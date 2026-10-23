from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Protocol

import httpx

from .config import ModelConfig
from .detectors import deterministic_spans
from .errors import InferenceError
from .models import ModelResponse


class InferenceBackend(Protocol):
    def complete(self, task: str, prompt: str, schema: dict[str, Any], max_tokens: int) -> ModelResponse:
        ...

    def close(self) -> None:
        ...


class VLLMBackend:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = httpx.Client(timeout=config.timeout_seconds,
                                   headers={"Authorization": f"Bearer {config.api_key}"})
        self._lock = threading.Lock()

    def complete(self, task: str, prompt: str, schema: dict[str, Any], max_tokens: int) -> ModelResponse:
        from .prompts import SYSTEM

        started = time.monotonic()
        retry_count = 0
        last_error: Exception | None = None
        last_code = "local_model_request_failed"
        for attempt in range(self.config.max_retries + 1):
            request = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": f"{task}_response", "strict": True, "schema": schema},
                },
                "chat_template_kwargs": {"enable_thinking": False},
            }
            try:
                response = self.client.post(f"{self.config.base_url.rstrip('/')}/chat/completions",
                                            json=request)
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise InferenceError("model_output_truncated")
                raw = choice["message"]["content"]
                payload = _parse_json_content(raw)
                usage = body.get("usage", {})
                return ModelResponse(
                    payload=payload,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    latency_seconds=time.monotonic() - started,
                    retry_count=retry_count,
                )
            except Exception as exc:
                last_error = exc
                if isinstance(exc, InferenceError):
                    last_code = str(exc)
                elif isinstance(exc, httpx.HTTPStatusError):
                    last_code = f"model_http_{exc.response.status_code}"
                elif isinstance(exc, (KeyError, TypeError, IndexError)):
                    last_code = "malformed_model_api_response"
                elif isinstance(exc, json.JSONDecodeError):
                    last_code = "invalid_model_json"
                else:
                    last_code = "local_model_request_failed"
                if attempt >= self.config.max_retries:
                    raise InferenceError(last_code) from exc
                retry_count += 1
        raise InferenceError(last_code) from last_error

    def close(self) -> None:
        self.client.close()


def _parse_json_content(raw: Any) -> dict[str, Any]:
    """Recover a single JSON object without logging protected model content."""
    if not isinstance(raw, str) or not raw.strip():
        raise InferenceError("empty_model_content")
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, count=1, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value, count=1)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as original:
        start = value.find("{")
        if start < 0:
            raise InferenceError("invalid_model_json") from original
        try:
            payload, _ = json.JSONDecoder().raw_decode(value[start:])
        except json.JSONDecodeError as exc:
            raise InferenceError("invalid_model_json") from exc
    if not isinstance(payload, dict):
        raise InferenceError("model_json_not_object")
    return payload


def _source(prompt: str) -> str:
    marker = "SOURCE_TEXT:\n"
    if marker in prompt:
        return prompt.rsplit(marker, 1)[1]
    marker = "CANDIDATE_TEXT:\n"
    return prompt.rsplit(marker, 1)[1] if marker in prompt else ""


def _field(value: Any, text: str, confidence: float = 0.97) -> dict[str, Any]:
    if value in {None, ""}:
        return {"value": None, "status": "not_found", "model_confidence": confidence,
                "start": None, "end": None}
    value = str(value)
    start = text.find(value)
    return {"value": value, "status": "found" if start >= 0 else "uncertain",
            "model_confidence": confidence, "start": start if start >= 0 else None,
            "end": start + len(value) if start >= 0 else None}


class MockBackend:
    """Deterministic offline backend for CI and synthetic smoke tests."""

    def complete(self, task: str, prompt: str, schema: dict[str, Any], max_tokens: int) -> ModelResponse:
        started = time.monotonic()
        text = _source(prompt)
        if task == "cv":
            payload = self._cv(text)
        elif task == "letter":
            payload = self._letter(text)
        elif task == "privacy":
            spans = deterministic_spans(text)
            payload = {"safe": not spans, "model_confidence": 0.99,
                       "findings": [{"identifier_type": row.identifier_type,
                                     "start": row.start, "end": row.end}
                                    for row in spans]}
        else:
            payload = {}
        return ModelResponse(payload=payload, latency_seconds=time.monotonic() - started)

    def _mentions(self, text: str) -> list[dict[str, Any]]:
        def contact_role(value: str) -> str:
            folded = value.casefold()
            if folded.startswith("rita"):
                return "recommender"
            if folded.startswith("alice"):
                return "applicant"
            return "unknown"

        rows = [{"identifier_type": span.identifier_type, "person_role": contact_role(span.text),
                 "text": span.text, "start": span.start, "end": span.end,
                 "model_confidence": 0.99} for span in deterministic_spans(text)]
        patterns = (
            (r"\bApplicant [A-Z][A-Za-z'-]+\b", "person_name", "applicant"),
            (r"\bRecommender [A-Z][A-Za-z'-]+\b", "person_name", "recommender"),
            (r"\bCoauthor [A-Z][A-Za-z'-]+\b", "person_name", "coauthor"),
            (r"\bExample University\b", "organization", "institution"),
            (r"\bExample Journal\b", "organization", "institution"),
            (r"\bImportant Paper\b", "work_title", "unknown"),
        )
        for pattern, kind, role in patterns:
            for match in re.finditer(pattern, text):
                rows.append({"identifier_type": kind, "person_role": role,
                             "text": match.group(), "start": match.start(), "end": match.end(),
                             "model_confidence": 0.98})
        return rows

    def _cv(self, text: str) -> dict[str, Any]:
        sections: dict[str, Any] = {}
        specs = {
            "education": ("EDUCATION|", ["institution", "degree_type", "field", "start_year", "end_year"]),
            "employment": ("EMPLOYMENT|", ["organization", "original_title", "job_category", "start_date", "end_date"]),
            "publications": ("PUBLICATION|", ["title", "publication_type", "year", "venue", "authors"]),
        }
        for name, (prefix, fields) in specs.items():
            items = []
            for line in text.splitlines():
                if line.startswith(prefix):
                    parts = line.split("|")[1:]
                    values = dict(zip(fields, parts))
                    start = text.find(line)
                    if name == "education":
                        item = {key: _field(values.get(key), text) for key in fields}
                        item["graduation_year"] = _field(values.get("end_year"), text)
                    elif name == "employment":
                        item = {key: _field(values.get(key), text) for key in fields}
                        item["position_type"] = _field("academic", text)
                    else:
                        item = {key: _field(values.get(key), text) for key in fields if key != "authors"}
                        item["doi"] = _field(None, text)
                        item["authors"] = [
                            {"name": _field(author.strip(), text), "author_position": index,
                             "is_applicant": "Applicant" in author}
                            for index, author in enumerate(values.get("authors", "").split(";"), 1)
                            if author.strip()
                        ]
                    item.update({"model_confidence": 0.97, "start": start, "end": start + len(line)})
                    items.append(item)
            sections[name] = {"status": "found" if items else "not_found",
                              "model_confidence": 0.97, "not_found_reason": None if items else
                              f"No {name} section found", "items": items}
        return {"sections": sections, "research_fields": [], "research_methods": [],
                "mentions": self._mentions(text)}

    def _letter(self, text: str) -> dict[str, Any]:
        mentions = self._mentions(text)
        applicant = next((row for row in mentions if row["person_role"] == "applicant"), None)
        recommender = next((row for row in mentions if row["person_role"] == "recommender"), None)
        relationships = []
        if applicant and recommender:
            relationships.append({"relationship_type": "wrote_recommendation_for",
                                  "source_start": recommender["start"],
                                  "target_start": applicant["start"],
                                  "model_confidence": 0.99})
        return {"status": "found", "model_confidence": 0.98,
                "mentions": mentions, "relationships": relationships}

    def close(self) -> None:
        return None


def create_backend(config: ModelConfig) -> InferenceBackend:
    return MockBackend() if config.backend == "mock" else VLLMBackend(config)
