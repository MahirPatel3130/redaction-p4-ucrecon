from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from . import PROMPT_VERSION
from .common import deterministic_identifiers


CV_PAGE_PROMPT = """You are parsing one page of an academic CV for a protected research pipeline.
Return JSON only. Never omit a requested section. Preserve your own confidence numbers; do not
explain them. Use exact visible strings for people, organizations, work titles, DOI values, and
contacts. Coordinates use [x_min,y_min,x_max,y_max] normalized 0..1 when visible.

Schema:
{"sections":{"education":{"status":"found|not_found|uncertain","model_confidence":0.0,
"not_found_reason":"","items":[]},"employment":{...},"publications":{...}},
"research_fields":[{"value":"","status":"found|uncertain","model_confidence":0.0}],
"methods":[{"value":"","status":"found|uncertain","model_confidence":0.0}],
"identifiers":[{"identifier_type":"person_name|email|phone_number|website|postal_address|unique_identifier|signature|organization|work_title|date",
"person_role":"applicant|coauthor|reference|advisor|committee_member|institution|unknown",
"text":"exact visible text","model_confidence":0.0,"bbox_normalized":[0,0,0,0]}]}.

Education item fields: institution, degree_type, field, start_year, end_year.
Employment item fields: organization, job_title, job_category, start_year, end_year.
Publication item fields: title, doi, publication_type, year, venue, authors. Each scalar field must
be {"value":...,"status":"found|not_found|uncertain|not_applicable","model_confidence":0.0}.
Authors are {"name":field,"author_position":integer,"is_applicant":boolean,
"model_confidence":0.0}. Every item must include item_model_confidence and source_page.
Allowed job_category: academic_faculty, postdoctoral, research, industry, government, other.
If a section is absent, return status not_found, a confidence, reason, and empty items.
Prompt version: %s.
""" % PROMPT_VERSION

CV_CONSOLIDATE_PROMPT = """Consolidate page-level academic CV parsing JSON into one document JSON.
Merge only clear continuations or duplicates. Preserve every model confidence from the source item
or field; do not calculate averages and do not invent confidence. Keep the same sections schema.
For a section with no items, return its explicit not_found/uncertain status and reason. Return JSON
only. Page results follow:\n"""

LETTER_PAGE_PROMPT = """Analyze one recommendation-letter page. Return JSON only:
{"transcript":{"text":"paragraph-normalized exact page text","status":"found|uncertain|model_failed",
"model_confidence":0.0},"entities":[{"identifier_type":"person_name|email|phone_number|website|postal_address|unique_identifier|signature|organization|work_title|date",
"person_role":"applicant|recommender|recipient|third_party|unknown","text":"exact substring",
"model_confidence":0.0,"bbox_normalized":[0,0,0,0]}]}.
When reliable PDF text is supplied, use it exactly and normalize only line wrapping and paragraphs.
Identify all people, institutions, employers, departments, journals, publication titles, dates,
addresses, contacts, identifiers, URLs, and signatures. Do not return commentary.
Prompt version: %s.
""" % PROMPT_VERSION

TRANSCRIBE_PROMPT = """Transcribe this scanned recommendation-letter page exactly, preserving
paragraphs. Return JSON only: {"text":"...","status":"found|uncertain|model_failed",
"model_confidence":0.0}. Do not summarize. Prompt version: %s.""" % PROMPT_VERSION

QA_PROMPT = """Review this de-identified academic document for remaining direct or linkable
identifiers. Placeholders such as [[PERSON:...]], [[ORG:...]], and [[WORK:...]] are safe. Return JSON
only: {"qa_passed":true,"model_confidence":0.0,"flagged_spans":[{"text":"exact remaining
identifier","identifier_type":"person|organization|work|contact|address|date|other"}]}.
Be conservative. Do not rewrite the document. Prompt version: %s.\nDOCUMENT:\n""" % PROMPT_VERSION


def extract_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output did not contain a JSON object")


def response_text(result: Any) -> str:
    generated = result[0].get("generated_text", result[0]) if isinstance(result, list) and result else result
    if isinstance(generated, list) and generated:
        generated = generated[-1]
    if isinstance(generated, dict):
        return str(generated.get("content", generated.get("text", "")))
    return str(generated)


class GemmaModel:
    def __init__(self, model_id: str):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers is required") from exc
        self.model_id = model_id
        self.pipe = pipeline("image-text-to-text", model=model_id, device_map="auto", dtype="auto")
        self.calls = 0
        self.retries = 0
        self.inference_seconds = 0.0
        self.raw_responses: list[str] = []

    def reset_metrics(self) -> None:
        self.calls = self.retries = 0
        self.inference_seconds = 0.0
        self.raw_responses = []

    def _call(self, prompt: str, image: Any | None = None, max_tokens: int = 4096) -> dict[str, Any]:
        content = []
        if image is not None:
            content.append({"type": "image", "url": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        last_error = None
        for attempt in range(2):
            started = time.monotonic()
            self.calls += 1
            result = self.pipe(messages, return_full_text=False,
                               generate_kwargs={"max_new_tokens": max_tokens, "do_sample": False})
            self.inference_seconds += time.monotonic() - started
            raw = response_text(result)
            self.raw_responses.append(raw)
            try:
                return extract_json(raw)
            except ValueError as exc:
                last_error = exc
                if attempt == 0:
                    self.retries += 1
        raise ValueError("Malformed model JSON after retry") from last_error

    def parse_cv_page(self, image: Any, native_text: str, page: int) -> dict[str, Any]:
        return self._call(f"{CV_PAGE_PROMPT}\nPAGE={page}\nPDF_TEXT:\n{native_text}", image)

    def consolidate_cv(self, pages: list[dict[str, Any]]) -> dict[str, Any]:
        return self._call(CV_CONSOLIDATE_PROMPT + json.dumps(pages, ensure_ascii=False), max_tokens=8192)

    def analyze_letter_page(self, image: Any, native_text: str, page: int) -> dict[str, Any]:
        return self._call(f"{LETTER_PAGE_PROMPT}\nPAGE={page}\nPDF_TEXT:\n{native_text}", image)

    def transcribe(self, image: Any) -> dict[str, Any]:
        return self._call(TRANSCRIBE_PROMPT, image, max_tokens=4096)

    def qa(self, text: str) -> dict[str, Any]:
        return self._call(QA_PROMPT + text, max_tokens=2048)


class MockModel:
    """Deterministic test model. CLI use requires RESEARCH_PIPELINE_ALLOW_MOCK=1."""

    model_id = "mock"

    def __init__(self):
        self.calls = self.retries = 0
        self.inference_seconds = 0.0
        self.raw_responses: list[str] = []

    def reset_metrics(self) -> None:
        self.calls = self.retries = 0
        self.inference_seconds = 0.0
        self.raw_responses = []

    @staticmethod
    def _field(value: Any, confidence: float = 0.96) -> dict[str, Any]:
        return {"value": value, "status": "found" if value not in (None, "") else "not_found",
                "model_confidence": confidence}

    def parse_cv_page(self, image: Any, native_text: str, page: int) -> dict[str, Any]:
        self.calls += 1
        sections = {name: {"status": "not_found", "model_confidence": 0.95,
                           "not_found_reason": f"No {name} on page", "items": []}
                    for name in ("education", "employment", "publications")}
        identifiers = deterministic_identifiers(native_text)
        for match in re.finditer(r"\bApplicant Alice\b", native_text):
            identifiers.append({"identifier_type": "person_name", "person_role": "applicant",
                                "text": match.group(), "model_confidence": 0.99})
        for line in native_text.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if line.startswith("EDUCATION|") and len(parts) >= 6:
                sections["education"] = {"status": "found", "model_confidence": 0.97, "items": [{
                    "item_model_confidence": 0.96, "source_page": page,
                    "institution": self._field(parts[1]), "degree_type": self._field(parts[2]),
                    "field": self._field(parts[3]), "start_year": self._field(parts[4]),
                    "end_year": self._field(parts[5]),
                }]}
                identifiers.append({"identifier_type": "organization", "person_role": "institution",
                                    "text": parts[1], "model_confidence": 0.97})
            elif line.startswith("EMPLOYMENT|") and len(parts) >= 6:
                sections["employment"] = {"status": "found", "model_confidence": 0.96, "items": [{
                    "item_model_confidence": 0.95, "source_page": page,
                    "organization": self._field(parts[1]), "job_title": self._field(parts[2]),
                    "job_category": self._field(parts[3]), "start_year": self._field(parts[4]),
                    "end_year": self._field(parts[5]),
                }]}
                identifiers.append({"identifier_type": "organization", "person_role": "institution",
                                    "text": parts[1], "model_confidence": 0.96})
            elif line.startswith("PUBLICATION|") and len(parts) >= 6:
                author_names = [name.strip() for name in parts[5].split(";") if name.strip()]
                sections["publications"] = {"status": "found", "model_confidence": 0.94, "items": [{
                    "item_model_confidence": 0.93, "source_page": page,
                    "title": self._field(parts[1]), "doi": self._field(None),
                    "publication_type": self._field(parts[2]), "year": self._field(parts[3]),
                    "venue": self._field(parts[4]), "authors": [
                        {"name": self._field(name), "author_position": index,
                         "is_applicant": index == 1, "model_confidence": 0.94}
                        for index, name in enumerate(author_names, 1)],
                }]}
                identifiers.extend([
                    {"identifier_type": "work_title", "person_role": "unknown", "text": parts[1],
                     "model_confidence": 0.94},
                    {"identifier_type": "organization", "person_role": "institution", "text": parts[4],
                     "model_confidence": 0.94},
                ])
                identifiers.extend({"identifier_type": "person_name", "person_role": "coauthor",
                                    "text": name, "model_confidence": 0.94} for name in author_names)
        return {"sections": sections, "research_fields": [], "methods": [],
                "identifiers": identifiers}

    def consolidate_cv(self, pages: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        result = {"sections": {}, "research_fields": [], "methods": [], "identifiers": []}
        for section in ("education", "employment", "publications"):
            items = [item for page in pages for item in page["sections"][section].get("items", [])]
            result["sections"][section] = ({"status": "found", "model_confidence": 0.93,
                                            "items": items} if items else
                                           {"status": "not_found", "model_confidence": 0.95,
                                            "not_found_reason": f"No {section} found", "items": []})
        result["identifiers"] = [item for page in pages for item in page.get("identifiers", [])]
        return result

    def analyze_letter_page(self, image: Any, native_text: str, page: int) -> dict[str, Any]:
        self.calls += 1
        entities = deterministic_identifiers(native_text)
        patterns = [
            (r"\bApplicant Alice\b", "person_name", "applicant"),
            (r"\bRecommender Rita\b", "person_name", "recommender"),
            (r"\bExample University\b", "organization", "unknown"),
            (r"\bImportant Paper\b", "work_title", "unknown"),
        ]
        for pattern, kind, role in patterns:
            for match in re.finditer(pattern, native_text):
                entities.append({"identifier_type": kind, "person_role": role,
                                 "text": match.group(), "model_confidence": 0.97})
        return {"transcript": {"text": native_text, "status": "found",
                               "model_confidence": 0.98}, "entities": entities}

    def transcribe(self, image: Any) -> dict[str, Any]:
        self.calls += 1
        return {"text": "", "status": "model_failed", "model_confidence": 0.1}

    def qa(self, text: str) -> dict[str, Any]:
        self.calls += 1
        flagged = []
        for value in ("Applicant Alice", "Recommender Rita", "Example University", "Important Paper"):
            if value.casefold() in text.casefold():
                flagged.append({"text": value, "identifier_type": "remaining_identifier"})
        return {"qa_passed": not flagged, "model_confidence": 0.98,
                "flagged_spans": flagged}


def create_model(model_id: str) -> GemmaModel | MockModel:
    if model_id == "mock":
        if os.environ.get("RESEARCH_PIPELINE_ALLOW_MOCK") != "1":
            raise RuntimeError("Mock model is disabled outside tests")
        return MockModel()
    return GemmaModel(model_id)
