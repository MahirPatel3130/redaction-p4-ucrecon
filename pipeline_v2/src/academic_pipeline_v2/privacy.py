from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .detectors import deterministic_spans
from .inference import InferenceBackend
from .models import (
    CVExtraction, CanonicalDocument, EntityMention, PrivacyFinding, PrivacyResult,
)
from .prompts import PRIVACY_PROMPT
from .util import sha256_bytes, stable_id


class RawPrivacyFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier_type: Literal[
        "person_name", "organization", "work_title", "email", "phone_number",
        "postal_address", "website", "unique_identifier", "signature", "date",
    ]
    start: int
    end: int


class RawPrivacyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    safe: bool
    model_confidence: float = Field(ge=0, le=1)
    # Bound each pass so constrained decoding closes the JSON object within its budget.
    # Additional findings are handled by the existing iterative repair loop.
    findings: list[RawPrivacyFinding] = Field(default_factory=list, max_length=50)


PRIORITY = {
    "email": 100, "unique_identifier": 95, "phone_number": 90,
    "postal_address": 85, "website": 80, "signature": 75,
    "person_name": 70, "organization": 60, "work_title": 50, "date": 10,
}


def placeholder(mention: EntityMention) -> str:
    entity = mention.resolved_entity_id
    kind = mention.identifier_type
    role = mention.person_role
    if kind == "person_name":
        label = {"applicant": "APPLICANT", "recommender": "RECOMMENDER"}.get(role, "PERSON")
        return f"[[{label}:{entity or mention.local_entity_id}]]"
    if kind == "organization":
        return f"[[ORG:{entity or mention.local_entity_id}]]"
    if kind == "work_title":
        return f"[[WORK:{entity or mention.local_entity_id}]]"
    if kind == "email":
        return f"[[EMAIL:{entity}]]" if entity and entity.startswith("person_") else "[[EMAIL]]"
    if kind == "phone_number":
        return f"[[PHONE:{entity}]]" if entity and entity.startswith("person_") else "[[PHONE]]"
    return {"postal_address": "[[ADDRESS]]", "website": "[[URL]]",
            "signature": "[[SIGNATURE]]", "unique_identifier": "[[IDENTIFIER]]",
            "date": "[[DATE]]"}.get(kind, "[[REDACTED]]")


def _non_overlapping(mentions: list[EntityMention]) -> list[EntityMention]:
    ordered = sorted(
        mentions,
        key=lambda row: (-PRIORITY.get(row.identifier_type, 0),
                         -(row.evidence.end - row.evidence.start), row.evidence.start),
    )
    selected: list[EntityMention] = []
    for mention in ordered:
        if mention.identifier_type == "date":
            continue
        if any(not (mention.evidence.end <= other.evidence.start
                    or mention.evidence.start >= other.evidence.end) for other in selected):
            continue
        selected.append(mention)
    return sorted(selected, key=lambda row: row.evidence.start)


def apply_replacements(document: CanonicalDocument, mentions: list[EntityMention]) -> tuple[
        str, list[dict[str, Any]]]:
    accepted = _non_overlapping([row for row in mentions if row.grounded])
    text = document.text
    rows: list[dict[str, Any]] = []
    for mention in sorted(accepted, key=lambda row: row.evidence.start, reverse=True):
        start, end = mention.evidence.start, mention.evidence.end
        if text[start:end].casefold() != mention.text.casefold():
            continue
        replacement = placeholder(mention)
        text = text[:start] + replacement + text[end:]
        rows.append({
            "redaction_id": stable_id("redaction", mention.mention_id),
            "mention_id": mention.mention_id,
            "application_id": document.application_id,
            "document_id": document.document_id,
            "identifier_type": mention.identifier_type,
            "person_role": mention.person_role,
            "resolved_entity_id": mention.resolved_entity_id,
            "text": mention.text,
            "placeholder": replacement,
            "page": mention.evidence.page,
            "start": start,
            "end": end,
            "model_confidence": mention.model_confidence,
            "detector": mention.detector,
            "grounded": mention.grounded,
        })
    return text, sorted(rows, key=lambda row: (row["page"], row["start"]))


def _scan(text: str, aliases: list[str]) -> list[PrivacyFinding]:
    masked = re.sub(r"\[\[[^\]]+\]\]", lambda match: " " * len(match.group()), text)
    findings = [PrivacyFinding(finding_type=row.identifier_type, start=row.start, end=row.end,
                               detector="regex", grounded=True)
                for row in deterministic_spans(masked)]
    folded = masked.casefold()
    for alias in sorted(set(aliases), key=len, reverse=True):
        alias = alias.strip()
        if len(alias) < 3 or alias.isdigit():
            continue
        start = folded.find(alias.casefold())
        if start >= 0:
            findings.append(PrivacyFinding(finding_type="protected_alias", start=start,
                                           end=start + len(alias), detector="alias", grounded=True))
    unique = {(row.finding_type, row.start, row.end): row for row in findings}
    return list(unique.values())


def _overlaps_placeholder(text: str, start: int, end: int) -> bool:
    return any(start < match.end() and end > match.start()
               for match in re.finditer(r"\[\[[^\]]+\]\]", text))


def validate_privacy(document: CanonicalDocument, deidentified_text: str,
                     redactions: list[dict[str, Any]], backend: InferenceBackend,
                     max_tokens: int, maximum_repairs: int) -> tuple[str, PrivacyResult,
                                                                    list[dict[str, Any]], dict[str, Any]]:
    aliases = [row["text"] for row in redactions]
    current = deidentified_text
    findings = _scan(current, aliases)
    repairs = 0
    calls = 0
    input_tokens = 0
    output_tokens = 0
    inference_seconds = 0.0
    model_confidence: float | None = None
    raw_responses: list[dict[str, Any]] = []
    while repairs <= maximum_repairs:
        deterministic_findings = _scan(current, aliases)
        prompt = PRIVACY_PROMPT.format(text=current)
        response = backend.complete("privacy", prompt,
                                    RawPrivacyReview.model_json_schema(), max_tokens)
        calls += 1
        input_tokens += response.input_tokens or 0
        output_tokens += response.output_tokens or 0
        inference_seconds += response.latency_seconds
        try:
            review = RawPrivacyReview.model_validate(response.payload)
        except Exception:
            response = backend.complete(
                "privacy", prompt + "\nReturn schema-valid JSON only; the prior object was invalid.",
                RawPrivacyReview.model_json_schema(), max_tokens)
            calls += 1
            input_tokens += response.input_tokens or 0
            output_tokens += response.output_tokens or 0
            inference_seconds += response.latency_seconds
            review = RawPrivacyReview.model_validate(response.payload)
        raw_responses.append(response.payload)
        model_confidence = review.model_confidence
        model_findings: list[PrivacyFinding] = []
        valid_spans: list[tuple[int, int, str]] = [
            (row.start, row.end, row.finding_type)
            for row in deterministic_findings if row.start is not None and row.end is not None
        ]
        for row in review.findings:
            if (0 <= row.start < row.end <= len(current)
                    and not _overlaps_placeholder(current, row.start, row.end)):
                model_findings.append(PrivacyFinding(
                    finding_type=row.identifier_type, start=row.start, end=row.end,
                    detector="model", grounded=True,
                ))
                valid_spans.append((row.start, row.end, row.identifier_type))
        findings = deterministic_findings + model_findings
        if not findings and review.safe:
            break
        if repairs >= maximum_repairs or not valid_spans:
            break
        selected_spans: list[tuple[int, int, str]] = []
        for candidate in sorted(set(valid_spans), key=lambda row: (row[0], -(row[1] - row[0]))):
            start, end, _ = candidate
            if any(not (end <= other[0] or start >= other[1]) for other in selected_spans):
                continue
            selected_spans.append(candidate)
        for start, end, identifier_type in sorted(selected_spans, reverse=True):
            raw = current[start:end]
            replacement = "[[PRIVACY_REPAIR]]"
            current = current[:start] + replacement + current[end:]
            redactions.append({
                "redaction_id": stable_id("redaction", f"{document.document_id}:repair:{repairs}:{start}"),
                "mention_id": None, "application_id": document.application_id,
                "document_id": document.document_id, "identifier_type": identifier_type,
                "person_role": "unknown", "resolved_entity_id": None, "text": raw,
                "placeholder": replacement, "page": None, "start": start, "end": end,
                "model_confidence": review.model_confidence, "detector": "privacy_repair",
                "grounded": True,
            })
        repairs += 1
    findings = _scan(current, [row["text"] for row in redactions]) + [
        row for row in findings if row.detector == "model" and row.start is not None
        and row.end is not None and row.end <= len(current)
        and "[[" not in current[row.start:row.end]
    ]
    unique = {(row.finding_type, row.start, row.end, row.detector): row for row in findings}
    findings = list(unique.values())
    released = not findings and model_confidence is not None and model_confidence >= 0.90
    reasons = [] if released else (["residual_privacy_finding"] if findings else
                                   ["privacy_model_confidence_below_threshold"])
    result = PrivacyResult(
        application_id=document.application_id, document_id=document.document_id,
        release_status="released" if released else "withheld", reason_codes=reasons,
        findings=findings, replacement_count=len(redactions), repair_attempts=repairs,
        model_qa_confidence=model_confidence,
        content_sha256=sha256_bytes(current.encode("utf-8")) if released else None,
    )
    return current, result, redactions, {"calls": calls, "repairs": repairs,
                                         "input_tokens": input_tokens,
                                         "output_tokens": output_tokens,
                                         "inference_seconds": inference_seconds,
                                         "_raw_responses": raw_responses}


def privacy_aliases(extraction: CVExtraction | Any) -> list[str]:
    return [mention.text for mention in extraction.mentions]
