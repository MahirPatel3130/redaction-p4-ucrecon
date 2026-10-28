from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .detectors import deterministic_spans
from .inference import InferenceBackend
from .models import (
    CVExtraction, CanonicalDocument, EntityMention, PrivacyFinding, PrivacyResult,
)
from .prompts import PRIVACY_PROMPT
from .util import sha256_bytes, stable_id


IdentifierType = Literal[
    "person_name", "organization", "work_title", "email", "phone_number",
    "postal_address", "website", "unique_identifier", "signature", "date",
]


class RawPrivacyFinding(BaseModel):
    """Detection-only model output. Character offsets are deliberately forbidden."""

    model_config = ConfigDict(extra="forbid")
    finding_kind: Literal["visible_identifier", "contextual_risk"]
    identifier_type: IdentifierType
    text: str = Field(min_length=1, max_length=500)
    line_number: int = Field(ge=1)
    model_confidence: float = Field(ge=0, le=1)


class RawPrivacyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    safe: bool
    model_confidence: float = Field(ge=0, le=1)
    findings: list[RawPrivacyFinding] = Field(default_factory=list, max_length=50)


PRIORITY = {
    "email": 100, "unique_identifier": 95, "phone_number": 90,
    "postal_address": 85, "website": 80, "signature": 75,
    "person_name": 70, "organization": 60, "work_title": 50, "date": 10,
    "protected_alias": 65,
}

PLACEHOLDER_PATTERN = re.compile(r"\[\[[^\]]+\]\]")


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    start: int
    lines: tuple[tuple[int, int, int, str], ...]

    def numbered_text(self) -> str:
        return "\n".join(f"[{number:04d}] {value.rstrip()}" for number, _, _, value in self.lines)


@dataclass(frozen=True)
class RepairSpan:
    start: int
    end: int
    identifier_type: str
    detector: str
    model_confidence: float | None = None
    person_role: str = "unknown"
    resolved_entity_id: str | None = None
    replacement: str | None = None


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


def _repair_placeholder(kind: str, document_id: str, start: int) -> str:
    suffix = stable_id("doc", f"{document_id}:{kind}:{start}")
    if kind == "person_name" or kind == "protected_alias":
        return f"[[PERSON:{suffix}]]"
    if kind == "organization":
        return f"[[ORG:{suffix}]]"
    if kind == "work_title":
        return f"[[WORK:{suffix}]]"
    return {"email": "[[EMAIL]]", "phone_number": "[[PHONE]]",
            "postal_address": "[[ADDRESS]]", "website": "[[URL]]",
            "unique_identifier": "[[IDENTIFIER]]", "signature": "[[SIGNATURE]]",
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


def _masked(text: str) -> str:
    return PLACEHOLDER_PATTERN.sub(lambda match: " " * len(match.group()), text)


def _scan(text: str, aliases: list[str]) -> list[PrivacyFinding]:
    masked = _masked(text)
    findings = [PrivacyFinding(finding_type=row.identifier_type, start=row.start, end=row.end,
                               detector="regex", grounded=True, disposition="confirmed")
                for row in deterministic_spans(masked)]
    folded = masked.casefold()
    for alias in sorted(set(aliases), key=len, reverse=True):
        alias = alias.strip()
        if len(alias) < 3 or alias.isdigit():
            continue
        needle = alias.casefold()
        cursor = 0
        while True:
            start = folded.find(needle, cursor)
            if start < 0:
                break
            findings.append(PrivacyFinding(finding_type="protected_alias", start=start,
                                           end=start + len(alias), detector="alias", grounded=True,
                                           disposition="confirmed"))
            cursor = start + max(1, len(alias))
    unique = {(row.finding_type, row.start, row.end): row for row in findings}
    return list(unique.values())


def _overlaps_placeholder(text: str, start: int, end: int) -> bool:
    return any(start < match.end() and end > match.start() for match in PLACEHOLDER_PATTERN.finditer(text))


def _model_text_is_plausible(identifier_type: str, value: str) -> bool:
    value = value.strip()
    if len(value) < 3 or PLACEHOLDER_PATTERN.search(value):
        return False
    folded = value.casefold()
    if identifier_type == "organization" and folded in {
        "university", "college", "school", "department", "institute", "institution",
        "organization", "committee",
    }:
        return False
    if identifier_type == "person_name" and folded in {
        "applicant", "candidate", "recommender", "professor", "doctor", "chair",
        "he", "she", "they", "him", "her", "them",
    }:
        return False
    if identifier_type == "work_title" and len(value.split()) < 2:
        return False
    return True


def _ground_model_text(text: str, value: str, start: int = 0,
                       end: int | None = None) -> list[tuple[int, int]]:
    """Map model-returned text to all deterministic occurrences in a bounded region."""
    value = value.strip()
    if not value:
        return []
    region_end = len(text) if end is None else end
    region = text[start:region_end]
    patterns = [re.escape(value)]
    tokens = value.split()
    if len(tokens) > 1:
        patterns.append(r"\s+".join(re.escape(token) for token in tokens))
    matches = [(start + match.start(), start + match.end()) for pattern in patterns
               for match in re.finditer(pattern, region, flags=re.IGNORECASE)]
    return sorted(set(matches))


def _chunks(text: str, maximum_characters: int = 3000) -> list[TextChunk]:
    raw_lines = text.splitlines(keepends=True) or [text]
    chunks: list[TextChunk] = []
    current: list[tuple[int, int, int, str]] = []
    current_length = 0
    cursor = 0
    line_number = 1
    for value in raw_lines:
        line_start, line_end = cursor, cursor + len(value)
        if current and current_length + len(value) > maximum_characters:
            chunk_start = current[0][1]
            chunk_end = current[-1][2]
            chunks.append(TextChunk(f"chunk_{len(chunks) + 1:04d}", text[chunk_start:chunk_end],
                                    chunk_start, tuple(current)))
            current = []
            current_length = 0
        current.append((line_number, line_start, line_end, value))
        current_length += len(value)
        cursor = line_end
        line_number += 1
    if current:
        chunk_start = current[0][1]
        chunk_end = current[-1][2]
        chunks.append(TextChunk(f"chunk_{len(chunks) + 1:04d}", text[chunk_start:chunk_end],
                                chunk_start, tuple(current)))
    return chunks


def _detect(text: str, backend: InferenceBackend, max_tokens: int,
            pass_name: str, policy: dict[str, Any] | None = None) -> tuple[
                list[RepairSpan], list[PrivacyFinding], dict[str, Any]]:
    grounded: list[RepairSpan] = []
    findings: list[PrivacyFinding] = []
    metrics: dict[str, Any] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                               "inference_seconds": 0.0, "raw_responses": [],
                               "model_confidences": []}
    for chunk in _chunks(text):
        replace_types = ", ".join(sorted(_approved_types(policy)))
        prompt = PRIVACY_PROMPT.format(pass_name=pass_name, chunk_id=chunk.chunk_id,
                                       replace_types=replace_types,
                                       text=chunk.numbered_text())
        response = backend.complete("privacy", prompt, RawPrivacyReview.model_json_schema(), max_tokens)
        review = RawPrivacyReview.model_validate(response.payload)
        metrics["calls"] += 1
        metrics["input_tokens"] += response.input_tokens or 0
        metrics["output_tokens"] += response.output_tokens or 0
        metrics["inference_seconds"] += response.latency_seconds
        metrics["raw_responses"].append({"pass": pass_name, "chunk_id": chunk.chunk_id,
                                         "payload": response.payload})
        metrics["model_confidences"].append(review.model_confidence)
        if (not review.safe and not review.findings) or len(review.findings) == 50:
            findings.append(PrivacyFinding(
                finding_type="model_detection_incomplete", detector="model", grounded=False,
                disposition="review_required", model_confidence=review.model_confidence,
            ))
        line_map = {number: (start, end) for number, start, end, _ in chunk.lines}
        for row in review.findings:
            if row.finding_kind == "contextual_risk":
                bounds = line_map.get(row.line_number)
                ranges = (_ground_model_text(text, row.text, *bounds) if bounds else [])
                if not ranges:
                    ranges = _ground_model_text(text, row.text, chunk.start,
                                                chunk.start + len(chunk.text))
                ranges = [(start, end) for start, end in ranges
                          if not _overlaps_placeholder(text, start, end)]
                if ranges:
                    for start, end in ranges:
                        findings.append(PrivacyFinding(
                            finding_type="contextual_reidentification_risk", start=start, end=end,
                            detector="model", grounded=True, disposition="review_required",
                            model_confidence=row.model_confidence,
                        ))
                else:
                    findings.append(PrivacyFinding(
                        finding_type="unsupported_contextual_advisory", detector="model",
                        grounded=False, disposition="advisory",
                        model_confidence=row.model_confidence,
                    ))
                continue
            if not _model_text_is_plausible(row.identifier_type, row.text):
                findings.append(PrivacyFinding(
                    finding_type="unsupported_model_advisory", detector="model",
                    grounded=False, disposition="advisory",
                    model_confidence=row.model_confidence,
                ))
                continue
            bounds = line_map.get(row.line_number)
            ranges = (_ground_model_text(text, row.text, *bounds) if bounds else [])
            if not ranges and _model_text_is_plausible(row.identifier_type, row.text):
                ranges = _ground_model_text(text, row.text, chunk.start,
                                            chunk.start + len(chunk.text))
            ranges = [(start, end) for start, end in ranges
                      if not _overlaps_placeholder(text, start, end)]
            if not ranges:
                findings.append(PrivacyFinding(
                    finding_type="unsupported_model_advisory", detector="model",
                    grounded=False, disposition="advisory",
                    model_confidence=row.model_confidence,
                ))
                continue
            for start, end in ranges:
                grounded.append(RepairSpan(start, end, row.identifier_type, "model",
                                           row.model_confidence))
                findings.append(PrivacyFinding(
                    finding_type=row.identifier_type, start=start, end=end, detector="model",
                    grounded=True, disposition="confirmed",
                    model_confidence=row.model_confidence,
                ))
    unique_spans = {(row.start, row.end, row.identifier_type): row for row in grounded}
    unique_findings = {(row.finding_type, row.start, row.end, row.detector,
                        row.disposition): row for row in findings}
    return list(unique_spans.values()), list(unique_findings.values()), metrics


def _approved_types(policy: dict[str, Any] | None) -> set[str]:
    if policy and isinstance(policy.get("replace_identifier_types"), list):
        return {str(value) for value in policy["replace_identifier_types"]}
    return {"person_name", "organization", "work_title", "email", "phone_number",
            "postal_address", "website", "unique_identifier", "signature"}


def _alias_repairs(text: str, redactions: list[dict[str, Any]]) -> list[RepairSpan]:
    result: list[RepairSpan] = []
    masked = _masked(text)
    folded = masked.casefold()
    for row in redactions:
        if row.get("detector") == "privacy_repair":
            continue
        alias = str(row.get("text", "")).strip()
        if len(alias) < 3 or alias.isdigit():
            continue
        cursor = 0
        while True:
            start = folded.find(alias.casefold(), cursor)
            if start < 0:
                break
            result.append(RepairSpan(
                start, start + len(alias), str(row.get("identifier_type") or "protected_alias"),
                "alias", row.get("model_confidence"), str(row.get("person_role") or "unknown"),
                row.get("resolved_entity_id"), row.get("placeholder"),
            ))
            cursor = start + len(alias)
    return result


def _deterministic_repairs(text: str) -> list[RepairSpan]:
    return [RepairSpan(row.start, row.end, row.identifier_type, "regex")
            for row in deterministic_spans(_masked(text))]


def _select_repairs(spans: list[RepairSpan]) -> list[RepairSpan]:
    selected: list[RepairSpan] = []
    for row in sorted(spans, key=lambda item: (-PRIORITY.get(item.identifier_type, 0),
                                               -(item.end - item.start), item.start)):
        if any(not (row.end <= other.start or row.start >= other.end) for other in selected):
            continue
        selected.append(row)
    return sorted(selected, key=lambda item: item.start)


def _mask_spans_preserving_offsets(text: str, spans: list[RepairSpan]) -> str:
    """Hide known candidates from a coverage pass without changing global offsets."""
    characters = list(text)
    for row in _select_repairs(spans):
        for index in range(row.start, min(row.end, len(characters))):
            if characters[index] not in {"\n", "\r"}:
                characters[index] = " "
    return "".join(characters)


def _apply_repair_batch(document: CanonicalDocument, text: str, spans: list[RepairSpan],
                        redactions: list[dict[str, Any]]) -> tuple[str, int]:
    selected = _select_repairs(spans)
    for row in sorted(selected, key=lambda item: item.start, reverse=True):
        raw = text[row.start:row.end]
        replacement = row.replacement or _repair_placeholder(
            row.identifier_type, document.document_id, row.start)
        text = text[:row.start] + replacement + text[row.end:]
        redactions.append({
            "redaction_id": stable_id(
                "redaction", f"{document.document_id}:privacy_gate:{row.start}:{row.end}"),
            "mention_id": None, "application_id": document.application_id,
            "document_id": document.document_id, "identifier_type": row.identifier_type,
            "person_role": row.person_role, "resolved_entity_id": row.resolved_entity_id,
            "text": raw, "placeholder": replacement, "page": None,
            "start": row.start, "end": row.end, "model_confidence": row.model_confidence,
            "detector": "privacy_repair", "grounded": True,
        })
    return text, len(selected)


def validate_privacy(document: CanonicalDocument, deidentified_text: str,
                     redactions: list[dict[str, Any]], backend: InferenceBackend,
                     max_tokens: int, maximum_repairs: int,
                     policy: dict[str, Any] | None = None) -> tuple[
                         str, PrivacyResult, list[dict[str, Any]], dict[str, Any]]:
    """Run a finite gate: detect, one deterministic repair batch, independently verify."""
    aliases = [str(row["text"]) for row in redactions if row.get("detector") != "privacy_repair"]
    approved = _approved_types(policy)
    candidate_hash = sha256_bytes(deidentified_text.encode("utf-8"))

    model_spans, first_findings, first_metrics = _detect(
        deidentified_text, backend, max_tokens, "candidate_detection", policy)
    repair_spans = _deterministic_repairs(deidentified_text) + _alias_repairs(
        deidentified_text, redactions)
    repair_spans.extend(row for row in model_spans if row.identifier_type in approved)
    coverage_input = _mask_spans_preserving_offsets(deidentified_text, repair_spans)
    coverage_spans, coverage_findings, coverage_metrics = _detect(
        coverage_input, backend, max_tokens, "candidate_coverage", policy)
    repair_spans.extend(row for row in coverage_spans if row.identifier_type in approved)
    repaired = deidentified_text
    repair_count = 0
    if repair_spans and maximum_repairs > 0:
        repaired, repair_count = _apply_repair_batch(
            document, deidentified_text, repair_spans, redactions)

    deterministic_final = _scan(repaired, aliases)
    _, final_model_findings, final_metrics = _detect(
        repaired, backend, max_tokens, "independent_final_review", policy)
    grounded_final = [row for row in final_model_findings
                      if row.grounded and row.finding_type in approved]
    review_items = [row for row in final_model_findings
                    if row.disposition == "review_required"]
    advisories = [row for row in first_findings + coverage_findings + final_model_findings
                  if row.disposition == "advisory"]

    if deterministic_final or grounded_final:
        status = "withheld"
        reasons = ["confirmed_residual_privacy_finding"]
    elif review_items:
        status = "review_required"
        reasons = sorted({
            ("model_detection_incomplete" if row.finding_type == "model_detection_incomplete"
             else "contextual_reidentification_review") for row in review_items
        })
    else:
        status = "validated_candidate"
        reasons = (["unsupported_model_advisory"] if advisories else [])

    confidences = (first_metrics["model_confidences"]
                   + coverage_metrics["model_confidences"]
                   + final_metrics["model_confidences"])
    model_confidence = min(confidences) if confidences else None
    final_findings = deterministic_final + grounded_final + review_items + advisories
    unique = {(row.finding_type, row.start, row.end, row.detector, row.disposition): row
              for row in final_findings}
    result = PrivacyResult(
        application_id=document.application_id, document_id=document.document_id,
        release_status=status, reason_codes=reasons, findings=list(unique.values()),
        replacement_count=len(redactions), repair_attempts=1 if repair_count else 0,
        model_qa_confidence=model_confidence,
        content_sha256=(sha256_bytes(repaired.encode("utf-8"))
                        if status == "validated_candidate" else None),
        advisory_count=len(advisories), confirmed_finding_count=(
            len(deterministic_final) + len(grounded_final)), validation_passes=3,
    )
    raw_responses = (first_metrics["raw_responses"] + coverage_metrics["raw_responses"]
                     + final_metrics["raw_responses"])
    return repaired, result, redactions, {
        "calls": (first_metrics["calls"] + coverage_metrics["calls"]
                  + final_metrics["calls"]),
        "repairs": 1 if repair_count else 0,
        "repair_span_count": repair_count,
        "input_tokens": (first_metrics["input_tokens"] + coverage_metrics["input_tokens"]
                         + final_metrics["input_tokens"]),
        "output_tokens": (first_metrics["output_tokens"] + coverage_metrics["output_tokens"]
                          + final_metrics["output_tokens"]),
        "inference_seconds": (first_metrics["inference_seconds"]
                              + coverage_metrics["inference_seconds"]
                              + final_metrics["inference_seconds"]),
        "candidate_content_sha256": candidate_hash,
        "final_content_sha256": sha256_bytes(repaired.encode("utf-8")),
        "unsupported_advisory_count": len(advisories),
        "confirmed_residual_count": len(deterministic_final) + len(grounded_final),
        "_raw_responses": raw_responses,
    }


def privacy_aliases(extraction: CVExtraction | Any) -> list[str]:
    return [mention.text for mention in extraction.mentions]
