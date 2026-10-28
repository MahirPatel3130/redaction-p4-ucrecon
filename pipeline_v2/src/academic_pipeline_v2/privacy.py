from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .detectors import deterministic_spans
from .inference import InferenceBackend
from .models import (
    CVExtraction, CanonicalDocument, EntityMention, PrivacyFinding, PrivacyResult,
)
from .ner import NERBackend, NERSpan, NullNERBackend
from .prompts import NER_CLASSIFICATION_PROMPT
from .util import sha256_bytes, stable_id


IdentifierType = Literal[
    "person_name", "organization", "work_title", "email", "phone_number",
    "postal_address", "website", "unique_identifier", "signature", "date",
]


class RawCandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    decision: Literal["replace", "preserve", "review"]
    identifier_type: IdentifierType
    model_confidence: float = Field(ge=0, le=1)
    reason_code: Literal["named_identifier", "generic_or_retained", "ambiguous"]


class RawCandidateReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[RawCandidateDecision] = Field(default_factory=list, max_length=100)


PRIORITY = {
    "email": 100, "unique_identifier": 95, "phone_number": 90,
    "postal_address": 85, "website": 80, "signature": 75,
    "person_name": 70, "organization": 60, "work_title": 50, "date": 10,
    "protected_alias": 65,
}

PLACEHOLDER_PATTERN = re.compile(r"\[\[[^\]]+\]\]")


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
    for alias in sorted(set(aliases), key=len, reverse=True):
        alias = alias.strip()
        if len(alias) < 3 or alias.isdigit():
            continue
        for start, end in _alias_occurrences(masked, alias):
            findings.append(PrivacyFinding(finding_type="protected_alias", start=start,
                                           end=end, detector="alias", grounded=True,
                                           disposition="confirmed"))
    unique = {(row.finding_type, row.start, row.end): row for row in findings}
    return list(unique.values())


def _overlaps_placeholder(text: str, start: int, end: int) -> bool:
    return any(start < match.end() and end > match.start() for match in PLACEHOLDER_PATTERN.finditer(text))


def _approved_types(policy: dict[str, Any] | None) -> set[str]:
    if policy and isinstance(policy.get("replace_identifier_types"), list):
        return {str(value) for value in policy["replace_identifier_types"]}
    return {"person_name", "organization", "work_title", "email", "phone_number",
            "postal_address", "website", "unique_identifier", "signature"}


def _alias_occurrences(text: str, alias: str) -> list[tuple[int, int]]:
    """Return complete-token alias matches, never arbitrary substrings."""
    tokens = alias.split()
    if not tokens:
        return []
    body = r"\s+".join(re.escape(token) for token in tokens)
    left = r"(?<!\w)" if tokens[0][0].isalnum() else ""
    right = r"(?!\w)" if tokens[-1][-1].isalnum() else ""
    return [(match.start(), match.end())
            for match in re.finditer(left + body + right, text, flags=re.IGNORECASE)]


def _alias_repairs(text: str, redactions: list[dict[str, Any]]) -> list[RepairSpan]:
    result: list[RepairSpan] = []
    masked = _masked(text)
    for row in redactions:
        if row.get("detector") == "privacy_repair":
            continue
        alias = str(row.get("text", "")).strip()
        if len(alias) < 3 or alias.isdigit():
            continue
        for start, end in _alias_occurrences(masked, alias):
            result.append(RepairSpan(
                start, end, str(row.get("identifier_type") or "protected_alias"),
                "alias", row.get("model_confidence"), str(row.get("person_role") or "unknown"),
                row.get("resolved_entity_id"), row.get("placeholder"),
            ))
    return result


def _expand_classified_repairs(text: str, spans: list[RepairSpan]) -> list[RepairSpan]:
    """Apply a classified named value to every complete-token occurrence in the document."""
    result = list(spans)
    for row in spans:
        value = text[row.start:row.end].strip()
        if len(value) < 3 or value.isdigit():
            continue
        for start, end in _alias_occurrences(_masked(text), value):
            result.append(RepairSpan(start, end, row.identifier_type, row.detector,
                                     row.model_confidence, row.person_role,
                                     row.resolved_entity_id, row.replacement))
    unique = {(row.start, row.end, row.identifier_type): row for row in result}
    return list(unique.values())


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
            "detector": "privacy_repair", "candidate_source": row.detector, "grounded": True,
        })
    return text, len(selected)


def _valid_ner_spans(text: str, rows: list[NERSpan]) -> list[NERSpan]:
    unique: dict[tuple[int, int, str], NERSpan] = {}
    for row in rows:
        if not (0 <= row.start < row.end <= len(text)):
            continue
        if text[row.start:row.end] != row.text or _overlaps_placeholder(text, row.start, row.end):
            continue
        unique[(row.start, row.end, row.identifier_type)] = row
    return sorted(unique.values(), key=lambda row: (row.start, row.end, row.identifier_type))


def _classify_ner_candidates(document: CanonicalDocument, text: str, rows: list[NERSpan],
                             backend: InferenceBackend, max_tokens: int, batch_size: int,
                             approved: set[str], pass_name: str) -> tuple[
                                 list[RepairSpan], list[PrivacyFinding], dict[str, Any]]:
    accepted: list[RepairSpan] = []
    findings: list[PrivacyFinding] = []
    metrics: dict[str, Any] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                               "inference_seconds": 0.0, "raw_responses": [],
                               "model_confidences": [], "preserved": 0}
    candidates: list[dict[str, Any]] = []
    row_by_id: dict[str, NERSpan] = {}
    for row in _valid_ner_spans(text, rows):
        candidate_id = stable_id(
            "candidate", f"{document.document_id}:{row.start}:{row.end}:{row.identifier_type}")
        row_by_id[candidate_id] = row
        candidates.append({
            "candidate_id": candidate_id,
            "proposed_type": row.identifier_type,
            "text": row.text,
            "context": text[max(0, row.start - 80):min(len(text), row.end + 80)],
            "ner_backend": row.backend,
            "ner_score": row.score,
        })
    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]
        prompt = NER_CLASSIFICATION_PROMPT.format(
            replace_types=", ".join(sorted(approved)),
            candidates=json.dumps(batch, ensure_ascii=False),
        )
        response = backend.complete(
            "privacy_candidate", prompt, RawCandidateReview.model_json_schema(), max_tokens)
        review = RawCandidateReview.model_validate(response.payload)
        metrics["calls"] += 1
        metrics["input_tokens"] += response.input_tokens or 0
        metrics["output_tokens"] += response.output_tokens or 0
        metrics["inference_seconds"] += response.latency_seconds
        metrics["raw_responses"].append({"pass": pass_name, "payload": response.payload})
        decisions = {row.candidate_id: row for row in review.decisions
                     if row.candidate_id in row_by_id}
        for candidate in batch:
            candidate_id = candidate["candidate_id"]
            source = row_by_id[candidate_id]
            decision = decisions.get(candidate_id)
            if decision is None:
                findings.append(PrivacyFinding(
                    finding_type="ner_candidate_unclassified", start=source.start, end=source.end,
                    detector="model", grounded=True, disposition="review_required",
                ))
                continue
            metrics["model_confidences"].append(decision.model_confidence)
            if decision.decision == "replace" and decision.identifier_type in approved:
                accepted.append(RepairSpan(
                    source.start, source.end, decision.identifier_type, "ner_model",
                    decision.model_confidence,
                ))
                findings.append(PrivacyFinding(
                    finding_type=decision.identifier_type, start=source.start, end=source.end,
                    detector="model", grounded=True, disposition="confirmed",
                    model_confidence=decision.model_confidence,
                ))
            elif decision.decision == "review":
                findings.append(PrivacyFinding(
                    finding_type="ambiguous_ner_candidate", start=source.start, end=source.end,
                    detector="model", grounded=True, disposition="review_required",
                    model_confidence=decision.model_confidence,
                ))
            else:
                metrics["preserved"] += 1
    unique_accepted = {(row.start, row.end, row.identifier_type): row for row in accepted}
    unique_findings = {(row.finding_type, row.start, row.end, row.disposition): row
                       for row in findings}
    return list(unique_accepted.values()), list(unique_findings.values()), metrics


def validate_privacy(document: CanonicalDocument, deidentified_text: str,
                     redactions: list[dict[str, Any]], backend: InferenceBackend,
                     max_tokens: int, maximum_repairs: int,
                     policy: dict[str, Any] | None = None,
                     ner_backend: NERBackend | None = None,
                     ner_candidates_per_call: int = 40) -> tuple[
                         str, PrivacyResult, list[dict[str, Any]], dict[str, Any]]:
    """Run a finite structured/NER gate with one deterministic replacement batch."""
    aliases = [str(row["text"]) for row in redactions if row.get("detector") != "privacy_repair"]
    approved = _approved_types(policy)
    candidate_hash = sha256_bytes(deidentified_text.encode("utf-8"))
    ner_backend = ner_backend or NullNERBackend()
    authoritative = _deterministic_repairs(deidentified_text) + _alias_repairs(
        deidentified_text, redactions)
    ner_input = _mask_spans_preserving_offsets(deidentified_text, authoritative)
    ner_rows = ner_backend.detect(ner_input)
    ner_repairs, _, first_metrics = _classify_ner_candidates(
        document, ner_input, ner_rows, backend, max_tokens, ner_candidates_per_call,
        approved, "candidate_classification")
    ner_repairs = _expand_classified_repairs(deidentified_text, ner_repairs)
    repair_spans = authoritative + ner_repairs
    repaired = deidentified_text
    repair_count = 0
    if repair_spans and maximum_repairs > 0:
        repaired, repair_count = _apply_repair_batch(
            document, deidentified_text, repair_spans, redactions)

    deterministic_final = _scan(repaired, aliases)
    final_ner_rows = ner_backend.detect(repaired)
    final_repairs, final_findings, final_metrics = _classify_ner_candidates(
        document, repaired, final_ner_rows, backend, max_tokens, ner_candidates_per_call,
        approved, "independent_final_classification")
    grounded_final = [row for row in final_findings if row.disposition == "confirmed"]
    review_items = [row for row in final_findings if row.disposition == "review_required"]
    ner_unavailable = getattr(ner_backend, "name", "none") == "none"

    if deterministic_final or grounded_final or final_repairs:
        status = "withheld"
        reasons = ["confirmed_residual_privacy_finding"]
    elif review_items:
        status = "review_required"
        reasons = ["ambiguous_ner_candidate_review"]
    elif ner_unavailable:
        status = "review_required"
        reasons = ["ner_validation_unavailable"]
    else:
        status = "validated_candidate"
        reasons = []

    confidences = first_metrics["model_confidences"] + final_metrics["model_confidences"]
    model_confidence = min(confidences) if confidences else None
    terminal_findings = deterministic_final + grounded_final + review_items
    if ner_unavailable:
        terminal_findings.append(PrivacyFinding(
            finding_type="ner_validation_unavailable", detector="model", grounded=False,
            disposition="review_required",
        ))
    unique = {(row.finding_type, row.start, row.end, row.detector, row.disposition): row
              for row in terminal_findings}
    result = PrivacyResult(
        application_id=document.application_id, document_id=document.document_id,
        release_status=status, reason_codes=reasons, findings=list(unique.values()),
        replacement_count=len(redactions), repair_attempts=1 if repair_count else 0,
        model_qa_confidence=model_confidence,
        content_sha256=(sha256_bytes(repaired.encode("utf-8"))
                        if status == "validated_candidate" else None),
        advisory_count=first_metrics["preserved"] + final_metrics["preserved"],
        confirmed_finding_count=(len(deterministic_final) + len(grounded_final)),
        validation_passes=2,
    )
    raw_responses = first_metrics["raw_responses"] + final_metrics["raw_responses"]
    return repaired, result, redactions, {
        "calls": first_metrics["calls"] + final_metrics["calls"],
        "repairs": 1 if repair_count else 0,
        "repair_span_count": repair_count,
        "input_tokens": first_metrics["input_tokens"] + final_metrics["input_tokens"],
        "output_tokens": first_metrics["output_tokens"] + final_metrics["output_tokens"],
        "inference_seconds": (first_metrics["inference_seconds"]
                              + final_metrics["inference_seconds"]),
        "candidate_content_sha256": candidate_hash,
        "final_content_sha256": sha256_bytes(repaired.encode("utf-8")),
        "ner_backend": getattr(ner_backend, "name", "none"),
        "ner_candidate_count": len(ner_rows),
        "ner_final_candidate_count": len(final_ner_rows),
        "ner_preserved_count": first_metrics["preserved"] + final_metrics["preserved"],
        "confirmed_residual_count": len(deterministic_final) + len(grounded_final),
        "_raw_responses": raw_responses,
    }


def privacy_aliases(extraction: CVExtraction | Any) -> list[str]:
    return [mention.text for mention in extraction.mentions]
