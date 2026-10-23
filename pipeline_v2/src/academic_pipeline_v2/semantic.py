from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from . import SCHEMA_VERSION
from .detectors import deterministic_spans, section_inventory
from .extraction import page_for_offset, text_hash
from .inference import InferenceBackend
from .models import (
    CVExtraction, CanonicalDocument, EducationItem, EducationSection, EmploymentItem,
    EmploymentSection, EntityMention, EvidenceSpan, ExtractedField, LetterExtraction,
    ProposedRelationship, PublicationAuthor, PublicationItem, PublicationSection, ValueStatus,
)
from .prompts import CV_PROMPT, LETTER_PROMPT, REPAIR_PROMPT
from .util import stable_id


class RawModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RawField(RawModel):
    value: str | int | float | bool | None
    status: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    start: int | None = None
    end: int | None = None


class RawEducation(RawModel):
    institution: RawField
    degree_type: RawField
    field: RawField
    start_year: RawField
    end_year: RawField
    graduation_year: RawField
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    start: int | None = None
    end: int | None = None


class RawEmployment(RawModel):
    organization: RawField
    original_title: RawField
    job_category: RawField
    position_type: RawField
    start_date: RawField
    end_date: RawField
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    start: int | None = None
    end: int | None = None


class RawAuthor(RawModel):
    name: RawField
    author_position: int | None = None
    is_applicant: bool | None = None


class RawPublication(RawModel):
    title: RawField
    doi: RawField
    publication_type: RawField
    year: RawField
    venue: RawField
    authors: list[RawAuthor] = Field(default_factory=list)
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    start: int | None = None
    end: int | None = None


class RawEducationSection(RawModel):
    status: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    not_found_reason: str | None = None
    items: list[RawEducation] = Field(default_factory=list)


class RawEmploymentSection(RawModel):
    status: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    not_found_reason: str | None = None
    items: list[RawEmployment] = Field(default_factory=list)


class RawPublicationSection(RawModel):
    status: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    not_found_reason: str | None = None
    items: list[RawPublication] = Field(default_factory=list)


class RawSections(RawModel):
    education: RawEducationSection
    employment: RawEmploymentSection
    publications: RawPublicationSection


class RawMention(RawModel):
    identifier_type: str
    person_role: str = "unknown"
    text: str
    start: int
    end: int
    model_confidence: float | None = Field(default=None, ge=0, le=1)


class RawCV(RawModel):
    sections: RawSections
    research_fields: list[RawField] = Field(default_factory=list)
    research_methods: list[RawField] = Field(default_factory=list)
    mentions: list[RawMention] = Field(default_factory=list)


class RawRelationship(RawModel):
    relationship_type: str
    source_start: int
    target_start: int
    model_confidence: float | None = Field(default=None, ge=0, le=1)


class RawLetter(RawModel):
    status: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    mentions: list[RawMention] = Field(default_factory=list)
    relationships: list[RawRelationship] = Field(default_factory=list)


def _chunks(text: str, maximum_chars: int) -> list[tuple[int, str]]:
    if len(text) <= maximum_chars:
        return [(0, text)]
    rows: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + maximum_chars)
        if end < len(text):
            boundary = text.rfind("\n", start + maximum_chars // 2, end)
            if boundary > start:
                end = boundary + 1
        rows.append((start, text[start:end]))
        start = end
    return rows


def _shift_offsets(value: Any, offset: int) -> Any:
    if isinstance(value, dict):
        shifted = {}
        for key, item in value.items():
            if key in {"start", "end", "source_start", "target_start"} and isinstance(item, int):
                shifted[key] = item + offset
            else:
                shifted[key] = _shift_offsets(item, offset)
        return shifted
    if isinstance(value, list):
        return [_shift_offsets(item, offset) for item in value]
    return value


def _merge_raw_cv(chunks: list[RawCV]) -> RawCV:
    sections: dict[str, Any] = {}
    classes = {"education": RawEducationSection, "employment": RawEmploymentSection,
               "publications": RawPublicationSection}
    for name in ("education", "employment", "publications"):
        sources = [getattr(chunk.sections, name) for chunk in chunks]
        items = [item for source in sources for item in source.items]
        found = next((source for source in sources if source.status == "found"), None)
        failed = next((source for source in sources if source.status == "model_failed"), None)
        sections[name] = classes[name](
            status="found" if items else ("model_failed" if failed else "not_found"),
            model_confidence=(found or failed or (sources[0] if sources else None)).model_confidence
            if sources else None,
            not_found_reason=None if items else f"No {name} items found across chunks",
            items=items,
        )
    return RawCV(
        sections=RawSections.model_validate(sections),
        research_fields=[item for chunk in chunks for item in chunk.research_fields],
        research_methods=[item for chunk in chunks for item in chunk.research_methods],
        mentions=[item for chunk in chunks for item in chunk.mentions],
    )


def _validated_completion(backend: InferenceBackend, task: str, prompt: str,
                          contract: type[RawModel], max_tokens: int) -> tuple[RawModel, list[Any]]:
    responses = []
    response = backend.complete(task, prompt, contract.model_json_schema(), max_tokens)
    responses.append(response)
    try:
        return contract.model_validate(response.payload), responses
    except Exception:
        correction = (prompt + "\nVALIDATION_CORRECTION: The previous JSON did not match the schema. "
                      "Return a complete, shorter schema-valid object; do not add prose.")
        response = backend.complete(task, correction, contract.model_json_schema(), max_tokens)
        responses.append(response)
        return contract.model_validate(response.payload), responses


def _status(value: str, allow_not_applicable: bool = False) -> ValueStatus:
    allowed = {item.value for item in ValueStatus}
    if not allow_not_applicable:
        allowed.discard(ValueStatus.NOT_APPLICABLE.value)
    return ValueStatus(value) if value in allowed else ValueStatus.MODEL_FAILED


def _evidence(document: CanonicalDocument, start: int | None, end: int | None,
              expected: str | None = None) -> tuple[list[EvidenceSpan], list[str]]:
    flags: list[str] = []
    if start is None or end is None or start < 0 or end <= start or end > len(document.text):
        if expected:
            matches = []
            cursor = 0
            while True:
                position = document.text.casefold().find(expected.casefold(), cursor)
                if position < 0:
                    break
                matches.append(position)
                cursor = position + max(1, len(expected))
            if len(matches) == 1:
                start, end = matches[0], matches[0] + len(expected)
                flags.append("offset_recovered_by_unique_exact_match")
            else:
                return [], ["ungrounded_value"]
        else:
            return [], ["missing_evidence"]
    source = document.text[start:end]
    if expected and source.casefold() != expected.casefold():
        return [], ["evidence_text_mismatch"]
    page = page_for_offset(document, start, end)
    if page is None:
        return [], ["evidence_crosses_page_boundary"]
    method = None if page.extraction_method == "failed" else page.extraction_method
    return [EvidenceSpan(page=page.page, start=start, end=end, text_sha256=text_hash(source),
                         extraction_method=method)], flags


def _field(document: CanonicalDocument, raw: RawField, derived: bool = False) -> ExtractedField:
    status = _status(raw.status, allow_not_applicable=True)
    value = raw.value
    evidence: list[EvidenceSpan] = []
    flags: list[str] = []
    if status == ValueStatus.FOUND and value is not None and not derived:
        evidence, flags = _evidence(document, raw.start, raw.end, str(value))
        if not evidence:
            status = ValueStatus.UNCERTAIN
    elif derived:
        flags.append("controlled_or_derived_value")
    return ExtractedField(value=value, status=status, model_confidence=raw.model_confidence,
                          evidence=evidence, validation_flags=flags)


def _item_evidence(document: CanonicalDocument, start: int | None,
                   end: int | None) -> list[EvidenceSpan]:
    evidence, _ = _evidence(document, start, end)
    return evidence


def _mentions(document: CanonicalDocument, raw_mentions: list[RawMention]) -> list[EntityMention]:
    candidates: list[tuple[RawMention, str]] = [(row, "model") for row in raw_mentions]
    for row in deterministic_spans(document.text):
        candidates.append((RawMention(identifier_type=row.identifier_type, person_role="unknown",
                                      text=row.text, start=row.start, end=row.end,
                                      model_confidence=None), "regex"))
    accepted: dict[tuple[int, int, str], EntityMention] = {}
    allowed_types = set(EntityMention.model_fields["identifier_type"].annotation.__args__)
    allowed_roles = set(EntityMention.model_fields["person_role"].annotation.__args__)
    for raw, detector in candidates:
        if raw.identifier_type not in allowed_types:
            continue
        evidence, _ = _evidence(document, raw.start, raw.end, raw.text)
        if not evidence:
            continue
        role = raw.person_role if raw.person_role in allowed_roles else "unknown"
        key = (raw.start, raw.end, raw.identifier_type)
        mention = EntityMention(
            mention_id=stable_id("mention", f"{document.document_id}:{raw.start}:{raw.end}:{raw.identifier_type}"),
            identifier_type=raw.identifier_type,
            person_role=role,
            text=raw.text,
            model_confidence=raw.model_confidence,
            detector=detector,
            evidence=evidence[0],
            grounded=True,
            local_entity_id=stable_id("local", f"{document.document_id}:{role}:{raw.text.casefold()}"),
        )
        current = accepted.get(key)
        if current is None or (current.person_role == "unknown" and role != "unknown"):
            accepted[key] = mention
    return sorted(accepted.values(), key=lambda row: (row.evidence.start, row.evidence.end))


def _empty_field() -> RawField:
    return RawField(value=None, status="model_failed", model_confidence=None)


def _parse_cv(document: CanonicalDocument, raw: RawCV,
              inventory: dict[str, dict[str, int | bool]]) -> CVExtraction:
    flags: list[str] = []
    sections = raw.sections

    education_items: list[EducationItem] = []
    for index, item in enumerate(sections.education.items):
        education_items.append(EducationItem(
            item_id=stable_id("education", f"{document.document_id}:{index}:{item.start}"),
            degree_type=_field(document, item.degree_type), field=_field(document, item.field),
            institution=_field(document, item.institution),
            start_year=_field(document, item.start_year), end_year=_field(document, item.end_year),
            graduation_year=_field(document, item.graduation_year),
            item_model_confidence=item.model_confidence,
            evidence=_item_evidence(document, item.start, item.end),
        ))
    employment_items: list[EmploymentItem] = []
    for index, item in enumerate(sections.employment.items):
        employment_items.append(EmploymentItem(
            item_id=stable_id("employment", f"{document.document_id}:{index}:{item.start}"),
            organization=_field(document, item.organization),
            original_title=_field(document, item.original_title),
            job_category=_field(document, item.job_category, derived=True),
            position_type=_field(document, item.position_type, derived=True),
            start_date=_field(document, item.start_date), end_date=_field(document, item.end_date),
            item_model_confidence=item.model_confidence,
            evidence=_item_evidence(document, item.start, item.end),
        ))
    publication_items: list[PublicationItem] = []
    for index, item in enumerate(sections.publications.items):
        publication_items.append(PublicationItem(
            item_id=stable_id("publication", f"{document.document_id}:{index}:{item.start}"),
            title=_field(document, item.title), doi=_field(document, item.doi),
            publication_type=_field(document, item.publication_type, derived=True),
            year=_field(document, item.year), venue=_field(document, item.venue),
            authors=[PublicationAuthor(name=_field(document, author.name),
                                       author_position=author.author_position,
                                       is_applicant=author.is_applicant)
                     for author in item.authors],
            item_model_confidence=item.model_confidence,
            evidence=_item_evidence(document, item.start, item.end),
        ))
    section_items = {"education": education_items, "employment": employment_items,
                     "publications": publication_items}
    section_objects: dict[str, Any] = {}
    section_classes = {"education": EducationSection, "employment": EmploymentSection,
                       "publications": PublicationSection}
    for name in ("education", "employment", "publications"):
        source = getattr(sections, name)
        validation_flags = []
        candidate_count = int(inventory.get(name, {}).get("candidate_entry_count", 0))
        if bool(inventory.get(name, {}).get("heading_found")) and not section_items[name]:
            validation_flags.append("detected_heading_without_extracted_items")
        if candidate_count and len(section_items[name]) < candidate_count:
            validation_flags.append("candidate_entry_coverage_gap")
        flags.extend(f"{name}:{flag}" for flag in validation_flags)
        section_objects[name] = section_classes[name](
            status=_status(source.status), model_confidence=source.model_confidence,
            not_found_reason=source.not_found_reason, items=section_items[name],
            validation_flags=validation_flags,
        )
    mentions = _mentions(document, raw.mentions)
    protected_fields: list[tuple[ExtractedField, str, str]] = []
    for item in education_items:
        protected_fields.append((item.institution, "organization", "institution"))
    for item in employment_items:
        protected_fields.append((item.organization, "organization", "institution"))
        protected_fields.append((item.original_title, "work_title", "unknown"))
    for item in publication_items:
        protected_fields.extend(((item.title, "work_title", "unknown"),
                                 (item.doi, "unique_identifier", "unknown"),
                                 (item.venue, "organization", "institution")))
        protected_fields.extend((author.name, "person_name",
                                 "applicant" if author.is_applicant else "coauthor")
                                for author in item.authors)
    existing = {(row.evidence.start, row.evidence.end, row.identifier_type) for row in mentions}
    for field, identifier_type, role in protected_fields:
        if field.value is None or not field.evidence:
            continue
        evidence = field.evidence[0]
        key = (evidence.start, evidence.end, identifier_type)
        if key in existing:
            continue
        mentions.append(EntityMention(
            mention_id=stable_id("mention", f"{document.document_id}:{evidence.start}:"
                                 f"{evidence.end}:{identifier_type}"),
            identifier_type=identifier_type, person_role=role, text=str(field.value),
            model_confidence=field.model_confidence, detector="model", evidence=evidence,
            grounded=True,
            local_entity_id=stable_id("local", f"{document.document_id}:{role}:"
                                      f"{str(field.value).casefold()}"),
        ))
        existing.add(key)
    mentions.sort(key=lambda row: (row.evidence.start, row.evidence.end))
    return CVExtraction(
        application_id=document.application_id, document_id=document.document_id,
        education=section_objects["education"], employment=section_objects["employment"],
        publications=section_objects["publications"],
        research_fields=[_field(document, field) for field in raw.research_fields],
        research_methods=[_field(document, field) for field in raw.research_methods],
        mentions=mentions, extraction_flags=sorted(set(flags)),
    )


def extract_cv(document: CanonicalDocument, backend: InferenceBackend,
               max_tokens: int, maximum_repairs: int,
               context_length: int = 32768) -> tuple[CVExtraction, dict[str, Any]]:
    inventory = section_inventory(document.text)
    chunks = _chunks(document.text, max(4000, int(context_length * 2.5)))
    responses = []
    raw_chunks = []
    for offset, chunk in chunks:
        prompt = CV_PROMPT.format(inventory=json.dumps(inventory, sort_keys=True), text=chunk)
        validated, attempts = _validated_completion(backend, "cv", prompt, RawCV, max_tokens)
        responses.extend(attempts)
        raw_chunks.append(RawCV.model_validate(_shift_offsets(validated.model_dump(), offset)))
    raw = _merge_raw_cv(raw_chunks)
    result = _parse_cv(document, raw, inventory)
    repairs = 0
    while len(chunks) == 1 and result.extraction_flags and repairs < maximum_repairs:
        issues = ",".join(result.extraction_flags)
        validated, attempts = _validated_completion(
            backend, "cv", REPAIR_PROMPT.format(issues=issues, text=document.text), RawCV, max_tokens)
        responses.extend(attempts)
        candidate = _parse_cv(document, validated, inventory)
        repairs += 1
        if len(candidate.extraction_flags) < len(result.extraction_flags):
            result = candidate
        else:
            break
    return result, {"calls": len(responses), "repairs": repairs,
                    "input_tokens": sum(row.input_tokens or 0 for row in responses),
                    "output_tokens": sum(row.output_tokens or 0 for row in responses),
                    "inference_seconds": sum(row.latency_seconds for row in responses),
                    "chunk_count": len(chunks),
                    "_raw_responses": [row.payload for row in responses]}


def extract_letter(document: CanonicalDocument, backend: InferenceBackend,
                   max_tokens: int, context_length: int = 32768) -> tuple[LetterExtraction, dict[str, Any]]:
    responses = []
    raw_chunks = []
    for offset, chunk in _chunks(document.text, max(4000, int(context_length * 2.5))):
        validated, attempts = _validated_completion(
            backend, "letter", LETTER_PROMPT.format(text=chunk), RawLetter, max_tokens)
        responses.extend(attempts)
        raw_chunks.append(RawLetter.model_validate(_shift_offsets(validated.model_dump(), offset)))
    raw = RawLetter(
        status="found" if any(row.status == "found" for row in raw_chunks) else "uncertain",
        model_confidence=next((row.model_confidence for row in raw_chunks
                               if row.model_confidence is not None), None),
        mentions=[item for row in raw_chunks for item in row.mentions],
        relationships=[item for row in raw_chunks for item in row.relationships],
    )
    mentions = _mentions(document, raw.mentions)
    mention_by_start = {mention.evidence.start: mention for mention in mentions}
    relationships: list[ProposedRelationship] = []
    for row in raw.relationships:
        source = mention_by_start.get(row.source_start)
        target = mention_by_start.get(row.target_start)
        if source and target and row.relationship_type in {
            "wrote_recommendation_for", "listed_as_reference_by", "educated_at",
            "worked_at", "authored", "published_in",
        }:
            relationships.append(ProposedRelationship(
                relationship_type=row.relationship_type,
                source_mention_id=source.mention_id,
                target_mention_id=target.mention_id,
                model_confidence=row.model_confidence,
                evidence=[source.evidence, target.evidence],
            ))
    result = LetterExtraction(
        application_id=document.application_id, document_id=document.document_id,
        mentions=mentions, relationships=relationships,
        extraction_status=_status(raw.status), model_confidence=raw.model_confidence,
    )
    return result, {"calls": len(responses), "repairs": 0,
                    "input_tokens": sum(row.input_tokens or 0 for row in responses),
                    "output_tokens": sum(row.output_tokens or 0 for row in responses),
                    "inference_seconds": sum(row.latency_seconds for row in responses),
                    "chunk_count": len(responses),
                    "_raw_responses": [row.payload for row in responses]}
