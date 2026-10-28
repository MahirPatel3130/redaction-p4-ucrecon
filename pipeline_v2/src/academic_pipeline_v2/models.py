from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import SCHEMA_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentType(str, Enum):
    CV = "cv"
    LETTER = "recommendation_letter"


class ValueStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"
    MODEL_FAILED = "model_failed"


class DocumentRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    application_id: str
    document_id: str
    document_type: DocumentType
    letter_index: int | None = None
    source_path: str
    source_relative_path: str
    source_sha256: str
    inventory_flags: list[str] = Field(default_factory=list)


class ApplicationRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    application_id: str
    source_root_namespace: str
    source_application_path: str
    status: Literal["inventoried", "inventory_flagged"]
    inventory_flags: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


class TextBlock(StrictModel):
    block_id: str
    page: int = Field(ge=1)
    page_start: int = Field(ge=0)
    page_end: int = Field(ge=0)
    document_start: int = Field(ge=0)
    document_end: int = Field(ge=0)
    text: str
    bbox: tuple[float, float, float, float] | None = None
    extraction_method: Literal["native", "ocr"]
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def valid_offsets(self) -> "TextBlock":
        if self.page_end < self.page_start or self.document_end < self.document_start:
            raise ValueError("invalid block offsets")
        return self


class CanonicalPage(StrictModel):
    page: int = Field(ge=1)
    text: str
    document_start: int = Field(ge=0)
    document_end: int = Field(ge=0)
    extraction_method: Literal["native", "ocr", "failed"]
    extraction_status: ValueStatus
    quality_flags: list[str] = Field(default_factory=list)
    blocks: list[TextBlock] = Field(default_factory=list)


class CanonicalDocument(StrictModel):
    schema_version: str = SCHEMA_VERSION
    application_id: str
    document_id: str
    document_type: DocumentType
    letter_index: int | None = None
    source_sha256: str
    source_reference: str
    text: str
    pages: list[CanonicalPage]
    extraction_status: ValueStatus
    extraction_flags: list[str] = Field(default_factory=list)


class EvidenceSpan(StrictModel):
    page: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text_sha256: str
    bbox: tuple[float, float, float, float] | None = None
    extraction_method: Literal["native", "ocr"] | None = None

    @model_validator(mode="after")
    def valid_span(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError("evidence end must be greater than start")
        return self


class ExtractedField(StrictModel):
    value: str | int | float | bool | None
    status: ValueStatus
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    validation_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_consistency(self) -> "ExtractedField":
        if self.status == ValueStatus.MODEL_FAILED and self.model_confidence is not None:
            raise ValueError("model_failed confidence must be null")
        if self.status == ValueStatus.FOUND and self.value is None:
            raise ValueError("found field requires a value")
        return self


class EntityMention(StrictModel):
    mention_id: str
    identifier_type: Literal[
        "person_name", "organization", "work_title", "email", "phone_number",
        "postal_address", "website", "unique_identifier", "signature", "date"
    ]
    person_role: Literal[
        "applicant", "recommender", "reference", "coauthor", "recipient",
        "advisor", "committee_member", "third_party", "institution", "unknown"
    ] = "unknown"
    text: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    detector: Literal["model", "regex", "alias", "privacy_repair"]
    evidence: EvidenceSpan
    grounded: bool = False
    local_entity_id: str | None = None
    resolved_entity_id: str | None = None


class EducationItem(StrictModel):
    item_id: str
    degree_type: ExtractedField
    field: ExtractedField
    institution: ExtractedField
    start_year: ExtractedField
    end_year: ExtractedField
    graduation_year: ExtractedField
    item_model_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class EmploymentItem(StrictModel):
    item_id: str
    organization: ExtractedField
    original_title: ExtractedField
    job_category: ExtractedField
    position_type: ExtractedField
    start_date: ExtractedField
    end_date: ExtractedField
    item_model_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class PublicationAuthor(StrictModel):
    name: ExtractedField
    author_position: int | None = Field(default=None, ge=1)
    is_applicant: bool | None = None


class PublicationItem(StrictModel):
    item_id: str
    title: ExtractedField
    doi: ExtractedField
    publication_type: ExtractedField
    year: ExtractedField
    venue: ExtractedField
    authors: list[PublicationAuthor] = Field(default_factory=list)
    item_model_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class Section(StrictModel):
    status: ValueStatus
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    not_found_reason: str | None = None
    validation_flags: list[str] = Field(default_factory=list)


class EducationSection(Section):
    items: list[EducationItem] = Field(default_factory=list)


class EmploymentSection(Section):
    items: list[EmploymentItem] = Field(default_factory=list)


class PublicationSection(Section):
    items: list[PublicationItem] = Field(default_factory=list)


class CVExtraction(StrictModel):
    schema_version: str = SCHEMA_VERSION
    application_id: str
    document_id: str
    education: EducationSection
    employment: EmploymentSection
    publications: PublicationSection
    research_fields: list[ExtractedField] = Field(default_factory=list)
    research_methods: list[ExtractedField] = Field(default_factory=list)
    mentions: list[EntityMention] = Field(default_factory=list)
    extraction_flags: list[str] = Field(default_factory=list)


class ProposedRelationship(StrictModel):
    relationship_type: Literal[
        "wrote_recommendation_for", "listed_as_reference_by", "educated_at",
        "worked_at", "authored", "published_in"
    ]
    source_mention_id: str
    target_mention_id: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class LetterExtraction(StrictModel):
    schema_version: str = SCHEMA_VERSION
    application_id: str
    document_id: str
    mentions: list[EntityMention] = Field(default_factory=list)
    relationships: list[ProposedRelationship] = Field(default_factory=list)
    extraction_status: ValueStatus
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    extraction_flags: list[str] = Field(default_factory=list)


class ResolvedRelationship(StrictModel):
    relationship_id: str
    relationship_type: str
    source_id: str
    target_id: str
    application_id: str
    document_id: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    validation_status: Literal["validated", "uncertain", "rejected"]


class PrivacyFinding(StrictModel):
    finding_type: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    detector: Literal["substitution", "regex", "alias", "model", "package"]
    grounded: bool = False
    disposition: Literal["confirmed", "review_required", "advisory"] = "confirmed"
    model_confidence: float | None = Field(default=None, ge=0, le=1)


class PrivacyResult(StrictModel):
    schema_version: str = SCHEMA_VERSION
    application_id: str
    document_id: str
    release_status: Literal["validated_candidate", "review_required", "withheld"]
    reason_codes: list[str] = Field(default_factory=list)
    findings: list[PrivacyFinding] = Field(default_factory=list)
    replacement_count: int = Field(0, ge=0)
    repair_attempts: int = Field(0, ge=0)
    model_qa_confidence: float | None = Field(default=None, ge=0, le=1)
    content_sha256: str | None = None
    advisory_count: int = Field(0, ge=0)
    confirmed_finding_count: int = Field(0, ge=0)
    validation_passes: int = Field(0, ge=0)


class DocumentCheckpoint(StrictModel):
    schema_version: str = SCHEMA_VERSION
    pipeline_version: str
    document_id: str
    source_sha256: str
    config_fingerprint: str
    stages: dict[str, str] = Field(default_factory=dict)
    stage_fingerprints: dict[str, str] = Field(default_factory=dict)
    artifact_checksums: dict[str, str] = Field(default_factory=dict)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    timings_seconds: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    failure_stage: str | None = None
    failure_code: str | None = None
    updated_at: str


class QualityRecord(StrictModel):
    application_id: str
    document_id: str
    document_type: DocumentType
    processing_status: str
    extraction_status: str
    privacy_status: str
    release_status: str
    repair_count: int = 0
    failure_code: str | None = None
    validation_flags: list[str] = Field(default_factory=list)


class ModelResponse(StrictModel):
    payload: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float
    retry_count: int = 0
