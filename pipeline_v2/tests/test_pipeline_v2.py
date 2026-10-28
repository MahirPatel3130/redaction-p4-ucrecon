from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pymupdf
import pytest
import httpx
from pydantic import ValidationError

from academic_pipeline_v2.audit import build_privacy_audit
from academic_pipeline_v2.config import ModelConfig, NERConfig, PipelineConfig
from academic_pipeline_v2.exporter import validate_candidate_package
from academic_pipeline_v2.evaluation import evaluate_run
from academic_pipeline_v2.extraction import extract_canonical
from academic_pipeline_v2.inventory import build_inventory
from academic_pipeline_v2.models import DocumentRecord, DocumentType
from academic_pipeline_v2.models import CanonicalDocument, CanonicalPage, ModelResponse, ValueStatus
from academic_pipeline_v2.ner import NERSpan
from academic_pipeline_v2.privacy import RawCandidateReview, _scan, validate_privacy
from academic_pipeline_v2.pipeline import (
    export_candidate_stage, finalize_review_stage, inventory_stage, process_stage, release_stage,
)
from academic_pipeline_v2.registry import EntityRegistry
from academic_pipeline_v2.inference import VLLMBackend
from academic_pipeline_v2.semantic import RawField, RawMention, _field, _mentions
from academic_pipeline_v2.util import sha256_file


def make_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 790), text, fontsize=9)
    document.save(path)
    document.close()


def make_application(root: Path, name: str = "application_001") -> Path:
    application = root / name
    application.mkdir(parents=True)
    cv = "\n".join([
        "Applicant Alice | alice@example.edu | 555-111-2222",
        "EDUCATION",
        "EDUCATION|Example University|PhD|Economics|2017|2022",
        "EMPLOYMENT",
        "EMPLOYMENT|Example University|Assistant Professor|academic_faculty|2022|present",
        "PUBLICATIONS",
        "PUBLICATION|Important Paper|journal_article|2024|Example Journal|Applicant Alice;Coauthor Carl",
    ])
    make_pdf(application / "Applicant_CV.pdf", cv)
    for index in range(1, 4):
        make_pdf(application / f"recommendation_{index}.pdf",
                 "Recommender Rita recommends Applicant Alice for this position.\n"
                 "Contact rita@example.edu at Example University.")
    return application


def config_for(tmp_path: Path) -> PipelineConfig:
    inputs = tmp_path / "applications"
    make_application(inputs)
    policy = tmp_path / "privacy.yaml"
    policy.write_text("version: '2.0'\n", encoding="utf-8")
    return PipelineConfig.model_validate({
        "input_roots": [inputs], "output_root": tmp_path / "run",
        "state_root": tmp_path / "state", "namespace": "synthetic",
        "cpu_workers": 2, "ocr": {"enabled": False},
        "model": {"backend": "mock", "model": "mock", "max_inflight": 2},
        "ner": {"enabled": True, "backend": "mock"},
        "privacy": {"policy_path": policy, "maximum_repairs": 2,
                    "review_fraction": 0.10, "review_minimum": 2,
                    "require_human_review": True},
    })


def test_inventory_and_canonical_offsets(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    applications, documents = build_inventory(config)
    assert len(applications) == 1
    assert len(documents) == 4
    assert not applications[0].inventory_flags
    cv = next(row for row in documents if row.document_type == DocumentType.CV)
    canonical = extract_canonical(cv, config.ocr)
    assert "Applicant Alice" in canonical.text
    assert canonical.pages[0].blocks
    for block in canonical.pages[0].blocks:
        assert canonical.text[block.document_start:block.document_end] == block.text


class FakeOCR:
    def extract(self, page: object, dpi: int):
        return [("Scanned Applicant Alice", (10.0, 10.0, 200.0, 30.0), 0.99)]


def test_selective_ocr_contract(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    document = pymupdf.open(); document.new_page(); document.save(pdf); document.close()
    record = DocumentRecord(application_id="application_x", document_id="document_x",
                            document_type=DocumentType.LETTER, letter_index=1,
                            source_path=str(pdf), source_relative_path="app/scan.pdf",
                            source_sha256=sha256_file(pdf))
    config = config_for(tmp_path / "configuration").ocr.model_copy(update={"enabled": True})
    canonical = extract_canonical(record, config, FakeOCR())
    assert canonical.pages[0].extraction_method == "ocr"
    assert "Scanned Applicant Alice" in canonical.text

    class LowConfidenceOCR:
        def extract(self, page: object, dpi: int):
            return [("Uncertain scan", None, 0.40)]

    low = extract_canonical(record, config, LowConfidenceOCR())
    assert low.extraction_status == ValueStatus.MODEL_FAILED
    assert "ocr_below_confidence" in low.pages[0].quality_flags


def test_end_to_end_candidate_review_release_and_no_leak(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    originals = {path: path.read_bytes() for path in config.input_roots[0].rglob("*.pdf")}
    inventory_stage(config)
    summary = process_stage(config)
    assert summary["document_count"] == 4
    assert summary["released_document_count"] == 4
    assert summary["structured_mention_count"] > 0
    assert summary["ner_candidate_count"] == 0  # structured extraction covered this fixture
    assert summary["contextual_risk"]["blocks_document_release"] is False
    contextual = json.loads((config.output_root / "protected" / "contextual_risk" /
                             "applications.json").read_text())
    assert contextual["scope"] == "application_level_nonblocking_contextual_assessment"
    assert next(iter(contextual["applications"].values()))["status"] == \
        "not_evaluated_insufficient_cohort"
    package = export_candidate_stage(config)
    validation = validate_candidate_package(config.output_root)
    assert validation["status"] == "valid"
    evaluation = evaluate_run(config)
    assert evaluation["release_rate"] == 1.0
    researcher_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore")
                                  for path in package.rglob("*") if path.is_file())
    for protected in ("Applicant Alice", "Recommender Rita", "Example University",
                      "Important Paper", "alice@example.edu", "rita@example.edu",
                      "Assistant Professor"):
        assert protected not in researcher_text
    letters = [json.loads(line) for line in
               (package / "recommendation_letters.jsonl").read_text().splitlines()]
    assert all("[[APPLICANT:" in row["text"] for row in letters)
    assert all("[[RECOMMENDER:" in row["text"] for row in letters)
    protected = (config.output_root / "protected" / "redactions.jsonl").read_text()
    assert "Applicant Alice" in protected
    assert (config.output_root / "protected").stat().st_mode & 0o777 == 0o700
    assert (config.state_root / "registry.sqlite").stat().st_mode & 0o777 == 0o600
    review = config.output_root / "protected" / "review" / "sample.csv"
    with review.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["review_decision"] = "pass"
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    finalize_review_stage(config, review)
    released = release_stage(config)
    assert (released / "manifest.json").exists()
    assert json.loads((released / "manifest.json").read_text())["release_status"] == "ready"
    for path, before in originals.items():
        assert path.read_bytes() == before


def test_resume_reuses_released_checkpoints(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    inventory_stage(config)
    first = process_stage(config)
    mtimes = {path: path.stat().st_mtime_ns for path in
              (config.output_root / "protected" / "checkpoints").rglob("semantic.json")}
    second = process_stage(config, retry_failed=True)
    assert first["released_document_count"] == second["released_document_count"]
    assert mtimes == {path: path.stat().st_mtime_ns for path in
                      (config.output_root / "protected" / "checkpoints").rglob("semantic.json")}


def test_v1_registry_import_is_explicit_and_idempotent(tmp_path: Path) -> None:
    registry = EntityRegistry(tmp_path / "registry.sqlite")
    payload = {"persons": {"person_existing": {"status": "active",
                                                "local_person_ids": ["app_x:applicant"],
                                                "roles": ["applicant"],
                                                "application_ids": ["app_x"],
                                                "document_ids": ["doc_x"],
                                                "identifiers": [{"identifier_type": "email",
                                                                 "value": "fictional@example.edu",
                                                                 "verified": True,
                                                                 "document_id": "doc_x"}]},
                           "retired": {"status": "retired"}}}
    preview = registry.import_v1(payload, "checksum", False)
    assert preview == {"active_people": 1, "applied": False}
    applied = registry.import_v1(payload, "checksum", True)
    assert applied["active_people"] == 1
    repeated = registry.import_v1(payload, "checksum", True)
    assert repeated["already_imported"] is True
    assert any(row["entity_id"] == "person_existing" for row in registry.public_entities())
    registry.close()


def test_vllm_is_loopback_only_and_retries_malformed_json() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(base_url="https://external.example/v1")
    config = ModelConfig(base_url="http://127.0.0.1:8000/v1", max_retries=1)
    backend = VLLMBackend(config)
    backend.client.close()
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = "not-json" if attempts == 1 else '{"safe":true}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    backend.client = httpx.Client(transport=httpx.MockTransport(handler))
    response = backend.complete("test", "prompt", {"type": "object"}, 32)
    backend.close()
    assert response.payload == {"safe": True}
    assert response.retry_count == 1


def test_vllm_recovers_fenced_json() -> None:
    config = ModelConfig(base_url="http://127.0.0.1:8000/v1", max_retries=0)
    backend = VLLMBackend(config)
    backend.client.close()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {
                "content": "```json\n{\"safe\": true}\n```",
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        })

    backend.client = httpx.Client(transport=httpx.MockTransport(handler))
    response = backend.complete("test", "prompt", {"type": "object"}, 32)
    backend.close()
    assert response.payload == {"safe": True}


def test_vllm_reports_truncated_output_without_content_leak() -> None:
    config = ModelConfig(base_url="http://127.0.0.1:8000/v1", max_retries=0)
    backend = VLLMBackend(config)
    backend.client.close()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "length", "message": {"content": "sensitive"}}],
        })

    backend.client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(Exception, match="model_output_truncated"):
        backend.complete("test", "prompt", {"type": "object"}, 32)
    backend.close()


def test_vllm_accepts_complete_json_with_length_finish_reason() -> None:
    config = ModelConfig(base_url="http://127.0.0.1:8000/v1", max_retries=0)
    backend = VLLMBackend(config)
    backend.client.close()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "length", "message": {
                "content": "{\"safe\": true}\n   ",
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        })

    backend.client = httpx.Client(transport=httpx.MockTransport(handler))
    response = backend.complete("test", "prompt", {"type": "object"}, 32)
    backend.close()
    assert response.payload == {"safe": True}


def test_same_verified_recommender_email_merges_across_applications(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    make_application(config.input_roots[0], "application_002")
    inventory_stage(config)
    summary = process_stage(config)
    assert summary["released_document_count"] == 8
    registry = EntityRegistry(config.state_root / "registry.sqlite")
    rows = registry.connection.execute(
        "SELECT DISTINCT source_id,application_id FROM relationships "
        "WHERE relationship_type='wrote_recommendation_for'"
    ).fetchall()
    registry.close()
    assert len({row["source_id"] for row in rows}) == 1
    assert len({row["application_id"] for row in rows}) == 2


def test_alias_scan_reports_every_repeated_occurrence() -> None:
    text = "Repeated Alias appears. Repeated Alias appears again. Repeated Alias."
    findings = [row for row in _scan(text, ["Repeated Alias"])
                if row.finding_type == "protected_alias"]
    assert len(findings) == 3
    assert [text[row.start:row.end] for row in findings] == ["Repeated Alias"] * 3


def test_alias_scan_never_matches_inside_an_unrelated_word() -> None:
    findings = [row for row in _scan("Ann wrote annually.", ["Ann"])
                if row.finding_type == "protected_alias"]
    assert len(findings) == 1


class PatternNER:
    name = "test_ner"

    def __init__(self, values: dict[str, str]):
        self.values = values

    def detect(self, text: str):
        rows = []
        for value, identifier_type in self.values.items():
            cursor = 0
            while True:
                start = text.find(value, cursor)
                if start < 0:
                    break
                rows.append(NERSpan(start, start + len(value), value, identifier_type,
                                    identifier_type, 0.98, self.name))
                cursor = start + len(value)
        return rows

    def close(self):
        return None


class CandidateBackend:
    def __init__(self, decisions: dict[str, str] | None = None):
        self.decisions = decisions or {}

    def complete(self, task, prompt, schema, max_tokens):
        assert task == "privacy_candidate"
        candidates = json.loads(prompt.rsplit("CANDIDATES_JSON:\n", 1)[1])
        rows = []
        for candidate in candidates:
            decision = self.decisions.get(candidate["text"], "replace")
            proposed = candidate["proposed_type"]
            identifier_type = "postal_address" if proposed == "location" else proposed
            rows.append({
                "candidate_id": candidate["candidate_id"], "decision": decision,
                "identifier_type": identifier_type, "model_confidence": 0.98,
                "reason_code": {"replace": "named_identifier",
                                "preserve": "generic_or_retained",
                                "review": "ambiguous"}[decision],
            })
        return ModelResponse(payload={"decisions": rows}, latency_seconds=0.01)

    def close(self):
        return None


def privacy_document(text: str) -> CanonicalDocument:
    return CanonicalDocument(
        application_id="application_x", document_id="document_x",
        document_type=DocumentType.LETTER, source_sha256="0" * 64,
        source_reference="document_x", text=text,
        pages=[CanonicalPage(page=1, text=text.split("\n", 1)[-1], document_start=11,
                             document_end=len(text), extraction_method="native",
                             extraction_status=ValueStatus.FOUND)],
        extraction_status=ValueStatus.FOUND,
    )


def test_candidate_review_schema_is_bounded_and_offset_free() -> None:
    schema = RawCandidateReview.model_json_schema()
    assert schema["properties"]["decisions"]["maxItems"] == 100
    decision = schema["$defs"]["RawCandidateDecision"]
    assert "candidate_id" in decision["required"]
    assert "start" not in decision["properties"]


def test_ner_candidate_is_classified_and_repaired_once() -> None:
    text = "[[PAGE:1]]\nSecret Alias wrote to Secret Alias."
    repaired, result, redactions, metrics = validate_privacy(
        privacy_document(text), text, [], CandidateBackend(), 256, 1, None,
        PatternNER({"Secret Alias": "person_name"}), 40)
    assert result.release_status == "validated_candidate"
    assert "Secret Alias" not in repaired
    assert len(redactions) == 2
    assert metrics["repairs"] == 1
    assert metrics["calls"] == 0
    assert metrics["ner_auto_accepted_count"] == 2


def test_classified_ner_value_expands_to_unseen_repeated_occurrence() -> None:
    class FirstOnlyNER(PatternNER):
        def detect(self, text: str):
            rows = super().detect(text)
            return rows[:1]

    text = "[[PAGE:1]]\nSecret Alias wrote to Secret Alias."
    repaired, result, redactions, _ = validate_privacy(
        privacy_document(text), text, [], CandidateBackend(), 256, 1, None,
        FirstOnlyNER({"Secret Alias": "person_name"}), 40)
    assert result.release_status == "validated_candidate"
    assert "Secret Alias" not in repaired
    assert len(redactions) == 2


def test_masked_context_scan_is_unioned_before_single_repair_batch() -> None:
    class ContextSensitiveNER:
        name = "context_sensitive"

        def detect(self, text: str):
            if "writer@example.edu" in text:
                return []
            value = "Secret Writer"
            start = text.find(value)
            return ([NERSpan(start, start + len(value), value, "person_name",
                             "PERSON", None, self.name)] if start >= 0 else [])

        def close(self):
            return None

    text = "[[PAGE:1]]\nSecret Writer writer@example.edu"
    repaired, result, redactions, metrics = validate_privacy(
        privacy_document(text), text, [], CandidateBackend(), 256, 1, None,
        ContextSensitiveNER(), 40)
    assert result.release_status == "validated_candidate"
    assert "Secret Writer" not in repaired
    assert "writer@example.edu" not in repaired
    assert metrics["repairs"] == 1
    assert len(redactions) == 2


def test_ner_candidate_can_be_preserved_without_redaction() -> None:
    text = "[[PAGE:1]]\nRiverside is a location, not a postal address."
    repaired, result, redactions, metrics = validate_privacy(
        privacy_document(text), text, [],
        CandidateBackend({"Riverside": "preserve"}), 256, 1, None,
        PatternNER({"Riverside": "location"}), 40)
    assert result.release_status == "validated_candidate"
    assert repaired == text
    assert redactions == []
    assert metrics["ner_preserved_count"] == 1


def test_ambiguous_ner_candidate_requires_review() -> None:
    text = "[[PAGE:1]]\nRiverside may be a location or part of an address."
    repaired, result, redactions, _ = validate_privacy(
        privacy_document(text), text, [], CandidateBackend({"Riverside": "review"}), 256, 1,
        None, PatternNER({"Riverside": "location"}), 40)
    assert result.release_status == "review_required"
    assert result.reason_codes == ["ambiguous_ner_candidate_review"]
    assert repaired == text
    assert redactions == []


def test_ner_residual_is_withheld_when_repair_disabled() -> None:
    text = "[[PAGE:1]]\nVisible Person remains."
    repaired, result, redactions, _ = validate_privacy(
        privacy_document(text), text, [], CandidateBackend(), 256, 0, None,
        PatternNER({"Visible Person": "person_name"}), 40)
    assert result.release_status == "withheld"
    assert result.confirmed_finding_count == 1
    assert repaired == text
    assert redactions == []


def test_placeholder_is_not_an_ner_candidate() -> None:
    text = "[[PAGE:1]]\n[[ORG:org_example]] provided support."
    repaired, result, redactions, metrics = validate_privacy(
        privacy_document(text), text, [], CandidateBackend(), 256, 1, None,
        PatternNER({"ORG": "organization"}), 40)
    assert result.release_status == "validated_candidate"
    assert repaired == text
    assert redactions == []
    assert metrics["calls"] == 0


def test_missing_ner_backend_prevents_automatic_validation() -> None:
    text = "[[PAGE:1]]\nGeneric evaluation prose."
    repaired, result, _, _ = validate_privacy(
        privacy_document(text), text, [], CandidateBackend(), 256, 1)
    assert result.release_status == "review_required"
    assert result.reason_codes == ["ner_validation_unavailable"]
    assert repaired == text


def test_structured_field_recovers_duplicate_value_inside_parent_bounds() -> None:
    text = "[[PAGE:1]]\nExample University first.\nExample University second."
    document = privacy_document(text)
    second = text.rfind("Example University")
    field = _field(document, RawField(
        value="Example University", status="found", model_confidence=0.9,
        start=None, end=None), bounds=(second, len(text)))
    assert field.status == ValueStatus.FOUND
    assert field.evidence[0].start == second
    assert "offset_recovered_by_bounded_exact_match" in field.validation_flags


def test_model_mention_with_bad_offsets_expands_to_exact_occurrences() -> None:
    text = "[[PAGE:1]]\nNamed Person appeared. Named Person returned."
    mentions = _mentions(privacy_document(text), [RawMention(
        identifier_type="person_name", person_role="third_party", text="Named Person",
        start=999, end=1011, model_confidence=0.9)])
    model_mentions = [row for row in mentions if row.detector == "model"]
    assert len(model_mentions) == 2
    assert all(row.grounded for row in model_mentions)


def test_real_ner_requires_explicit_local_model_path() -> None:
    with pytest.raises(ValidationError):
        NERConfig(enabled=True, backend="spacy")
    value = NERConfig(enabled=True, backend="mock")
    assert value.model_path is None


def test_privacy_audit_is_protected_and_stratified(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    inventory_stage(config)
    process_stage(config)
    result = build_privacy_audit(config, per_type=2)
    path = Path(result["audit_path"])
    assert result["classification"] == "P4-sensitive-protected-only"
    assert path.stat().st_mode & 0o777 == 0o600
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["document_type"] in {"cv", "recommendation_letter"} for row in rows)
