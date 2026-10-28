from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pymupdf
import pytest
import httpx
from pydantic import ValidationError

from academic_pipeline_v2.config import ModelConfig, PipelineConfig
from academic_pipeline_v2.exporter import validate_candidate_package
from academic_pipeline_v2.evaluation import evaluate_run
from academic_pipeline_v2.extraction import extract_canonical
from academic_pipeline_v2.inventory import build_inventory
from academic_pipeline_v2.models import DocumentRecord, DocumentType
from academic_pipeline_v2.models import CanonicalDocument, CanonicalPage, ModelResponse, ValueStatus
from academic_pipeline_v2.privacy import RawPrivacyReview, _scan, validate_privacy
from academic_pipeline_v2.pipeline import (
    export_candidate_stage, finalize_review_stage, inventory_stage, process_stage, release_stage,
)
from academic_pipeline_v2.registry import EntityRegistry
from academic_pipeline_v2.inference import VLLMBackend
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


def test_privacy_model_finding_is_automatically_repaired() -> None:
    class Backend:
        def complete(self, task, prompt, schema, max_tokens):
            text = prompt.rsplit("CANDIDATE_TEXT:\n", 1)[-1]
            if "Secret Alias" in text:
                start = text.index("Secret Alias")
                payload = {"safe": False, "model_confidence": 0.99,
                           "findings": [{"identifier_type": "person_name",
                                         "start": start, "end": start + 12}]}
            else:
                payload = {"safe": True, "model_confidence": 0.99, "findings": []}
            return ModelResponse(payload=payload, latency_seconds=0.01)

        def close(self):
            return None

    text = "[[PAGE:1]]\nSecret Alias wrote this letter."
    document = CanonicalDocument(
        application_id="application_x", document_id="document_x",
        document_type=DocumentType.LETTER, source_sha256="0" * 64,
        source_reference="document_x", text=text,
        pages=[CanonicalPage(page=1, text="Secret Alias wrote this letter.",
                             document_start=11, document_end=len(text),
                             extraction_method="native", extraction_status=ValueStatus.FOUND)],
        extraction_status=ValueStatus.FOUND,
    )
    repaired, result, redactions, metrics = validate_privacy(
        document, text, [], Backend(), 256, 2)
    assert result.release_status == "released"
    assert "Secret Alias" not in repaired
    assert "[[PRIVACY_REPAIR]]" in repaired
    assert metrics["repairs"] == 1
    assert redactions[0]["detector"] == "privacy_repair"


def test_privacy_review_schema_bounds_findings() -> None:
    schema = RawPrivacyReview.model_json_schema()
    assert schema["properties"]["findings"]["maxItems"] == 50
    finding_schema = schema["$defs"]["RawPrivacyFinding"]
    assert "ORG" not in finding_schema["properties"]["identifier_type"]["enum"]
    assert "organization" in finding_schema["properties"]["identifier_type"]["enum"]
    assert "model_confidence" in schema["required"]


def test_privacy_review_ignores_placeholder_spans() -> None:
    class Backend:
        def complete(self, task, prompt, schema, max_tokens):
            text = prompt.rsplit("CANDIDATE_TEXT:\n", 1)[-1]
            start = text.index("ORG")
            return ModelResponse(payload={
                "safe": False,
                "model_confidence": 0.99,
                "findings": [{"identifier_type": "organization",
                              "start": start, "end": start + 3}],
            }, latency_seconds=0.01)

        def close(self):
            return None

    text = "[[PAGE:1]]\n[[ORG:org_example]] provided support."
    document = CanonicalDocument(
        application_id="application_x", document_id="document_x",
        document_type=DocumentType.LETTER, source_sha256="0" * 64,
        source_reference="document_x", text=text,
        pages=[CanonicalPage(page=1, text="[[ORG:org_example]] provided support.",
                             document_start=11, document_end=len(text),
                             extraction_method="native", extraction_status=ValueStatus.FOUND)],
        extraction_status=ValueStatus.FOUND,
    )
    repaired, result, redactions, _ = validate_privacy(
        document, text, [], Backend(), 256, 2)
    assert result.release_status == "released"
    assert repaired == text
    assert redactions == []


def test_alias_scan_reports_every_repeated_occurrence() -> None:
    text = "Repeated Alias appears. Repeated Alias appears again. Repeated Alias."
    findings = [row for row in _scan(text, ["Repeated Alias"])
                if row.finding_type == "protected_alias"]
    assert len(findings) == 3
    assert [text[row.start:row.end] for row in findings] == ["Repeated Alias"] * 3


def test_privacy_repair_fragment_does_not_become_alias() -> None:
    class Backend:
        calls = 0

        def complete(self, task, prompt, schema, max_tokens):
            self.calls += 1
            text = prompt.rsplit("CANDIDATE_TEXT:\n", 1)[-1]
            if self.calls == 1:
                start = text.index("ordinary")
                payload = {"safe": False, "model_confidence": 0.99,
                           "findings": [{"identifier_type": "work_title",
                                         "start": start, "end": start + len("ordinary")}]}
            else:
                payload = {"safe": True, "model_confidence": 0.99, "findings": []}
            return ModelResponse(payload=payload, latency_seconds=0.01)

        def close(self):
            return None

    text = "[[PAGE:1]]\nordinary prose and another ordinary phrase."
    document = CanonicalDocument(
        application_id="application_x", document_id="document_x",
        document_type=DocumentType.LETTER, source_sha256="0" * 64,
        source_reference="document_x", text=text,
        pages=[CanonicalPage(page=1, text="ordinary prose and another ordinary phrase.",
                             document_start=11, document_end=len(text),
                             extraction_method="native", extraction_status=ValueStatus.FOUND)],
        extraction_status=ValueStatus.FOUND,
    )
    repaired, result, redactions, _ = validate_privacy(
        document, text, [], Backend(), 256, 2)
    assert result.release_status == "released"
    assert repaired.count("ordinary") == 1
    assert len(redactions) == 1
