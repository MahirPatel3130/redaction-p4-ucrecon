from __future__ import annotations

import json
import logging
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from . import PIPELINE_VERSION, PROMPT_VERSION, SCHEMA_VERSION
from .checkpoints import CheckpointStore, stage_fingerprint
from .config import PipelineConfig
from .errors import ConfigurationError, ContractError, PipelineError
from .exporter import build_candidate, release_candidate, select_review_sample, validate_review
from .exporter import validate_candidate_package
from .extraction import extract_canonical
from .inference import create_backend
from .inventory import build_inventory
from .models import (
    ApplicationRecord, CVExtraction, CanonicalDocument, DocumentRecord, DocumentType,
    LetterExtraction, PrivacyResult,
)
from .privacy import apply_replacements, validate_privacy
from .registry import EntityRegistry
from .semantic import extract_cv, extract_letter
from .util import (
    atomic_json, atomic_write, ensure_private_tree, fingerprint, read_json, sha256_file,
    write_csv, write_jsonl,
)


LOG = logging.getLogger(__name__)


def _safe_failure_code(exc: Exception) -> str:
    """Return only controlled, non-sensitive failure codes."""
    if isinstance(exc, (InferenceError, ConfigurationError, ContractError)):
        value = str(exc)
        if value and value.replace("_", "").replace("-", "").isalnum():
            return value
    return type(exc).__name__


def prepare(config: PipelineConfig) -> tuple[Path, Path]:
    protected = config.output_root / "protected"
    metrics = config.output_root / "metrics"
    for directory in (protected, protected / "checkpoints", protected / "raw_text",
                      protected / "extraction_results", protected / "source_evidence",
                      protected / "entity_mappings",
                      protected / "qa_reports", protected / "review",
                      protected / "model_responses", metrics,
                      config.state_root):
        directory.mkdir(parents=True, exist_ok=True)
    ensure_private_tree(protected)
    os.chmod(config.state_root, 0o700)
    run_identity = fingerprint({"namespace": config.namespace,
                                "input_roots": [str(path.resolve()) for path in config.input_roots]})
    state_path = protected / "run_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("run_identity") != run_identity:
            raise ConfigurationError("output_root_belongs_to_different_input_collection")
    else:
        atomic_json(state_path, {"schema_version": SCHEMA_VERSION,
                                 "pipeline_version": PIPELINE_VERSION,
                                 "run_identity": run_identity})
    return protected, metrics


def inventory_stage(config: PipelineConfig) -> tuple[list[ApplicationRecord], list[DocumentRecord]]:
    protected, _ = prepare(config)
    applications, documents = build_inventory(config)
    atomic_json(protected / "inventory.json", {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "applications": [row.model_dump(mode="json") for row in applications],
        "documents": [row.model_dump(mode="json") for row in documents],
    })
    LOG.info("Inventory completed applications=%d documents=%d flags=%d",
             len(applications), len(documents),
             sum(bool(row.inventory_flags) for row in applications))
    return applications, documents


def load_inventory(config: PipelineConfig) -> tuple[list[ApplicationRecord], list[DocumentRecord]]:
    path = config.output_root / "protected" / "inventory.json"
    if not path.exists():
        return inventory_stage(config)
    payload = read_json(path)
    return ([ApplicationRecord.model_validate(row) for row in payload.get("applications", [])],
            [DocumentRecord.model_validate(row) for row in payload.get("documents", [])])


def _extraction_fingerprint(config: PipelineConfig, record: DocumentRecord) -> str:
    return stage_fingerprint("canonical_text_v2", record.source_sha256,
                             config.ocr.model_dump(mode="json"))


def extract_stage(config: PipelineConfig, retry_failed: bool = False) -> list[CanonicalDocument]:
    protected, _ = prepare(config)
    _, documents = load_inventory(config)
    store = CheckpointStore(protected, config.cache_fingerprint())

    def one(record: DocumentRecord) -> CanonicalDocument | None:
        checkpoint = store.load_metadata(record)
        stage_fp = _extraction_fingerprint(config, record)
        cached = store.load_artifact(checkpoint, "canonical", CanonicalDocument, stage_fp)
        if cached is not None and (not retry_failed or cached.extraction_status.value != "model_failed"):
            return cached
        started = time.monotonic()
        try:
            result = extract_canonical(record, config.ocr)
            checkpoint.timings_seconds["canonical"] = time.monotonic() - started
            store.save_artifact(checkpoint, "canonical", result, stage_fp)
            atomic_json(protected / "raw_text" / f"{record.document_id}.json", {
                "document_id": record.document_id, "canonical_text": result.text,
                "pages": [page.model_dump(mode="json") for page in result.pages],
            })
            return result
        except Exception as exc:
            store.fail(checkpoint, "canonical", type(exc).__name__)
            LOG.error("Canonical extraction failed document_id=%s error=%s",
                      record.document_id, type(exc).__name__)
            return None

    results: list[CanonicalDocument] = []
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        futures = {executor.submit(one, record): record.document_id for record in documents}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    LOG.info("Canonical extraction completed successful=%d failed=%d",
             len(results), len(documents) - len(results))
    return sorted(results, key=lambda row: row.document_id)


def _semantic_fingerprint(config: PipelineConfig, canonical: CanonicalDocument) -> str:
    return stage_fingerprint("semantic_v2", PROMPT_VERSION, config.model.model,
                             config.model.backend, sha256_file(
                                 config.output_root / "protected" / "checkpoints" /
                                 canonical.document_id / "canonical.json"))


def _privacy_fingerprint(config: PipelineConfig, semantic_fp: str,
                         resolved: dict[str, Any]) -> str:
    policy = yaml.safe_load(config.privacy.policy_path.read_text(encoding="utf-8"))
    return stage_fingerprint("privacy_v2", semantic_fp, config.model.model,
                             fingerprint(policy), resolved.get("mention_map", {}))


def _load_canonicals(config: PipelineConfig, documents: list[DocumentRecord]) -> dict[str, CanonicalDocument]:
    store = CheckpointStore(config.output_root / "protected", config.cache_fingerprint())
    result = {}
    for record in documents:
        checkpoint = store.load_metadata(record)
        value = store.load_artifact(checkpoint, "canonical", CanonicalDocument,
                                    _extraction_fingerprint(config, record))
        if value is not None:
            result[record.document_id] = value
    return result


def process_stage(config: PipelineConfig, retry_failed: bool = False) -> dict[str, Any]:
    protected, metrics = prepare(config)
    applications, documents = load_inventory(config)
    extract_stage(config, retry_failed=retry_failed)
    canonicals = _load_canonicals(config, documents)
    store = CheckpointStore(protected, config.cache_fingerprint())
    backend = create_backend(config.model)
    semantic: dict[str, CVExtraction | LetterExtraction] = {}
    semantic_metrics: dict[str, dict[str, Any]] = {}

    def semantic_one(record: DocumentRecord) -> tuple[str, CVExtraction | LetterExtraction,
                                                       dict[str, Any]] | None:
        canonical = canonicals.get(record.document_id)
        if canonical is None or canonical.extraction_status.value == "model_failed":
            return None
        checkpoint = store.load_metadata(record)
        stage_fp = _semantic_fingerprint(config, canonical)
        model_type = CVExtraction if record.document_type == DocumentType.CV else LetterExtraction
        cached = store.load_artifact(checkpoint, "semantic", model_type, stage_fp)
        cached_failed = False
        if isinstance(cached, CVExtraction):
            cached_failed = any(section.status.value == "model_failed" for section in
                                (cached.education, cached.employment, cached.publications))
        elif isinstance(cached, LetterExtraction):
            cached_failed = cached.extraction_status.value == "model_failed"
        if cached is not None and (not retry_failed or not cached_failed):
            return record.document_id, cached, {
                **checkpoint.metrics.get("semantic", {}), "cached": True,
            }
        started = time.monotonic()
        try:
            if record.document_type == DocumentType.CV:
                result, model_metrics = extract_cv(canonical, backend,
                                                   config.model.cv_max_tokens,
                                                   config.privacy.maximum_repairs,
                                                   config.model.context_length)
            else:
                result, model_metrics = extract_letter(canonical, backend,
                                                       config.model.letter_max_tokens,
                                                       config.model.context_length)
            raw_responses = model_metrics.pop("_raw_responses", [])
            if config.model.store_raw_responses:
                atomic_json(protected / "model_responses" /
                            f"{record.document_id}.semantic.json",
                            {"document_id": record.document_id, "responses": raw_responses})
            checkpoint.timings_seconds["semantic"] = time.monotonic() - started
            checkpoint.metrics["semantic"] = model_metrics
            store.save_artifact(checkpoint, "semantic", result, stage_fp)
            atomic_json(protected / "extraction_results" / f"{record.document_id}.json", result)
            atomic_json(protected / "source_evidence" / f"{record.document_id}.json", {
                "document_id": record.document_id,
                "source_sha256": record.source_sha256,
                "mentions": [mention.model_dump(mode="json") for mention in result.mentions],
            })
            return record.document_id, result, model_metrics
        except Exception as exc:
            code = _safe_failure_code(exc)
            store.fail(checkpoint, "semantic", code)
            LOG.error("Semantic extraction failed document_id=%s error=%s code=%s",
                      record.document_id, type(exc).__name__, code)
            return None

    with ThreadPoolExecutor(max_workers=config.model.max_inflight) as executor:
        futures = [executor.submit(semantic_one, record) for record in documents]
        for future in as_completed(futures):
            value = future.result()
            if value:
                document_id, extraction, model_metrics = value
                semantic[document_id] = extraction
                semantic_metrics[document_id] = model_metrics

    registry = EntityRegistry(config.state_root / "registry.sqlite")
    processed: list[dict[str, Any]] = []
    redactions: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    try:
        resolved_by_document: dict[str, dict[str, Any]] = {}
        for record in documents:
            canonical = canonicals.get(record.document_id)
            extraction = semantic.get(record.document_id)
            if canonical is None or extraction is None:
                continue
            resolved = registry.resolve_document(canonical, extraction)
            resolved_by_document[record.document_id] = resolved
            atomic_json(protected / "entity_mappings" / f"{record.document_id}.json", resolved)

        def privacy_one(record: DocumentRecord) -> dict[str, Any] | None:
            canonical = canonicals.get(record.document_id)
            extraction = semantic.get(record.document_id)
            resolved = resolved_by_document.get(record.document_id)
            if canonical is None or extraction is None or resolved is None:
                return None
            checkpoint = store.load_metadata(record)
            semantic_fp = _semantic_fingerprint(config, canonical)
            stage_fp = _privacy_fingerprint(config, semantic_fp, resolved)
            cached = store.load_artifact(checkpoint, "privacy", None, stage_fp)
            cached_privacy = (PrivacyResult.model_validate(cached["privacy"])
                              if cached is not None else None)
            if cached is not None and (not retry_failed or cached_privacy.release_status == "released"):
                return {"document": record, "canonical": canonical, "extraction": extraction,
                        "resolved": resolved,
                        "privacy": cached_privacy,
                        "deidentified_text": cached.get("deidentified_text", ""),
                        "redactions": cached.get("redactions", []),
                        "processing_status": "completed", "extraction_status": "completed",
                        "privacy_metrics": {**checkpoint.metrics.get("privacy", {}),
                                            "cached": True}}
            started = time.monotonic()
            try:
                deidentified, document_redactions = apply_replacements(canonical, extraction.mentions)
                deidentified, privacy, document_redactions, privacy_metrics = validate_privacy(
                    canonical, deidentified, document_redactions, backend,
                    config.model.qa_max_tokens, config.privacy.maximum_repairs,
                )
                raw_responses = privacy_metrics.pop("_raw_responses", [])
                if config.model.store_raw_responses:
                    atomic_json(protected / "model_responses" /
                                f"{record.document_id}.privacy.json",
                                {"document_id": record.document_id, "responses": raw_responses})
                artifact = {"privacy": privacy.model_dump(mode="json"),
                            "deidentified_text": deidentified if privacy.release_status == "released" else "",
                            "redactions": document_redactions}
                checkpoint.timings_seconds["privacy"] = time.monotonic() - started
                checkpoint.metrics["privacy"] = privacy_metrics
                store.save_artifact(checkpoint, "privacy", artifact, stage_fp)
                atomic_json(protected / "qa_reports" / f"{record.document_id}.json",
                            privacy.model_dump(mode="json"))
                return {"document": record, "canonical": canonical, "extraction": extraction,
                        "resolved": resolved, "privacy": privacy,
                        "deidentified_text": artifact["deidentified_text"],
                        "redactions": document_redactions, "processing_status": "completed",
                        "extraction_status": "completed", "privacy_metrics": privacy_metrics}
            except Exception as exc:
                code = _safe_failure_code(exc)
                store.fail(checkpoint, "privacy", code)
                LOG.error("Privacy validation failed document_id=%s error=%s code=%s",
                          record.document_id, type(exc).__name__, code)
                return None

        with ThreadPoolExecutor(max_workers=config.model.max_inflight) as executor:
            futures = {executor.submit(privacy_one, record): record for record in documents}
            for future in as_completed(futures):
                row = future.result()
                if row:
                    processed.append(row)

        processed_ids = {row["document"].document_id for row in processed}
        for record in documents:
            if record.document_id in processed_ids:
                continue
            checkpoint = store.load_metadata(record)
            privacy = PrivacyResult(
                application_id=record.application_id, document_id=record.document_id,
                release_status="withheld", reason_codes=["processing_failed"],
            )
            processed.append({"document": record, "canonical": canonicals.get(record.document_id),
                              "extraction": semantic.get(record.document_id), "resolved": {},
                              "privacy": privacy, "deidentified_text": "", "redactions": [],
                              "processing_status": "failed", "extraction_status":
                              "failed" if record.document_id not in semantic else "completed",
                              "failure_code": checkpoint.failure_code})
        processed.sort(key=lambda row: row["document"].document_id)
        redactions = [redaction for row in processed for redaction in row.get("redactions", [])]
        write_jsonl(protected / "redactions.jsonl", redactions, 0o600)
        redaction_fields = sorted({key for row in redactions for key in row}) or ["redaction_id"]
        write_csv(protected / "redactions.csv", redaction_fields, redactions, 0o600)
        for row in processed:
            document_id = row["document"].document_id
            checkpoint = store.load_metadata(row["document"])
            semantic_row = semantic_metrics.get(document_id, {})
            privacy_row = row.get("privacy_metrics", {})
            timing_rows.append({
                "application_id": row["document"].application_id,
                "document_id": document_id,
                "document_type": row["document"].document_type.value,
                "page_count": len(row["canonical"].pages) if row.get("canonical") else 0,
                "ocr_page_count": sum(page.extraction_method == "ocr" for page in row["canonical"].pages)
                    if row.get("canonical") else 0,
                "canonical_seconds": checkpoint.timings_seconds.get("canonical", 0.0),
                "semantic_seconds": checkpoint.timings_seconds.get("semantic", 0.0),
                "privacy_seconds": checkpoint.timings_seconds.get("privacy", 0.0),
                "model_inference_seconds": (semantic_row.get("inference_seconds", 0.0)
                                            + privacy_row.get("inference_seconds", 0.0)),
                "input_tokens": (semantic_row.get("input_tokens", 0)
                                 + privacy_row.get("input_tokens", 0)),
                "output_tokens": (semantic_row.get("output_tokens", 0)
                                  + privacy_row.get("output_tokens", 0)),
                "model_calls": semantic_row.get("calls", 0) + privacy_row.get("calls", 0),
                "repair_count": semantic_row.get("repairs", 0) + privacy_row.get("repairs", 0),
                "status": row["processing_status"], "release_status": row["privacy"].release_status,
                "failure_code": row.get("failure_code"),
            })
        write_jsonl(metrics / "document_timings.jsonl", timing_rows, 0o600)
        fields = list(timing_rows[0]) if timing_rows else ["document_id"]
        write_csv(metrics / "document_timings.csv", fields, timing_rows, 0o600)
        summary = summarize(processed, timing_rows)
        atomic_json(metrics / "run_summary.json", summary)
        atomic_json(protected / "processed_index.json", {
            "schema_version": SCHEMA_VERSION,
            "document_ids": [row["document"].document_id for row in processed],
            "summary": summary,
        })
        LOG.info("Processing completed documents=%d released=%d withheld=%d",
                 len(processed), summary["released_document_count"],
                 summary["withheld_document_count"])
        return summary
    finally:
        registry.close()
        backend.close()
        ensure_private_tree(protected)


def summarize(processed: list[dict[str, Any]], timings: list[dict[str, Any]]) -> dict[str, Any]:
    released = sum(row["privacy"].release_status == "released" for row in processed)
    totals = [sum(float(row.get(key, 0.0)) for key in
                  ("canonical_seconds", "semantic_seconds", "privacy_seconds")) for row in timings]
    ordered = sorted(totals)
    def percentile(percent: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * percent + 0.999)))
        return ordered[index]
    total_seconds = sum(totals)
    pages = sum(int(row.get("page_count", 0)) for row in timings)
    return {
        "schema_version": SCHEMA_VERSION, "document_count": len(processed),
        "released_document_count": released,
        "withheld_document_count": len(processed) - released,
        "failed_document_count": sum(row["processing_status"] == "failed" for row in processed),
        "mean_document_seconds": statistics.fmean(totals) if totals else 0.0,
        "median_document_seconds": statistics.median(totals) if totals else 0.0,
        "p90_document_seconds": percentile(0.90),
        "p95_document_seconds": percentile(0.95),
        "pages_per_minute": pages / (total_seconds / 60) if total_seconds else 0.0,
        "documents_per_hour": len(timings) / (total_seconds / 3600) if total_seconds else 0.0,
        "model_call_count": sum(int(row.get("model_calls", 0)) for row in timings),
        "input_token_count": sum(int(row.get("input_tokens", 0)) for row in timings),
        "output_token_count": sum(int(row.get("output_tokens", 0)) for row in timings),
        "repair_count": sum(int(row.get("repair_count", 0)) for row in timings),
        "peak_gpu_vram_bytes": None,
        "peak_gpu_vram_note": "Collect from the colocated vLLM server metrics on the restricted host",
    }


def load_processed(config: PipelineConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    applications, documents = load_inventory(config)
    protected = config.output_root / "protected"
    store = CheckpointStore(protected, config.cache_fingerprint())
    registry = EntityRegistry(config.state_root / "registry.sqlite")
    rows: list[dict[str, Any]] = []
    try:
        for record in documents:
            checkpoint = store.load_metadata(record)
            canonical = store.load_artifact(checkpoint, "canonical", CanonicalDocument,
                                            _extraction_fingerprint(config, record))
            if canonical is None:
                privacy = PrivacyResult(application_id=record.application_id,
                                        document_id=record.document_id,
                                        release_status="withheld",
                                        reason_codes=["processing_failed"])
                rows.append({"document": record, "canonical": None, "extraction": None,
                             "resolved": {}, "privacy": privacy, "redactions": [],
                             "deidentified_text": "", "processing_status": "failed",
                             "extraction_status": "failed", "failure_code": checkpoint.failure_code})
                continue
            model_type = CVExtraction if record.document_type == DocumentType.CV else LetterExtraction
            semantic_fp = _semantic_fingerprint(config, canonical)
            extraction = store.load_artifact(checkpoint, "semantic", model_type, semantic_fp)
            if extraction is None:
                privacy = PrivacyResult(application_id=record.application_id,
                                        document_id=record.document_id,
                                        release_status="withheld",
                                        reason_codes=["processing_failed"])
                rows.append({"document": record, "canonical": canonical, "extraction": None,
                             "resolved": {}, "privacy": privacy, "redactions": [],
                             "deidentified_text": "", "processing_status": "failed",
                             "extraction_status": "failed", "failure_code": checkpoint.failure_code})
                continue
            resolved_path = protected / "entity_mappings" / f"{record.document_id}.json"
            resolved = read_json(resolved_path) if resolved_path.exists() else registry.resolve_document(
                canonical, extraction)
            privacy_fp = _privacy_fingerprint(config, semantic_fp, resolved)
            artifact = store.load_artifact(checkpoint, "privacy", None, privacy_fp)
            if artifact is None:
                privacy = PrivacyResult(application_id=record.application_id,
                                        document_id=record.document_id,
                                        release_status="withheld",
                                        reason_codes=["privacy_not_completed"])
                artifact = {"privacy": privacy.model_dump(mode="json"), "redactions": [],
                            "deidentified_text": ""}
            rows.append({"document": record, "canonical": canonical, "extraction": extraction,
                         "resolved": resolved, "privacy": PrivacyResult.model_validate(artifact["privacy"]),
                         "redactions": artifact.get("redactions", []),
                         "deidentified_text": artifact.get("deidentified_text", ""),
                         "processing_status": "completed", "extraction_status": "completed"})
        return [row.model_dump(mode="json") for row in applications], rows
    finally:
        registry.close()


def export_candidate_stage(config: PipelineConfig) -> Path:
    applications, processed = load_processed(config)
    registry = EntityRegistry(config.state_root / "registry.sqlite")
    try:
        package = build_candidate(config.output_root, applications, processed, registry)
        sample = select_review_sample(processed, config.privacy.review_fraction,
                                      config.privacy.review_minimum)
        review_path = config.output_root / "protected" / "review" / "sample.csv"
        write_csv(review_path,
                  ["document_id", "application_id", "document_type", "review_decision",
                   "reviewer_note"], sample, 0o600)
        atomic_json(config.output_root / "protected" / "review" / "sample_manifest.json", {
            "candidate_manifest_sha256": sha256_file(package / "manifest.json"),
            "document_ids": [row["document_id"] for row in sample],
        })
        return package
    finally:
        registry.close()


def finalize_review_stage(config: PipelineConfig, decisions: Path) -> dict[str, Any]:
    manifest_path = config.output_root / "protected" / "review" / "sample_manifest.json"
    if not manifest_path.exists():
        raise ContractError("review_sample_manifest_missing")
    manifest = read_json(manifest_path)
    candidate_manifest = config.output_root / "researcher_candidate" / "manifest.json"
    if not candidate_manifest.exists() or sha256_file(candidate_manifest) != manifest["candidate_manifest_sha256"]:
        raise ContractError("candidate_changed_after_review_sample")
    validate_candidate_package(config.output_root)
    approval = validate_review(decisions, set(manifest["document_ids"]))
    approval["candidate_manifest_sha256"] = manifest["candidate_manifest_sha256"]
    atomic_json(config.output_root / "protected" / "review" / "approval.json", approval)
    return approval


def release_stage(config: PipelineConfig) -> Path:
    approval_path = config.output_root / "protected" / "review" / "approval.json"
    if config.privacy.require_human_review and not approval_path.exists():
        raise ContractError("human_review_approval_missing")
    approval = read_json(approval_path) if approval_path.exists() else {"reviewed_document_count": 0}
    validate_candidate_package(config.output_root)
    return release_candidate(config.output_root, approval)


def status_stage(config: PipelineConfig) -> dict[str, Any]:
    summary_path = config.output_root / "metrics" / "run_summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    inventory_path = config.output_root / "protected" / "inventory.json"
    if inventory_path.exists():
        inventory = read_json(inventory_path)
        return {"status": "inventoried", "application_count": len(inventory.get("applications", [])),
                "document_count": len(inventory.get("documents", []))}
    return {"status": "not_started"}
