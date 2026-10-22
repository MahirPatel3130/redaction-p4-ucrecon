from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import PROMPT_VERSION, SCHEMA_VERSION
from .common import (
    OpaqueRegistry, atomic_json, deterministic_identifiers,
    five_year_bucket, leakage_findings, normalize, replace_aliases, sha256_file, stable_id,
    timing_statistics, valid_confidence, write_csv, write_json, write_jsonl,
)
from .model import create_model


CV_NAME = re.compile(r"(?:cv|curriculum[_ -]?vitae)", re.IGNORECASE)
SECTION_NAMES = ("education", "employment", "publications")
FIELD_STATUSES = {"found", "not_found", "uncertain", "not_applicable", "model_failed"}
SECTION_STATUSES = {"found", "not_found", "uncertain", "model_failed"}


def application_id(relative: Path) -> str:
    return stable_id("app", relative.as_posix())


def document_id(relative: Path) -> str:
    return stable_id("doc", relative.as_posix())


def discover(input_roots: list[Path], limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    applications: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    seen_apps: set[str] = set()
    seen_docs: set[str] = set()
    for root in input_roots:
        if not root.is_dir():
            raise ValueError(f"Input root does not exist: {root}")
        directories = sorted((path for path in root.iterdir()
                              if path.is_dir() and not path.name.startswith(".")),
                             key=lambda path: path.name.casefold())
        for directory in directories:
            relative_app = directory.relative_to(root)
            app_id = application_id(relative_app)
            if app_id in seen_apps:
                raise ValueError("Application ID collision across input roots")
            seen_apps.add(app_id)
            pdfs = sorted((path for path in directory.iterdir()
                           if path.is_file() and path.suffix.casefold() == ".pdf"),
                          key=lambda path: path.name.casefold())
            cvs = [path for path in pdfs if CV_NAME.search(path.stem)]
            letters = [path for path in pdfs if path not in cvs]
            flags = []
            if len(cvs) != 1:
                flags.append("missing_cv" if not cvs else "multiple_cv_candidates")
            if len(letters) != 3:
                flags.append("unexpected_recommendation_letter_count")
            record = {"application_id": app_id, "status": "inventory_flagged" if flags else "inventoried",
                      "has_cv": len(cvs) == 1, "recommendation_letter_count": len(letters),
                      "inventory_flags": flags}
            applications.append(record)
            selected = [(cvs[0], "cv", None)] if len(cvs) == 1 else []
            selected.extend((path, "recommendation_letter", index)
                            for index, path in enumerate(letters, 1))
            for source, doc_type, letter_index in selected:
                relative = relative_app / source.name
                doc_id = document_id(relative)
                if doc_id in seen_docs:
                    raise ValueError("Document ID collision across input roots")
                seen_docs.add(doc_id)
                tasks.append({"application_id": app_id, "document_id": doc_id,
                              "document_type": doc_type, "letter_index": letter_index,
                              "source_path": str(source.resolve()),
                              "source_relative_path": relative.as_posix(),
                              "source_sha256": sha256_file(source)})
    applications.sort(key=lambda row: row["application_id"])
    tasks.sort(key=lambda row: (row["application_id"], row["document_type"],
                                row.get("letter_index") or 0, row["document_id"]))
    if limit is not None:
        chosen = {row["application_id"] for row in applications[:limit]}
        applications = [row for row in applications if row["application_id"] in chosen]
        tasks = [row for row in tasks if row["application_id"] in chosen]
    return applications, tasks


def render_page(page: Any, dpi: int) -> Any:
    import pymupdf
    from PIL import Image

    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def is_native_text_reliable(text: str) -> bool:
    return sum(character.isalnum() for character in text) >= 40


def temporary_placeholder(entity: dict[str, Any], index: int) -> str:
    kind = str(entity.get("identifier_type", "other"))
    label = {
        "person_name": "PERSON", "organization": "ORG", "work_title": "WORK",
        "email": "EMAIL", "phone_number": "PHONE", "postal_address": "ADDRESS",
        "website": "URL", "signature": "SIGNATURE", "date": "DATE",
        "unique_identifier": "IDENTIFIER",
    }.get(kind, "REDACTED")
    return f"[[{label}:TEMP_{index}]]"


def provisional_deidentify(text: str, entities: list[dict[str, Any]]) -> str:
    replacements = []
    for index, entity in enumerate(entities):
        raw = str(entity.get("text", "")).strip()
        if raw:
            replacements.append((raw, temporary_placeholder(entity, index)))
    for raw, placeholder in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        text = re.sub(re.escape(raw), lambda _: placeholder, text, flags=re.IGNORECASE)
    return text


def safe_confidence(value: Any) -> float | None:
    try:
        return valid_confidence(value, required=False)
    except ValueError:
        return None


def validate_entity(entity: dict[str, Any], page: int | None,
                    page_text: str = "") -> dict[str, Any] | None:
    if not isinstance(entity, dict):
        return None
    kind = str(entity.get("identifier_type", ""))
    raw = str(entity.get("text", "")).strip()
    if not kind or not raw:
        return None
    confidence = safe_confidence(entity.get("model_confidence"))
    box = entity.get("bbox_normalized")
    if box is not None:
        if not isinstance(box, list) or len(box) != 4:
            box = None
        else:
            try:
                box = [max(0.0, min(1.0, float(value))) for value in box]
            except (TypeError, ValueError):
                box = None
    grounded = bool(raw.casefold() in page_text.casefold()) if page_text else kind == "signature"
    return {"identifier_type": kind,
            "person_role": str(entity.get("person_role", "unknown")),
            "text": raw, "model_confidence": confidence,
            "bbox_normalized": box, "page": page,
            "detection_source": entity.get("detection_source", "gemma"),
            "source_grounded": grounded}


def deduplicate_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entity in entities:
        key = (entity.get("page"), entity.get("identifier_type"),
               normalize(str(entity.get("text", ""))))
        current = result.get(key)
        current_score = (0 if current is None else
                         int(current.get("detection_source") != "deterministic_regex")
                         + int(current.get("person_role") not in {None, "unknown"}))
        entity_score = (int(entity.get("detection_source") != "deterministic_regex")
                        + int(entity.get("person_role") not in {None, "unknown"}))
        if current is None or entity_score > current_score:
            result[key] = entity
    return list(result.values())


def validate_field(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"value": None, "status": "model_failed", "model_confidence": None}
    status = str(value.get("status", "model_failed"))
    if status not in FIELD_STATUSES:
        status = "model_failed"
    confidence = safe_confidence(value.get("model_confidence"))
    if status == "model_failed":
        confidence = None
    return {"value": value.get("value"), "status": status, "model_confidence": confidence}


def validate_section(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"status": "model_failed", "model_confidence": None,
                "failure_code": "missing_section_object", "items": []}
    status = str(value.get("status", "model_failed"))
    if status not in SECTION_STATUSES:
        status = "model_failed"
    confidence = safe_confidence(value.get("model_confidence"))
    if status == "model_failed":
        confidence = None
    items = value.get("items", [])
    if not isinstance(items, list):
        return {"status": "model_failed", "model_confidence": confidence,
                "failure_code": "invalid_section_items", "items": []}
    expected = {
        "education": ("institution", "degree_type", "field", "start_year", "end_year"),
        "employment": ("organization", "job_title", "job_category", "start_year", "end_year"),
        "publications": ("title", "doi", "publication_type", "year", "venue"),
    }[name]
    validated = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = {key: validate_field(item.get(key)) for key in expected}
        row["item_model_confidence"] = safe_confidence(item.get("item_model_confidence"))
        row["source_page"] = item.get("source_page")
        if name == "publications":
            authors = []
            for author in item.get("authors", []) if isinstance(item.get("authors", []), list) else []:
                if isinstance(author, dict):
                    authors.append({"name": validate_field(author.get("name")),
                                    "author_position": author.get("author_position"),
                                    "is_applicant": bool(author.get("is_applicant", False)),
                                    "model_confidence": safe_confidence(
                                        author.get("model_confidence"))})
            row["authors"] = authors
        validated.append(row)
    result = {"status": status, "model_confidence": confidence, "items": validated}
    if value.get("not_found_reason"):
        result["not_found_reason"] = str(value["not_found_reason"])
    if status == "found" and not validated:
        result.update({"status": "model_failed", "model_confidence": None,
                       "failure_code": "found_without_items"})
    return result


def fallback_consolidate(page_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Lossless page-level fallback; it never invents semantic confidence."""
    result: dict[str, Any] = {"sections": {}, "research_fields": [], "methods": []}
    for name in SECTION_NAMES:
        items = [item for page in page_results
                 for item in page.get("sections", {}).get(name, {}).get("items", [])]
        if items:
            result["sections"][name] = {
                "status": "found", "model_confidence": None, "items": items,
                "validation_flags": ["deterministic_page_merge_fallback"],
            }
        else:
            statuses = [page.get("sections", {}).get(name, {}).get("status")
                        for page in page_results]
            status = "uncertain" if "uncertain" in statuses else "not_found"
            result["sections"][name] = {
                "status": status, "model_confidence": None, "items": [],
                "not_found_reason": "No page-level items were recovered",
                "validation_flags": ["deterministic_page_merge_fallback"],
            }
    result["research_fields"] = [item for page in page_results
                                 for item in page.get("research_fields", [])]
    result["methods"] = [item for page in page_results for item in page.get("methods", [])]
    return result


def process_document(task: dict[str, Any], config: dict[str, Any], model: Any,
                     checkpoint_path: Path, debug_path: Path | None) -> dict[str, Any]:
    import pymupdf

    model.reset_metrics()
    total_started = time.monotonic()
    render_seconds = extraction_seconds = consolidation_seconds = qa_seconds = 0.0
    native_pages = scanned_pages = 0
    failure_stage = "pdf_open"
    raw: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "config_hash": config["config_hash"],
                           **task, "status": "failed", "pages": []}
    try:
        document = pymupdf.open(task["source_path"])
        try:
            raw["page_count"] = document.page_count
            page_results = []
            all_entities = []
            letter_parts = []
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                failure_stage = f"page_{page_number}_text_extraction"
                started = time.monotonic()
                native_text = page.get_text("text", sort=True)
                extraction_seconds += time.monotonic() - started
                reliable = is_native_text_reliable(native_text)
                native_pages += int(reliable)
                scanned_pages += int(not reliable)
                started = time.monotonic()
                failure_stage = f"page_{page_number}_rendering"
                image = render_page(page, config["dpi"])
                render_seconds += time.monotonic() - started
                if task["document_type"] == "cv":
                    failure_stage = f"page_{page_number}_cv_parsing"
                    try:
                        result = model.parse_cv_page(
                            image, native_text if reliable else "", page_number)
                    except ValueError:
                        result = {"sections": {
                            name: {"status": "model_failed", "model_confidence": None,
                                   "items": [], "failure_code": "malformed_page_json"}
                            for name in SECTION_NAMES},
                            "research_fields": [], "methods": [], "identifiers": []}
                        raw.setdefault("validation_flags", []).append(
                            f"page_{page_number}_model_json_failed")
                    sections = result.get("sections", {}) if isinstance(result, dict) else {}
                    validated_page = {"page": page_number, "sections": {
                        name: validate_section(sections.get(name), name) for name in SECTION_NAMES},
                        "research_fields": result.get("research_fields", []),
                        "methods": result.get("methods", []), "identifiers": []}
                    model_entities = result.get("identifiers", [])
                    for entity in model_entities if isinstance(model_entities, list) else []:
                        validated = validate_entity(entity, page_number, native_text)
                        if validated:
                            validated_page["identifiers"].append(validated)
                    page_results.append(validated_page)
                    all_entities.extend(validated_page["identifiers"])
                else:
                    failure_stage = f"page_{page_number}_letter_analysis"
                    if not reliable:
                        try:
                            transcript = model.transcribe(image)
                        except ValueError:
                            transcript = {"text": "", "status": "model_failed",
                                          "model_confidence": None}
                            raw.setdefault("validation_flags", []).append(
                                f"page_{page_number}_transcription_json_failed")
                        transcript_confidence = safe_confidence(
                            transcript.get("model_confidence"))
                        native_text = str(transcript.get("text", ""))
                    else:
                        transcript_confidence = None
                    try:
                        result = model.analyze_letter_page(image, native_text, page_number)
                    except ValueError:
                        result = {"transcript": {"text": native_text,
                                                  "status": "uncertain",
                                                  "model_confidence": None},
                                  "entities": []}
                        raw.setdefault("validation_flags", []).append(
                            f"page_{page_number}_model_json_failed")
                    transcript = result.get("transcript", {}) if isinstance(result, dict) else {}
                    page_text = str(transcript.get("text", native_text))
                    confidence = safe_confidence(transcript.get("model_confidence"))
                    model_entities = result.get("entities", [])
                    entities = [validated for entity in (
                                    model_entities if isinstance(model_entities, list) else [])
                                if (validated := validate_entity(entity, page_number, page_text))]
                    entities.extend(validated for entity in deterministic_identifiers(page_text)
                                    if (validated := validate_entity(entity, page_number, page_text)))
                    entities = deduplicate_entities(entities)
                    page_results.append({"page": page_number, "text": page_text,
                                         "transcription_source": "pdf_text" if reliable else "gemma_vision",
                                         "transcription_model_confidence": (
                                             confidence if reliable else transcript_confidence),
                                         "entities": entities})
                    all_entities.extend(entities)
                    letter_parts.append(f"[[PAGE:{page_number}]]\n\n{page_text.strip()}")
            if task["document_type"] == "cv":
                failure_stage = "cv_consolidation"
                started = time.monotonic()
                try:
                    consolidated = model.consolidate_cv(page_results)
                except ValueError:
                    consolidated = fallback_consolidate(page_results)
                    raw.setdefault("validation_flags", []).append(
                        "cv_consolidation_model_json_failed")
                consolidation_seconds += time.monotonic() - started
                sections = consolidated.get("sections", {}) if isinstance(consolidated, dict) else {}
                raw["cv_parse"] = {"sections": {
                    name: validate_section(sections.get(name), name) for name in SECTION_NAMES},
                    "research_fields": consolidated.get("research_fields", []),
                    "methods": consolidated.get("methods", [])}
                raw["identifiers"] = deduplicate_entities(all_entities)
                provisional = provisional_deidentify(
                    json.dumps(raw["cv_parse"], ensure_ascii=False), raw["identifiers"])
            else:
                raw["letter_text"] = "\n\n".join(letter_parts)
                raw["identifiers"] = deduplicate_entities(all_entities)
                raw["page_transcription"] = page_results
                provisional = provisional_deidentify(raw["letter_text"], raw["identifiers"])
            started = time.monotonic()
            failure_stage = "privacy_qa"
            try:
                qa = model.qa(provisional)
            except ValueError:
                qa = {"qa_passed": False, "model_confidence": None,
                      "flagged_spans": [], "failure_code": "malformed_qa_json"}
            qa_seconds += time.monotonic() - started
            raw["privacy_qa"] = {"qa_passed": bool(qa.get("qa_passed", False)),
                                 "model_confidence": safe_confidence(qa.get("model_confidence")),
                                 "flagged_count": len(qa.get("flagged_spans", []))
                                 if isinstance(qa.get("flagged_spans", []), list) else 0}
            raw["status"] = "extracted"
        finally:
            document.close()
    except Exception as exc:
        raw["status"] = "failed"
        raw["failure_code"] = type(exc).__name__
        raw["failure_stage"] = failure_stage
    raw["timing"] = {
        "application_id": task["application_id"], "document_id": task["document_id"],
        "document_type": task["document_type"], "page_count": raw.get("page_count", 0),
        "native_page_count": native_pages, "scanned_page_count": scanned_pages,
        "render_seconds": round(render_seconds, 3),
        "text_extraction_seconds": round(extraction_seconds, 3),
        "consolidation_seconds": round(consolidation_seconds, 3),
        "resolution_seconds": 0.0,
        "deidentification_seconds": 0.0,
        "qa_seconds": round(qa_seconds, 3),
        "model_inference_seconds": round(model.inference_seconds, 3),
        "model_calls": model.calls, "retry_count": model.retries,
        "total_seconds": round(time.monotonic() - total_started, 3),
        "worker_id": config.get("worker_id", 0), "gpu_id": config.get("gpu_id", "0"),
        "status": raw["status"], "failure_code": raw.get("failure_code"),
    }
    atomic_json(checkpoint_path, raw)
    if debug_path is not None:
        write_json(debug_path, {"responses": model.raw_responses}, mode=0o600)
    return raw


def _worker_loop(worker_id: int, device: str, config: dict[str, Any], task_queue: Any,
                 result_queue: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    local_config = {**config, "worker_id": worker_id, "gpu_id": device}
    try:
        model = create_model(config["model"])
    except Exception as exc:
        result_queue.put({"worker_error": type(exc).__name__, "worker_id": worker_id,
                          "gpu_id": device, "failure_stage": "model_initialization",
                          "detail": str(exc)[:1000]})
        return
    while True:
        task = task_queue.get()
        if task is None:
            break
        checkpoint = Path(config["checkpoint_root"]) / f"{task['document_id']}.json"
        debug = (Path(config["debug_root"]) / f"{task['document_id']}.model.json"
                 if config["debug_artifacts"] else None)
        result = process_document(task, local_config, model, checkpoint, debug)
        result_queue.put({"document_id": task["document_id"], "status": result["status"]})


def extract_documents(tasks: list[dict[str, Any]], config: dict[str, Any],
                      resume: bool, retry_failed: bool) -> list[dict[str, Any]]:
    checkpoint_root = Path(config["checkpoint_root"])
    pending = []
    for task in tasks:
        checkpoint = checkpoint_root / f"{task['document_id']}.json"
        if checkpoint.exists() and resume:
            old = json.loads(checkpoint.read_text(encoding="utf-8"))
            reusable = (old.get("source_sha256") == task["source_sha256"]
                        and old.get("config_hash") == config["config_hash"])
            if reusable and (old.get("status") == "extracted" or not retry_failed):
                continue
        pending.append(task)
    if pending:
        context = mp.get_context("spawn")
        task_queue, result_queue = context.Queue(), context.Queue()
        workers = []
        devices = config["devices"]
        for worker_id in range(config["workers"]):
            process = context.Process(target=_worker_loop,
                                      args=(worker_id, devices[worker_id], config,
                                            task_queue, result_queue))
            process.start()
            workers.append(process)
        for task in pending:
            task_queue.put(task)
        for _ in workers:
            task_queue.put(None)
        completed = 0
        errors = []
        while completed < len(pending):
            message = result_queue.get()
            if "worker_error" in message:
                errors.append(message)
                logging.error(
                    "Model worker failed worker_id=%s gpu_id=%s stage=%s error=%s detail=%s",
                    message.get("worker_id"), message.get("gpu_id"),
                    message.get("failure_stage"), message.get("worker_error"),
                    message.get("detail", "unavailable"))
                break
            completed += 1
            logging.info("Document extraction completed document_id=%s status=%s",
                         message["document_id"], message["status"])
        for process in workers:
            process.join()
        bad_exits = [{"pid": process.pid, "exit_code": process.exitcode}
                     for process in workers if process.exitcode != 0]
        for bad_exit in bad_exits:
            logging.error("Model worker exited unexpectedly pid=%s exit_code=%s",
                          bad_exit["pid"], bad_exit["exit_code"])
        if errors or bad_exits:
            raise RuntimeError("One or more model workers failed")
    return [json.loads((checkpoint_root / f"{task['document_id']}.json").read_text(encoding="utf-8"))
            for task in tasks]


def _qa_identifier_type(value: str, raw: str) -> str:
    value = value.strip().casefold()
    if value == "person": return "person_name"
    if value == "organization": return "organization"
    if value == "work": return "work_title"
    if value == "address": return "postal_address"
    if value == "date": return "date"
    if value == "contact":
        detected = deterministic_identifiers(raw)
        return detected[0]["identifier_type"] if detected else "other_identifying_data"
    return "other_identifying_data"


def _page_for_span(extraction: dict[str, Any], span: str) -> int | None:
    for page in extraction.get("page_transcription", []):
        if span.casefold() in str(page.get("text", "")).casefold():
            return page.get("page")
    return None


def repair_privacy_qa(extractions: list[dict[str, Any]], config: dict[str, Any]) -> int:
    """Rerun only failed privacy QA and turn exact grounded findings into redactions."""
    candidates = [row for row in extractions if row.get("status") == "extracted"
                  and (not row.get("privacy_qa", {}).get("qa_passed")
                       or row.get("privacy_qa", {}).get("flagged_count", 0))]
    if not candidates:
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config["devices"][0]
    model = create_model(config["model"])
    repaired = 0
    for extraction in candidates:
        model.reset_metrics()
        if extraction["document_type"] == "cv":
            source = json.dumps(extraction.get("cv_parse", {}), ensure_ascii=False)
        else:
            source = extraction.get("letter_text", "")
        provisional = provisional_deidentify(source, extraction.get("identifiers", []))
        try:
            first = model.qa(provisional)
        except ValueError:
            first = {"qa_passed": False, "model_confidence": None,
                     "flagged_spans": [], "failure_code": "malformed_qa_json"}
        additions = []
        unresolved = 0
        for item in first.get("flagged_spans", []) if isinstance(first, dict) else []:
            if not isinstance(item, dict):
                unresolved += 1
                continue
            span = str(item.get("text", "")).strip()
            if (not span or span.casefold() not in provisional.casefold()
                    or span.casefold() not in source.casefold()):
                unresolved += 1
                continue
            kind = _qa_identifier_type(str(item.get("identifier_type", "other")), span)
            additions.append({"identifier_type": kind, "person_role": "unknown",
                              "text": span, "model_confidence": safe_confidence(
                                  first.get("model_confidence")),
                              "bbox_normalized": None,
                              "page": _page_for_span(extraction, span),
                              "detection_source": "gemma_privacy_qa",
                              "source_grounded": True})
        additions = deduplicate_entities(additions)
        extraction["identifiers"] = deduplicate_entities(
            extraction.get("identifiers", []) + additions)
        remediated = provisional_deidentify(source, extraction["identifiers"])
        if additions:
            try:
                final = model.qa(remediated)
            except ValueError:
                final = {"qa_passed": False, "model_confidence": None,
                         "flagged_spans": [], "failure_code": "malformed_qa_json"}
        else:
            final = first
        flagged = final.get("flagged_spans", []) if isinstance(final, dict) else []
        extraction["privacy_qa"] = {
            "qa_passed": bool(final.get("qa_passed", False)) and unresolved == 0,
            "model_confidence": safe_confidence(final.get("model_confidence")),
            "flagged_count": len(flagged) if isinstance(flagged, list) else 0,
            "remediated_span_count": len(additions),
            "unresolved_span_count": unresolved,
        }
        extraction.setdefault("timing", {})["qa_repair_model_calls"] = model.calls
        extraction["timing"]["qa_repair_seconds"] = round(model.inference_seconds, 3)
        checkpoint = Path(config["checkpoint_root"]) / f"{extraction['document_id']}.json"
        atomic_json(checkpoint, extraction)
        if config.get("debug_artifacts"):
            write_json(Path(config["debug_root"]) /
                       f"{extraction['document_id']}.qa-repair.model.json",
                       {"responses": model.raw_responses}, mode=0o600)
        repaired += 1
        logging.info("Privacy QA repaired document_id=%s additions=%d final_pass=%s",
                     extraction["document_id"], len(additions),
                     extraction["privacy_qa"]["qa_passed"])
    return repaired


def _raw_field(field: Any) -> Any:
    return field.get("value") if isinstance(field, dict) else None


def resolve_entities(extractions: list[dict[str, Any]], identity_registry_path: Path,
                     state_root: Path) -> dict[str, Any]:
    """Resolve direct people plus exact organizations and works in protected registries."""
    import resolve_identities_and_relationships as identity

    parties: dict[str, dict[str, Any]] = {}

    def party(local_id: str, role: str, extraction: dict[str, Any]) -> dict[str, Any]:
        row = parties.setdefault(local_id, {
            "local_person_id": local_id, "roles": set(), "application_ids": set(),
            "document_ids": set(), "identifiers": [], "entity_ids": set(),
            "source": "unified_pipeline",
        })
        row["roles"].add(role)
        row["application_ids"].add(extraction["application_id"])
        row["document_ids"].add(extraction["document_id"])
        return row

    entity_local: dict[tuple[str, int], str] = {}
    resolution_by_doc: dict[str, float] = defaultdict(float)
    for extraction in extractions:
        started = time.monotonic()
        if extraction.get("status") != "extracted":
            continue
        app_id, doc_id = extraction["application_id"], extraction["document_id"]
        applicant_id = f"{app_id}:applicant"
        party(applicant_id, "applicant", extraction)
        if extraction["document_type"] == "recommendation_letter":
            party(f"{doc_id}:recommender", "recommender", extraction)
        for index, entity in enumerate(extraction.get("identifiers", [])):
            kind, role = entity.get("identifier_type"), entity.get("person_role", "unknown")
            if role == "applicant":
                local_id = applicant_id
            elif role == "recommender":
                local_id = f"{doc_id}:recommender"
            elif kind == "person_name":
                local_id = f"{doc_id}:{role}:{stable_id('local', normalize(str(entity.get('text', ''))))}"
            elif role in {"applicant", "recommender"}:
                local_id = applicant_id if role == "applicant" else f"{doc_id}:recommender"
            else:
                continue
            entity_local[(doc_id, index)] = local_id
            target = party(local_id, role, extraction)
            if kind in identity.IDENTITY_TYPES:
                value = str(entity.get("text", "")).strip()
                normalized = identity.normalize(kind, value)
                if normalized:
                    target["identifiers"].append({
                        "identifier_type": kind, "value": value,
                        "normalized_value": normalized,
                        "verified": bool(entity.get("source_grounded", False)),
                        "document_id": doc_id,
                        "entity_id": stable_id("entity", f"{doc_id}|{kind}|{normalized}"),
                        "mention_ids": [],
                    })
        resolution_by_doc[doc_id] += time.monotonic() - started
    registry = identity.load_registry(identity_registry_path)
    previous = set(registry["persons"])
    identity.ensure_persons(registry, parties)
    conflicts = identity.auto_merge_strong(registry, parties, previous)
    advisories = identity.auto_merge_balanced(registry, parties)
    registry["updated_at"] = identity.now_utc()
    identity.atomic_write_registry(identity_registry_path, registry)

    local_to_person = {local_id: identity.person_for_local(registry, local_id)
                       for local_id in parties}
    org_registry = OpaqueRegistry(state_root / "organization_registry.json", "org")
    work_registry = OpaqueRegistry(state_root / "work_registry.json", "work")
    org_map: dict[tuple[str, str], str] = {}
    work_map: dict[tuple[str, str], str] = {}
    for extraction in extractions:
        started = time.monotonic()
        if extraction.get("status") != "extracted":
            continue
        doc_id = extraction["document_id"]
        for entity in extraction.get("identifiers", []):
            value = str(entity.get("text", "")).strip()
            kind = entity.get("identifier_type")
            evidence = {"document_id": doc_id, "page": entity.get("page")}
            if value and kind == "organization":
                org_map[(doc_id, normalize(value))] = org_registry.resolve(
                    value, "organization", evidence)[0]
            elif value and kind == "work_title":
                work_map[(doc_id, normalize(value))] = work_registry.resolve(
                    value, "publication_or_work", evidence)[0]
        sections = extraction.get("cv_parse", {}).get("sections", {})
        for section, key in (("education", "institution"), ("employment", "organization")):
            for item in sections.get(section, {}).get("items", []):
                value = _raw_field(item.get(key))
                if value:
                    org_map[(doc_id, normalize(str(value)))] = org_registry.resolve(
                        str(value), "education_institution" if section == "education" else "employer",
                        {"document_id": doc_id, "section": section})[0]
        for item in sections.get("publications", {}).get("items", []):
            title, doi = _raw_field(item.get("title")), _raw_field(item.get("doi"))
            year, venue = _raw_field(item.get("year")), _raw_field(item.get("venue"))
            if title:
                strong = str(doi) if doi else f"{normalize(str(title))}|{year or ''}"
                work_map[(doc_id, normalize(str(title)))] = work_registry.resolve(
                    str(title), "publication", {"document_id": doc_id}, strong_key=strong)[0]
            if venue:
                org_map[(doc_id, normalize(str(venue)))] = org_registry.resolve(
                    str(venue), "publication_venue", {"document_id": doc_id})[0]
        resolution_by_doc[doc_id] += time.monotonic() - started
    org_registry.save()
    work_registry.save()
    for extraction in extractions:
        extraction.setdefault("timing", {})["resolution_seconds"] = round(
            resolution_by_doc.get(extraction["document_id"], 0.0), 3)
    return {"identity_registry": registry, "local_to_person": local_to_person,
            "entity_local": entity_local, "organization_registry": org_registry,
            "work_registry": work_registry, "org_map": org_map, "work_map": work_map,
            "advisory_count": len(advisories) + len(conflicts)}


def _placeholder(kind: str, role: str, person_id: str | None,
                 org_id: str | None, work_id: str | None, value: str) -> str:
    if kind == "person_name":
        label = {"applicant": "APPLICANT", "recommender": "RECOMMENDER"}.get(role, "PERSON")
        return f"[[{label}:{person_id or 'UNRESOLVED'}]]"
    if kind == "organization":
        return f"[[ORG:{org_id or 'UNRESOLVED'}]]"
    if kind == "work_title":
        return f"[[WORK:{work_id or 'UNRESOLVED'}]]"
    if kind == "email":
        return f"[[EMAIL:{person_id or 'UNRESOLVED'}]]"
    if kind == "phone_number":
        return f"[[PHONE:{person_id or 'UNRESOLVED'}]]"
    if kind == "postal_address":
        return "[[ADDRESS]]"
    if kind == "website" or kind == "unique_identifier":
        return "[[URL]]" if kind == "website" else "[[IDENTIFIER]]"
    if kind == "signature":
        return "[[SIGNATURE]]"
    if kind == "date":
        return f"[[DATE:{five_year_bucket(value) or 'WITHHELD'}]]"
    return "[[REDACTED]]"


def build_replacements(extraction: dict[str, Any], resolved: dict[str, Any]) -> tuple[
        list[tuple[str, str]], list[dict[str, Any]]]:
    replacements, redactions = [], []
    doc_id = extraction["document_id"]
    for index, entity in enumerate(extraction.get("identifiers", [])):
        raw = str(entity.get("text", "")).strip()
        if not raw:
            continue
        kind, role = entity.get("identifier_type", "other"), entity.get("person_role", "unknown")
        local_id = resolved["entity_local"].get((doc_id, index))
        person_id = resolved["local_to_person"].get(local_id)
        org_id = resolved["org_map"].get((doc_id, normalize(raw))) if kind == "organization" else None
        work_id = resolved["work_map"].get((doc_id, normalize(raw))) if kind == "work_title" else None
        placeholder = _placeholder(kind, role, person_id, org_id, work_id, raw)
        mention_id = stable_id("mention", f"{doc_id}|{entity.get('page')}|{kind}|{normalize(raw)}|{index}")
        entity_id = stable_id("entity", f"{doc_id}|{kind}|{normalize(raw)}")
        replacements.append((raw, placeholder))
        redactions.append({
            "application_id": extraction["application_id"], "document_id": doc_id,
            "redaction_id": mention_id, "mention_id": mention_id, "entity_id": entity_id,
            "person_id": person_id, "identifier_category": kind, "person_role": role,
            "text": raw, "placeholder": placeholder, "page": entity.get("page"),
            "bbox_normalized": entity.get("bbox_normalized"),
            "model_confidence": entity.get("model_confidence"),
            "detection_source": entity.get("detection_source", "gemma"),
            "geometry_source": "gemma" if entity.get("bbox_normalized") else None,
            "grounding_status": "grounded" if entity.get("source_grounded") else "unverified",
            "validation_status": "accepted", "redaction_reason": "direct_or_linkable_identifier",
        })
    return replacements, redactions


def _public_field(field: dict[str, Any], transform: Any = None) -> dict[str, Any]:
    value = field.get("value")
    if value is not None and transform:
        value = transform(value)
    return {"value": value, "status": field.get("status", "model_failed"),
            "model_confidence": field.get("model_confidence")}


def _section_shell(section: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"status": section.get("status", "model_failed"),
              "model_confidence": section.get("model_confidence"), "items": items}
    if section.get("not_found_reason"):
        result["not_found_reason"] = section["not_found_reason"]
    if section.get("failure_code"):
        result["failure_code"] = section["failure_code"]
    return result


def transform_cv(extraction: dict[str, Any], resolved: dict[str, Any]) -> tuple[
        dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
        list[dict[str, Any]], list[dict[str, Any]]]:
    doc_id, app_id = extraction["document_id"], extraction["application_id"]
    applicant = resolved["local_to_person"].get(f"{app_id}:applicant")
    sections = extraction.get("cv_parse", {}).get("sections", {})
    education, employment, publications, authors, relationships = [], [], [], [], []
    for entity in extraction.get("identifiers", []):
        if entity.get("identifier_type") == "person_name" and entity.get("person_role") == "reference":
            local = f"{doc_id}:reference:{stable_id('local', normalize(str(entity.get('text', ''))))}"
            reference = resolved["local_to_person"].get(local)
            if reference and applicant:
                relationships.append({
                    "relationship_id": stable_id("rel", f"listed_as_reference_by|{doc_id}|{reference}"),
                    "relationship_type": "listed_as_reference_by", "source_id": reference,
                    "target_id": applicant, "document_id": doc_id,
                    "model_confidence": entity.get("model_confidence")})
    for index, item in enumerate(sections.get("education", {}).get("items", []), 1):
        raw_org = _raw_field(item["institution"])
        org_id = resolved["org_map"].get((doc_id, normalize(str(raw_org)))) if raw_org else None
        row_id = stable_id("education", f"{doc_id}|{index}")
        row = {"education_id": row_id, "application_id": app_id, "document_id": doc_id,
               "person_id": applicant, "organization_id": org_id,
               "degree_type": _public_field(item["degree_type"]),
               "academic_field": _public_field(item["field"]),
               "start_period": _public_field(item["start_year"], five_year_bucket),
               "end_period": _public_field(item["end_year"], five_year_bucket),
               "item_model_confidence": item.get("item_model_confidence")}
        education.append(row)
        if applicant and org_id:
            relationships.append({"relationship_id": stable_id("rel", f"educated_at|{row_id}"),
                                  "relationship_type": "educated_at", "source_id": applicant,
                                  "target_id": org_id, "document_id": doc_id,
                                  "model_confidence": item.get("item_model_confidence")})
    for index, item in enumerate(sections.get("employment", {}).get("items", []), 1):
        raw_org = _raw_field(item["organization"])
        org_id = resolved["org_map"].get((doc_id, normalize(str(raw_org)))) if raw_org else None
        row_id = stable_id("employment", f"{doc_id}|{index}")
        row = {"employment_id": row_id, "application_id": app_id, "document_id": doc_id,
               "person_id": applicant, "organization_id": org_id,
               "job_category": _public_field(item["job_category"]),
               "start_period": _public_field(item["start_year"], five_year_bucket),
               "end_period": _public_field(item["end_year"], five_year_bucket),
               "item_model_confidence": item.get("item_model_confidence")}
        employment.append(row)
        if applicant and org_id:
            relationships.append({"relationship_id": stable_id("rel", f"worked_at|{row_id}"),
                                  "relationship_type": "worked_at", "source_id": applicant,
                                  "target_id": org_id, "document_id": doc_id,
                                  "model_confidence": item.get("item_model_confidence")})
    for index, item in enumerate(sections.get("publications", {}).get("items", []), 1):
        title, venue = _raw_field(item["title"]), _raw_field(item["venue"])
        work_id = resolved["work_map"].get((doc_id, normalize(str(title)))) if title else None
        org_id = resolved["org_map"].get((doc_id, normalize(str(venue)))) if venue else None
        row_id = stable_id("publication", f"{doc_id}|{index}")
        publications.append({"publication_id": row_id, "application_id": app_id,
                             "document_id": doc_id, "work_id": work_id,
                             "venue_organization_id": org_id,
                             "publication_type": _public_field(item["publication_type"]),
                             "publication_year": _public_field(item["year"]),
                             "item_model_confidence": item.get("item_model_confidence")})
        if work_id and org_id:
            relationships.append({"relationship_id": stable_id("rel", f"published_in|{row_id}"),
                                  "relationship_type": "published_in", "source_id": work_id,
                                  "target_id": org_id, "document_id": doc_id,
                                  "model_confidence": item.get("item_model_confidence")})
        for author in item.get("authors", []):
            name = _raw_field(author["name"])
            if author.get("is_applicant"):
                person_id = applicant
            else:
                local = f"{doc_id}:coauthor:{stable_id('local', normalize(str(name or 'unknown')))}"
                person_id = resolved["local_to_person"].get(local)
            author_row = {"publication_id": row_id, "work_id": work_id, "person_id": person_id,
                          "author_position": author.get("author_position"),
                          "is_applicant": author.get("is_applicant"),
                          "model_confidence": author.get("model_confidence")}
            authors.append(author_row)
            if person_id and work_id:
                relationships.append({"relationship_id": stable_id("rel", f"authored|{row_id}|{person_id}"),
                                      "relationship_type": "authored", "source_id": person_id,
                                      "target_id": work_id, "document_id": doc_id,
                                      "model_confidence": author.get("model_confidence")})
    profile = {"application_id": app_id, "document_id": doc_id, "person_id": applicant,
               "education": _section_shell(sections.get("education", {}), education),
               "employment": _section_shell(sections.get("employment", {}), employment),
               "publications": _section_shell(sections.get("publications", {}), publications),
               "research_fields": extraction.get("cv_parse", {}).get("research_fields", []),
               "methods": extraction.get("cv_parse", {}).get("methods", [])}
    return profile, education, employment, publications, authors, relationships


def package_dataset(applications: list[dict[str, Any]], extractions: list[dict[str, Any]],
                    resolved: dict[str, Any], output_root: Path, require_review: bool) -> dict[str, Any]:
    protected = output_root / "protected"
    candidate = output_root / "researcher_candidate" if require_review else output_root / "researcher"
    metrics = output_root / "metrics"
    for directory in (protected, candidate, metrics, protected / "replacement_maps",
                      protected / "raw_text", protected / "raw_structured_extractions",
                      protected / "qa_reports"):
        directory.mkdir(parents=True, exist_ok=True)
    redactions: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    education: list[dict[str, Any]] = []
    employment: list[dict[str, Any]] = []
    publications: list[dict[str, Any]] = []
    authors: list[dict[str, Any]] = []
    letters: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    replacements_by_document: dict[str, tuple[list[tuple[str, str]], list[dict[str, Any]]]] = {}
    for extraction in extractions:
        if extraction.get("status") == "extracted":
            replacements_by_document[extraction["document_id"]] = build_replacements(
                extraction, resolved)
    redactions = [row for _, rows in replacements_by_document.values() for row in rows]
    aliases_by_document: dict[str, list[str]] = defaultdict(list)
    aliases_by_application: dict[str, list[str]] = defaultdict(list)
    cross_document_types = {"person_name", "email", "phone_number", "website",
                            "postal_address", "unique_identifier"}
    for row in redactions:
        if row["identifier_category"] == "date":
            continue
        aliases_by_document[row["document_id"]].append(row["text"])
        if row["identifier_category"] in cross_document_types:
            aliases_by_application[row["application_id"]].append(row["text"])
    withhold_count = 0
    for extraction in extractions:
        deidentification_started = time.monotonic()
        doc_id = extraction["document_id"]
        if extraction.get("status") != "extracted":
            documents.append({"application_id": extraction["application_id"], "document_id": doc_id,
                              "document_type": extraction["document_type"],
                              "letter_index": extraction.get("letter_index"), "status": "model_failed",
                              "release_status": "withheld", "failure_code": extraction.get("failure_code")})
            withhold_count += 1
            continue
        replacements, document_redactions = replacements_by_document[doc_id]
        protected_aliases = (aliases_by_document[doc_id]
                             + aliases_by_application[extraction["application_id"]])
        write_json(protected / "replacement_maps" / f"{doc_id}.json",
                   {"document_id": doc_id, "replacements": document_redactions}, mode=0o600)
        if extraction["document_type"] == "recommendation_letter":
            write_json(protected / "raw_text" / f"{doc_id}.json",
                       {"document_id": doc_id, "text": extraction.get("letter_text", "")}, mode=0o600)
            deidentified, _ = replace_aliases(extraction.get("letter_text", ""), replacements)
            findings = leakage_findings(deidentified, protected_aliases)
            qa = extraction.get("privacy_qa", {})
            scanned_low = any(
                page.get("transcription_source") == "gemma_vision"
                and (page.get("transcription_model_confidence") is None
                     or page.get("transcription_model_confidence") < 0.85)
                for page in extraction.get("page_transcription", []))
            safe = (qa.get("qa_passed") and qa.get("model_confidence") is not None
                    and qa["model_confidence"] >= 0.90 and qa.get("flagged_count", 0) == 0
                    and not findings and not scanned_low)
            letter = {"application_id": extraction["application_id"], "document_id": doc_id,
                      "person_id": resolved["local_to_person"].get(f"{doc_id}:recommender"),
                      "applicant_person_id": resolved["local_to_person"].get(
                          f"{extraction['application_id']}:applicant"),
                      "letter_index": extraction.get("letter_index"),
                      "status": "released" if safe else "withheld",
                      "text": deidentified if safe else None,
                      "withhold_reason": None if safe else "privacy_qa_failed"}
            letters.append(letter)
            if letter["person_id"] and letter["applicant_person_id"]:
                relationships.append({
                    "relationship_id": stable_id("rel", f"wrote|{doc_id}"),
                    "relationship_type": "wrote_recommendation_for",
                    "source_id": letter["person_id"], "target_id": letter["applicant_person_id"],
                    "document_id": doc_id, "model_confidence": None})
            withhold_count += int(not safe)
            write_json(protected / "qa_reports" / f"{doc_id}.json",
                       {"model_qa": qa, "deterministic_flags": findings,
                        "scanned_transcription_below_threshold": scanned_low,
                        "released": safe}, mode=0o600)
        else:
            write_json(protected / "raw_structured_extractions" / f"{doc_id}.json",
                       extraction.get("cv_parse", {}), mode=0o600)
            profile, ed, emp, pubs, auth, rels = transform_cv(extraction, resolved)
            serialized = json.dumps(profile, ensure_ascii=False)
            findings = leakage_findings(serialized, protected_aliases)
            qa = extraction.get("privacy_qa", {})
            safe = (qa.get("qa_passed") and qa.get("model_confidence") is not None
                    and qa["model_confidence"] >= 0.90 and qa.get("flagged_count", 0) == 0
                    and not findings)
            if safe:
                profiles.append(profile); education.extend(ed); employment.extend(emp)
                publications.extend(pubs); authors.extend(auth); relationships.extend(rels)
            else:
                withhold_count += 1
            write_json(protected / "qa_reports" / f"{doc_id}.json",
                       {"model_qa": qa, "deterministic_flags": findings, "released": safe}, mode=0o600)
        documents.append({"application_id": extraction["application_id"], "document_id": doc_id,
                          "document_type": extraction["document_type"],
                          "letter_index": extraction.get("letter_index"), "status": "completed",
                          "release_status": "released" if safe else "withheld",
                          "page_count": extraction.get("page_count", 0), "failure_code": None})
        extraction.setdefault("timing", {})["deidentification_seconds"] = round(
            time.monotonic() - deidentification_started, 3)
    # Public registries contain no aliases.
    identity_registry = resolved["identity_registry"]
    people = []
    used_people = {row["person_id"] for row in profiles if row.get("person_id")}
    used_people |= {value for row in relationships for value in (row.get("source_id"), row.get("target_id"))
                    if isinstance(value, str) and value.startswith("person_")}
    for person_id, person in identity_registry["persons"].items():
        if person.get("status") == "active" and person_id in used_people:
            people.append({"person_id": person_id, "roles": person.get("roles", []),
                           "application_count": len(person.get("application_ids", [])),
                           "document_count": len(person.get("document_ids", [])),
                           "identity_linkage_confidence": person.get("identity_linkage_confidence", 1.0)})
    used_orgs = {row.get("organization_id") for row in education + employment}
    used_orgs |= {row.get("venue_organization_id") for row in publications}
    used_works = {row.get("work_id") for row in publications}
    organizations = resolved["organization_registry"].public_rows({x for x in used_orgs if x})
    works = resolved["work_registry"].public_rows({x for x in used_works if x})
    applications_public = [{"application_id": row["application_id"], "status": row["status"],
                            "has_cv": row["has_cv"],
                            "recommendation_letter_count": row["recommendation_letter_count"],
                            "inventory_flags": ";".join(row["inventory_flags"])} for row in applications]
    section_status = [{"application_id": row["application_id"], "document_id": row["document_id"],
                       "section": name, "status": row[name]["status"],
                       "model_confidence": row[name]["model_confidence"],
                       "item_count": len(row[name]["items"])}
                      for row in profiles for name in SECTION_NAMES]
    tables = {"applications": applications_public, "documents": documents, "people": people,
              "organizations": organizations, "works": works, "relationships": relationships}
    dataset = {"schema_version": SCHEMA_VERSION,
               "release_status": "pending_review" if require_review else "ready",
               "review_required": require_review, **tables,
               "counts": {key: len(value) for key, value in tables.items()}}
    write_jsonl(candidate / "cv_profiles.jsonl", profiles)
    write_jsonl(candidate / "recommendation_letters.jsonl", letters)
    write_csv(candidate / "cv_section_status.csv",
              ["application_id", "document_id", "section", "status", "model_confidence", "item_count"], section_status)
    csv_specs = {
        "education.csv": (education, ["education_id", "application_id", "document_id", "person_id",
                                      "organization_id", "degree_type", "academic_field",
                                      "start_period", "end_period", "item_model_confidence"]),
        "employment.csv": (employment, ["employment_id", "application_id", "document_id", "person_id",
                                         "organization_id", "job_category", "start_period", "end_period",
                                         "item_model_confidence"]),
        "publications.csv": (publications, ["publication_id", "application_id", "document_id", "work_id",
                                           "venue_organization_id", "publication_type", "publication_year",
                                           "item_model_confidence"]),
        "publication_authors.csv": (authors, ["publication_id", "work_id", "person_id",
                                               "author_position", "is_applicant", "model_confidence"]),
        "applications.csv": (applications_public, ["application_id", "status", "has_cv",
                                                     "recommendation_letter_count", "inventory_flags"]),
        "documents.csv": (documents, ["application_id", "document_id", "document_type", "letter_index",
                                        "status", "release_status", "page_count", "failure_code"]),
        "people.csv": (people, ["person_id", "roles", "application_count", "document_count",
                                  "identity_linkage_confidence"]),
        "organizations.csv": (organizations, ["org_id", "entity_type", "occurrence_count",
                                                "resolution_status"]),
        "works.csv": (works, ["work_id", "entity_type", "occurrence_count", "resolution_status"]),
        "relationships.csv": (relationships, ["relationship_id", "relationship_type", "source_id",
                                                "target_id", "document_id", "model_confidence"]),
    }
    for filename, (rows, default_columns) in csv_specs.items():
        columns = sorted({key for row in rows for key in row}) or default_columns
        flattened = [{key: json.dumps(value) if isinstance(value, (dict, list)) else value
                      for key, value in row.items()} for row in rows]
        write_csv(candidate / filename, columns, flattened)
    write_json(candidate / "dataset.json", dataset)
    dictionary = {"schema_version": SCHEMA_VERSION,
                  "notes": "All IDs are opaque. Field confidence values are model supplied.",
                  "tables": {**{name: sorted({key for row in rows for key in row}) or defaults
                                  for name, (rows, defaults) in csv_specs.items()},
                             "cv_profiles": sorted({key for row in profiles for key in row}),
                             "recommendation_letters": sorted({key for row in letters for key in row})}}
    write_json(candidate / "data_dictionary.json", dictionary)
    quality = {"document_count": len(documents), "released_document_count": len(documents) - withhold_count,
               "withheld_document_count": withhold_count,
               "identity_advisory_count": resolved["advisory_count"]}
    write_json(candidate / "quality_summary.json", quality)
    (candidate / "README.md").write_text("""# De-identified academic application dataset

Classification: controlled de-identified research data. This directory intentionally excludes raw
names, contact values, organization names, publication titles, source paths, filenames, document
geometry, model responses, and replacement maps. Stable opaque IDs remain linkable and must still
be handled under the approved research protocol.

`dataset.json` is the canonical index. `applications.csv` and `documents.csv` describe coverage and
withholding. `cv_profiles.jsonl` preserves section/field statuses and model-supplied confidence;
`cv_section_status.csv` is its compact quality view. `education.csv`, `employment.csv`,
`publications.csv`, and `publication_authors.csv` are normalized analysis tables.
`recommendation_letters.jsonl` contains paragraph-normalized, de-identified text with `[[PAGE:n]]`
markers, or a null text plus a withholding reason when privacy checks fail. `people.csv`,
`organizations.csv`, and `works.csv` contain only opaque entities. `relationships.csv` contains
`wrote_recommendation_for`, `listed_as_reference_by`, `educated_at`, `worked_at`, `authored`, and
`published_in` edges when supported by released records. Use `data_dictionary.json` for field
lists, `quality_summary.json` for coverage, and `manifest.json` for checksums.

All `model_confidence` values come directly from Gemma; null means no semantic model confidence was
available. The pipeline does not average or invent confidence. Education and employment years use
fixed five-year buckets, publication years remain exact, and job titles are replaced by controlled
job categories. An explicit `not_found`, `uncertain`, or `model_failed` status is analytically
different from a missing row.
""", encoding="utf-8")
    write_jsonl(protected / "redactions.jsonl", redactions, mode=0o600)
    redaction_columns = sorted({key for row in redactions for key in row}) or ["redaction_id"]
    write_csv(protected / "redactions.csv", redaction_columns,
              [{key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()} for row in redactions], mode=0o600,
              protect_formulas=True)
    for path in protected.rglob("*"):
        if path.is_dir(): os.chmod(path, 0o700)
        elif path.is_file(): os.chmod(path, 0o600)
    os.chmod(protected, 0o700)
    return {"candidate": candidate, "dataset": dataset, "quality": quality,
            "redaction_count": len(redactions)}


def write_metrics(extractions: list[dict[str, Any]], metrics_root: Path,
                  application_count: int, worker_count: int) -> dict[str, Any]:
    timings = [row["timing"] for row in extractions]
    write_jsonl(metrics_root / "document_timings.jsonl", timings)
    columns = sorted({key for row in timings for key in row}) or ["document_id"]
    write_csv(metrics_root / "document_timings.csv", columns, timings)
    successful = [row for row in timings if row["status"] == "extracted"]
    seconds = [float(row["total_seconds"]) for row in successful]
    pages = sum(int(row.get("page_count", 0)) for row in successful)
    total = sum(seconds)
    by_type = {kind: timing_statistics([float(row["total_seconds"]) for row in successful
                                        if row["document_type"] == kind])
               for kind in ("cv", "recommendation_letter")}
    observed_hours = total / 3600 / max(1, worker_count)
    per_application = observed_hours / max(1, application_count)
    projections = {str(apps): {str(workers): round(per_application * apps / workers, 2)
                               for workers in range(1, 5)}
                   for apps in sorted({100, 500, 1000, application_count})}
    summary = {"schema_version": SCHEMA_VERSION, "document_count": len(timings),
               "successful_document_count": len(successful),
               "failed_document_count": len(timings) - len(successful),
               "overall_seconds": timing_statistics(seconds), "by_document_type": by_type,
               "pages_per_minute": round(pages / (total / 60), 3) if total else 0.0,
               "documents_per_hour": round(len(successful) / (total / 3600), 3) if total else 0.0,
               "retry_rate": round(sum(row.get("retry_count", 0) for row in timings)
                                   / max(1, sum(row.get("model_calls", 0) for row in timings)), 4),
               "failure_rate": round((len(timings) - len(successful)) / max(1, len(timings)), 4),
               "projected_wall_hours_by_applications_and_gpu_workers": projections,
               "projection_note": "Linear estimates from observed document time; not benchmark guarantees."}
    write_json(metrics_root / "run_summary.json", summary)
    return summary


def write_manifest(package_root: Path, source_tasks: list[dict[str, Any]],
                   release_status: str) -> None:
    files = {path.relative_to(package_root).as_posix(): sha256_file(path)
             for path in package_root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    write_json(package_root / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "release_status": release_status,
        "classification": "de-identified-controlled-research-data",
        "files": files,
        "source_checksums": {task["document_id"]: task["source_sha256"] for task in source_tasks},
    })


def select_review_sample(extractions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in extractions if row.get("status") == "extracted"]
    count = min(len(eligible), max(20, int(len(eligible) * 0.10 + 0.999)))
    ordered = sorted(eligible, key=lambda row: (
        row["document_type"], -row.get("timing", {}).get("scanned_page_count", 0), row["document_id"]))
    if count >= len(ordered): chosen = ordered
    else:
        step = len(ordered) / count
        chosen = [ordered[min(int(index * step), len(ordered) - 1)] for index in range(count)]
    return [{"document_id": row["document_id"], "application_id": row["application_id"],
             "document_type": row["document_type"], "review_decision": "",
             "reviewer_note": ""} for row in chosen]


def finalize_review(output_root: Path, decisions_csv: Path) -> None:
    candidate, researcher = output_root / "researcher_candidate", output_root / "researcher"
    if not candidate.is_dir():
        raise ValueError("researcher_candidate is missing")
    with decisions_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("review_decision", "").strip().casefold() != "pass" for row in rows):
        raise ValueError("Every sampled document must have review_decision=pass")
    if researcher.exists():
        raise ValueError("researcher output already exists")
    shutil.copytree(candidate, researcher)
    dataset = json.loads((researcher / "dataset.json").read_text(encoding="utf-8"))
    dataset.update({"release_status": "ready", "review_required": False,
                    "reviewed_document_count": len(rows)})
    write_json(researcher / "dataset.json", dataset)
    old_manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    source_checksums = old_manifest.get("source_checksums", {})
    files = {path.relative_to(researcher).as_posix(): sha256_file(path)
             for path in researcher.rglob("*") if path.is_file() and path.name != "manifest.json"}
    write_json(researcher / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "release_status": "ready",
        "classification": "de-identified-controlled-research-data",
        "files": files, "source_checksums": source_checksums,
    })
