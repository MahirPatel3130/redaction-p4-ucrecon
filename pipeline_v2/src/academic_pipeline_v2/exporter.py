from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .detectors import deterministic_spans
from .errors import ReleaseError
from .models import CVExtraction, DocumentType
from .registry import EntityRegistry
from .util import (
    atomic_json, atomic_write, ensure_private_tree, normalized_key, sha256_file, stable_id,
    write_csv, write_jsonl,
)


def five_year_bucket(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).strip()
    if text.casefold() in {"present", "current", "ongoing"}:
        return "present"
    match = re.search(r"\b(?:19|20)\d{2}\b", text)
    if not match:
        return None
    year = int(match.group())
    start = year - year % 5
    return f"{start}-{start + 4}"


def public_field(field: Any, transform: Any = None) -> dict[str, Any]:
    value = field.value
    if transform and value is not None:
        value = transform(value)
    return {"value": value, "status": field.status.value,
            "model_confidence": field.model_confidence,
            "validation_flags": field.validation_flags}


def public_cv(extraction: CVExtraction, registry: EntityRegistry,
              applicant_id: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    people_by_name = {
        normalized_key(mention.text): mention.resolved_entity_id
        for mention in extraction.mentions
        if mention.identifier_type == "person_name" and mention.resolved_entity_id
    }
    education: list[dict[str, Any]] = []
    for item in extraction.education.items:
        org_id = registry.find_alias(str(item.institution.value), ("organization_name",)) \
            if item.institution.value else None
        education.append({
            "education_id": item.item_id, "application_id": extraction.application_id,
            "document_id": extraction.document_id, "person_id": applicant_id,
            "organization_id": org_id, "degree_type": public_field(item.degree_type),
            "academic_field": public_field(item.field),
            "start_period": public_field(item.start_year, five_year_bucket),
            "end_period": public_field(item.end_year, five_year_bucket),
            "graduation_period": public_field(item.graduation_year, five_year_bucket),
            "item_model_confidence": item.item_model_confidence,
        })
    employment: list[dict[str, Any]] = []
    for item in extraction.employment.items:
        org_id = registry.find_alias(str(item.organization.value), ("organization_name",)) \
            if item.organization.value else None
        employment.append({
            "employment_id": item.item_id, "application_id": extraction.application_id,
            "document_id": extraction.document_id, "person_id": applicant_id,
            "organization_id": org_id, "job_category": public_field(item.job_category),
            "position_type": public_field(item.position_type),
            "start_period": public_field(item.start_date, five_year_bucket),
            "end_period": public_field(item.end_date, five_year_bucket),
            "item_model_confidence": item.item_model_confidence,
        })
    publications: list[dict[str, Any]] = []
    authors: list[dict[str, Any]] = []
    for item in extraction.publications.items:
        strong = str(item.doi.value) if item.doi.value else (
            f"{item.title.value}|{item.year.value}" if item.title.value else item.item_id)
        work_id = registry.find_alias(strong, ("work_key",))
        venue_id = registry.find_alias(str(item.venue.value), ("venue_name", "organization_name")) \
            if item.venue.value else None
        publication_id = item.item_id
        publications.append({
            "publication_id": publication_id, "application_id": extraction.application_id,
            "document_id": extraction.document_id, "work_id": work_id,
            "venue_organization_id": venue_id,
            "publication_type": public_field(item.publication_type),
            "publication_year": public_field(item.year),
            "item_model_confidence": item.item_model_confidence,
        })
        for author in item.authors:
            person_id = applicant_id if author.is_applicant else (
                people_by_name.get(normalized_key(str(author.name.value)))
                if author.name.value else None)
            authors.append({
                "publication_id": publication_id, "work_id": work_id, "person_id": person_id,
                "author_position": author.author_position, "is_applicant": author.is_applicant,
                "model_confidence": author.name.model_confidence,
            })
    profile = {
        "schema_version": SCHEMA_VERSION, "application_id": extraction.application_id,
        "document_id": extraction.document_id, "person_id": applicant_id,
        "education": {"status": extraction.education.status.value,
                      "model_confidence": extraction.education.model_confidence,
                      "not_found_reason": extraction.education.not_found_reason,
                      "validation_flags": extraction.education.validation_flags,
                      "items": education},
        "employment": {"status": extraction.employment.status.value,
                       "model_confidence": extraction.employment.model_confidence,
                       "not_found_reason": extraction.employment.not_found_reason,
                       "validation_flags": extraction.employment.validation_flags,
                       "items": employment},
        "publications": {"status": extraction.publications.status.value,
                        "model_confidence": extraction.publications.model_confidence,
                        "not_found_reason": extraction.publications.not_found_reason,
                        "validation_flags": extraction.publications.validation_flags,
                        "items": publications},
        "research_fields": [public_field(row) for row in extraction.research_fields],
        "research_methods": [public_field(row) for row in extraction.research_methods],
        "extraction_flags": extraction.extraction_flags,
    }
    return profile, {"education": education, "employment": employment,
                     "publications": publications, "publication_authors": authors}


def _atomic_directory(target: Path, build: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        build(temporary)
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _package_leaks(root: Path, protected_aliases: list[str]) -> list[str]:
    findings: list[str] = []
    aliases = sorted({alias.casefold() for alias in protected_aliases
                      if len(alias.strip()) >= 3 and not alias.strip().isdigit()}, key=len, reverse=True)
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        masked = re.sub(r"\[\[[^\]]+\]\]", " ", text)
        masked = re.sub(
            r"\b(?:application|document|person|org|work|relationship|education|employment|"
            r"publication|mention|local|redaction)_[0-9a-f]+\b",
            " ", masked, flags=re.IGNORECASE,
        )
        if deterministic_spans(masked):
            findings.append("deterministic_identifier_in_package")
        folded = masked.casefold()
        if any(alias in folded for alias in aliases):
            findings.append("protected_alias_in_package")
    return sorted(set(findings))


def build_candidate(output_root: Path, applications: list[dict[str, Any]],
                    processed: list[dict[str, Any]], registry: EntityRegistry) -> Path:
    target = output_root / "researcher_candidate"
    released_ids = {row["document"].document_id for row in processed
                    if row["privacy"].release_status == "validated_candidate"}
    protected_aliases = [redaction["text"] for row in processed
                         for redaction in row.get("redactions", [])]

    def build(root: Path) -> None:
        documents: list[dict[str, Any]] = []
        quality: list[dict[str, Any]] = []
        profiles: list[dict[str, Any]] = []
        letters: list[dict[str, Any]] = []
        tables = {key: [] for key in ("education", "employment", "publications",
                                      "publication_authors")}
        for row in processed:
            document = row["document"]
            privacy = row["privacy"]
            released = privacy.release_status == "validated_candidate"
            documents.append({
                "schema_version": SCHEMA_VERSION, "application_id": document.application_id,
                "document_id": document.document_id, "document_type": document.document_type.value,
                "letter_index": document.letter_index,
                "page_count": len(row["canonical"].pages) if row.get("canonical") else 0,
                "ocr_page_count": sum(page.extraction_method == "ocr"
                                      for page in row["canonical"].pages)
                if row.get("canonical") else 0,
                "processing_status": row.get("processing_status", "completed"),
                "release_status": privacy.release_status,
                "failure_code": row.get("failure_code"),
            })
            quality.append({
                "application_id": document.application_id, "document_id": document.document_id,
                "document_type": document.document_type.value,
                "processing_status": row.get("processing_status", "completed"),
                "extraction_status": row.get("extraction_status", "completed"),
                "privacy_status": privacy.release_status,
                "release_status": privacy.release_status,
                "repair_count": privacy.repair_attempts,
                "failure_code": row.get("failure_code"),
                "validation_flags": privacy.reason_codes,
            })
            if not released:
                continue
            if document.document_type == DocumentType.CV:
                profile, normalized = public_cv(row["extraction"], registry,
                                                row["resolved"]["applicant_id"])
                profiles.append(profile)
                for key in tables:
                    tables[key].extend(normalized[key])
            else:
                letters.append({
                    "schema_version": SCHEMA_VERSION, "application_id": document.application_id,
                    "document_id": document.document_id, "letter_index": document.letter_index,
                    "recommender_person_id": registry.local_entity(
                        f"{document.document_id}:recommender", "person", "recommender",
                        document.application_id, document.document_id),
                    "applicant_person_id": row["resolved"]["applicant_id"],
                    "status": "released", "text": row["deidentified_text"],
                    "extraction_model_confidence": row["extraction"].model_confidence,
                    "privacy_model_confidence": privacy.model_qa_confidence,
                })
        public_apps = [{"application_id": row["application_id"], "status": row["status"],
                        "inventory_flags": row["inventory_flags"],
                        "document_count": len(row["document_ids"])} for row in applications]
        write_csv(root / "applications.csv",
                  ["application_id", "status", "inventory_flags", "document_count"], public_apps)
        write_jsonl(root / "documents.jsonl", documents)
        write_jsonl(root / "cv_profiles.jsonl", profiles)
        write_jsonl(root / "recommendation_letters.jsonl", letters)
        specifications = {
            "education": ["education_id", "application_id", "document_id", "person_id",
                          "organization_id", "degree_type", "academic_field", "start_period",
                          "end_period", "graduation_period", "item_model_confidence"],
            "employment": ["employment_id", "application_id", "document_id", "person_id",
                           "organization_id", "job_category", "position_type", "start_period",
                           "end_period", "item_model_confidence"],
            "publications": ["publication_id", "application_id", "document_id", "work_id",
                             "venue_organization_id", "publication_type", "publication_year",
                             "item_model_confidence"],
            "publication_authors": ["publication_id", "work_id", "person_id", "author_position",
                                    "is_applicant", "model_confidence"],
        }
        for name, fields in specifications.items():
            write_csv(root / f"{name}.csv", fields, tables[name])
        entities = registry.public_entities()
        write_csv(root / "entities.csv",
                  ["entity_id", "entity_type", "document_count", "application_count"], entities)
        relationships = registry.public_relationships(released_ids)
        write_csv(root / "relationships.csv",
                  ["relationship_id", "relationship_type", "source_id", "target_id",
                   "application_id", "document_id", "confidence", "validation_status"], relationships)
        write_csv(root / "quality.csv",
                  ["application_id", "document_id", "document_type", "processing_status",
                   "extraction_status", "privacy_status", "release_status", "repair_count",
                   "failure_code", "validation_flags"], quality)
        dictionary = {
            "schema_version": SCHEMA_VERSION,
            "join_keys": ["application_id", "document_id", "entity_id", "work_id",
                          "organization_id", "person_id"],
            "canonical_files": ["documents.jsonl", "cv_profiles.jsonl",
                                "recommendation_letters.jsonl"],
            "notes": ["All IDs are opaque.", "Confidence values are model supplied.",
                      "Withheld documents contain no research content."],
        }
        atomic_json(root / "data_dictionary.json", dictionary, 0o644)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "document_count": len(documents),
            "released_document_count": len(released_ids),
            "review_required_document_count": sum(
                row["privacy"].release_status == "review_required" for row in processed),
            "withheld_document_count": sum(
                row["privacy"].release_status == "withheld" for row in processed),
        }
        atomic_json(root / "quality_summary.json", summary, 0o644)
        atomic_write(root / "README.md", RESEARCHER_README, 0o644)
        leaks = _package_leaks(root, protected_aliases)
        if leaks:
            raise ReleaseError("candidate_package_privacy_validation_failed")
        files = {path.relative_to(root).as_posix(): sha256_file(path)
                 for path in root.rglob("*") if path.is_file()}
        atomic_json(root / "manifest.json", {
            "schema_version": SCHEMA_VERSION, "release_status": "candidate",
            "classification": "controlled-deidentified-research-data", "files": files,
            "counts": summary,
        }, 0o644)

    _atomic_directory(target, build)
    return target


RESEARCHER_README = """# Academic application research dataset — Pipeline V2

This candidate package contains de-identified structured CV data and source-preserving
recommendation-letter text. Direct identifiers, organization names, work titles, source paths,
filenames, evidence geometry, and protected mappings are excluded. Stable opaque IDs are join
keys and remain controlled linkable data.

Start with `quality_summary.json`, `documents.jsonl`, and `data_dictionary.json`. Use
`cv_profiles.jsonl` for complete nested CV records, normalized education/employment/publication
CSV tables for analysis, and `recommendation_letters.jsonl` for de-identified narrative text.
`entities.csv` and `relationships.csv` support graph analysis. Failed or withheld documents appear
only in coverage and quality metadata.
"""


def select_review_sample(processed: list[dict[str, Any]], fraction: float,
                         minimum: int) -> list[dict[str, Any]]:
    eligible = [row for row in processed
                if row["privacy"].release_status == "validated_candidate"]
    count = min(len(eligible), max(minimum, int(len(eligible) * fraction + 0.999)))
    ranked = sorted(eligible, key=lambda row: (
        -int(any(page.extraction_method == "ocr" for page in row["canonical"].pages)),
        -row["privacy"].repair_attempts,
        -len(row["canonical"].pages),
        row["document"].document_id,
    ))
    if count >= len(ranked):
        chosen = ranked
    else:
        head = ranked[:max(1, count // 2)]
        remainder = ranked[len(head):]
        step = len(remainder) / max(1, count - len(head))
        chosen = head + [remainder[min(int(index * step), len(remainder) - 1)]
                         for index in range(count - len(head))]
    return [{"document_id": row["document"].document_id,
             "application_id": row["document"].application_id,
             "document_type": row["document"].document_type.value,
             "review_decision": "", "reviewer_note": ""} for row in chosen]


def validate_review(decisions: Path, expected_ids: set[str]) -> dict[str, Any]:
    with decisions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {row.get("document_id", "") for row in rows}
    if ids != expected_ids or any(row.get("review_decision", "").strip().casefold() != "pass"
                                  for row in rows):
        raise ReleaseError("review_sample_incomplete_or_failed")
    return {"reviewed_document_count": len(rows), "document_ids": sorted(ids)}


def release_candidate(output_root: Path, approval: dict[str, Any]) -> Path:
    candidate = output_root / "researcher_candidate"
    target = output_root / "researcher"
    if not candidate.is_dir():
        raise ReleaseError("candidate_package_missing")

    def build(root: Path) -> None:
        for source in candidate.rglob("*"):
            relative = source.relative_to(candidate)
            destination = root / relative
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest["release_status"] = "ready"
        manifest["review"] = approval
        manifest["files"] = {path.relative_to(root).as_posix(): sha256_file(path)
                             for path in root.rglob("*")
                             if path.is_file() and path.name != "manifest.json"}
        atomic_json(root / "manifest.json", manifest, 0o644)

    _atomic_directory(target, build)
    return target


def validate_candidate_package(output_root: Path) -> dict[str, Any]:
    candidate = output_root / "researcher_candidate"
    protected_redactions = output_root / "protected" / "redactions.jsonl"
    if not candidate.is_dir() or not (candidate / "manifest.json").exists():
        raise ReleaseError("candidate_package_missing")
    aliases: list[str] = []
    if protected_redactions.exists():
        for line in protected_redactions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line).get("text")
                if isinstance(value, str):
                    aliases.append(value)
    leaks = _package_leaks(candidate, aliases)
    if leaks:
        raise ReleaseError("candidate_package_privacy_validation_failed")
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.get("files", {}).items():
        path = candidate / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ReleaseError("candidate_manifest_checksum_failed")
    entities: set[str] = set()
    with (candidate / "entities.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            entities.add(row["entity_id"])
    with (candidate / "relationships.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_id"] not in entities or row["target_id"] not in entities:
                raise ReleaseError("relationship_foreign_key_failed")
    return {"status": "valid", "file_count": len(manifest.get("files", {})),
            "entity_count": len(entities)}
