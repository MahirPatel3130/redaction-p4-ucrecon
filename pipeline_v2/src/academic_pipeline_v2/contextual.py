from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .models import CVExtraction, ValueStatus
from .util import atomic_json, sha256_bytes


MINIMUM_COHORT = 20


def _retained_signature(extraction: CVExtraction) -> tuple[str, ...]:
    values: set[str] = set()
    for field in extraction.research_fields + extraction.research_methods:
        if field.status == ValueStatus.FOUND and isinstance(field.value, str):
            normalized = " ".join(field.value.casefold().split())
            if normalized:
                values.add(normalized)
    return tuple(sorted(values))


def assess_contextual_risk(applications: list[Any], processed: list[dict[str, Any]],
                           output_root: Path) -> dict[str, Any]:
    """Assess dataset-level quasi-identifier combinations without driving span deletion."""
    signatures: dict[str, tuple[str, ...]] = {}
    for row in processed:
        extraction = row.get("extraction")
        if isinstance(extraction, CVExtraction):
            signature = _retained_signature(extraction)
            if signature:
                signatures[row["document"].application_id] = signature
    cohort_size = len(applications)
    counts = Counter(signatures.values())
    by_application: dict[str, dict[str, Any]] = {}
    for application in applications:
        application_id = (application.application_id if hasattr(application, "application_id")
                          else str(application.get("application_id")))
        signature = signatures.get(application_id)
        if cohort_size < MINIMUM_COHORT:
            status = "not_evaluated_insufficient_cohort"
            factors: list[str] = []
        elif signature and counts[signature] == 1:
            status = "advisory"
            factors = ["unique_research_field_method_combination"]
        else:
            status = "clear"
            factors = []
        by_application[application_id] = {
            "application_id": application_id,
            "status": status,
            "risk_factor_codes": factors,
            "retained_signature_hash": (
                sha256_bytes("\u241f".join(signature).encode("utf-8")) if signature else None),
            "matching_application_count": counts.get(signature, 0) if signature else 0,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "2.0",
        "scope": "application_level_nonblocking_contextual_assessment",
        "cohort_size": cohort_size,
        "minimum_cohort_size": MINIMUM_COHORT,
        "applications": by_application,
    }
    atomic_json(output_root / "applications.json", artifact, mode=0o600)
    return {
        "scope": artifact["scope"],
        "cohort_size": cohort_size,
        "evaluated_application_count": sum(
            row["status"] != "not_evaluated_insufficient_cohort"
            for row in by_application.values()),
        "advisory_application_count": sum(
            row["status"] == "advisory" for row in by_application.values()),
        "blocks_document_release": False,
    }
