from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .util import atomic_json


def evaluate_run(config: PipelineConfig) -> dict[str, Any]:
    summary_path = config.output_root / "metrics" / "run_summary.json"
    if not summary_path.exists():
        raise ValueError("run_summary_missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ground_truth_candidates = [root / "ground_truth.json" for root in config.input_roots]
    ground_truth = next((path for path in ground_truth_candidates if path.exists()), None)
    accuracy: dict[str, Any] = {}
    if ground_truth is not None:
        truth = json.loads(ground_truth.read_text(encoding="utf-8"))
        inventory = json.loads((config.output_root / "protected" / "inventory.json").read_text(
            encoding="utf-8"))
        app_by_path = {row["source_application_path"]: row["application_id"]
                       for row in inventory.get("applications", [])}
        observed: dict[str, dict[str, int]] = {}
        for table, key in (("education.csv", "education_count"),
                           ("employment.csv", "employment_count"),
                           ("publications.csv", "publication_count")):
            path = config.output_root / "researcher_candidate" / table
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    observed.setdefault(row["application_id"], {}).setdefault(key, 0)
                    observed[row["application_id"]][key] += 1
        totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        expected_identifiers: list[tuple[str, str]] = []
        for app in truth.get("applications", []):
            app_id = app_by_path.get(app.get("application"))
            if not app_id:
                continue
            for key in ("education_count", "employment_count", "publication_count"):
                expected = int(app.get(key, 0))
                actual = observed.get(app_id, {}).get(key, 0)
                totals["true_positive"] += min(expected, actual)
                totals["false_positive"] += max(0, actual - expected)
                totals["false_negative"] += max(0, expected - actual)
            for field in ("applicant", "recommender"):
                if isinstance(app.get(field), str):
                    expected_identifiers.append((app_id, app[field]))
        denominator_precision = totals["true_positive"] + totals["false_positive"]
        denominator_recall = totals["true_positive"] + totals["false_negative"]
        redactions: set[tuple[str, str]] = set()
        redaction_path = config.output_root / "protected" / "redactions.jsonl"
        if redaction_path.exists():
            for line in redaction_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                redactions.add((row.get("application_id", ""), str(row.get("text", "")).casefold()))
        identifier_hits = sum((app_id, value.casefold()) in redactions
                              for app_id, value in expected_identifiers)
        accuracy = {
            "structured_item_precision": totals["true_positive"] / denominator_precision
            if denominator_precision else 1.0,
            "structured_item_recall": totals["true_positive"] / denominator_recall
            if denominator_recall else 1.0,
            "seeded_identifier_recall": identifier_hits / len(expected_identifiers)
            if expected_identifiers else 1.0,
            "expected_identifier_count": len(expected_identifiers),
        }
    result = {
        "model": config.model.model,
        "backend": config.model.backend,
        "document_count": summary.get("document_count", 0),
        "release_rate": (summary.get("released_document_count", 0)
                         / max(1, summary.get("document_count", 0))),
        "failure_rate": (summary.get("failed_document_count", 0)
                         / max(1, summary.get("document_count", 0))),
        "model_call_count": summary.get("model_call_count", 0),
        "ground_truth_available": ground_truth is not None,
        "accuracy": accuracy,
        "accuracy_note": "Synthetic count/span metrics are not proof of P4 safety"
                         if ground_truth else "No local ground-truth file was present",
    }
    atomic_json(config.output_root / "metrics" / "evaluation.json", result)
    return result
