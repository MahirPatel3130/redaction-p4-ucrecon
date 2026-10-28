from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .util import read_json, write_csv


def _strings(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            result.update(_strings(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_strings(item))
    elif isinstance(value, str) and len(value.strip()) >= 3:
        result.add(value.strip().casefold())
    return result


def build_privacy_audit(config: PipelineConfig, per_type: int = 5) -> dict[str, Any]:
    """Write a protected, stratified audit sample. Never call this for researcher export."""
    protected = config.output_root / "protected"
    output = protected / "audit" / "privacy_findings.csv"
    inventory = read_json(protected / "inventory.json")
    document_types = {row["document_id"]: row["document_type"]
                      for row in inventory.get("documents", [])}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((protected / "checkpoints").glob("*/privacy.json")):
        artifact = read_json(path)
        privacy = artifact.get("privacy", {})
        text = str(artifact.get("deidentified_text", ""))
        semantic_path = path.parent / "semantic.json"
        structured_values = _strings(read_json(semantic_path)) if semantic_path.exists() else set()
        for finding in privacy.get("findings", []):
            start, end = finding.get("start"), finding.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
                value = ""
                context = ""
            else:
                value = text[start:end]
                context = text[max(0, start - 80):min(len(text), end + 80)].replace("\n", " ")
            document_id = path.parent.name
            document_type = document_types.get(document_id, "unknown")
            groups[(document_type, str(finding.get("finding_type", "unknown")))].append({
                "document_id": path.parent.name,
                "document_type": document_type,
                "release_status": privacy.get("release_status"),
                "finding_type": finding.get("finding_type"),
                "detector": finding.get("detector"),
                "disposition": finding.get("disposition"),
                "grounded": finding.get("grounded"),
                "model_confidence": finding.get("model_confidence"),
                "start": start,
                "end": end,
                "text": value,
                "context": context,
                "matches_structured_extraction": bool(value and value.casefold() in structured_values),
                "audit_classification": "",
                "reviewer_note": "",
            })
    sampled = [row for key in sorted(groups) for row in groups[key][:per_type]]
    fields = [
        "document_id", "document_type", "release_status", "finding_type", "detector", "disposition",
        "grounded", "model_confidence", "start", "end", "text", "context",
        "matches_structured_extraction", "audit_classification", "reviewer_note",
    ]
    write_csv(output, fields, sampled, mode=0o600)
    return {"audit_path": str(output), "sample_count": len(sampled),
            "classification": "P4-sensitive-protected-only"}
