from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]*)?\(?\d{3}\)?[ .-]*\d{3}[ .-]*\d{4}(?!\d)")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s,;]+|\b[A-Z0-9.-]+\.(?:edu|com|org|net)(?:/[^\s,;]*)?")
DOI_RE = re.compile(r"(?i)\b(?:doi\s*:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, value: str, length: int = 16) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if mode is not None:
        os.chmod(path, mode)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if mode is not None:
        os.chmod(path, mode)


def csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]],
              mode: int | None = None, protect_formulas: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(value) if protect_formulas else value
                             for key, value in row.items()})
    if mode is not None:
        os.chmod(path, mode)


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip())


def normalize(value: str) -> str:
    value = compact(value).casefold()
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip()


def valid_confidence(value: Any, required: bool = True) -> float | None:
    if value is None and not required:
        return None
    if isinstance(value, bool):
        raise ValueError("Model confidence must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model confidence must be a number") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError("Model confidence must be between 0 and 1")
    return number


def five_year_bucket(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = compact(str(value))
    if text.casefold() in {"present", "current", "ongoing"}:
        return "present"
    match = YEAR_RE.search(text)
    if not match:
        return None
    year = int(match.group())
    start = year - year % 5
    return f"{start}-{start + 4}"


def field_value(value: Any, status: str = "found", confidence: Any = None) -> dict[str, Any]:
    allowed = {"found", "not_found", "uncertain", "not_applicable", "model_failed"}
    if status not in allowed:
        raise ValueError("Unsupported field status")
    return {"value": value, "status": status,
            "model_confidence": valid_confidence(confidence, required=False)}


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_statistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("total", "mean", "median", "minimum", "maximum", "p90", "p95")}
    return {
        "total": round(sum(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "minimum": round(min(values), 3),
        "maximum": round(max(values), 3),
        "p90": round(percentile(values, 0.90), 3),
        "p95": round(percentile(values, 0.95), 3),
    }


class OpaqueRegistry:
    """Persistent exact-alias registry for protected organization/work resolution."""

    def __init__(self, path: Path, kind: str):
        self.path = path
        self.kind = kind
        if path.exists():
            self.data = read_json(path)
            if self.data.get("kind") != kind or not isinstance(self.data.get("entities"), dict):
                raise ValueError(f"Malformed {kind} registry")
        else:
            self.data = {"schema_version": "1.0", "kind": kind, "created_at": now_utc(),
                         "updated_at": now_utc(), "entities": {}, "alias_index": {}, "merges": []}

    def resolve(self, raw_value: str, entity_type: str, evidence: dict[str, Any] | None = None,
                strong_key: str | None = None) -> tuple[str, str]:
        raw_value = compact(raw_value)
        if not raw_value:
            raise ValueError(f"Cannot resolve empty {self.kind} value")
        alias_key = normalize(strong_key or raw_value)
        existing = self.data["alias_index"].get(alias_key)
        if existing:
            entity = self.data["entities"].get(existing)
            if not entity or entity.get("status") != "active":
                raise ValueError(f"Broken {self.kind} alias index")
            self._add_alias(entity, raw_value, evidence)
            return existing, "exact_alias"
        entity_id = f"{self.kind}_{uuid.uuid4().hex}"
        self.data["entities"][entity_id] = {
            "entity_id": entity_id, "entity_type": entity_type, "status": "active",
            "created_at": now_utc(), "aliases": [raw_value], "evidence": [evidence] if evidence else [],
        }
        self.data["alias_index"][alias_key] = entity_id
        return entity_id, "new_entity"

    def _add_alias(self, entity: dict[str, Any], raw_value: str,
                   evidence: dict[str, Any] | None) -> None:
        if raw_value not in entity["aliases"]:
            entity["aliases"].append(raw_value)
        if evidence and evidence not in entity["evidence"]:
            entity["evidence"].append(evidence)

    def save(self) -> None:
        self.data["updated_at"] = now_utc()
        atomic_json(self.path, self.data)

    def public_rows(self, used_ids: set[str]) -> list[dict[str, Any]]:
        return sorted(({
            f"{self.kind}_id": entity_id,
            "entity_type": entity.get("entity_type", "unknown"),
            "occurrence_count": len(entity.get("evidence", [])),
            "resolution_status": "resolved",
        } for entity_id, entity in self.data["entities"].items()
            if entity_id in used_ids and entity.get("status") == "active"),
            key=lambda row: row[f"{self.kind}_id"])


def replace_aliases(text: str, replacements: list[tuple[str, str]]) -> tuple[str, list[str]]:
    applied: list[str] = []
    unique = {}
    for raw, placeholder in replacements:
        raw = compact(raw)
        if raw and len(raw) >= 2:
            unique[(raw.casefold(), placeholder)] = (raw, placeholder)
    for raw, placeholder in sorted(unique.values(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(re.escape(raw), re.IGNORECASE)
        text, count = pattern.subn(lambda _: placeholder, text)
        if count:
            applied.append(placeholder)
    return text, sorted(set(applied))


def deterministic_identifiers(text: str) -> list[dict[str, Any]]:
    found = []
    for identifier_type, pattern in (("email", EMAIL_RE), ("phone_number", PHONE_RE),
                                     ("website", URL_RE), ("unique_identifier", DOI_RE)):
        for match in pattern.finditer(text):
            found.append({"identifier_type": identifier_type, "person_role": "unknown",
                          "text": match.group(), "model_confidence": None,
                          "detection_source": "deterministic_regex"})
    return found


def leakage_findings(text: str, protected_aliases: Iterable[str]) -> list[str]:
    findings = []
    scan_text = re.sub(r"\[\[[^\]]+\]\]", " ", text)
    scan_text = re.sub(
        r"\b(?:person|org|work|doc|app|rel|education|employment|publication|mention|entity)_[0-9a-f]+\b",
        " ", scan_text, flags=re.IGNORECASE)
    for label, pattern in (("email", EMAIL_RE), ("phone", PHONE_RE),
                           ("url", URL_RE), ("doi", DOI_RE)):
        if pattern.search(scan_text):
            findings.append(f"remaining_{label}")
    folded = text.casefold()
    for alias in protected_aliases:
        alias = compact(alias)
        if len(alias) >= 4 and not alias.isdigit() and alias.casefold() in folded:
            findings.append("remaining_protected_alias")
            break
    return sorted(set(findings))
