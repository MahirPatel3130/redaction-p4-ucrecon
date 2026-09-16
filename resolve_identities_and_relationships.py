#!/usr/bin/env python3
"""Resolve direct-person identities and export a de-identified relationship dataset."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
SUCCESS_STATUSES = {"completed", "json_completed"}
STRONG_TYPES = {"email", "unique_identifier"}
REVIEW_TYPES = {"person_name", "phone_number", "website"}
IDENTITY_TYPES = REVIEW_TYPES | STRONG_TYPES | {"postal_address"}
DECISIONS = {"accept", "reject", "defer"}
CV_REFERENCE_CONTACTS = {"email", "phone_number", "website", "postal_address"}
HONORIFICS = re.compile(r"(?i)\b(?:dr|prof|professor|mr|mrs|ms|ph\.?d)\.?\b")
REVIEW_COLUMNS = [
    "candidate_id", "candidate_type", "priority", "left_person_id", "right_person_id",
    "left_local_person_ids", "right_local_person_ids", "left_roles", "right_roles",
    "evidence_types", "evidence_values", "confidence", "conflicts", "decision",
    "reviewer_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-json", type=Path, required=True)
    parser.add_argument("--recommendation-json", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path)
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, value: str, length: int = 16) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def csv_safe(value: str) -> str:
    """Prevent spreadsheet formula execution in the restricted review export."""
    return f"'{value}" if value.startswith(("=", "+", "-", "@")) else value


def atomic_write_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                             dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(registry, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip())


def normalize(identifier_type: str, value: str) -> str:
    value = compact(value)
    if identifier_type == "email":
        return value.casefold().rstrip(".,;:")
    if identifier_type == "phone_number":
        return re.sub(r"\D", "", value)
    if identifier_type == "website":
        value = re.sub(r"(?i)^https?://", "", value.casefold()).rstrip("/.,;:")
        return re.sub(r"^www\.", "", value)
    if identifier_type == "unique_identifier":
        return re.sub(r"(?i)^https?://orcid\.org/", "", value).upper()
    if identifier_type == "person_name":
        value = HONORIFICS.sub(" ", value)
        value = re.sub(r"[^\w\s'-]", " ", value, flags=re.UNICODE)
        return compact(value).casefold()
    return compact(value).casefold()


def entity_type(entity: dict[str, Any]) -> str:
    identifier_type = str(entity.get("identifier_type", ""))
    if identifier_type:
        return identifier_type
    category = str(entity.get("category", ""))
    return {
        "applicant_name": "person_name",
        "reference_identity": "person_name",
        "address": "postal_address",
    }.get(category, category)


def entity_role(entity: dict[str, Any]) -> str:
    return str(entity.get("person_role") or entity.get("owner_role") or "unknown")


def entity_verified(entity: dict[str, Any]) -> bool:
    return bool(entity.get("value_verified_against_pdf", entity.get("verified", False)))


def entity_mentions(entity: dict[str, Any]) -> list[str]:
    return sorted({str(item.get("mention_id")) for item in entity.get("occurrences", [])
                   if isinstance(item, dict) and item.get("mention_id")})


def entity_boxes(entity: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = []
    for item in entity.get("occurrences", []):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_normalized")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                values = [float(number) for number in bbox]
            except (TypeError, ValueError):
                continue
            boxes.append({"page": item.get("page"), "bbox": values,
                          "mention_id": item.get("mention_id")})
    return boxes


def evidence_from_entity(entity: dict[str, Any], document_id: str,
                         application_id: str) -> dict[str, Any]:
    identifier_type = entity_type(entity)
    value = str(entity.get("value", ""))
    return {
        "identifier_type": identifier_type,
        "value": value,
        "normalized_value": normalize(identifier_type, value),
        "verified": entity_verified(entity),
        "entity_id": str(entity.get("entity_id", "")),
        "mention_ids": entity_mentions(entity),
        "document_id": document_id,
        "application_id": application_id,
    }


def empty_party(local_id: str, role: str) -> dict[str, Any]:
    return {"local_person_id": local_id, "roles": {role}, "application_ids": set(),
            "document_ids": set(), "identifiers": [], "entity_ids": set(),
            "source": "structural"}


def add_entity(party: dict[str, Any], entity: dict[str, Any], document_id: str,
               application_id: str) -> None:
    party["roles"].add(entity_role(entity))
    party["application_ids"].add(application_id)
    party["document_ids"].add(document_id)
    if entity.get("entity_id"):
        party["entity_ids"].add(str(entity["entity_id"]))
    evidence = evidence_from_entity(entity, document_id, application_id)
    key = (evidence["identifier_type"], evidence["normalized_value"], document_id,
           evidence["entity_id"])
    existing = {(item["identifier_type"], item["normalized_value"], item["document_id"],
                 item["entity_id"]) for item in party["identifiers"]}
    if evidence["normalized_value"] and key not in existing:
        party["identifiers"].append(evidence)


def horizontal_gap(left: list[float], right: list[float]) -> float:
    if left[2] >= right[0] and right[2] >= left[0]:
        return 0.0
    return min(abs(left[2] - right[0]), abs(right[2] - left[0]))


def reference_assignment(contact: dict[str, Any], anchors: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    candidates: list[tuple[float, str]] = []
    for contact_box in entity_boxes(contact):
        for anchor in anchors:
            for anchor_box in anchor["boxes"]:
                if contact_box["page"] != anchor_box["page"]:
                    continue
                cb, ab = contact_box["bbox"], anchor_box["bbox"]
                vertical = max(0.0, cb[1] - ab[3])
                if cb[1] < ab[1] - 0.02 or vertical > 0.20:
                    continue
                horizontal = horizontal_gap(cb, ab)
                if horizontal > 0.25:
                    continue
                candidates.append((vertical + horizontal * 0.5, anchor["local_person_id"]))
    best_by_anchor: dict[str, float] = {}
    for score, local_id in candidates:
        best_by_anchor[local_id] = min(score, best_by_anchor.get(local_id, 10.0))
    ranked = sorted((score, local_id) for local_id, score in best_by_anchor.items())
    if not ranked:
        return None, []
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.03:
        return None, [local_id for _, local_id in ranked[:3]]
    return ranked[0][1], [local_id for _, local_id in ranked[:3]]


def party_id_for_reference(document_id: str, entity: dict[str, Any]) -> str:
    mentions = entity_mentions(entity)
    anchor = mentions[0] if mentions else str(entity.get("entity_id", "reference"))
    return f"{document_id}:reference:{hashlib.sha256(anchor.encode()).hexdigest()[:10]}"


def make_grouping_candidate(contact_local_id: str, anchor_id: str | None,
                            entity: dict[str, Any]) -> dict[str, Any]:
    right = anchor_id or "no_candidate_anchor"
    candidate_id = stable_id("match", f"reference_grouping|{contact_local_id}|{right}")
    return {
        "candidate_id": candidate_id, "candidate_type": "reference_grouping",
        "priority": "medium", "left_local_ids": [contact_local_id],
        "right_local_ids": [anchor_id] if anchor_id else [], "signals": [entity_type(entity)],
        "evidence_values": [str(entity.get("value", ""))], "confidence": 0.0,
        "conflicts": ["ambiguous_or_unassigned_reference_contact"],
        "document_id": document_id,
    }


def validate_aggregate(value: dict[str, Any], label: str) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported {label} schema_version")
    if not isinstance(value.get("applications"), dict) or not isinstance(value.get("documents"), dict):
        raise ValueError(f"Malformed {label} aggregate")
    seen: set[str] = set()
    application_paths: dict[str, str] = {}
    for app_id, application in value["applications"].items():
        if not isinstance(application, dict) or application.get("application_id") != app_id:
            raise ValueError(f"Malformed {label} application record")
        source_path = str(application.get("source_application_path", ""))
        if source_path and stable_id("app", Path(source_path).as_posix()) != app_id:
            raise ValueError(f"Invalid {label} application_id")
        application_paths[app_id] = source_path
    for key, record in value["documents"].items():
        if not isinstance(record, dict):
            raise ValueError(f"Malformed {label} document record")
        doc_id = str(record.get("document_id", ""))
        if not doc_id or doc_id in seen:
            raise ValueError(f"Missing or duplicate {label} document_id")
        seen.add(doc_id)
        if not record.get("application_id") or not record.get("document_type"):
            raise ValueError(f"Missing {label} document identity fields")
        if record["application_id"] not in application_paths:
            raise ValueError(f"Unknown {label} document application_id")
        expected_type = "cv" if label == "CV" else "recommendation_letter"
        if record["document_type"] != expected_type:
            raise ValueError(f"Unexpected {label} document_type")
        if record.get("source_relative_path") and record["source_relative_path"] != key:
            raise ValueError(f"Document key/path mismatch in {label}")
        if stable_id("doc", Path(key).as_posix()) != doc_id:
            raise ValueError(f"Invalid {label} document_id")


def collect_parties(cv: dict[str, Any], recommendations: dict[str, Any]) -> tuple[
        dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    parties: dict[str, dict[str, Any]] = {}
    relationship_seeds: list[dict[str, Any]] = []
    grouping_candidates: list[dict[str, Any]] = []

    def get_party(local_id: str, role: str) -> dict[str, Any]:
        party = parties.setdefault(local_id, empty_party(local_id, role))
        party["roles"].add(role)
        return party

    for record in recommendations["documents"].values():
        if record.get("status") not in SUCCESS_STATUSES:
            continue
        app_id, doc_id = str(record["application_id"]), str(record["document_id"])
        applicant_local = f"{app_id}:applicant"
        recommender_local = f"{doc_id}:recommender"
        for local_id, role in ((applicant_local, "applicant"),
                               (recommender_local, "recommender")):
            party = get_party(local_id, role)
            party["application_ids"].add(app_id)
            party["document_ids"].add(doc_id)
        for entity in record.get("entities", []):
            if (not isinstance(entity, dict)
                    or entity_role(entity) not in {"applicant", "recommender"}
                    or entity_type(entity) not in IDENTITY_TYPES):
                continue
            role = entity_role(entity)
            local_id = applicant_local if role == "applicant" else recommender_local
            add_entity(get_party(local_id, role), entity, doc_id, app_id)
        mention_ids = sorted({mention for entity in record.get("entities", [])
                              if isinstance(entity, dict)
                              for mention in entity_mentions(entity)})
        relationship_seeds.append({
            "relationship_type": "wrote_recommendation_for",
            "subject_local_id": recommender_local, "object_local_id": applicant_local,
            "application_id": app_id, "document_id": doc_id,
            "evidence_mention_ids": mention_ids, "confidence": 1.0,
            "resolution_method": "recommendation_letter_structure",
            "review_status": "verified",
        })

    for record in cv["documents"].values():
        if record.get("status") not in SUCCESS_STATUSES:
            continue
        app_id, doc_id = str(record["application_id"]), str(record["document_id"])
        applicant_local = f"{app_id}:applicant"
        applicant = get_party(applicant_local, "applicant")
        applicant["application_ids"].add(app_id)
        applicant["document_ids"].add(doc_id)
        entities = [entity for entity in record.get("entities", []) if isinstance(entity, dict)]
        for entity in entities:
            if entity_role(entity) == "applicant" and entity_type(entity) in IDENTITY_TYPES:
                add_entity(applicant, entity, doc_id, app_id)
        anchor_entities = [entity for entity in entities
                           if entity_role(entity) == "reference"
                           and entity_type(entity) == "person_name"]
        anchors = []
        for entity in anchor_entities:
            local_id = party_id_for_reference(doc_id, entity)
            party = get_party(local_id, "reference")
            party["source"] = "cv_reference_anchor"
            add_entity(party, entity, doc_id, app_id)
            anchors.append({"local_person_id": local_id, "entity": entity,
                            "boxes": entity_boxes(entity)})
            relationship_seeds.append({
                "relationship_type": "listed_as_reference_by",
                "subject_local_id": local_id, "object_local_id": applicant_local,
                "application_id": app_id, "document_id": doc_id,
                "evidence_mention_ids": entity_mentions(entity), "confidence": 0.95,
                "resolution_method": "cv_reference_section",
                "review_status": "structurally_inferred",
            })
        contacts = [entity for entity in entities if entity_role(entity) == "reference"
                    and entity_type(entity) in CV_REFERENCE_CONTACTS]
        for entity in contacts:
            assigned, candidates = reference_assignment(entity, anchors)
            if assigned:
                add_entity(get_party(assigned, "reference"), entity, doc_id, app_id)
                continue
            local_id = f"{doc_id}:reference_unassigned:{hashlib.sha256(str(entity.get('entity_id', '')).encode()).hexdigest()[:10]}"
            party = get_party(local_id, "reference")
            party["source"] = "unassigned_reference_contact"
            add_entity(party, entity, doc_id, app_id)
            possible_anchors = candidates or [anchor["local_person_id"] for anchor in anchors]
            if possible_anchors:
                grouping_candidates.extend(make_grouping_candidate(
                    local_id, anchor_id, entity) for anchor_id in possible_anchors)
            else:
                grouping_candidates.append(make_grouping_candidate(
                    local_id, None, entity))
    return parties, relationship_seeds, grouping_candidates


def new_registry() -> dict[str, Any]:
    stamp = now_utc()
    return {"schema_version": SCHEMA_VERSION, "created_at": stamp, "updated_at": stamp,
            "persons": {}, "local_party_index": {}, "rejected_candidates": [],
            "decisions": [], "merges": []}


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return new_registry()
    registry = read_json(path)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported identity registry schema_version")
    for field, expected in (("persons", dict), ("local_party_index", dict),
                            ("rejected_candidates", list), ("decisions", list),
                            ("merges", list)):
        if not isinstance(registry.get(field), expected):
            raise ValueError(f"Malformed registry field: {field}")
    return registry


def active_person_id(registry: dict[str, Any], person_id: str) -> str:
    visited = set()
    while person_id in registry["persons"] and registry["persons"][person_id].get("merged_into"):
        if person_id in visited:
            raise ValueError("Registry contains a person merge cycle")
        visited.add(person_id)
        person_id = registry["persons"][person_id]["merged_into"]
    return person_id


def create_person(registry: dict[str, Any], local_id: str) -> str:
    person_id = f"person_{uuid.uuid4().hex}"
    registry["persons"][person_id] = {
        "person_id": person_id, "status": "active", "created_at": now_utc(),
        "merged_into": None, "local_person_ids": [local_id], "roles": [],
        "application_ids": [], "document_ids": [], "identifiers": [],
        "resolution_methods": ["singleton"],
    }
    registry["local_party_index"][local_id] = person_id
    return person_id


def ensure_persons(registry: dict[str, Any], parties: dict[str, dict[str, Any]]) -> None:
    for local_id in sorted(parties):
        person_id = registry["local_party_index"].get(local_id)
        if person_id:
            person_id = active_person_id(registry, person_id)
            if person_id not in registry["persons"]:
                raise ValueError("Registry local-party index references a missing person")
            registry["local_party_index"][local_id] = person_id
        else:
            person_id = create_person(registry, local_id)
        person = registry["persons"][person_id]
        party = parties[local_id]
        person["local_person_ids"] = sorted(set(person["local_person_ids"]) | {local_id})
        person["roles"] = sorted(set(person["roles"]) | party["roles"])
        person["application_ids"] = sorted(set(person["application_ids"]) | party["application_ids"])
        person["document_ids"] = sorted(set(person["document_ids"]) | party["document_ids"])
        existing = {(item["identifier_type"], item["normalized_value"], item["document_id"],
                     item["entity_id"]) for item in person["identifiers"]}
        for item in party["identifiers"]:
            key = (item["identifier_type"], item["normalized_value"], item["document_id"],
                   item["entity_id"])
            if key not in existing:
                person["identifiers"].append(item)
                existing.add(key)


def merge_people(registry: dict[str, Any], person_ids: Iterable[str], method: str,
                 evidence: str) -> str:
    active_ids = sorted({active_person_id(registry, person_id) for person_id in person_ids})
    if not active_ids:
        raise ValueError("Cannot merge an empty person set")
    if len(active_ids) == 1:
        return active_ids[0]
    canonical = min(active_ids, key=lambda person_id: (
        registry["persons"][person_id].get("created_at", ""), person_id))
    target = registry["persons"][canonical]
    for retired_id in active_ids:
        if retired_id == canonical:
            continue
        retired = registry["persons"][retired_id]
        for field in ("local_person_ids", "roles", "application_ids", "document_ids",
                      "resolution_methods"):
            target[field] = sorted(set(target[field]) | set(retired.get(field, [])))
        existing = {(item["identifier_type"], item["normalized_value"], item["document_id"],
                     item["entity_id"]) for item in target["identifiers"]}
        for item in retired.get("identifiers", []):
            key = (item["identifier_type"], item["normalized_value"], item["document_id"],
                   item["entity_id"])
            if key not in existing:
                target["identifiers"].append(item)
                existing.add(key)
        retired["status"] = "retired"
        retired["merged_into"] = canonical
        for local_id in retired.get("local_person_ids", []):
            registry["local_party_index"][local_id] = canonical
        registry["merges"].append({"retired_person_id": retired_id,
                                    "surviving_person_id": canonical, "method": method,
                                    "evidence": evidence, "timestamp": now_utc()})
    target["resolution_methods"] = sorted(set(target["resolution_methods"]) | {method})
    return canonical


def parse_decisions(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.is_file():
        raise ValueError("Decisions CSV does not exist")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"candidate_id", "left_local_person_ids", "right_local_person_ids", "decision"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Decisions CSV is missing required columns")
        decisions = []
        for row in reader:
            decision = str(row.get("decision", "")).strip().casefold()
            if not decision:
                continue
            if decision not in DECISIONS:
                raise ValueError("Decisions CSV contains an invalid decision")
            if not row.get("candidate_id"):
                raise ValueError("Decisions CSV contains a decision without candidate_id")
            row["decision"] = decision
            decisions.append({key: str(value or "") for key, value in row.items()})
    return decisions


def apply_decisions(registry: dict[str, Any], decisions: list[dict[str, str]]) -> None:
    recorded = {(item.get("candidate_id"), item.get("decision"))
                for item in registry["decisions"]}
    rejected = set(registry["rejected_candidates"])
    actions_by_candidate: dict[str, str] = {}
    for decision in decisions:
        candidate_id = decision["candidate_id"]
        action = decision["decision"]
        previous_action = actions_by_candidate.setdefault(candidate_id, action)
        if previous_action != action:
            raise ValueError("Decisions CSV contains conflicting decisions")
        left_ids = [value for value in decision.get("left_local_person_ids", "").split("|")
                    if value]
        right_ids = [value for value in decision.get("right_local_person_ids", "").split("|")
                     if value]
        signals = sorted(value for value in decision.get("evidence_types", "").split("|") if value)
        candidate_type = decision.get("candidate_type", "")
        if candidate_type == "reference_grouping":
            expected = stable_id(
                "match", f"reference_grouping|{left_ids[0] if left_ids else ''}|{right_ids[0] if right_ids else 'no_candidate_anchor'}")
        else:
            if not left_ids or not right_ids:
                raise ValueError("Identity decision is missing local parties")
            ordered = sorted((left_ids[0], right_ids[0]))
            expected = stable_id(
                "match", f"{candidate_type}|{ordered[0]}|{ordered[1]}|{'|'.join(signals)}")
        if expected != candidate_id:
            raise ValueError("Decisions CSV contains an invalid candidate_id")
        unknown = [value for value in left_ids + right_ids
                   if value not in registry["local_party_index"]]
        if unknown:
            raise ValueError("Decisions CSV references unknown local parties")
        if action == "accept":
            local_ids = left_ids + right_ids
            person_ids = [registry["local_party_index"].get(local_id) for local_id in local_ids]
            if len([value for value in person_ids if value]) < 2:
                raise ValueError("Accepted decision references unknown local parties")
            merge_people(registry, [value for value in person_ids if value],
                         "manual_review", candidate_id)
            rejected.discard(candidate_id)
        elif action == "reject":
            rejected.add(candidate_id)
        if (candidate_id, action) not in recorded:
            registry["decisions"].append({
                "candidate_id": candidate_id, "decision": action,
                "reviewer_note": decision.get("reviewer_note", ""), "timestamp": now_utc(),
            })
    registry["rejected_candidates"] = sorted(rejected)


def auto_merge_strong(registry: dict[str, Any], parties: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    local_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for local_id, party in parties.items():
        person_id = active_person_id(registry, registry["local_party_index"][local_id])
        for item in party["identifiers"]:
            if item["identifier_type"] in STRONG_TYPES and item["verified"]:
                key = (item["identifier_type"], item["normalized_value"])
                by_key[key].add(person_id)
                local_by_key[key].add(local_id)
    conflicts = []
    for key, person_ids in sorted(by_key.items()):
        active_ids = {active_person_id(registry, value) for value in person_ids}
        established = {person_id for person_id in active_ids
                       if len(registry["persons"][person_id].get("local_person_ids", [])) > 1}
        if len(established) > 1:
            conflicts.append({"key": key, "person_ids": sorted(active_ids),
                              "local_ids": sorted(local_by_key[key])})
            continue
        merge_people(registry, active_ids, f"exact_verified_{key[0]}", stable_id("evidence", "|".join(key)))
    return conflicts


def person_for_local(registry: dict[str, Any], local_id: str) -> str:
    return active_person_id(registry, registry["local_party_index"][local_id])


def name_parts(value: str) -> tuple[str, str]:
    tokens = normalize("person_name", value).replace(",", " ").split()
    if not tokens:
        return "", ""
    return tokens[-1], tokens[0][:1]


def candidate_from_parties(candidate_type: str, left: dict[str, Any], right: dict[str, Any],
                           signals: list[str], values: list[str], confidence: float,
                           priority: str = "medium", conflicts: list[str] | None = None) -> dict[str, Any]:
    left_id, right_id = left["local_person_id"], right["local_person_id"]
    ordered = sorted((left_id, right_id))
    candidate_id = stable_id("match", f"{candidate_type}|{ordered[0]}|{ordered[1]}|{'|'.join(sorted(signals))}")
    return {"candidate_id": candidate_id, "candidate_type": candidate_type,
            "priority": priority, "left_local_ids": [left_id], "right_local_ids": [right_id],
            "signals": sorted(set(signals)), "evidence_values": values,
            "confidence": round(confidence, 3), "conflicts": conflicts or []}


def generate_review_candidates(registry: dict[str, Any], parties: dict[str, dict[str, Any]],
                               grouping: list[dict[str, Any]],
                               strong_conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = list(grouping)
    local_ids = sorted(parties)
    for index, left_id in enumerate(local_ids):
        left = parties[left_id]
        for right_id in local_ids[index + 1:]:
            right = parties[right_id]
            if person_for_local(registry, left_id) == person_for_local(registry, right_id):
                continue
            signals, values, scores = [], [], []
            for left_item in left["identifiers"]:
                if left_item["identifier_type"] not in REVIEW_TYPES:
                    continue
                for right_item in right["identifiers"]:
                    if left_item["identifier_type"] != right_item["identifier_type"]:
                        continue
                    kind = left_item["identifier_type"]
                    left_value, right_value = left_item["normalized_value"], right_item["normalized_value"]
                    if not left_value or not right_value:
                        continue
                    if left_value == right_value:
                        signals.append(f"exact_{kind}")
                        values.extend([left_item["value"], right_item["value"]])
                        scores.append({"person_name": 0.80, "phone_number": 0.90,
                                       "website": 0.88}[kind])
                    elif kind == "person_name":
                        left_surname, left_initial = name_parts(left_item["value"])
                        right_surname, right_initial = name_parts(right_item["value"])
                        ratio = difflib.SequenceMatcher(None, left_value, right_value).ratio()
                        if left_surname and left_surname == right_surname and left_initial == right_initial and ratio >= 0.94:
                            signals.append("high_similarity_person_name")
                            values.extend([left_item["value"], right_item["value"]])
                            scores.append(0.72)
            if signals:
                candidates.append(candidate_from_parties(
                    "identity_match", left, right, signals, list(dict.fromkeys(values)),
                    max(scores)))
    for conflict in strong_conflicts:
        local = conflict["local_ids"]
        pair = next(((left_id, right_id) for index, left_id in enumerate(local)
                     for right_id in local[index + 1:]
                     if person_for_local(registry, left_id) != person_for_local(registry, right_id)),
                    None)
        if pair is None:
            continue
        left, right = parties[pair[0]], parties[pair[1]]
        kind, raw_key = conflict["key"]
        candidates.append(candidate_from_parties(
            "strong_identifier_conflict", left, right, [f"conflicting_{kind}"],
            [raw_key], 0.0, "high", ["multiple_established_people_share_strong_identifier"]))
    rejected = set(registry["rejected_candidates"])
    unique = {}
    for candidate in candidates:
        if candidate["candidate_id"] in rejected:
            continue
        left_ids, right_ids = candidate.get("left_local_ids", []), candidate.get("right_local_ids", [])
        if left_ids and right_ids:
            known_left = [value for value in left_ids if value in registry["local_party_index"]]
            known_right = [value for value in right_ids if value in registry["local_party_index"]]
            if known_left and known_right and any(
                    person_for_local(registry, left) == person_for_local(registry, right)
                    for left in known_left for right in known_right):
                continue
        unique[candidate["candidate_id"]] = candidate
    return sorted(unique.values(), key=lambda item: (
        {"high": 0, "medium": 1, "low": 2}.get(item["priority"], 3), item["candidate_id"]))


def serialize_party(party: dict[str, Any], person_id: str) -> dict[str, Any]:
    return {"local_person_id": party["local_person_id"], "person_id": person_id,
            "roles": sorted(party["roles"]), "application_ids": sorted(party["application_ids"]),
            "document_ids": sorted(party["document_ids"]),
            "entity_ids": sorted(party["entity_ids"]), "identifiers": party["identifiers"],
            "source": party["source"]}


def build_relationships(seeds: list[dict[str, Any]], registry: dict[str, Any]) -> list[dict[str, Any]]:
    relationships = []
    for seed in seeds:
        subject = person_for_local(registry, seed["subject_local_id"])
        object_id = person_for_local(registry, seed["object_local_id"])
        relationship_id = stable_id(
            "rel", f"{seed['document_id']}|{seed['relationship_type']}|{seed['subject_local_id']}|{seed['object_local_id']}")
        relationships.append({"relationship_id": relationship_id, **seed,
                              "subject_person_id": subject, "object_person_id": object_id})
    return sorted(relationships, key=lambda item: item["relationship_id"])


def combined_catalog(cv: dict[str, Any], recommendations: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    app_paths: dict[str, str] = {}
    path_ids: dict[str, str] = {}
    applications: dict[str, dict[str, Any]] = {}
    documents = []
    seen_documents = set()
    for source, label in ((cv, "cv"), (recommendations, "recommendation")):
        for app_id, app in source["applications"].items():
            path = str(app.get("source_application_path", ""))
            if app_id in app_paths and path and app_paths[app_id] and path != app_paths[app_id]:
                raise ValueError("Application path conflict between aggregates")
            if path and path in path_ids and path_ids[path] != app_id:
                raise ValueError("Application ID conflict between aggregates")
            app_paths.setdefault(app_id, path)
            if path:
                path_ids[path] = app_id
            row = applications.setdefault(app_id, {"application_id": app_id, "has_cv": False,
                                                    "recommendation_letter_count": 0,
                                                    "source_statuses": []})
            row["source_statuses"].append(str(app.get("status", "unknown")))
        for record in source["documents"].values():
            doc_id = str(record["document_id"])
            if doc_id in seen_documents:
                raise ValueError("Document ID occurs in both aggregates")
            seen_documents.add(doc_id)
            app_id = str(record["application_id"])
            row = applications.setdefault(app_id, {"application_id": app_id, "has_cv": False,
                                                    "recommendation_letter_count": 0,
                                                    "source_statuses": []})
            doc_type = str(record["document_type"])
            if doc_type == "cv":
                row["has_cv"] = True
            elif doc_type == "recommendation_letter":
                row["recommendation_letter_count"] += 1
            documents.append({"document_id": doc_id, "application_id": app_id,
                              "document_type": doc_type, "letter_index": record.get("letter_index"),
                              "status": str(record.get("status", "unknown"))})
    for row in applications.values():
        statuses = row.pop("source_statuses")
        row["status"] = "completed" if statuses and all(
            value in SUCCESS_STATUSES or value == "completed" for value in statuses) else "partial_or_failed"
    return sorted(applications.values(), key=lambda item: item["application_id"]), sorted(
        documents, key=lambda item: item["document_id"])


def review_rows(candidates: list[dict[str, Any]], parties: dict[str, dict[str, Any]],
                registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in candidates:
        left_ids, right_ids = item.get("left_local_ids", []), item.get("right_local_ids", [])
        left_parties = [parties[value] for value in left_ids if value in parties]
        right_parties = [parties[value] for value in right_ids if value in parties]
        left_people = sorted({person_for_local(registry, value) for value in left_ids
                              if value in registry["local_party_index"]})
        right_people = sorted({person_for_local(registry, value) for value in right_ids
                               if value in registry["local_party_index"]})
        rows.append({
            "candidate_id": item["candidate_id"], "candidate_type": item["candidate_type"],
            "priority": item["priority"], "left_person_id": "|".join(left_people),
            "right_person_id": "|".join(right_people),
            "left_local_person_ids": "|".join(left_ids),
            "right_local_person_ids": "|".join(right_ids),
            "left_roles": "|".join(sorted({role for party in left_parties for role in party["roles"]})),
            "right_roles": "|".join(sorted({role for party in right_parties for role in party["roles"]})),
            "evidence_types": "|".join(item.get("signals", [])),
            "evidence_values": csv_safe(" || ".join(item.get("evidence_values", []))),
            "confidence": item.get("confidence", 0.0),
            "conflicts": "|".join(item.get("conflicts", [])), "decision": "",
            "reviewer_note": "",
        })
    return rows


def active_research_people(registry: dict[str, Any], parties: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    current_ids = sorted({person_for_local(registry, local_id) for local_id in parties})
    rows = []
    for person_id in current_ids:
        person = registry["persons"][person_id]
        local_count = len([value for value in person["local_person_ids"] if value in parties])
        methods = set(person.get("resolution_methods", []))
        if "manual_review" in methods:
            status = "reviewed_match"
        elif any(value.startswith("exact_verified_") for value in methods):
            status = "strong_identifier_match"
        elif local_count > 1:
            status = "structurally_linked"
        else:
            status = "singleton"
        current_parties = [parties[value] for value in person["local_person_ids"] if value in parties]
        rows.append({"person_id": person_id,
                     "roles": sorted({role for party in current_parties for role in party["roles"]}),
                     "application_count": len({app for party in current_parties for app in party["application_ids"]}),
                     "document_count": len({doc for party in current_parties for doc in party["document_ids"]}),
                     "resolution_status": status})
    return rows


def deidentified_relationships(relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: item[key] for key in (
        "relationship_id", "relationship_type", "subject_person_id", "object_person_id",
        "application_id", "document_id", "confidence", "review_status")}
            for item in relationships]


def emit_outputs(output_root: Path, run_id: str, status: str,
                 applications: list[dict[str, Any]], documents: list[dict[str, Any]],
                 people: list[dict[str, Any]], parties: dict[str, dict[str, Any]],
                 registry: dict[str, Any], relationships: list[dict[str, Any]],
                 candidates: list[dict[str, Any]], input_checksums: dict[str, str]) -> None:
    restricted = output_root / "restricted"
    researcher = output_root / "researcher"
    rows = review_rows(candidates, parties, registry)
    identity_resolution = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "parties": [serialize_party(parties[local_id], person_for_local(registry, local_id))
                    for local_id in sorted(parties)],
        "people": [registry["persons"][person["person_id"]] for person in people],
    }
    restricted.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(restricted, 0o700)
    previous_umask = os.umask(0o077)
    try:
        write_json(restricted / "identity_resolution.json", identity_resolution)
        write_json(restricted / "relationship_evidence.json", {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "relationships": relationships})
        write_csv(restricted / "review_queue.csv", REVIEW_COLUMNS, rows)
        write_json(restricted / "resolution_summary.json", {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "status": status,
            "counts": {"applications": len(applications), "documents": len(documents),
                       "people": len(people), "relationships": len(relationships),
                       "pending_review": len(candidates)},
            "input_checksums": input_checksums,
        })
    finally:
        os.umask(previous_umask)
    public_relationships = deidentified_relationships(relationships)
    dataset = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "status": status,
               "counts": {"applications": len(applications), "documents": len(documents),
                          "people": len(people), "relationships": len(public_relationships),
                          "pending_review": len(candidates)},
               "applications": applications, "documents": documents,
               "people": people, "relationships": public_relationships}
    write_json(researcher / "dataset.json", dataset)
    write_csv(researcher / "applications.csv",
              ["application_id", "has_cv", "recommendation_letter_count", "status"], applications)
    write_csv(researcher / "documents.csv",
              ["document_id", "application_id", "document_type", "letter_index", "status"], documents)
    write_csv(researcher / "people.csv",
              ["person_id", "roles", "application_count", "document_count", "resolution_status"],
              ({**row, "roles": "|".join(row["roles"])} for row in people))
    write_csv(researcher / "relationships.csv",
              ["relationship_id", "relationship_type", "subject_person_id", "object_person_id",
               "application_id", "document_id", "confidence", "review_status"],
              public_relationships)
    generated = [path for path in sorted(researcher.iterdir()) if path.name != "manifest.json"]
    write_json(researcher / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "status": status,
        "input_checksums": input_checksums,
        "files": {path.name: file_sha256(path) for path in generated},
    })


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()
    cv_path = args.cv_json.resolve()
    recommendation_path = args.recommendation_json.resolve()
    registry_path = args.registry.resolve()
    output_root = args.output_root.resolve()
    for path, label in ((cv_path, "CV aggregate"),
                        (recommendation_path, "recommendation aggregate")):
        if not path.is_file():
            logging.error("Missing %s path=%s", label, path)
            return 2
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        logging.error("Output directory must be new or empty path=%s", output_root)
        return 2
    input_directories = {cv_path.parent, recommendation_path.parent}
    if (output_root == registry_path.parent or output_root in registry_path.parents
            or output_root in input_directories
            or any(directory in output_root.parents for directory in input_directories)):
        logging.error("Output directory must be separate from inputs and registry")
        return 2
    try:
        cv = read_json(cv_path)
        recommendations = read_json(recommendation_path)
        validate_aggregate(cv, "CV")
        validate_aggregate(recommendations, "recommendation")
        input_checksums = {"cv_json_sha256": file_sha256(cv_path),
                           "recommendation_json_sha256": file_sha256(recommendation_path)}
        applications, documents = combined_catalog(cv, recommendations)
        parties, seeds, grouping = collect_parties(cv, recommendations)
        registry = load_registry(registry_path)
        ensure_persons(registry, parties)
        decisions = parse_decisions(args.decisions_csv.resolve() if args.decisions_csv else None)
        apply_decisions(registry, decisions)
        strong_conflicts = auto_merge_strong(registry, parties)
        candidates = generate_review_candidates(registry, parties, grouping, strong_conflicts)
        relationships = build_relationships(seeds, registry)
        people = active_research_people(registry, parties)
        status = "ready_with_pending_review" if candidates else "ready"
        run_id = f"run_{uuid.uuid4().hex}"
        output_root.mkdir(parents=True, exist_ok=True)
        emit_outputs(output_root, run_id, status, applications, documents, people, parties,
                     registry, relationships, candidates, input_checksums)
        registry["updated_at"] = now_utc()
        atomic_write_registry(registry_path, registry)
    except (csv.Error, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logging.error("Resolution failed error=%s", type(exc).__name__)
        return 2
    logging.info("Resolution completed applications=%d documents=%d people=%d relationships=%d pending_review=%d seconds=%.1f",
                 len(applications), len(documents), len(people), len(relationships),
                 len(candidates), time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
