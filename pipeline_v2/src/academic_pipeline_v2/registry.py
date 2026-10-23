from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import (
    CVExtraction, CanonicalDocument, EntityMention, LetterExtraction, ResolvedRelationship,
)
from .util import normalized_key, stable_id, utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    alias_type TEXT NOT NULL,
    normalized_value TEXT NOT NULL,
    raw_value TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    document_id TEXT,
    UNIQUE(entity_id, alias_type, normalized_value, document_id)
);
CREATE INDEX IF NOT EXISTS aliases_lookup ON aliases(alias_type, normalized_value, verified);
CREATE TABLE IF NOT EXISTS local_parties (
    local_key TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    application_id TEXT,
    document_id TEXT,
    role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mentions (
    mention_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    application_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    identifier_type TEXT NOT NULL,
    role TEXT NOT NULL,
    page INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY,
    relationship_type TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES entities(entity_id),
    target_id TEXT NOT NULL REFERENCES entities(entity_id),
    application_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    confidence REAL,
    validation_status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    document_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    path TEXT NOT NULL,
    checksum TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(document_id, stage)
);
CREATE TABLE IF NOT EXISTS imports (
    import_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_checksum TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
"""


class EntityRegistry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()
        os.chmod(path, 0o600)

    def close(self) -> None:
        self.connection.close()
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.path) + suffix)
            if path.exists():
                os.chmod(path, 0o600)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _new_entity(self, entity_type: str, preferred_id: str | None = None,
                    metadata: dict[str, Any] | None = None) -> str:
        entity_id = preferred_id or f"{entity_type}_{uuid.uuid4().hex}"
        self.connection.execute(
            "INSERT OR IGNORE INTO entities(entity_id,entity_type,status,created_at,metadata_json) "
            "VALUES(?,?,?,?,?)",
            (entity_id, entity_type, "active", utc_now(), json.dumps(metadata or {}, sort_keys=True)),
        )
        return entity_id

    def local_entity(self, local_key: str, entity_type: str, role: str,
                     application_id: str, document_id: str | None = None,
                     preferred_id: str | None = None) -> str:
        row = self.connection.execute(
            "SELECT entity_id FROM local_parties WHERE local_key=?", (local_key,)
        ).fetchone()
        if row:
            return str(row["entity_id"])
        entity_id = self._new_entity(entity_type, preferred_id)
        self.connection.execute(
            "INSERT INTO local_parties(local_key,entity_id,application_id,document_id,role) "
            "VALUES(?,?,?,?,?)",
            (local_key, entity_id, application_id, document_id, role),
        )
        return entity_id

    def exact_entity(self, entity_type: str, alias_type: str, raw_value: str,
                     document_id: str, verified: bool = True) -> str:
        key = normalized_key(raw_value)
        row = self.connection.execute(
            "SELECT entity_id FROM aliases WHERE alias_type=? AND normalized_value=? "
            "AND verified=1 ORDER BY alias_id LIMIT 1", (alias_type, key)
        ).fetchone()
        entity_id = str(row["entity_id"]) if row else self._new_entity(entity_type)
        self.add_alias(entity_id, alias_type, raw_value, document_id, verified)
        return entity_id

    def add_alias(self, entity_id: str, alias_type: str, raw_value: str,
                  document_id: str, verified: bool) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO aliases(entity_id,alias_type,normalized_value,raw_value,verified,document_id) "
            "VALUES(?,?,?,?,?,?)",
            (entity_id, alias_type, normalized_key(raw_value), raw_value, int(verified), document_id),
        )

    def _merge_entity(self, retired_id: str, survivor_id: str) -> None:
        if retired_id == survivor_id:
            return
        aliases = self.connection.execute(
            "SELECT alias_type,normalized_value,raw_value,verified,document_id FROM aliases "
            "WHERE entity_id=?", (retired_id,)
        ).fetchall()
        for row in aliases:
            self.connection.execute(
                "INSERT OR IGNORE INTO aliases(entity_id,alias_type,normalized_value,raw_value,verified,document_id) "
                "VALUES(?,?,?,?,?,?)", (survivor_id, *tuple(row))
            )
        self.connection.execute("DELETE FROM aliases WHERE entity_id=?", (retired_id,))
        self.connection.execute("UPDATE local_parties SET entity_id=? WHERE entity_id=?",
                                (survivor_id, retired_id))
        self.connection.execute("UPDATE mentions SET entity_id=? WHERE entity_id=?",
                                (survivor_id, retired_id))
        self.connection.execute("UPDATE relationships SET source_id=? WHERE source_id=?",
                                (survivor_id, retired_id))
        self.connection.execute("UPDATE relationships SET target_id=? WHERE target_id=?",
                                (survivor_id, retired_id))
        self.connection.execute("UPDATE entities SET status='retired' WHERE entity_id=?", (retired_id,))

    def resolve_document(self, document: CanonicalDocument,
                         extraction: CVExtraction | LetterExtraction) -> dict[str, Any]:
        with self.transaction():
            applicant_id = self.local_entity(
                f"{document.application_id}:applicant", "person", "applicant",
                document.application_id,
            )
            mention_map: dict[str, str] = {}
            person_mentions: list[tuple[EntityMention, str]] = []
            for mention in extraction.mentions:
                entity_id: str
                if mention.identifier_type == "person_name":
                    if mention.person_role == "applicant":
                        entity_id = applicant_id
                    elif mention.person_role == "recommender":
                        entity_id = self.local_entity(
                            f"{document.document_id}:recommender", "person", "recommender",
                            document.application_id, document.document_id,
                        )
                    else:
                        entity_id = self.local_entity(
                            mention.local_entity_id or mention.mention_id, "person", mention.person_role,
                            document.application_id, document.document_id,
                        )
                    self.add_alias(entity_id, "name", mention.text, document.document_id, False)
                    person_mentions.append((mention, entity_id))
                elif mention.identifier_type == "organization":
                    entity_id = self.exact_entity("org", "organization_name", mention.text,
                                                  document.document_id, True)
                elif mention.identifier_type == "work_title":
                    entity_id = self.exact_entity("work", "work_title", mention.text,
                                                  document.document_id, True)
                elif mention.identifier_type in {"email", "unique_identifier", "phone_number", "website"}:
                    role_candidates = [row for row in person_mentions
                                       if mention.person_role != "unknown"
                                       and row[0].person_role == mention.person_role]
                    nearest = min(
                        role_candidates or person_mentions,
                        key=lambda row: abs(row[0].evidence.start - mention.evidence.start),
                        default=None,
                    )
                    if nearest and abs(nearest[0].evidence.start - mention.evidence.start) <= 500:
                        entity_id = nearest[1]
                        strong = mention.identifier_type in {"email", "unique_identifier"}
                        existing = self.connection.execute(
                            "SELECT entity_id FROM aliases WHERE alias_type=? AND normalized_value=? "
                            "AND verified=1 ORDER BY alias_id LIMIT 1",
                            (mention.identifier_type, normalized_key(mention.text)),
                        ).fetchone()
                        if existing and strong:
                            survivor = str(existing["entity_id"])
                            retired = entity_id
                            self._merge_entity(retired, survivor)
                            entity_id = survivor
                            for previous, previous_id in person_mentions:
                                if previous_id != survivor and previous_id == nearest[1]:
                                    mention_map[previous.mention_id] = survivor
                                    previous.resolved_entity_id = survivor
                            person_mentions = [(previous, survivor if previous_id == retired else previous_id)
                                               for previous, previous_id in person_mentions]
                        self.add_alias(entity_id, mention.identifier_type, mention.text,
                                       document.document_id, strong)
                    else:
                        entity_id = self.local_entity(
                            mention.local_entity_id or mention.mention_id, "identifier", "unknown",
                            document.application_id, document.document_id,
                        )
                else:
                    entity_id = self.local_entity(
                        mention.local_entity_id or mention.mention_id, "identifier", mention.person_role,
                        document.application_id, document.document_id,
                    )
                mention.resolved_entity_id = entity_id
                mention_map[mention.mention_id] = entity_id
                self.connection.execute(
                    "INSERT OR REPLACE INTO mentions VALUES(?,?,?,?,?,?,?,?,?)",
                    (mention.mention_id, entity_id, document.application_id, document.document_id,
                     mention.identifier_type, mention.person_role, mention.evidence.page,
                     mention.evidence.start, mention.evidence.end),
                )
            relationships = self._relationships(document, extraction, mention_map, applicant_id)
            return {"applicant_id": applicant_id, "mention_map": mention_map,
                    "relationships": relationships}

    def _relationships(self, document: CanonicalDocument,
                       extraction: CVExtraction | LetterExtraction,
                       mention_map: dict[str, str], applicant_id: str) -> list[ResolvedRelationship]:
        rows: list[ResolvedRelationship] = []
        if isinstance(extraction, LetterExtraction):
            recommender_id = self.local_entity(
                f"{document.document_id}:recommender", "person", "recommender",
                document.application_id, document.document_id,
            )
            rows.append(self._relationship("wrote_recommendation_for", recommender_id,
                                           applicant_id, document, None, "validated"))
            for proposed in extraction.relationships:
                source = mention_map.get(proposed.source_mention_id)
                target = mention_map.get(proposed.target_mention_id)
                if source and target:
                    rows.append(self._relationship(proposed.relationship_type, source, target,
                                                   document, proposed.model_confidence, "validated"))
        else:
            for mention in extraction.mentions:
                if mention.identifier_type == "person_name" and mention.person_role == "reference":
                    reference_id = mention_map.get(mention.mention_id)
                    if reference_id:
                        rows.append(self._relationship("listed_as_reference_by", reference_id,
                                                       applicant_id, document,
                                                       mention.model_confidence, "validated"))
            for item in extraction.education.items:
                if item.institution.value:
                    org_id = self.exact_entity("org", "organization_name",
                                               str(item.institution.value), document.document_id)
                    rows.append(self._relationship("educated_at", applicant_id, org_id, document,
                                                   item.item_model_confidence, "validated"))
            for item in extraction.employment.items:
                if item.organization.value:
                    org_id = self.exact_entity("org", "organization_name",
                                               str(item.organization.value), document.document_id)
                    rows.append(self._relationship("worked_at", applicant_id, org_id, document,
                                                   item.item_model_confidence, "validated"))
            for item in extraction.publications.items:
                strong = str(item.doi.value) if item.doi.value else (
                    f"{item.title.value}|{item.year.value}" if item.title.value else item.item_id)
                work_id = self.exact_entity("work", "work_key", strong, document.document_id)
                rows.append(self._relationship("authored", applicant_id, work_id, document,
                                               item.item_model_confidence, "validated"))
                for author in item.authors:
                    if author.is_applicant:
                        author_id = applicant_id
                    elif author.name.value:
                        key = normalized_key(str(author.name.value))
                        author_id = next((mention_map.get(mention.mention_id)
                                          for mention in extraction.mentions
                                          if mention.identifier_type == "person_name"
                                          and normalized_key(mention.text) == key), None)
                    else:
                        author_id = None
                    if author_id:
                        rows.append(self._relationship("authored", author_id, work_id, document,
                                                       author.name.model_confidence, "validated"))
                if item.venue.value:
                    venue_id = self.exact_entity("org", "venue_name", str(item.venue.value),
                                                 document.document_id)
                    rows.append(self._relationship("published_in", work_id, venue_id, document,
                                                   item.item_model_confidence, "validated"))
        unique = {row.relationship_id: row for row in rows}
        return list(unique.values())

    def _relationship(self, kind: str, source: str, target: str,
                      document: CanonicalDocument, confidence: float | None,
                      status: str) -> ResolvedRelationship:
        relationship_id = stable_id(
            "relationship", f"{document.document_id}:{kind}:{source}:{target}"
        )
        row = ResolvedRelationship(
            relationship_id=relationship_id, relationship_type=kind,
            source_id=source, target_id=target, application_id=document.application_id,
            document_id=document.document_id, model_confidence=confidence,
            validation_status=status,
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO relationships VALUES(?,?,?,?,?,?,?,?)",
            (row.relationship_id, row.relationship_type, row.source_id, row.target_id,
             row.application_id, row.document_id, row.model_confidence, row.validation_status),
        )
        return row

    def public_entities(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT e.entity_id,e.entity_type,COUNT(DISTINCT m.document_id) document_count,"
            "COUNT(DISTINCT m.application_id) application_count FROM entities e "
            "LEFT JOIN mentions m ON e.entity_id=m.entity_id WHERE e.status='active' "
            "GROUP BY e.entity_id,e.entity_type ORDER BY e.entity_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def find_alias(self, raw_value: str, alias_types: tuple[str, ...]) -> str | None:
        if not alias_types:
            return None
        placeholders = ",".join("?" for _ in alias_types)
        row = self.connection.execute(
            f"SELECT entity_id FROM aliases WHERE normalized_value=? AND alias_type IN ({placeholders}) "
            "ORDER BY verified DESC, alias_id LIMIT 1",
            (normalized_key(raw_value), *alias_types),
        ).fetchone()
        return str(row["entity_id"]) if row else None

    def public_relationships(self, released_documents: set[str]) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM relationships ORDER BY relationship_id").fetchall()
        return [dict(row) for row in rows if row["document_id"] in released_documents]

    def import_v1(self, payload: dict[str, Any], source_checksum: str,
                  apply: bool) -> dict[str, Any]:
        persons = payload.get("persons", {})
        if not isinstance(persons, dict):
            raise ValueError("invalid_v1_person_registry")
        active = [(person_id, value) for person_id, value in persons.items()
                  if isinstance(value, dict) and value.get("status", "active") == "active"
                  and str(person_id).startswith("person_")]
        summary = {"active_people": len(active), "applied": apply}
        if not apply:
            return summary
        import_id = stable_id("import", source_checksum)
        with self.transaction():
            previous = self.connection.execute("SELECT 1 FROM imports WHERE import_id=?",
                                               (import_id,)).fetchone()
            if previous:
                return {**summary, "already_imported": True}
            for person_id, value in active:
                self._new_entity("person", person_id, {"source": "v1_import"})
                for identifier in value.get("identifiers", []):
                    if not isinstance(identifier, dict):
                        continue
                    raw = identifier.get("value")
                    kind = identifier.get("identifier_type")
                    if isinstance(raw, str) and raw.strip() and isinstance(kind, str):
                        self.add_alias(person_id, kind, raw,
                                       str(identifier.get("document_id", "v1_import")),
                                       bool(identifier.get("verified", False)))
                roles = [str(role) for role in value.get("roles", []) if role]
                application_ids = [str(app) for app in value.get("application_ids", []) if app]
                document_ids = [str(doc) for doc in value.get("document_ids", []) if doc]
                for local_id in value.get("local_person_ids", []):
                    if not isinstance(local_id, str) or not local_id:
                        continue
                    self.connection.execute(
                        "INSERT OR IGNORE INTO local_parties(local_key,entity_id,application_id,document_id,role) "
                        "VALUES(?,?,?,?,?)",
                        (local_id, person_id, application_ids[0] if application_ids else None,
                         document_ids[0] if document_ids else None,
                         roles[0] if roles else "unknown"),
                    )
            self.connection.execute(
                "INSERT INTO imports VALUES(?,?,?,?,?)",
                (import_id, "v1_identity_registry", source_checksum, utc_now(),
                 json.dumps(summary, sort_keys=True)),
            )
        return summary
