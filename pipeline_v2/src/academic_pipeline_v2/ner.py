from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import NERConfig
from .errors import ConfigurationError


@dataclass(frozen=True)
class NERSpan:
    start: int
    end: int
    text: str
    identifier_type: str
    label: str
    score: float | None
    backend: str


class NERBackend(Protocol):
    name: str

    def detect(self, text: str) -> list[NERSpan]:
        ...

    def close(self) -> None:
        ...


class NullNERBackend:
    name = "none"

    def detect(self, text: str) -> list[NERSpan]:
        return []

    def close(self) -> None:
        return None


class MockNERBackend:
    """Small deterministic recognizer for fictional tests only."""

    name = "mock"

    def detect(self, text: str) -> list[NERSpan]:
        import re

        rows: list[NERSpan] = []
        patterns = (
            (r"\b(?:Applicant|Recommender|Coauthor) [A-Z][A-Za-z'-]+\b", "person_name"),
            (r"\bExample (?:University|Journal)\b", "organization"),
            (r"\bImportant Paper\b", "work_title"),
        )
        for pattern, identifier_type in patterns:
            for match in re.finditer(pattern, text):
                rows.append(NERSpan(match.start(), match.end(), match.group(), identifier_type,
                                    identifier_type, 0.99, self.name))
        return _deduplicate(rows)

    def close(self) -> None:
        return None


def _chunks(text: str, maximum: int) -> list[tuple[int, str]]:
    if len(text) <= maximum:
        return [(0, text)]
    result: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + maximum)
        if end < len(text):
            boundary = text.rfind("\n", start + maximum // 2, end)
            if boundary > start:
                end = boundary + 1
        result.append((start, text[start:end]))
        start = end
    return result


SPACY_LABELS = {
    "PERSON": "person_name",
    "ORG": "organization",
    "FAC": "organization",
    "WORK_OF_ART": "work_title",
    "GPE": "location",
    "LOC": "location",
}

GLINER_LABELS = {
    "person name": "person_name",
    "organization": "organization",
    "publication title": "work_title",
    "postal address": "postal_address",
    "location": "location",
}


class SpacyNERBackend:
    name = "spacy"

    def __init__(self, config: NERConfig):
        try:
            import spacy
        except ImportError as exc:
            raise ConfigurationError("spacy_ner_not_installed") from exc
        path = _require_local_model(config.model_path)
        try:
            self.engine = spacy.load(str(path))
            self.engine.max_length = max(self.engine.max_length, config.chunk_characters + 1000)
        except Exception as exc:
            raise ConfigurationError("spacy_ner_model_load_failed") from exc
        self.config = config
        self._lock = threading.Lock()

    def detect(self, text: str) -> list[NERSpan]:
        chunks = _chunks(text, self.config.chunk_characters)
        with self._lock:
            documents = list(self.engine.pipe((value for _, value in chunks), batch_size=16))
        rows: list[NERSpan] = []
        for (offset, _), document in zip(chunks, documents, strict=True):
            for entity in document.ents:
                identifier_type = SPACY_LABELS.get(entity.label_)
                if identifier_type:
                    rows.append(NERSpan(
                        start=offset + entity.start_char, end=offset + entity.end_char,
                        text=entity.text, identifier_type=identifier_type,
                        label=entity.label_, score=None, backend=self.name,
                    ))
        return _deduplicate(rows)

    def close(self) -> None:
        return None


class GlinerNERBackend:
    name = "gliner"

    def __init__(self, config: NERConfig):
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise ConfigurationError("gliner_ner_not_installed") from exc
        path = _require_local_model(config.model_path)
        try:
            self.engine = GLiNER.from_pretrained(str(path), local_files_only=True)
            if config.device != "cpu" and hasattr(self.engine, "to"):
                self.engine.to(config.device)
        except Exception as exc:
            raise ConfigurationError("gliner_ner_model_load_failed") from exc
        self.config = config
        self._lock = threading.Lock()

    def detect(self, text: str) -> list[NERSpan]:
        rows: list[NERSpan] = []
        labels = list(GLINER_LABELS)
        for offset, chunk in _chunks(text, self.config.chunk_characters):
            with self._lock:
                entities: list[dict[str, Any]] = self.engine.predict_entities(
                    chunk, labels, threshold=self.config.minimum_score)
            for entity in entities:
                label = str(entity.get("label", "")).casefold()
                identifier_type = GLINER_LABELS.get(label)
                start, end = entity.get("start"), entity.get("end")
                score = entity.get("score")
                if (identifier_type and isinstance(start, int) and isinstance(end, int)
                        and 0 <= start < end <= len(chunk)):
                    rows.append(NERSpan(
                        start=offset + start, end=offset + end, text=chunk[start:end],
                        identifier_type=identifier_type, label=label,
                        score=float(score) if score is not None else None, backend=self.name,
                    ))
        return _deduplicate(rows)

    def close(self) -> None:
        return None


def _require_local_model(path: Path | None) -> Path:
    if path is None or not path.exists():
        raise ConfigurationError("local_ner_model_path_missing")
    return path


def _deduplicate(rows: list[NERSpan]) -> list[NERSpan]:
    unique = {(row.start, row.end, row.identifier_type): row for row in rows}
    return sorted(unique.values(), key=lambda row: (row.start, row.end, row.identifier_type))


def create_ner_backend(config: NERConfig) -> NERBackend:
    if not config.enabled or config.backend == "none":
        return NullNERBackend()
    if config.backend == "mock":
        return MockNERBackend()
    if config.backend == "spacy":
        return SpacyNERBackend(config)
    if config.backend == "gliner":
        return GlinerNERBackend(config)
    raise ConfigurationError("unsupported_ner_backend")
