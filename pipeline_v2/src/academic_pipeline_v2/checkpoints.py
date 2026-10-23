from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from . import PIPELINE_VERSION
from .models import DocumentCheckpoint, DocumentRecord
from .util import atomic_json, fingerprint, read_json, sha256_file, utc_now


T = TypeVar("T", bound=BaseModel)


class CheckpointStore:
    def __init__(self, protected_root: Path, config_fingerprint: str):
        self.root = protected_root / "checkpoints"
        self.config_fingerprint = config_fingerprint
        self.root.mkdir(parents=True, exist_ok=True)

    def document_root(self, document_id: str) -> Path:
        path = self.root / document_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def metadata_path(self, document_id: str) -> Path:
        return self.document_root(document_id) / "checkpoint.json"

    def load_metadata(self, record: DocumentRecord) -> DocumentCheckpoint:
        path = self.metadata_path(record.document_id)
        if path.exists():
            checkpoint = DocumentCheckpoint.model_validate(read_json(path))
            if (checkpoint.source_sha256 == record.source_sha256
                    and checkpoint.pipeline_version == PIPELINE_VERSION):
                checkpoint.config_fingerprint = self.config_fingerprint
                return checkpoint
        return DocumentCheckpoint(
            pipeline_version=PIPELINE_VERSION,
            document_id=record.document_id,
            source_sha256=record.source_sha256,
            config_fingerprint=self.config_fingerprint,
            updated_at=utc_now(),
        )

    def save_metadata(self, checkpoint: DocumentCheckpoint) -> None:
        checkpoint.updated_at = utc_now()
        atomic_json(self.metadata_path(checkpoint.document_id), checkpoint)

    def artifact_path(self, document_id: str, stage: str) -> Path:
        return self.document_root(document_id) / f"{stage}.json"

    def save_artifact(self, checkpoint: DocumentCheckpoint, stage: str,
                      value: BaseModel | dict[str, Any], stage_fingerprint_value: str) -> Path:
        path = self.artifact_path(checkpoint.document_id, stage)
        atomic_json(path, value)
        checkpoint.stages[stage] = "completed"
        checkpoint.stage_fingerprints[stage] = stage_fingerprint_value
        checkpoint.artifact_checksums[stage] = sha256_file(path)
        checkpoint.failure_stage = None
        checkpoint.failure_code = None
        self.save_metadata(checkpoint)
        return path

    def load_artifact(self, checkpoint: DocumentCheckpoint, stage: str,
                      model: type[T] | None = None,
                      stage_fingerprint_value: str | None = None) -> T | dict[str, Any] | None:
        path = self.artifact_path(checkpoint.document_id, stage)
        expected = checkpoint.artifact_checksums.get(stage)
        if checkpoint.stages.get(stage) != "completed" or not path.exists() or not expected:
            return None
        if (stage_fingerprint_value is not None
                and checkpoint.stage_fingerprints.get(stage) != stage_fingerprint_value):
            return None
        if sha256_file(path) != expected:
            return None
        value = read_json(path)
        return model.model_validate(value) if model else value

    def fail(self, checkpoint: DocumentCheckpoint, stage: str, code: str) -> None:
        checkpoint.stages[stage] = "failed"
        checkpoint.failure_stage = stage
        checkpoint.failure_code = code
        checkpoint.retry_counts[stage] = checkpoint.retry_counts.get(stage, 0) + 1
        self.save_metadata(checkpoint)


def stage_fingerprint(*values: Any) -> str:
    return fingerprint(values)
