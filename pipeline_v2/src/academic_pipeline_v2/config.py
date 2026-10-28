from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigurationError
from .util import fingerprint


class OCRConfig(BaseModel):
    enabled: bool = True
    device: str = "cpu"
    language: str = "en"
    dpi: int = Field(180, ge=96, le=400)
    minimum_alphanumeric: int = Field(40, ge=0)
    minimum_printable_ratio: float = Field(0.85, ge=0, le=1)
    maximum_replacement_ratio: float = Field(0.05, ge=0, le=1)
    minimum_ocr_confidence: float = Field(0.85, ge=0, le=1)
    text_detection_model_dir: Path | None = None
    text_recognition_model_dir: Path | None = None
    textline_orientation_model_dir: Path | None = None


class ModelConfig(BaseModel):
    backend: str = "vllm"
    model: str = "Qwen/Qwen3-14B"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "local-only"
    context_length: int = Field(32768, ge=4096)
    max_inflight: int = Field(4, ge=1, le=32)
    cv_max_tokens: int = Field(8192, ge=512)
    letter_max_tokens: int = Field(4096, ge=256)
    qa_max_tokens: int = Field(2048, ge=128)
    timeout_seconds: float = Field(600, gt=0)
    max_retries: int = Field(1, ge=0, le=3)
    store_raw_responses: bool = False

    @field_validator("backend")
    @classmethod
    def valid_backend(cls, value: str) -> str:
        if value not in {"vllm", "mock"}:
            raise ValueError("backend must be vllm or mock")
        return value

    @model_validator(mode="after")
    def local_vllm_only(self) -> "ModelConfig":
        if self.backend == "vllm":
            host = (urlparse(self.base_url).hostname or "").casefold()
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("vLLM endpoint must be loopback-only")
        return self


class NERConfig(BaseModel):
    enabled: bool = False
    backend: str = "none"
    model_path: Path | None = None
    device: str = "cpu"
    minimum_score: float = Field(0.65, ge=0, le=1)
    chunk_characters: int = Field(2000, ge=256, le=20000)
    candidates_per_call: int = Field(40, ge=1, le=100)

    @field_validator("backend")
    @classmethod
    def valid_backend(cls, value: str) -> str:
        if value not in {"none", "spacy", "gliner", "mock"}:
            raise ValueError("NER backend must be none, spacy, gliner, or mock")
        return value

    @model_validator(mode="after")
    def enabled_backend(self) -> "NERConfig":
        if self.enabled and self.backend == "none":
            raise ValueError("enabled NER requires a non-none backend")
        if self.enabled and self.backend != "mock" and self.model_path is None:
            raise ValueError("enabled NER requires a local model_path")
        return self


class PrivacyConfig(BaseModel):
    policy_path: Path
    # Retained for configuration compatibility. The finite gate performs at most one
    # deterministic repair batch whenever this value is greater than zero.
    maximum_repairs: int = Field(2, ge=0, le=5)
    review_fraction: float = Field(0.10, gt=0, le=1)
    review_minimum: int = Field(20, ge=1)
    require_human_review: bool = True


class PipelineConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_roots: list[Path]
    output_root: Path
    state_root: Path
    namespace: str = "academic-recruitment"
    cpu_workers: int = Field(4, ge=1)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    ner: NERConfig = Field(default_factory=NERConfig)
    privacy: PrivacyConfig

    @field_validator("input_roots")
    @classmethod
    def require_inputs(cls, value: list[Path]) -> list[Path]:
        if not value:
            raise ValueError("input_roots must not be empty")
        return value

    def resolved(self, config_path: Path) -> "PipelineConfig":
        base = config_path.resolve().parent
        data = self.model_dump()
        for key in ("output_root", "state_root"):
            path = Path(data[key])
            data[key] = path if path.is_absolute() else (base / path).resolve()
        data["input_roots"] = [path if Path(path).is_absolute() else (base / path).resolve()
                               for path in data["input_roots"]]
        for key in ("text_detection_model_dir", "text_recognition_model_dir",
                    "textline_orientation_model_dir"):
            path_value = data["ocr"].get(key)
            if path_value:
                path = Path(path_value)
                data["ocr"][key] = path if path.is_absolute() else (base / path).resolve()
        ner_path = data["ner"].get("model_path")
        if ner_path:
            path = Path(ner_path)
            data["ner"]["model_path"] = path if path.is_absolute() else (base / path).resolve()
        policy = Path(data["privacy"]["policy_path"])
        data["privacy"]["policy_path"] = policy if policy.is_absolute() else (base / policy).resolve()
        return PipelineConfig.model_validate(data)

    def cache_fingerprint(self) -> str:
        value = self.model_dump(mode="json")
        value.pop("output_root", None)
        value.pop("state_root", None)
        return fingerprint(value)


def load_config(path: Path) -> PipelineConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be an object")
        return PipelineConfig.model_validate(raw).resolved(path)
    except Exception as exc:
        raise ConfigurationError("invalid_configuration") from exc
