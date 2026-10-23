from __future__ import annotations

import re
from pathlib import Path

from .config import PipelineConfig
from .errors import ExtractionError
from .models import ApplicationRecord, DocumentRecord, DocumentType
from .util import sha256_file, stable_id


CV_PATTERN = re.compile(r"(?:^|[^a-z])(?:cv|curriculum[ _-]?vitae)(?:[^a-z]|$)", re.IGNORECASE)


def build_inventory(config: PipelineConfig) -> tuple[list[ApplicationRecord], list[DocumentRecord]]:
    applications: list[ApplicationRecord] = []
    documents: list[DocumentRecord] = []
    seen_applications: set[str] = set()
    seen_documents: set[str] = set()
    root_names = [root.name.casefold() for root in config.input_roots]
    if len(root_names) != len(set(root_names)):
        raise ExtractionError("duplicate_input_root_names_require_unique_names")
    for root in config.input_roots:
        if not root.is_dir():
            raise ExtractionError("input_root_missing")
        root_namespace = f"{config.namespace}:{root.name}"
        for application_dir in sorted(
            (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
            key=lambda path: path.name.casefold(),
        ):
            relative_app = application_dir.relative_to(root).as_posix()
            app_id = stable_id("application", f"{root_namespace}/{relative_app}")
            if app_id in seen_applications:
                raise ExtractionError("application_id_collision")
            seen_applications.add(app_id)
            pdfs = sorted(
                (path for path in application_dir.iterdir()
                 if path.is_file() and path.suffix.casefold() == ".pdf"),
                key=lambda path: path.name.casefold(),
            )
            cvs = [path for path in pdfs if CV_PATTERN.search(path.stem)]
            letters = [path for path in pdfs if path not in cvs]
            flags: list[str] = []
            if not cvs:
                flags.append("missing_cv")
            elif len(cvs) > 1:
                flags.append("multiple_cv_candidates")
            if len(letters) not in {2, 3}:
                flags.append("unexpected_recommendation_letter_count")
            selected: list[tuple[Path, DocumentType, int | None]] = []
            if len(cvs) == 1:
                selected.append((cvs[0], DocumentType.CV, None))
            selected.extend((path, DocumentType.LETTER, index)
                            for index, path in enumerate(letters, 1))
            document_ids: list[str] = []
            for path, document_type, letter_index in selected:
                relative_document = f"{relative_app}/{path.name}"
                doc_id = stable_id("document", f"{root_namespace}/{relative_document}")
                if doc_id in seen_documents:
                    raise ExtractionError("document_id_collision")
                seen_documents.add(doc_id)
                document_ids.append(doc_id)
                documents.append(DocumentRecord(
                    application_id=app_id,
                    document_id=doc_id,
                    document_type=document_type,
                    letter_index=letter_index,
                    source_path=str(path.resolve()),
                    source_relative_path=relative_document,
                    source_sha256=sha256_file(path),
                    inventory_flags=list(flags),
                ))
            applications.append(ApplicationRecord(
                application_id=app_id,
                source_root_namespace=root_namespace,
                source_application_path=relative_app,
                status="inventory_flagged" if flags else "inventoried",
                inventory_flags=flags,
                document_ids=document_ids,
            ))
    return applications, documents
