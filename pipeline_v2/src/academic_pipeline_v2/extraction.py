from __future__ import annotations

import logging
import math
import unicodedata
from pathlib import Path
from typing import Any, Protocol

from .config import OCRConfig
from .errors import ExtractionError
from .models import (
    CanonicalDocument, CanonicalPage, DocumentRecord, TextBlock, ValueStatus,
)
from .util import sha256_bytes, stable_id


LOG = logging.getLogger(__name__)


class OCRBackend(Protocol):
    def extract(self, page: Any, dpi: int) -> list[
            tuple[str, tuple[float, float, float, float] | None, float | None]]:
        ...


class PaddleOCRBackend:
    """Lazy local OCR adapter; no model download is allowed by this package."""

    def __init__(self, config: OCRConfig):
        if not config.text_detection_model_dir or not config.text_recognition_model_dir:
            raise ExtractionError("local_ocr_model_paths_required")
        if (not config.text_detection_model_dir.is_dir()
                or not config.text_recognition_model_dir.is_dir()):
            raise ExtractionError("local_ocr_model_path_missing")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ExtractionError("paddleocr_not_installed") from exc
        try:
            arguments: dict[str, Any] = {
                "lang": config.language,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": bool(config.textline_orientation_model_dir),
                "device": config.device,
                "text_detection_model_dir": str(config.text_detection_model_dir),
                "text_recognition_model_dir": str(config.text_recognition_model_dir),
            }
            if config.textline_orientation_model_dir:
                if not config.textline_orientation_model_dir.is_dir():
                    raise ExtractionError("local_ocr_orientation_model_path_missing")
                arguments["textline_orientation_model_dir"] = str(
                    config.textline_orientation_model_dir)
            self.engine = PaddleOCR(**arguments)
        except Exception as exc:
            raise ExtractionError("paddleocr_initialization_failed") from exc

    def extract(self, page: Any, dpi: int) -> list[
            tuple[str, tuple[float, float, float, float] | None, float | None]]:
        import numpy as np

        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=(scale, scale), alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n)
        try:
            results = self.engine.predict(image)
            rows: list[tuple[str, tuple[float, float, float, float] | None, float | None]] = []
            for result in results:
                payload = result.json if hasattr(result, "json") else result
                payload = payload.get("res", payload) if isinstance(payload, dict) else {}
                texts = payload.get("rec_texts", [])
                boxes = payload.get("rec_boxes", [])
                scores = payload.get("rec_scores", [])
                for index, text in enumerate(texts):
                    box = boxes[index] if index < len(boxes) else None
                    bbox = tuple(float(value) / scale for value in box[:4]) if box is not None else None
                    score = float(scores[index]) if index < len(scores) else None
                    rows.append((str(text), bbox, score))
            return rows
        except Exception as exc:
            raise ExtractionError("paddleocr_page_failed") from exc


def _native_quality(text: str, page: Any, config: OCRConfig) -> list[str]:
    flags: list[str] = []
    alphanumeric = sum(character.isalnum() for character in text)
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    replacement = text.count("\ufffd")
    if alphanumeric < config.minimum_alphanumeric:
        flags.append("insufficient_native_text")
    if text and printable / len(text) < config.minimum_printable_ratio:
        flags.append("low_printable_ratio")
    if text and replacement / len(text) > config.maximum_replacement_ratio:
        flags.append("encoding_replacement_characters")
    if abs(int(getattr(page, "rotation", 0))) % 180:
        flags.append("rotated_page")
    image_coverage = 0.0
    area = max(1.0, float(page.rect.width * page.rect.height))
    for image in page.get_images(full=True):
        try:
            for rectangle in page.get_image_rects(image[0]):
                image_coverage += float(rectangle.width * rectangle.height) / area
        except Exception:
            continue
    if image_coverage > 0.65 and alphanumeric < config.minimum_alphanumeric * 4:
        flags.append("image_dominant_page")
    return sorted(set(flags))


def _native_blocks(page: Any, page_number: int) -> list[
        tuple[str, tuple[float, float, float, float], float | None]]:
    rows = []
    for block in page.get_text("blocks", sort=True):
        text = str(block[4]).strip()
        if text:
            rows.append((text, tuple(float(value) for value in block[:4]), None))
    return rows


def extract_canonical(record: DocumentRecord, config: OCRConfig,
                      ocr: OCRBackend | None = None) -> CanonicalDocument:
    import pymupdf

    pages: list[CanonicalPage] = []
    document_parts: list[str] = []
    document_cursor = 0
    document_flags: list[str] = []
    try:
        pdf = pymupdf.open(record.source_path)
    except Exception as exc:
        raise ExtractionError("pdf_open_failed") from exc
    try:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            native_text = page.get_text("text", sort=True)
            flags = _native_quality(native_text, page, config)
            use_ocr = bool(flags) and config.enabled
            method = "native"
            block_rows: list[tuple[str, tuple[float, float, float, float] | None, float | None]]
            if use_ocr:
                if ocr is None:
                    try:
                        ocr = PaddleOCRBackend(config)
                    except ExtractionError:
                        ocr = None
                if ocr is not None:
                    block_rows = ocr.extract(page, config.dpi)
                    method = "ocr"
                    if not block_rows:
                        flags.append("ocr_returned_no_text")
                else:
                    block_rows = _native_blocks(page, page_number)
                    flags.append("ocr_unavailable")
            else:
                block_rows = _native_blocks(page, page_number)
            page_parts: list[str] = []
            blocks: list[TextBlock] = []
            page_cursor = 0
            for block_index, (text, bbox, confidence) in enumerate(block_rows):
                text = unicodedata.normalize("NFKC", text).strip()
                if not text:
                    continue
                separator = "\n" if page_parts else ""
                page_parts.append(separator + text)
                page_start = page_cursor + len(separator)
                page_end = page_start + len(text)
                document_start = document_cursor + page_start
                blocks.append(TextBlock(
                    block_id=stable_id("block", f"{record.document_id}:{page_number}:{block_index}:{text}"),
                    page=page_number,
                    page_start=page_start,
                    page_end=page_end,
                    document_start=document_start,
                    document_end=document_start + len(text),
                    text=text,
                    bbox=bbox,
                    extraction_method=method,
                    extraction_confidence=confidence,
                ))
                page_cursor = page_end
            page_text = "".join(page_parts)
            page_start_global = document_cursor
            page_end_global = page_start_global + len(page_text)
            if method == "ocr" and any(
                    block.extraction_confidence is None
                    or block.extraction_confidence < config.minimum_ocr_confidence
                    for block in blocks):
                flags.append("ocr_below_confidence")
            unusable = (not page_text.strip() or "ocr_unavailable" in flags
                        or "ocr_below_confidence" in flags)
            status = ValueStatus.MODEL_FAILED if unusable else ValueStatus.FOUND
            if status == ValueStatus.MODEL_FAILED:
                flags.append("page_text_unavailable")
            pages.append(CanonicalPage(
                page=page_number,
                text=page_text,
                document_start=page_start_global,
                document_end=page_end_global,
                extraction_method=method if page_text.strip() else "failed",
                extraction_status=status,
                quality_flags=sorted(set(flags)),
                blocks=blocks,
            ))
            marker = f"[[PAGE:{page_number}]]\n"
            document_parts.append(marker + page_text)
            # Subsequent global offsets include the canonical page marker and separator.
            delta = len(marker)
            for block in blocks:
                block.document_start += delta
                block.document_end += delta
            pages[-1].document_start += delta
            pages[-1].document_end += delta
            document_cursor += len(marker) + len(page_text) + 2
            document_parts.append("\n\n")
            document_flags.extend(flags)
    finally:
        pdf.close()
    text = "".join(document_parts).rstrip()
    failed_pages = [page.page for page in pages if page.extraction_status == ValueStatus.MODEL_FAILED]
    return CanonicalDocument(
        application_id=record.application_id,
        document_id=record.document_id,
        document_type=record.document_type,
        letter_index=record.letter_index,
        source_sha256=record.source_sha256,
        source_reference=record.document_id,
        text=text,
        pages=pages,
        extraction_status=ValueStatus.MODEL_FAILED if failed_pages else ValueStatus.FOUND,
        extraction_flags=sorted(set(document_flags + (["one_or_more_pages_failed"] if failed_pages else []))),
    )


def page_for_offset(document: CanonicalDocument, start: int, end: int) -> CanonicalPage | None:
    for page in document.pages:
        if start >= page.document_start and end <= page.document_end:
            return page
    return None


def text_hash(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))
