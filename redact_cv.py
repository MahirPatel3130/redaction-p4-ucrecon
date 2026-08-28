#!/usr/bin/env python3
"""Offline, human-review-required CV redaction with Gemma vision and PDF geometry."""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_DEFAULT = "google/gemma-4-31B-it"
CV_NAME = re.compile(r"^(?:cv|curriculum[_ -]?vitae)\.pdf$", re.IGNORECASE)
CATEGORIES = {
    "applicant_name", "email", "phone_number", "website", "address",
    "coauthor_identity", "reference_identity", "publication_information",
    "other_identifying_data",
}
OWNER_ROLES = {"applicant", "coauthor", "reference", "advisor", "committee_member",
               "institution", "document_furniture", "unknown"}

CONTACT_PROMPT = """Detect every identifying contact or identity region on this CV page.
Return JSON only: {"detections":[{"box_2d":[y_min,x_min,y_max,x_max],
"category":"email","owner_role":"applicant","text":"visible text"}]}.
Coordinates MUST use Gemma's native 0..1000 grid in y,x,y,x order. Valid categories:
applicant_name, email, phone_number, website, address, reference_identity,
coauthor_identity, other_identifying_data. Valid owner roles: applicant, coauthor,
reference, advisor, committee_member, institution, document_furniture, unknown.
Include names, emails, phones/faxes, URLs, addresses, usernames, IDs, advisors,
committee members, references, and collaborators. Do not use markdown or commentary."""

SCHOLARLY_PROMPT = """Detect every identifying scholarly-information region on this CV page.
Return JSON only: {"detections":[{"box_2d":[y_min,x_min,y_max,x_max],
"category":"publication_information","owner_role":"applicant","text":"visible text"}]}.
Coordinates MUST use Gemma's native 0..1000 grid in y,x,y,x order. Detect complete
publication citations, titles, DOI/ISBN values, presentations, grants, named authors,
collaborators, advisors, and committee members. Use category publication_information for
complete citations/titles, coauthor_identity for standalone collaborator names, and
other_identifying_data for grants or other linkable scholarly details. Valid owner roles:
applicant, coauthor, reference, advisor, committee_member, institution, unknown.
Do not use markdown or commentary."""

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]*)?\(?\d{3}\)?[ .-]*\d{3}[ .-]*\d{4}(?!\d)")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s,;]+|\b[A-Z0-9.-]+\.(?:edu|com|org|net)(?:/[^\s,;]*)?")
DOI_RE = re.compile(r"(?i)\b(?:doi\s*:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--dpi", type=int, default=144)
    p.add_argument("--margin-points", type=float, default=1.5)
    p.add_argument("--debug-artifacts", action="store_true")
    p.add_argument("--json-only", action="store_true",
                   help="Produce the entity inventory without generating a PDF")
    p.add_argument("--reuse-model-responses-from", type=Path,
                   help="Replay protected *.model-responses.json files without loading the model")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args()
    if not 72 <= args.dpi <= 600:
        p.error("--dpi must be between 72 and 600")
    if not 0 <= args.margin_points <= 12:
        p.error("--margin-points must be between 0 and 12")
    return args


def compact_text(value: str) -> str:
    """Comparison form resilient to punctuation and letter-spaced PDF text."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def pdf_lines(page: Any) -> list[dict[str, Any]]:
    lines = []
    for group in line_groups(page_words(page)):
        if not group:
            continue
        lines.append({"text": " ".join(w["text"] for w in group),
                      "words": group, "rect": union_rect([w["rect"] for w in group])})
    return lines


def native_box(item: dict[str, Any], page: Any) -> list[float] | None:
    box = item.get("box_2d")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        y0, x0, y1, x1 = [max(0.0, min(1000.0, float(v))) / 1000 for v in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0 * page.rect.width, y0 * page.rect.height, x1 * page.rect.width, y1 * page.rect.height]


def best_pdf_phrase(model_text: str, lines: list[dict[str, Any]], box: list[float] | None,
                    max_words: int = 14) -> tuple[str, list[float] | None, float]:
    """Return the most similar exact PDF word sequence near a model region."""
    if not model_text or not lines:
        return "", None, 0.0
    nearby = [line for line in lines if box and rect_intersects(line["rect"],
              [box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5])]
    candidates = nearby or lines
    target = compact_text(model_text)
    target_words = max(1, len(model_text.split()))
    best = ("", None, 0.0)
    for line in candidates:
        words = line["words"]
        low, high = max(1, target_words - 3), min(len(words), max_words, target_words + 4)
        for size in range(low, high + 1):
            for start in range(0, len(words) - size + 1):
                chosen = words[start:start + size]
                text = " ".join(w["text"] for w in chosen)
                score = difflib.SequenceMatcher(None, target, compact_text(text)).ratio()
                if score > best[2]:
                    best = (text, union_rect([w["rect"] for w in chosen]), score)
    return best


def looks_like_identity(value: str) -> bool:
    if not value or len(value.split()) > 12:
        return False
    lowered = value.lower()
    if any(word in lowered for word in ("escherichia", "biosynthesis", "mutation", "university of")):
        return False
    has_initial = bool(re.search(r"\b[A-Z]\.", value))
    capitals = re.findall(r"\b[A-Z][A-Za-z'-]+", value)
    return has_initial or len(capitals) >= 2


def exact_record(category: str, owner: str, value: str, page: Any, page_number: int,
                 rect: list[float], source: str, confidence: float = 1.0) -> dict[str, Any]:
    owner = owner if owner in OWNER_ROLES else "unknown"
    return {"category": category, "owner_role": owner, "text": value.strip(), "page": page_number,
            "bbox_normalized": [rect[0] / page.rect.width, rect[1] / page.rect.height,
                                 rect[2] / page.rect.width, rect[3] / page.rect.height],
            "detection_source": source, "geometry_source": "pdf_text",
            "value_verified_against_pdf": True, "match_confidence": round(confidence, 3)}


def infer_contact_owner(page_number: int, page_count: int, rect: list[float], category: str,
                        reference_y: float | None = None) -> str:
    y = rect[1]
    # Repeated footer classification is refined later across occurrences.
    if y / 792 > .92:
        return "institution"
    if reference_y is not None and y > reference_y:
        return "reference"
    if page_number == 1 and y / 792 < .35:
        return "applicant"
    return "unknown"


def text_in_box(lines: list[dict[str, Any]], box: list[float]) -> tuple[str, list[float] | None]:
    selected = [line for line in lines if rect_intersects(line["rect"],
                [box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3])]
    if not selected:
        return "", None
    return " ".join(line["text"] for line in selected), union_rect([line["rect"] for line in selected])


def trim_address(value: str) -> str:
    value = EMAIL_RE.sub("", value)
    value = PHONE_RE.sub("", value)
    return re.sub(r"[•·|,;\s]+$", "", re.sub(r"\s+", " ", value)).strip()


def citation_authors(value: str) -> list[str]:
    prefix = re.split(r'[“\"]|\b(?:19|20)\d{2}\b', value, maxsplit=1)[0]
    return [match.group(0).strip(" ,;") for match in
            re.finditer(r"\b[A-Z][A-Za-z'-]+,\s*(?:[A-Z]\.\s*){1,3}", prefix)]


def person_surname(value: str) -> str:
    value = re.sub(r"(?i)\b(?:ph\.?d\.?|dr\.?|curriculum vitae|advisor|pi)\b", "", value)
    value = value.replace(":", " ").strip(" ,.;")
    if "," in value and len(value.split(",", 1)[0].split()) == 1:
        return compact_text(value.split(",", 1)[0])
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", value)
    return compact_text(words[-1]) if words else ""


def split_scholarly_value(value: str) -> list[str]:
    value = re.sub(r"_+", " ", value)
    # This also handles a quote or publication year immediately after initials.
    author_pattern = r"\b[A-Z][A-Za-z'-]+,\s*(?:[A-Z]\.\s*){1,3}(?=[,;]|\s+(?:19|20)\d{2}|\s*[“\"])"
    author_start = re.search(author_pattern, value)
    if author_start and not re.match(r"(?i)^\s*(?:thesis|dissertation)", value):
        value = value[author_start.start():]
    starts = [m.start() for m in re.finditer(author_pattern, value)]
    pieces = []
    if len(starts) <= 1:
        pieces = [value]
    else:
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(value)
            piece = value[start:end]
            # Do not split at additional authors before the title quote.
            if pieces and not re.search(r'[“\"]|\b(?:19|20)\d{2}\b', pieces[-1]):
                pieces[-1] += " " + piece
            else:
                pieces.append(piece)
    cleaned = []
    for piece in pieces:
        piece = re.split(r"(?i)\b(?:TEACHING EXPERIENCE|AWARDS|REFERENCES|PRESENTATIONS)\b", piece)[0]
        piece = re.sub(r"\s+", " ", piece).strip(" ,.;")
        if looks_like_scholarly_information(piece) and len(piece.split()) >= 4:
            cleaned.append(piece)
    return cleaned


def looks_like_scholarly_information(value: str) -> bool:
    return bool(re.search(r'(?i)[“\"]|\b(?:publication|presented|journal|thesis|dissertation|abstract|doi|isbn)\b', value))


def clean_json_inventory(source: Path, response_path: Path, model: str) -> dict[str, Any]:
    """Build a conservative, PDF-grounded inventory from protected Gemma responses."""
    import pymupdf
    raw_pages = json.loads(response_path.read_text(encoding="utf-8"))
    doc = pymupdf.open(source)
    records: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    try:
        page_count = doc.page_count
        for index, page in enumerate(doc):
            page_number = index + 1
            lines = pdf_lines(page)
            reference_y = next((line["rect"][1] for line in lines
                                if line["text"].strip().upper() in {"REFERENCES", "REFERENCE"}), None)
            if reference_y is not None:
                for line in lines:
                    if line["rect"][1] <= reference_y:
                        continue
                    match = re.search(r"\b[A-Z][A-Za-z'-]+(?:\s+[A-Z]\.)?\s+[A-Z][A-Za-z'-]+,\s*Ph\.?D\.?", line["text"])
                    if match:
                        records.append(exact_record("reference_identity", "reference", match.group(0),
                                                    page, page_number, line["rect"], "pdf_reference_section"))
            # Exact deterministic contact values are authoritative and never merged across categories.
            for item in deterministic_detections(page, page_words(page), page_number, 0.0):
                if item["category"] not in {"email", "phone_number", "website", "publication_information"}:
                    continue
                rect = [item["bbox_normalized"][0] * page.rect.width,
                        item["bbox_normalized"][1] * page.rect.height,
                        item["bbox_normalized"][2] * page.rect.width,
                        item["bbox_normalized"][3] * page.rect.height]
                owner = infer_contact_owner(page_number, page_count, rect, item["category"], reference_y)
                records.append(exact_record(item["category"], owner, item["text"], page,
                                            page_number, rect, "pdf_regex"))
            for pass_data in raw_pages.get(str(page_number), []):
                for raw in pass_data.get("responses", []):
                    try:
                        detections = extract_json(raw)["detections"]
                    except ValueError:
                        discarded.append({"page": page_number, "reason": "malformed_saved_response"})
                        continue
                    for detection in detections:
                        category = str(detection.get("category", detection.get("label", ""))).lower()
                        owner = str(detection.get("owner_role", "unknown")).lower()
                        model_value = str(detection.get("text", "")).strip()
                        box = native_box(detection, page)
                        if category not in CATEGORIES or not box or not model_value:
                            continue
                        # Regex-backed categories must resolve to an exact source value. This rejects hallucinated contacts.
                        if category in {"email", "phone_number", "website"}:
                            phrase, rect, score = best_pdf_phrase(model_value, lines, box, max_words=8)
                            compact_verified = compact_text(model_value) in compact_text(page.get_text())
                            if (score < .86 or not rect) and compact_verified:
                                phrase, rect, score = model_value, box, .9
                            elif score < .86 or not rect:
                                discarded.append({"page": page_number, "category": category,
                                                  "reason": "contact_value_not_verified", "model_text": model_value})
                                continue
                            pattern = {"email": EMAIL_RE, "phone_number": PHONE_RE, "website": URL_RE}[category]
                            match = pattern.search(phrase)
                            value = match.group(0) if match else model_value
                            # If the compact model value exists in the PDF, preserve it for fragmented footer text.
                            if not compact_verified:
                                discarded.append({"page": page_number, "category": category,
                                                  "reason": "contact_value_not_in_pdf", "model_text": model_value})
                                continue
                            records.append(exact_record(category, owner, value, page, page_number, rect,
                                                        "gemma+pdf_match", score))
                            continue
                        phrase, rect, score = best_pdf_phrase(model_value, lines, box,
                                                              max_words=48 if category == "publication_information" else 16)
                        if category == "publication_information" and score < .72:
                            boxed_text, boxed_rect = text_in_box(lines, box)
                            if boxed_rect and len(boxed_text.split()) >= 4 and re.search(
                                    r'(?i)[“\"]|publication|presented|journal|thesis|dissertation|abstract|doi|isbn', boxed_text):
                                phrase, rect, score = boxed_text, boxed_rect, .8
                        threshold = .55 if category in {"publication_information", "other_identifying_data"} else .72
                        if score < threshold or not rect:
                            discarded.append({"page": page_number, "category": category,
                                              "reason": "model_value_not_grounded", "model_text": model_value,
                                              "best_score": round(score, 3)})
                            continue
                        if category == "publication_information" and not looks_like_scholarly_information(phrase):
                            discarded.append({"page": page_number, "category": category,
                                              "reason": "failed_publication_shape", "model_text": model_value,
                                              "matched_text": phrase})
                            continue
                        if category in {"applicant_name", "coauthor_identity", "reference_identity"} and not looks_like_identity(phrase):
                            discarded.append({"page": page_number, "category": category,
                                              "reason": "failed_identity_shape", "model_text": model_value,
                                              "matched_text": phrase})
                            continue
                        if category == "address":
                            phrase = trim_address(phrase)
                            if len(phrase) < 5:
                                continue
                        records.append(exact_record(category, owner, phrase, page, page_number, rect,
                                                    "gemma+pdf_match", score))
    finally:
        doc.close()

    # Derive exact co-author names from PDF-grounded citation prefixes when Gemma supplied a citation.
    # Prefer exact regex contacts when the same value/page was also proposed by Gemma.
    exact_contacts = {(x["category"], normalized_value(x["category"], x["text"]), x["page"])
                      for x in records if x["detection_source"] == "pdf_regex"}
    records = [x for x in records if x["detection_source"] == "pdf_regex" or
               (x["category"], normalized_value(x["category"], x["text"]), x["page"]) not in exact_contacts]
    # Split broad scholarly regions into individual exact citation/thesis entries.
    split_records = []
    for item in records:
        if item["category"] != "publication_information":
            split_records.append(item); continue
        for piece in split_scholarly_value(item["text"]):
            copy = dict(item); copy["text"] = piece; split_records.append(copy)
    records = split_records
    filtered_records = []
    for item in records:
        if item["category"] != "publication_information":
            filtered_records.append(item); continue
        compact = compact_text(item["text"])
        redundant = any(other is not item and other["category"] == "publication_information" and
                        other["page"] == item["page"] and compact in compact_text(other["text"]) and
                        len(compact) < len(compact_text(other["text"])) * .85 for other in records)
        if not redundant:
            filtered_records.append(item)
    records = filtered_records
    applicant_surnames = {person_surname(x["text"]) for x in records if x["category"] == "applicant_name"}
    derived = []
    for item in records:
        if item["category"] != "publication_information":
            continue
        for author in citation_authors(item["text"]):
            if compact_text(author.split(",", 1)[0]) in applicant_surnames:
                continue
            copy = dict(item); copy.update({"category": "coauthor_identity", "owner_role": "coauthor",
                                            "text": author, "detection_source": "pdf_citation_parser",
                                            "match_confidence": .95})
            derived.append(copy)
    records.extend(derived)
    for item in records:
        if item["category"] == "coauthor_identity" and person_surname(item["text"]) in applicant_surnames:
            item["category"], item["owner_role"] = "applicant_name", "applicant"
        elif item["category"] == "coauthor_identity" and item["owner_role"] == "reference":
            item["category"] = "reference_identity"
        elif item["category"] == "coauthor_identity" and item["owner_role"] == "advisor":
            item["category"] = "other_identifying_data"
        if item["category"] in {"email", "website"}:
            item["text"] = item["text"].strip().rstrip(".,;:)")
    # Exact-category dedupe only. Never allow a broad model region to relabel a contact value.
    unique: list[dict[str, Any]] = []
    seen = set()
    for item in records:
        key = (item["category"], normalized_value(item["category"], item["text"]), item["page"],
               tuple(round(v, 3) for v in item["bbox_normalized"]))
        if key in seen:
            continue
        seen.add(key); item["reason"] = role_reason(item); unique.append(item)
    classify_furniture(unique)
    entities = build_entities(unique)
    for entity in entities:
        entity["value_verified_against_pdf"] = True
        entity["match_confidence"] = min(
            item["match_confidence"] for item in unique
            if item["category"] == entity["category"] and
            normalized_value(item["category"], item["text"]) == normalized_value(entity["category"], entity["value"])
        )
    coverage = {}
    for category in sorted(CATEGORIES):
        count = sum(1 for entity in entities if entity["category"] == category)
        coverage[category] = {"entity_count": count, "covered": count > 0}
    review_flags = [f"no_{category}_detected" for category, result in coverage.items()
                    if not result["covered"] and category in {"applicant_name", "email", "phone_number",
                    "website", "address", "coauthor_identity", "reference_identity", "publication_information"}]
    return {"source_pdf": source.name, "source_relative_path": str(source), "redacted_pdf": None,
            "model": model, "status": "json_completed", "review_required": True,
            "page_count": page_count, "coverage": coverage, "review_flags": review_flags,
            "entities": entities, "redactions": unique, "discarded_model_detections": discarded,
            "errors": [], "verification_errors": []}


def find_cvs(root: Path) -> list[Path]:
    """Find exactly one CV directly inside each immediate application directory."""
    application_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    sources = []
    for application_dir in application_dirs:
        matches = sorted(
            p for p in application_dir.iterdir()
            if p.is_file() and CV_NAME.fullmatch(p.name)
        )
        relative = application_dir.relative_to(root)
        if not matches:
            logging.warning("Skipping application without a matching CV: %s", relative)
            continue
        if len(matches) > 1:
            logging.error("Skipping application with multiple matching CVs: %s count=%d",
                          relative, len(matches))
            continue
        sources.append(matches[0])
    return sources


def extract_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, list):
                value = {"detections": value}
            if isinstance(value, dict) and isinstance(value.get("detections"), list):
                return value
            if isinstance(value, dict) and isinstance(value.get("redactions"), list):
                return {"detections": value["redactions"]}
        except json.JSONDecodeError:
            pass
    raise ValueError("model response contained malformed JSON")


def response_text(result: Any) -> str:
    generated = result[0].get("generated_text", result[0]) if isinstance(result, list) and result else result
    if isinstance(generated, list) and generated:
        generated = generated[-1]
    if isinstance(generated, dict):
        return str(generated.get("content", generated.get("text", "")))
    return str(generated)


def load_pipeline(model_id: str) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("transformers is required; see README.md") from exc
    logging.info("Loading model %s", model_id)
    return pipeline("image-text-to-text", model=model_id, device_map="auto", dtype="auto")


def render_page(page: Any, dpi: int) -> Any:
    import pymupdf as fitz
    from PIL import Image
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def model_call(pipe: Any, image: Any, prompt: str) -> tuple[list[dict[str, Any]], str]:
    messages = [{"role": "user", "content": [
        {"type": "image", "url": image}, {"type": "text", "text": prompt}
    ]}]
    result = pipe(messages, return_full_text=False,
                  generate_kwargs={"max_new_tokens": 2048, "do_sample": False})
    raw = response_text(result)
    return extract_json(raw)["detections"], raw


def model_pass(pipe: Any, image: Any, prompt: str) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    for _ in range(2):
        try:
            detections, raw = model_call(pipe, image, prompt)
            return detections, [raw]
        except ValueError as exc:
            failures.append(str(exc))
    # Dense malformed pages are retried as vertical halves with transformed y coordinates.
    combined: list[dict[str, Any]] = []
    raws: list[str] = []
    width, height = image.size
    for tile_index, (top, bottom) in enumerate(((0, height // 2), (height // 2, height))):
        tile = image.crop((0, top, width, bottom))
        detections, raw = model_call(pipe, tile, prompt)
        raws.append(raw)
        for detection in detections:
            box = detection.get("box_2d")
            if isinstance(box, list) and len(box) == 4:
                offset = 500 * tile_index
                detection["box_2d"] = [box[0] / 2 + offset, box[1], box[2] / 2 + offset, box[3]]
            combined.append(detection)
    return combined, raws


def normalize_model_detection(item: Any, page_number: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    category = str(item.get("category", "")).strip().lower()
    owner = str(item.get("owner_role", "unknown")).strip().lower()
    box = item.get("box_2d")
    if category not in CATEGORIES or not isinstance(box, list) or len(box) != 4:
        return None
    try:
        y0, x0, y1, x1 = [max(0.0, min(1000.0, float(v))) / 1000 for v in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return {"category": category, "owner_role": owner if owner in OWNER_ROLES else "unknown",
            "model_text": str(item.get("text", "")).strip(), "page": page_number,
            "bbox_normalized": [x0, y0, x1, y1]}


def page_words(page: Any) -> list[dict[str, Any]]:
    words = []
    for x0, y0, x1, y1, text, block, line, number in page.get_text("words", sort=True):
        words.append({"rect": [x0, y0, x1, y1], "text": text, "block": block,
                      "line": line, "number": number})
    return words


def union_rect(rects: list[list[float]]) -> list[float]:
    return [min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects)]


def rect_intersects(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def snap_model_detection(page: Any, words: list[dict[str, Any]], detection: dict[str, Any],
                         margin: float) -> dict[str, Any]:
    width, height = page.rect.width, page.rect.height
    x0, y0, x1, y1 = detection["bbox_normalized"]
    approx = [x0 * width, y0 * height, x1 * width, y1 * height]
    search = [approx[0] - 4, approx[1] - 3, approx[2] + 4, approx[3] + 3]
    hits = [w for w in words if rect_intersects(w["rect"], search)]
    if hits:
        rect = union_rect([w["rect"] for w in hits])
        text = " ".join(w["text"] for w in sorted(hits, key=lambda w: (w["block"], w["line"], w["number"])))
        geometry_source = "pdf_text"
    else:
        rect, text, geometry_source = approx, detection["model_text"], "vision"
    rect = [max(0, rect[0] - margin), max(0, rect[1] - margin),
            min(width, rect[2] + margin), min(height, rect[3] + margin)]
    detection.update({"text": text or "[visual redaction]", "rect_points": rect,
                      "bbox_normalized": [rect[0] / width, rect[1] / height,
                                           rect[2] / width, rect[3] / height],
                      "detection_source": "gemma", "geometry_source": geometry_source})
    detection.pop("model_text", None)
    return detection


def line_groups(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for word in words:
        groups[(word["block"], word["line"])].append(word)
    return [sorted(group, key=lambda w: w["number"]) for _, group in sorted(groups.items())]


def deterministic_detections(page: Any, words: list[dict[str, Any]], page_number: int,
                             margin: float) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    patterns = (("email", EMAIL_RE), ("phone_number", PHONE_RE), ("website", URL_RE),
                ("publication_information", DOI_RE))
    for group in line_groups(words):
        line, spans, cursor = "", [], 0
        for index, word in enumerate(group):
            if index:
                line += " "; cursor += 1
            start = cursor; line += word["text"]; cursor += len(word["text"])
            spans.append((start, cursor, word))
        email_ranges = [(m.start(), m.end()) for m in EMAIL_RE.finditer(line)]
        for category, pattern in patterns:
            for match in pattern.finditer(line):
                if category == "website" and any(match.start() >= start and match.end() <= end
                                                  for start, end in email_ranges):
                    continue
                selected = [word for start, end, word in spans if start < match.end() and end > match.start()]
                if not selected:
                    continue
                rect = union_rect([w["rect"] for w in selected])
                rect = [max(0, rect[0] - margin), max(0, rect[1] - margin),
                        min(page.rect.width, rect[2] + margin), min(page.rect.height, rect[3] + margin)]
                found.append({"category": category, "owner_role": "unknown", "text": match.group(0),
                              "page": page_number, "rect_points": rect,
                              "bbox_normalized": [rect[0] / page.rect.width, rect[1] / page.rect.height,
                                                   rect[2] / page.rect.width, rect[3] / page.rect.height],
                              "detection_source": "pdf_regex", "geometry_source": "pdf_text"})
    return found


def iou(a: list[float], b: list[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0


def role_reason(item: dict[str, Any]) -> str:
    role = item["owner_role"].replace("_", " ").title() if item["owner_role"] != "unknown" else "Unclassified"
    label = {"phone_number": "phone or fax number", "email": "email address", "website": "website",
             "applicant_name": "name", "address": "postal address", "coauthor_identity": "co-author identity",
             "reference_identity": "reference identity", "publication_information": "publication information",
             "other_identifying_data": "identifying information"}[item["category"]]
    return f"{role} {label}"


def merge_occurrences(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in items:
        old = next((x for x in merged if x["page"] == item["page"] and
                    (x["category"] == item["category"] or {x["category"], item["category"]} <= {"email", "website"}) and
                    iou(x["bbox_normalized"], item["bbox_normalized"]) >= .35), None)
        if not old:
            item["reason"] = role_reason(item); merged.append(item); continue
        old["rect_points"] = union_rect([old["rect_points"], item["rect_points"]])
        old["bbox_normalized"] = union_rect([old["bbox_normalized"], item["bbox_normalized"]])
        sources = set(old["detection_source"].split("+")) | set(item["detection_source"].split("+"))
        old["detection_source"] = "+".join(sorted(sources))
        if old["owner_role"] == "unknown" and item["owner_role"] != "unknown":
            old["owner_role"] = item["owner_role"]
        if item["geometry_source"] == "pdf_text": old["geometry_source"] = "pdf_text"
        # Prefer exact PDF-regex values over model/line text for deterministic identifiers.
        if item["detection_source"] == "pdf_regex": old["text"] = item["text"]
        old["reason"] = role_reason(old)
    return merged


def normalized_value(category: str, value: str) -> str:
    if category == "phone_number": return re.sub(r"\D", "", value)
    if category in {"email", "website"}:
        return compact_text(value.rstrip(".,;:)"))
    if category in {"applicant_name", "coauthor_identity", "reference_identity"}:
        if category == "coauthor_identity":
            initial = re.search(r",\s*([A-Z])", value)
            return person_surname(value) + initial.group(1).lower() if initial else compact_text(value)
        return compact_text(re.sub(r"(?i)\b(?:ph\.?d\.?|dr\.?|curriculum vitae)\b", "", value))
    return re.sub(r"\s+", " ", value.strip().lower())


def build_entities(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = normalized_value(item["category"], item["text"])
        if item["category"] == "applicant_name" and item["owner_role"] == "applicant":
            key = person_surname(item["text"])
        groups[(item["category"], key)].append(item)
    entities = []
    counts: dict[str, int] = defaultdict(int)
    for (category, _), occurrences in groups.items():
        deduped = []
        for occurrence in occurrences:
            if any(old["page"] == occurrence["page"] and
                   iou(old["bbox_normalized"], occurrence["bbox_normalized"]) >= .30 for old in deduped):
                continue
            deduped.append(occurrence)
        occurrences = deduped
        counts[category] += 1
        first = occurrences[0]
        aliases = list(dict.fromkeys(o["text"] for o in occurrences))
        entities.append({"entity_id": f"{category}_{counts[category]:03d}", "category": category,
                         "value": first["text"], "owner_role": first["owner_role"],
                         "aliases": aliases, "reason": first["reason"], "occurrences": [
                             {k: o[k] for k in ("page", "bbox_normalized", "detection_source", "geometry_source")}
                             for o in occurrences]})
    return entities


def classify_furniture(items: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items: groups[(item["category"], normalized_value(item["category"], item["text"]))].append(item)
    for occurrences in groups.values():
        pages = {x["page"] for x in occurrences}
        ys = [x["bbox_normalized"][1] for x in occurrences]
        if len(pages) > 1 and max(ys) - min(ys) < .03 and (max(ys) > .92 or min(ys) < .08):
            for item in occurrences:
                item["owner_role"] = "document_furniture"; item["reason"] = role_reason(item)


def apply_redactions(source: Path, destination: Path, items: list[dict[str, Any]]) -> None:
    import pymupdf as fitz
    doc = fitz.open(source)
    try:
        for item in items: doc[item["page"] - 1].add_redact_annot(fitz.Rect(item["rect_points"]), fill=(0, 0, 0))
        for page in doc: page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        doc.save(destination, garbage=4, deflate=True)
    finally: doc.close()


def verify_pdf(path: Path, entities: list[dict[str, Any]]) -> list[str]:
    import pymupdf
    doc = pymupdf.open(path)
    try: text = "\n".join(page.get_text() for page in doc)
    finally: doc.close()
    failures = []
    for entity in entities:
        value = entity["value"]
        if value != "[visual redaction]" and len(value) >= 4 and value.lower() in text.lower():
            failures.append(f"entity remained extractable: {entity['entity_id']}")
    for label, pattern in (("email", EMAIL_RE), ("phone", PHONE_RE), ("website", URL_RE), ("doi", DOI_RE)):
        if pattern.search(text): failures.append(f"{label} pattern remained extractable")
    return sorted(set(failures))


def process_cv(source: Path, input_root: Path, output_root: Path, pipe: Any, model: str,
               dpi: int, margin: float, debug: bool, json_only: bool = False) -> dict[str, Any]:
    import pymupdf
    relative = source.relative_to(input_root)
    redacted_relative = relative.with_name(relative.stem + "_redacted.pdf")
    json_path = output_root / relative.with_suffix(".json")
    candidate = output_root / redacted_relative.with_name(redacted_relative.stem + ".candidate.pdf")
    errors, items, raw_pages = [], [], {}
    started = time.monotonic(); doc = pymupdf.open(source)
    try:
        page_count = doc.page_count
        for index, page in enumerate(doc):
            number = index + 1; words = page_words(page); image = render_page(page, dpi)
            try:
                raw_pages[str(number)] = []
                model_items: list[dict[str, Any]] = []
                for pass_name, prompt in (("contact", CONTACT_PROMPT), ("scholarly", SCHOLARLY_PROMPT)):
                    detected, raws = model_pass(pipe, image, prompt)
                    raw_pages[str(number)].append({"pass": pass_name, "responses": raws})
                    model_items.extend(x for raw in detected if (x := normalize_model_detection(raw, number)))
                aligned = [snap_model_detection(page, words, x, margin) for x in model_items]
                deterministic = deterministic_detections(page, words, number, margin)
                page_items = merge_occurrences(aligned + deterministic); items.extend(page_items)
                logging.info("Processed %s page=%d model=%d deterministic=%d merged=%d",
                             relative, number, len(aligned), len(deterministic), len(page_items))
            except Exception as exc:
                errors.append({"page": number, "error": f"{type(exc).__name__}: {exc}"})
                logging.error("Failed %s page=%d error=%s", relative, number, type(exc).__name__)
    finally: doc.close()
    if json_only:
        response_path = json_path.with_name(json_path.stem + ".model-responses.json")
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(json.dumps(raw_pages, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record = clean_json_inventory(source, response_path, model)
        record["source_relative_path"] = relative.as_posix()
        json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logging.info("Finished %s status=json_completed pages=%d entities=%d seconds=%.1f",
                     relative, page_count, len(record["entities"]), time.monotonic() - started)
        return record
    classify_furniture(items); entities = build_entities(items)
    verification_errors: list[str] = []
    if not errors:
        apply_redactions(source, candidate, items)
        verification_errors = verify_pdf(candidate, entities)
        if not verification_errors:
            final_path = output_root / redacted_relative; final_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(final_path)
        else:
            candidate.unlink(missing_ok=True)
    status = "completed" if not errors and not verification_errors else "failed_verification" if verification_errors else "failed"
    record = {"source_pdf": source.name, "source_relative_path": relative.as_posix(),
              "redacted_pdf": redacted_relative.as_posix() if status == "completed" else None,
              "model": model, "status": status, "review_required": True, "page_count": page_count,
              "entities": entities, "redactions": [{k: v for k, v in x.items() if k != "rect_points"} for x in items],
              "errors": errors, "verification_errors": verification_errors}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if debug:
        debug_path = json_path.with_name(json_path.stem + ".model-responses.json")
        debug_path.write_text(json.dumps(raw_pages, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Finished %s status=%s pages=%d entities=%d seconds=%.1f", relative, status,
                 page_count, len(entities), time.monotonic() - started)
    return record


def main() -> int:
    args = parse_args(); logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    if not input_root.is_dir(): logging.error("Input root is not a directory: %s", input_root); return 2
    if input_root == output_root or input_root in output_root.parents:
        logging.error("Output root must not be the input root or nested inside it"); return 2
    sources = find_cvs(input_root)
    if not sources:
        logging.warning("No application folders containing exactly one matching CV found under %s",
                        input_root)
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate = {}
    if args.reuse_model_responses_from:
        reuse_root = args.reuse_model_responses_from.resolve()
        for source in sources:
            relative = source.relative_to(input_root)
            response_path = reuse_root / relative.with_suffix("").with_name(relative.stem + ".model-responses.json")
            if not response_path.exists():
                aggregate[relative.as_posix()] = {"source_pdf": source.name,
                    "source_relative_path": relative.as_posix(), "status": "failed",
                    "review_required": True, "entities": [], "redactions": [],
                    "errors": [{"page": None, "error": f"Missing saved responses: {response_path}"}]}
                continue
            record = clean_json_inventory(source, response_path, args.model)
            record["source_relative_path"] = relative.as_posix()
            destination = output_root / relative.with_suffix(".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            aggregate[relative.as_posix()] = record
            logging.info("Replayed %s entities=%d discarded=%d", relative, len(record["entities"]),
                         len(record["discarded_model_detections"]))
        (output_root / "redactions.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 1 if any(x.get("status") != "json_completed" for x in aggregate.values()) else 0
    pipe = load_pipeline(args.model)
    for source in sources:
        relative = source.relative_to(input_root).as_posix()
        try:
            aggregate[relative] = process_cv(source, input_root, output_root, pipe, args.model,
                                              args.dpi, args.margin_points, args.debug_artifacts,
                                              args.json_only)
        except Exception as exc:
            logging.exception("Document failed path=%s error=%s", relative, type(exc).__name__)
            aggregate[relative] = {"source_pdf": source.name, "source_relative_path": relative,
                                   "redacted_pdf": None, "model": args.model, "status": "failed",
                                   "review_required": True, "entities": [], "redactions": [],
                                   "errors": [{"page": None, "error": f"{type(exc).__name__}: {exc}"}],
                                   "verification_errors": []}
    (output_root / "redactions.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if any(x["status"] != "completed" for x in aggregate.values()) else 0


if __name__ == "__main__": sys.exit(main())
