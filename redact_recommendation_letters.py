#!/usr/bin/env python3
"""Offline, human-review-required direct-identifier redaction for recommendation letters."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
MODEL_DEFAULT = "google/gemma-4-31B-it"
EXPECTED_LETTERS = 3
CV_NAME = re.compile(r"(?:cv|curriculum[_ -]?vitae)", re.IGNORECASE)
IDENTIFIER_TYPES = {
    "person_name", "email", "phone_number", "website", "postal_address",
    "unique_identifier", "signature",
}
PERSON_ROLES = {"applicant", "recommender", "recipient", "third_party", "unknown"}
CONTACT_TYPES = {"email", "phone_number", "website", "postal_address", "unique_identifier"}

DIRECT_IDENTIFIER_PROMPT = """Find only direct personal identifiers visible on this recommendation-letter page.
Return JSON only in this exact shape:
{"detections":[{"box_2d":[y_min,x_min,y_max,x_max],"identifier_type":"person_name","person_role":"applicant","text":"exact visible text"}]}.
Coordinates must use a 0..1000 grid in y,x,y,x order. Valid identifier_type values:
person_name, email, phone_number, website, postal_address, unique_identifier, signature.
Valid person_role values: applicant, recommender, recipient, third_party, unknown.
Detect applicant, letter-writer, recipient, and other individual names; email addresses; phone/fax
numbers; postal addresses; personal or professional URLs; ORCID, employee/student/researcher IDs
and usernames; and handwritten or digital signatures. For a signature use text "[signature]".
Do not detect standalone universities, companies, departments, job titles, dates, publications,
logos, letterhead branding, or evaluation prose. Use tight boxes around only the identifier.
Do not return markdown, explanations, or any keys outside the JSON object."""

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]*)?\(?\d{3}\)?[ .-]*\d{3}[ .-]*\d{4}(?!\d)")
URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s,;]+|\b[A-Z0-9.-]+\.(?:edu|com|org|net)(?:/[^\s,;]*)?"
)
ORCID_RE = re.compile(r"(?i)\b(?:https?://orcid\.org/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
LABELED_ID_RE = re.compile(
    r"(?i)\b(?:employee|student|faculty|researcher|user(?:name)?|applicant)\s*ID\s*[:#]?\s*[A-Z0-9][A-Z0-9._-]{3,}\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--margin-points", type=float, default=1.5)
    parser.add_argument("--write-redacted-pdfs", action="store_true")
    parser.add_argument("--debug-artifacts", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi must be between 72 and 600")
    if not 0 <= args.margin_points <= 12:
        parser.error("--margin-points must be between 0 and 12")
    return args


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def application_id(relative_path: Path) -> str:
    return stable_id("app", relative_path.as_posix())


def document_id(relative_path: Path) -> str:
    return stable_id("doc", relative_path.as_posix())


def is_pdf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".pdf"


def is_cv(path: Path) -> bool:
    return is_pdf(path) and bool(CV_NAME.search(path.stem))


def discover_applications(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.casefold(),
    )
    for directory in directories:
        relative = directory.relative_to(root)
        app_id = application_id(relative)
        letters = sorted(
            (path for path in directory.iterdir() if is_pdf(path) and not is_cv(path)),
            key=lambda path: path.name.casefold(),
        )
        record = {
            "application_id": app_id,
            "source_application_path": relative.as_posix(),
            "expected_letter_count": EXPECTED_LETTERS,
            "discovered_letter_count": len(letters),
            "status": "pending" if len(letters) == EXPECTED_LETTERS else "invalid_letter_count",
            "documents": [],
            "errors": [],
        }
        if len(letters) != EXPECTED_LETTERS:
            record["errors"].append({
                "error": "invalid_letter_count",
                "expected": EXPECTED_LETTERS,
                "discovered": len(letters),
            })
            logging.error("Skipping application path=%s expected_letters=%d discovered_letters=%d",
                          relative, EXPECTED_LETTERS, len(letters))
        else:
            valid.append({"path": directory, "relative": relative,
                          "application_id": app_id, "letters": letters})
        records[app_id] = record
    return valid, records


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
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            value = {"detections": value}
        if isinstance(value, dict) and isinstance(value.get("detections"), list):
            return value
        if isinstance(value, dict) and isinstance(value.get("redactions"), list):
            return {"detections": value["redactions"]}
    raise ValueError("model output did not contain a valid detections array")


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
    logging.info("Loading model=%s", model_id)
    return pipeline("image-text-to-text", model=model_id, device_map="auto", dtype="auto")


def render_page(page: Any, dpi: int) -> Any:
    import pymupdf
    from PIL import Image

    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def model_call(pipe: Any, image: Any) -> tuple[list[dict[str, Any]], str]:
    messages = [{"role": "user", "content": [
        {"type": "image", "url": image},
        {"type": "text", "text": DIRECT_IDENTIFIER_PROMPT},
    ]}]
    result = pipe(messages, return_full_text=False,
                  generate_kwargs={"max_new_tokens": 2048, "do_sample": False})
    raw = response_text(result)
    return extract_json(raw)["detections"], raw


def model_pass(pipe: Any, image: Any) -> tuple[list[dict[str, Any]], list[str]]:
    for _ in range(2):
        try:
            detections, raw = model_call(pipe, image)
            return detections, [raw]
        except ValueError:
            pass
    combined: list[dict[str, Any]] = []
    raw_responses: list[str] = []
    width, height = image.size
    for tile_index, (top, bottom) in enumerate(((0, height // 2), (height // 2, height))):
        detections, raw = model_call(pipe, image.crop((0, top, width, bottom)))
        raw_responses.append(raw)
        for detection in detections:
            box = detection.get("box_2d")
            if isinstance(box, list) and len(box) == 4:
                detection["box_2d"] = [box[0] / 2 + 500 * tile_index,
                                       box[1], box[2] / 2 + 500 * tile_index, box[3]]
            combined.append(detection)
    return combined, raw_responses


def page_words(page: Any) -> list[dict[str, Any]]:
    return [
        {"rect": [x0, y0, x1, y1], "text": text, "block": block,
         "line": line, "number": number}
        for x0, y0, x1, y1, text, block, line, number
        in page.get_text("words", sort=True)
    ]


def line_groups(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for word in words:
        groups[(word["block"], word["line"])].append(word)
    return [sorted(group, key=lambda word: word["number"])
            for _, group in sorted(groups.items())]


def union_rect(rectangles: list[list[float]]) -> list[float]:
    return [min(rect[0] for rect in rectangles), min(rect[1] for rect in rectangles),
            max(rect[2] for rect in rectangles), max(rect[3] for rect in rectangles)]


def rect_intersects(first: list[float], second: list[float]) -> bool:
    return (min(first[2], second[2]) > max(first[0], second[0]) and
            min(first[3], second[3]) > max(first[1], second[1]))


def iou(first: list[float], second: list[float]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def normalized_value(identifier_type: str, value: str) -> str:
    if identifier_type == "phone_number":
        return re.sub(r"\D", "", value)
    if identifier_type in {"email", "website", "unique_identifier"}:
        return compact_text(value.rstrip(".,;:)"))
    if identifier_type == "person_name":
        value = re.sub(r"(?i)\b(?:dr|professor|prof|mr|mrs|ms)\.?\s+", "", value)
    return re.sub(r"\s+", " ", value.strip().casefold())


def native_box(item: dict[str, Any], page: Any) -> list[float] | None:
    box = item.get("box_2d")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        y0, x0, y1, x1 = [max(0.0, min(1000.0, float(value))) / 1000 for value in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0 * page.rect.width, y0 * page.rect.height,
            x1 * page.rect.width, y1 * page.rect.height]


def normalize_model_detection(item: Any, page_number: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    identifier_type = str(item.get("identifier_type", item.get("category", ""))).strip().lower()
    identifier_type = {"name": "person_name", "address": "postal_address",
                       "phone": "phone_number", "id": "unique_identifier"}.get(
                           identifier_type, identifier_type)
    role = str(item.get("person_role", item.get("owner_role", "unknown"))).strip().lower()
    role = {"writer": "recommender", "author": "recommender",
            "candidate": "applicant", "subject": "applicant"}.get(role, role)
    if identifier_type not in IDENTIFIER_TYPES:
        return None
    if role not in PERSON_ROLES:
        role = "unknown"
    return {"identifier_type": identifier_type, "person_role": role,
            "model_text": str(item.get("text", "")).strip(), "page": page_number,
            "box_2d": item.get("box_2d")}


def best_pdf_phrase(model_text: str, words: list[dict[str, Any]], box: list[float],
                    max_words: int) -> tuple[str, list[float] | None, float]:
    if not model_text or not words:
        return "", None, 0.0
    nearby = [group for group in line_groups(words)
              if rect_intersects(union_rect([word["rect"] for word in group]),
                                 [box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5])]
    candidates = nearby or line_groups(words)
    target = compact_text(model_text)
    target_words = max(1, len(model_text.split()))
    best: tuple[str, list[float] | None, float] = ("", None, 0.0)
    for group in candidates:
        low = max(1, target_words - 3)
        high = min(len(group), max_words, target_words + 4)
        for size in range(low, high + 1):
            for start in range(len(group) - size + 1):
                selected = group[start:start + size]
                text = " ".join(word["text"] for word in selected)
                score = difflib.SequenceMatcher(None, target, compact_text(text)).ratio()
                if score > best[2]:
                    best = (text, union_rect([word["rect"] for word in selected]), score)
    return best


def looks_like_person_name(value: str) -> bool:
    cleaned = re.sub(r"(?i)^(?:dear|dr|professor|prof|mr|mrs|ms)\.?\s+", "", value).strip(" ,:")
    tokens = cleaned.split()
    lowered = cleaned.casefold()
    blocked = ("university", "college", "department", "school", "committee", "hospital",
               "institute", "laboratory", "corporation", "company", "foundation")
    if not 2 <= len(tokens) <= 7 or any(word in lowered for word in blocked):
        return False
    capitalized = sum(bool(re.fullmatch(r"[A-Z][A-Za-z'’-]*[.,]?", token)) for token in tokens)
    initials = sum(bool(re.fullmatch(r"[A-Z]\.?", token)) for token in tokens)
    return capitalized + initials >= 2


def looks_like_address(value: str) -> bool:
    return bool(re.search(
        r"(?i)\b(?:\d{1,6}\s+\S+|P\.?O\.?\s+Box\s+\d+).*(?:street|st\.?|road|rd\.?|"
        r"avenue|ave\.?|boulevard|blvd\.?|drive|dr\.?|lane|ln\.?|way|court|ct\.?)\b|"
        r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", value))


def exact_pattern_value(identifier_type: str, value: str) -> str | None:
    pattern = {"email": EMAIL_RE, "phone_number": PHONE_RE, "website": URL_RE,
               "unique_identifier": ORCID_RE}.get(identifier_type)
    if pattern is None:
        return value
    match = pattern.search(value)
    return match.group(0).rstrip(".,;:)") if match else None


def expand_rect(rect: list[float], page: Any, margin: float) -> list[float]:
    return [max(0, rect[0] - margin), max(0, rect[1] - margin),
            min(page.rect.width, rect[2] + margin), min(page.rect.height, rect[3] + margin)]


def ground_model_detection(page: Any, words: list[dict[str, Any]], detection: dict[str, Any],
                           margin: float) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    box = native_box(detection, page)
    if box is None:
        return None, {"page": detection["page"], "identifier_type": detection["identifier_type"],
                      "reason": "invalid_bounding_box"}
    page_area = page.rect.width * page.rect.height
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area / page_area > 0.25:
        return None, {"page": detection["page"], "identifier_type": detection["identifier_type"],
                      "reason": "bounding_box_too_large"}
    identifier_type = detection["identifier_type"]
    role = detection["person_role"]
    if identifier_type == "signature":
        record = {"identifier_type": identifier_type,
                  "person_role": role if role != "unknown" else "recommender",
                  "text": "[signature]", "page": detection["page"],
                  "rect_points": expand_rect(box, page, margin),
                  "detection_source": "gemma", "geometry_source": "vision",
                  "value_verified_against_pdf": False, "match_confidence": 0.75}
        return record, None
    if not detection["model_text"]:
        return None, {"page": detection["page"], "identifier_type": identifier_type,
                      "reason": "missing_model_text"}
    if not words:
        record = {"identifier_type": identifier_type, "person_role": role,
                  "text": detection["model_text"], "page": detection["page"],
                  "rect_points": expand_rect(box, page, margin),
                  "detection_source": "gemma", "geometry_source": "vision",
                  "value_verified_against_pdf": False, "match_confidence": 0.6}
        return record, None
    max_words = 14 if identifier_type == "postal_address" else 8
    phrase, rect, score = best_pdf_phrase(detection["model_text"], words, box, max_words)
    threshold = 0.68 if identifier_type == "postal_address" else 0.76
    if rect is None or score < threshold:
        return None, {"page": detection["page"], "identifier_type": identifier_type,
                      "reason": "model_value_not_grounded", "best_score": round(score, 3)}
    value = exact_pattern_value(identifier_type, phrase)
    if value is None:
        return None, {"page": detection["page"], "identifier_type": identifier_type,
                      "reason": "identifier_shape_not_verified"}
    if identifier_type == "person_name" and not looks_like_person_name(value):
        return None, {"page": detection["page"], "identifier_type": identifier_type,
                      "reason": "failed_person_name_shape"}
    if identifier_type == "postal_address" and not looks_like_address(value):
        return None, {"page": detection["page"], "identifier_type": identifier_type,
                      "reason": "failed_address_shape"}
    record = {"identifier_type": identifier_type, "person_role": role, "text": value,
              "page": detection["page"], "rect_points": expand_rect(rect, page, margin),
              "detection_source": "gemma+pdf_match", "geometry_source": "pdf_text",
              "value_verified_against_pdf": True, "match_confidence": round(score, 3)}
    return record, None


def deterministic_detections(page: Any, words: list[dict[str, Any]], page_number: int,
                             margin: float) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    patterns = (("email", EMAIL_RE), ("phone_number", PHONE_RE),
                ("unique_identifier", ORCID_RE), ("unique_identifier", LABELED_ID_RE),
                ("website", URL_RE))
    for group in line_groups(words):
        line, spans, cursor = "", [], 0
        for index, word in enumerate(group):
            if index:
                line += " "
                cursor += 1
            start = cursor
            line += word["text"]
            cursor += len(word["text"])
            spans.append((start, cursor, word))
        email_ranges = [(match.start(), match.end()) for match in EMAIL_RE.finditer(line)]
        orcid_ranges = [(match.start(), match.end()) for match in ORCID_RE.finditer(line)]
        for identifier_type, pattern in patterns:
            for match in pattern.finditer(line):
                if identifier_type == "website" and any(
                    match.start() >= start and match.end() <= end
                    for start, end in email_ranges + orcid_ranges
                ):
                    continue
                selected = [word for start, end, word in spans
                            if start < match.end() and end > match.start()]
                if not selected:
                    continue
                rect = expand_rect(union_rect([word["rect"] for word in selected]), page, margin)
                found.append({"identifier_type": identifier_type, "person_role": "unknown",
                              "text": match.group(0).rstrip(".,;:)"), "page": page_number,
                              "rect_points": rect, "detection_source": "pdf_regex",
                              "geometry_source": "pdf_text",
                              "value_verified_against_pdf": True, "match_confidence": 1.0})
    return found


def reason_for(item: dict[str, Any]) -> str:
    role = item["person_role"].replace("_", " ").title()
    label = {"person_name": "name", "email": "email address",
             "phone_number": "phone or fax number", "website": "website",
             "postal_address": "postal address", "unique_identifier": "unique identifier",
             "signature": "signature"}[item["identifier_type"]]
    return f"{role} {label}"


def merge_occurrences(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in items:
        old = next((candidate for candidate in merged
                    if candidate["page"] == item["page"]
                    and candidate["identifier_type"] == item["identifier_type"]
                    and iou(candidate["rect_points"], item["rect_points"]) >= 0.35), None)
        if old is None:
            item["reason"] = reason_for(item)
            merged.append(item)
            continue
        old["rect_points"] = union_rect([old["rect_points"], item["rect_points"]])
        sources = set(old["detection_source"].split("+")) | set(item["detection_source"].split("+"))
        old["detection_source"] = "+".join(sorted(sources))
        if old["person_role"] == "unknown" and item["person_role"] != "unknown":
            old["person_role"] = item["person_role"]
        if item["geometry_source"] == "pdf_text":
            old["geometry_source"] = "pdf_text"
        if item["detection_source"] == "pdf_regex":
            old["text"] = item["text"]
        old["value_verified_against_pdf"] = (
            old["value_verified_against_pdf"] or item["value_verified_against_pdf"])
        old["match_confidence"] = max(old["match_confidence"], item["match_confidence"])
        old["reason"] = reason_for(old)
    unique: list[dict[str, Any]] = []
    seen = set()
    for item in merged:
        key = (item["identifier_type"], item["person_role"],
               normalized_value(item["identifier_type"], item["text"]), item["page"],
               tuple(round(value, 2) for value in item["rect_points"]))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def local_person_id(app_id: str, doc_id: str, role: str, normalized: str) -> str:
    if role == "applicant":
        return f"{app_id}:applicant"
    if role == "recommender":
        return f"{doc_id}:recommender"
    if role == "recipient":
        return f"{doc_id}:recipient"
    return f"{doc_id}:{role}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:10]}"


def build_entities(items: list[dict[str, Any]], app_id: str,
                   doc_id: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        normalized = normalized_value(item["identifier_type"], item["text"])
        groups[(item["identifier_type"], item["person_role"], normalized)].append(item)
    entities: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)
    for (identifier_type, role, normalized), occurrences in sorted(groups.items()):
        counters[identifier_type] += 1
        entity_id = f"{identifier_type}_{counters[identifier_type]:03d}"
        local_id = local_person_id(app_id, doc_id, role, normalized)
        aliases = list(dict.fromkeys(item["text"] for item in occurrences))
        occurrence_records = []
        for item in sorted(occurrences, key=lambda value: (value["page"], value["rect_points"])):
            width, height = item.pop("page_width"), item.pop("page_height")
            rect = item["rect_points"]
            item["bbox_normalized"] = [rect[0] / width, rect[1] / height,
                                       rect[2] / width, rect[3] / height]
            mention_key = f"{doc_id}|{entity_id}|{item['page']}|" + ",".join(
                f"{value:.5f}" for value in item["bbox_normalized"])
            mention_id = stable_id("men", mention_key)
            item.update({"entity_id": entity_id, "mention_id": mention_id,
                         "local_person_id": local_id, "person_id": None})
            occurrence_records.append({
                "mention_id": mention_id,
                "page": item["page"],
                "bbox_normalized": item["bbox_normalized"],
                "detection_source": item["detection_source"],
                "geometry_source": item["geometry_source"],
                "value_verified_against_pdf": item["value_verified_against_pdf"],
                "match_confidence": item["match_confidence"],
            })
        entities.append({
            "entity_id": entity_id,
            "identifier_type": identifier_type,
            "person_role": role,
            "local_person_id": local_id,
            "person_id": None,
            "value": occurrences[0]["text"],
            "aliases": aliases,
            "reason": occurrences[0]["reason"],
            "value_verified_against_pdf": all(
                item["value_verified_against_pdf"] for item in occurrences),
            "match_confidence": min(item["match_confidence"] for item in occurrences),
            "occurrences": occurrence_records,
        })
    return entities


def build_parties(entities: list[dict[str, Any]], app_id: str,
                  doc_id: str) -> list[dict[str, Any]]:
    base = {
        f"{app_id}:applicant": {"local_person_id": f"{app_id}:applicant",
                                "person_id": None, "role": "applicant",
                                "entity_ids": [], "identifier_types_observed": []},
        f"{doc_id}:recommender": {"local_person_id": f"{doc_id}:recommender",
                                  "person_id": None, "role": "recommender",
                                  "entity_ids": [], "identifier_types_observed": []},
    }
    for entity in entities:
        local_id = entity["local_person_id"]
        party = base.setdefault(local_id, {"local_person_id": local_id, "person_id": None,
                                           "role": entity["person_role"], "entity_ids": [],
                                           "identifier_types_observed": []})
        party["entity_ids"].append(entity["entity_id"])
        party["identifier_types_observed"].append(entity["identifier_type"])
    for party in base.values():
        party["entity_ids"] = sorted(set(party["entity_ids"]))
        party["identifier_types_observed"] = sorted(set(party["identifier_types_observed"]))
    return sorted(base.values(), key=lambda party: party["local_person_id"])


def build_relationship(doc_id: str, app_id: str) -> list[dict[str, Any]]:
    relationship_type = "wrote_recommendation_for"
    return [{
        "relationship_id": stable_id("rel", f"{doc_id}|{relationship_type}"),
        "subject_local_person_id": f"{doc_id}:recommender",
        "subject_person_id": None,
        "relationship_type": relationship_type,
        "object_local_person_id": f"{app_id}:applicant",
        "object_person_id": None,
        "document_id": doc_id,
        "resolution_method": "recommendation_letter_structure",
        "confidence": 1.0,
    }]


def coverage_and_flags(entities: list[dict[str, Any]], items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    coverage = {}
    for identifier_type in sorted(IDENTIFIER_TYPES):
        count = sum(entity["identifier_type"] == identifier_type for entity in entities)
        coverage[identifier_type] = {"entity_count": count, "covered": count > 0}
    flags = []
    if not any(entity["identifier_type"] == "person_name" and
               entity["person_role"] == "applicant" for entity in entities):
        flags.append("no_applicant_name_detected")
    if not any(entity["identifier_type"] == "person_name" and
               entity["person_role"] == "recommender" for entity in entities):
        flags.append("no_recommender_name_detected")
    if not any(entity["identifier_type"] in CONTACT_TYPES and
               entity["person_role"] == "recommender" for entity in entities):
        flags.append("no_recommender_contact_detected")
    if any(not item["value_verified_against_pdf"] for item in items):
        flags.append("vision_only_identifier_requires_review")
    if any(item["identifier_type"] == "signature" for item in items):
        flags.append("signature_requires_visual_review")
    return coverage, flags


def apply_redactions(source: Path, destination: Path,
                     items: list[dict[str, Any]]) -> None:
    import pymupdf

    document = pymupdf.open(source)
    try:
        for item in items:
            document[item["page"] - 1].add_redact_annot(
                pymupdf.Rect(item["rect_points"]), fill=(0, 0, 0))
        for page in document:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        document.save(destination, garbage=4, deflate=True)
    finally:
        document.close()


def verify_pdf(path: Path, entities: list[dict[str, Any]]) -> list[str]:
    import pymupdf

    document = pymupdf.open(path)
    try:
        text = "\n".join(page.get_text() for page in document)
    finally:
        document.close()
    failures = []
    for entity in entities:
        value = entity["value"]
        if value != "[signature]" and len(value) >= 4 and value.casefold() in text.casefold():
            failures.append(f"entity remained extractable: {entity['entity_id']}")
    for label, pattern in (("email", EMAIL_RE), ("phone", PHONE_RE), ("website", URL_RE),
                           ("ORCID", ORCID_RE), ("labeled ID", LABELED_ID_RE)):
        if pattern.search(text):
            failures.append(f"{label} pattern remained extractable")
    return sorted(set(failures))


def serializable_redactions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items()
             if key not in {"rect_points"}} for item in items]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_letter(source: Path, input_root: Path, output_root: Path, pipe: Any,
                   model: str, dpi: int, margin: float, write_pdf: bool,
                   debug: bool, app_id: str, letter_index: int) -> dict[str, Any]:
    import pymupdf

    relative = source.relative_to(input_root)
    doc_id = document_id(relative)
    json_path = output_root / relative.with_suffix(".json")
    redacted_relative = relative.with_name(relative.stem + "_redacted.pdf")
    candidate = output_root / redacted_relative.with_name(redacted_relative.stem + ".candidate.pdf")
    errors: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    raw_pages: dict[str, list[str]] = {}
    items: list[dict[str, Any]] = []
    started = time.monotonic()
    document = pymupdf.open(source)
    try:
        page_count = document.page_count
        for index, page in enumerate(document):
            page_number = index + 1
            words = page_words(page)
            model_items: list[dict[str, Any]] = []
            try:
                detections, raw = model_pass(pipe, render_page(page, dpi))
                raw_pages[str(page_number)] = raw
                for detection in detections:
                    normalized = normalize_model_detection(detection, page_number)
                    if normalized is None:
                        discarded.append({"page": page_number,
                                          "reason": "invalid_model_detection"})
                        continue
                    grounded, rejection = ground_model_detection(page, words, normalized, margin)
                    if grounded is not None:
                        model_items.append(grounded)
                    elif rejection is not None:
                        discarded.append(rejection)
            except Exception as exc:
                errors.append({"page": page_number,
                               "error": f"{type(exc).__name__}: {exc}"})
                logging.error("Page failed path=%s page=%d error=%s",
                              relative, page_number, type(exc).__name__)
            page_items = merge_occurrences(
                model_items + deterministic_detections(page, words, page_number, margin))
            for item in page_items:
                item["page_width"] = page.rect.width
                item["page_height"] = page.rect.height
            items.extend(page_items)
            logging.info("Processed path=%s page=%d model=%d accepted=%d discarded=%d",
                         relative, page_number, len(model_items), len(page_items),
                         sum(value.get("page") == page_number for value in discarded))
    finally:
        document.close()

    entities = build_entities(items, app_id, doc_id)
    parties = build_parties(entities, app_id, doc_id)
    relationships = build_relationship(doc_id, app_id)
    coverage, review_flags = coverage_and_flags(entities, items)
    verification_errors: list[str] = []
    redacted_pdf: str | None = None
    if write_pdf and not errors:
        apply_redactions(source, candidate, items)
        verification_errors = verify_pdf(candidate, entities)
        if verification_errors:
            candidate.unlink(missing_ok=True)
        else:
            final_path = output_root / redacted_relative
            final_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(final_path)
            redacted_pdf = redacted_relative.as_posix()
    if errors:
        status = "failed"
    elif verification_errors:
        status = "failed_verification"
    elif write_pdf:
        status = "completed"
    else:
        status = "json_completed"
    record = {
        "schema_version": SCHEMA_VERSION,
        "application_id": app_id,
        "document_id": doc_id,
        "letter_index": letter_index,
        "document_type": "recommendation_letter",
        "source_pdf": source.name,
        "source_relative_path": relative.as_posix(),
        "redacted_pdf": redacted_pdf,
        "model": model,
        "status": status,
        "review_required": True,
        "page_count": page_count,
        "parties": parties,
        "entities": entities,
        "relationships": relationships,
        "redactions": serializable_redactions(items),
        "coverage": coverage,
        "review_flags": review_flags,
        "discarded_model_detections": discarded,
        "errors": errors,
        "verification_errors": verification_errors,
    }
    write_json(json_path, record)
    if debug:
        write_json(json_path.with_name(json_path.stem + ".model-responses.json"), raw_pages)
    logging.info("Finished path=%s status=%s pages=%d entities=%d seconds=%.1f",
                 relative, status, page_count, len(entities), time.monotonic() - started)
    return record


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        logging.error("Input root is not a directory path=%s", input_root)
        return 2
    if input_root == output_root or input_root in output_root.parents:
        logging.error("Output root must not be the input root or nested inside it")
        return 2
    valid, application_records = discover_applications(input_root)
    aggregate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "applications": application_records,
        "documents": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_root / "recommendation_redactions.json"
    if not valid:
        write_json(aggregate_path, aggregate)
        logging.error("No application contained exactly %d recommendation-letter PDFs",
                      EXPECTED_LETTERS)
        return 1
    pipe = load_pipeline(args.model)
    for application in valid:
        app_id = application["application_id"]
        app_record = application_records[app_id]
        for index, source in enumerate(application["letters"], start=1):
            relative = source.relative_to(input_root)
            doc_id = document_id(relative)
            app_record["documents"].append(doc_id)
            try:
                record = process_letter(
                    source, input_root, output_root, pipe, args.model, args.dpi,
                    args.margin_points, args.write_redacted_pdfs, args.debug_artifacts,
                    app_id, index,
                )
            except Exception as exc:
                logging.exception("Document failed path=%s error=%s",
                                  relative, type(exc).__name__)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "application_id": app_id,
                    "document_id": doc_id,
                    "letter_index": index,
                    "document_type": "recommendation_letter",
                    "source_pdf": source.name,
                    "source_relative_path": relative.as_posix(),
                    "redacted_pdf": None,
                    "model": args.model,
                    "status": "failed",
                    "review_required": True,
                    "page_count": None,
                    "parties": [], "entities": [], "relationships": [],
                    "redactions": [], "coverage": {},
                    "review_flags": ["document_processing_failed"],
                    "discarded_model_detections": [],
                    "errors": [{"page": None,
                                "error": f"{type(exc).__name__}: {exc}"}],
                    "verification_errors": [],
                }
                write_json(output_root / relative.with_suffix(".json"), record)
            aggregate["documents"][relative.as_posix()] = record
            write_json(aggregate_path, aggregate)
        successful = {"completed"} if args.write_redacted_pdfs else {"json_completed"}
        document_records = [aggregate["documents"][path]
                            for path in aggregate["documents"]
                            if aggregate["documents"][path]["application_id"] == app_id]
        app_record["status"] = "completed" if all(
            record["status"] in successful for record in document_records) else "failed"
        write_json(aggregate_path, aggregate)
    return 1 if any(record["status"] != "completed"
                    for record in application_records.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
