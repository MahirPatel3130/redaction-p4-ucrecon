from __future__ import annotations

import re
from dataclasses import dataclass


EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?1[ .()-]*)?(?:\d[ .()-]*){10}(?!\d)")
URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s,;<>]+|\b[A-Z0-9.-]+\.(?:edu|com|org|net)(?:/[^\s,;<>]*)?")
ORCID = re.compile(r"(?i)\b(?:orcid\s*[: ]\s*)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
DOI = re.compile(r"(?i)\b(?:doi\s*:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class DetectedSpan:
    identifier_type: str
    start: int
    end: int
    text: str


def deterministic_spans(text: str) -> list[DetectedSpan]:
    rows: list[DetectedSpan] = []
    patterns = (
        ("email", EMAIL),
        ("phone_number", PHONE),
        ("website", URL),
        ("unique_identifier", ORCID),
        ("unique_identifier", DOI),
    )
    seen: set[tuple[int, int, str]] = set()
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), kind)
            if key not in seen:
                rows.append(DetectedSpan(kind, match.start(), match.end(), match.group()))
                seen.add(key)
    return sorted(rows, key=lambda row: (row.start, row.end, row.identifier_type))


CV_HEADINGS = {
    "education": re.compile(r"(?im)^\s*(?:education|academic background|degrees?)\s*:?[ \t]*$"),
    "employment": re.compile(
        r"(?im)^\s*(?:employment|appointments?|professional experience|academic positions?)\s*:?[ \t]*$"
    ),
    "publications": re.compile(
        r"(?im)^\s*(?:publications?|selected publications?|articles?|research papers?)\s*:?[ \t]*$"
    ),
    "research": re.compile(r"(?im)^\s*(?:research|research interests?|methods?)\s*:?[ \t]*$"),
}


def section_inventory(text: str) -> dict[str, dict[str, int | bool]]:
    matches: list[tuple[int, str]] = []
    for name, pattern in CV_HEADINGS.items():
        matches.extend((match.start(), name) for match in pattern.finditer(text))
    matches.sort()
    result: dict[str, dict[str, int | bool]] = {
        name: {"heading_found": False, "candidate_entry_count": 0}
        for name in CV_HEADINGS
    }
    for index, (start, name) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        body = text[start:end]
        lines = [line.strip() for line in body.splitlines()[1:] if line.strip()]
        candidates = sum(
            bool(re.search(r"\b(?:19|20)\d{2}\b", line))
            or bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line))
            for line in lines
        )
        result[name] = {"heading_found": True, "candidate_entry_count": int(candidates)}
    return result

