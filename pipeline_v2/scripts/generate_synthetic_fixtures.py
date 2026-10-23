#!/usr/bin/env python3
"""Generate fictional native-text PDFs and protected ground truth for local evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf


def pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(45, 45, 565, 795), text, fontsize=9)
    document.save(path)
    document.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    applications = args.output_root / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    people = [
        ("application_001", "Applicant Alice", "Recommender Rita"),
        ("application_002", "Applicant Ben", "Recommender Rita"),
    ]
    truth = {"applications": []}
    for app_name, applicant, recommender in people:
        root = applications / app_name
        root.mkdir()
        cv = [
            f"{applicant}\nEDUCATION\nEDUCATION|Example University|PhD|Economics|2017|2022\n"
            "EMPLOYMENT\nEMPLOYMENT|Example University|Assistant Professor|academic_faculty|2022|present\n"
            f"PUBLICATIONS\nPUBLICATION|Important Paper|journal_article|2024|Example Journal|{applicant};Coauthor Carl"
        ]
        pdf(root / f"{applicant.replace(' ', '_')}_CV.pdf", cv)
        for index in range(1, 4):
            pdf(root / f"letter_{index}.pdf", [
                f"{recommender} recommends {applicant}. Contact rita@example.edu at Example University."
            ])
        truth["applications"].append({"application": app_name, "applicant": applicant,
                                      "recommender": recommender,
                                      "education_count": 1, "employment_count": 1,
                                      "publication_count": 1})
    (applications / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n",
                                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
