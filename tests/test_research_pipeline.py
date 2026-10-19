from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pymupdf

from research_pipeline.common import five_year_bucket, replace_aliases, timing_statistics
from research_pipeline.model import GemmaModel, MockModel, extract_json
from research_pipeline.pipeline import deduplicate_entities, process_document, validate_section


ROOT = Path(__file__).resolve().parents[1]


def make_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=10)
    document.save(path)
    document.close()


class CommonTests(unittest.TestCase):
    def test_bucket_and_overlap_replacement(self) -> None:
        self.assertEqual(five_year_bucket("2022"), "2020-2024")
        text, applied = replace_aliases("Applicant Alice Alice", [
            ("Applicant Alice", "[[APPLICANT:person_1]]"),
            ("Alice", "[[PERSON:person_1]]"),
        ])
        self.assertNotIn("Alice", text)
        self.assertTrue(applied)

    def test_timing_statistics(self) -> None:
        result = timing_statistics([1.0, 2.0, 3.0])
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["total"], 6.0)

    def test_model_json_and_explicit_failure(self) -> None:
        self.assertEqual(extract_json("```json\n{\"ok\": true}\n```"), {"ok": True})
        section = validate_section(None, "education")
        self.assertEqual(section["status"], "model_failed")
        self.assertIsNone(section["model_confidence"])

    def test_malformed_model_json_retries(self) -> None:
        replies = iter([[{"generated_text": "not json"}],
                        [{"generated_text": '{"ok": true}'}]])
        model = GemmaModel.__new__(GemmaModel)
        model.pipe = lambda *args, **kwargs: next(replies)
        model.calls = model.retries = 0
        model.inference_seconds = 0.0
        model.raw_responses = []
        self.assertEqual(model._call("test"), {"ok": True})
        self.assertEqual(model.retries, 1)

    def test_model_and_regex_duplicate_prefers_model_role(self) -> None:
        rows = deduplicate_entities([
            {"page": 1, "identifier_type": "email", "person_role": "unknown",
             "text": "a@example.edu", "detection_source": "deterministic_regex"},
            {"page": 1, "identifier_type": "email", "person_role": "applicant",
             "text": "a@example.edu", "detection_source": "gemma"},
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["person_role"], "applicant")


class EndToEndMockTests(unittest.TestCase):
    def test_scanned_page_uses_confident_transcription(self) -> None:
        class ScannedModel(MockModel):
            def transcribe(self, image):
                self.calls += 1
                return {"text": "Recommender Rita recommends Applicant Alice.",
                        "status": "found", "model_confidence": 0.91}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "letter.pdf"
            document = pymupdf.open(); document.new_page(); document.save(pdf); document.close()
            task = {"application_id": "app_test", "document_id": "doc_test",
                    "document_type": "recommendation_letter", "letter_index": 1,
                    "source_path": str(pdf), "source_relative_path": "app/letter.pdf",
                    "source_sha256": "fixture"}
            result = process_document(task, {"config_hash": "test", "dpi": 72,
                                             "worker_id": 0, "gpu_id": "0"},
                                      ScannedModel(), root / "checkpoint.json", None)
            self.assertEqual(result["timing"]["scanned_page_count"], 1)
            self.assertEqual(result["page_transcription"][0]["transcription_model_confidence"],
                             0.91)

    def test_full_package_resume_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "applications"
            app = inputs / "application_001"
            app.mkdir(parents=True)
            cv_text = "\n".join([
                "Applicant Alice alice@example.edu 555-111-2222",
                "EDUCATION|Example University|PhD|Economics|2017|2022",
                "EMPLOYMENT|Example University|Assistant Professor|academic_faculty|2022|present",
                "PUBLICATION|Important Paper|journal_article|2024|Example Journal|Applicant Alice;Coauthor Carl",
            ])
            make_pdf(app / "Applicant_CV.pdf", cv_text)
            for index in range(1, 4):
                make_pdf(app / f"letter_{index}.pdf",
                         "Recommender Rita recommends Applicant Alice. "
                         "Contact rita@example.edu. Example University supports this letter.")
            checksums = {path.name: path.read_bytes() for path in app.iterdir()}
            output, state = root / "run", root / "state"
            registry = state / "identity_registry.json"
            command = [sys.executable, str(ROOT / "build_research_dataset.py"),
                       "--input-root", str(inputs), "--output-root", str(output),
                       "--identity-registry", str(registry), "--state-root", str(state),
                       "--model", "mock", "--devices", "0", "--workers", "1",
                       "--require-review", "--log-level", "DEBUG"]
            environment = {**os.environ, "RESEARCH_PIPELINE_ALLOW_MOCK": "1"}
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            candidate = output / "researcher_candidate"
            dataset = json.loads((candidate / "dataset.json").read_text())
            self.assertEqual(dataset["release_status"], "pending_review")
            profiles = [json.loads(line) for line in (candidate / "cv_profiles.jsonl").read_text().splitlines()]
            self.assertEqual(profiles[0]["education"]["model_confidence"], 0.93)
            self.assertEqual(profiles[0]["employment"]["items"][0]["start_period"]["value"], "2020-2024")
            researcher_text = "\n".join(path.read_text(errors="ignore")
                                         for path in candidate.iterdir() if path.is_file())
            for raw in ("Applicant Alice", "Recommender Rita", "Example University",
                        "alice@example.edu", "rita@example.edu", "Important Paper"):
                self.assertNotIn(raw, researcher_text)
            protected_text = (output / "protected/redactions.jsonl").read_text()
            self.assertIn("Applicant Alice", protected_text)
            redaction_rows = [json.loads(line) for line in protected_text.splitlines()]
            self.assertEqual(len({row["redaction_id"] for row in redaction_rows}),
                             len(redaction_rows))
            self.assertTrue((output / "metrics/document_timings.csv").exists())
            self.assertEqual((output / "protected").stat().st_mode & 0o777, 0o700)
            self.assertEqual((output / "protected/redactions.jsonl").stat().st_mode & 0o777,
                             0o600)
            manifest = json.loads((candidate / "manifest.json").read_text())
            self.assertIn("dataset.json", manifest["files"])
            for path in app.iterdir():
                self.assertEqual(path.read_bytes(), checksums[path.name])
            subprocess.run(command + ["--resume"], check=True, cwd=ROOT, env=environment,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            review = output / "protected/review_sample.csv"
            with review.open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["review_decision"] = "pass"
            with review.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            finalize = [sys.executable, str(ROOT / "build_research_dataset.py"),
                        "--output-root", str(output), "--finalize-review", str(review)]
            subprocess.run(finalize, check=True, cwd=ROOT, env=environment,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            final = json.loads((output / "researcher/dataset.json").read_text())
            self.assertEqual(final["release_status"], "ready")


if __name__ == "__main__":
    unittest.main()
