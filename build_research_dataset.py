#!/usr/bin/env python3
"""Build protected extraction artifacts and a de-identified academic research dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from research_pipeline import PROMPT_VERSION, SCHEMA_VERSION
from research_pipeline.common import atomic_json, config_hash
from research_pipeline.pipeline import (
    discover, extract_documents, finalize_review, package_dataset, resolve_entities,
    repair_privacy_qa, select_review_sample, write_manifest, write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--identity-registry", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit-applications", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--repair-qa", action="store_true",
                        help="Rerun privacy QA only where prior QA failed and redact grounded spans")
    parser.add_argument("--require-review", action="store_true")
    parser.add_argument("--finalize-review", type=Path)
    parser.add_argument("--debug-artifacts", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(output_root: Path, level: str) -> None:
    metrics = output_root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(metrics / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def validate_args(args: argparse.Namespace) -> list[str]:
    if args.finalize_review:
        if args.input_root:
            raise ValueError("--input-root cannot be used with --finalize-review")
        return []
    if not args.input_root:
        raise ValueError("At least one --input-root is required")
    if not args.identity_registry or not args.state_root:
        raise ValueError("--identity-registry and --state-root are required")
    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    if not devices:
        raise ValueError("--devices must name at least one GPU")
    if args.workers < 1 or args.workers > len(devices):
        raise ValueError("--workers must be between 1 and the number of --devices")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    if args.limit_applications is not None and args.limit_applications < 1:
        raise ValueError("--limit-applications must be positive")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    return devices


def prepare_output(root: Path, resume: bool) -> None:
    if root.exists():
        entries = list(root.iterdir())
        if entries and not resume:
            raise ValueError("Output root is nonempty; choose a new directory or use --resume")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("protected/checkpoints", "protected/model_responses", "metrics"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for directory in (root / "protected", root / "protected/checkpoints",
                      root / "protected/model_responses"):
        os.chmod(directory, 0o700)


def main() -> int:
    args = parse_args()
    try:
        devices = validate_args(args)
        if args.finalize_review:
            configure_logging(args.output_root, args.log_level)
            finalize_review(args.output_root, args.finalize_review)
            logging.info("Review finalized output_root=%s", args.output_root)
            return 0
        prepare_output(args.output_root, args.resume)
        configure_logging(args.output_root, args.log_level)
        roots = [path.resolve() for path in args.input_root]
        applications, tasks = discover(roots, args.limit_applications)
        inventory = {"schema_version": SCHEMA_VERSION, "application_count": len(applications),
                     "document_count": len(tasks), "applications": applications,
                     "documents": [{key: value for key, value in task.items()
                                    if key not in {"source_path"}} for task in tasks]}
        protected = args.output_root / "protected"
        atomic_json(protected / "inventory.json", inventory)
        if args.preflight_only:
            logging.info("Preflight completed applications=%d documents=%d flags=%d",
                         len(applications), len(tasks),
                         sum(bool(row["inventory_flags"]) for row in applications))
            return 0
        configuration = {
            "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
            "model": args.model, "dpi": args.dpi, "devices": devices,
            "workers": args.workers,
            "source_inventory_hash": config_hash(
                {task["document_id"]: task["source_sha256"] for task in tasks}),
        }
        configuration["config_hash"] = config_hash(configuration)
        run_state_path = protected / "run_state.json"
        if run_state_path.exists():
            old = json.loads(run_state_path.read_text(encoding="utf-8"))
            if old.get("config_hash") != configuration["config_hash"]:
                raise ValueError("Resume configuration does not match the existing run")
        atomic_json(run_state_path, {**configuration, "status": "extracting"})
        worker_config = {**configuration,
                         "checkpoint_root": str(protected / "checkpoints"),
                         "debug_root": str(protected / "model_responses"),
                         "debug_artifacts": args.debug_artifacts}
        extractions = extract_documents(tasks, worker_config, args.resume, args.retry_failed)
        if args.repair_qa:
            repaired = repair_privacy_qa(extractions, worker_config)
            logging.info("Privacy QA repair completed documents=%d", repaired)
            extractions = extract_documents(tasks, worker_config, True, False)
        resolved = resolve_entities(extractions, args.identity_registry.resolve(),
                                    args.state_root.resolve())
        package = package_dataset(applications, extractions, resolved, args.output_root,
                                  args.require_review)
        summary = write_metrics(extractions, args.output_root / "metrics", len(applications),
                                args.workers)
        if args.require_review:
            sample = select_review_sample(extractions)
            from research_pipeline.common import write_csv
            write_csv(protected / "review_sample.csv",
                      ["document_id", "application_id", "document_type",
                       "review_decision", "reviewer_note"], sample, mode=0o600)
        write_manifest(package["candidate"], tasks, package["dataset"]["release_status"])
        atomic_json(run_state_path, {**configuration, "status": "completed",
                                    "researcher_package": package["candidate"].name,
                                    "redaction_count": package["redaction_count"],
                                    "failed_documents": summary["failed_document_count"]})
        logging.info("Pipeline completed applications=%d documents=%d redactions=%d withheld=%d package=%s",
                     len(applications), len(tasks), package["redaction_count"],
                     package["quality"]["withheld_document_count"], package["candidate"])
        return 0
    except Exception as exc:
        logging.error("Pipeline failed error=%s", type(exc).__name__)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.exception("Debug traceback")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
