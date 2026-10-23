from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .evaluation import evaluate_run
from .exporter import validate_candidate_package
from .pipeline import (
    export_candidate_stage, extract_stage, finalize_review_stage, inventory_stage,
    process_stage, release_stage, status_stage,
)
from .registry import EntityRegistry
from .util import read_json, sha256_file


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="academic-pipeline-v2")
    root.add_argument("--log-level", default="INFO",
                      choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("inventory", "extract", "process", "resume", "status", "evaluate",
                 "validate-release"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--config", type=Path, required=True)
    mode = export.add_mutually_exclusive_group(required=True)
    mode.add_argument("--candidate", action="store_true")
    mode.add_argument("--release", action="store_true")
    finalize = commands.add_parser("finalize-review")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--decisions", type=Path, required=True)
    migrate = commands.add_parser("migrate-v1-registry")
    migrate.add_argument("--config", type=Path, required=True)
    migrate.add_argument("--v1-registry", type=Path, required=True)
    action = migrate.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--apply", action="store_true")
    return root


def configure_logging(config: object, level: str) -> None:
    output_root = getattr(config, "output_root")
    metrics = output_root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(getattr(logging, level))
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(metrics / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(stream)
    logging.getLogger().addHandler(file_handler)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        configure_logging(config, args.log_level)
        if args.command == "inventory":
            applications, documents = inventory_stage(config)
            result: object = {"application_count": len(applications),
                              "document_count": len(documents)}
        elif args.command == "extract":
            result = {"canonical_document_count": len(extract_stage(config))}
        elif args.command == "process":
            result = process_stage(config, retry_failed=False)
        elif args.command == "resume":
            result = process_stage(config, retry_failed=True)
        elif args.command == "status":
            result = status_stage(config)
        elif args.command == "evaluate":
            result = evaluate_run(config)
        elif args.command == "validate-release":
            result = validate_candidate_package(config.output_root)
        elif args.command == "export":
            result = {"package": str(export_candidate_stage(config) if args.candidate
                                      else release_stage(config))}
        elif args.command == "finalize-review":
            result = finalize_review_stage(config, args.decisions)
        elif args.command == "migrate-v1-registry":
            registry = EntityRegistry(config.state_root / "registry.sqlite")
            try:
                payload = read_json(args.v1_registry)
                result = registry.import_v1(payload, sha256_file(args.v1_registry), args.apply)
            finally:
                registry.close()
        else:
            raise ValueError("unsupported_command")
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        logging.error("Command failed command=%s error=%s", args.command, type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

