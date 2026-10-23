#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from academic_pipeline_v2.models import (
    CVExtraction, CanonicalDocument, DocumentCheckpoint, EntityMention, LetterExtraction,
    PrivacyResult, QualityRecord, ResolvedRelationship,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = (CanonicalDocument, CVExtraction, LetterExtraction, EntityMention,
              PrivacyResult, QualityRecord, DocumentCheckpoint, ResolvedRelationship)
    for model in models:
        path = args.output / f"{model.__name__}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
