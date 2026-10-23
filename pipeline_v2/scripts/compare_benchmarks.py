#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare paired v2 evaluation JSON files")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.results:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"path": str(path), "model": value.get("model"),
                     "release_rate": value.get("release_rate"),
                     "failure_rate": value.get("failure_rate"),
                     "model_call_count": value.get("model_call_count"),
                     **value.get("accuracy", {})})
    print(json.dumps({"paired_results": rows,
                      "decision": "No winner is selected automatically; review quality, privacy, "
                                  "latency, token, and VRAM metrics together."},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
