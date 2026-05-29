"""Result persistence: append JSONL rows, materialize CSV from JSONL.

JSONL is the append-only source of truth (survives crashes, multiple invocations);
CSV is a flat view rebuilt from it on demand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSONL file (one JSON object per line), creating dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def rebuild_csv_from_jsonl(jsonl_path: Path, csv_path: Path, fields: list[str]) -> int:
    """Rewrite `csv_path` as a flat table of all rows in `jsonl_path`.

    Returns the number of rows written. Extra keys not in `fields` are ignored.
    """
    rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)
