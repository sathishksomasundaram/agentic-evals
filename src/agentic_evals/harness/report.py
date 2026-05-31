"""Result persistence: append JSONL rows, materialize CSV from JSONL.

JSONL is the append-only source of truth (survives crashes, multiple invocations);
CSV is a flat view rebuilt from it on demand.
"""

from __future__ import annotations

import csv
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSONL file (one JSON object per line), creating dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_run(out_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Persist one experiment run for proof.

    Writes two files under `out_dir`:
      - ``runs/<run-id>.json`` — an *immutable* per-run archive (never overwritten),
        so every execution leaves a reproducible record;
      - ``raw-results.json`` — the canonical *latest* run (overwritten each time),
        which the experiment write-ups link to.

    The run id is a UTC timestamp plus a short random suffix (collision-proof for
    runs in the same second). It and an ISO `created_utc` field are injected into
    the stored payload. Returns (archived_path, latest_path).
    """
    now = datetime.now(UTC)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    stored = {"run_id": run_id, "created_utc": now.isoformat(), **payload}
    text = json.dumps(stored, indent=2)

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    archived = runs_dir / f"{run_id}.json"
    archived.write_text(text)

    latest = out_dir / "raw-results.json"
    latest.write_text(text)
    return archived, latest


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
