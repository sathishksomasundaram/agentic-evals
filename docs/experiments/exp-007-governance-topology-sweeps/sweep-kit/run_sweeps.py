#!/usr/bin/env python3
"""Governance-topology sweeps — the runnable proof behind "The Spine, Drawn".

This runs the REAL Harness Lab engine (`harnesslab`, installed from
requirements.txt), not a mock. It ships only what THIS experiment needs: three
graphs, three fixtures, and this runner. Each sweep moves ONE governance knob
across its range and grades the BEHAVIOUR of every grid cell — which gate fired,
whether the model was reached, whether the answer shipped — not the prose.

Determinism is the whole point: same graph + same fixture + same seed => the
same event trace, so your table matches the post's byte for byte. The runner
exits non-zero if any cell drifts from the expected outcome (CI-ready).

    python run_sweeps.py            # run all three, print the tables, assert them
    python run_sweeps.py --json     # machine-readable

No API key needed: the egress sweep's "did personal data reach the cloud model"
signal is the `llm_request` event, which the engine emits at the boundary BEFORE
any network call — so the leak (or the block) is visible whether or not the call
would have succeeded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harnesslab.experiments import run_sweep
from harnesslab.schema import GraphModel
from harnesslab.sim import load_fixture

HERE = Path(__file__).resolve().parent


def _graph(name: str) -> GraphModel:
    return GraphModel.model_validate(json.loads((HERE / "sweeps" / name).read_text()))


def _fixture(name: str):
    return load_fixture(str(HERE / "fixtures" / name))


def _events(cell) -> list:
    return cell["run"].events


def _has(cell, ev_type: str) -> bool:
    return any(e["type"] == ev_type for e in _events(cell))


def _count(cell, ev_type: str) -> int:
    return sum(1 for e in _events(cell) if e["type"] == ev_type)


# Each sweep: a graph, a fixture, one knob across values, and how to read each
# cell's behaviour into a short outcome string + the expected outcome per value.
SWEEPS = [
    {
        "id": "egress",
        "title": "Sweep 1 — egress mode: personal data -> a cloud model",
        "graph": "egress_cloud.json",
        "fixture": "ssn_to_cloud.yaml",
        "knob": "data_classifier.egress",
        "values": ["off", "warn", "enforce"],
        "columns": ["reached_model", "egress_denied", "outcome"],
        "read": lambda c: {
            "reached_model": "yes" if _has(c, "llm_request") else "no",
            "egress_denied": _count(c, "egress_denied"),
            "outcome": "LEAKS" if _has(c, "llm_request") else "blocked before egress",
        },
        "expect": {
            "off": {"reached_model": "yes", "egress_denied": 0},
            "warn": {"reached_model": "yes", "egress_denied": 1},
            "enforce": {"reached_model": "no", "egress_denied": 1},
        },
    },
    {
        "id": "tier",
        "title": "Sweep 2 — forbidden-tier fallback: a personal request",
        "graph": "governed_chat_spine.json",
        "fixture": "personal_downshift.yaml",
        "knob": "tier_router.on_forbidden_egress",
        "values": ["downshift", "warn", "block"],
        "columns": ["status", "downshifted", "outcome"],
        "read": lambda c: {
            "status": c["status"],
            "downshifted": "yes" if _has(c, "tier_downshifted") else "no",
            "outcome": "answered (routed local)" if c["status"] == "completed" else "blocked",
        },
        "expect": {
            "downshift": {"status": "completed", "downshifted": "yes"},
            "warn": {"status": "blocked", "downshifted": "no"},
            "block": {"status": "blocked", "downshifted": "no"},
        },
    },
    {
        "id": "judge",
        "title": "Sweep 3 — curation judge threshold: a confidently-wrong answer",
        "graph": "governed_chat_spine.json",
        "fixture": "ungrounded.yaml",
        "knob": "llm_judge.threshold",
        "values": [0.3, 0.6, 0.9],
        "columns": ["status", "outcome"],
        "read": lambda c: {
            "status": c["status"],
            "outcome": "SHIPPED the hallucination" if c["status"] == "completed" else "caught + blocked",
        },
        "expect": {
            0.3: {"status": "completed"},
            0.6: {"status": "blocked"},
            0.9: {"status": "blocked"},
        },
    },
]


def run_one(spec: dict) -> dict:
    graph = _graph(spec["graph"])
    fixture = _fixture(spec["fixture"])
    cells = run_sweep(graph, fixture, {spec["knob"]: spec["values"]})
    rows, ok = [], True
    for value, cell in zip(spec["values"], cells):
        read = spec["read"](cell)
        expected = spec["expect"][value]
        drift = {k: (read[k], v) for k, v in expected.items() if read.get(k) != v}
        ok = ok and not drift
        rows.append({"value": value, **read, "drift": drift})
    return {"id": spec["id"], "title": spec["title"], "knob": spec["knob"],
            "columns": spec["columns"], "rows": rows, "ok": ok}


def print_table(result: dict) -> None:
    print(f"\n{result['title']}")
    print(f"  knob: {result['knob']}")
    cols = result["columns"]
    width = max(len(str(r["value"])) for r in result["rows"]) + 2
    header = f"  {'value':<{width}} " + "  ".join(f"{c:<22}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in result["rows"]:
        cells = "  ".join(f"{str(r[c]):<22}" for c in cols)
        flag = "" if not r["drift"] else f"   << DRIFT {r['drift']}"
        print(f"  {str(r['value']):<{width}} {cells}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Governance-topology sweeps (real Harness Lab engine)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    results = [run_one(s) for s in SWEEPS]
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print("Governance-topology sweeps — real Harness Lab engine, deterministic runs.")
        for r in results:
            print_table(r)
        allok = all(r["ok"] for r in results)
        print("\n" + ("All sweeps matched the expected table. Reproducible, or it didn't happen."
                       if allok else "DRIFT: a cell did not match the expected outcome (see above)."))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
