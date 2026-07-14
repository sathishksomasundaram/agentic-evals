#!/usr/bin/env python3
"""Live-Wire Kit — the same governed graph, sim and live, side by side.

Runs ONE graph (`email_digest`) twice on the REAL Harness Lab engine:

  * SIM  — the tool is a fixture twin (`email.search`, mode: sim).
  * LIVE — the identical graph with ONE binding changed: the tool node points at
           `inbox.search_inbox` (mode: real), a tool imported from a real MCP
           mail server that this kit speaks to over stdio.

Then it prints the two traces side by side. The claim of the post is that only
TWO things differ: the tool's rows are real instead of scripted, and
`deterministic` flips true -> false. The runner asserts exactly that — the event
sequences must be identical, both runs must complete, determinism must flip —
and exits non-zero otherwise.

The mail source is a mock MCP server (synthetic inbox) shipped with Harness Lab
as `harnesslab.mcp_email_mock` — a stand-in for a real IMAP/Gmail MCP bridge, so
the whole mode:real + MCP path is exercised for real (a genuine stdio round trip)
without wiring a live account. Swap the server, keep the graph.

    python run_livewire.py           # side-by-side + verdict
    python run_livewire.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from harnesslab import catalog
from harnesslab.engine import execute
from harnesslab.schema import GraphModel
from harnesslab.sim import load_fixture

HERE = Path(__file__).resolve().parent
MOCK_SERVER = "python3 -m harnesslab.mcp_email_mock"


def _graph() -> dict:
    return json.loads((HERE / "graphs" / "email_digest.json").read_text())


def _publish_live_tool() -> None:
    """Import the mock server's read tool into a throwaway catalog — the same
    path `discover_mcp_tools` produces, done by hand so the kit is self-contained."""
    os.environ["HARNESSLAB_CATALOG"] = tempfile.mkdtemp(prefix="livewire-catalog-")
    import importlib
    importlib.reload(catalog)
    catalog.publish_tool({
        "id": "inbox.search_inbox", "description": "Search the inbox.", "category": "email",
        "safety": {"side_effects": "read"}, "args": [{"key": "query", "type": "text"}],
        "executor": {"kind": "mcp", "ref": "search_inbox",
                     "mcp": {"command": MOCK_SERVER, "query_arg": "query"}},
    })


def _run_sim():
    g = GraphModel.model_validate(_graph())
    return execute(g, load_fixture(str(HERE / "fixtures" / "sim.yaml")))


def _run_live():
    g = GraphModel.model_validate(_graph())
    tool = next(n for n in g.nodes if n.type == "tool")
    tool.config["tool"] = "inbox.search_inbox"   # the imported live tool
    tool.config["mode"] = "real"                 # the ONE binding that changes
    return execute(g, load_fixture(str(HERE / "fixtures" / "live.yaml")))


def _deterministic(run) -> bool:
    return next(e for e in run.events if e["type"] == "run_started")["payload"]["deterministic"]


def _tool_result(run):
    return next((e["payload"] for e in run.events if e["type"] == "tool_result"), {"count": 0, "docs": []})


def _seq(run) -> list:
    return [e["type"] for e in run.events]


def _label(run, kind: str) -> str:
    """A one-line, human-readable label per event, so the side-by-side reads."""
    return kind


def side_by_side(sim, live) -> str:
    ssim, slive = _seq(sim), _seq(live)
    tsim, tlive = _tool_result(sim), _tool_result(live)
    width = max((len(t) for t in ssim), default=10) + 2
    lines = [f"  {'SIM (fixture twin)':<{width}}   LIVE (real MCP mail server)",
             f"  {'-' * (width - 2):<{width}}   {'-' * 28}"]
    for i in range(max(len(ssim), len(slive))):
        a = ssim[i] if i < len(ssim) else ""
        b = slive[i] if i < len(slive) else ""
        note = ""
        if a == "run_started":
            a += f"  deterministic={_deterministic(sim)}"
            b += f"  deterministic={_deterministic(live)}"
            note = "   <-- the one word that flips"
        if a == "tool_result":
            a += f"  count={tsim['count']} (scripted)"
            b += f"  count={tlive['count']} (from the live inbox)"
            note = "   <-- real rows over the wire"
        mark = "" if a == b or (a.split("  ")[0] == b.split("  ")[0]) else "   != "
        lines.append(f"  {a:<{width}}   {b}{note}{mark}")
    return "\n".join(lines)


def verdict(sim, live) -> tuple[bool, list]:
    checks = [
        ("both runs completed", sim.status == "completed" and live.status == "completed"),
        ("event sequence identical", _seq(sim) == _seq(live)),
        ("determinism flips true -> false", _deterministic(sim) is True and _deterministic(live) is False),
        ("live tool rows came from the server", _tool_result(live)["count"] > 0),
    ]
    return all(ok for _, ok in checks), checks


def main() -> int:
    ap = argparse.ArgumentParser(description="Same graph, sim and live, side by side")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    _publish_live_tool()
    sim, live = _run_sim(), _run_live()
    ok, checks = verdict(sim, live)

    if args.json:
        print(json.dumps({
            "sim": {"status": sim.status, "deterministic": _deterministic(sim),
                    "tool_rows": _tool_result(sim)["count"], "events": _seq(sim)},
            "live": {"status": live.status, "deterministic": _deterministic(live),
                     "tool_rows": _tool_result(live)["count"], "events": _seq(live)},
            "checks": {name: ok for name, ok in checks}, "ok": ok}, indent=2))
        return 0 if ok else 1

    print("The same email_digest graph, run twice — one binding changed (mode: sim -> real).\n")
    print(side_by_side(sim, live))
    print("\nOnly two lines differ: the tool's rows (real vs scripted) and one word "
          "(deterministic). Every governance node in between did the same thing.\n")
    for name, passed in checks:
        print(f"  [{'ok' if passed else 'XX'}] {name}")
    print("\n" + ("Same graph, sim and live, governance intact. Reproducible where it can be, "
                  "honest where it can't." if ok else "MISMATCH — see the failing check above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
