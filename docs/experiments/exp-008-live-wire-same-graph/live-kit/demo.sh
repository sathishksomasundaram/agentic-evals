#!/usr/bin/env bash
# Run the SAME email_digest graph twice on the REAL Harness Lab engine — once in
# sim, once live against a real MCP mail server over stdio — and print the two
# traces side by side. Exit 0 iff the event-type sequences are identical (same
# topology, every governance node fired in the same place), determinism flips
# true -> false, and the live rows actually came over the wire. Data-dependent
# payloads differ where the data differs — that's the point, and it's printed.
set -uo pipefail
cd "$(dirname "$0")"
if ! python3 -c "import harnesslab, mcp" 2>/dev/null; then
  echo "Needs the real engine + MCP transport:  pip install -r requirements.txt"
  echo "(installs the open-source engine straight from GitHub until the PyPI release)"
  exit 2
fi
python3 run_livewire.py "$@"
