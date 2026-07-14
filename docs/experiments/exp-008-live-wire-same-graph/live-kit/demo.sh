#!/usr/bin/env bash
# Run the SAME email_digest graph twice on the REAL Harness Lab engine — once in
# sim, once live against a real MCP mail server over stdio — and print the two
# traces side by side. Exit 0 iff only the two expected lines differ (the tool's
# rows, and the `deterministic` flag) and every governance node behaved the same.
set -uo pipefail
cd "$(dirname "$0")"
if ! python3 -c "import harnesslab, mcp" 2>/dev/null; then
  echo "Needs the real engine + MCP transport:  pip install -r requirements.txt"
  echo "(Harness Lab is heading to open source; install from the repo until then.)"
  exit 2
fi
python3 run_livewire.py "$@"
