#!/usr/bin/env bash
# Run the three governance-topology sweeps on the REAL Harness Lab engine and
# print the outcome tables. No mock: this imports `harnesslab` and executes the
# actual deterministic engine over the three shipped graphs/fixtures.
#
# Expected (deterministic — matches the post byte for byte):
#   egress:  off -> LEAKS · warn -> LEAKS(+denied) · enforce -> blocked
#   tier:    downshift -> answered · warn -> blocked · block -> blocked
#   judge:   0.3 -> SHIPPED · 0.6 -> blocked · 0.9 -> blocked
# Exit code is 0 iff every cell matches its expected outcome.
set -uo pipefail
cd "$(dirname "$0")"

if ! python3 -c "import harnesslab" 2>/dev/null; then
  echo "The Harness Lab engine isn't installed yet. It's the real thing this kit runs on."
  echo "Install it (Harness Lab is heading to open source):"
  echo "    pip install -r requirements.txt        # once harnesslab is on PyPI"
  echo "    pip install -e /path/to/harness-lab/backend   # from a local checkout"
  exit 2
fi

python3 run_sweeps.py "$@"
