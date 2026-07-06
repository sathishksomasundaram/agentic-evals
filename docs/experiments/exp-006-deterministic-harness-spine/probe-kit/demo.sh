#!/usr/bin/env bash
# Self-contained demo: start the mock agent, run all three probe batteries against
# it, then stop the mock. Requires only Python 3.9+ and curl (both standard).
# Expected: every battery clean (routing GROUNDED, injection/redteam DEFENDED).
set -uo pipefail
cd "$(dirname "$0")"

python3 mock_agent.py >/dev/null 2>&1 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null' EXIT
sleep 1

python3 run_probes.py --agent agent.demo.json --battery all --repeats 3
