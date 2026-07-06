#!/usr/bin/env bash
# Tuning-lever sweep — an agent-agnostic TEMPLATE.
#
# This is the shape of the campaign behind exp-006: run the same behaviour suites
# across iterations, flipping ONE knob per iteration, and record whether correctness
# moved (it won't) vs. latency (it will). The finding: the golden config is the
# defaults; the knobs move latency, not outcomes.
#
# The knob names below are GENERIC. Map each to your own agent's feature flag:
#
#   generic knob            what it toggles
#   --------------------    --------------------------------------------------------
#   semantic_reroute        embedding-based intent re-router over the keyword router
#   output_safety_judge     an LLM guard screening the final answer for unsafe content
#   leak_judge              an LLM judge blocking PII / secret leakage in the reply
#   grounding_judge         checks the answer's claims trace to retrieved sources
#   tier_escalation         diagnose a weak cheap-tier answer → escalate / clarify / reroute
#
# Requires: an agent under test that (a) accepts these flags as environment and
# (b) exposes a scenario runner over the suites in ./suites. As run for exp-006 the
# agent was a local-first assistant with a YAML scenario runner; substitute yours.
# The reusable, runnable-by-anyone artifact is ../probe-kit (see ../README.md).
set -uo pipefail

RUNNER="${RUNNER:-./run-suite.sh}"   # your command: takes a suite path, prints "N/M passed" + per-scenario "…ms"
SUITES_DIR="${SUITES_DIR:-./suites}"
OUT="${OUT:-./campaign-results.csv}"

echo "iter,knob,routing_pass,injection_pass,fastpath_med_ms,worst_ms" > "$OUT"

measure() {  # $1 = suite file → echoes "PASS MED_MS WORST_MS"
  local log; log="$("$RUNNER" "$1" 2>/dev/null)"
  local pass; pass="$(printf '%s' "$log" | grep -oE '[0-9]+/[0-9]+ passed' | head -1 | sed 's/ passed//')"
  local ms;   ms="$(printf '%s' "$log" | grep -oE '[0-9]+ms' | tr -d 'ms')"
  local med worst
  med="$(printf '%s\n' $ms | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]; else print 0}')"
  worst="$(printf '%s\n' $ms | sort -n | tail -1)"
  echo "${pass:-0/0} ${med:-0} ${worst:-0}"
}

iteration() {  # $1 iter  $2 knob-label  $3.. KEY=VAL flags
  local n="$1" knob="$2"; shift 2
  ( for kv in "$@"; do export "$kv"; done
    read -r rp rmed rworst <<<"$(measure "$SUITES_DIR/routing-breadth.yaml")"
    read -r ip _ _        <<<"$(measure "$SUITES_DIR/injection-pressure.yaml")"
    printf '%s,%s,%s,%s,%s,%s\n' "$n" "$knob" "$rp" "$ip" "$rmed" "$rworst" >> "$OUT"
    echo "iter $n [$knob]: routing=$rp injection=$ip fast=${rmed}ms worst=${rworst}ms"
  )
}

iteration 1 "baseline (defaults)"
iteration 2 "semantic_reroute"        semantic_reroute=1
iteration 3 "output_safety_judge"     output_safety_judge=1
iteration 4 "leak+grounding judges"   leak_judge=1 grounding_judge=1
iteration 5 "tier_escalation (shadow)" tier_escalation=1
iteration 6 "all on"                  semantic_reroute=1 output_safety_judge=1 leak_judge=1 grounding_judge=1 tier_escalation=1

echo; echo "=== results → $OUT ==="
column -t -s, "$OUT" 2>/dev/null || cat "$OUT"
echo "Golden = the row with the best pass rates at the lowest latency (expected: the baseline)."
