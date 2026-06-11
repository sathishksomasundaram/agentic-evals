# exp-005 — Router shootout: can any small local LLM beat a regex table?

> Numbering note: exp-004 is reserved for the Ollama↔MLX serving-stack
> bake-off (planned); this experiment landed first because Phase 2 of the
> IRIS testing program produced its eval set and a surprising baseline.

## Claim

The IRIS testing program (project-iris, Phase 2) measured the production
LLM router — llama3.2:3b with a structured-JSON routing prompt — at **72.7%**
on a 77-case routing golden set, **losing to the keyword regex classifier's
88.3%** while costing ~330ms p50 vs ~0ms. The folk claim being tested is the
intuition behind every "use a small LLM as your router" recommendation
(including our own exp-001 verdict that llama3.2:3b is the Tier-1 router
pick):

**Thesis:** *Some small local LLM (≤9B, Ollama, as configured for IRIS
routing) beats the keyword classifier's 88.3% on the routing golden set at
p50 ≤ 500ms — either standalone or as the low-confidence secondary in a
keyword-first hybrid.*

Accept/reject is decided per configuration; the practical output is the
best routing configuration for IRIS with measured evidence.

## Goal

Done means: a leaderboard of router configurations (pure-keyword, pure-LLM
per model, keyword-first hybrid per model) with accuracy (overall + by case
class) and latency (p50/p95), an ACCEPT/REJECT verdict on the thesis, the
winner replicated n=3, and a RECOMMENDATIONS.md that names the config IRIS
should ship.

## Fixed setup

| Item | Value |
|---|---|
| Hardware | Apple Silicon (M4 Max, 36 GB), same box as exp-002/003 |
| Serving | Ollama (native API), models as installed locally |
| Router prompt | IRIS production prompt, vendored verbatim from `project-iris` `src/iris/core/intent_router.py` @ `3cb65b3` |
| Parser/guards | IRIS production logic: first-JSON-object extraction, valid intent/agent vocab, intent→agent consistency guard, keyword fallback on any failure |
| Eval set | 77-case routing golden set, vendored from `project-iris` `scenarios/data/routing_golden.yaml` @ `3cb65b3` (60 clear / 10 ambiguous / 7 adversarial) |
| Sampling | temperature 0.1, max 256 output tokens, num_ctx 1024 — the production router tier settings |
| Scoring | case passes iff intent AND agent both match the golden expectation |

## Variations (independent variables)

1. **Model** (spike 0001): llama3.2:3b (production baseline), qwen3.5:4b,
   granite4 (4B), qwen2.5:7b-instruct, gemma2:9b.
2. **Architecture** (spike 0002): pure-LLM vs keyword-first hybrid (keyword
   answers when its rule fires at confidence ≥0.8; LLM only on fallback).
3. (Reserve) Prompt variants — only if 1–2 leave the thesis undecided.

## Expected range (priors)

- Pure-LLM: 65–85%. Phase 2 measured llama3.2:3b at 72.7%; a 7–9B model
  should do better but the consistency guard already rescues many failures —
  surprises live above 88.3%.
- Hybrid: ≥88.3% by construction on keyword-confident cases (73/77); the
  question is whether the LLM lifts the 4 fallback cases without new errors.
  Ceiling for hybrid = keyword wrong-answers replaced is impossible (keyword
  answers are final), so hybrid max = 88.3% + (4 fallback cases) ≈ 93.5%.
- Latency: 3–4B ≈ 200–400ms p50; 7–9B ≈ 400–900ms p50.

## Metrics & outputs

Per configuration: overall accuracy, accuracy by class (clear/ambiguous/
adversarial), parse-fallback rate, latency p50/p95/max, per-case JSONL.
Outputs: `results/leaderboard.csv`, `results/raw-results.jsonl`, spike
REPORTs, consolidated REPORT.md, RECOMMENDATIONS.md.

## Iteration budget

4 spikes max (model sweep, hybrid, winner replication, one contingency).

## Success / Reject criteria

- **ACCEPT** the thesis if any configuration achieves >88.3% overall at
  p50 ≤ 500ms, replicated (n=3 within ±2 points).
- **REJECT** if no configuration clears the keyword baseline within budget —
  the practical recommendation is then keyword(-first) routing, which is
  itself a publishable anti-hype result.

## Stop conditions

Thesis decided at n=3; or iteration budget exhausted; or an environment
blocker (Ollama instability) survives one retry round.
