# exp-005 REPORT — Router shootout

**Thesis:** some small local LLM (≤9B) beats IRIS's keyword classifier
(88.3%, ~0ms) on the 77-case routing golden set at p50 ≤ 500ms, standalone or
as the keyword-first hybrid's secondary.

**Verdict: REJECTED** (2 spikes + n=3 replication; budget 4, used 2).

## Final leaderboard (full table: `results/leaderboard.csv`)

| Rank | Config | Accuracy | p50 | Note |
|---|---|---|---|---|
| 1 | **keyword regex table** | **88.3%** | **~0ms** | deterministic, byte-identical |
| – | qwen3.5:4b (LLM or hybrid) | 88.3%* | 5,403ms | *100% parse-fallback → it IS the keyword score |
| 2 | hybrid:granite4 | 87.0% | ~0ms p50 | LLM touches only 3 cases, loses 1 |
| 3 | llm:granite4 (n=4) | 83.8 ± 1.7% | 322ms | best genuine LLM; wins adversarial, loses clear |
| 4 | llm:llama3.2:3b (production pick) | 75.3% | 301ms | 34% propped up by parse-fallback |
| 5 | llm:qwen2.5:7b | 75.3% | 514ms | cleanest JSON (8% fallback), mediocre routing |
| 6 | llm:gemma2:9b | 62.3% | 776ms | biggest model, worst router |

## The five findings

1. **A regex table beats every small LLM at routing its own workload** —
   on clear cases (the bulk of real traffic) it is unbeaten at 100% vs the
   best LLM's 88%, at zero latency and zero variance. LLMs only win on
   adversarial phrasing (71% vs 43%), which is rarer.
2. **The production hybrid has a structural ceiling.** Binary keyword
   confidence (0.85 if any rule fires) means the LLM sees only no-rule
   fallback cases — 3 of 77 here, all already correct. The keyword stage's
   9 errors are all *confident* errors the LLM never gets to touch. The
   architecture caps at the keyword score by construction.
3. **Escalation headroom exists but needs an arbiter.** granite4 fixes 6/9
   keyword errors yet breaks 9 keyword-correct cases when it disagrees —
   naive escalation nets negative; a perfect arbiter would hit 96.1%.
   The cheapest path to that headroom is probably *better rules* (several
   traps look rule-fixable), not a second model.
4. **Thinking models silently impersonate accuracy.** qwen3.5:4b posts
   88.3% — every reply unparseable (thinking ate the 256-token budget),
   every case keyword-fallback, 5.4s wasted per request. Without the
   fallback-rate column this would have looked like a tie for first.
5. **Parse-fallback is load-bearing.** The production pick llama3.2:3b
   "scores" 75.3% with a third of its answers actually coming from the
   keyword fallback. Accuracy claims for LLM routers must always be
   reported alongside fallback rate.

## Replication & determinism

granite4 n=4: 84.4 / 83.1 / 85.7 / 81.8 (mean 83.8 ± 1.7) at temp 0.1 —
LLM routing is ±2-point noisy run-to-run even near-greedy. The keyword
classifier is deterministic (n>1 byte-identical, no replication needed).

## Limits

One golden set (77 cases, English, IRIS's intent taxonomy), one prompt
(production verbatim), one serving stack (Ollama), one hardware class.
The golden set's clear:ambiguous:adversarial mix (60:10:7) approximates but
is not measured real traffic. Models tested are what's installed locally;
a purpose-tuned router model (e.g. a fine-tune on the taxonomy) is untested
and is the most credible way the thesis could still be rescued.

## Cross-links

- Eval set + baseline provenance: project-iris Phase 2
  (`docs/testing-program/phase-2-scenario-harness.md` @ `3cb65b3`).
- Production gap found: IRIS's router parser does not strip `<think>`
  blocks (this experiment's parser does, as a measured accommodation) —
  filed as a carry-forward for project-iris.
