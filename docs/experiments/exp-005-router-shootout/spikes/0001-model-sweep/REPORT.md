# Spike 0001 — Model sweep: no candidate beats the regex table

**Question:** Does any installed small local LLM (3–9B), given IRIS's exact
production router prompt + guarded parser, beat the keyword classifier's
88.3% on the 77-case golden set?

**Answer: No.** Best genuine LLM: granite4 (4B) at 84.4% / 322ms p50.

| Config | Accuracy | clear | ambiguous | adversarial | p50 | Parse-fallback |
|---|---|---|---|---|---|---|
| keyword (baseline) | **88.3%** | 100% | 50% | 43% | ~0ms | — |
| llm:qwen3.5:4b | 88.3%* | — | — | — | **5,403ms** | **100%** |
| llm:granite4 | 84.4% | 88% | 70% | 71% | 322ms | 19% |
| llm:llama3.2:3b (prod config) | 75.3% | 80% | 50% | 57% | 301ms | 34% |
| llm:qwen2.5:7b | 75.3% | 78% | 60% | 71% | 514ms | 8% |
| llm:gemma2:9b | 62.3% | 62% | 60% | 71% | 776ms | 3% |

Run: `uv run agentic-evals exp-005 model-sweep` (temp 0.1, num_ctx 1024,
num_predict 256 — production router tier settings). Raw per-case rows in
`results/raw-results.jsonl`.

## Findings

1. **qwen3.5:4b's 88.3% is an illusion.** It is a thinking-by-default model;
   with the production 256-token output cap, the entire budget is consumed by
   `<think>` content and **every single reply fails JSON parsing** (fallback
   rate 100%). Its score IS the keyword classifier's score, reached after a
   5.4-second detour per request. Two lessons: (a) thinking models are
   disqualified at production router settings; (b) high fallback rates can
   silently impersonate good accuracy — always report fallback rate next to
   accuracy. (Echoes exp-001's DeepSeek-R1 verdict: reasoning tuning breaks
   structured emission under tight budgets.)
2. **Bigger is not better at routing.** gemma2:9b is the largest and worst
   (62.3%) — it emits clean JSON (3% fallback) that is confidently wrong.
   qwen2.5:7b parses best (8% fallback) and still lands at 75.3%.
3. **granite4 (4B) is the only LLM with a defensible profile** — and it's
   the model IRIS already uses for Tier-1 *execution*, not routing. It beats
   the keyword table on adversarial (71% vs 43%) and ambiguous (70% vs 50%)
   cases but loses on clear ones (88% vs 100%) — and clear cases dominate
   real traffic.
4. **The production pick (llama3.2:3b) replicated Phase 2 closely** (75.3%
   here vs 72.7% in project-iris; different process, same shape), with a
   34% parse-fallback rate propping it up.
5. **Determinism note:** at temp 0.1 LLM accuracy moves run-to-run (see
   spike 0003 n=3: granite4 81.8–85.7%); the keyword table is byte-identical.

## Data notes

- `results/raw-results.jsonl` is append-only across spikes; granite4 has two
  `run=0` row sets (initial sweep + replication batch), which pool to 83.8%
  in `leaderboard.csv`'s `run=0` line. Individual runs: 84.4 / 83.1 / 85.7 /
  81.8.
