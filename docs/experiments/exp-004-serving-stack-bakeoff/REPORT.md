# exp-004 REPORT — Serving-stack bake-off

**Thesis:** for at least one IRIS tier workload, an alternative serving
stack (raw llama.cpp or MLX) beats Ollama on the tier's deciding metric by
enough to justify the operational change.

**Verdict: ACCEPTED — for exactly one tier (Tier 3, MoE on MLX: +39%
decode, −4.1 GB vs Ollama on identical weights). REJECTED everywhere
else.** Four spikes of a 5-spike budget.

## Per-tier decisions

| Tier | Decision | Evidence |
|---|---|---|
| Router | Keyword classifier — no LLM, no stack | exp-005 (+v1): unbeaten by 9 LLM configs; spike 1: schema constraint made LLM routing *worse* (−6..−10 pts) |
| Tier 1 / 2 / code_exec (dense 3–7B) | **Stay on Ollama, default flags** | Spike 2: stacks near-identical (accuracy travels with weights; decode hardware-bound at 67–70 tok/s); spike 3: "free" tuning flags cost −6% decode → reverted |
| Tier 3 (complex tasks) | **Switch model to Qwen3.6-35B-A3B MoE, serve via MLX** | Spike 4: dense-27B incumbent 18.6 tok/s; same-daemon MoE 75.2; MLX MoE 104.4 @ 19.7 GB (replicates exp-003) |

## The five headline findings

1. **Accuracy travels with the weights, not the server** — identical
   scores on the same GGUF across Ollama/llama.cpp; MLX quant within
   noise (spike 2).
2. **Guaranteed-valid ≠ correct** — schema-constrained decoding removed
   parse failures at 0.6× latency and *cost* 6–10 accuracy points: the
   keyword fallback was an ensemble member, and parse failure was a
   confidence signal (spike 1).
3. **The "free wins" weren't** — flash attention + q8 KV measured −6%
   decode for memory savings IRIS can't use; reverted (spike 3).
4. **The Tier-3 incumbent was the wrong model before it was the wrong
   stack** — the MoE is 4× faster on the same daemon; MLX adds +39% on
   top at −4.1 GB (spike 4A).
5. **exp-001's cliff rule survives production prompt shapes** — clean
   step function between buffer ratios 0.50 and 0.55 on the IRIS router
   prompt at 2k tokens, deterministic (spike 4B). The published
   recommendation is now in-harness-validated.

## Pattern worth naming

Four plausible, widely-recommended improvements were measured this
program: the LLM router, schema-constrained routing, "free" serving
flags, and wholesale stack migration. **Three lost to the boring
incumbent; the one that won did so only where the architecture
(MoE) actually differs — and the model change mattered more than the
stack change.** Measure before you migrate.

## Limits

One machine (M4 Max 36 GB), one daemon version pair (Ollama 0.30.7,
mlx-lm 0.31.3), Tier-3 quality compared via exp-003's grading rather than
re-graded per stack, memory methods differ per stack (documented per
spike). Operability spike (multi-tier topology under load) not run —
budget retained; the Tier-3 adoption checklist covers its essentials.
