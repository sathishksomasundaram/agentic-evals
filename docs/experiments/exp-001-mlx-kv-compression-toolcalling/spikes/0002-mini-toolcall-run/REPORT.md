# Mini-spike 0002 — REPORT

**Run date:** 2026-05-27 · **Status:** Complete · **Outcome:** Strong signal; v0 plan needs revision.

> **Archival note:** this spike's runner ([`mini_toolcall.py`](../../../../../src/agentic_evals/experiments/exp001/mini_toolcall.py)) is pinned to the `rachittshah/mlx-turboquant` port (`TurboQuantKVCache`), superseded by `yzamari/mlx-turboquant` in [spike 0005](../0005-yzamari-port/REPORT.md). It won't import under a fresh `uv sync` — kept as a record of the trail, not a runnable target.

## What we ran

- Model: `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` (28 layers, head_dim=128)
- Framework: `mlx-lm 0.31.3` on Apple Silicon (Metal device)
- TurboQuant port: `rachittshah/mlx-turboquant` @ commit `f39a74b3`
- Two runs back-to-back, `max_tokens=200`, identical prompt (system+user from [prompt-001.md](prompt-001.md))
- Raw JSON: [raw-results.json](raw-results.json)

## Headline results

| Run | Cache | Wall time | Peak GPU mem | Tool-call correctness |
|---|---|---|---|---|
| 1 | mlx-lm default (FP) | **3.38 s** | 4.585 GB | ⚠ Reasoning-preamble — never reached JSON in 200 tokens |
| 2 | `TurboQuantKVCache(bits=3)` | 5.528 s | 4.62 GB | ❌ **Complete collapse** — emitted `"!!!!!!"` × 200 |

Run 1 output (truncated to last 200 tokens of context):
> *"Okay, so I need to figure out the current weather in San Francisco. I remember that the user asked for this information, and I have a tool called web_search that I can use ... I'll structure the JSON with the tool as 'web_search', the query as 'current weather in San Francisco', and the time_range as an empty string ... I'm a bit nervous about the formatting, but I think as long as I follow the structure, it should be fine. I'll go ahead and"* — [cut off]

Run 2 output:
> `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` × 200 tokens. Degenerate token loop.

## Honest interpretation (Rule 2)

### Finding 1 — DeepSeek-R1-Distill is **wrong for tool-calling at small token budgets**
R1-Distill models are reasoning-tuned: they "think out loud" before answering. The model *was* about to emit the right JSON (the thought process shows it understood the task and was approaching the correct call). It just ran out of token budget before getting there.

Implication: this model is **not a good first wedge target for tool-calling**. Either:
- Use `Qwen-2.5-7B-Instruct` (iris's actual Tier 2 model — non-reasoning-tuned, will answer directly)
- Or test R1-Distill with `max_tokens=600+` (also adds noise — reasoning preamble varies)
- Or suppress reasoning via prompt (fragile)

**Recommendation: switch v0's first model to `mlx-community/Qwen2.5-7B-Instruct-4bit`.** R1-Distill stays in the matrix but specifically for *reasoning* capability evaluations, not tool-calling.

### Finding 2 — `TurboQuantKVCache(bits=3)` symmetric **destroys** this model's coherence
The output collapse to `!!!` is not subtle quality degradation; it's complete model breakdown. The KV cache is so corrupted that the next-token distribution degenerates into a single high-probability junk token that the model then echoes.

This directly contradicts the [POC's](../0001-inference-knobs-landscape/turboquant-poc/POC-REVIEW.md) optimistic K=V=3 result (66% needle retrieval on synthetic data ≠ "model still works"). It **validates yzamari's honest disclosure** that "needle retrieval fails on compressed tokens."

**What this proves:** synthetic-data KV-quant benchmarks don't predict real-model behavior under autoregressive decoding. This is exactly the gap open-model-benchmark exists to measure. It's a feature, not a bug, that the spike surfaced this.

**However:** before declaring rachittshah's port unfit, two confounds need ruling out:
1. **Untested bit-widths.** `bits=4` may work fine. The cliff is somewhere between 4 and 3.
2. **Port correctness vs. POC.** The POC's NumPy reference should give the same encode/decode roundtrip as rachittshah's MLX impl. If they diverge, the port has a bug (not the algorithm).

### Finding 3 — Memory savings are invisible at this prompt scale
Δ peak memory: 4.62 GB (TQ on) – 4.585 GB (TQ off) = **+35 MB**. TurboQuant marginally *increased* peak memory at this scale — the cache codebook + bookkeeping outweighs the savings on a 735-char prompt + 200-token generation. The model weights dominate; KV cache is negligible at short contexts.

**Implication:** memory measurements only become meaningful at **long contexts (≥4 K tokens)**. v0's measurement plan must include at least one long-context probe, or memory results will be uninterpretable.

### Finding 4 — TQ-on was 64% slower at this scale
5.528 s vs 3.38 s. Likely overhead from encode/decode each step. rachittshah's port is pure-MLX (no fused Metal kernel). yzamari's port would invert this (it has the Metal kernel that bypasses decompression). **Speed is a port-quality question, not a TurboQuant question.**

## Specific v0 plan changes implied

| ADR 0008/0009 said | Mini-spike implies |
|---|---|
| Use DeepSeek-R1-Distill-Qwen-7B for tool-calling | **No** — use Qwen-2.5-7B-Instruct for tool-calling; keep R1-Distill for reasoning evals |
| TurboQuant K4/V2 on/off, symmetric proxy at bits=3 | **Bits=3 symmetric is broken on this stack.** Test bits=4 first; characterize the cliff between 4 and 3. |
| Hold context fixed | **Add a long-context (4K+) probe** — short-context memory measurements are meaningless |
| Port choice: rachittshah primary | **Re-evaluate.** Pure-MLX overhead is real. yzamari's Metal kernel matters once you measure latency. |
| Methodology: extend POC's needle/KL/cosine to live runs | **POC quality metrics on synthetic data do NOT predict real-model behavior** — translation isn't direct |

## Recommended next decisions

1. **ADR 0011** — supersede ADR 0008 + 0009's model list: use Qwen-2.5-7B-Instruct as the v0 tool-calling primary; keep DeepSeek-R1-Distill-Qwen-7B reserved for reasoning evals only.
2. **Spike 0003** — repeat this run with `bits=4` symmetric and `bits=2`. Establishes the quality cliff on real models.
3. **Spike 0004** — switch port to yzamari/mlx-turboquant; re-run the same prompt; characterize speed delta and any quality difference (asymmetric K3/V2 supported there).
4. **Spike 0005** — add a long-context prompt (~4 K tokens) to expose memory differences that this short test missed.

## What's NOT in this report (skipped per Rule 3)

- No statistical significance (n=1 per condition; will need ≥3 runs in v0).
- No correctness validation against the POC's NumPy reference (planned for spike 0003 — confirm rachittshah's port encode/decode matches POC math).
- No comparison against yzamari/mlx-turboquant (deferred to spike 0004).
- No analysis of *which layer* TurboQuant breaks first (skipped — would require instrumented model).
- No iris-via-API tool-call test (intentionally — we tested via raw model, not iris's full agentic stack; that comes later).

## Mini-spike conclusion

**The test apparatus produces strong, interpretable signal — the spike succeeded.** The signal happens to be "the v0 plan needs three concrete fixes before scale-up," which is exactly the point of running a spike instead of building the full matrix first.
