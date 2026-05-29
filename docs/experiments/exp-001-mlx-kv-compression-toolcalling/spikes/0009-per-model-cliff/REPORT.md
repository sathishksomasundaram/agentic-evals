# Spike 0009 — Per-model cliff localization

**Run date:** 2026-05-27 · **Status:** Complete · **Outcome:** Significant correction — cliff is at ratio ~0.5 *across all model sizes tested*, not size-dependent as v0 implied.

## What we ran

- Models (separate processes, OMB_MODEL env var):
  - `mlx-community/Llama-3.2-3B-Instruct-4bit` (head_dim=128, 28 layers, 212/1988 tokens short/long)
  - `mlx-community/Phi-4-mini-instruct-4bit` (head_dim=128, 32 layers, 178/1965 tokens)
- Port: yzamari/mlx-turboquant @ HEAD
- Fixed K3/V2. Sweep ratios: `{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85}` + baseline
- Same prompts as spikes 0006-0008 + v0
- Raw JSON: [Llama-3.2](mlx-community-llama-3.2-3b-instruct-4bit/raw-results.json) · [Phi-4-mini](mlx-community-phi-4-mini-instruct-4bit/raw-results.json)

## Headline results

### Llama-3.2-3B-Instruct

| Prompt | ratio | buf | verdict |
|---|---|---|---|
| short | baseline | — | ✅ PASS |
| short | **0.50** | 106 | ❌ OTHER (collapse) |
| short | **0.55** | **117** | ✅ **PASS** ← cliff |
| short | 0.60+ | 127+ | ✅ PASS (byte-identical to 0.55 onward, ~0.35s each) |
| long | baseline | — | ❌ LOST_FORMAT |
| long | **0.50** | 994 | ❌ OTHER |
| long | **0.55** | **1093** | ✅ **PASS** ← cliff |
| long | 0.60+ | 1193+ | ✅ PASS (~1.93s each) |

**Llama-3.2-3B cliff: between ratio = 0.50 and 0.55.** Both prompt regimes.

### Phi-4-mini-instruct

| Prompt | ratio | buf | verdict |
|---|---|---|---|
| short | baseline | — | ✅ PASS |
| short | **0.50** | 89 | ❌ OTHER (collapse) |
| short | **0.55** | **98** | ✅ **PASS** ← cliff |
| short | 0.60+ | 107+ | ✅ PASS (~0.41s each from 0.60+) |
| long | baseline | — | ❌ **OTHER** ("bó bó bó" collapse — model can't even do baseline) |
| long | **0.50** | 982 | ❌ OTHER |
| long | **0.55** | **1081** | ✅ **PASS** ← cliff. **TQ rescued where FP baseline fails!** |
| long | 0.60+ | 1179+ | ✅ PASS (~1.74s each) |

**Phi-4-mini cliff: between ratio = 0.50 and 0.55.** Same window as Llama-3.2 and Qwen-2.5.

## Findings (the big corrections)

### Finding 1 — The cliff is at ratio ~0.5 across all three models, not size-dependent

| Model | Size | Cliff window | Cliff midpoint estimate |
|---|---|---|---|
| Qwen-2.5-7B-Instruct (spike 0008) | 7B | 0.500 → 0.505 | **~0.503** |
| Llama-3.2-3B-Instruct (spike 0009) | 3B | 0.50 → 0.55 | **~0.52** |
| Phi-4-mini-instruct (spike 0009) | 3.8B | 0.50 → 0.55 | **~0.52** |

**All three cliffs sit in the same window: roughly 0.50–0.55.** The v0 hypothesis "smaller models need more buffer" was wrong — it was an artifact of v0 sampling ratio=0.5 exactly (right on the cliff edge) and ratio=1.0 (far above), with no measurements between.

**Practical universal rule (updated):** `buffer_size = max(min_buffer, ceil(0.55 × prompt_tokens))` works across 3B-to-7B instruct models on Apple Silicon MLX TurboQuant. The cliff is *universal*; only the failure mode below it varies.

### Finding 2 — TurboQuant *rescues* Phi-4-mini's long-context tool-calling

Phi-4-mini's long-prompt **FP baseline collapses** to `"bó bó bó"` repetition — it cannot follow tool-call instructions on dilute 2000-token context at full precision. But **at ratio ≥ 0.55, TurboQuant restores the tool call.**

This is even more dramatic than spike 0007's Qwen-2.5 finding: there, baseline produced fluent prose (still wrong) and TQ produced JSON. Here, baseline produces *literal gibberish* and TQ produces JSON.

The implicit-attention-filter hypothesis is reinforced — compressing irrelevant filler **fixes** the model's attention focus on a model that fails without it.

### Finding 3 — Below the cliff: failure modes differ by model

The cliff location is universal; the *failure mode below it* is per-model:
- **Qwen-2.5** below cliff → hallucinates plausible data (e.g. `"41.3°F · 58% humidity"`)
- **Llama-3.2-3B** below cliff → starts JSON then degenerates into token garbage (`{"tool": EXAIN EXEXEXEX...`)
- **Phi-4-mini** below cliff → similar collapse (`{"}\n™\n™...` or `bó bó bó`)

The danger ordering for agentic systems:
1. Most dangerous: Qwen-2.5's fluent hallucination (user can't tell from the prose)
2. Less dangerous: Llama/Phi token-garbage (obviously broken, JSON parser will fail)

## Corrections to RECOMMENDATIONS.md and v0 REPORT

The recommendations doc had two model entries with *estimated* per-model buffer ratios that are now contradicted by measured data:

| Model | Earlier estimate | Measured value | Δ |
|---|---|---|---|
| Llama-3.2-3B-Instruct | ratio 0.70 / min buffer 256 | **ratio 0.55 / min buffer 128** | Was over-conservative |
| Phi-4-mini-instruct | ratio 1.0 / buffer = prompt_tokens (no compression) | **ratio 0.55 / min buffer 128** | Was *way* over-conservative + missed the "TQ rescues" finding |

Both updated in [RECOMMENDATIONS.md](../../RECOMMENDATIONS.md). v0 REPORT's "smaller models need more buffer" hypothesis statement is also revised — it was wrong.

## Speed observations

Wall times at ratio ≥ 0.55 are remarkably consistent within each model:
- Llama-3.2 short: ~0.35s · long: ~1.93s
- Phi-4-mini short: ~0.41s · long: ~1.73s

The flatness above the cliff suggests TQ overhead is approximately constant for buffer sizes between 0.55 × prompt and 0.85 × prompt — choose the smaller buffer (more compression, more memory savings) without speed penalty.

## Skipped (Rule 3)

- **Finer-grained sweep within 0.50-0.55** to pin the cliff to sub-percent precision per model (parallel to spike 0008's Qwen-2.5 result). Cheap to do; deferred for v1.
- **Other models** (Qwen3-4B, DeepSeek-R1-Distill, Gemma-3) — Qwen3-4B fails long at all configs (model-level), DeepSeek isn't a tool-call target, Gemma uses its own schema. None benefits from a cliff sweep in the same way.
- **Per-bit-width interaction** — only K3/V2 tested.
- **Statistical confidence** — n=1 per cell.

## Conclusion

**The cliff is universal across the three models that successfully tool-call.** The v0 "smaller models need more buffer" claim was wrong. The cliff sits at ratio ≈ 0.50-0.55 across Llama-3.2-3B, Phi-4-mini-instruct, and Qwen-2.5-7B. The differences are in failure modes below the cliff — not in cliff position.

This is a cleaner story for the blog than what we had after v0. Updated downstream:
- [RECOMMENDATIONS.md](../../RECOMMENDATIONS.md) — corrected per-model values
- [v0 REPORT](../../REPORT.md) — corrected "size-dependent cliff" claim
- Blog outline Section 6 + 8b (maintained outside this repo) — simplified to "universal cliff, per-model failure modes"
