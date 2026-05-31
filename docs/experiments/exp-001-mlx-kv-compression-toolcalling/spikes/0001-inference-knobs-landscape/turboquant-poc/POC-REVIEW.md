# Review of the TurboQuant POC

**Files reviewed:** [turboquant.py](turboquant.py), [benchmark.py](benchmark.py), [plot.py](plot.py), [IRIS_INTEGRATION.md](IRIS_INTEGRATION.md), [tradeoff.png](tradeoff.png)

**Source:** User's prior POC from Claude project (provided 2026-05-27 via files.zip).

## What the POC actually is

A clean **NumPy reference implementation** of the TurboQuant algorithm + a **quality-validation harness** measuring the metrics that matter for KV-cache quantization:

- **`turboquant.py`** — the algorithm spec
  - `lloyd_max_gaussian()` — empirical Lloyd-Max codebook for N(0,1)
  - `random_rotation()` — QR-based orthogonal rotation (O(d²); production needs Hadamard for O(d log d))
  - `TurboQuant` class — encode/decode with optional QJL residual sign bits
  - `AsymmetricKV` — asymmetric K/V wrapper (different bit-widths per K and V)

- **`benchmark.py`** — the quality harness, measuring per-config:
  - Needle-in-haystack retrieval % (paper's headline metric)
  - Attention KL divergence vs FP16
  - K cosine similarity
  - Output (context vector) cosine similarity
  - At two head dimensions: 128 (Llama-style) and 256 (Gemma-style)

- **`tradeoff.png`** — the headline chart

## What the POC actually proves (from IRIS_INTEGRATION.md + the chart)

| Configuration | Compression | Needle @ d=128 | Needle @ d=256 |
|---|---|---|---|
| `sym 4b` | ~3.9x | ~80% | ~84.5% |
| `asym K4/V3` | ~4.5x | ~80% | ~84.5% |
| **`asym K4/V2`** | **~5.1x** | **~80%** | **~84.5%** |
| `sym 3b` | ~5.1x | ~66% | ~70% |
| `asym K3/V2` | ~6.1x | ~66% | ~70% |
| `sym 2b` | ~7.5x | ~42% | ~50% |

**Headline result confirmed:** **asymmetric K4/V2 hits 5x compression at the quality of full 4-bit symmetric.** This is the practical sweet spot the IRIS_INTEGRATION.md recommends as default policy.

## What the POC does NOT prove (acknowledged in IRIS_INTEGRATION.md)

- **No real-model token/s** (NumPy, not a Metal kernel)
- **No perplexity on real text** (synthetic outlier KV cache only)
- **No downstream agentic-task quality** — the missing piece our benchmark would actually contribute

## ⚠ Stack inconsistency flagged (Rule 2)

IRIS_INTEGRATION.md says:

> *"iris (Next.js / NestJS) ... iris's TS/Nest layer"*

But the actual iris repo I read is **Python 3.12 + FastAPI + LangChain/LangGraph + Poetry**, with a React frontend. There is no Next.js / NestJS layer in iris that I found. This suggests:
- The POC's strategic narrative was generated against an incorrect/assumed iris stack
- Most likely cause: the POC was generated in an isolated session without access to iris's actual repo
- **The algorithm code itself is unaffected** — it's pure NumPy, framework-agnostic
- **The integration recommendations need re-grounding** against iris's real Python stack before action

## Impact on the agentic-evals plan

This POC materially changes my picture of the TurboQuant axis. Per the original [turboquant-google.md](../turboquant-google.md), I marked TurboQuant-MLX as "high relevance, no canonical port, parallel research thread." That assessment now needs upgrading:

| Before | After |
|---|---|
| No canonical TurboQuant impl to test against | **Have a clean NumPy reference + quality harness** the user authored |
| "Pick one of 7 community MLX ports" was the gating decision | Can validate any community MLX port against this reference's outputs |
| TurboQuant deferred from wedge v0 due to research-stage uncertainty | The quality-vs-compression curve is **already characterized** on synthetic data — what's missing is the downstream agentic-task impact, which is *exactly* what agentic-evals is designed to measure |
| Asymmetric K/V was "a research detail" | **Asymmetric K/V is the recommended default** (K4/V2 with first/last 2 layers in fp16) — should be the configuration the benchmark tests, not symmetric quants |

## Recommended action

This POC is foundational. Two natural integrations:

1. **Use the POC's quality harness as the canonical KV-compression measurement methodology.** Adopt `needle retrieval %`, `attention KL`, `output cosine` as core metrics for any KV-compression axis we add. Don't reinvent.

2. **Reconsider the v0 wedge.** Originally (ADR 0008) v0 = quantization-only across 3 models, with TurboQuant in v1+. Now TurboQuant + asymmetric K/V is a credible **second axis** for v0 because:
   - The algorithm spec is in hand
   - The quality methodology is in hand
   - Adding it doesn't require new research — it requires picking a community MLX port to validate against this reference
   - It's the most novel signal the OSS community lacks: *"asymmetric KV K4/V2 on real agentic workloads at scale"*

   See proposal in this folder + open question for the user.

## Open question for user

Should we **amend ADR 0008** to add asymmetric-KV-TurboQuant as a planned second axis for v0, or keep v0 lean (quant only) and tackle TurboQuant in v1 as originally scoped?
