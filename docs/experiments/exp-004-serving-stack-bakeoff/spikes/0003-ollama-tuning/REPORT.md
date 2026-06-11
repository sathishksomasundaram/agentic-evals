# Spike 0003 — Ollama tuning flags: the "free wins" lose when measured

**Question:** IRIS's `.env.example` suggests `OLLAMA_FLASH_ATTENTION=1`
("~20–40% memory savings on long contexts") and `OLLAMA_KV_CACHE_TYPE=q8_0`
("halve KV cache vs fp16; quality cost negligible") — the nearest Ollama
analog to exp-001's KV recommendation, never actually enabled. Are they
the free wins they look like?

**Answer: no — on this hardware and Ollama 0.30.7 they cost ~6% decode
throughput and ~6% request latency for a memory saving IRIS can't use.
REVERTED.**

## Method

Same-day A/B on the identical daemon binary (brew Ollama 0.30.7, launchd
service), identical model (qwen2.5:7b-instruct, IRIS Tier 2), only the two
env vars toggled via the service plist. Probes: the 77-case routing
workload (accuracy, p50), 3×256-token decode runs (server-reported
eval stats), and model footprint at num_ctx 16384 (`ollama ps`).

## Results

| Metric | Defaults | FA=1 + KV q8_0 | Δ |
|---|---|---|---|
| Decode tok/s (3 runs each) | 69.7 / 69.5–69.8 | 65.3 / 65.3–65.3 | **−6.3%** |
| Routing p50 | 557ms | 589ms | −5.7% |
| Routing accuracy | 75.3% | 72.7% | −2.6 pts (≈2 cases, within run noise) |
| Footprint @ 16k ctx | 5.6 GB | 5.2 GB | −0.4 GB |

## Reading

- The decode regression is the solid signal: byte-stable across all three
  tuned runs vs all three default runs. On this Metal backend, flash
  attention + quantized KV is a *throughput cost*, not a win, at 7B/4-bit.
- The memory saving (−0.4 GB at 16k context) matches the expected ~halving
  of the KV slab — and is worthless to IRIS, whose tiers run 1–8k contexts
  (KV ≤ ~0.2 GB) on a 36 GB machine. These flags pay under memory
  pressure at long contexts; IRIS has neither.
- The accuracy drop is within plausible noise and not the deciding factor.

## Verdict: REJECT — flags reverted, daemon back on defaults.

Third instance of the program's recurring pattern: a plausible,
widely-recommended improvement (LLM router, schema-constrained decoding,
"free" serving flags) loses to the measured incumbent. Config hints in
docs — including our own `.env.example` — are claims, and claims get
graded. The `.env.example` comment should be updated to cite this spike.
