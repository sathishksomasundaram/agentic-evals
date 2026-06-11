# Spike 0002 — Same model, three stacks: the stack wars are overrated at 7B

**Question:** Does serving qwen2.5:7b (IRIS's Tier-2 workhorse) via raw
llama.cpp or MLX beat Ollama on routing accuracy, latency, or decode
throughput?

**Answer: at dense-7B/4-bit scale the three stacks are near-identical.**
Accuracy is a model property (74.0–75.3% everywhere); decode throughput
is hardware-bound (67–70 tok/s everywhere); only request overhead
differs, and llama.cpp wins it modestly.

## Results (routing golden set + 3×256-token decode probes)

| Stack | Routing accuracy | p50 | p95 | Fallback | Decode tok/s |
|---|---|---|---|---|---|
| Ollama 0.30.7 | 75.3% | 557ms | 573ms | 7.8% | 69.7 (server-reported) |
| llama.cpp (same GGUF blob) | 75.3% | **465ms** | 484ms | 9.1% | 69.6 (server-reported) |
| MLX (mlx-community 4bit) | 74.0% | 558ms | 580ms | **2.6%** | 67.0 (wall-clock floor) |

Also banked from spike 1 (granite4, no MLX quant exists for it):
llama.cpp 84.4% @ 293ms vs Ollama 83.8 ± 1.7% @ 322–359ms — same shape.

## Findings

1. **Accuracy travels with the weights, not the server.** Ollama and
   llama.cpp on the *same GGUF file* score identically (75.3%); the MLX
   quant (a different 4-bit scheme) lands within noise at 74.0%. Anyone
   attributing accuracy differences to their serving stack at this scale
   is probably measuring their quant, their template, or their parser.
2. **Decode is hardware-bound at 7B-dense/4-bit:** 67–70 tok/s on all
   three. exp-002/003's 100+ tok/s MLX numbers came from a 30–35B **MoE
   with ~3B active parameters** — that result does not generalize to
   dense models, and this spike is the proof.
3. **llama.cpp has the lowest per-request overhead** (p50 −17% vs both
   others). For a chat turn it's noise; for high-frequency short calls
   (routing, judges, extraction) it compounds.
4. **MLX produced the cleanest JSON** (2.6% fallback vs 7.8–9.1%) —
   plausibly chat-template differences. Small but consistent.
5. **Ops note:** llama.cpp reads Ollama's blob store directly (one model
   store, two servers). MLX requires separate downloads and, in
   `mlx_lm.server` form, one process per model — the weakest multi-tier
   serving story of the three.

## Limitations

- Memory not compared rigorously here: RSS misrepresents Metal unified
  memory; Ollama self-reports 4.6 GB for this model. exp-002/003 measured
  MLX peak properly (native stats) — a rigorous three-way memory probe
  needs per-stack instrumentation and belongs with spike 3.
- One model class (dense 7B @ 4-bit). The MoE/large-tier story (where
  MLX's published numbers shine) is exactly what spike 3 should test —
  qwen3.6:27b GGUF vs an MLX MoE — along with the TurboQuant cliff rule.
- MLX decode is a wall-clock floor (server reports no timings); true
  decode is slightly higher than 67 tok/s, not enough to change the
  conclusion.

## Verdict input (per exp-004 criteria)

For **Tier 2 (dense 7B)**: REJECT the migration — no stack beats Ollama
by ≥10% on any deciding metric. llama.cpp's −17% request overhead is
real but below threshold; adopt only if it wins elsewhere too (it
already lost spike 1 for routing). The open door remains **Tier 3 /
MoE + TurboQuant** (spike 3) and shape-critical workloads.
