# Spike 0002 — long-context prefill throughput

**Question.** Spike 0001 measured prefill on 56–88-token prompts, where prefill
tok/s is noisy and not comparable to the article's "1851 tokens/sec prefill"
(almost certainly a long-context number). Can the model approach ~1851 tok/s
prefill when given a long prompt — and how does it scale with length?

**Setup.** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` on Apple **M4 Max,
36 GB**, `mlx-lm` 0.31.3, greedy. A code-shaped filler is grown to hit target
prompt lengths; generation is capped at 8 tokens so the prompt-eval phase
dominates the timing. n=1.

## Result

| Target | Actual prompt tok | Prefill tok/s | Peak GB |
|---|---|---|---|
| 512 | 540 | 291.4 | 17.8 |
| 1024 | 1060 | **1273.4** | 17.9 |
| 2048 | 2100 | 1215.0 | 18.1 |
| 4096 | 4115 | 1189.5 | 18.3 |
| 8192 | 8210 | 1013.4 | 18.7 |

**Best prefill 1273 tok/s = 69% of the article's ~1851 tok/s.**

## Reading

- **Prefill peaks then declines with length.** ~1273 tok/s at ~1k tokens, easing
  to ~1013 tok/s at ~8k — expected, since attention cost grows with context. The
  540-token point is low because the prompt-eval phase is too short to amortize
  fixed overhead (the same noise that made spike 0001's short-prompt prefill
  meaningless).
- **69% of the M5 Max claim, on an M4 Max.** The gap is consistent with one chip
  generation of GPU/bandwidth uplift (M5 Max would need ~45% more prefill
  throughput to hit 1851 — plausible). So the prefill claim is **not refuted**;
  we simply can't reach it on M4.
- **Memory corroborates the article.** Peak rises 17.8 → 18.7 GB from 540 → 8210
  tokens, approaching the article's "19–22 GB **depending on KV cache**" — which
  now reads as a long-context figure, consistent with our measurements.

## Caveats

- **n=1**, one model, one chip, synthetic code-shaped filler.
- Cross-chip: a lower number here does not refute the M5 claim; it bounds M4.
- Prefill tok/s as reported by mlx-lm; no independent wall-clock cross-check.
