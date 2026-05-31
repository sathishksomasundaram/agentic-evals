# Google TurboQuant + MLX ports

**Status:** Researched 2026-05-26. Updated 2026-05-27 with user-provided POC (see [turboquant-poc/POC-REVIEW.md](turboquant-poc/POC-REVIEW.md)).

## What it really is

A **KV-cache compression** technique from Google (paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026). The name "TurboQuant" is sometimes used interchangeably with "PolarQuant" in community ports — needs verifying against the paper directly.

Target: shrink the KV cache (which grows linearly with context length) so longer contexts fit in the same VRAM / unified memory.

## How it works

Combines: **random orthogonal rotation** of K/V (flattens the distribution before quantization), **Lloyd-Max scalar quantization** on the rotated values, **QJL projection** for residual sign bits, **group quantization**. Produces an *unbiased* estimator for attention-score inner products at low bit-widths. Headline depth: **3-bit keys / 2-bit values**.

## Frameworks that run it today (MLX community ports)

| Repo | Notes |
|---|---|
| [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) | PoC; 5.5x KV-cache compression claim |
| [yzamari/mlx-turboquant](https://github.com/yzamari/mlx-turboquant) | Prince Canuma involvement; 6/6 NIAH pass at 64K; zero-decompression Metal kernels |
| [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) | Frames it as "PolarQuant"; 3-5x compression, near-lossless |
| [helgklaizar/turboquant-mlx](https://github.com/helgklaizar/turboquant-mlx) | Production-oriented; ≤5x memory reduction |
| [matt-k-wong/turboquant-mlx-full](https://github.com/matt-k-wong/turboquant-mlx-full) | Covers **both weights and KV cache** — first such full impl in MLX |
| [lingengyuan/qjl-mlx](https://github.com/lingengyuan/qjl-mlx) | First MLX-native QJL + TurboQuant |
| [Incept5/gemma4-benchmark](https://github.com/Incept5/gemma4-benchmark) | Gemma 4 + Qwen 3.5 on Apple Silicon using TurboQuant KV |

vLLM / NVIDIA path: see `turboquant-academic.md`.

## Hardware

- **Apple Silicon (M-series):** yes via MLX ports above — iris-relevant path.
- **NVIDIA CUDA:** see academic page.
- **AMD:** no known support.

## What's measurable

- **Memory:** KV-cache footprint at given context length (towardsai: 13.3 GB → 4.9 GB on Qwen3.5-35B at 64K)
- **Latency:** TTFT, decode tokens/sec (towardsai cites ~98% of FP16 decode)
- **Quality:** attention-score cosine similarity vs FP16; downstream task quality (NIAH, agentic)
- **Max context** before OOM at given hardware

## iris-relevance

**High.** iris runs Ollama on Apple Silicon — no TurboQuant today. Benchmarking iris's Tier 2 (`qwen2.5-coder:7b`) or Tier 3 (`qwen3.6:27b`) on MLX-with-TurboQuant vs current Ollama yields:
- Memory headroom for longer context windows
- Quality preservation under aggressive KV compression
- Decision input: does iris add MLX+TurboQuant as a long-context deployment option?

## Blockers (updated)

1. ~~Multiple competing MLX ports, none canonical~~ — **partially resolved**: user has a NumPy reference implementation + quality harness in [turboquant-poc/](turboquant-poc/) that any community MLX port can be validated against. Still need to pick a port to evaluate.
2. **TurboQuant ↔ PolarQuant naming overlap** — verify against the paper.
3. ~~No standard benchmark harness~~ — **resolved**: the POC harness (`needle retrieval %`, `attention KL`, `output cosine` at d=128 and d=256, symmetric + asymmetric K/V) is the canonical measurement methodology. Adopt directly.
4. **Apple Silicon hardware variance** — M1 / M2 / M3 / M4 (and Pro / Max / Ultra) have different memory bandwidth + unified-memory sizes. Numbers don't transfer cleanly across chips.
5. **No real-model validation yet** — POC measures on synthetic outlier KV. Confirmed-as-missing: downstream impact on real models running real agentic workloads, which is the gap agentic-evals fills.

## References

- Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [towardsai article — Run 32b Models on Your Mac with 5x Less Memory](https://pub.towardsai.net/run-32b-models-on-your-mac-with-5x-less-memory-googles-turboquant-hits-apple-silicon-ad59f9292d1c)
- MLX repos linked in the table above
