# DFlash — block-diffusion speculative decoding

**Status:** Researched 2026-05-26. No code/installs/measurements.

## What it really is

A **lightweight block-diffusion model** designed to serve as the **draft model** in speculative decoding. Produces high-quality multi-token drafts cheaply, so the larger target model verifies many tokens at once → end-to-end inference speedup.

Differs from MTP: MTP embeds the drafter inside the same model (auxiliary heads). DFlash is an *external* draft model that pairs with any compatible target.

## How it works (README detail is thin)

- **Block diffusion:** instead of left-to-right next-token drafting, generate a block of tokens via diffusion over the block — claimed to produce more coherent multi-token drafts.
- Pre-trained DFlash draft models published for: Gemma 4, MiniMax (M2.5, M2.7), Kimi (K2.5, K2.6), Qwen (3.5, 3.6, 3 series), LLaMA-3.1, GPT-OSS.
- Inference: load DFlash draft alongside the target; framework wires the verification.

Block-diffusion mechanics aren't fully described in README — likely needs the underlying paper for the math (TODO if we adopt).

## Frameworks

| Framework | Status |
|---|---|
| **vLLM** (≥ 0.20.1) | Supported via API flag |
| **SGLang** | Supported |
| **Transformers** | Limited — only Qwen3 + LLaMA-3.1 |
| **MLX** | **Supported** — Apple Silicon path exists |

Of the four spike techniques, this is the only one with both **MLX support** *and* a **published draft-model zoo** the user can pick up today.

## Hardware

- **NVIDIA** via vLLM / SGLang
- **Apple Silicon** via MLX
- **AMD** not mentioned

## What's measurable

- **Decode throughput / tokens-per-sec** with DFlash-as-drafter vs no spec-decode baseline
- **Acceptance rate** of DFlash drafts
- **End-to-end latency** on agentic loops (tool-calling round-trips)
- **Quality** — speculative decoding lossless by construction; verify exact-match output
- **Net efficiency** = target-speedup − drafter cost (memory + compute)

⚠ **The README publishes no speedup numbers.** Referenced benchmarks (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench) are quality benchmarks for the target model, not DFlash throughput. We'd be measuring DFlash speedup as fresh data — a feature (novel result) and a risk (no oracle to validate harness against).

## iris-relevance

**High and direct.** Of the four techniques, this is the most plausibly adoptable by iris:
- MLX backend = user's Apple Silicon
- Draft models published for Qwen 3.5 / 3.6 — same family as iris's Tier 2 / Tier 3
- Speculative decoding is framework-level, doesn't touch iris's agentic core

A wedge comparing "Qwen 3.5 on MLX baseline" vs "Qwen 3.5 on MLX + DFlash drafter" for iris's tool-calling tests would produce numbers the OSS community doesn't have yet.

## Blockers

1. **No published speedup baseline** — first to measure → first to report.
2. **Draft-model footprint** — disk + memory on top of the target model.
3. **Quality testing** — lossless-by-construction in theory; implementation edge cases can leak. Need exact-match output checks.
4. **Block-diffusion mechanics underspecified** — for fair measurement, may need the underlying paper.

## References

- Repo: [github.com/z-lab/dflash](https://github.com/z-lab/dflash)
- Underlying paper: not located yet — TODO if we proceed
