# Academic TurboQuant (0xSero/turboquant)

**Status:** Researched 2026-05-26. No code/installs/measurements.

## What it really is

A vLLM-targeted implementation of the **same Google TurboQuant paper** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026). Tracked as a separate page because framework + hardware support are completely different from the MLX community ports, with a different measurement story.

## How it works

Same paper as `turboquant-google.md` — random rotation + Lloyd-Max + QJL projection + group quantization, 3-bit K / 2-bit V. Implementation specifics: monkey-patch into vLLM (`free_kv_cache`, hybrid decode), tested on vLLM 0.18.0, PyTorch 2.10, CUDA 12.8.

## Frameworks

**vLLM only.** No Ollama / llama.cpp / MLX / exllamav2 integration. Installs as `pip install -e .` against vLLM.

## Hardware

**NVIDIA only.** Tested on RTX 5090 (32 GB) and 8× RTX 3090 (24 GB) clusters. CUDA 12.8 required. No Apple Silicon, no AMD, no Intel.

## What's measurable (from the README)

- **KV memory freed:** ~30 GB on Qwen3.5-27B-AWQ
- **Token capacity:** ~2× at same VRAM ceiling
- **Latency overhead:** +5.7% prefill, +3.1% decode on RTX 5090
- **Quality:** cosine similarity 0.940 (2-bit V), 0.997 (4-bit V)

## iris-relevance

**Low for iris's current stack.** iris is Apple Silicon + Ollama. Exercising this implementation requires:
- A separate NVIDIA box
- vLLM stack
- iris reconfigured to use vLLM for at least one tier — a deployment-shape change, not a config flip

Defer unless / until we build the federated multi-hardware leaderboard (ADR 0007's hardware-honesty path). Then this is the canonical NVIDIA-side TurboQuant reference.

## Blockers

1. **No iris hardware match.** Requires NVIDIA GPU (RTX 3090+).
2. **vLLM-only.** iris would need framework changes to consume the result.
3. **MoE limitation** flagged in README — only full-attention layers compressed; linear-attention/Mamba layers skipped.
4. **PyTorch 2.10 + CUDA 12.8** pinning; environment-brittle.

## References

- Repo: [github.com/0xSero/turboquant](https://github.com/0xSero/turboquant)
- Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
