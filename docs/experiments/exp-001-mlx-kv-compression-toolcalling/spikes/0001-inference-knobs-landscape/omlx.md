# oMLX — Mac-native MLX inference server

**Status:** Researched 2026-05-26. No code/installs/measurements.

⚠ **Naming clarification:** User initially listed oMLX in the same breath as "TurboQuant for MLX." **oMLX is not TurboQuant.** It is a separate inference framework / runtime for Apple Silicon. Tracked here because it is a credible candidate to add alongside `mlx-lm` and Ollama as the iris-side framework matrix.

## What it really is

A **runtime / inference server** for Apple Silicon — "LLM inference, optimized for your Mac, managed directly from your menu bar." Functionally peer to Ollama, but Mac-only and MLX-native. Lives at [omlx.ai](https://omlx.ai) with the source at [github.com/jundot/omlx](https://github.com/jundot/omlx) (15.3k stars, very active — 1.3k commits, latest release `v0.3.12` on 2026-05-27, 74 open PRs).

## Architecture

- Built on **Apple's MLX + mlx-lm + mlx-vlm** (so any mlx-lm model "just works")
- FastAPI server layer; OpenAI-compatible `/v1` endpoints on port 8000
- Web dashboard at `/admin`
- Native macOS app + Homebrew (`brew install omlx`) + source install
- **Tiered KV cache:** hot tier in RAM, cold tier on SSD, using safetensors — its own approach to long-context memory pressure (different from TurboQuant's compression-based approach)
- **Continuous batching** for throughput across concurrent sessions
- Apache 2.0 licensed

## Models supported

- **LLMs:** any mlx-lm checkpoint (Qwen, DeepSeek, Llama, etc.)
- **VLMs:** Qwen3.5, GLM-4V, Pixtral
- **OCR:** DeepSeek-OCR, DOTS-OCR, GLM-OCR
- **Embeddings + rerankers:** BERT, BGE-M3, ModernBERT

## Hardware

Apple Silicon only (M1 / M2 / M3 / M4). macOS 15.0+ (Sequoia). Python 3.10+.

## What it does NOT do

- **Not** TurboQuant or PolarQuant. Does not implement any third-party KV-compression scheme — its tiered-KV approach is its own.
- No NVIDIA / AMD / Intel support.
- Does not include benchmarking tooling.

## iris-relevance

**High as a candidate framework.** iris's stack today is Ollama + LM Studio (+ MLX optionally for Gemma). oMLX is a credible third Mac path:
- **API-compatible with iris's existing client** (OpenAI `/v1` shape) — drop-in test
- **Better long-context behavior** than naive MLX-LM (tiered KV)
- **Continuous batching** matters if iris ever multiplexes user sessions

For the benchmark: **oMLX is a candidate to add to the framework axis** alongside mlx-lm. Doing so would let us answer: "for the same model + quant, does running on oMLX beat running on Ollama on Apple Silicon?" That's directly iris-actionable.

## Blockers

1. **First-time setup overhead** — install, configure model dir, learn the admin dashboard.
2. **No native quant flag** — quant level is determined by the input mlx-lm checkpoint, not a runtime knob.
3. **Tiered KV trade-off** — SSD-tier reads could add latency unpredictably. Worth measuring.
4. **Project maturity** — 15.3k stars and active but still pre-1.0 (`v0.3.x`); breaking changes possible.

## References

- Website: [omlx.ai](https://omlx.ai)
- Repo: [github.com/jundot/omlx](https://github.com/jundot/omlx)
- License: Apache 2.0
