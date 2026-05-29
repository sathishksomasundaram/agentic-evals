# MTP — Multi-Token Prediction (DeepSeek-style)

**Status:** Researched 2026-05-26. No code/installs/measurements.

## What it really is

An auxiliary training module that lets a transformer predict **multiple future tokens** at each position from the same hidden state, instead of just the next token. Introduced in DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)). MTP heads train jointly with the standard next-token loss; at inference they enable speculative decoding **without** a separate draft model.

## How it works

- Small transformer head(s) on top of the main hidden state predict tokens t+2, t+3, …
- Heads respect causal order (sequential) but are much smaller than the main model
- **Training:** densifies the loss signal → better data efficiency
- **Inference:** MTP heads act as a built-in drafter; the base model verifies the drafts (lossless by construction)
- DeepSeek-V3 reports >80% acceptance rate on MTP1 → ~1.8× decode speedup
- DeepSeek-V3 MTP module: ~14B params on top of the 671B base (11.5B unique + 2.5B shared)

## Frameworks that run it today

| Framework | MTP status |
|---|---|
| **vLLM** | Native MTP-aware speculative decoding for DeepSeek-V3 |
| **SGLang** | Supported; AMD ROCm tutorials reference it |
| **NVIDIA NeMo / Megatron-Bridge** | First-class MTP training + inference |
| **llama.cpp** | **No native MTP**; treats DeepSeek as dense, loses MoE *and* MTP |
| **Ollama** | Inherits llama.cpp limitations — no MTP |
| **MLX** | No native MTP support known yet |

For OSS local-first agents like iris, this is the friction: Ollama / llama.cpp / MLX all currently miss MTP.

## Hardware

Hardware-agnostic in principle; practical support follows the framework matrix. Best supported today: **NVIDIA + vLLM**, with some AMD via SGLang.

## What's measurable

- **Decode throughput** with MTP on vs off (primary signal)
- **Acceptance rate** of MTP drafts (DeepSeek-V3 baseline: ≥80%)
- **Quality preservation** — lossless by construction, but verify
- **End-to-end agentic-loop latency** — each tool-call round-trip benefits proportionally

## iris-relevance

**Medium, indirect.** None of iris's current Tier 1 / 2 / 3 models (`llama3.2:3b`, `qwen2.5-coder:7b`, `qwen3.6:27b`) ship with MTP heads. To exercise MTP for iris, we'd need:
1. Adding a DeepSeek-V3 / DeepSeek-R1 model to iris's tier mix
2. Switching that tier to vLLM or SGLang

That's a deployment-shape ask. More realistic posture: benchmark MTP as a **reference point** ("here's the cutting-edge speedup ceiling") rather than something iris adopts directly.

## Blockers

1. **Model availability.** Need MTP-capable open models. DeepSeek-V3 family is the main option; Qwen3 announced MTP for some checkpoints — verify.
2. **Framework gap.** Ollama / llama.cpp / MLX don't support MTP. Benchmarking MTP requires vLLM = NVIDIA hardware, which the user doesn't have locally.
3. **MTP × MoE coupling.** llama.cpp's lack of MoE compounds the MTP gap.

## References

- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)
- [DeepWiki: MTP in DeepSeek-V3](https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.4-multi-token-prediction-(mtp))
- [NVIDIA Megatron-Bridge MTP docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/multi-token-prediction.html)
- [Accelerating DeepSeek-V3 with MTP in SGLang (AMD ROCm)](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/mtp.html)
