# Model distillation

**Status:** Researched 2026-05-26. No code/installs/measurements.

## What it really is

A **training-time** technique — fundamentally different from the other three spike techniques (Turboquant, MTP, DFlash) which are inference-time knobs. Distillation produces a smaller "student" model that mimics a larger "teacher" model's behavior. The student then runs through any standard inference framework.

For this benchmark: **we measure the distilled output model**, not the distillation process. Distillation belongs on a "post-training method" axis next to **base-model selection**, not next to inference knobs.

## Variants (terminology disambiguation)

- **Response / knowledge distillation:** student trained on teacher's logits or generated outputs
- **Feature distillation:** student matches teacher's intermediate hidden states
- **Symbolic / chain-of-thought distillation:** student trained on teacher-generated reasoning traces (what DeepSeek-R1-Distill does)

## OSS distilled models worth knowing (May 2026)

The **DeepSeek-R1-Distill** family is the most prominent open-source distilled lineup:

| Model | Base | Size | Distilled from |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5 | 1.5B | DeepSeek-R1 |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2.5 | 7B | DeepSeek-R1 |
| DeepSeek-R1-Distill-Llama-8B | Llama3 | 8B | DeepSeek-R1 |
| DeepSeek-R1-Distill-Qwen-14B | Qwen2.5 | 14B | DeepSeek-R1 |
| DeepSeek-R1-Distill-Qwen-32B | Qwen2.5 | 32B | DeepSeek-R1 — claims to beat OpenAI-o1-mini |
| DeepSeek-R1-Distill-Llama-70B | Llama3 | 70B | DeepSeek-R1 |

Methodology: DeepSeek generates ~800K high-quality reasoning samples from R1, SFT's the base models on those samples. No RL on the small ones.

Other lineages worth tracking:
- **Phi family** (Microsoft) — curated + synthetic data, debatable as "distillation" strictly speaking
- **Llama Guard** distilled safety classifiers
- **Qwen3** distilled variants via community Open R1 reproductions

## Frameworks / hardware

**Any.** Output is a standard model checkpoint — runs through Ollama, llama.cpp, MLX, vLLM, anywhere the base architecture runs. No framework lock-in. No hardware lock-in.

## What's measurable

This is the key difference from inference-knob techniques. We measure **the distilled model itself**, like any other model:
- Quality across capability categories (the central agentic-capabilities axis of the benchmark)
- Latency / memory at given hardware + quantization
- Direct comparisons like: *DeepSeek-R1-Distill-Qwen-7B vs base Qwen-2.5-7B vs DeepSeek-R1-Distill-Llama-8B* on iris's tool-calling tests, at Apple Silicon, Q5_K_M.

This maps **naturally onto the benchmark's main axis** — "which model is best at capability X" — without needing a separate harness.

## iris-relevance

**High and natural.** iris's Tier 2 question ("what's the best 7-9B model for code/tool-calling on local hardware") is exactly the question DeepSeek-R1-Distill-Qwen-7B is designed to compete on. Including 2-3 distilled models in the first wedge alongside vanilla Qwen-2.5-7B costs almost nothing extra — they all run through the same harness.

## Blockers

**None significant.** Lowest-friction technique of the four to include:
- Models on Ollama / Hugging Face / LM Studio
- No new framework
- No new harness primitives
- Standard model-loading and prompting

## Scope clarification (Rule 2)

The user listed distillation alongside Turboquant / MTP / DFlash. To be explicit: **distillation isn't a "knob" we toggle on a given model.** It's the choice of *which model* to load. So in the benchmark, distillation collapses into the **model selection** axis. There's no "with distillation / without distillation" toggle — there's only "base model X vs distilled-from-Y model Z."

## References

- [DeepSeek-R1 (GitHub)](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-R1-Distill-Qwen-7B (Hugging Face)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [DeepSeek-R1-Distill-Qwen-1.5B (Hugging Face)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [Ollama: deepseek-r1](https://ollama.com/library/deepseek-r1)
- [BentoML: Complete Guide to DeepSeek Models](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond)
