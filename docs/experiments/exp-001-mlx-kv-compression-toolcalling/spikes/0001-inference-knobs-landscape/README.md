# Spike 0001 — Inference-Knobs Landscape

- **Status:** Open / research in progress
- **Started:** 2026-05-26
- **Type:** Research only — no code, no installs, no measurements
- **Why:** Per Rule 10. Before we pick a wedge or design any harness, we need ground truth on the optimization/quantization/decoding techniques the user wants the benchmark to eventually cover. The full list explodes the combinatorial matrix; the spike clarifies which techniques are real, runnable, and iris-relevant *today*.

## Goal

Produce **one short page per technique** that answers:

1. **What it really is** (one paragraph in plain English)
2. **How it works** (1-2 paragraphs, technical but not exhaustive)
3. **Frameworks that run it today** (concrete: Ollama / llama.cpp / MLX / vLLM / SGLang / exllamav2 / Transformers / other)
4. **Hardware support** (NVIDIA / Apple Silicon / AMD / Intel — be specific about gaps)
5. **What's measurable** (latency, throughput, memory, quality — and at what cost)
6. **iris-relevance** (does iris's current stack support it? what would change?)
7. **Blockers** (anything that makes benchmarking this hard or impossible today)
8. **References** (papers, repos, blog posts, with links)

## Out of scope for this spike

- No model downloads, no installs, no measurements
- No harness design
- No dashboard work
- No code samples beyond illustrative snippets

## Techniques to cover (in this folder)

| # | Technique | File | Status |
|---|---|---|---|
| 1 | Google's TurboQuant (+ MLX ports) | [turboquant-google.md](turboquant-google.md) | Done |
| 2 | Academic TurboQuant (`0xSero/turboquant`, vLLM, ICLR 2026) | [turboquant-academic.md](turboquant-academic.md) | Done |
| 3 | MTP — Multi-Token Prediction (DeepSeek-style) | [mtp.md](mtp.md) | Done |
| 4 | DFlash (block-diffusion speculative decoding) | [dflash.md](dflash.md) | Done |
| 5 | Model distillation | [distillation.md](distillation.md) | Done |
| 6 | oMLX (Mac MLX inference server — added 2026-05-26 after user research) | [omlx.md](omlx.md) | Done — framework candidate, not a quantization technique |
| 7 | User-provided TurboQuant POC (NumPy reference + quality harness) | [turboquant-poc/POC-REVIEW.md](turboquant-poc/POC-REVIEW.md) | Reviewed 2026-05-27 — materially upgrades the TurboQuant axis; flagged iris-stack inconsistency in POC's integration brief |

## Exit criteria

When all pages are filled in, we revisit the wedge question with grounded info: pick the axis, the capability, the models, the frameworks, the hardware target. **Output of the spike → ADR 0008 selecting the wedge.**
