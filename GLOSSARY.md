# Glossary

The handful of terms used throughout this repo's experiments and reports.

**Validation-with-proof**:
The standard everything here is held to. A finding is only published with runnable code, captured data, and a report you can reproduce. Not opinion, not vibes.

**Claim**:
A third-party assertion about model performance, tuning, or capability — from an open-source project, a vendor, a paper, or a video — that we test. Neutral until tested.

**Proof**:
The reproducible artifact set behind a finding: the runner code, the raw results (JSONL/CSV), and the report. Enough for you to re-run it and get the same answer.

**Agentic capability**:
A model behavior tested *in an agent context* — tool-calling, routing, classification, instruction-following, reasoning, RAG — not raw text-generation quality. The unit of what we measure.

**Harness**:
The test apparatus that exercises a model (or a compression technique) and produces measured signal under real autoregressive decoding. Lives in [`src/agentic_evals/`](src/agentic_evals/).

**Spike**:
A small, time-boxed experiment that validates or refutes one hypothesis before any larger build. Each lives in [`docs/experiments/exp-001-mlx-kv-compression-toolcalling/spikes/NNNN-slug/`](docs/experiments/exp-001-mlx-kv-compression-toolcalling/spikes/) with a REPORT.

**System under test (SUT)**:
The thing being validated — a model, a quantization config, a tuning technique, or an inference framework.

**Finding**:
A validated, reproducible conclusion from a spike or benchmark run, graded by confidence (n=1 exploratory → n≥3 publication-grade).

**KV-cache buffer / `buffer_size`**:
When the key/value cache is compressed, the most recent `buffer_size` tokens are kept *uncompressed* for quality. The ratio `buffer_size / prompt_tokens` turns out to be the dominant quality knob — see [exp-001](docs/experiments/exp-001-mlx-kv-compression-toolcalling/README.md).
