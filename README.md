# agentic-evals

**Reproducible proofs for the agentic capabilities of local, open-weight AI models.**

Free, local, secure AI is powerful — and the open-source ecosystem around it is vast, scattered, and full of unverified claims. This repo is where those claims get *tested*: on real models, under real decoding, with code you can run yourself.

Every finding here ships with the exact code that produced it. Clone it, run one command, reproduce the result on your own hardware. **Proof, or it didn't happen.**

> Built by Sathish, who's building a local-first AI assistant — coming soon — on the foundations tested here. A companion blog series is in the works.

## What's inside

- **[docs/EXPERIMENT-METHODOLOGY.md](docs/EXPERIMENT-METHODOLOGY.md)** — **start here.** How every experiment in this repo is run: the claim → spike → proof method and the spec template (thesis, fixed setup, variations, expected range, metrics, iteration budget, success/reject criteria).
- **[docs/experiments/exp-001-mlx-kv-compression-toolcalling/](docs/experiments/exp-001-mlx-kv-compression-toolcalling/README.md)** — the first full experiment: the methodology filled in, then the full narrative — how it started, the baselines, the method, how each result was derived, and where it ended up.
- **[docs/experiments/exp-001-mlx-kv-compression-toolcalling/spikes/](docs/experiments/exp-001-mlx-kv-compression-toolcalling/spikes/)** — the experiment trail. Each spike is a small, focused test with a REPORT documenting what it asked and what it found (including the times the conclusions were wrong and got corrected).
- **[docs/experiments/exp-001-mlx-kv-compression-toolcalling/](docs/experiments/exp-001-mlx-kv-compression-toolcalling/)** — the first multi-model leaderboard: [REPORT](docs/experiments/exp-001-mlx-kv-compression-toolcalling/REPORT.md), [RECOMMENDATIONS](docs/experiments/exp-001-mlx-kv-compression-toolcalling/RECOMMENDATIONS.md) (per-model verdicts + copy-paste configs), [charts](docs/experiments/exp-001-mlx-kv-compression-toolcalling/results/charts/), and raw data (`raw-results.jsonl`, `leaderboard.csv`).
- **[docs/experiments/](docs/experiments/)** — three closed experiments: **exp-001** (KV-cache compression × tool-calling — the hallucination cliff), **exp-002** (laptop MoE coding viability, substitute model — a deterministic bug-fix collapse), and **exp-003** (the same claim on the article's *real* thinking MoE — coding is a thinking-budget story).
- **[docs/experiments/exp-006-deterministic-harness-spine/](docs/experiments/exp-006-deterministic-harness-spine/README.md)** — the harness turn: two campaigns (a tuning-lever sweep and a judges-on red-team) showing that reliability and safety come from an agent's *deterministic spine*, not the model or its knobs. Ships the **Harness Probe Kit** — an agent-agnostic, stdlib-only behavioural tester you point at *any* local agent (`cd probe-kit && bash demo.sh`). Companion to the *"The Harness Has a Spine"* blog post.
- **[src/agentic_evals/](src/agentic_evals/)** — the code: the generic `harness/` core, per-experiment runners under `experiments/`, and the `agentic-evals` CLI that dispatches them (`uv run agentic-evals --list`).
- **[GLOSSARY.md](GLOSSARY.md)** — the handful of terms used throughout.

## Quickstart

Requires Apple Silicon (current experiments use Apple's MLX), Python 3.12, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sathishksomasundaram/agentic-evals
cd agentic-evals
uv sync
uv run pre-commit install   # optional, for contributors
```

## Reproduce the headline finding

The **buffer-size cliff** — where a local agent stops using its tools and starts hallucinating instead:

```bash
uv run agentic-evals --list                    # see every experiment + runner
uv run agentic-evals exp-001 cliff-fine-sweep   # the headline finding
```

Watch the same model flip from a correct tool call to inventing fake data as `buffer_size` crosses ~50% of the prompt length. Full explanation in the [exp-001 README](docs/experiments/exp-001-mlx-kv-compression-toolcalling/README.md).

## Status

**v0 — early, honest, evolving.** Most cells are single-run (n=1); the one replicated finding (n=3) is flagged as such in its report. Apple Silicon + MLX + tool-calling only, so far. The point isn't a finished benchmark — it's a *reproducible* one. If a result doesn't hold on your hardware, open a Discussion — that's the most useful contribution you can make.

## Contributing

Results from other models, hardware, frameworks, and tasks are welcome. The runners are parameterized and outputs are structured (JSONL/CSV). Found a case where a finding doesn't replicate? Especially welcome — that's how we both get to the truth.

## License

[Apache License 2.0](LICENSE). Use it, fork it, build on it.
