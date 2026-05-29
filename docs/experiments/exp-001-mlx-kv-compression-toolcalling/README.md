# exp-001 — MLX KV-compression × tool-calling

An instance of the [experiment methodology](../../EXPERIMENT-METHODOLOGY.md). Read top to bottom to know how it started, what was fixed vs varied, how each result was derived, and where it ended up.

---

## Specification

| Field | Value |
|---|---|
| **Thesis** | "Aggressive KV-cache compression (TurboQuant) is effectively lossless for small/mid local *instruct* models doing tool-calling." (The ecosystem's implicit claim.) |
| **Goal** | Find *whether and where* compression preserves tool-calling correctness on real models under real decoding — and produce per-model deployable configs. |
| **Fixed setup** | Apple Silicon · `mlx-lm` + `yzamari/mlx-turboquant` · task = emit a correct JSON call to a `web_search(query, time_range)` tool · greedy/deterministic sampling. |
| **Variations** | Models (6: Qwen-2.5-7B, Llama-3.2-3B, Qwen3-4B, DeepSeek-R1-Distill-7B, Phi-4-mini, Gemma-3-4B) · bit-widths (K/V) · `buffer_size` (ratio sweep) · prompt length (short ~189 tok / long ~1966 tok). |
| **Expected range** | Prior claim → near-lossless at 4-bit. We expected, at worst, *graceful* degradation as bits/buffer dropped. |
| **Metrics & outputs** | Verdict per run: PASS / LOST_FORMAT / COLLAPSE / HALLUC / ALT · wall time · peak memory. Results in [`results/`](results/); analysis in [REPORT.md](REPORT.md); deployable configs in [RECOMMENDATIONS.md](RECOMMENDATIONS.md). |
| **Iteration budget** | 9 spikes + 1 multi-model run (the trail below). |
| **Verdict** | **Thesis REJECTED in its naive form, REFINED into a conditional one.** Compression is *not* uniformly lossless — there is a sharp cliff at `buffer_ratio ≈ 0.5` below which models hallucinate. Above `≈ 0.55` it is effectively lossless (sometimes *better* than baseline). |
| **Headline finding** | The **buffer-size hallucination cliff**. |

## The method

Claim → spike → proof. Each hypothesis got a small, time-boxed spike with its own REPORT — including the times a conclusion was wrong and what corrected it. Nothing asserted without runnable code.

## The trail, spike by spike

| Spike | Question | What it found |
|---|---|---|
| [0001](spikes/0001-inference-knobs-landscape/) | What compression techniques even exist? | Landscape: TurboQuant, MTP, DFlash, distillation. Picked TurboQuant KV compression on MLX. |
| [0002](spikes/0002-mini-toolcall-run/REPORT.md) | Does the apparatus produce signal? | DeepSeek-R1-Distill is wrong for tool-calling (it "thinks" instead of emitting JSON); one port collapsed the model to `"!!!!"`. |
| [0003](spikes/0003-bit-width-sweep/REPORT.md) | Where's the bit-width cliff? | All bit-widths failed *identically* — a bug signature, not a quality curve. Baseline = perfect tool call. |
| [0004](spikes/0004-port-vs-poc-roundtrip/REPORT.md) | Algorithm or integration broken? | Algorithm matches a NumPy reference on *synthetic* data — but on **real Qwen-2.5 K/V**, reconstruction cosine drops 0.98 → 0.33. The "data-oblivious" assumption fails on real distributions. |
| [0005](spikes/0005-yzamari-port/REPORT.md) | Does a different port behave differently? | **Yes.** `yzamari`'s `buffer_size` workaround makes it work — byte-identical to baseline at a 12% *speedup*. Buffer dominates bit-width. |
| [0006](spikes/0006-buffer-cliff/REPORT.md) | Where's the buffer cliff? | A sharp cliff at ~50% of prompt length. The long-context *baseline* fails too — and TurboQuant sometimes *beats* it. |
| [0007](spikes/0007-buffer-cliff-replication/REPORT.md) | Is "compression beats baseline" real? | **Replicated, n=3, deterministic:** baseline 0/3, TurboQuant 3/3, 10% faster. |
| [0008](spikes/0008-cliff-fine-sweep/REPORT.md) | Exactly where is the cliff? | Pinned to `buffer_ratio = 0.5051 ± 0.0026`. **Below it, the model hallucinates plausible data** ("41.3°F · 58% humidity") instead of calling the tool. |
| [0009](spikes/0009-per-model-cliff/REPORT.md) | Does the cliff move with model size? | **No — universal at ~0.50-0.55** across three models. Phi-4-mini, which collapses at full precision on long context, is *rescued* by TurboQuant. |

## Results & where it ended up

Full leaderboard across 6 models → [REPORT.md](REPORT.md). Per-model verdicts + copy-paste configs → [RECOMMENDATIONS.md](RECOMMENDATIONS.md). The practical rule:

> **`buffer_size = max(128, ceil(0.55 × prompt_tokens))`** for tool-calling on these models. Below ~0.5 ratio, expect silent failure. Verify on your own workload.

## Reproduce

The headline in one command:

```bash
uv run agentic-evals exp-001 cliff-fine-sweep   # correct tool call → hallucination as buffer drops
```

Every finding maps to a runner (`uv run agentic-evals --list` shows them all). With greedy/deterministic sampling these reproduce byte-for-byte:

| Finding / output | Spike | Command |
|---|---|---|
| The sharp cliff + hallucination below it | [0008](spikes/0008-cliff-fine-sweep/REPORT.md) | `uv run agentic-evals exp-001 cliff-fine-sweep` |
| Buffer ratio dominates bit-width | [0005](spikes/0005-yzamari-port/REPORT.md) | `uv run agentic-evals exp-001 yzamari-sweep` |
| Compression beats baseline (n=3, deterministic) | [0007](spikes/0007-buffer-cliff-replication/REPORT.md) | `uv run agentic-evals exp-001 replication-buf-cliff` |
| Per-model cliff localization | [0009](spikes/0009-per-model-cliff/REPORT.md) | `uv run agentic-evals exp-001 cliff-per-model --model <hf-id>` |
| Full v0 leaderboard (one model per process) | [REPORT.md](REPORT.md) | `uv run agentic-evals exp-001 v0-runner --model <hf-id>` |
| Rebuild the charts | — | `uv run agentic-evals exp-001 charts` |

The full evidence base (leaderboard + all 7 findings + charts) is in [REPORT.md](REPORT.md); deployable per-model configs are in [RECOMMENDATIONS.md](RECOMMENDATIONS.md).

## Honest caveats

- **Mostly n=1.** Only spike 0007 is replicated (n=3). Single-run cells are exploratory.
- **One task** (tool-calling), **one prompt structure**, **one hardware class** (Apple Silicon), **one bit-width pair** (K3/V2 for most sweeps).
- **Gemma-3 uses its own tool-call schema** — tested against a foreign schema, so its "failures" are partly a mismatch.
- **Upstream port quirk:** `yzamari/mlx-turboquant` can crash when multiple models load in one process; we run each model in its own process.

These gaps are the roadmap. Closing one — or finding a result that doesn't replicate — is the most valuable contribution.
