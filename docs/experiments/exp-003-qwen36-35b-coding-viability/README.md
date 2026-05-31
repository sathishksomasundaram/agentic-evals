# exp-003 — Qwen3.6-35B-A3B (the article's *actual* model) coding viability

An instance of the [experiment methodology](../../EXPERIMENT-METHODOLOGY.md), and the faithful follow-up to [exp-002](../exp-002-mlx-moe-coding-viability/README.md). It re-runs the same Medium claim — that a single quantized MoE LLM on a 36 GB Apple-Silicon laptop is fast and good enough to replace cloud coding assistants — on the model the article *actually* named.

Source claim: [*"I cancelled ChatGPT, Cursor and Midjourney this week — my MacBook Pro M5 Max quietly replaced all"*](https://medium.com/@shreetejghodekar/i-cancelled-chatgpt-cursor-and-midjourney-this-week-my-macbook-pro-m5-max-quietly-replaced-all-91cbd7f3c78b) (Medium).

## Why this is a separate experiment

exp-002 measured a **substitute** (`Qwen3-Coder-30B-A3B-Instruct-4bit`) under the mistaken belief that the named model didn't exist. It does: [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0), MLX build [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit). It is a *different, newer* model — `model_type: qwen3_5_moe`, a **multimodal, thinking-by-default MoE** (256 experts, hybrid Gated-DeltaNet), reporting SWE-bench Verified 73.4% / LiveCodeBench v6 80.4%. Re-running here (rather than swapping the model inside exp-002) keeps the substitute's numbers intact for comparison and isolates the two harness changes the real model needs.

## Specification

| Field | Value |
|---|---|
| **Thesis** | The article's headline LLM claim — **19-22 GB**, **90-130 tok/s decode**, "competitive enough for day-to-day coding" — holds for the *actual* named model `Qwen3.6-35B-A3B` (4-bit) on a 36 GB Apple-Silicon Mac. |
| **Goal** | Reproduce exp-002's headline check (decode tok/s, peak memory, executable coding pass-rate on the *same* 8-task battery) on the real model, then ACCEPT / REJECT / REFINE — and compare against the exp-002 substitute. |
| **Fixed setup** | Apple **M4 Max, 36 GB** (article: M5 Max, 36 GB) · `mlx-lm` 0.31.3 (its `qwen3_5_moe` loader treats the model as a **text-only LM**, dropping vision weights — no `mlx-vlm` needed) · greedy/deterministic · model = `mlx-community/Qwen3.6-35B-A3B-4bit`. |
| **Variations** | Same coding battery as exp-002 (3 easy + 5 hard, incl. the `fix_binary_search` bug-fix). Start n=1 per cell. Thinking-by-default, so a large token budget (`max_tokens=3072`) and post-`</think>` grading. |
| **Expected range** | If the article holds: decode 90-130 tok/s (likely lower on M4 than M5), peak 19-22 GB. Coding: prior is **≥** the exp-002 substitute (the real model scores higher on public coding benchmarks). The bug-fix collapse from exp-002 may or may not reproduce — thinking may rescue it. |
| **Metrics & outputs** | Per task: `generation_tps`, `generation_tokens`, `peak_mem_gb`, whether a `<think>` block was emitted, and PASS/FAIL by executing the post-think code against asserts. Raw runs land under `spikes/0001-coding-viability/`; analysis in REPORT.md (both written after the first run). |
| **Iteration budget** | ~4 spikes. |
| **Success criteria** | ACCEPT if decode lands in 90-130 tok/s **and** peak ≤ 22 GB **and** the battery mostly passes. |
| **Reject criteria** | REJECT if decode is far below band, memory exceeds 22 GB (won't fit the claim), or the battery mostly fails. |
| **Stop conditions** | Thesis decided · budget exhausted · hard blocker (won't load / OOM on 36 GB). |
| **Verdict** | **CLOSED — ACCEPTED (n=1) for throughput + memory; coding ACCEPTED with a thinking-budget caveat.** Decode **106.8-107.6 tok/s** (in 90-130 band) and peak **19.7-20.0 GB** (in 19-22 band, a *better* fit than the exp-002 substitute's 17.4-18.7 GB). Spikes 0002-0003 peeled apart the spike-0001 truncation confound: `lru_cache_class` and `fix_binary_search` were both **budget artifacts** — given enough thinking budget (the bug-fix needs ~12.7k tokens) they close `</think>` and pass cleanly. Coding is **effectively 8/8 executable, 7/8 by clean termination**. The bug-fix task that collapsed on the exp-002 substitute **works on the real model**. The lone non-clean task is `min_stack`: it never terminates `</think>` even at 18,432 tokens (passes only by grader fallback) — a *termination* quirk, not a correctness failure. Full write-up in [REPORT.md](REPORT.md). |

## The trail, spike by spike

| Spike | Question | What it found |
|---|---|---|
| 0001-coding-viability | Decode tok/s, peak memory, and coding pass-rate on the real model (same battery as exp-002)? | Decode **107.6 tok/s** + peak **19.8 GB** both in-band (memory fits the claim *better* than the substitute). Coding **6/8** but under-measured — 3/8 tasks truncated at the 3072-token cap; all 5 natural-`stop` tasks passed. Throughput + memory **SUPPORTED**; coding deferred to a larger-budget re-run. |
| 0002-token-budget | Re-grade coding with a fair split budget (6144 think + 2048 answer) so truncation is no longer the confound — what's the real coding number? | Coding **7/8** (easy 3/3, hard 4/5); decode **106.8 tok/s** + peak **19.8 GB** still in-band. `lru_cache_class` was a real budget artifact → now passes. **6/8 finish at a clean `stop`, all 6 pass.** Two tasks (`min_stack`, `fix_binary_search`) hit `think_cap` — generated the full 8192 tokens without ever closing `</think>`; `min_stack` passed by luck, `fix_binary_search` failed. |
| 0003-think-termination | Is that runaway thinking intrinsic, or just a still-too-small budget? Re-run the two `think_cap` tasks at a 16384 thinking budget. | The two split. **`fix_binary_search` was a budget artifact** — at 16k it closes `</think>` after ~12.7k thinking tokens and **passes cleanly** (`stop`); the bug-fix that collapsed on the exp-002 substitute works on the real model. **`min_stack` is the stubborn one** — still no `</think>` at the 18,432-token ceiling, passing only by grader fallback. A *termination* quirk on a trivial task, not a correctness failure. |

## Reproduce

```bash
uv run agentic-evals exp-003 coding-viability    # spike 0001 (first run downloads ~18-20 GB)
uv run agentic-evals exp-003 coding-budget       # spike 0002 (fair split-budget re-grade)
uv run agentic-evals exp-003 think-termination   # spike 0003 (16k-budget termination probe)
```

## Honest caveats

- **Different chip.** M4 Max here vs the article's M5 Max — throughput is not directly comparable.
- **Text-only slice of a multimodal model.** `mlx-lm` drops the vision tower; we test only the LM. The article's coding use is text, so this is the right slice, but it is a slice.
- **Thinking-by-default.** We grade the answer after `</think>`. A large `max_tokens` is needed; tasks that exhaust the budget mid-think will FAIL on a truncated answer (recorded via `finish_reason`).
