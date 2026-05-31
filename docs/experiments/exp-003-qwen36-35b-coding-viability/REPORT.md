# exp-003 — REPORT (closed)

Consolidated findings for [exp-003](README.md), the faithful re-test of the Medium
article's headline LLM claim on the model it *actually* named —
`mlx-community/Qwen3.6-35B-A3B-4bit` — after [exp-002](../exp-002-mlx-moe-coding-viability/README.md)
measured a substitute. See the [spec](README.md) for the thesis and accept/reject
criteria.

## Verdict (n=1, exploratory) — CLOSED

**ACCEPTED for throughput + memory; coding ACCEPTED with a caveat about thinking
budget.** On an Apple **M4 Max, 36 GB** — one chip generation *older* than the
article's M5 Max — the article's *actual* model lands inside the decode band, inside
the memory band (a *better* fit than the substitute), and passes the coding battery
once given a thinking budget large enough to let it finish reasoning.

| Metric | Article (M5 Max, 36 GB) | Measured (M4 Max, 36 GB) | Verdict |
|---|---|---|---|
| Decode throughput | 90–130 tok/s | **106.8–107.6 tok/s** | ✅ in-band |
| Peak memory | 19–22 GB (w/ KV) | **19.7 → 20.0 GB** (rises with context) | ✅ in-band (better fit than the substitute) |
| Day-to-day coding | "competitive enough" | **effectively 8/8** executable; **7/8** by clean termination | ✅ with budget caveat |

Evidence: [spike 0001](spikes/0001-coding-viability/REPORT.md) (decode + memory +
first coding pass) · [spike 0002](spikes/0002-token-budget/REPORT.md) (fair split
thinking/answer budget) · [spike 0003](spikes/0003-think-termination/REPORT.md)
(think-termination probe).

## Findings

1. **The throughput claim holds on the real model — on an older chip.** Median decode
   106.8–107.6 tok/s sits inside the article's 90–130 band, measured on M4 Max, not
   the claimed M5 Max. The claim is plausible and likely conservative for the M5.
2. **The memory claim holds — and fits *better* than the substitute did.** Peak
   19.7 GB short, rising to 20.0 GB at the longest contexts we drove. That lands
   *inside* the article's "19–22 GB depending on KV cache," whereas the exp-002
   substitute sat just below it at 17.4–18.7 GB. The named model is the better match.
3. **Coding is a budget story, not a capability story.** At a flat `max_tokens=3072`
   (spike 0001) the model scored 6/8, but **all** failures were `finish_reason=length`
   — truncation, not wrong answers. A split budget (6144 think + 2048 answer, spike
   0002) and detecting `</think>` mid-stream lifted it to 7/8 with **6/8 finishing at a
   clean `stop`, all 6 passing.** The remaining two confounds were *thinking-budget*
   artifacts, not coding errors.
4. **The bug-fix task that broke the substitute *works* on the real model — it just
   needs room to think.** `fix_binary_search` (the exp-002 collapse case) timed out at
   8192 tokens but, given a 16384 thinking budget (spike 0003), closed `</think>` after
   **~12,696 thinking tokens**, answered in 89, finished at `stop`, and passed with a
   correct fix. The exp-002 substitute failed this same task by a token-level
   Python→Java collapse at the maximum-entropy `mid = (`; the real thinking model
   instead simply needed a long reasoning trace. Different model, different failure
   mode, opposite outcome.
5. **The lone genuine quirk is non-termination on a *trivial* task.** `min_stack` (an
   O(1) stack-with-min) never closed `</think>` even at the 18,432-token ceiling. It
   passed only by grader fallback — a correct code fence sat inside the unterminated
   reasoning. This is a *termination* failure, not a correctness one. Hypothesis
   (unproven): the "all four operations must run in O(1)" clause sends the model into an
   open-ended self-verification loop. **Difficulty label ≠ thinking cost** — the hard
   bug-fix terminated; the easy stack did not.

## What changed versus exp-002 (the substitute)

| | exp-002 substitute (`Qwen3-Coder-30B-A3B-Instruct-4bit`) | exp-003 real model (`Qwen3.6-35B-A3B-4bit`) |
|---|---|---|
| Arch | text-only, non-thinking coding MoE | multimodal **thinking-by-default** MoE (loaded text-only) |
| Decode | 103.5 tok/s | 106.8–107.6 tok/s |
| Peak mem | 17.4–18.7 GB (just *below* the band) | 19.7–20.0 GB (*inside* the band) |
| `fix_binary_search` | **FAIL** — token-level Python→Java collapse | **PASS** — needs ~12.7k thinking tokens |
| Dominant limiter | token-level decode fragility | **thinking-token budget** |

## Open / not yet proven

- **Throughput/memory still n=1** — directional, not confirmed. A few replicates would
  confirm.
- **Single chip** — numbers bound M4 Max; they make M5 "plausible," not proven.
- **`min_stack` non-termination ceiling** — we stopped at 16384 thinking tokens; we did
  not find whether it ever terminates, or why this task specifically loops.
- **"Replaces Cursor" is broader** — multi-file edits, repo-scale context, and
  tool-using agent loops are unmeasured. The self-contained-function battery is a
  floor, not the real bar.
- **Cross-hardware** — Apple Silicon / MLX only. Windows + CUDA (or Windows + Apple-less
  setups) is explicitly **out of scope** and unmeasured (see caveat in the README).
- **Image-gen / transcription** claims from the article remain unverified (out of scope).

## Recommendations (for running this model locally)

1. **Budget thinking generously and separately from the answer.** A flat token cap is
   the wrong knob for a thinking-by-default model — it truncates the answer or strands
   the model mid-reasoning. Split it: a large thinking budget (≥ 8k for non-trivial
   tasks) plus a separate answer budget, and grade only the post-`</think>` text.
2. **Detect `</think>` to bound runaway reasoning.** Some tasks (here, a *trivial* one)
   never terminate. Cap the thinking phase explicitly and have a fallback so a
   non-terminating run fails loudly rather than silently consuming compute.
3. **Grade by execution, not by reading.** `min_stack` "passed" only because a correct
   fence sat inside unterminated reasoning — a reader skimming the output would have
   been misled. Run the code.
4. **Expect 19–20 GB resident on a 36 GB machine** at 4-bit with working KV cache —
   comfortable, with headroom, exactly as the article claimed.

## Status

**Experiment closed.** Thesis decided (ACCEPT for throughput + memory; ACCEPT-with-
caveat for coding). Headline learnings rolled up into the cross-experiment synthesis and
the capstone blog post (maintained in the private strategy repo; a companion blog series
is in the works).
