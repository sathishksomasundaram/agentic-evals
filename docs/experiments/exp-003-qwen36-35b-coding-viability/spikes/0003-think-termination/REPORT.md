# Spike 0003 — is the runaway thinking intrinsic, or just a bigger budget?

**Question.** Spike 0002 left two tasks that generated the *entire* 8192-token budget
without ever closing `</think>` (`think_cap`): `min_stack` (passed by luck) and
`fix_binary_search` (failed). n=1 at 8192 couldn't say whether that non-termination is
**intrinsic to the task on this model** or **just a budget that was still too small**.
This spike answers it by doubling the thinking budget.

**Setup.** Same model/chip/greedy as spikes 0001-0002
(`mlx-community/Qwen3.6-35B-A3B-4bit`, M4 Max 36 GB). Re-run **only the two
`think_cap` tasks** at a **16384** thinking budget (2x spike 0002, ~5x spike 0001) +
the same 2048 answer budget. The decisive field is `closed_think` — did `</think>`
ever appear? n=1.

## Result

| Task | Verdict | closed </think> | gen tok | think | ans | finish | peak |
|---|---|---|---|---|---|---|---|
| min_stack | PASS | **False** | **18432** | 0 | 0 | think_cap | 20.0 GB |
| fix_binary_search | **PASS** | **True** | 12785 | **12696** | 89 | **stop** | 20.0 GB |

**Headline:** the two tasks split. **`fix_binary_search` was a budget artifact after
all** — at a bigger budget it closes `</think>` (after ~12.7k thinking tokens), answers
in 89 tokens, finishes at `stop`, and **passes**. **`min_stack` is the genuinely
stubborn one** — it still never closes `</think>` even at the 18,432-token ceiling.

## Reading

- **The bug-fix task is *not* an intrinsic failure on the real model.** This reverses
  spike 0002's tentative read. Given ~12.7k thinking tokens, the model repairs the
  buggy binary search correctly and terminates cleanly. The same task collapsed at the
  token level on the exp-002 substitute (greedy Python→Java switch at `mid = (`) and
  timed out at 8192 tokens here — but with enough room to reason, the real model gets
  it right. **The "replaces Cursor" bar survives the bug-fix task on the real model —
  it just needs a large thinking budget.**
- **`min_stack` is the surprise.** A conceptually *trivial* task ("a stack with O(1)
  `get_min`") is the one that won't stop reasoning — 18,432 tokens, no `</think>`. It
  has passed in every spike, but **always by luck**: a correct code fence sits inside
  the unterminated reasoning and the grader's fallback picks it up. So this is not a
  *correctness* failure (the draft code is right) — it is a **termination** failure.
  Hypothesis (unproven): the "all four operations must run in O(1)" clause sends the
  model into an open-ended self-verification loop it never decides to exit.
- **Difficulty label ≠ thinking cost.** The "easy"/"hard" tags track human-perceived
  complexity, not how long this model reasons. The hard bug-fix terminates at ~12.7k;
  the easy-ish MinStack doesn't terminate at all. Thinking length is its own axis.
- **Peak memory crept to 20.0 GB** at these longer contexts (vs 19.8 at ≤8k) — still
  comfortably inside the article's 19-22 GB band.

## Verdict

**Non-termination is task-specific, not intrinsic.** `fix_binary_search` was a budget
artifact (now a clean `stop` PASS at a 16k budget); `min_stack` is a real
non-termination case that more budget did **not** fix (still `think_cap` at 18,432
tokens), though its code is correct so it passes by fallback. Net for exp-003: with a
sufficiently large thinking budget the battery's *executable* pass rate is effectively
**8/8**, but **clean termination is 7/8** — `min_stack` is the lone task this model
won't stop thinking about.

## Caveats

- One model, one chip, n=1; greedy.
- `min_stack`'s pass is a grader-fallback artifact (correct fence inside unterminated
  reasoning), not a clean completion — counted as PASS but flagged.
- We did not push past 16384 to find `min_stack`'s ceiling (if any). The point was made:
  doubling the budget flipped `fix_binary_search` to a clean pass but left `min_stack`
  unchanged, so the two failures had different causes.
- Text-only slice of a multimodal model (vision tower dropped by the loader).
