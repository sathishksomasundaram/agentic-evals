# Spike 0002 — a *fair* token budget for the real model's coding battery

**Question.** Spike 0001 graded the real model at a flat `max_tokens=3072` and 3/8
tasks hit that cap (`finish=length`), so the 6/8 was a budget artifact. With a
**generous, split budget** — does the coding number become fair, and what's left
when truncation is no longer the confound?

**Setup.** Same model, chip, and 8-task battery as spike 0001
(`mlx-community/Qwen3.6-35B-A3B-4bit`, Apple **M4 Max, 36 GB**, `mlx-lm` 0.31.3,
greedy). The one change is the generation loop:
- **Split budgets** — **6144** thinking tokens + a **separate 2048** answer budget
  (total ceiling 8192), instead of one shared 3072. A long reasoning trace can no
  longer starve the answer.
- **Mid-stream `</think>` detection** — answer tokens are only counted *after* the
  model closes its thinking block; each phase is capped independently.
- **Honest finish reason** — `stop` (natural EOS), `answer_cap` (answer ran past
  2048 — a real conciseness failure), or `think_cap` (still thinking at the 8192
  ceiling — never produced an answer). n=1.

## Result

| Task | Diff | Verdict | gen tok | think | ans | finish | Detail |
|---|---|---|---|---|---|---|---|
| is_palindrome | easy | PASS | 935 | 890 | 45 | stop | ok |
| two_sum | easy | PASS | 943 | 873 | 70 | stop | ok |
| lru_cache_class | easy | **PASS** | 3267 | 2908 | 359 | **stop** | ok (was a 3072-cap FAIL in 0001) |
| merge_intervals | hard | PASS | 1612 | 1498 | 114 | stop | ok |
| coin_change | hard | PASS | 1735 | 1622 | 113 | stop | ok |
| min_stack | hard | PASS | **8192** | 0 | 0 | **think_cap** | ok (passed despite runaway thinking) |
| group_anagrams | hard | PASS | 1648 | 1587 | 61 | stop | ok |
| fix_binary_search | hard | **FAIL** | **8192** | 0 | 0 | **think_cap** | timeout>15s (never stopped thinking) |

**Headline:** median decode **106.8 tok/s** (in the article's 90-130 band) · peak
**19.8 GB** (in the article's 19-22 band) · coding **7/8** · **6/8 finished at a clean
`stop`, and all 6 passed.**

## Reading

- **The `lru_cache_class` failure was a genuine budget artifact — and the fix
  confirms it.** In 0001 it hit the 3072 cap mid-answer (SyntaxError on truncated
  code). Here it thinks for 2908 tokens, writes a 359-token answer, finishes at
  `stop`, and **passes**. That is exactly the "raise the budget" win we predicted.
- **6 of 8 tasks now finish thinking and answer naturally (`stop`) — and 6/6 of
  those pass.** That is the honest coding signal for the real model: when it
  converges, the battery passes.
- **Two hard tasks never closed `</think>` even at 8192 tokens (`think_cap`).** This
  is a *different* failure from answer-truncation:
  - `min_stack` — generated the full 8192 tokens of unterminated reasoning, but a
    valid code fence happened to sit inside the draft, so the grader's fallback
    passed it. **A lucky pass, not a clean one.**
  - `fix_binary_search` — also ran the full 8192 tokens without ever answering; the
    best fence inside the runaway reasoning still had the infinite-loop bug →
    timeout. **This is not a budget artifact** — more budget did not rescue it; the
    model fails to *terminate* its reasoning on this task.
- **The bug-fix task is the persistent fragile spot across both models — but the
  failure mode differs.** exp-002's substitute collapsed at the token level (a
  greedy Python→Java switch at `mid = (`). The real thinking model instead falls
  into **non-terminating reasoning** on the same buggy-binary-search repair. Same
  task, two different ways to fail.

## Verdict

**SUPPORTED (n=1) on the real model for throughput + memory; coding is now a fair
7/8 with one real, non-budget failure.** Decode (106.8 tok/s) and peak (19.8 GB)
both stay in-band. The fair-budget re-grade resolved the `lru_cache_class` artifact
(6→7/8) and shows **6/6 clean-`stop` tasks pass**. What remains is a genuine signal,
not a confound: `fix_binary_search` fails by *runaway thinking* (8192 tokens, no
`</think>`), and `min_stack` only passed by luck under the same runaway behavior.

## Caveats

- One model, one chip, n=1; greedy.
- Text-only slice of a multimodal model (vision tower dropped by the loader).
- `think_cap` is recorded as `think_tokens=0 / answer_tokens=0` because the
  `</think>` marker that gates those counters never appeared — read it as "all 8192
  tokens were unterminated thinking" (`gen tok` column confirms 8192).
- **Open question for a follow-up:** is `fix_binary_search`'s non-termination
  intrinsic, or would an even larger budget eventually let it close `</think>`? n=1
  can't tell. But the contrast with `min_stack` (also runaway, only passed by luck)
  suggests the budget is no longer the dominant limiter for these two — convergence
  is.
