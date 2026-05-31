# Spike 0001 — the article's real model on the exp-002 coding battery

**Question.** On the article's *actual* model `mlx-community/Qwen3.6-35B-A3B-4bit`,
what are the headline decode tok/s, peak memory, and executable coding pass-rate —
and how do they compare to the exp-002 substitute?

**Setup.** `mlx-community/Qwen3.6-35B-A3B-4bit` on Apple **M4 Max, 36 GB**,
`mlx-lm` 0.31.3 (loaded as a **text-only LM** via the `qwen3_5_moe` loader — vision
weights dropped), **greedy**, `max_tokens=3072`. Thinking-by-default: we grade the
code *after* `</think>`. Same 8-task battery as exp-002 (3 easy + 5 hard incl. the
`fix_binary_search` bug-fix). n=1.

## Result

| Task | Diff | Verdict | gen tok | finish | think | Detail |
|---|---|---|---|---|---|---|
| is_palindrome | easy | PASS | 935 | stop | yes | ok |
| two_sum | easy | PASS | 943 | stop | yes | ok |
| lru_cache_class | easy | **FAIL** | **3072** | **length** | yes | SyntaxError (truncated) |
| merge_intervals | hard | PASS | 1612 | stop | yes | ok |
| coin_change | hard | PASS | 1735 | stop | yes | ok |
| min_stack | hard | PASS | **3072** | **length** | no | ok (passed despite truncation) |
| group_anagrams | hard | PASS | 1648 | stop | yes | ok |
| fix_binary_search | hard | **FAIL** | **3072** | **length** | no | timeout>15s (truncated) |

**Headline:** median decode **107.6 tok/s** (in the article's 90-130 band) · peak
**19.8 GB** (in the article's 19-22 band) · coding **6/8** — but see the caveat.

## Reading

- **Both performance claims hold on the *real* model.** Decode 107.6 tok/s is
  in-band on an M4 Max (a generation older than the article's M5). Peak **19.8 GB
  lands squarely inside the article's "19-22 GB depending on KV cache"** — a *better*
  match than the exp-002 substitute (which sat at 17.4-18.7 GB, just below the band).
- **The 6/8 coding grade is under-measured, not a fair score.** All three tasks that
  touched the **3072-token cap** (`finish=length`) are confounds, not clean results:
  - `lru_cache_class` — finished thinking, then the *answer* was truncated mid-code →
    SyntaxError.
  - `fix_binary_search` — hit the cap **while still thinking** (`think=false`, no
    `</think>`), so the grader fell back to a *draft* code fence inside the reasoning,
    which still had the infinite-loop bug → timeout. This is a budget artifact, **not**
    the exp-002 language-collapse (no Python→Java switch occurred).
  - `min_stack` — also truncated (`length`), but a valid code block happened to land
    before the cut, so it passed by luck.
- **Of the 5 tasks that ran to a natural `stop`, all 5 passed.** That is the honest
  signal: when the model is given room to finish, the battery passes.

## Verdict

**SUPPORTED (n=1) on the real model for the throughput + memory claims; coding
deferred.** Decode and peak memory both land in the article's bands — and memory fits
the claim better than the substitute did. The coding pass-rate is **not yet a fair
number** because thinking-by-default blew past `max_tokens=3072` on 3/8 tasks. Spike
0002 should re-grade coding with a much larger budget (and/or a generation cap that
separates the thinking budget from the answer budget).

## Caveats

- One model, one chip, n=1; greedy.
- Text-only slice of a multimodal model (vision tower dropped by the loader).
- The dominant limitation is the **token budget vs. thinking length** — a known trap
  for reasoning models (cf. the R1-Distill issue in exp-001). The fix is mechanical
  (raise the budget), so the coding number here should be read as a floor.
