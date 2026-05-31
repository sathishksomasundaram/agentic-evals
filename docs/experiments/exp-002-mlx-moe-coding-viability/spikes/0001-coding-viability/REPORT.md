# Spike 0001 — coding viability + headline throughput/memory

**Question.** Does the apparatus work, and what are the headline decode/prefill
throughput, peak memory, and coding pass-rate for the closest real MoE model?

**Setup.** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` on Apple **M4 Max,
36 GB**, `mlx-lm` 0.31.3, greedy sampling, `max_tokens=1024`. Coding grade =
execute the model's extracted code against asserts (PASS only if asserts hold).
Throughput/memory come from mlx-lm's own per-generation stats. n=1.

> **Correction.** This is a *substitute*, not the article's actual model. The named
> `Qwen/Qwen3.6-35B-A3B` is real (we initially thought it wasn't) and is a newer
> multimodal/thinking MoE, so a faithful re-test is deferred to a separate
> experiment (exp-003). See the [experiment REPORT](../../REPORT.md), Finding 6.

**Battery.** Hardened from the initial 3-task smoke to **8 tasks**: 3 easy
(`is_palindrome`, `two_sum`, `LRUCache`) + 5 hard (`merge_intervals`,
`coin_change`, `MinStack`, `group_anagrams`, and a `fix_binary_search` *bug-fix*
task). All 8 asserts were validated against reference solutions first.

## Result

| Task | Diff | Verdict | Decode tok/s | Peak GB |
|---|---|---|---|---|
| is_palindrome | easy | PASS | 104.4 | 17.3 |
| two_sum | easy | PASS | 102.9 | 17.3 |
| lru_cache_class | easy | PASS | 102.7 | 17.3 |
| merge_intervals | hard | PASS | 103.4 | 17.3 |
| coin_change | hard | PASS | 103.8 | 17.3 |
| min_stack | hard | PASS | 103.7 | 17.3 |
| group_anagrams | hard | PASS | 104.5 | 17.3 |
| fix_binary_search | hard | **FAIL** | 102.1 | 17.4 |

**Median decode 103.5 tok/s · peak 17.4 GB · coding 7/8 PASS (easy 3/3, hard 4/5).**

## vs the article's claims (M5 Max, 36 GB)

| Metric | Article | Measured (M4 Max) | Verdict |
|---|---|---|---|
| Decode | 90–130 tok/s | **103.5 tok/s** | **in-band** |
| Peak memory | 19–22 GB | **17.4 GB** | **below** (lighter) |
| Coding quality | "competitive enough" | **7/8 PASS** | mostly supports |
| Prefill | ~1851 tok/s | noisy on short prompts → see [spike 0002](../0002-prefill-scaling/REPORT.md) | deferred |

## The one failure is a real finding: a mid-stream language collapse

`fix_binary_search` failed with a `SyntaxError`. Inspecting the raw reply, the
model began a **correct Python** fix:

```
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo +
```

…then catastrophically **code-switched to Java mid-token**:

```
        mid = (lo +public static int binarySearch(int[] nums, int target) {
```

This is reproducible under greedy decoding — not a grading artifact (the grader
correctly caught it). The bug-fix task (echoing buggy code, then producing a fix)
appears to be where the 4-bit Coder MoE is most fragile. Worth a focused replication.

## Reading

- **Decode lands in the claimed band on an *older* chip** (M4, not M5) → the
  decode claim is plausible, likely conservative, for the M5.
- **Memory is *better* than claimed** (17.4 vs 19–22 GB).
- **Quality holds on 7/8**, including 4/5 hard algorithmic tasks — but the bug-fix
  task exposed a real failure mode.

## Caveats / what's not proven

- **n=1**, eight tasks, one model, one chip.
- Prefill claim handled separately in spike 0002.
- "Replaces Cursor/ChatGPT" is broader than self-contained functions — multi-file,
  long-context, tool-using workflows are unmeasured.
