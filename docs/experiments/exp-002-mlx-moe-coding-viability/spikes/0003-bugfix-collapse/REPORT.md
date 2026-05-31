# Spike 0003 — bug-fix language collapse: replication + cause

**Question.** Spike 0001 saw the model start a correct Python fix on a buggy
`binary_search`, then code-switch to Java mid-token. Two questions: (1) is it
reproducible (n≥3), and (2) what triggers it — the echo-then-fix *structure*, the
binary-search *content*, or the specific combination?

**Setup.** `Qwen3-Coder-30B-A3B-Instruct-4bit`, M4 Max, `mlx-lm` 0.31.3,
`max_tokens=512`. Per variant: 1 greedy run + 4 seeded sampled runs
(temp=0.8, top_p=0.95, `mx.random.seed` ∈ {1,2,3,4}) → **n=5**, fully
reproducible. "Collapse" = the extracted code contains Java/C-family markers.

> Note: this variant's prompt is a lightly shortened rewording of spike 0001's
> `fix_binary_search` (bug description trimmed). The failure reproduces anyway,
> which makes it robust to that wording.

## Result

| Variant | What it isolates | PASS | Collapse |
|---|---|---|---|
| `echo_fix_binsearch` | echo buggy code → fix (the original) | **0/5** | 1/5 |
| `scratch_binsearch` | same algorithm, written from scratch | 4/5 | 0/5 |
| `echo_fix_factorial` | echo-then-fix on a *different* function | 5/5 | 0/5 |

## The mechanism: a deterministic break point, a variable failure surface

Every `echo_fix_binsearch` failure copies the buggy code verbatim, then derails
at the **identical token** — the `(` in `mid = (`:

```
def binary_search(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (        ← derails here, every time
```

What follows the `(` varies by seed — the failure *surface* is stochastic, the
break *point* is not:

| Run | Continuation after `mid = (` | Failure |
|---|---|---|
| greedy | `import math\ndef binary_search…` | SyntaxError |
| seed 1 | `import math…` | SyntaxError |
| seed 2 | `"""` | unterminated triple-quote |
| seed 3 | `x + y) // 2` (undefined vars) | NameError |
| seed 4 | `public` … (Java) | unclosed `(` — **the spike-0001 collapse** |

So the spike-0001 Java code-switch **replicates** (seed 4), but it is one of
several derailments that all share the same trigger token.

## Reading

- **It is not the echo-then-fix structure.** `echo_fix_factorial` (same
  structure, different buggy function) passed 5/5.
- **It is not binary search per se.** `scratch_binsearch` (same algorithm, no
  buggy code to echo) passed 4/5 (the lone miss was an indentation slip, not a
  collapse).
- **It is the specific combination** — echoing *this* buggy binary search and
  being asked to fix it. The model reproduces the buggy line up to `mid = (` and
  then cannot commit to the next token, scattering into Java, stray imports,
  triple-quotes, or undefined variables.
- **Hypothesis (unproven):** the buggy line `mid = (lo + hi) // 2` sits next to
  the actual bug (`lo = mid` instead of `lo = mid + 1`); the model is torn
  between faithfully echoing the buggy arithmetic and "fixing" it, and `mid = (`
  is the maximum-entropy branch point. Confirming this would need logit/entropy
  inspection at that token.

## Confidence

**Confirmed (n=5, seeded/reproducible)** that this prompt fails 0/5 and that the
break point is deterministic. The *cause* (why `mid = (`) is a hypothesis. The
cross-variant contrast (0/5 vs 4/5 vs 5/5) is strong evidence the trigger is the
specific echo-and-fix instance, not the general structure or content.

## Caveats

- One model, one chip, one buggy function; "collapse" is a regex heuristic, not a
  parser. The mechanism hypothesis is not yet instrumented.
- Practical takeaway for the article's "replaces Cursor" claim: this 4-bit Coder
  MoE can *silently and reproducibly* fail to repair certain real buggy code —
  exactly the everyday task it would face — even while writing the same code
  correctly from scratch.
