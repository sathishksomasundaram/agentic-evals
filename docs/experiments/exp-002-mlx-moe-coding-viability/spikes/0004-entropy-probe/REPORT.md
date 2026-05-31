# Spike 0004 — entropy probe: why the bug-fix collapses at `mid = (`

**Question.** Spike 0003 confirmed (n=5) that the echo-and-fix-this-buggy-binary-search
prompt derails at the *identical* token — the open paren in `mid = (` — and
*hypothesized* that this is a **maximum-entropy branch point**. This spike tests that
hypothesis directly by reading the model's own next-token distribution.

**Setup.** Same substitute model `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`,
M4 Max, `mlx-lm` 0.31.3, **greedy** (deterministic). `mlx-lm`'s
`GenerationResponse.logprobs` is the full log-probability vector over the vocabulary
at each step, so for every generated token we compute the Shannon entropy (bits) of
that decision and record the top-5 candidates. We locate the decision *immediately
after* `mid = (` and compare its entropy and rank across three variants (the same
controls as spike 0003).

## Result — the hypothesis holds, sharply

| Variant | Decision after `mid = (` | Entropy | Rank in generation | Outcome |
|---|---|---|---|---|
| `echo_fix_binsearch` | scatter | **6.447 bits** | **1 / 113** (the single highest) | collapse |
| `scratch_binsearch` | `left` (p≈1.0) | **0.001 bits** | 17 / 89 | passes |
| `echo_fix_factorial` | *(never emits `mid = (`)* | — (gen max 0.552) | — | passes |

In `echo_fix_binsearch` the post-`(` decision is **the maximum-entropy token of the
entire 113-token generation** — 6.447 bits (≈ 87 effectively-equiprobable tokens) —
against a **median of 0.000 bits**. The model is essentially deterministic everywhere
*except* this one token. The window makes the spike unmistakable:

```
step 28  ' mid'    0.002 bits
step 29  ' ='      0.000
step 30  ' ('      0.049
step 31  'import'  6.447   ← the entire generation's entropy is concentrated here
step 32  ' math'   0.141
step 33  '\n'      0.099
step 34  'def'     2.074   ← second, smaller derail
```

## What the model is actually torn between (a refinement of spike 0003)

Top-5 candidates after `mid = (` in `echo_fix_binsearch`:

| prob | token |
|---|---|
| 0.153 | `import` |
| 0.120 | `public` |
| 0.072 | `#` |
| 0.064 | `#!/` |
| 0.056 | `from` |

This refines — and partly **corrects** — spike 0003's guess. The mass is *not* split
between "faithfully echo the buggy `lo + hi`" and "fix the arithmetic"; the correct
continuation (`lo`) is not even in the top 5. Instead the probability collapses onto
**"start a new top-level construct" tokens spanning several languages**: `import` /
`from` (Python), `public` (Java), `#` / `#!/` (comment / shebang). Greedy takes
`import` → `import math` (the SyntaxError path); spike-0003 seed 4 took `public` → the
Java collapse. **The stochastic failure surface from spike 0003 is literally this
top-k distribution being sampled.**

So the precise mechanism: having faithfully echoed up to the open paren, the model
**loses the thread of "I am inside an arithmetic expression"** and its distribution
defaults to high-level boilerplate openers — at exactly one token, with near-uniform
uncertainty.

## The control is the clincher

`scratch_binsearch` writes the **identical text** `mid = (` (steps 32–34, all
0.000-bit confident) and then continues with `left` at probability **1.0000** —
entropy 0.001 bits, and the whole generation never exceeds 0.997 bits. The same
surface tokens are near-deterministic when *writing from scratch* and maximum-entropy
when *echoing-and-fixing*. The trigger is the echo-and-fix **context**, not the tokens.

`echo_fix_factorial` never emits `mid = (` and peaks at 0.552 bits (the closing
fence) — confident throughout, consistent with its 5/5 pass.

## Confidence

**Confirmed (deterministic, greedy).** Spike 0003's "maximum-entropy branch point"
hypothesis is no longer a hypothesis: the break point is *the* highest-entropy
decision in the generation (rank 1/113, 6.447 bits vs 0.000 median), the same text
from scratch is 0.001 bits, and the competing candidates are exactly the
cross-language derailments observed in spike 0003. The one refinement: the model is
not torn between echoing vs. fixing the arithmetic — at the open paren it collapses
toward new-construct openers.

## Caveats

- One model, one chip, one buggy function; entropy is read from the 4-bit quantized
  logits (quantization itself may flatten the distribution at this token — untested).
- Greedy only. Entropy is a property of the distribution, not of sampling, so n>1
  seeds would not change it; but a second model / a non-quantized run would test
  whether the spike is intrinsic or a 4-bit artifact.
