# exp-002 — REPORT

Consolidated findings for [exp-002](README.md). See the [spec](README.md) for the
thesis and accept/reject criteria.

## Verdict so far (n=1, exploratory)

**The article's headline LLM claim is largely SUPPORTED on real hardware — but
measured on a *substitute* model.** On an M4 Max — one chip generation *older* than
the article's M5 Max — a real MoE lands in the claimed decode band, stays under the
claimed memory band, and passes 7/8 of a hardened coding battery. The one prefill
shortfall and the one coding failure are both explained, not hand-waved.

> **Correction (see Finding 6).** The article's actual model, `Qwen/Qwen3.6-35B-A3B`,
> **does exist** (an earlier write-up of ours wrongly said it didn't). We tested the
> older text-only `Qwen3-Coder-30B-A3B-Instruct-4bit` as a substitute; the real model
> is a newer multimodal thinking MoE we could not yet run. These numbers therefore
> bound the substitute, and a faithful re-test is deferred to spike 0004.

| Metric | Article (M5 Max, 36 GB) | Measured (M4 Max, 36 GB) | Verdict |
|---|---|---|---|
| Decode throughput | 90–130 tok/s | **103.5 tok/s** | ✅ in-band |
| Peak memory | 19–22 GB (w/ KV) | **17.4 GB** (short) → **18.7 GB** (8k ctx) | ✅ consistent |
| Day-to-day coding | "competitive enough" | **7/8 PASS** (easy 3/3, hard 4/5) | ✅ mostly |
| Prefill throughput | ~1851 tok/s | **1273 tok/s peak** (69%) | ⚠️ plausible for M5, not reached on M4 |
| Model identity | "Qwen 3.6-35B-A3B" | **real** (we tested a *substitute*) | ⚠️ see correction below |

Evidence: [spike 0001](spikes/0001-coding-viability/REPORT.md) (coding + decode +
memory) · [spike 0002](spikes/0002-prefill-scaling/REPORT.md) (prefill scaling) ·
[spike 0003](spikes/0003-bugfix-collapse/REPORT.md) (collapse replication + cause) ·
[spike 0004](spikes/0004-entropy-probe/REPORT.md) (entropy probe of the break point).

## Findings

1. **Decode in-band on an older chip.** 103.5 tok/s median sits inside the
   article's 90–130 band — measured on M4 Max, not the claimed M5 Max. The claim
   is plausible, likely conservative, for the M5.
2. **Memory lighter than / consistent with the claim.** 17.4 GB on short prompts,
   rising to 18.7 GB at 8k context — matching the article's "19–22 GB depending
   on KV cache" as a long-context figure.
3. **Coding correctness, executed: 7/8.** All 3 easy + 4/5 hard tasks produced
   code that passed asserts in a fresh subprocess.
4. **A real, reproducible failure mode — bug-fix collapse with a deterministic
   break point (n=5).** Spike 0003 replicated the spike-0001 Java code-switch and
   isolated its cause. The echo-and-fix-this-buggy-binary-search prompt fails
   **0/5** (1 greedy + 4 seeded sampled), and every failure copies the buggy code
   verbatim then derails at the **identical token** — the `(` in `mid = (`. What
   follows is *stochastic* (stray `import math`, an unterminated `"""`, undefined
   `x + y`, or the spike-0001 Java `public …`), but the break *point* is fixed.
   Two controls localize the trigger: `scratch_binsearch` (same algorithm, no
   buggy code to echo) passes **4/5**, and `echo_fix_factorial` (same echo-then-fix
   structure, different function) passes **5/5**. So it is neither the prompt
   structure nor binary search per se — it is the specific combination of echoing
   *this* buggy code and being asked to repair it. **Spike 0004 then read the
   model's own logits and confirmed the mechanism:** the decision right after
   `mid = (` is the **maximum-entropy token of the entire generation** (6.447 bits,
   rank 1/113, vs a 0.000-bit median), while the *identical* `mid = (` text written
   from scratch is 0.001 bits / rank 17/89. The probability mass at the break point
   sits on cross-language "new construct" openers (`import` 0.15, `public` 0.12, `#`,
   `from`) — i.e. the stochastic failure surface of spike 0003 *is* this distribution
   being sampled. (Refinement: the model is not torn between echoing vs. fixing the
   arithmetic — at the open paren it loses the thread and defaults to boilerplate
   openers.)
5. **Prefill reaches 69% of the claim on M4.** Peaks at 1273 tok/s (~1k ctx),
   declining to 1013 tok/s at 8k. The gap to 1851 is consistent with one chip
   generation of uplift; not refuted.
6. **Correction — the named model is real; we tested a substitute.** Our earlier
   write-up claimed "Qwen 3.6-35B-A3B" did not exist. **It does:**
   [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0),
   with an MLX 4-bit build at `mlx-community/Qwen3.6-35B-A3B-4bit`. It is a *newer,
   different* model than what we measured: `model_type: qwen3_5_moe`, a **multimodal
   (vision+text) thinking-by-default MoE** (256 experts, hybrid Gated-DeltaNet),
   reporting SWE-bench Verified 73.4% / LiveCodeBench v6 80.4%. We substituted the
   older, **text-only, non-thinking, coding-specialized** `Qwen3-Coder-30B-A3B-Instruct-4bit`.
   So **all spike-0001/0002/0003 numbers describe the substitute, not the article's
   model** — and the article's model is, on paper, *more* capable at coding. A
   faithful re-test is deferred to a **separate experiment (exp-003)**. It is *not*
   toolchain-blocked: `mlx-lm` 0.31.3 ships a `qwen3_5_moe` loader that loads it as a
   text-only LM (dropping vision weights), so the only real differences to handle are
   thinking-by-default (larger token budget + a thinking-aware harness) and the
   ~18–20 GB download.

## Open / not yet proven

- **Throughput/memory still n=1** — directional, not confirmed. (The bug-fix
  collapse (4) is confirmed at n=5 by spike 0003, and its *cause* — the maximum-
  entropy break point at `mid = (` — is now confirmed by spike 0004's logit
  inspection. One residual unknown: whether the entropy spike is intrinsic or a
  4-bit-quantization artifact, which a non-quantized run would settle.)
- **Single chip** — numbers bound M4 Max; they neither confirm nor refute M5
  beyond "plausible." A borrowed M5 Max would close this.
- **"Replaces Cursor" is broader** — multi-file edits, long-context repo Q&A, and
  tool-using agent loops are unmeasured. The self-contained-function battery is a
  floor, not the real bar.
- **Image-gen / transcription claims** from the article remain unverified (out of
  scope for this experiment).

## Next spikes

- ~~`0003` — replicate the bug-fix language collapse (n≥3) and probe whether it's
  prompt-structure specific.~~ **Done** — confirmed n=5, trigger isolated to the
  specific echo-and-fix instance. See [spike 0003](spikes/0003-bugfix-collapse/REPORT.md).
- ~~`0004` — instrument the `mid = (` break point (per-token entropy / logit
  inspection).~~ **Done** — confirmed it is the maximum-entropy token of the
  generation. See [spike 0004](spikes/0004-entropy-probe/REPORT.md).
- The **faithful re-test on the article's actual model**
  `mlx-community/Qwen3.6-35B-A3B-4bit` moves to a **separate experiment (exp-003)** —
  it is a newer multimodal/thinking arch needing a toolchain upgrade, so it is not a
  drop-in swap inside exp-002.
- Later — confirm the entropy spike survives outside 4-bit (non-quantized / second
  model); 8-bit DWQ as a quality axis; harder, multi-step coding tasks toward the
  real "replaces Cursor" bar.
