# exp-002 — MLX MoE coding viability on Apple Silicon

An instance of the [experiment methodology](../../EXPERIMENT-METHODOLOGY.md). It tests the central claim of a widely-shared blog post — that a single quantized Mixture-of-Experts LLM on a 36 GB Apple-Silicon laptop is fast enough *and* good enough to replace cloud coding assistants.

Source claim: [*"I cancelled ChatGPT, Cursor and Midjourney this week — my MacBook Pro M5 Max quietly replaced all"*](https://medium.com/@shreetejghodekar/i-cancelled-chatgpt-cursor-and-midjourney-this-week-my-macbook-pro-m5-max-quietly-replaced-all-91cbd7f3c78b) (Medium).

---

## Specification

| Field | Value |
|---|---|
| **Thesis** | "A 4-bit ~30-35B-A3B MoE LLM on a 36 GB Apple-Silicon Mac runs in **19-22 GB** at **90-130 tok/s decode** and is **competitive enough for day-to-day coding**." (The article's headline LLM claim.) |
| **Goal** | Measure decode + prefill throughput and peak memory for the closest *real* MoE model on this hardware, and grade it on a small executable coding battery — then ACCEPT / REJECT / REFINE the article's numbers. |
| **Fixed setup** | Apple **M4 Max, 36 GB** (the article used an M5 Max, 36 GB — one chip generation newer) · `mlx-lm` 0.31.3 · greedy/deterministic sampling · model = `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (see model note below). |
| **Variations** | Coding task (battery of executable prompts, short → long) · later: quantization (4-bit vs 8-bit DWQ). Start n=1 per cell. |
| **Expected range** | Article: 90-130 tok/s decode, 19-22 GB. On M4 (not M5) Max we expect decode somewhat lower (~60-100 tok/s); memory ~16-20 GB. Coding battery: prior is "most simple tasks pass." |
| **Metrics & outputs** | Per run: `prompt_tps`, `generation_tps`, `peak_mem_gb`, coding verdict (PASS / FAIL by executing the generated code against asserts). Results in [`results/`](results/); analysis in REPORT.md; deployable takeaways in RECOMMENDATIONS.md. |
| **Iteration budget** | ~5 spikes. |
| **Success criteria** | ACCEPT if decode lands in the article's 90-130 tok/s band **and** peak memory ≤ 22 GB **and** the coding battery mostly passes. |
| **Reject criteria** | REJECT if decode is far below the band, memory exceeds 22 GB (won't fit the claim), or the coding battery mostly fails. |
| **Stop conditions** | Thesis decided · budget exhausted · hard blocker (model won't load / OOM). |
| **Verdict** | **SUPPORTED so far — on a substitute model (n=1).** On M4 Max: decode **103.5 tok/s** (in the 90-130 band), peak **17.4-18.7 GB** (consistent with the 19-22 KV claim), coding **7/8 PASS**. Prefill **1273 tok/s = 69%** of the 1851 claim — plausible for M5, not reached on M4. ⚠️ Measured on `Qwen3-Coder-30B-A3B`, **not** the article's real (newer, multimodal) `Qwen3.6-35B-A3B` — see Model note. See [REPORT.md](REPORT.md). |

## Model note (a correction)

The article names **"Qwen 3.6-35B-A3B"**. An earlier version of this write-up claimed no such model existed — **that was wrong.** The model is real: [**`Qwen/Qwen3.6-35B-A3B`**](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0), with an MLX 4-bit build at **`mlx-community/Qwen3.6-35B-A3B-4bit`**. It is a *newer, different* model than the one we measured: `model_type: qwen3_5_moe`, a **multimodal (vision+text), thinking-by-default MoE** (256 experts, hybrid Gated-DeltaNet), reporting SWE-bench Verified 73.4% / LiveCodeBench v6 80.4%.

What spikes 0001–0003 actually tested is a **substitute**: the older, **text-only, non-thinking, coding-specialized** **`Qwen3-Coder-30B-A3B-Instruct-4bit`** (~17.2 GB on disk). It shares the 30B-class-A3B MoE shape but is not the article's model — and the article's model is, on paper, *more* capable at coding. So all current numbers bound the substitute. A faithful re-test on the real model is a **separate experiment, exp-003** (not a drop-in swap inside exp-002). It is *not* blocked on the toolchain after all — `mlx-lm` 0.31.3 ships a `qwen3_5_moe` loader that treats the model as a **text-only LM** (it drops the `vision_tower`/`model.visual` weights), so no `mlx-vlm` or version bump is needed. The differences exp-003 must handle are that it is **thinking-by-default** (needs a thinking-aware harness + larger token budget) and a larger (~18–20 GB) download.

## The method

Claim → spike → proof. One hypothesis per spike, each with captured throughput/memory data and (where applicable) executed code. Reports document surprises and self-corrections honestly.

## The trail, spike by spike

| Spike | Question | What it found |
|---|---|---|
| [0001-coding-viability](spikes/0001-coding-viability/REPORT.md) | Headline tok/s, memory, and coding pass-rate on a hardened 8-task battery? | Decode **103.5 tok/s** (in the article's 90-130 band, on an *older* M4 Max), peak **17.4 GB**, coding **7/8 PASS** (easy 3/3, hard 4/5). The lone failure: the model code-switched Python→Java mid-token on the bug-fix task. Tested a substitute (`Qwen3-Coder-30B-A3B`); the article's real model `Qwen3.6-35B-A3B` exists but is a newer multimodal arch we couldn't yet run — see Model note. |
| [0002-prefill-scaling](spikes/0002-prefill-scaling/REPORT.md) | Can prefill reach the article's ~1851 tok/s on a long prompt? | Prefill peaks at **1273 tok/s** (~1k ctx) = **69%** of the claim, declining to 1013 tok/s at 8k. Memory rises to 18.7 GB at 8k — matching the article's "19-22 GB depending on KV cache." Gap to 1851 plausible for the faster M5. |
| [0003-bugfix-collapse](spikes/0003-bugfix-collapse/REPORT.md) | Is the spike-0001 Python→Java collapse reproducible, and what triggers it? | **Confirmed n=5.** The echo-and-fix-this-buggy-binary-search prompt fails **0/5**; every failure derails at the *identical* token `mid = (`, with the garbage after it varying by seed (Java, stray `import`, `"""`, undefined vars). Controls isolate the trigger: `scratch_binsearch` **4/5**, `echo_fix_factorial` **5/5** → it is the specific echo-and-fix instance, not the structure or the algorithm. |
| [0004-entropy-probe](spikes/0004-entropy-probe/REPORT.md) | *Why* does it break at `mid = (`? | **Confirmed mechanism.** The decision right after `mid = (` is the **maximum-entropy token of the whole generation** (6.447 bits, rank **1/113**, vs 0.000 median); the *same* `mid = (` written from scratch is 0.001 bits. Mass sits on cross-language openers (`import` 0.15, `public` 0.12, `#`, `from`) — the spike-0003 failure surface *is* this distribution being sampled. |

## Reproduce

```bash
uv run agentic-evals exp-002 coding-viability   # spike 0001 — decode/memory/coding battery
uv run agentic-evals exp-002 prefill-scaling     # spike 0002 — prefill throughput sweep
uv run agentic-evals exp-002 bugfix-collapse     # spike 0003 — collapse replication (n=5)
uv run agentic-evals exp-002 entropy-probe       # spike 0004 — entropy of the break point
# override the model:
uv run agentic-evals exp-002 coding-viability --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
```

## Honest caveats

- **Different chip.** We measure on M4 Max; the article claims M5 Max. Throughput numbers are not directly comparable — a slower result here does not by itself refute the M5 claim.
- **Substitute model.** The article's named model `Qwen3.6-35B-A3B` is real (a newer multimodal/thinking MoE); spikes 0001–0003 measure the older text-only `Qwen3-Coder-30B-A3B` instead. The real model *is* loadable as a text LM by the pinned `mlx-lm`, so the faithful re-test = the separate experiment exp-003 (needs a thinking-aware harness).
- **Small battery, n=1.** The coding grade is a smoke test of executable correctness, not a SWE-bench-grade evaluation.
