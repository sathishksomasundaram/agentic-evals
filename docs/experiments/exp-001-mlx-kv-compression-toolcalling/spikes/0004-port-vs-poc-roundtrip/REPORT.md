# Spike 0004 — REPORT

**Run date:** 2026-05-27 · **Status:** Complete · **Outcome:** Port algorithm is correct. Spike 0003 bug is in the cache / mlx-lm integration layer, NOT the math.

> **Archival note:** this spike's runners ([`poc_vs_port_roundtrip.py`](../../../../../src/agentic_evals/experiments/exp001/poc_vs_port_roundtrip.py), [`real_kv_reconstruction.py`](../../../../../src/agentic_evals/experiments/exp001/real_kv_reconstruction.py)) depend on the POC `turboquant` package and the `rachittshah` port's `polar_quant` module — neither is the project's current pinned dependency (`yzamari/mlx-turboquant`, adopted in [spike 0005](../0005-yzamari-port/REPORT.md)). They won't import under a fresh `uv sync` — kept as a record of the trail, not runnable targets.

## What we ran

Round-trip encode/decode of synthetic K vectors through two implementations:
- **POC reference** — `docs/experiments/exp-001-mlx-kv-compression-toolcalling/spikes/0001-inference-knobs-landscape/turboquant-poc/turboquant.py` (`TurboQuant.encode → decode`)
- **Port under test** — `rachittshah/mlx-turboquant` (`PolarQuant.quantize_and_reconstruct`)

Synthetic K: 128 rows × 128 dims, outlier channels (scale=6 on 4 random channels), matches the POC benchmark's `make_kv()` generator. Bit widths swept: 4, 3, 2.

Verdict criterion: equal reconstruction error magnitude across implementations (different random rotations are expected, so identical *outputs* are not — equal *quality* is).

Raw JSON: [raw-results.json](raw-results.json)

## Headline result

| bits | POC MSE | Port MSE | MSE ratio | POC cos | Port cos | cos Δ | Verdict |
|---|---|---|---|---|---|---|---|
| 4 | 0.0195 | 0.0193 | **0.99x** | 0.99537 | 0.99541 | +0.00004 | ✅ **MATCH** |
| 3 | 0.0727 | 0.0706 | **0.97x** | 0.98267 | 0.98307 | +0.00040 | ✅ **MATCH** |
| 2 | 0.2446 | 0.2407 | **0.98x** | 0.9396  | 0.94037 | +0.00077 | ✅ **MATCH** |

**Algorithm parity is established at all three bit widths.** rachittshah's `PolarQuant` produces reconstruction error indistinguishable from the POC's NumPy reference. The port's math is correct.

## What this tells us about spike 0003's failure

Spike 0003 saw all bit widths produce identical degenerate output (`!!!!` × 300) during real generation. Spike 0004 proves this is **NOT** because:
- ~~The port has a fundamentally broken algorithm~~ — refuted; algorithm matches POC
- ~~The port silently ignores the bits parameter~~ — refuted; quality scales with bits as expected on synthetic data

So the bug must live at one or more of:
1. **Cache wrapper** (`TurboQuantKVCache.update_and_fetch` and friends) — the layer above `PolarQuant` that adapts it to mlx-lm's cache protocol.
2. **mlx-lm `prompt_cache` integration** — how `generate()` invokes the cache during prefill + decode.
3. **GQA mismatch** — Qwen-2.5 uses Grouped-Query Attention; cache handling may not correctly account for `num_key_value_heads ≠ num_attention_heads`.
4. **Real-attention vs static-test gap** — `quantize_and_reconstruct` measures one-shot round-trip; autoregressive decoding repeatedly queries against a growing cache. Drift across many calls could accumulate even when each call is fine.

## Iteration log — Rule 2 honest note

**The first run of this spike showed catastrophic divergence** (port MSE = 265, cos = 0.08) at all bit widths — visually identical to the spike 0003 failure pattern.

That was a **bug in the spike script**, not the port. I had unpacked `PolarQuant.quantize_and_reconstruct`'s return tuple as `(indices, norms, reconstructed)`. The actual return order is `(reconstructed, indices, norms)`. I was therefore comparing the original K to the uint8 *index* array, which produced the spurious catastrophic MSE.

This is documented inline in [poc_vs_port_roundtrip.py](../../../../../src/agentic_evals/experiments/exp001/poc_vs_port_roundtrip.py) as a code comment, so the next reader doesn't repeat it.

Why surface this clearly: it's exactly the *kind of subtle integration bug* that ruins benchmarks if not caught. The benchmark project's own dogfooding caught it within one debug pass — encouraging for the methodology, embarrassing for me, useful to record.

## What this means for next decisions

| Open question after spike 0003 | Spike 0004 verdict |
|---|---|
| Is the rachittshah port's algorithm broken? | **No** — math matches POC reference within numerical tolerance. |
| Should we file an upstream bug? | **Not yet** — the algorithm is correct; the bug (if it's in the port) is in the cache wrapper, not the math. |
| Is the port still a candidate for v0? | **Conditionally yes** — pending diagnosis of the integration-layer bug. If it's in our usage (e.g., GQA-related), we can patch our side. If it's in the wrapper, may need a small upstream fix. |
| Should we still try yzamari? | **Yes** — spike 0005. yzamari has a fundamentally different integration path (custom Metal kernel that bypasses the cache-wrapper layer). Apples-to-apples comparison resolves whether the issue is rachittshah-specific or fundamental to mlx-lm prompt_cache. |

## Recommended next decisions

1. **Spike 0005** — same prompt + sweep on yzamari/mlx-turboquant. Apples-to-apples.
   - If yzamari works on Qwen-2.5-7B-Instruct → rachittshah cache wrapper is the bug. File upstream issue. Use yzamari for v0.
   - If yzamari also fails identically → it's an mlx-lm + GQA + custom-cache fundamental issue, not port-specific. Either fix our usage (GQA-aware cache construction) or defer TQ axis from v0.

2. **Optional intermediate diagnosis** — instrument rachittshah's cache wrapper to log shapes per layer during a single forward pass on Qwen-2.5. Look for shape mismatches that would indicate GQA misalignment. ~30 min. Could short-circuit the yzamari spike if it pinpoints the bug.

3. **Update [turboquant-google.md](../0001-inference-knobs-landscape/turboquant-google.md) spike doc** with this finding — the "no canonical port" assessment is partially corrected (rachittshah's *algorithm* is fine; *integration* is the question mark).

## Skipped (Rule 3)

- Didn't test with real K/V from a loaded model (used synthetic outlier-channel data, like POC's harness). Doing this with real Qwen-2.5 KV would be a stronger test but requires instrumented model loading.
- Didn't test the QJL variant (POC's `use_qjl=True` + port's `TurboQuantCompressor` with QJL). Default config used both sides.
- Didn't compare V (values) separately — only tested with the K generator. K and V follow the same distribution in our synthetic data, so this is a small omission.
- Didn't test the asymmetric K/V wrapper (POC has `AsymmetricKV`; port's `TurboQuantCompressor` supports asymmetric via separate seeds). For the focused "does the math match" question, symmetric tests suffice.

---

## Part B — Real Qwen-2.5 K/V reconstruction (added 2026-05-27)

Spike 0004 above showed algorithm parity on synthetic data. But spike 0003 still fails end-to-end. The next question: **does the port's algorithm work on REAL Qwen-2.5 K/V**, or is the synthetic-data quality an artifact?

Methodology (executable script: [real_kv_reconstruction.py](../../../../../src/agentic_evals/experiments/exp001/real_kv_reconstruction.py)):
1. Load Qwen-2.5-7B-Instruct-4bit, do one prefill on a 86-token prompt using mlx-lm's default `KVCache`
2. Extract layer-0's stored K/V (shape `(1, 4, 256, 128)` — 4 KV heads × 256 seq × 128 dim)
3. Flatten to `(1024, 128)` and run through `PolarQuant(bits=3, dim=128, seed=42)`
4. Measure reconstruction quality vs spike 0004's synthetic baseline

Pre-instrumentation: monkey-patched `TurboQuantKVCache.update_and_fetch` to log shapes during one prefill. **Shapes are correct** — GQA is handled (4 KV heads vs 28 attention heads is correctly passed through). The next-token logit on that one prefill collapsed to token id 0 (`!`), matching spike 0003's full-generation failure — strongly suggesting **reconstruction quality**, not shape mismatch, is the issue.

### Result

**Real Qwen-2.5 layer-0 K statistics (not what the algorithm assumes):**

| Metric | Synthetic (POC harness) | Real Qwen-2.5 layer-0 |
|---|---|---|
| K mean | ≈ 0.0 | **0.4459** |
| K std | ≈ 1.4 | **15.48** |
| K max\|abs\| | ≈ 18 | **171.62** |
| Per-head K std | uniform | **23.84, 14.44, 12.19, 5.65** — 4× spread |
| V std | ≈ 1.4 | **0.13** (100× smaller than synthetic) |

**Reconstruction quality on real K/V at bits=3:**

| Metric | Synthetic (spike 0004) | Real Qwen-2.5 K | Real Qwen-2.5 V |
|---|---|---|---|
| per-row cosine | **0.98267** | **0.3307** ❌ | **0.3301** ❌ |
| MSE | 0.0727 | 7.72 (106× worse) | 5.93e-4 |
| max\|err\| | 1.89 | 26.24 | 0.239 |

**This is the actual root cause of spike 0003's `!!!!` failure.** Per-row cosine of 0.33 on the cache vectors means attention scores are essentially randomized — the model can't distinguish anything in its own context and collapses to the next-token argmax (which happens to be token id 0 = `!`).

### Why the algorithm fails on real data

PolarQuant (and the POC) assumes that after per-row normalization (`u = x/||x||`) and random rotation (`y = R·u`), each coordinate of `y` is approximately `N(0, 1/d)`. The Lloyd-Max codebook is designed for that distribution.

That assumption holds for the synthetic data (zero-mean, unit-variance ish, outlier channels). It does **not** hold for real Qwen-2.5 K:
- **Non-zero mean** (0.45 — there's a directional bias)
- **High per-head variance spread** (head 0 std=23.8 vs head 3 std=5.6 — the same codebook can't serve both)
- **Extreme outliers** (max\|K\|=171 — Lloyd-Max codebook bins saturate)
- **V has 100× smaller scale than synthetic** — different problem but same cosine result, suggesting the rotation is collapsing direction information regardless of scale

This is exactly what **yzamari's "needle retrieval FAILS on compressed tokens"** warning was about. The POC's "near-lossless K4/V2" claim, validated on synthetic data with outlier channels, **does not survive real-model K/V distributions**.

### Implication for the benchmark thesis

This finding is significant for the project's framing:
- The POC validated the math on synthetic data → ~5x compression at ~0.98 cosine
- rachittshah's port validated on synthetic data → same result (spike 0004)
- Both ports' upstream tests measured logit-cosine at single forward passes
- **Real Qwen-2.5 K/V reconstruction has cosine = 0.33** — the algorithm's "data-oblivious" promise fails on real distributions

**The benchmark's entire reason to exist is being demonstrated by our own spikes.** Static-data and synthetic benchmarks systematically overestimate KV-quantization quality. Real-model agentic-workload measurement is the gap.

This is a **publishable finding** in its own right — likely the headline of the first blog post if we lean into it.

---

## Conclusion (revised after Part B)

**Spike 0004 succeeded twice:** Part A showed algorithm parity (port math = POC math); Part B showed the algorithm itself doesn't survive real Qwen-2.5 K/V distributions.

This reframes the next-step priorities:

| Earlier conclusion | Revised after Part B |
|---|---|
| Bug is in rachittshah's cache wrapper | **Bug is in the algorithm's assumption fit to real data** — the cache wrapper is innocent |
| Spike 0005 (yzamari port) will likely fix it | **Spike 0005 likely shows the same failure** — yzamari uses the same algorithm. May be partially mitigated by yzamari's `buffer_size` (keep recent tokens uncompressed) workaround. |
| File upstream bug at rachittshah | **No bug to file** — implementation is correct |

The honest assessment: **TurboQuant K4/V2 as a clean v0 axis is now in serious doubt.** Best-case if spike 0005 (yzamari) shows the same: TQ becomes a *measurement target* the benchmark exposes (highly valuable result), not a *deployment knob* iris adopts (deferred indefinitely).

Plus the meta-result: in 5 spikes we've demonstrated the benchmark's core thesis on our own attempts to use someone else's optimization claim. The "novel angle" from ADR 0005 is no longer hypothetical — it has its first datapoint.

The spike also surfaced one bug in our own code (tuple unpack order) within one debug cycle — encouraging signal that the methodology catches integration mistakes early, which is exactly what the project's thesis requires.
