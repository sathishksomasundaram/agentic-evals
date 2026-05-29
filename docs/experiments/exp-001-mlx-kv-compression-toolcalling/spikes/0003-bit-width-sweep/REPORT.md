# Spike 0003 — REPORT

**Run date:** 2026-05-27 · **Status:** Complete · **Outcome:** Strong, surprising signal — TurboQuant port appears broken at all tested bit widths.

> **Archival note:** this spike's runner ([`bit_width_sweep.py`](../../../../../src/agentic_evals/experiments/exp001/bit_width_sweep.py)) is pinned to the `rachittshah/mlx-turboquant` port (`TurboQuantKVCache`), superseded by `yzamari/mlx-turboquant` in [spike 0005](../0005-yzamari-port/REPORT.md). It won't import under a fresh `uv sync` — kept as a record of the trail, not a runnable target.

## What we ran

- Model: `mlx-community/Qwen2.5-7B-Instruct-4bit` (28 layers, head_dim=128)
- Framework: `mlx-lm 0.31.3` on Apple Silicon
- TurboQuant port: `rachittshah/mlx-turboquant` @ `f39a74b3` (same as spike 0002)
- `max_tokens=300`, identical tool-calling prompt as spike 0002
- Bit widths swept: `None` (baseline), `4`, `3`, `2`
- Raw JSON: [raw-results.json](raw-results.json)

## Headline results

| Config | Wall time | Peak GPU mem | Tool-call correctness |
|---|---|---|---|
| **baseline** (no TQ) | **0.698 s** | 4.598 GB | ✅ **PERFECT** |
| TQ bits=4 | 6.744 s | 4.565 GB | ❌ collapsed to `!!!!!` × 300 |
| TQ bits=3 | 6.341 s | 4.630 GB | ❌ collapsed to `!!!!!` × 300 |
| TQ bits=2 | 6.361 s | 4.651 GB | ❌ collapsed to `!!!!!` × 300 |

### Baseline output (verbatim)
```json
{"tool": "web_search", "args": {"query": "weather in San Francisco", "time_range": ""}}
```

**A textbook perfect tool call.** ADR 0011 is fully validated: Qwen-2.5-7B-Instruct hits the prompt cleanly with zero preamble, in 0.7 seconds. This is what tool-calling on a properly-tuned model looks like.

### TQ outputs (all bit widths, all identical)
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ... (× 300)
```

## Honest interpretation (Rule 2)

### Finding 1 — All bit widths fail identically. This is not a "quality cliff."

A genuine quality cliff would look like:
- bits=4 → mostly works, occasional errors
- bits=3 → meaningfully degraded but coherent
- bits=2 → severe degradation

What we got: **all three bit widths produced exactly the same degenerate token loop.** The uniformity is the signal. The port is producing the *same broken behavior* regardless of how many bits we asked for. This is a bug signature, not a quality curve.

### Finding 2 — Baseline integration is correct
The non-TQ run produces a perfect JSON tool call. So:
- `mlx-lm.generate()` works correctly
- The chat template works
- The model is fine
- Our prompt is fine

The break is specifically in the TurboQuant-cache code path.

### Finding 3 — Most likely root causes (hypotheses)

In rough order of probability:

1. **GQA / Grouped-Query Attention mismatch.** Qwen-2.5 uses GQA (different number of KV heads than Q heads). `TurboQuantKVCache(bits=N, head_dim=128)` may not be aware of this — passing the wrong head_dim or not accounting for the K/V head count vs Q head count would corrupt every layer's cache. Spike 0002's failure on DeepSeek-R1-Distill-Qwen-7B (same Qwen2.5 architecture) is consistent with this hypothesis.

2. **`prompt_cache` interface mismatch with mlx-lm 0.31.3.** rachittshah's README example uses `model(tokens, cache=cache)` directly — a *single forward pass*. We're calling `generate(model, tokenizer, prompt, prompt_cache=cache)` — full generation. The `prompt_cache` interface in mlx-lm may expect specific methods (e.g. `update`, `state`, `offset`) that `TurboQuantKVCache` doesn't implement matching the latest mlx-lm version. Their tests showed *logit cosine similarity vs FP16* — measured at the prefill stage, not after autoregressive decoding.

3. **Per-layer init / initial-state issue.** The TurboQuant cache might require a warm-up / prefill phase before its rotation matrices are properly applied, which mlx-lm's standard generate loop doesn't trigger.

4. **The port's claimed "drop-in replacement" status is overstated.** It works for the narrow case the author tested (Llama-3.2 + single forward pass logit comparison), but breaks under generation.

### Finding 4 — Peak memory is essentially flat across runs
4.565 — 4.651 GB across all configurations. Model weights dominate at 7B-4bit. KV cache savings irrelevant at this prompt length. Confirms spike 0002's finding 3 — **memory measurements only become interpretable at long contexts** (4K+ tokens).

### Finding 5 — Speed cost is real even when output is junk
TQ runs took ~6.5s vs baseline 0.7s, but most of this delta is "TQ ran to max_tokens=300 because it never emits EOS" while baseline stopped at ~90 tokens. Per-token:
- Baseline: ~130 tok/s
- TQ runs: ~46 tok/s

Even discounting the EOS issue, TQ is ~3x slower per token on rachittshah's pure-MLX port. yzamari's Metal kernel would close this gap, but we have a correctness problem to solve first.

## What this means for v0

| Spike 0003 implication | v0 plan delta |
|---|---|
| TurboQuant + rachittshah port + Qwen-2.5-7B-Instruct is broken (or used incorrectly) | **Cannot validate ADR 0009's TQ axis with this port + this model** |
| Baseline run produces clean tool call | ADR 0011's model switch is correct; **v0 measurement methodology works** end-to-end without TQ |
| Quality cliff diagnostics need port-level validation | Before scaling: must run POC NumPy reference vs rachittshah port encode/decode round-trip — see Spike 0004 |
| yzamari has Metal kernel AND honest disclosure about lossy retrieval | Strong candidate to test next as alternative port |

## Recommended next decisions (priority order)

1. **Spike 0004 (highest priority): port correctness validation.**
   - Load Qwen-2.5-7B-Instruct, extract a real KV cache snapshot during prefill
   - Run that K/V through both POC's `turboquant.py` (NumPy reference) AND `mlx_turboquant.cache.TurboQuantKVCache.update_and_fetch` (port under test)
   - Compare decoded outputs — should match within ~1e-5 numerical tolerance
   - **If they diverge: port has a bug.** Report upstream; pivot to yzamari.
   - **If they match: the issue is in port × mlx-lm integration**, not the algorithm itself. Either patch the integration or pivot to yzamari.

2. **Spike 0005: replicate spike 0003 with yzamari/mlx-turboquant.**
   - Same model, same prompt, same bit widths
   - Determines if the issue is port-specific or fundamental
   - yzamari's Metal kernel may behave very differently

3. **Defer the TQ on/off axis from v0 if neither port works on real instruct models.**
   - v0 can still ship: 2 models × N quants × tool-calling correctness
   - TQ becomes a v1 axis specifically once a port-level fix lands
   - This is not failure — it's *the benchmark working*: we caught a real-world gap between algorithm-spec validation (POC) and production-readiness (real model + real generation)

## Skipped (Rule 3)

- No POC round-trip validation in this spike (planned for 0004)
- No yzamari comparison (planned for 0005)
- No upstream issue filed at rachittshah/mlx-turboquant — should do this after spike 0004 confirms whether the bug is in the port or in our usage
- No long-context probe yet (planned for 0005-ish)
- No statistical significance — n=1 per condition

## Conclusion

**Spike 0003 succeeded by failing usefully.** Two sessions ago we were planning to scale immediately to 18-30 rows on a "K4/V2 on/off" axis. The mini-spike (0002) caught the wrong-model issue; this spike (0003) caught a port-level issue that **would have invalidated every TQ-on row** of the v0 matrix.

The benchmark's thesis — *"real-model autoregressive decoding is the test that matters; synthetic logit-similarity benchmarks do not predict it"* — is being validated by our own spikes. The POC was right at the math level. rachittshah's port might be right at the math level (TBD via spike 0004). But neither produces working tool-calls under real generation on a real instruct model.

This is the exact gap open-model-benchmark exists to close.
