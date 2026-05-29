# v0 — Recommendations per model

**For:** developers picking an open-weight model + KV-cache config for **agentic tool-calling on Apple Silicon via MLX**.

**Status:** Single-pass v0 data (mostly n=1 per cell; spike 0007 confirmed n=3 deterministic on Qwen-2.5). Use as a starting point, **replicate before any production deployment decision.**

**Stack assumed:** mlx-lm + yzamari/mlx-turboquant + greedy sampling + tool-calling task (one tool, JSON output schema). Different stacks/tasks may shift cliffs.

---

## At-a-glance verdict matrix

| Model | Verdict | Use for | Avoid for |
|---|---|---|---|
| Qwen-2.5-7B-Instruct-4bit | ✅ **RECOMMENDED** | General tool-calling, code, reasoning at 7B class | Below ratio 0.50 — silent hallucination |
| Llama-3.2-3B-Instruct-4bit | ✅ **RECOMMENDED** | Tier-1 routing, fast tool calls (short or long) | Below cliff — collapses to garbage tokens (loud failure) |
| Qwen3-4B-Instruct-2507-6bit | ⚠ **SHORT-PROMPT ONLY** | Short interactive tool-calling | Long-context tool-calling — model-level limit, NOT a TQ issue |
| DeepSeek-R1-Distill-Qwen-7B-4bit | ❌ **NOT FOR TOOL-CALLING** | Math, code, logic where reasoning matters | Format-strict tool-calling (reasoning preamble eats budget) |
| Phi-4-mini-instruct-4bit | ✅ **RECOMMENDED + TQ-NECESSARY** | STEM/coding + long-context agentic flows (TQ rescues baseline) | FP baseline on long context — it collapses without compression |
| Gemma-3-4B-it-4bit | ⚠ **NEEDS NATIVE SCHEMA** | Chat, multilingual; tool-calling with its own `{"tool":"<name>","query":"..."}` convention | Forcing non-Gemma tool-call format |

**Per-model cliff data:** ratio ≈ 0.50-0.55 is the cliff for **every model that passes tool-calling** (Qwen-2.5-7B, Llama-3.2-3B, Phi-4-mini — pinned by [spike 0008](spikes/0008-cliff-fine-sweep/REPORT.md) for Qwen-2.5 to 0.505 precision; [spike 0009](spikes/0009-per-model-cliff/REPORT.md) for the other two to the 0.50→0.55 window). Use `ratio = 0.55` as a universal safety-margin default.

---

## Per-model deep dive

### 1. Qwen-2.5-7B-Instruct-4bit · ✅ RECOMMENDED

**Repo:** [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) · head_dim=128 · 28 layers

**Behavior summary:** Cleanest cliff dynamics in the test set. Short tool calls always work; long-context tool calls require careful buffer sizing OR sufficient FP context (with `ratio=1.0` config). Below the cliff: **hallucinates plausible weather data** instead of calling the tool (spike 0008's headline finding).

**Recommended config:**

```python
from mlx_turboquant import make_turboquant_cache, patch_model
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
model = patch_model(model)

# Compute buffer with safety margin above the ~0.505 cliff
prompt_tokens = len(tokenizer.encode(prompt))
buffer_size = max(128, int(0.55 * prompt_tokens))  # 10% margin above cliff

cache = make_turboquant_cache(
    model,
    key_bits=3,
    value_bits=2,        # K3/V2 — POC-recommended asymmetric
    buffer_size=buffer_size,
)
output = generate(model, tokenizer, prompt, max_tokens=200, prompt_cache=cache)
```

**Cliff (Qwen-2.5 specific):** ratio = 0.505 ± 0.003 — see [spike 0008](spikes/0008-cliff-fine-sweep/REPORT.md).
**Use `ratio = 0.55` (+10% safety margin) as the default.**

**Caveats:**
- Long-context **baseline** (no TQ) also fails tool-calling — model gets distracted by filler. Add TQ even if you don't need compression.
- Below cliff: silent hallucination, not visible failure. Set up correctness monitoring.

**Best for:** Tier 2 agentic workloads — tool-calling, code assist, structured extraction.

---

### 2. Llama-3.2-3B-Instruct-4bit · ✅ RECOMMENDED (was CONDITIONAL — upgraded by spike 0009)

**Repo:** [`mlx-community/Llama-3.2-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) · head_dim=128 · 28 layers

**Behavior summary:** [Spike 0009](spikes/0009-per-model-cliff/REPORT.md) pinned the cliff to ratio 0.50→0.55 — **same window as Qwen-2.5-7B**, not higher as initially estimated. At ratio ≥ 0.55, short and long tool calls both pass. Below the cliff, output **collapses into token garbage** (`{"tool": EXAIN EXEXEXEXOM...`) rather than hallucinating fluently — easier to detect than Qwen-2.5's failure mode.

**Recommended config:**

```python
buffer_size = max(128, int(0.55 * prompt_tokens))  # same as Qwen-2.5

cache = make_turboquant_cache(
    model,
    key_bits=3,
    value_bits=2,
    buffer_size=buffer_size,
)
```

**Caveats:**
- Below cliff doesn't gracefully degrade — it **collapses into garbage tokens**. JSON parser will fail loudly — good for detectability.
- Long-context FP baseline fails (model gets distracted by filler). TQ at ratio ≥ 0.55 actually helps.

**Best for:** iris Tier 1 — intent classification, routing, fast simple tool calls. Low-latency path. ~0.35s short prompts, ~1.93s long.

---

### 3. Qwen3-4B-Instruct-2507-6bit · ⚠ SHORT-PROMPT ONLY

**Repo:** [`mlx-community/Qwen3-4B-Instruct-2507-6bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-6bit) · head_dim=128 · 36 layers

**Behavior summary:** Short tool calls pass at every config. **Long tool-calling fails at every config — including `ratio=1.0` (no compression).** This is a model-level limitation, not a TurboQuant artifact. Qwen3-4B-Instruct simply doesn't follow tool-call instructions when context is heavily diluted with irrelevant filler.

**Recommended config:**

```python
# Short prompts only. For long context, switch models.
buffer_size = max(128, int(0.55 * prompt_tokens))

cache = make_turboquant_cache(
    model,
    key_bits=3,
    value_bits=2,
    buffer_size=buffer_size,
)
```

**Caveats:**
- **Don't use for long-context tool-calling.** Use Qwen-2.5-7B (with TQ at `ratio=1.0`) or Llama-3.2-3B instead for long context.
- "Long context" in our test = ~2000 tokens with 1700 of irrelevant filler. Realistic dilution ratios may differ.

**Best for:** snappy short tool-calling, replacing larger models when latency matters more than context.

---

### 4. DeepSeek-R1-Distill-Qwen-7B-4bit · ❌ NOT FOR TOOL-CALLING

**Repo:** [`mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit) · head_dim=128 · 28 layers

**Behavior summary:** Reasoning-tuned model. Spends `max_tokens` on chain-of-thought ("Okay, so I need to figure out…") before getting to the JSON. **Fails all short tool-call configs**. Long-prompt + `ratio=1.0` accidentally passes — dilute context apparently shortens its reasoning preamble.

**Recommended config:**

```python
# For REASONING tasks, not tool-calling:
buffer_size = max(128, int(0.55 * prompt_tokens))
max_tokens = 1000  # allow room for thinking-out-loud

cache = make_turboquant_cache(
    model, key_bits=3, value_bits=2, buffer_size=buffer_size
)
```

**Caveats:**
- For **tool-calling**, use a non-reasoning model (Qwen-2.5-7B-Instruct).
- For reasoning: `max_tokens` must accommodate the thinking preamble — set to ≥ 1000 not 200.

**Best for:** math, logic, code-reasoning tasks. Reserve for the workloads it's actually trained for.

---

### 5. Phi-4-mini-instruct-4bit · ✅ RECOMMENDED — TurboQuant RESCUES long-context failure (was FRAGILE — upgraded by spike 0009)

**Repo:** [`mlx-community/Phi-4-mini-instruct-4bit`](https://huggingface.co/mlx-community/Phi-4-mini-instruct-4bit) · head_dim=128 · 32 layers

**Behavior summary:** [Spike 0009](spikes/0009-per-model-cliff/REPORT.md) found cliff at ratio 0.50→0.55 — same universal window. The surprising part: **long-context FP baseline COLLAPSES** to `"bó bó bó"` repetition, but **TurboQuant at ratio ≥ 0.55 RESTORES the tool call.** Compressing irrelevant filler appears to be net-helpful for this model's attention focus.

**Recommended config:**

```python
buffer_size = max(128, int(0.55 * prompt_tokens))

cache = make_turboquant_cache(
    model, key_bits=3, value_bits=2, buffer_size=buffer_size
)
```

**Caveats:**
- **Don't trust the baseline** at long context — it collapses. Use TurboQuant *because* you need correctness on long prompts, not despite it.
- Below cliff (ratio < 0.55), output is gibberish (`{"}\n™\n™...`). At ratio ≥ 0.55, output is byte-stable.

**Best for:** STEM/coding queries (its strength) **and** long-context agentic flows when paired with TQ at ratio ≥ 0.55. Compression is a *feature* here, not a tax. ~0.41s short, ~1.73s long.

---

### 6. Gemma-3-4B-it-4bit · ⚠ NEEDS NATIVE SCHEMA

**Repo:** [`mlx-community/gemma-3-4b-it-4bit`](https://huggingface.co/mlx-community/gemma-3-4b-it-4bit) · **head_dim=256** (largest in test set) · 34 layers

**Behavior summary:** Short tool calls pass at all configs. Long `ratio=0.5` emits an **alternate tool-call schema** wrapped in code fence:

```
\`\`\`tool
{"tool": "query", "query": "weather in San Francisco"}
\`\`\`
```

It's not failing — it's using Gemma's own tool-call convention rather than our prompt's. Long `ratio=1.0` returned empty `\`\`\`\\n\`\`\`\\n` — collapsed differently.

**Recommended config:**

```python
# Test with Gemma's NATIVE tool-call format, not a forced foreign schema.
# Gemma expects ```tool ... ``` fence with {"tool": "<name>", "query": ...}
buffer_size = max(128, int(0.55 * prompt_tokens))

cache = make_turboquant_cache(
    model, key_bits=3, value_bits=2, buffer_size=buffer_size
)
```

**Caveats:**
- **Native schema matters.** Forcing our `{"tool":"web_search", "args":{...}}` shape on Gemma is a misuse. Use Gemma's `{"tool":"<name>", "query":"..."}` instead.
- head_dim=256 means yzamari recomputes Lloyd-Max codebooks at load time (logged inline). Costs a few seconds on first model load.
- Long-context behavior is unpredictable in our test — alternate schema vs empty depending on buffer.

**Best for:** chat, multilingual, general open-ended generation. Tool-calling with **Gemma-native format** (not yet measured by us).

---

## Universal rules

Regardless of model:

1. **Set `buffer_size` dynamically as a fraction of prompt length** — never use a fixed absolute value across varying prompts.
2. **Aim for `buffer_ratio = 0.55`** (updated by [spike 0009](spikes/0009-per-model-cliff/REPORT.md)). The cliff is at ratio ≈ 0.50-0.55 across the three models we measured — not size-dependent as initially estimated.
3. **Below the cliff, models fail differently — but always fail.** Qwen-2.5 hallucinates fluently (dangerous: prose looks right). Llama-3.2 and Phi-4-mini collapse to token garbage (loud: JSON parser will fail). Set up correctness monitoring either way.
4. **TurboQuant isn't only a compression tool — sometimes it's a quality tool.** On long-dilute prompts, the compressed-attention path can pass where FP baseline fails (Qwen-2.5 spike 0007; Phi-4-mini spike 0009).
5. **Verify on YOUR workload before deploying.** Our tests are one tool, one English prompt structure. Your tasks may differ.
6. **Run each model in its own process** until [yzamari's cross-model state leak](spikes/0007-buffer-cliff-replication/REPORT.md) is fixed upstream. Use `uv run agentic-evals exp-001 v0-runner --model <id>` as the per-process invocation pattern.

---

## What this guide does NOT yet tell you

- **Per-model cliff locations** — only Qwen-2.5 is pinned. Llama-3.2 and Phi-4-mini cliffs are estimated.
- **Cross-hardware behavior** — Apple Silicon only. NVIDIA-via-vLLM path will have different characteristics.
- **Sustained / multi-turn dynamics** — single-turn benchmark. Conversational state across many turns is unmeasured.
- **Other capabilities** — tool-calling only. Routing, classification, code, RAG not yet measured.
- **Bit-width × buffer interaction** — K3/V2 fixed. K2/V2 worked in spike 0005 on Qwen-2.5 short, untested elsewhere.
- **Memory savings at long context** — captured but not yet analyzed for actionable thresholds.

These gaps are the v1 roadmap.
