# Spike 0004 — Tier 3: the MoE changes everything, and MLX earns its tier

Two parts: (A) the Tier-3 serving decision — same MoE across stacks plus
the dense incumbent; (B) exp-001's buffer-cliff rule finally tested on an
IRIS-shaped prompt.

## Part A — Tier-3 throughput/memory (n=3 each, same day, same machine)

| Config | Decode tok/s | 256-token wall | Memory | Method |
|---|---|---|---|---|
| qwen3.6:27b dense @ Ollama (**incumbent**) | 18.6 (18.5–18.7) | 14.2s | 18.5 GB | ollama ps |
| qwen3.6:35b-a3b MoE @ Ollama | 75.2 (75.1–75.4) | 3.7s | 23.8 GB | ollama ps |
| Qwen3.6-35B-A3B-4bit @ MLX (in-process) | **104.4** (102.3–107.1) | **2.7s** | **19.7 GB** | mx.get_peak_memory |

### Findings

1. **The incumbent is the wrong model before it's the wrong stack.** The
   dense 27B decodes at 18.6 tok/s — a 14-second wall for one 256-token
   answer. The 35B-A3B MoE (3B active parameters) is **4× faster on the
   very same Ollama daemon**. Whatever else happens, Tier 3 should be the
   MoE.
2. **MLX beats Ollama on the identical MoE: +39% decode, −4.1 GB.** This
   is the first and only configuration in the entire bake-off to clear
   the ≥10% ACCEPT bar — and it replicates exp-003's independent
   measurement (106.8–107.6 tok/s, 19.8 GB) almost exactly.
3. **The memory delta matters operationally:** 23.8 GB resident (Ollama
   MoE) on a 36 GB machine leaves little room for the Tier-1/2 models;
   19.7 GB (MLX) restores the headroom.
4. The one-process-per-model weakness of `mlx_lm.server` (spike 2) does
   not apply here: Tier 3 is a single dedicated model — one dedicated
   process is the natural topology.

### Caveats

- Quality on the MLX side is established (exp-003: 7/8 coding with split
  thinking budgets); the Ollama-side MoE quality was not re-graded here
  (same weights family, different quant pipeline — assumed comparable,
  flagged for the adoption checklist).
- Thinking-budget handling (exp-003's recommendations: split think/answer
  budgets, `</think>` detection) is REQUIRED regardless of stack — the
  IRIS Tier-3 config has neither today.

## Part B — exp-001's cliff rule on an IRIS-shaped prompt

Production router system prompt diluted to ~2,000 tokens (exp-001
methodology), 5 golden routes, greedy, Qwen2.5-7B-Instruct-4bit (the
pinned-cliff model), TurboQuant K3/V2:

| buffer_ratio | PASS | BROKEN_OUTPUT |
|---|---|---|
| 0.45 | 0/5 | 5/5 |
| 0.50 | 0/5 | 5/5 |
| 0.55 | **5/5** | 0/5 |
| 1.00 | 5/5 | 0/5 |

**The published rule (`buffer ≥ 0.55 × prompt_tokens`) survives contact
with the production prompt shape exactly as published** — a clean step
function between 0.50 and 0.55, deterministic, on a different task
(routing JSON) than the one it was derived from (weather tool-call).
exp-001's headline recommendation is now in-harness-validated, closing
the follow-through gap flagged when this experiment was scoped.

## Verdict input

**Tier 3: ACCEPT — two-step adoption.** Step 1 (stack-independent, do
first): replace the dense 27B with the 35B-A3B MoE — 4× decode on the
existing daemon. Step 2: serve that MoE via MLX (dedicated process /
LM Studio provider, which IRIS already has) for +39% decode and 4 GB of
headroom — contingent on wiring thinking-budget handling into the Tier-3
config, which is required either way.
