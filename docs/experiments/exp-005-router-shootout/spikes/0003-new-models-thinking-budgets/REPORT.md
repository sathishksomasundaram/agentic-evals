# Spike 0003 — New models (gemma4 QAT) + thinking-budget fairness

**Question:** Do the model releases that landed after spike 0001 — gemma4
QAT (e4b, 12b) — beat the keyword baseline? And were thinking models
unfairly handicapped by the production 256-token output budget?

**Answer: the verdict survives, with an important refinement.** Three
configurations now reach or pass 88.3% on paper — every one of them is
fallback-inflated and 12–80× over the latency budget. But the fairness
runs revealed something real: **on the cases they complete, the newest
thinking models out-route the regex table (91–95%). Their failure mode is
completion, not judgment.**

## Environment note

Ollama daemon upgraded 0.24.0 → 0.30.7 mid-experiment (gemma4 requires it).
Cross-version anchor: granite4 re-run on 0.30.7 scored 83.1% @ 359ms —
inside its prior n=4 spread (81.8–85.7% @ 322–326ms). Comparability holds.
The new daemon also separates `thinking` from `content` in responses and
supports `think: false`, which this spike uses.

## Results (77-case golden set, production settings unless noted)

| Config | Accuracy | Own-answer acc. | p50 | Fallback |
|---|---|---|---|---|
| keyword (baseline) | **88.3%** | 88.3% | **~0ms** | — |
| qwen3.5:4b, think, 2048-token budget | 90.9% | **95.1%** (39/41) | 26,590ms | **47%** |
| gemma4:12b-qat, think, 256 cap | 89.6% | 93.6% (44/47) | 5,950ms | 39% |
| qwen3.6:27b (largest local, 256 cap) | 88.3% | 91.4% (53/58) | 15,217ms | 25% |
| gemma4:12b-qat, **no think** (n=3) | 83.1 / 83.1 / 83.1% | 83.1% | 1,225ms | 0% |
| gemma4:e4b-qat, no think | 83.1% | 83.1% | 701ms | 4% |
| gemma4:12b-qat, think, **uncapped** | 81.8% | 81.8% | 5,735ms (max 56s) | 1% |
| qwen3.5:4b, no think | 77.9% | 78.3% | 968ms | 5% |

(Registry notes: there is no gemma4 9B — 12b-it-qat is the nearest; qwen3.7
does not exist on Ollama — qwen3.6 tops the family and ships no small sizes.)

## Findings

1. **Nothing clean beats the baseline.** The three ≥88.3% scores carry
   25–47% keyword-fallback content and 5–26s p50 latencies. Subtract the
   fallback and none covers the full case set. The ACCEPT criterion
   (>88.3% at p50 ≤ 500ms, n=3) is missed by 12–53× on latency.
2. **Thinking models are good-but-slow routers, not bad ones** (revision
   of spike 0001's framing). Given budget to finish, qwen3.5:4b answers
   95.1% of its completed cases correctly — better than the keyword table.
   But "given budget" means 26.6s p50, and it *still* blows through 2048
   thinking tokens on 47% of cases. The production constraint isn't
   arbitrary: a router that takes half a minute isn't a router.
3. **The overthinking inversion (new):** gemma4:12b scored *worse* with
   uncapped thinking (81.8%, one case 56s) than with thinking truncated
   (89.6%, fallback-rescued) — and worse than with thinking disabled
   (83.1%). More deliberation actively degraded routing. The 256-token
   "handicap" was accidentally protective.
4. **No-think mode is the only deployable LLM configuration** — clean JSON,
   0–5% fallback, ~0.7–1.2s — and every no-think score (77.9–83.1%) lands
   *below* the keyword baseline. gemma4:12b no-think replicated 83.1%
   three times identically.
5. **Size ceiling confirmed:** the largest installed model (27B) exactly
   ties the regex table while taking 15 seconds and borrowing a quarter of
   its answers from it.

## Implication for the verdict

REJECTED stands, now against 9 model configurations including the newest
QAT releases and with thinking-budget fairness controls. The refined
recommendation: if per-case routing quality ever matters more than
latency (it doesn't, for an interactive assistant), the right shape is
*selective escalation of known-hard cases to a thinking model running
async* — not a thinking model in the hot path. The rule-fix pass
(project-iris) already closed the measured hard cases at 0ms, so even
that niche is currently empty.
