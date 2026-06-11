# exp-005 RECOMMENDATIONS — deployable takeaways

## For IRIS (apply now)

1. **Keep keyword(-first) routing as the production router.** Do not spend
   an LLM call per turn on routing: it costs 300–800ms, adds run-to-run
   variance, and measured worse on every model available (best: granite4 at
   83.8 ± 1.7% vs keyword's deterministic 88.3%).
2. **Attack the 9 confident keyword errors with rules, not models.** They
   cluster in 5 trap families (communication keywords `reply|send|email`
   firing on non-email intents; `who am i` profile trap; `what time` trap;
   `weather` trap; `find` → search trap). Each candidate rule change is
   cheap to grade: rerun the golden set (`exp-005 model-sweep`, keyword row,
   or project-iris `scenarios.run routing-keyword`). Rule-order fixes that
   lift adversarial accuracy without touching clear-case 100% are the
   highest-ROI routing work available.
3. **If LLM escalation is ever revisited, escalate selectively** — only the
   trap-family rules, only to granite4, never trust-on-disagreement
   globally (measured net −3). And not llama3.2:3b: the current
   `llm_tiers.yaml` router model is the second-worst genuine router tested.
4. **Fix the router parser to strip `<think>` blocks** before JSON
   extraction (production gap; this experiment had to add it). Any
   thinking-capable model behind the router otherwise guarantees 100%
   fallback at 256 output tokens.

## Universal rules (blog-worthy)

- **Report parse-fallback rate next to any LLM-router accuracy number.**
  Fallback can silently impersonate accuracy (qwen3.5: "88.3%" with 100%
  fallback at 5.4s/request; llama3.2: a third of its answers were not its
  own).
- **Don't size up for routing.** 9B routed worse than 4B; clean JSON ≠
  correct JSON.
- **Thinking models are disqualified from tight-budget structured tasks**
  unless the harness splits thinking/answer budgets (exp-003's lesson,
  reconfirmed in a different domain).
- **A deterministic baseline is a feature, not a fallback.** Zero latency,
  zero variance, byte-identical replication — an LLM replacement must beat
  it by enough to pay for losing all three.

## v1 addendum (new models)

- gemma4 QAT and the 27B flagship don't change the deployment call: their
  only ≥baseline scores are fallback-inflated at 6–27s per route. No-think
  modes (the only deployable configuration) all land below the baseline.
- New rule: **never put a thinking model in the routing hot path** — and
  if one sneaks in, cap its budget; uncapped thinking measured *worse*
  (gemma4:12b: 81.8% uncapped vs 83.1% no-think).
- The niche worth watching: thinking models' own-answer accuracy (91–95%)
  beats the table — if local inference ever gets 50× faster, selective
  async escalation becomes interesting. Not before.

## Open follow-ups

- Rule-fix pass on the 5 trap families, graded against the golden set
  (project-iris side).
- Confidence-aware selective escalation prototype (keyword rule identity →
  escalate list) — only if rule fixes plateau.
- Purpose-tuned router fine-tune — the one untested path that could rescue
  the LLM-router thesis.
- exp-004 (serving-stack bake-off) may change latency numbers but not the
  accuracy ordering.
