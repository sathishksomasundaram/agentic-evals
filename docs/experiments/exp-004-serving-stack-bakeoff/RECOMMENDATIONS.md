# exp-004 RECOMMENDATIONS — deployable takeaways

## For IRIS (`config/llm_tiers.yaml` changes)

1. **Tier 3, step 1 — change the model (do this first, zero ops risk):**
   `qwen3.6:27b` → `qwen3.6:35b-a3b` on the existing Ollama daemon.
   4× decode (18.6 → 75.2 tok/s) on identical infrastructure. Costs
   +5.3 GB resident; pair with step 2 to claw it back.
2. **Tier 3, step 2 — serve the MoE via MLX** (dedicated process; IRIS's
   existing `lmstudio` provider, or `mlx_lm.server`): 104.4 tok/s at
   19.7 GB — +39% decode and −4.1 GB vs the same weights on Ollama.
   **Precondition (required regardless of stack):** wire exp-003's
   thinking-budget recommendations into the Tier-3 config — split
   think/answer budgets and `</think>` detection. The model thinks by
   default; a flat `max_tokens` will truncate mid-thought.
3. **Everything else stays:** router = keyword classifier; Tiers 1/2 and
   code_exec on Ollama with **default** daemon flags
   (`OLLAMA_FLASH_ATTENTION`/`OLLAMA_KV_CACHE_TYPE` measured as a −6%
   decode regression on this hardware — update the `.env.example`
   comment to cite spike 0003).
4. **If TurboQuant ever enters the serving path:** the buffer rule
   `buffer ≥ 0.55 × prompt_tokens` is now validated on IRIS-shaped
   prompts (spike 0004B) — apply it as published, with the cliff at
   0.50–0.55 confirmed as a step function.

## Universal rules (blog-worthy)

- **Change the model before the stack.** The Tier-3 win was 80% model
  architecture (dense → MoE, 4×) and 20% serving stack (+39%). Most
  "switch to X" advice conflates the two.
- **Constrained decoding is a shape tool, not a quality tool.** It
  guarantees parseability and deletes your fallback ensemble in the same
  move. Use it where shape failure dominates (tool args, extraction),
  never to "fix" a judgment problem.
- **A serving stack earns a tier, not a fleet.** One process per model is
  a weakness for a four-tier fleet and a non-issue for a dedicated
  big-model tier.
- **Grade vendor/config hints like any other claim** — including your own
  `.env.example`.

## Open follow-ups

- IRIS-side adoption PR: tier3 → 35b-a3b (+ thinking-budget config), then
  the MLX provider wiring; re-run the scenario harness before/after for
  the production proof.
- Ollama-side MoE quality re-grade (exp-003's battery on the GGUF quant)
  if the Ollama-MoE fallback path is ever load-bearing.
- Operability spike under real multi-tier load (deferred; budget held).
