# Spike 0002 — Hybrid ceiling + disagreement-escalation headroom

**Question:** Does the production keyword-first hybrid (LLM only on
keyword-fallback cases) beat pure keyword? And is there headroom in an
escalate-on-disagreement design?

**Answer: The production hybrid architecture cannot beat the keyword
classifier on this golden set — its ceiling equals the keyword score, and
every model only loses points from there.**

## Part A — hybrid measurement (`uv run agentic-evals exp-005 hybrid-sweep`)

| Config | Accuracy | Δ vs keyword |
|---|---|---|
| keyword | 88.3% | — |
| hybrid:qwen3.5:4b | 88.3% | ±0 (100% parse-fallback = keyword in disguise) |
| hybrid:granite4 | 87.0% | **−1.3** |
| hybrid:llama3.2:3b / qwen2.5:7b / gemma2:9b | 85.7% | **−2.6** |

**Why the ceiling is structural:** the keyword stage answers with confidence
0.85 whenever *any* rule fires and 0.5 otherwise — binary confidence. On this
set a rule fires on 74/77 cases, so only 3 cases (gen-01 fun-fact, gen-02
translate, gen-05 haiku) ever reach the LLM — and all 3 already pass via the
keyword fallback's `general/system` default. The hybrid's reachable maximum
is therefore exactly the keyword score; the LLMs misrouted 1–2 of the 3
("translate good morning to japanese" → communication, again).

Meanwhile **all 9 keyword errors are confidently wrong** (a rule fired at
0.85): five communication-keyword traps (reply/email/send), the who-am-i
profile trap, the what-time trap, the weather trap, the find→search trap.
The hybrid never escalates them, so no LLM can ever fix them.

## Part B — disagreement-escalation headroom (offline, from spike 0001 rows)

If instead the router escalated *whenever keyword and LLM disagree*:

| LLM | Fixes (of keyword's 9 errors) | Breaks (keyword-correct, LLM disagrees & is wrong) | Net |
|---|---|---|---|
| granite4 | **6** (adv-07/08/09/18/20/21) | 9 | **−3** |
| qwen2.5:7b | 5 | 15 | −10 |
| gemma2:9b | 5 | 25 | −20 |
| llama3.2:3b | 2 | 12 | −10 |

Naively trusting the LLM on disagreement is identical to running the LLM
standalone — a loss. A **perfect arbiter** picking the right side of every
granite4 disagreement would score **74/77 = 96.1%**, so the headroom is real
(+7.8 points) but requires an arbitration signal neither side provides
today. Candidate signals for a follow-up: keyword rule identity (the 5
communication-trap rules produce most errors — escalate only those), LLM
self-reported confidence (untested), or simply better rules (adv-07's
find-trap and adv-18's who-am-i-trap look rule-fixable without a model).

## Verdict input

No hybrid beats 88.3%. Combined with spike 0001 (no standalone LLM beats it
either) and n=3 replication of the best LLM (granite4: 84.4/83.1/85.7/81.8,
mean 83.8 ± 1.7), the thesis is **REJECTED** within budget.
