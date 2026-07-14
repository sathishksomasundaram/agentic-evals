# Governance Sweep Kit

The runnable proof for *The Spine, Drawn*. Three governance sweeps on the **real
Harness Lab engine** — no mock, no reimplementation. It ships only what this
experiment needs: three graphs, three fixtures, and one runner you can read in
full. It grades **behaviour, not prose** — which gate fired, whether the model
was reached, whether the answer shipped.

## Run it

```bash
pip install -r requirements.txt     # the real Harness Lab engine (thin; opening soon)
bash demo.sh                        # all three sweeps, expected tables asserted
```

Exit `0` iff every cell matches its expected outcome (CI-ready). Deterministic:
same graph + fixture + seed ⇒ same trace, so your tables match the post's.

## What's in the box

| Path | What it is |
| --- | --- |
| `run_sweeps.py` | The runner (~170 lines, stdlib + `harnesslab`). Loads each graph, sweeps one knob, grades the outcome of every cell, exits non-zero on drift. |
| `sweeps/egress_cloud.json` | Minimal `classify → cloud model` graph — the egress sweep needs its own graph (a tier router would downshift personal content and mask the contrast). |
| `sweeps/governed_chat_spine.json` | The full governed chat spine, as exported from Harness Lab — the graph the tier and judge sweeps run against. |
| `fixtures/ssn_to_cloud.yaml` | Personal content (synthetic SSN) → cloud, for the egress sweep. |
| `fixtures/personal_downshift.yaml` | A personal request whose intent maps to a cloud tier, for the tier sweep. |
| `fixtures/ungrounded.yaml` | A confidently-wrong answer ($305 vs $212.40), for the judge sweep. |
| `requirements.txt` | The real engine: `harnesslab` (thin, stdlib-based). |
| `demo.sh` | Green-on-clone runner once the engine is installed. |

## Point it at your own governance idea

The three `--set` knobs address config by node **type**, so they read like the
lever they turn — `data_classifier.egress`, `tier_router.on_forbidden_egress`,
`llm_judge.threshold`. Edit a graph in `sweeps/`, add a value to a sweep in
`run_sweeps.py`, rerun, and watch the outcome table move. The same three sweeps
run headless via the engine's CLI, too:

```bash
harnesslab sweep sweeps/governed_chat_spine.json \
  --fixture fixtures/ungrounded.yaml \
  --set llm_judge.threshold=0.3,0.6,0.9
```

## The point

The governance question — *"what would a stricter rule have done?"* — has an
answer you can **run**, not a hunch you argue. That's what a declarative,
deterministic harness buys you that a code-based one doesn't: the safety line is
a number you can see before you deploy. Point it at your own graph, push past
where the post stopped, and tell me where the shape bends.
