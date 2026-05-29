# Experiment Methodology

How every experiment in this repo is run. This is the **contract**: a human or an AI agent should be able to pick up an experiment, run it, and judge it — without guessing the structure. Each experiment under [`experiments/`](experiments/) is an *instance* of the template below.

---

## The method: claim → spike → proof

1. **Claim.** Start from a falsifiable claim — usually something the ecosystem asserts but rarely verifies (a paper, a vendor, a YouTube video).
2. **Spike.** Test one hypothesis at a time with a small, time-boxed experiment. No big-bang builds; each step is validated before the next.
3. **Proof.** A finding is only recorded when backed by runnable code + captured data + a report a reader can reproduce. Reports document failures and self-corrections honestly — the trail *is* part of the proof.

## Experiment specification (the template)

Every experiment's `README.md` fills in these fields **up front**:

| Field | What it pins down |
|---|---|
| **Thesis** | The falsifiable claim under test. One sentence, accept-or-reject-able. |
| **Goal** | The specific question to answer / what "done" means. |
| **Fixed setup (inputs)** | What's held constant: models, framework, hardware, task, sampling. The baselines. |
| **Variations (axes)** | The independent variables swept, each with its range. |
| **Expected range** | The prior/hypothesized outcome bounds — so a *surprise* is recognizable as one. |
| **Metrics & outputs** | What's measured + the grading/verdict scheme + where results land. |
| **Iteration budget** | Max spikes/runs allowed before a decision is forced. |
| **Success criteria** | Explicit condition(s) to **ACCEPT** the thesis. |
| **Reject criteria** | Explicit condition(s) to **REJECT** the thesis. |
| **Stop conditions** | When to halt: thesis decided · budget exhausted · hard blocker. |

A thesis is rarely simply accepted or rejected — more often **refined** (the naive claim is rejected, a conditional version survives). Record the refinement.

## Confidence grading

Label every finding by confidence:
- **n=1 — exploratory.** A single run. Directional, not conclusive.
- **n≥3 — confirmed.** Replicated. With greedy/deterministic sampling, byte-identical replicates are strong evidence.
- **publication-grade.** Confirmed + understood mechanism + cross-checked.

## Folder anatomy (per experiment)

```
docs/experiments/exp-NNN-slug/   # the write-up (docs only)
├── README.md            # this template, filled in for the experiment
├── REPORT.md            # the consolidated findings / leaderboard
├── RECOMMENDATIONS.md   # the practical, deployable takeaways
├── spikes/              # the trail — NNNN-slug/ each with its own REPORT
└── results/             # raw data (jsonl/csv) + generated charts/

src/agentic_evals/               # the code
├── harness/             # generic, reusable core — every experiment imports it
├── experiments/expNNN/  # this experiment's runners (one module per spike + runner + charts)
└── cli.py               # `agentic-evals` console script that dispatches runners
```

The generic **harness** (`agentic_evals.harness`) is the reusable core. Each experiment's runners live in a subpackage (`agentic_evals.experiments.expNNN`), and every runner is invoked uniformly through the `agentic-evals` CLI:

```bash
uv run agentic-evals --list                 # discover experiments + runners
uv run agentic-evals exp-NNN <runner>       # run one (runner slug = module name, hyphenated)
```

That uniform entry point is deliberate: it is the substrate for the autonomous-experiment loop below — an agent only needs `{experiment, runner}` to drive a spike, with no bespoke per-runner scaffolding.

## Naming

- **Experiments** name the *investigation* (subject + method), **not** the finding: `exp-NNN-mlx-kv-compression-toolcalling`, not `exp-NNN-hallucination-cliff`. (Findings change; the investigation is stable. Zero-padded NNN.)
- **Spikes** are `NNNN-slug/` within an experiment.
- The headline *finding* lives in the README/REPORT and the blog title.

## The autonomous-experiment loop (north star)

The reason the template is rigid: **so an AI agent can run an experiment unattended.** Hand it a filled spec — `{thesis, goal, fixed setup, variations, expected range, metrics, iteration budget, success/reject criteria}` — and a budget. The agent then:

```
while iterations_left and not thesis_decided:
    pick the next most-informative spike (within the variations)
    run it; capture proof
    update belief about the thesis
    if success_criteria met:  ACCEPT  (stop)
    if reject_criteria  met:  REJECT  (stop)
iterations exhausted → report best current belief + what's still open
```

This is the Karpathy-style framing: a bounded, goal-directed search that *succeeds or rejects a thesis* within a fixed iteration budget — rather than open-ended tinkering. Today this loop is run **manually (human + AI)**; the template is the machine-followable contract that lets us formalize it into a semi-autonomous runner later. Every field above exists because an unattended agent needs it to know *what to try* and *when to stop*.

## Worked example

[`exp-001-mlx-kv-compression-toolcalling`](experiments/exp-001-mlx-kv-compression-toolcalling/README.md) is the first full instance — the template filled in, then the spike trail, results, and the (refined) verdict. Read it alongside this doc to see the methodology in practice.
