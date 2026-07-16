# Live-Wire Kit

The runnable proof for
[*Simulation First, Then Go Live*](https://www.tech4talk.com/blog/local-ai/the-same-harness-live/).
It runs the same `email_digest` graph twice on the **real Harness Lab engine** —
once in sim, once live against a real MCP mail server over stdio — and prints
the two traces side by side so you can see, yourself, exactly what changed and
what didn't: the **event topology is identical** (every governance node fires in
the same place), while the tool's rows and the `deterministic` flag are the
annotated differences. Data-dependent payloads differ where the data differs.

## Run it

```bash
pip install -r requirements.txt     # the open-source engine + MCP transport, from GitHub
bash demo.sh                        # side-by-side + verdict
```

Exit `0` iff the event-**type** sequences are identical, both runs complete,
`deterministic` flips true → false, the live rows actually came over the wire,
and the sim run's canonical trace digest (`c1:sha256:…`) reproduces across two
executions — the engine's published determinism contract, demonstrated rather
than assumed (CI-ready). The live digest is printed but never asserted stable:
that run says `deterministic: false`, and the kit believes it.

## What's in the box

| Path | What it is |
| --- | --- |
| `run_livewire.py` | The runner (~170 lines, `harnesslab` + `mcp`). Runs sim twice + live once, diffs the traces, asserts the topology holds, determinism flips, and the sim digest reproduces. |
| `graphs/email_digest.json` | The graph — a deterministic digest tool, a narrating model, a judge chain. As exported from Harness Lab. |
| `fixtures/sim.yaml` | The sim run (fixture-twin tool, scripted rows). |
| `fixtures/live.yaml` | The live run's scripted narrative + judge scores (the model is held on sim in both runs, so the *tool* is the only variable). |
| `requirements.txt` | The real engine + MCP transport: `harnesslab[mcp]`. |
| `demo.sh` | Green-on-clone runner once the engine is installed. |

## The point

Going live is one binding, not a rewrite — and the governance travels with the
graph because it acts on the flow, not the tool. Swap the mock MCP mail server
for a real IMAP/Gmail bridge and the runner, the graph, and the guarantees don't
change; only the inbox does. Point it at your own, and tell me where the shape
bends.
