# exp-008 — the same governed graph, sim and live

The runnable proof behind the post **[*Sim on Tuesday, My Inbox on Wednesday*](https://www.tech4talk.com/blog/local-ai/the-same-harness-live/)**.

[exp-007](../exp-007-governance-topology-sweeps/) swept the governance knobs of a
harness on the bench, in sim. The obvious objection: *does any of it survive
contact with the real world, or does it only hold because the tools are fake?*
This kit answers it. It runs **one** graph (`email_digest` — a deterministic
tool computes the inbox facts, the model only phrases them, a judge chain blocks
a dropped or invented item) **twice**:

- **sim** — the tool is a fixture twin (`email.search`, `mode: sim`);
- **live** — the identical graph with **one binding changed**: the tool node
  points at `inbox.search_inbox` (`mode: real`), a tool imported from a real MCP
  mail server the kit speaks to over stdio.

…then prints the two traces side by side.

## The finding

The two traces are the **same event sequence**. Exactly two lines differ:

```
  SIM (fixture twin)                    LIVE (real MCP mail server)
  run_started  deterministic=True   |   run_started  deterministic=False   <-- one word flips
  tool_result  count=3 (scripted)   |   tool_result  count=4 (live inbox)  <-- real rows over the wire
```

Everything between — `content` classification, the model call, the judge chain,
redaction, audit — is the same nodes doing the same thing. **The governance
didn't notice it went live**, because the guards act on the *flow* (the data
crossing an edge), not on the tool's implementation. The full generated
comparison is in [assets/run-log.txt](assets/run-log.txt).

The one honest cost is the one word that flips: a live run depends on a mailbox
that changes under you, so it can't promise reproducibility — and the trace says
so (`deterministic: false`) rather than pretending.

## Run it

```bash
cd live-kit
pip install -r requirements.txt     # the real engine + MCP transport (opening soon)
bash demo.sh                        # runs sim + live, prints the side-by-side, asserts the diff
```

`demo.sh` exits non-zero unless the event sequences are identical, both runs
complete, and determinism flips true → false. The runner is ~150 lines you can
read in full; it uses the **real** engine — no reimplementation.

## The honest boundary

The mail source is a **mock** MCP server over a synthetic inbox
(`harnesslab.mcp_email_mock`, shipped with Harness Lab) — a stand-in for a real
IMAP/Gmail MCP bridge, so the whole `mode: real` + MCP path runs for real (a
genuine stdio round trip) without wiring a live account. Going from this mock to
a genuine mailbox is a **server swap**, not a graph change: point the import at a
real MCP mail bridge and re-import — the graph, and every guarantee in it, is
unchanged. That portability *is* the finding; the specific mock is not. And to
keep the tool the only variable, the model stays on the sim provider in both
runs — the experiment is about flipping the *tool* to real, not the model.
