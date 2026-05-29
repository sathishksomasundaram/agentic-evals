# Mini-spike 0002 — Setup

Spec for the smallest possible v0 dry-run. One model, one quant, one prompt, with and without TurboQuant. Per Rule 10.

## MLX model checkpoints (Task #8)

For the mini-spike we only need ONE model. Confirmed availability on Hugging Face under `mlx-community`:

**DeepSeek-R1-Distill-Qwen-7B (confirmed):**
- [`mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit) — primary target for mini-spike (~4 GB)
- [`mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit`](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit)
- `mlx-community/DeepSeek-R1-Distill-Qwen-7B-MLX` (multi-quant repo: 2/3/4/6/8 bits available)
- Per search: 2-bit, 3-bit, 6-bit variants also published

**Qwen-2.5-7B and Qwen-3 (deferred verification):** not yet checked; will verify before scaling v0. The mini-spike only needs the first model.

## TurboQuant MLX port choice (Task #9)

Three candidate ports compared:

| Repo | Paper cited | API style | Asym K/V? | Has Metal kernel? | Honest about lossy? |
|---|---|---|---|---|---|
| [matt-k-wong/turboquant-mlx-full](https://github.com/matt-k-wong/turboquant-mlx-full) | arXiv:2406.04723 (⚠ different from POC's 2504.19874) | Wrapper API: `make_turboquant_cache()` + `tq_generate()` | No (single `--kv-bits`) | No (NumPy?) | Not explicit |
| [yzamari/mlx-turboquant](https://github.com/yzamari/mlx-turboquant) | arXiv:2504.19874 (matches POC) | Custom Metal: `mx.fast.turboquant_attention` | Documented K=3/V=2 | **Yes** | **Yes** — flags "needle retrieval fails on compressed tokens" |
| [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) | arXiv:2504.19874 (matches POC; calls it "PolarQuant") | Drop-in `TurboQuantKVCache(bits=N)` — mlx-lm compatible | Channel-split fractional bits, but not K-vs-V asymmetric | No (pure MLX) | Methodology in REPORT.md |

### Decision: rachittshah primary, yzamari secondary

**Primary: [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant)**

Why:
- Same paper as the POC (arXiv:2504.19874) — methodology aligns
- Drop-in mlx-lm `KVCache` replacement — minimal harness code
- Methodology documented in REPORT.md — auditable
- uv-based install — matches our project stack
- Tested with Qwen3 and Llama-3.2 — relevant model families

Limitations to be aware of:
- **Asymmetric K-vs-V not natively supported** — `bits=N` applies uniformly. The POC's K4/V2 pattern would require either (a) symmetric 3-bit as a proxy (similar compression, possibly worse quality), or (b) patching the port. Will note in REPORT.md and pick approach during the spike.
- No Metal kernel — slower than yzamari at inference time. For the mini-spike we care about *quality + correctness*, not throughput; revisit framework choice in v0 scale-up if speed differences are meaningful.

**Secondary reference: [yzamari/mlx-turboquant](https://github.com/yzamari/mlx-turboquant)**

Use as a cross-check on quality claims. Its honest disclosure ("needle retrieval fails on compressed tokens") directly contradicts the POC's "K4/V2 at 5x = full-quality" headline. This *is* the question our benchmark should answer for real agentic workloads. Mini-spike findings will guide whether yzamari's pessimism or the POC's optimism is more accurate for tool-calling.

### Not chosen
- matt-k-wong — cites a different arXiv (2406.04723); needs verification of which TurboQuant variant it implements before trusting comparisons.

## Iris tool to test (Task #10)

Selected: **`web_search`** from `src/iris/tools/web_search.py`.

Rationale: simplest signature in iris's tool catalog (`query: str, time_range: str | None`); clear semantic expectations; the model has a "right answer" that's checkable.

### Tool schema (what a model must emit)

```json
{
  "name": "web_search",
  "description": "Run a DuckDuckGo search and return a structured block of results. When time_range is None the function infers a window from the query (e.g. 'weather today' → 'd'). Pass an explicit 'd'/'w'/'m'/'y' to override the inference, or an empty string to disable narrowing.",
  "parameters": {
    "query":      { "type": "string", "required": true,  "description": "The search query." },
    "time_range": { "type": "string", "required": false, "description": "One of 'd', 'w', 'm', 'y', or empty. Optional; inferred when omitted." }
  }
}
```

### Mini-spike prompt (the test case)

See [prompt-001.md](prompt-001.md).

## What "running the mini-spike" entails (Task #11 footprint)

- `uv add mlx-lm` — adds runtime dep (~150 MB of deps, mostly torch's transitive closure though MLX itself is light)
- Install rachittshah/mlx-turboquant (via `uv add git+https://github.com/rachittshah/mlx-turboquant` or pip-install-from-source)
- Download `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` (~4 GB) into `./models/`
- Write `src/agentic_evals/experiments/exp001/mini_toolcall.py` (~50 lines: load model, run prompt twice — TQ off then TQ on, capture outputs)
- Run twice; capture transcripts; hand-grade correctness; record timing + memory
- Write `docs/experiments/exp-001-mlx-kv-compression-toolcalling/spikes/0002-mini-toolcall-run/REPORT.md` with findings

Peak working memory while running 4-bit 7B on MLX: ~5-6 GB unified memory. Comfortable on 16 GB+ Macs.
