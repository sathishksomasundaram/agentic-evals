# Mini-spike prompt 001 — `web_search` invocation

## Test target
Get the model to emit a structurally correct tool call to iris's `web_search` tool, given a user query that clearly requires it.

## Test prompt

System prompt (concise; designed to fit small-model context budgets):

```
You are a helpful assistant. You have access to one tool:

web_search(query: str, time_range: str | None = None) -> str
  Description: Run a DuckDuckGo search and return results.
  Parameters:
    query (required, string): the search query
    time_range (optional, string): one of "d" (day), "w" (week), "m" (month),
      "y" (year), or empty string. Defaults to inferred-from-query when omitted.

When you need to use the tool, respond with EXACTLY this JSON format and nothing
else:

  {"tool": "web_search", "args": {"query": "<your query>", "time_range": "<bucket-or-null>"}}

When you have the answer, respond normally without JSON.
```

User prompt:

```
What's the weather in San Francisco right now?
```

## Expected output (ground truth for hand-grading)

The model should emit a JSON tool call. Acceptable forms:

✅ Strict pass (preferred):
```json
{"tool": "web_search", "args": {"query": "weather San Francisco", "time_range": "d"}}
```

✅ Acceptable variations:
- `query` may be paraphrased: "current weather in San Francisco", "San Francisco weather today" — any reasonable rendering of the user intent
- `time_range` may be `"d"`, `"today"`, `null`, or omitted entirely (since the query says "right now" — inference would pick "d")
- May include a `name` field instead of `tool` if cleanly mapped to the schema

❌ Failure modes to grade as fail:
- Model answers the weather question directly (no tool call) — hallucination
- Model emits malformed JSON (unclosed braces, single quotes only, etc.)
- Wrong tool name (`search`, `web`, `lookup`, etc.)
- Missing required `query` arg
- Outputs `time_range` as a non-string non-null (e.g., 1, true, list)
- Extraneous prose around the JSON (e.g., "Sure! Here's the call: { ... }")

## What we're measuring (mini-spike)

| Metric | Baseline (TQ OFF) | TQ ON (K4/V2 or 3-bit symmetric proxy) |
|---|---|---|
| Tool-call correctness | pass / fail / partial — hand-graded | pass / fail / partial — hand-graded |
| Generated tokens (rough count) | observed | observed |
| Wall-clock latency (first response) | wall clock | wall clock |
| Peak memory while running | `ps`/Activity Monitor reading | same |
| KV-cache memory if measurable | optional | optional |

## What we're NOT measuring (yet)

- Multi-turn behavior
- Multiple tools / tool-selection accuracy
- Statistical significance (n=1 in mini-spike)
- Different sampling params (temp=0 throughout)
- Throughput at scale

## Why one prompt is enough for the mini-spike

We are not building a benchmark in this spike. We are answering: *"can our test apparatus produce signal we can interpret?"* If a single round-trip works, we scale to v0 with confidence. If it produces gibberish, we change approach before investing in 18-30 rows.
