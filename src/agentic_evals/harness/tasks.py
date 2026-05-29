"""The tool-calling evaluation task: prompts + grading.

This is experiment-001's task definition, factored out so future experiments and
the platform can reuse it or compare against it. A "task" here = the prompts that
pose the test + a grader that turns raw model output into a verdict.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT_BASE = """You are a helpful assistant. You have access to one tool:

web_search(query: str, time_range: str | None = None) -> str
  Description: Run a DuckDuckGo search and return results.
  Parameters:
    query (required, string): the search query
    time_range (optional, string): one of "d" (day), "w" (week), "m" (month),
      "y" (year), or empty string. Defaults to inferred-from-query when omitted.

When you need to use the tool, respond with EXACTLY this JSON format and nothing else:

  {"tool": "web_search", "args": {"query": "<your query>", "time_range": "<bucket-or-null>"}}

When you have the answer, respond normally without JSON."""

USER_PROMPT = "What's the weather in San Francisco right now?"

# Realistic-feeling technical filler. Mentions the tool's domain so the
# context is plausibly "background reading" an agent would have ingested.
FILLER_PARAGRAPH = (
    "The DuckDuckGo search engine indexes web content using a combination of "
    "crawler-based discovery and partner-API ingestion from sources including "
    "Bing's web index, its own crawler, and curated knowledge bases such as "
    "Wikipedia. Results are ranked by a relevance signal that combines anchor-"
    "text quality with link-graph reachability and a freshness decay factor. "
    "The time_range parameter constrains the date filter applied at query "
    "time, mapped from natural-language hints into one of four buckets: daily, "
    "weekly, monthly, or yearly. The system supports query rewriting for "
    "common typos and offers instant-answer overlays for queries that match "
    "structured data sources such as weather, currency conversion, and "
    "calculator expressions. Privacy is preserved by stripping search terms "
    "from referrer headers and avoiding third-party trackers. Search results "
    "are returned as structured documents containing the URL, title, and a "
    "short snippet, plus optional metadata such as publication date and a "
    "language hint. "
)


def build_long_system(target_filler_tokens: int, tokenizer: Any) -> str:
    """Return a long/dilute SYSTEM message: the tool definition + enough filler
    to reach roughly `target_filler_tokens`. Callers apply the chat template.
    """
    filler = ""
    while len(tokenizer.encode(filler)) < target_filler_tokens:
        filler += FILLER_PARAGRAPH
    return SYSTEM_PROMPT_BASE + "\n\nBackground reading (for context):\n\n" + filler


def grade(output: str) -> str:
    """Grade a generation against the expected `web_search` tool call.

    Returns one of: PASS (correct tool call), COLLAPSE (degenerate token loop),
    LOST_FORMAT (answered the topic without a tool call), or OTHER.
    Case-insensitive; anchored on the `"tool"` key + `web_search` + the topic.
    """
    o = output.strip()
    o_low = o.lower()
    has_tool_key = '"tool"' in o or "'tool'" in o
    has_websearch = "web_search" in o_low
    has_search_topic = "san francisco" in o_low or "weather" in o_low
    if has_tool_key and has_websearch and has_search_topic:
        return "PASS"
    if o.count("!") > 50 or "!!!!" in o:
        return "COLLAPSE"
    if has_search_topic and not has_tool_key:
        return "LOST_FORMAT"
    return "OTHER"
