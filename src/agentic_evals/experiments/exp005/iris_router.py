"""IRIS production router logic, vendored for exp-005.

Frozen snapshot of ``project-iris`` ``src/iris/core/intent_router.py`` @
``3cb65b3`` — the system prompt, the JSON parser with vocabulary + intent→agent
consistency guards, and the keyword rule table (including the Phase 2 cal-05
calendar fix). Vendored so the experiment is reproducible from this repo alone;
any drift from production is a property of the snapshot, by design.

One experiment-side accommodation beyond production: ``parse_llm_route``
strips ``<think>…</think>`` blocks before JSON extraction, because some
candidate models (qwen3.5 family) think by default. Production's parser does
NOT do this — if a thinking model wins the shootout, that parser gap becomes
an IRIS bug to fix (tracked in the spike report).
"""

from __future__ import annotations

import json
import re

# ─── Keyword rule table (production order matters) ─────────────────────────

KEYWORD_RULES: list[tuple[str, str, str]] = [
    (
        r"\b("
        r"(?:what|tell\s+me)\s+(?:do\s+you\s+know|about)\s+(?:about\s+)?(?:me|myself)|"
        r"who\s+am\s+i|what'?s\s+my\s+(?:name|role|profession|focus|stack|"
        r"profile|background|preference|preferences|setup|style|"
        r"values?|mission|ambition|ambitions|priority|priorities|personality)|"
        r"what\s+do\s+i\s+(?:do|prefer|use|like|work\s+on|build|focus|value|believe)|"
        r"what\s+\w+\s+do\s+i\s+(?:prefer|use|like|work\s+with)|"
        r"(?:name|list|tell\s+me)\s+\w+(?:\s+\w+){0,4}\s+i\s+(?:prefer|use|work\s+with|like)|"
        r"what\s+(?:are|is)\s+my\s+(?:preferences?|stack|background|languages?|"
        r"goals?|interests?|profile|role|focus|values?|ambitions?|"
        r"priorit(?:y|ies)|principles?|mission|style)|"
        r"what\s+kind\s+of\s+\w+(?:\s+\w+){0,3}\s+(?:am|do)\s+i\b|"
        r"do\s+i\s+(?:use|prefer|like|work\s+with|believe|value)\s+\w+|"
        r"my\s+(?:profile|background|technical\s+preferences|stack|values?)\b"
        r")\b",
        "profile_query",
        "system",
    ),
    (
        r"\b(what\s+is\s+the\s+best|what'?s\s+the\s+best|"
        r"best\s+way\s+to|best\s+approach|best\s+practices?\s*(?:for|to)\b|"
        r"how\s+should\s+(?:i|we)\s+(?:approach|handle|implement|design|"
        r"structure|organize|integrate|do|use|build|convert|store|index)|"
        r"recommend(?:ed|ation)?\s+(?:approach|way|method|tool|library|solution)|"
        r"what\s+do\s+you\s+recommend|advise\s+on|"
        r"which\s+(?:tool|approach|method|library|way)\s+(?:is|should|would|do))\b",
        "general",
        "system",
    ),
    (r"^how\s+(do|can|would|should)\s+(i|we|one)\b", "general", "system"),
    (r"\b(help|what can you|how do|explain)\b", "help", "system"),
    (
        r"\b(sandbox|run[_\s-]?shell|use\s+the\s+sandbox\s+tool|"
        r"execute\s+in\s+the\s+sandbox)\b",
        "code_exec",
        "code_exec",
    ),
    (
        r"\b(pdf|excel|xlsx|csv|tsv|json|yaml|html|markdown|md|"
        r"image|png|jpg|jpeg|svg|chart|graph|plot|"
        r"file|document|report|spreadsheet|workbook|notebook)\b"
        r".{0,120}"
        r"\b(format|reformat|layout|spacing|style|article|polish|clean\s*up|"
        r"improve|fix|refine|rewrite|restyle|beautify)\b"
        r"|"
        r"\b(format|reformat|layout|spacing|style|article|polish|clean\s*up|"
        r"improve|fix|refine|rewrite|restyle|beautify)\b"
        r".{0,120}"
        r"\b(pdf|excel|xlsx|csv|tsv|json|yaml|html|markdown|md|"
        r"image|png|jpg|jpeg|svg|chart|graph|plot|"
        r"file|document|report|spreadsheet|workbook|notebook)\b",
        "code_exec",
        "code_exec",
    ),
    (
        r"\b(calculate|compute|plot)\b"
        r"|"
        r"\b(create|make|generate|produce|build|export|render|convert|run|execute|"
        r"get|fetch|refresh|update|summarize)\b"
        r".{0,80}"
        r"\b(pdf|excel|xlsx|csv|tsv|json|yaml|html|markdown|md|"
        r"image|png|jpg|jpeg|svg|chart|graph|plot|"
        r"file|document|report|spreadsheet|workbook|notebook|"
        r"script|command|"
        r"game|puzzle|simulation|demo|animation|playable)\b",
        "code_exec",
        "code_exec",
    ),
    (
        r"\b(build|create|make|develop|design|generate|produce|write|implement|code|refactor|debug|unittest)\b"
        r".{0,80}"
        r"\b(agent|tool|service|bot|system|app|application|script|pipeline|workflow|plugin|extension|function|class|module|program|test)\b",
        "coding",
        "coding_agent",
    ),
    (r"\b(code|implement|refactor|debug|unittest)\b|fix\s+bug", "coding", "coding_agent"),
    (
        r"\b(gmail|e-?mails?|inbox|mailbox|mail|messages?|send|reply|forward)\b",
        "communication",
        "email",
    ),
    (
        r"\b(calendar|schedule|meeting|appointment|event|remind|"
        r"standup|recurring|every\s+(?:day|weekday|week|morning|evening))\b",
        "calendar",
        "calendar",
    ),
    (r"\b(file|folder|document|upload|download|storage)\b", "files", "filemanager"),
    (r"\b(weather|temperature|forecast|rain|sunny)\b", "weather", "system"),
    (r"\b(time|date|clock|timezone|what time)\b", "system", "system"),
    (r"\b(search|find|look up|research|query)\b", "search", "rag"),
]


def classify_keyword(query: str) -> dict[str, object]:
    """Production KeywordClassifier semantics: first rule wins, else fallback."""
    lowered = query.lower()
    for pattern, intent, agent in KEYWORD_RULES:
        if re.search(pattern, lowered):
            return {"intent": intent, "agent": agent, "confidence": 0.85, "source": "keyword"}
    return {"intent": "general", "agent": "system", "confidence": 0.5, "source": "fallback"}


# ─── LLM router prompt + guarded parser ────────────────────────────────────

VALID_INTENTS = frozenset(
    {
        "communication", "calendar", "files", "coding", "code_exec",
        "search", "weather", "system", "help", "general", "profile_query",
    }
)
VALID_AGENTS = frozenset(
    {"email", "calendar", "filemanager", "coding_agent", "code_exec", "rag", "system"}
)
INTENT_TO_AGENT = {
    "communication": "email",
    "calendar": "calendar",
    "files": "filemanager",
    "coding": "coding_agent",
    "code_exec": "code_exec",
    "search": "rag",
    "weather": "system",
    "system": "system",
    "help": "system",
    "general": "system",
    "profile_query": "system",
}

ROUTER_SYSTEM_PROMPT = (
    "You are IRIS's request router. Classify the user message into a single JSON object.\n"
    "\n"
    "Output JSON shape (and NOTHING else — no prose, no markdown fences):\n"
    '{"intent": "<one of: communication, calendar, files, coding, code_exec, '
    'search, weather, system, help, general, profile_query>", '
    '"agent_type": "<one of: email, calendar, filemanager, coding_agent, '
    'code_exec, rag, system>", '
    '"is_multi_step": <true|false>, '
    '"confidence": <number between 0 and 1>}\n'
    "\n"
    "Rules:\n"
    "- 'code_exec' / 'code_exec': the user wants a concrete artifact PRODUCED "
    "(PDF, Excel, CSV, image, chart) or a script/calculation actually RUN. "
    "Examples: 'create a pdf summary of X', 'refresh AI news and give me an updated "
    "pdf with summaries', 'plot Y as a chart', 'convert this "
    "JSON to CSV', 'calculate the IRR of these cashflows'. Pick this over "
    "'coding' whenever execution + a deliverable is needed.\n"
    "- If the user explicitly asks to use the sandbox or run_shell tool, route "
    "to 'code_exec' / 'code_exec'.\n"
    "- 'coding' / 'coding_agent': writing, editing, debugging, refactoring code, or "
    "building software artifacts (CLIs, agents, services, scripts) — when the "
    "user wants the SOURCE CODE itself, not a finished file.\n"
    "- 'communication' / 'email': Gmail/mailbox/inbox setup, reading, extracting, "
    "summarizing, drafting, replying, or sending email.\n"
    "- 'calendar' / 'calendar': scheduling, appointments, reminders.\n"
    "- 'files' / 'filemanager': file/folder/document operations.\n"
    "- 'search' / 'rag': research, looking things up on the web, factual lookups.\n"
    "- 'weather' / 'system': weather/forecast/temperature questions (route to system; "
    "the system handler will fetch real-time data).\n"
    "- 'profile_query' / 'system': the user is asking about THEMSELVES — name, "
    "role, profession, preferences, technical stack, languages they use, what "
    "the agent knows about them. Examples: 'what's my name', 'who am I', "
    "'what do I do', 'what languages do I prefer', 'tell me about myself', "
    "'do I use Rust'. Routed to Tier 2 for faithful prompt-reading.\n"
    "- 'system' / 'system': general chat, time/date, identity, capabilities, IRIS itself.\n"
    "- 'help' / 'system': the user is asking for help or how to use something.\n"
    "- 'general' / 'system': anything else / unsure.\n"
    "- Set is_multi_step = true ONLY when the user asks for two distinct actions in one "
    "turn (e.g. 'check my email AND update the calendar').\n"
    "- Confidence: 0.9+ when the request is unambiguous, 0.5-0.7 when uncertain."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def parse_llm_route(raw: str) -> dict[str, object] | None:
    """Production parser + guards. Returns None when the keyword fallback
    would take over (parse failure, unknown vocab, intent/agent mismatch)."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = _THINK_BLOCK_RE.sub("", raw).strip()  # experiment-side accommodation
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None

    intent = str(loaded.get("intent", "")).strip().lower()
    agent = str(loaded.get("agent_type", "")).strip().lower()
    if intent not in VALID_INTENTS or agent not in VALID_AGENTS:
        return None
    if INTENT_TO_AGENT.get(intent) != agent:
        return None
    return {"intent": intent, "agent": agent, "source": "llm"}


def route_llm_with_fallback(raw: str, query: str) -> dict[str, object]:
    """Full production semantics: guarded LLM answer, keyword on any failure."""
    parsed = parse_llm_route(raw)
    if parsed is not None:
        return parsed
    result = classify_keyword(query)
    result["source"] = f"fallback-{result['source']}"
    return result
