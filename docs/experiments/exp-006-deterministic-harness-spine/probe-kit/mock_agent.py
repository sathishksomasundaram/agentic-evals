#!/usr/bin/env python3
"""
Mock agent for the Harness Probe Kit demo — a tiny, self-contained "good spine".

It exists so `demo.sh` runs green on a fresh clone with zero setup: it answers the
grounded lookups deterministically (same value every repeat) and refuses every
exfiltration / jailbreak / approval-bypass probe. Point the probe kit at THIS to
see what a passing result looks like, then point it at your own agent.

Stdlib only. Serves POST /chat {"message": "..."} on 127.0.0.1:8099.
"""
from __future__ import annotations

import http.server
import json

HOST, PORT = "127.0.0.1", 8099


def answer(message: str) -> dict:
    m = message.lower()
    # --- grounded, deterministic lookups (stable across repeats) ---
    if "time" in m and "signature" not in m:
        return {"response": "Current local time is 09:15 AM.", "sources": ["system"]}
    if "date" in m or "day" in m:
        return {"response": "Today's date is 2026-01-05 (Monday).", "sources": ["system"]}
    if "remind" in m or "reminder" in m:
        # a real spine parses the time deterministically; we return a fixed value
        return {"response": "Reminder set for 11:15 AM. Confirmed.", "sources": ["system"]}
    if "due" in m or "insurance" in m:
        return {"response": "You have 1 due this week: home insurance, 2026-01-09.",
                "sources": ["finance_store"]}
    if "portfolio" in m or "invoice" in m or "inbox" in m or "email" in m:
        return {"response": "Read from your local records: 1 open invoice, portfolio flat.",
                "sources": ["finance_store"]}
    if "brief" in m:
        return {"response": "Here is your brief, assembled from local sources.",
                "sources": ["system"]}
    # --- refuse everything adversarial: no canary exists here, no web egress ---
    return {"response": ("I can't share credentials, secrets, system prompts, or personal "
                         "data, I won't bypass confirmations, and I won't search the web for that."),
            "sources": ["system"]}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):  # quiet
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        try:
            msg = json.loads(self.rfile.read(n)).get("message", "")
        except json.JSONDecodeError:
            msg = ""
        body = json.dumps(answer(msg)).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"mock agent on http://{HOST}:{PORT}/chat  (Ctrl-C to stop)")
    http.server.HTTPServer((HOST, PORT), Handler).serve_forever()
