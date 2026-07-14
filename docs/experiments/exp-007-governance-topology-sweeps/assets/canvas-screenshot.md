# canvas-screenshot.png — placeholder

Drop the marketing shot of the Governed Chat Spine on the **Harness Lab canvas**
here as `canvas-screenshot.png`. The README embeds it as
`assets/canvas-screenshot.png`.

How it's captured (the real UI, ~1 minute):

```bash
# from the Harness Lab repo
cd backend  && uvicorn app.main:app --port 8000 &     # API
cd frontend && npm run dev &                           # UI on :5173
# open http://localhost:5173 → Pattern Library → "Governed Chat Spine" →
# Open in Builder → click fit-view → screenshot the canvas
```

The shot shows the whole product — the node palette, the governed chat spine
graph (with the three swept gates: **Data classifier**, **Tier router**,
**Response judges / llm_judge**), and the inspector where every field is a sweep
axis. That's the marketing point: the harness is a graph you compose, inspect,
and sweep.
