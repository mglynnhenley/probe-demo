# Probe Demo Frontend

A small React chat UI that demos the probe-demo backend. Each assistant token is
color-tinted by its probe score so you can see where the probes fire as the
response streams in.

## Run

```bash
cd frontend
npm install
npm run dev
```

The UI defaults to `http://localhost:8000` and posts to
`POST /v1/chat/completions` with `model: "gpt-4.1"` and `stream: true`. Edit the
backend URL in the top bar at runtime.

## What it does

- Streams chat completions from the backend (SSE).
- Renders each generated token as a `<span>` whose background opacity tracks
  the active probe's score for that token.
- Sidebar lists every probe seen in the most recent assistant message with a
  per-token sparkline, mean, max, and the peak token. Click a probe to switch
  which one tints the response.
- Hover any token to see the full per-probe scores in the tooltip.

## Notes

- The backend's `model` field is essentially a label — the real model is set at
  server startup via `MODEL_NAME`. The UI sends `gpt-4.1` so the demo reads
  correctly; the probe scores come from whatever model the backend has loaded.
- If no probes are loaded (no `PROBE_PATH`), the panel says so and the chat
  still works.
