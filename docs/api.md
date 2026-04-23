# Probe-Demo Chat API — Spec for Client Apps

An OpenAI-compatible chat completions API with a single extension: per-token
**probe scores** returned alongside generated text. Any client that already
talks to OpenAI's `/v1/chat/completions` will work unchanged; probe scores
appear in one extra field.

- **Default base URL:** `http://localhost:8000`
- **OpenAI-compatible root:** `http://localhost:8000/v1`
- **Auth:** none. The server accepts any `Authorization` header (OpenAI SDKs
  require one, so pass `"unused"` or similar).
- **CORS:** `*` for origins, methods, and headers.
- **Transport:** HTTP/1.1 JSON. Streaming uses Server-Sent Events.

---

## Endpoints

### `GET /health`

Liveness probe.

**Response 200 (application/json)**

```json
{ "status": "ok" }
```

---

### `GET /v1/models`

Lists the single served model. Exists for OpenAI SDK compatibility.

**Response 200**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1714000000,
      "owned_by": "organization"
    }
  ]
}
```

### `GET /v1/models/{model_id}`

Returns the model card if `model_id` matches the served model, otherwise 404.
`model_id` may contain slashes (e.g. `Qwen/Qwen2.5-0.5B-Instruct`).

---

### `POST /v1/chat/completions`

Chat completion with optional streaming and optional probe scores.

#### Request body

| Field             | Type                   | Default | Notes |
| ----------------- | ---------------------- | ------- | ----- |
| `model`           | string                 | —       | Accepted but ignored — the server returns whichever model it has loaded. Passing `"default"` is fine. |
| `messages`        | `ChatMessage[]`        | —       | `role ∈ {"system", "user", "assistant"}`, `content: string`. A default system prompt is prepended if the first message isn't a system message. |
| `stream`          | boolean                | `false` | When `true`, response is SSE (see below). |
| `max_tokens`      | integer \| null        | `null`  | Token cap for the completion. `null` means "until EOS or context limit". |
| `temperature`     | number                 | `0.7`   | `<= 0` ⇒ greedy. |
| `top_p`           | number                 | `0.9`   | Nucleus sampling; only applied when `temperature > 0`. |
| `include_scores`  | boolean                | `true`  | **Extension.** When `false`, the server skips probe evaluation and omits the `scores` field. |
| `probe_path`      | string \| null         | `null`  | **Extension.** Absolute path (or trainer output directory) to a `ValueHeadProbe` checkpoint. Falls back to the server's `PROBE_PATH` env var. |

Unknown top-level fields are rejected by Pydantic (422).

#### Non-streaming response (`stream: false`, 200)

```json
{
  "id": "chatcmpl-ab12cd34ef56",
  "object": "chat.completion",
  "created": 1714000000,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Paris." },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 23,
    "completion_tokens": 2,
    "total_tokens": 25
  },
  "scores": {
    "hallucination": [0.01, 0.02]
  }
}
```

**`scores` (extension)** — `{ probe_name: float[] }`. Each list has length
`usage.completion_tokens`, aligned element-wise with the generated tokens.
The server asserts this length invariant and returns 500 if it's violated.
Absent when `include_scores` is `false` or no probes are loaded.

#### Streaming response (`stream: true`)

- `Content-Type: text/event-stream`
- Each SSE event is `data: <json>\n\n`.
- Stream terminates with a literal `data: [DONE]\n\n` line.

Chunks have this shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion.chunk",
  "created": 1714000000,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "delta": { "content": "Par" },
      "finish_reason": null
    }
  ],
  "scores": { "hallucination": 0.01 }
}
```

Chunk sequence for a single completion:

1. **Role chunk** — `delta = { "role": "assistant" }`, no `scores`.
2. **Content chunks** — one per generated token. `delta.content` holds the
   token's text fragment and `scores` is `{ probe_name: float }` for that
   token. `scores` is `null` when `include_scores=false` or no probes exist.
3. **Finish chunk** — `delta = {}`, `finish_reason = "stop"`, no `scores`.
4. **`data: [DONE]`** sentinel line.

If the HTTP connection drops, the server aborts generation server-side.

#### Error codes

| Status | When |
| ------ | ---- |
| 400 / 422 | Malformed JSON, unknown field, bad enum value. |
| 404 | `GET /v1/models/{id}` when `id` doesn't match the loaded model. |
| 500 | Internal error during generation (e.g. probe checkpoint missing). |

Errors follow FastAPI's default JSON shape (`{"detail": "..."}`).

---

## Probe score semantics

- **Alignment.** Score index `i` corresponds to generated token `i` (0-indexed).
  Streaming scores arrive in lockstep with their content delta; non-streaming
  scores are returned as parallel arrays.
- **Range.** `[0.0, 1.0]` — sigmoid of the probe's logit. Higher = stronger
  evidence the token belongs to the probe's target class (e.g. hallucination,
  sycophancy). Thresholds are probe-specific; treat as a calibrated-ish signal,
  not a hard label.
- **Semantics of the captured hidden state.** The probe is scored on the
  hidden state **after** the token is consumed — the state that the model
  would use to predict the following token. This matches how the probe was
  trained, so server-side scores are directly comparable with offline
  training-time probe outputs.
- **Multiple probes.** If the server was started with
  `PROBE_PATH=probe_a,probe_b` (comma-separated list), every scored response
  contains both keys. Per-request `probe_path` overrides and returns a single
  probe.
- **Probe naming.** The key in `scores` is derived from the probe checkpoint
  path: the parent directory name when the file is `probe_head.bin`
  (standard trainer layout), otherwise the filename stem.

---

## Client examples

### `curl` — non-streaming

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 16,
    "temperature": 0,
    "include_scores": true
  }' | jq .
```

### `curl` — streaming (SSE)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Python — OpenAI SDK (non-streaming)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    temperature=0,
    extra_body={"include_scores": True},  # default; shown for clarity
)

# Standard OpenAI fields
print(response.choices[0].message.content)

# Extension: per-probe, per-token scores aligned with generated tokens.
# The OpenAI SDK parks unknown fields on model_extra.
scores = (response.model_extra or {}).get("scores") or {}
for probe_name, series in scores.items():
    print(probe_name, series)
```

### Python — OpenAI SDK (streaming)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain black holes briefly."}],
    stream=True,
    extra_body={"include_scores": True},
)

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
    per_token = (chunk.model_extra or {}).get("scores")  # {probe_name: float} | None
    if per_token:
        # inline-annotate, log to a store, render a heatmap, etc.
        ...
```

### JavaScript / TypeScript (streaming via fetch)

```ts
const res = await fetch("http://localhost:8000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "default",
    messages: [{ role: "user", content: "Hello" }],
    stream: true,
    include_scores: true,
  }),
});

const reader = res.body!.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  // SSE events are separated by blank lines
  let idx;
  while ((idx = buffer.indexOf("\n\n")) !== -1) {
    const frame = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 2);
    if (!frame.startsWith("data:")) continue;
    const payload = frame.slice(5).trim();
    if (payload === "[DONE]") return;

    const chunk = JSON.parse(payload);
    const delta = chunk.choices?.[0]?.delta?.content;
    const scores = chunk.scores; // { probe_name: number } | null
    if (delta) process.stdout.write(delta);
    if (scores) { /* ... */ }
  }
}
```

---

## Operational notes

- **Model selection.** The server loads exactly one model at startup. Pass
  anything in the request's `model` field — the server echoes back whatever
  it actually loaded.
- **Serialization.** Generation is serialized behind a lock; concurrent
  requests are processed sequentially.
- **Token budget.** `max_tokens` is clamped to the model's remaining context
  window (`MAX_MODEL_LEN - prompt_tokens`). Passing a larger value isn't an
  error — it's silently truncated.
- **Stop condition.** Generation stops on EOS, on reaching `max_tokens`, or
  if the HTTP client disconnects.

For background on the probe itself (what the scores mean, how the head is
trained), see `docs/probe_concepts.md`.
