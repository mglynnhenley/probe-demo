# Probe-Demo Backend API

OpenAI-compatible chat completions API that attaches per-token hallucination probe scores to every response. The server runs a local open-weight model (Gemma 4 31B) and returns probe scores alongside each token as it is generated.

## Quickstart

```bash
# Start the server (requires PROBE_PATH and MODEL_NAME set)
python main.py

# Or via Modal
modal deploy modal_backend.py
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Generate text with probe scores |

---

## Probe scores

Every response includes a `scores` field alongside the standard OpenAI fields. Because the OpenAI SDK strips unknown fields by default, access them via `model_extra`:

```python
# Non-streaming: scores["probe_name"] is a list[float] aligned with completion tokens
scores = (response.model_extra or {}).get("scores")

# Streaming: scores["probe_name"] is a single float for the token in this chunk
scores = (chunk.model_extra or {}).get("scores")
```

Score values are probabilities in `[0, 1]`. Higher values indicate higher predicted hallucination likelihood for that token.

---

## Modes

### 1. Local generation

Pass the local model name (or `"default"`) as the `model`. The server generates text and probes each token as it is decoded.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is the boiling point of water?"}],
    extra_body={"include_scores": True},
)

content = response.choices[0].message.content
scores = (response.model_extra or {}).get("scores")
# scores == {"hallucination": [0.03, 0.11, 0.08, ...]}  one float per completion token
```

### 2. Closed-source generation

Pass any model name that is not the local model (e.g. an OpenRouter or Anthropic model ID). The server streams tokens from that external model and forces each one through the local model to extract probe scores. The `model` field in the response reflects the external model.

Requires `OPENROUTER_API_KEY` and/or `ANTHROPIC_API_KEY` set in the server environment.

```python
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Who invented the telephone?"}],
    extra_body={"include_scores": True},
)

content = response.choices[0].message.content   # text from Claude
scores = (response.model_extra or {}).get("scores")
# scores == {"hallucination": [0.02, 0.05, ...]}  probed via local model
```

### 3. Analyze (probe existing text)

Append `-analyze` to any model name. Instead of generating new text, the server runs the provided assistant message through the local model as a prefill pass and returns per-token probe scores for that text.

This is useful for scoring text that was already generated elsewhere (e.g. from a closed-source API call you made independently).

**Non-streaming:** `choices[0].message.content` contains the original assistant text reconstructed from per-token decodes. `scores` is a `dict[str, list[float]]` with one float per token.

**Streaming:** each chunk carries a single-token `scores` dict. `choices[0].delta.content` is always empty — tokens are not re-emitted, only scores stream.

```python
# Non-streaming analyze
response = client.chat.completions.create(
    model="default-analyze",
    messages=[
        {"role": "user", "content": "Who invented the telephone?"},
        {"role": "assistant", "content": "Alexander Graham Bell invented the telephone in 1876."},
    ],
    extra_body={"include_scores": True},
)

content = response.choices[0].message.content  # original text, reconstructed from tokens
scores = (response.model_extra or {}).get("scores")
# scores == {"hallucination": [0.01, 0.02, 0.03, 0.71, 0.68, 0.05, ...]}
# one float per token in the assistant message

# Streaming analyze — scores only, no content deltas
for chunk in client.chat.completions.create(
    model="default-analyze",
    messages=[
        {"role": "user", "content": "Who invented the telephone?"},
        {"role": "assistant", "content": "Alexander Graham Bell invented the telephone in 1876."},
    ],
    stream=True,
    extra_body={"include_scores": True},
):
    scores = (chunk.model_extra or {}).get("scores")
    if scores:
        print(scores)  # {"hallucination": 0.03}
```

The `-analyze` suffix works with any model name as the prefix:

```
"default-analyze"                         # analyze with local model
"anthropic/claude-sonnet-4-6-analyze"     # prefix is ignored, still uses local model for probing
```

---

## Streaming

All three modes support `stream=True`. Probe scores are attached to each chunk as a single float per active probe:

```python
for chunk in client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain DNA replication."}],
    stream=True,
    extra_body={"include_scores": True},
):
    delta = chunk.choices[0].delta.content or ""
    scores = (chunk.model_extra or {}).get("scores")
    if scores:
        prob = scores.get("hallucination", 0.0)
        print(f"{delta}[{prob:.2f}]", end="", flush=True)
```

Role and finish-reason chunks do not carry scores — `model_extra.get("scores")` returns `None` on those.

---

## cURL examples

**Health check:**
```bash
curl http://localhost:8000/health
```

**Non-streaming with probe scores:**
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "include_scores": true,
    "max_tokens": 50
  }' | jq '{content: .choices[0].message.content, scores: .scores}'
```

**Streaming with probe scores:**
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Who was Napoleon Bonaparte?"}],
    "stream": true,
    "include_scores": true,
    "max_tokens": 80
  }'
```

**Analyze existing text:**
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default-analyze",
    "messages": [
      {"role": "user", "content": "Who invented the telephone?"},
      {"role": "assistant", "content": "Alexander Graham Bell invented the telephone in 1876."}
    ],
    "include_scores": true
  }' | jq '.scores'
```

**Closed-source model via OpenRouter:**
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "What is the speed of light?"}],
    "include_scores": true,
    "max_tokens": 60
  }' | jq '{model: .model, content: .choices[0].message.content, scores: .scores}'
```

---

## Request reference

Standard OpenAI fields that are forwarded to the local vLLM model:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | `str` | required | Any value; triggers routing (see Modes above) |
| `messages` | `list` | required | Standard OpenAI message format |
| `stream` | `bool` | `false` | |
| `max_tokens` | `int` | `null` | Also accepts `max_completion_tokens` |
| `temperature` | `float` | `1.0` | |
| `top_p` | `float` | `1.0` | |
| `n` | `int` | `1` | |
| `stop` | `str \| list[str]` | `null` | |
| `presence_penalty` | `float` | `0.0` | |
| `frequency_penalty` | `float` | `0.0` | |
| `seed` | `int` | `null` | |
| `logprobs` | `bool` | `null` | |
| `top_logprobs` | `int` | `null` | |
| `logit_bias` | `dict` | `null` | Keys are stringified token IDs |

Probe-specific fields (passed via `extra_body` in the OpenAI SDK, or as top-level JSON in cURL):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `include_scores` | `bool` | `true` | Set to `false` to suppress probe scoring |
| `probe_path` | `str` | `null` | Path to a specific probe checkpoint; falls back to `PROBE_PATH` env var |
| `closed_source_model` | `str` | `null` | Explicit override for the external model; overrides `model`-based routing |
| `block_size` | `int` | `1` | Tokens probed concurrently in closed-source mode; higher values trade latency for throughput |

---

## Response reference

Responses are standard OpenAI `ChatCompletion` / `ChatCompletionChunk` objects with one extra field:

**Non-streaming** — `scores` is a `dict[str, list[float]]` where each list is aligned with completion tokens:
```json
{
  "id": "chatcmpl-abc123",
  "model": "google/gemma-4-31B-it",
  "choices": [{"message": {"role": "assistant", "content": "Paris."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
  "scores": {"hallucination": [0.03, 0.07, 0.02, 0.04]}
}
```

**Streaming** — each content chunk carries `scores` as a `dict[str, float]` (one value per probe for that token):
```json
{"choices": [{"delta": {"content": " Paris"}}], "scores": {"hallucination": 0.03}}
```

In analyze mode (non-streaming), `choices[0].message.content` is the original assistant text reconstructed from per-token decodes, and `scores` covers those same tokens. In streaming analyze, `delta.content` is always empty — only `scores` streams.
