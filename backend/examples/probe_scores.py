#!/usr/bin/env python
"""Score extensions — per-probe, per-token scores on both streaming and
non-streaming responses.

Requires the backend to have PROBE_PATH set. Tests are skipped (not failed)
when no probes are loaded.

Usage:
    python probe_scores.py
"""

from _client import get_client, chunk_scores, response_scores

client = get_client()

MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


# Non-streaming scores
response = client.chat.completions.create(
    model="default",
    messages=MESSAGES,
    extra_body={"include_scores": True},
)
scores = response_scores(response)

if scores is None:
    print("SKIP  non-streaming scores (no probe loaded on server)")
else:
    n_tokens = response.usage.completion_tokens
    assert all(len(v) == n_tokens for v in scores.values()), \
        f"score arrays not aligned with completion_tokens ({n_tokens}): " \
        f"{{{', '.join(f'{k}: {len(v)}' for k, v in scores.items())}}}"
    assert all(0.0 <= s <= 1.0 for v in scores.values() for s in v), \
        "scores outside [0, 1]"
    for name, values in scores.items():
        print(f"PASS  non-streaming scores[{name!r}]: {len(values)} tokens, "
              f"range {min(values):.3f}–{max(values):.3f}")

# Scores absent when include_scores=False
response_no_scores = client.chat.completions.create(
    model="default",
    messages=MESSAGES,
    extra_body={"include_scores": False},
)
assert response_scores(response_no_scores) is None, \
    "scores present despite include_scores=False"
print("PASS  scores absent when include_scores=False")

# Streaming scores
scored_chunks: list[dict[str, float]] = []
for chunk in client.chat.completions.create(
    model="default",
    messages=MESSAGES,
    stream=True,
    extra_body={"include_scores": True},
):
    s = chunk_scores(chunk)
    if s is not None:
        scored_chunks.append(s)

if not scored_chunks:
    print("SKIP  streaming scores (no probe loaded on server)")
else:
    assert all(0.0 <= v <= 1.0 for chunk in scored_chunks for v in chunk.values()), \
        "streaming scores outside [0, 1]"
    probe_names = set(scored_chunks[0].keys())
    assert all(set(c.keys()) == probe_names for c in scored_chunks), \
        "probe names inconsistent across chunks"
    print(f"PASS  streaming scores: {len(scored_chunks)} scored chunks, "
          f"probes: {sorted(probe_names)}")
