#!/usr/bin/env python
"""Analyze mode — score pre-existing text token-by-token without generating anything.

Pass model="<anything>-analyze". The server runs the last assistant message through
the local model as a prefill pass and returns per-token probe scores.

Non-streaming: choices[0].message.content contains the original text reconstructed
from per-token decodes; scores is a list[float] per probe aligned with those tokens.

Streaming: delta.content is always empty — only scores stream per chunk.

Usage:
    python analyze.py
"""

from _client import get_client, response_scores

client = get_client()

TEXT = (
    "Alexander Graham Bell is widely credited with inventing the telephone in 1876, "
    "though Elisha Gray filed a similar patent on the same day."
)
USER_PROMPT = "Who invented the telephone?"

# Non-streaming analyze
response = client.chat.completions.create(
    model="default-analyze",
    messages=[
        {"role": "user", "content": USER_PROMPT},
        {"role": "assistant", "content": TEXT},
    ],
    extra_body={"include_scores": True},
)

content = response.choices[0].message.content
assert isinstance(content, str), f"expected str content, got {type(content)}"
# content is the original text reconstructed from per-token decodes; it should
# be non-empty and closely match TEXT (minor whitespace differences are possible)
assert len(content) > 0, "analyze non-streaming returned empty content"

scores = response_scores(response)
assert scores is not None, "no scores returned — is PROBE_PATH set on the server?"

for probe_name, token_scores in scores.items():
    assert len(token_scores) > 0, f"empty scores for probe {probe_name!r}"
    assert all(0.0 <= s <= 1.0 for s in token_scores), \
        f"scores out of [0,1] for probe {probe_name!r}"
    print(
        f"PASS  analyze non-streaming [{probe_name}]: "
        f"{len(token_scores)} tokens, range {min(token_scores):.3f}–{max(token_scores):.3f}"
    )

# Streaming analyze — scores arrive per token, content delta is always empty
scored_chunks: list[dict[str, float]] = []
content_chunks: list[str] = []

for chunk in client.chat.completions.create(
    model="default-analyze",
    messages=[
        {"role": "user", "content": USER_PROMPT},
        {"role": "assistant", "content": TEXT},
    ],
    stream=True,
    extra_body={"include_scores": True},
):
    delta = chunk.choices[0].delta.content if chunk.choices else None
    if delta:
        content_chunks.append(delta)
    s = (chunk.model_extra or {}).get("scores")
    if s is not None:
        scored_chunks.append(s)

assert not content_chunks, \
    f"analyze stream should produce no content, got {content_chunks!r}"
assert scored_chunks, "no scored chunks received in streaming analyze"
assert all(0.0 <= v <= 1.0 for chunk in scored_chunks for v in chunk.values()), \
    "streaming analyze scores outside [0, 1]"
print(
    f"PASS  analyze streaming: {len(scored_chunks)} scored chunks, "
    f"probes: {sorted(scored_chunks[0].keys())}"
)
