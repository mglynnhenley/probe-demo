#!/usr/bin/env python
"""POST /v1/chat/completions with stream=True — streaming chat.

Usage:
    python chat_streaming.py
"""

from _client import get_client

client = get_client()

# Basic streaming — collect all deltas
chunks = []
for chunk in client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    stream=True,
):
    delta = chunk.choices[0].delta.content if chunk.choices else None
    if delta:
        chunks.append(delta)

full_text = "".join(chunks)
assert full_text, "no content received"
print(f"PASS  basic streaming → {full_text!r}")

# Streamed output matches non-streaming output (temperature=0)
non_stream_response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say only the word: banana"}],
    temperature=0,
)
non_stream_text = non_stream_response.choices[0].message.content

stream_chunks = []
for chunk in client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say only the word: banana"}],
    temperature=0,
    stream=True,
):
    delta = chunk.choices[0].delta.content if chunk.choices else None
    if delta:
        stream_chunks.append(delta)
stream_text = "".join(stream_chunks)

assert stream_text == non_stream_text, \
    f"streaming/non-streaming mismatch:\n  stream:     {stream_text!r}\n  non-stream: {non_stream_text!r}"
print(f"PASS  streaming matches non-streaming → {stream_text!r}")

# First chunk has role=assistant, subsequent have content
role_chunks = []
content_chunks = []
for chunk in client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hi"}],
    stream=True,
):
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta.role:
        role_chunks.append(delta.role)
    if delta.content:
        content_chunks.append(delta.content)

assert role_chunks == ["assistant"], f"unexpected roles: {role_chunks}"
assert content_chunks, "no content chunks"
print(f"PASS  chunk structure (role then content)")
