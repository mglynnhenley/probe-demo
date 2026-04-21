#!/usr/bin/env python
"""POST /v1/chat/completions — non-streaming chat, covering standard parameters.

Usage:
    python chat_completion.py
"""

from _client import get_client

client = get_client()

# Basic completion
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
content = response.choices[0].message.content
assert content, "empty response"
assert response.usage.total_tokens > 0, "missing usage"
print(f"PASS  basic completion → {content!r}")

# System prompt
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a pirate. End every reply with 'Arrr!'."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
)
content = response.choices[0].message.content
assert content, "empty response"
print(f"PASS  system prompt → {content!r}")

# max_tokens
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Write me a very long essay about the ocean."}],
    max_tokens=10,
)
assert response.usage.completion_tokens <= 15, \
    f"max_tokens not respected: got {response.usage.completion_tokens} tokens"
print(f"PASS  max_tokens=10 → {response.usage.completion_tokens} completion tokens")

# temperature=0 is deterministic
r1 = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say the single word: hello"}],
    temperature=0,
)
r2 = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say the single word: hello"}],
    temperature=0,
)
assert r1.choices[0].message.content == r2.choices[0].message.content, \
    f"temperature=0 not deterministic:\n  {r1.choices[0].message.content!r}\n  {r2.choices[0].message.content!r}"
print(f"PASS  temperature=0 determinism → {r1.choices[0].message.content!r}")

# Multi-turn conversation
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "My name is Claude."},
        {"role": "assistant", "content": "Hello Claude, nice to meet you!"},
        {"role": "user", "content": "What is my name?"},
    ],
)
content = response.choices[0].message.content
assert content, "empty response"
print(f"PASS  multi-turn → {content!r}")
