#!/usr/bin/env python
"""Quick interactive demo — streams a response with inline probe scores.

For the full test suite run each example file directly, or:
    python run_all.py

Usage:
    python probe_chat.py [prompt]
    python probe_chat.py "Explain black holes in one sentence"
"""

import sys

from _client import get_client, chunk_scores

client = get_client()
prompt = " ".join(sys.argv[1:]) or "What is the capital of France?"

print(f"Prompt: {prompt}\n")

token_count = 0
for chunk in client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
    extra_body={"include_scores": True},
):
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)

        scores = chunk_scores(chunk)
        if scores:
            avg = sum(scores.values()) / len(scores)
            token_count += 1
            if token_count <= 5:
                print(f"\033[2m[{avg:.3f}]\033[0m", end="", flush=True)

print(f"\n\n({token_count} tokens with probe scores)")
