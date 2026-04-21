#!/usr/bin/env python
"""GET /v1/models and GET /v1/models/{id} — model listing and retrieval.

Usage:
    python models.py
"""

import sys

from _client import get_client

client = get_client()

# List
models = client.models.list()
assert len(models.data) >= 1, "no models returned"
model_id = models.data[0].id
print(f"PASS  GET /v1/models → {[m.id for m in models.data]}")

# Retrieve
retrieved = client.models.retrieve(model_id)
assert retrieved.id == model_id, f"id mismatch: {retrieved.id!r} != {model_id!r}"
print(f"PASS  GET /v1/models/{model_id} → {retrieved.id}")

# Retrieve unknown model should 404
import openai
try:
    client.models.retrieve("nonexistent-model")
    print("FAIL  GET /v1/models/nonexistent-model — expected 404, got success")
    sys.exit(1)
except openai.NotFoundError:
    print("PASS  GET /v1/models/nonexistent-model → 404 as expected")
