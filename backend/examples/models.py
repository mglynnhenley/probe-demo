#!/usr/bin/env python
"""GET /v1/models and GET /v1/models/{id} — model listing and retrieval.

Usage:
    python models.py
"""

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

# Any model ID is accepted (the server supports closed-source routing by model name)
retrieved_unknown = client.models.retrieve("nonexistent-model")
assert retrieved_unknown.id == "nonexistent-model", \
    f"unexpected id: {retrieved_unknown.id!r}"
print("PASS  GET /v1/models/nonexistent-model → accepted (closed-source routing supported)")
