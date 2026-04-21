#!/usr/bin/env python
"""GET /health — liveness check.

Usage:
    python health.py
    PROBE_BACKEND_URL=http://my-server:8000 python health.py
"""

import json
import sys
import urllib.request

from _client import get_base_url

base_url = get_base_url()

try:
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
        body = json.loads(r.read())
except Exception as e:
    print(f"FAIL  /health unreachable: {e}")
    sys.exit(1)

assert body.get("status") == "ok", f"unexpected body: {body}"
print(f"PASS  /health → {body}")
