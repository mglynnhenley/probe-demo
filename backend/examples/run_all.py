#!/usr/bin/env python
"""Run all example/test files and report a summary.

Usage:
    python run_all.py
    PROBE_BACKEND_URL=http://my-server:8000 python run_all.py
"""

import subprocess
import sys
from pathlib import Path

examples = [
    "health.py",
    "models.py",
    "chat_completion.py",
    "chat_streaming.py",
    "probe_scores.py",
    "analyze.py",
]

here = Path(__file__).parent
results = []

for name in examples:
    print(f"\n── {name} {'─' * (40 - len(name))}")
    result = subprocess.run(
        [sys.executable, str(here / name)],
        cwd=here,
    )
    results.append((name, result.returncode == 0))

print("\n" + "═" * 44)
for name, ok in results:
    status = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  {status}  {name}")

passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} passed")
if passed < len(results):
    sys.exit(1)
