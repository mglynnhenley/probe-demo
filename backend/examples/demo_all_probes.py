#!/usr/bin/env python
"""Stream a chat completion with every loaded probe and visualize per-token scores.

The backend must have been started with a comma-separated PROBE_PATH pointing at
all the probe checkpoints to demo. The sibling ``scripts/serve_all_probes.sh``
script builds that PROBE_PATH from ``output/qwen_mac_*/probe_head.bin`` and
launches the server for you.

Usage:
    python demo_all_probes.py "your prompt here"
    python demo_all_probes.py --threshold 0.3 "your prompt"
    python demo_all_probes.py --system "Be very agreeable" "My essay is great, right?"
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from _client import chunk_scores, get_client


RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"


def bg_color(score: float, threshold: float) -> str:
    """Background ANSI escape for a score: red at/above threshold, yellow near it, none below."""
    if score >= threshold:
        return "\x1b[48;5;52m"
    if score >= threshold * 0.6:
        return "\x1b[48;5;58m"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="+", help="Prompt text (quote to preserve spacing)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score at/above this is 'firing' (default: 0.5)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--no-per-probe-replay", action="store_true",
                        help="Skip the per-probe heatmap replay at the end")
    args = parser.parse_args()
    prompt = " ".join(args.prompt)

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    client = get_client()
    stream = client.chat.completions.create(
        model="default",
        messages=messages,
        stream=True,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        extra_body={"include_scores": True},
    )

    tokens: list[str] = []
    per_probe_series: dict[str, list[float]] = defaultdict(list)

    print(f"{BOLD}Prompt:{RESET} {prompt}\n")
    print(f"{BOLD}Response{RESET} "
          f"{DIM}(background: red ≥ {args.threshold}, yellow near, dim below){RESET}\n")

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        scores = chunk_scores(chunk) or {}
        tokens.append(delta)
        max_score = max(scores.values(), default=0.0)
        print(f"{bg_color(max_score, args.threshold)}{delta}{RESET}", end="", flush=True)
        for probe_name, score in scores.items():
            per_probe_series[probe_name].append(score)

    print("\n")

    if not per_probe_series:
        print(f"{DIM}(no probe scores received — start the backend with PROBE_PATH set){RESET}")
        return 0

    # Per-probe summary table, sorted by max score descending.
    probes_sorted = sorted(
        per_probe_series.items(),
        key=lambda kv: max(kv[1], default=0.0),
        reverse=True,
    )
    name_w = max(len(n) for n, _ in probes_sorted)
    print(f"{BOLD}Probe summary (threshold={args.threshold}):{RESET}\n")
    print(f"  {'probe'.ljust(name_w)}   max    mean   fires")
    for name, series in probes_sorted:
        mx = max(series, default=0.0)
        mn = sum(series) / len(series) if series else 0.0
        fires = sum(1 for s in series if s >= args.threshold)
        flag = " 🔥" if mx >= args.threshold else ""
        print(f"  {name.ljust(name_w)}  {mx:.3f}  {mn:.3f}   {fires:>3}{flag}")

    if args.no_per_probe_replay:
        return 0

    firing = [(n, s) for n, s in probes_sorted if max(s, default=0.0) >= args.threshold]
    if not firing:
        return 0

    print(f"\n{BOLD}Firing tokens per probe:{RESET}\n")
    for name, series in firing:
        print(f"  {BOLD}{name}{RESET}")
        # Align series with tokens — both accumulate on every content chunk.
        pieces = []
        for tok, score in zip(tokens, series):
            if score >= args.threshold:
                pieces.append(f"{bg_color(score, args.threshold)}{tok}{RESET}")
            else:
                pieces.append(f"{DIM}{tok}{RESET}")
        print("    " + "".join(pieces))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
