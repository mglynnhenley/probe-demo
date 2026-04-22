#!/usr/bin/env python3
"""Summary figure of probe outcomes across both policies × both splits.

Loads the four probe_scores_*.npz artifacts produced by evaluate_probe_roc.py
and produces a 2×2 grid of per-sequence-score distributions plus ROC curves,
so you can eyeball at a glance how well the probe separates violating vs
clean completions and whether that separation survives the held-out split.

Output: output/probe_results_summary.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    {
        "policy": "tipping_off",
        "split": "train+val",
        "path": ROOT / "output/qwen_mac_tipping_off/probe_scores_all.npz",
    },
    {
        "policy": "tipping_off",
        "split": "held-out",
        "path": ROOT / "output/qwen_mac_tipping_off/probe_scores_tipping_off_heldout_annotated.npz",
    },
    {
        "policy": "hallucinated_citations",
        "split": "train+val",
        "path": ROOT / "output/qwen_mac_hallucinated/probe_scores_all.npz",
    },
    {
        "policy": "hallucinated_citations",
        "split": "held-out",
        "path": ROOT / "output/qwen_mac_hallucinated/probe_scores_hallucinated_citations_heldout_annotated.npz",
    },
]

POLICIES = ["tipping_off", "hallucinated_citations"]
SPLITS = ["train+val", "held-out"]


def load(run):
    d = np.load(run["path"], allow_pickle=True)
    return {
        "seq_scores": d["seq_scores"],
        "seq_labels": d["seq_labels"],
        "token_auc": float(d["roc_auc"][0]),
        "seq_auc": float(d["seq_roc_auc"][0]),
        "roc_fpr": d["roc_fpr"],
        "roc_tpr": d["roc_tpr"],
    }


def main():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    # Left two columns: per-sequence score distributions.
    # Right two columns: ROC curves (token).
    for row, policy in enumerate(POLICIES):
        for col, split in enumerate(SPLITS):
            run = next(r for r in RUNS if r["policy"] == policy and r["split"] == split)
            data = load(run)

            # --- Per-sequence distribution ---
            ax = axes[row, col]
            violating = data["seq_scores"][data["seq_labels"] == 1]
            clean = data["seq_scores"][data["seq_labels"] == 0]
            bins = np.linspace(0, 1, 21)
            ax.hist(clean, bins=bins, alpha=0.55, color="#5b8def", label=f"clean (n={len(clean)})", edgecolor="white")
            ax.hist(violating, bins=bins, alpha=0.55, color="#d64545", label=f"violating (n={len(violating)})", edgecolor="white")
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("max prob per completion")
            ax.set_ylabel("# completions")
            ax.set_title(f"{policy} — {split}\nseq AUC={data['seq_auc']:.3f}  token AUC={data['token_auc']:.3f}")
            ax.legend(loc="upper center", fontsize=8)
            ax.grid(True, alpha=0.25)

            # --- ROC curve ---
            ax_roc = axes[row, col + 2]
            ax_roc.plot(data["roc_fpr"], data["roc_tpr"], linewidth=1.8, color="#222", label=f"token AUC={data['token_auc']:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
            ax_roc.set_xlim(0, 1)
            ax_roc.set_ylim(0, 1)
            ax_roc.set_xlabel("false positive rate")
            ax_roc.set_ylabel("true positive rate")
            ax_roc.set_title(f"ROC — {policy}\n{split}")
            ax_roc.legend(loc="lower right", fontsize=8)
            ax_roc.grid(True, alpha=0.25)

    fig.suptitle(
        "Probe results: Qwen2.5-0.5B-Instruct, layer 10, MLP head — 100-row pilot",
        fontsize=12,
        fontweight="bold",
    )

    out = ROOT / "output/probe_results_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
