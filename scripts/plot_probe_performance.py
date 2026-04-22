#!/usr/bin/env python3
"""Performance summary: MLP vs covseq head, both policies × both splits."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (policy, split, head, npz path)
RUNS = [
    ("tipping_off", "train+val", "mlp", ROOT / "output/qwen_mac_tipping_off/probe_scores_all.npz"),
    ("tipping_off", "train+val", "covseq", ROOT / "output/qwen_mac_tipping_off_covseq/probe_scores_all.npz"),
    ("tipping_off", "held-out", "mlp", ROOT / "output/qwen_mac_tipping_off/probe_scores_tipping_off_heldout_annotated.npz"),
    ("tipping_off", "held-out", "covseq", ROOT / "output/qwen_mac_tipping_off_covseq/probe_scores_tipping_off_heldout_annotated.npz"),
    ("hallucinated_citations", "train+val", "mlp", ROOT / "output/qwen_mac_hallucinated/probe_scores_all.npz"),
    ("hallucinated_citations", "train+val", "covseq", ROOT / "output/qwen_mac_hallucinated_covseq/probe_scores_all.npz"),
    ("hallucinated_citations", "held-out", "mlp", ROOT / "output/qwen_mac_hallucinated/probe_scores_hallucinated_citations_heldout_annotated.npz"),
    ("hallucinated_citations", "held-out", "covseq", ROOT / "output/qwen_mac_hallucinated_covseq/probe_scores_hallucinated_citations_heldout_annotated.npz"),
]


def load_auc(path: Path):
    d = np.load(path, allow_pickle=True)
    return float(d["roc_auc"][0]), float(d["seq_roc_auc"][0])


def main():
    # Group by (policy, split), one bar cluster per group.
    groups = [("tipping_off", "train+val"),
              ("tipping_off", "held-out"),
              ("hallucinated_citations", "train+val"),
              ("hallucinated_citations", "held-out")]

    token_mlp, token_cov, seq_mlp, seq_cov = [], [], [], []
    for pol, split in groups:
        mlp = next(r for r in RUNS if r[0] == pol and r[1] == split and r[2] == "mlp")
        cov = next(r for r in RUNS if r[0] == pol and r[1] == split and r[2] == "covseq")
        t_m, s_m = load_auc(mlp[3])
        t_c, s_c = load_auc(cov[3])
        token_mlp.append(t_m); token_cov.append(t_c)
        seq_mlp.append(s_m);  seq_cov.append(s_c)

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True, sharey=True)

    x = np.arange(len(groups))
    width = 0.38

    def _bars(ax, mlp_vals, cov_vals, title):
        b1 = ax.bar(x - width / 2, mlp_vals, width, label="MLP head", color="#5b8def")
        b2 = ax.bar(x + width / 2, cov_vals, width, label="covseq head", color="#d64545")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="random")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n{s}" for p, s in groups], fontsize=9)
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel("ROC AUC")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.axvline(1.5, color="lightgray", linewidth=0.8, alpha=0.6)

    _bars(ax_t, token_mlp, token_cov, "Token-level AUC")
    _bars(ax_s, seq_mlp, seq_cov, "Sequence-level AUC (max prob per completion)")

    fig.suptitle(
        "Probe head comparison — Qwen2.5-0.5B-Instruct, layer 10\n"
        "100-row training pilot vs 40-row held-out",
        fontsize=12,
        fontweight="bold",
    )

    out = ROOT / "output/probe_performance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
