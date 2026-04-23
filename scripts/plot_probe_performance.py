#!/usr/bin/env python3
"""Bar-chart summary: token + seq AUC per probe, train+val vs held-out.

Produces a 2-panel figure (token AUC left, sequence AUC right) with one
bar-cluster per probe. Skips probes whose npz files aren't on disk yet, so
it is safe to run mid-pipeline to see partial results.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# One row per probe. `dir_name` is the output directory under output/; the
# pair of files (train+val / held-out) follows evaluate_probe_roc.py naming.
PROBES = [
    ("tipping_off",             "qwen_mac_tipping_off",             "probe_scores_tipping_off_heldout_annotated.npz"),
    ("hallucinated_citations",  "qwen_mac_hallucinated",            "probe_scores_hallucinated_citations_heldout_annotated.npz"),
    ("religious_truth_claim",   "qwen_mac_religious_truth_claim",   "probe_scores_religious_truth_claim_heldout_annotated.npz"),
    ("partisan_endorsement",    "qwen_mac_partisan_endorsement",    "probe_scores_partisan_endorsement_heldout_annotated.npz"),
    ("definitive_diagnosis",    "qwen_mac_definitive_diagnosis",    "probe_scores_definitive_diagnosis_heldout_annotated.npz"),
    ("personal_investment_rec", "qwen_mac_personal_investment_rec", "probe_scores_personal_investment_rec_heldout_annotated.npz"),
    ("moralising",              "qwen_mac_moralising",              "probe_scores_moralising_heldout_annotated.npz"),
    ("sycophancy",              "qwen_mac_sycophancy",              "probe_scores_sycophancy_heldout_annotated.npz"),
    ("unsolicited_disclaimer",  "qwen_mac_unsolicited_disclaimer",  "probe_scores_unsolicited_disclaimer_heldout_annotated.npz"),
    ("fabricated_quote",        "qwen_mac_fabricated_quote",        "probe_scores_fabricated_quote_heldout_annotated.npz"),
    ("structuring",             "qwen_mac_structuring",             "probe_scores_structuring_heldout_annotated.npz"),
    ("guaranteed_returns",      "qwen_mac_guaranteed_returns",      "probe_scores_guaranteed_returns_heldout_annotated.npz"),
]


def _load(path: Path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return float(d["roc_auc"][0]), float(d["seq_roc_auc"][0])


def main():
    labels, t_tv, s_tv, t_ho, s_ho = [], [], [], [], []
    for policy, dir_name, heldout_fname in PROBES:
        tv = _load(ROOT / f"output/{dir_name}/probe_scores_all.npz")
        ho = _load(ROOT / f"output/{dir_name}/{heldout_fname}")
        if tv is None and ho is None:
            continue
        labels.append(policy)
        t_tv.append(tv[0] if tv else np.nan)
        s_tv.append(tv[1] if tv else np.nan)
        t_ho.append(ho[0] if ho else np.nan)
        s_ho.append(ho[1] if ho else np.nan)

    if not labels:
        print("No probe results found on disk. Run evaluate_probe_roc.py first.")
        return

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(max(12, 1.2 * len(labels) + 4), 6),
                                     constrained_layout=True, sharey=True)

    x = np.arange(len(labels))
    width = 0.38

    def _bars(ax, tv_vals, ho_vals, title):
        b1 = ax.bar(x - width / 2, tv_vals, width, label="train+val", color="#5b8def")
        b2 = ax.bar(x + width / 2, ho_vals, width, label="held-out", color="#d64545")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="random")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
        ax.set_ylim(0.4, 1.02)
        ax.set_ylabel("ROC AUC")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                if np.isfinite(h):
                    ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.2f}",
                            ha="center", va="bottom", fontsize=7)

    _bars(ax_t, t_tv, t_ho, "Token-level AUC")
    _bars(ax_s, s_tv, s_ho, "Sequence-level AUC (max prob per completion)")

    fig.suptitle(
        f"Probe performance — Qwen2.5-0.5B-Instruct layer 10, MLP head, {len(labels)} policies",
        fontsize=12,
        fontweight="bold",
    )

    out = ROOT / "output/probe_performance_12.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out} ({len(labels)} probes)")


if __name__ == "__main__":
    main()
