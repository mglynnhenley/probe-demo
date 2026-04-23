#!/usr/bin/env python3
"""12-probe summary figure: per-sequence score distributions (held-out split).

For each probe, plots the histogram of max-prob-per-completion separated by
label (clean vs violating), with token + sequence AUC in the title. Skips
probes whose npz isn't on disk, so it's safe to run mid-pipeline.

Output: output/probe_results_summary_12.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (policy_label, output_dir, heldout_npz_filename)
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
    return {
        "seq_scores": d["seq_scores"],
        "seq_labels": d["seq_labels"],
        "token_auc": float(d["roc_auc"][0]),
        "seq_auc": float(d["seq_roc_auc"][0]),
    }


def main():
    # 3 rows × 4 cols = 12 panels, one per probe.
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    axes = axes.ravel()

    bins = np.linspace(0, 1, 21)
    plotted = 0

    for ax, (policy, dir_name, heldout_fname) in zip(axes, PROBES):
        data = _load(ROOT / f"output/{dir_name}/{heldout_fname}")
        if data is None:
            ax.set_title(f"{policy}\n(no held-out npz yet)", fontsize=10, color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        violating = data["seq_scores"][data["seq_labels"] == 1]
        clean = data["seq_scores"][data["seq_labels"] == 0]
        ax.hist(clean, bins=bins, alpha=0.55, color="#5b8def",
                label=f"clean (n={len(clean)})", edgecolor="white")
        ax.hist(violating, bins=bins, alpha=0.55, color="#d64545",
                label=f"violating (n={len(violating)})", edgecolor="white")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("max prob per completion", fontsize=8)
        ax.set_ylabel("# completions", fontsize=8)
        ax.set_title(f"{policy}\nseq AUC={data['seq_auc']:.3f}  token AUC={data['token_auc']:.3f}",
                     fontsize=9)
        ax.legend(loc="upper center", fontsize=7)
        ax.grid(True, alpha=0.25)
        plotted += 1

    fig.suptitle(
        f"Probe held-out results — Qwen2.5-0.5B-Instruct layer 10, MLP head "
        f"({plotted}/{len(PROBES)} probes evaluated)",
        fontsize=13,
        fontweight="bold",
    )

    out = ROOT / "output/probe_results_summary_12.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out} ({plotted}/{len(PROBES)} probes plotted)")


if __name__ == "__main__":
    main()
