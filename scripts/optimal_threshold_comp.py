#!/usr/bin/env python
"""computing the optimal threshold for a probe based on class imbalance and relative costs of false positives and false negatives"""

import numpy as np
import click
from pathlib import Path


# def compute_weighted_j_statistic(tpr: np.ndarray, fpr: np.ndarray, fp_cost: float, fn_cost: float) -> :


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.argument("labels_path", type=click.Path(exists=True, path_type=Path))
# @click.argument("scores_path", type=click.Path(exists=True, path_type=Path))
@click.option("--fp_cost", type=float, default=1.0, help="cost of a false positive")
@click.option("--fn_cost", type=float, default=1.0, help="cost of a false negative")
def main(labels_path: Path, fp_cost: float, fn_cost: float):
    """
    main entry function if this script is being executed directly
    
    note that this function assumes that labels and scores for some probe have already been computed
    """
    # relevant path: output/gemma4_31b_probe_covseq_2/probe_scores_all.npz

    # load labels
    data = np.load(labels_path)
    data_fields = data.files
    # files: ['split', 'logits', 'probs', 'labels', 'example_indices', 'token_indices', 'window_lengths', 'roc_fpr', 'roc_tpr', 'roc_thresholds', 'roc_auc']

    for field in data_fields:
        print(f"Field: {field}")
        print(type(data[field]))
        try:
            print(f"Data: {data[field][:10]}")
        except Exception as _:
            print(f"Data: {data[field]}")

    # computing the weighted j statistic
    optimal_idx = np.argmax((data["roc_tpr"] * fn_cost) - (data["roc_fpr"] * fp_cost))
    optimal_threshold = data["roc_thresholds"][optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")

    # distribution over for token_indices and window_lengths
    # unique_token_indices, unique_token_counts = np.unique(data["token_indices"], return_counts=True)
    # unique_window_lengths, unique_window_length_counts = np.unique(data["window_lengths"], return_counts=True)
    # print(f"Unique token indices: {unique_token_indices}")
    # print(f"Unique token counts: {unique_token_counts}")
    # print(f"Unique window lengths: {unique_window_lengths}")
    # print(f"Unique window length counts: {unique_window_length_counts}")


if __name__ == "__main__":
    main()
