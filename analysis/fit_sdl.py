#!/usr/bin/env python
"""
Sparse Dictionary Learning probe: learn a dictionary on activation space,
encode activations as sparse codes, then fit a logistic classifier on those codes.

Sweeps over dictionary size (n_components) by default.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import normalize
from tqdm import tqdm

_ANALYSIS_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _ANALYSIS_DIR.parent / "train"
_RESULTS_DIR = _ANALYSIS_DIR / "results"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from load_activ import load_activations, prepare_records  # noqa: E402


def build_flat_features(records: List[Dict[str, Any]], desc: str = "Building features") -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        labels = rec["token_labels"]
        keep = labels != -100
        if not keep.any():
            continue
        xs.append(rec["hidden_states"][keep].astype(np.float32))
        ys.append(labels[keep].astype(np.float32))
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def eval_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = (scores > 0).astype(float)
    viol = labels == 1.0
    ok   = labels == 0.0
    tp = pred[viol].sum() if viol.any() else 0.0
    fp = pred[ok].sum()   if ok.any()   else 0.0
    fn = (1 - pred[viol]).sum() if viol.any() else 0.0
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    tnr  = (1 - pred[ok]).mean() if ok.any() else 0.0
    acc  = (pred == labels).mean()
    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tnr": float(tnr),
        "accuracy": float(acc),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def plot_curves(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))
    for name, (scores, labels) in models_data.items():
        prec, rec, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax_pr.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    pos_rate = float(next(iter(models_data.values()))[1].mean())
    ax_pr.axhline(pos_rate, color="k", linestyle="--", linewidth=0.8, label=f"No skill ({pos_rate:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall")
    ax_pr.legend()
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC")
    ax_roc.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Curves saved to {output_path}")


# ---------------------------------------------------------------------------
# SDL training
# ---------------------------------------------------------------------------

def fit_sdl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_components: int,
    alpha: float,
    transform_algorithm: str,
    fit_algorithm: str,
    max_dict_train: int,
    batch_size: int,
    max_iter: int,
    lr_epochs: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    1. Learn dictionary on (a subset of) X_train.
    2. Encode train and test sets as sparse codes.
    3. Fit logistic regression on sparse codes.
    Returns (metrics, decision_scores, y_test).
    """
    rng = np.random.default_rng(42)

    # Normalise activations so dictionary atoms have unit norm
    X_train_norm = normalize(X_train, norm="l2")
    X_test_norm  = normalize(X_test,  norm="l2")

    # Subsample for dictionary learning if needed
    if len(X_train_norm) > max_dict_train:
        idx = rng.choice(len(X_train_norm), max_dict_train, replace=False)
        X_dict = X_train_norm[idx]
        print(f"  Dictionary learning on {max_dict_train:,} / {len(X_train_norm):,} train tokens")
    else:
        X_dict = X_train_norm

    print(f"  Fitting dictionary  n_components={n_components}  alpha={alpha}  fit={fit_algorithm}  transform={transform_algorithm}")
    t0 = time.time()
    fit_bar = tqdm(total=max_iter, desc="  Dictionary fit", unit="iter", dynamic_ncols=True)

    def _fit_callback(_dict):
        fit_bar.update(1)

    sdl = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        transform_algorithm=transform_algorithm,
        transform_alpha=alpha,
        fit_algorithm=fit_algorithm,
        random_state=42,
        verbose=False,
        callback=_fit_callback,
    )
    sdl.fit(X_dict)
    fit_bar.close()
    print(f"  Dictionary fitted in {time.time() - t0:.1f}s")

    encode_chunk = 10_000
    t0 = time.time()
    Z_train = np.vstack([
        sdl.transform(X_train_norm[i : i + encode_chunk])
        for i in tqdm(range(0, len(X_train_norm), encode_chunk),
                      desc="  Encoding train", unit="chunk", dynamic_ncols=True)
    ])
    print(f"  Train encoded in {time.time() - t0:.1f}s  shape={Z_train.shape}")

    Z_test = np.vstack([
        sdl.transform(X_test_norm[i : i + encode_chunk])
        for i in tqdm(range(0, len(X_test_norm), encode_chunk),
                      desc="  Encoding test", unit="chunk", dynamic_ncols=True)
    ])

    classes = np.array([0, 1])
    y_train_int = y_train.astype(int)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_int)
    clf = SGDClassifier(
        loss="log_loss",
        class_weight={0: weights[0], 1: weights[1]},
        learning_rate="optimal",
        random_state=42,
    )
    for _ in tqdm(range(lr_epochs), desc="  Logistic regression", unit="epoch", dynamic_ncols=True):
        perm = rng.permutation(len(Z_train))
        clf.partial_fit(Z_train[perm], y_train_int[perm], classes=classes)

    scores = clf.decision_function(Z_test)
    metrics = eval_metrics(scores, y_test)
    return metrics, scores, y_test


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option("--activations-dir", type=click.Path(exists=True, path_type=Path),
              default="/pool/lmawby/data/probe_ann_activations")
@click.option("--annotations-jsonl", type=click.Path(exists=True, path_type=Path),
              default="data/annotated.jsonl")
@click.option("--model-name", default="google/gemma-4-31B-it")
@click.option("--num-to-load", type=int, default=None)
@click.option("--val-fraction", type=float, default=0.1)
@click.option("--n-components", default="64,128,256,512",
              help="Comma-separated dictionary sizes to sweep")
@click.option("--alpha", type=float, default=0.1,
              help="Sparsity regularisation strength")
@click.option("--transform-algorithm", default="omp",
              type=click.Choice(["omp", "lasso_lars", "lasso_cd", "threshold"]),
              help="Sparse coding algorithm for transform step")
@click.option("--fit-algorithm", default="lars",
              type=click.Choice(["lars", "cd"]),
              help="Algorithm for dictionary atom updates (lars avoids CD convergence warnings)")
@click.option("--max-dict-train", type=int, default=50_000,
              help="Max tokens used to learn the dictionary (subsample if larger)")
@click.option("--dict-batch-size", type=int, default=256,
              help="Mini-batch size for dictionary learning")
@click.option("--dict-max-iter", type=int, default=1000,
              help="Number of mini-batch iterations for dictionary learning")
@click.option("--lr-epochs", type=int, default=20,
              help="SGD epochs for logistic regression on sparse codes")
@click.option("--seed", type=int, default=42)
@click.option("--output-json", type=click.Path(path_type=Path), default=_RESULTS_DIR / "fit_sdl.json")
@click.option("--plot-path", type=click.Path(path_type=Path), default=_RESULTS_DIR / "fit_sdl_curves.png")
def main(
    activations_dir: Path,
    annotations_jsonl: Path,
    model_name: str,
    num_to_load: Optional[int],
    val_fraction: float,
    n_components: str,
    alpha: float,
    transform_algorithm: str,
    fit_algorithm: str,
    max_dict_train: int,
    dict_batch_size: int,
    dict_max_iter: int,
    lr_epochs: int,
    seed: int,
    output_json: Optional[Path],
    plot_path: Optional[Path],
):
    """Sparse dictionary learning probe: sweep over dictionary sizes."""
    np.random.seed(seed)
    n_components_list = [int(x) for x in n_components.split(",") if x.strip()]

    data = load_activations(activations_dir, annotations_jsonl, num_to_load=num_to_load)
    if not data:
        print("No data loaded — exiting.")
        return

    records = prepare_records(data, model_name, chat_template_kwargs={"enable_thinking": False})
    print(f"Prepared {len(records)} records")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    split = int(len(records) * (1 - val_fraction))
    train_records = [records[i] for i in indices[:split]]
    test_records  = [records[i] for i in indices[split:]]
    print(f"Split: {len(train_records)} train / {len(test_records)} test records")

    X_train, y_train = build_flat_features(train_records, desc="Building train features")
    X_test,  y_test  = build_flat_features(test_records,  desc="Building test features")
    print(f"Train tokens: {X_train.shape[0]:,}  test tokens: {X_test.shape[0]:,}")
    print(f"Violation rate  train={y_train.mean():.4f}  test={y_test.mean():.4f}")

    run_config = {
        "activations_dir": str(activations_dir),
        "annotations_jsonl": str(annotations_jsonl),
        "model_name": model_name,
        "val_fraction": val_fraction,
        "n_components_sweep": n_components_list,
        "alpha": alpha,
        "transform_algorithm": transform_algorithm,
        "max_dict_train": max_dict_train,
        "seed": seed,
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
    }
    results: Dict[str, Any] = {}
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    sweep_bar = tqdm(n_components_list, desc="SDL sweep", unit="model", dynamic_ncols=True)
    for nc in sweep_bar:
        sweep_bar.set_postfix(n_components=nc)
        tqdm.write(f"\n── SDL  n_components={nc} ──")
        metrics, scores, labels = fit_sdl(
            X_train, y_train, X_test, y_test,
            n_components=nc,
            alpha=alpha,
            transform_algorithm=transform_algorithm,
            fit_algorithm=fit_algorithm,
            max_dict_train=max_dict_train,
            batch_size=dict_batch_size,
            max_iter=dict_max_iter,
            lr_epochs=lr_epochs,
        )
        results[f"sdl_{nc}"] = metrics
        curves_data[f"SDL n={nc}"] = (scores, labels)
        tqdm.write(f"  f1={metrics['f1']:.4f}  auprc={metrics['auprc']:.4f}  auroc={metrics['auroc']:.4f}")
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps({"config": run_config, "results": results}, indent=2))
            tqdm.write(f"[checkpoint] Wrote {output_json}")

    # ── Results table ─────────────────────────────────────────────────────
    print("\n── Results ──")
    col_w = 12
    keys = list(results.keys())
    header = f"{'metric':<12}" + "".join(f"{k:>{col_w}}" for k in keys)
    print(header)
    print("-" * len(header))
    for metric in ["f1", "precision", "recall", "tnr", "accuracy", "auprc", "auroc"]:
        row = f"{metric:<12}" + "".join(f"{results[k][metric]:>{col_w}.4f}" for k in keys)
        print(row)

    if plot_path is not None:
        plot_curves(curves_data, plot_path)


if __name__ == "__main__":
    main()
