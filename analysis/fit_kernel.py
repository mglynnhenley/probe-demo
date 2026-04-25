#!/usr/bin/env python
"""
Kernel-method probes on activation data.

Each kernel is defined as a (name, Nystroem-transformer-or-None) pair.
LinearSVC is used directly for the linear kernel; for all others a
Nystroem approximation is prepended so the pipeline scales to large token counts.

─────────────────────────────────────────────────────────────────────────────
  ADD YOUR OWN KERNELS in the KERNELS list below.
─────────────────────────────────────────────────────────────────────────────
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
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

_ANALYSIS_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _ANALYSIS_DIR.parent / "train"
_RESULTS_DIR = _ANALYSIS_DIR / "results"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from load_activ import load_activations, prepare_records  # noqa: E402


def _make_dot_exp_sin_squared(
    dot_weight: float,
    periodicity: float,
    length_scale: float,
):
    """
    Returns a kernel k(x,y) = w*(x·y/d) + (1-w)*exp(-2 sin²(π‖x-y‖/p) / l²)

    The linear component is divided by d so that both terms are O(1) after
    StandardScaler (raw dot products are O(d) in scale).
    """
    def kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        linear_k = (X @ Y.T) / d
        sq = (
            np.sum(X ** 2, axis=1, keepdims=True)
            - 2.0 * (X @ Y.T)
            + np.sum(Y ** 2, axis=1)
        )
        dists = np.sqrt(np.maximum(sq, 0.0))
        periodic_k = np.exp(
            -2.0 * np.sin(np.pi * dists / periodicity) ** 2 / length_scale ** 2
        )
        return dot_weight * linear_k + (1.0 - dot_weight) * periodic_k
    return kernel


# =============================================================================
#  KERNEL DEFINITIONS
#  Each entry is a (label, transformer_or_None) tuple.
#
#  • transformer = None  →  LinearSVC directly (exact linear kernel, fast).
#  • transformer = Nystroem(...)  →  approximate kernel via random features.
#
#  Nystroem(kernel=...) accepts any string recognised by sklearn's pairwise
#  kernels ("rbf", "poly", "sigmoid", "laplacian", "chi2", "cosine") or a
#  callable  kernel_fn(X, Y) → ndarray.
#
#  ── ADD YOUR OWN KERNELS HERE ────────────────────────────────────────────
#
#  Example custom callable kernel:
#
#      def my_kernel(X, Y):
#          # X: (n, d)  Y: (m, d) → (n, m)
#          return np.exp(-0.5 * cdist(X, Y, "cosine"))
#
#      ("my_kernel", Nystroem(kernel=my_kernel, n_components=500)),
#
# =============================================================================
def _build_kernels(
    n_components: int,
    gamma: Optional[float],
    exp_sin_periodicity: float,
    exp_sin_length_scale: float,
    exp_sin_dot_weight: float,
) -> List[Tuple[str, Any]]:
    """
    Returns the list of (name, transformer_or_None) pairs to sweep over.

    n_components:        number of Nystroem random features.
    gamma:               RBF / poly / laplacian scale; None → sklearn default (1/d).
                         Note: for laplacian, gamma=None is overridden to 1/sqrt(d) at
                         fit time — the default 1/d causes kernel concentration in high dims.
    exp_sin_periodicity: period p of the exp-sin-squared component (Euclidean distance
                         units post-StandardScaler; sqrt(2*hidden_dim) is a natural scale).
    exp_sin_length_scale: length-scale l of the exp-sin-squared component.
    exp_sin_dot_weight:  weight w of the linear component (1-w to the periodic component).
    """
    kernels = [
        # ── built-in kernels ──────────────────────────────────────────────
        ("linear",          None),
        ("rbf",             Nystroem(kernel="rbf",       gamma=gamma, n_components=n_components, random_state=42)),
        ("poly-2",          Nystroem(kernel="poly",      degree=2,    n_components=n_components, random_state=42)),
        ("poly-3",          Nystroem(kernel="poly",      degree=3,    n_components=n_components, random_state=42)),
        # gamma=None is fixed to 1/sqrt(d) at fit time (see fit_kernel).
        ("laplacian",       Nystroem(kernel="laplacian", gamma=gamma, n_components=n_components, random_state=42)),
        # Cosine similarity concentrates near 0 in high dimensions (std ≈ 1/sqrt(d)),
        # making the Nystroem landmark matrix near-degenerate. Retained for diagnostics.
        ("cosine",          Nystroem(kernel="cosine",                 n_components=n_components, random_state=42)),

        # ── custom kernels ────────────────────────────────────────────────
        ("dot+exp-sin²",    Nystroem(
            kernel=_make_dot_exp_sin_squared(exp_sin_dot_weight, exp_sin_periodicity, exp_sin_length_scale),
            n_components=n_components, random_state=42,
        )),
        # ── ADD YOUR OWN KERNELS BELOW ────────────────────────────────────
        # ("my_kernel",    Nystroem(kernel=my_kernel, n_components=n_components, random_state=42)),
        # ── END CUSTOM KERNELS ────────────────────────────────────────────
    ]
    return kernels
# =============================================================================
#  END KERNEL DEFINITIONS
# =============================================================================


def build_flat_features(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for rec in records:
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
    viol, ok = labels == 1.0, labels == 0.0
    tp = pred[viol].sum() if viol.any() else 0.0
    fp = pred[ok].sum()   if ok.any()   else 0.0
    fn = (1 - pred[viol]).sum() if viol.any() else 0.0
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    tnr  = (1 - pred[ok]).mean() if ok.any() else 0.0
    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tnr": float(tnr),
        "accuracy": float((pred == labels).mean()),
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
        ax_pr.plot(rec, prec, label=f"{name} (AP={average_precision_score(labels, scores):.3f})")
        fpr, tpr, _ = roc_curve(labels, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(labels, scores):.3f})")
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
# Fitting
# ---------------------------------------------------------------------------

def fit_kernel(
    name: str,
    transformer,           # None for linear, Nystroem for others
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    svm_C: float,
    chunk_size: int,
    n_epochs: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit one kernel SVM and return (metrics, decision_scores_on_test)."""
    rng = np.random.default_rng(42)
    n_train = len(X_train)

    t = time.time()
    tqdm.write("  [1/3] Fitting scaler ...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    tqdm.write(f"        done in {time.time() - t:.1f}s")

    if transformer is None:
        # ── Linear: exact LinearSVC on all training tokens ─────────────────
        tqdm.write("  [2/3] No transformer (linear kernel)")
        t = time.time()
        tqdm.write(f"  [3/3] Fitting LinearSVC on {n_train:,} tokens ...")
        svm = LinearSVC(C=svm_C, class_weight="balanced", max_iter=2000, random_state=42)
        svm.fit(scaler.transform(X_train), y_train)
        tqdm.write(f"        done in {time.time() - t:.1f}s  (n_iter={svm.n_iter_})")
        scores = svm.decision_function(scaler.transform(X_test))
    else:
        # ── Non-linear: Nystroem (fit on subset) + SGDClassifier (chunked) ─
        #
        # Laplacian with gamma=None uses 1/d, which in high dimensions makes
        # kernel values concentrate in a tiny range → near-rank-1 landmark
        # matrix → degenerate Nystroem features. Override to 1/sqrt(d).
        if getattr(transformer, "kernel", None) == "laplacian" and getattr(transformer, "gamma", None) is None:
            transformer.gamma = float(1.0 / np.sqrt(X_train.shape[1]))
            tqdm.write(f"  (laplacian auto-gamma: {transformer.gamma:.5f})")

        n_fit = min(n_train, transformer.n_components * 20)
        fit_idx = rng.choice(n_train, n_fit, replace=False)
        t = time.time()
        tqdm.write(
            f"  [2/3] Fitting {name} transformer on {n_fit:,} samples "
            f"({transformer.n_components} components) ..."
        )
        transformer.fit(scaler.transform(X_train[fit_idx]))
        tqdm.write(f"        done in {time.time() - t:.1f}s")

        # Sanity-check output dimensionality before committing to full training.
        probe_out = transformer.transform(scaler.transform(X_train[:1]))
        n_out = probe_out.shape[1]
        if n_out == 0:
            raise ValueError(
                f"Kernel '{name}' produced 0-dimensional Nystroem features. "
                "The kernel matrix is degenerate (near-constant) in this "
                "high-dimensional space."
            )
        tqdm.write(f"        Nystroem output dim: {n_out}")

        clf = SGDClassifier(
            loss="hinge",
            alpha=1.0 / (svm_C * n_train),
            class_weight="balanced",
            random_state=42,
        )

        t = time.time()
        tqdm.write(
            f"  [3/3] SGDClassifier on {n_train:,} tokens "
            f"({n_epochs} epoch(s), chunk={chunk_size:,}) ..."
        )
        for epoch in range(n_epochs):
            perm = rng.permutation(n_train)
            for start in range(0, n_train, chunk_size):
                idx = perm[start:start + chunk_size]
                Xc = transformer.transform(scaler.transform(X_train[idx]))
                clf.partial_fit(Xc, y_train[idx], classes=np.array([0.0, 1.0]))
        tqdm.write(f"        done in {time.time() - t:.1f}s")

        chunks = []
        for start in range(0, len(X_test), chunk_size):
            Xc = transformer.transform(scaler.transform(X_test[start:start + chunk_size]))
            chunks.append(clf.decision_function(Xc))
        scores = np.concatenate(chunks)

    return eval_metrics(scores, y_test), scores


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
@click.option("--nystroem-components", type=int, default=500,
              help="Number of Nystroem random features for approximate kernels")
@click.option("--gamma", type=float, default=None,
              help="RBF/poly gamma; omit to use sklearn's default (1/n_features). "
                   "Laplacian overrides this to 1/sqrt(n_features) automatically.")
@click.option("--svm-c", type=float, default=1.0,
              help="SVM regularisation (smaller = stronger)")
@click.option("--chunk-size", type=int, default=50_000,
              help="Tokens per SGD chunk for non-linear kernels (controls peak memory)")
@click.option("--n-epochs", type=int, default=3,
              help="SGD epochs per non-linear kernel")
@click.option("--exp-sin-periodicity", type=float, default=100.0,
              help="Period p for the dot+exp-sin² kernel (Euclidean distance units "
                   "post-StandardScaler; sqrt(2*hidden_dim) is a natural scale)")
@click.option("--exp-sin-length-scale", type=float, default=1.0,
              help="Length scale l for the exp-sin-squared component")
@click.option("--exp-sin-dot-weight", type=float, default=0.5,
              help="Weight w of the linear component; (1-w) goes to the periodic component")
@click.option("--seed", type=int, default=42)
@click.option("--output-json", type=click.Path(path_type=Path), default=_RESULTS_DIR / "fit_kernel.json")
@click.option("--plot-path", type=click.Path(path_type=Path), default=_RESULTS_DIR / "fit_kernel_curves.png")
def main(
    activations_dir: Path,
    annotations_jsonl: Path,
    model_name: str,
    num_to_load: Optional[int],
    val_fraction: float,
    nystroem_components: int,
    gamma: Optional[float],
    svm_c: float,
    chunk_size: int,
    n_epochs: int,
    exp_sin_periodicity: float,
    exp_sin_length_scale: float,
    exp_sin_dot_weight: float,
    seed: int,
    output_json: Optional[Path],
    plot_path: Optional[Path],
):
    """Kernel-method sweep over activation data."""
    np.random.seed(seed)

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

    print("\nBuilding flat feature arrays ...")
    X_train, y_train = build_flat_features(train_records)
    X_test,  y_test  = build_flat_features(test_records)
    print(f"Train tokens: {X_train.shape[0]:,}  test tokens: {X_test.shape[0]:,}")
    print(f"Violation rate  train={y_train.mean():.4f}  test={y_test.mean():.4f}")

    kernels = _build_kernels(
        nystroem_components, gamma,
        exp_sin_periodicity, exp_sin_length_scale, exp_sin_dot_weight,
    )
    run_config = {
        "activations_dir": str(activations_dir),
        "annotations_jsonl": str(annotations_jsonl),
        "model_name": model_name,
        "val_fraction": val_fraction,
        "nystroem_components": nystroem_components,
        "gamma": gamma,
        "svm_c": svm_c,
        "chunk_size": chunk_size,
        "n_epochs": n_epochs,
        "exp_sin_periodicity": exp_sin_periodicity,
        "exp_sin_length_scale": exp_sin_length_scale,
        "exp_sin_dot_weight": exp_sin_dot_weight,
        "seed": seed,
        "kernels": [name for name, _ in kernels],
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
    }
    all_results: Dict[str, Dict] = {}
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    kernel_bar = tqdm(kernels, desc="Kernel sweep", unit="kernel", dynamic_ncols=True)
    for name, transformer in kernel_bar:
        kernel_bar.set_postfix(kernel=name)
        tqdm.write(f"\n── {name} ──")
        try:
            metrics, scores = fit_kernel(
                name, transformer, X_train, y_train, X_test, y_test,
                svm_C=svm_c,
                chunk_size=chunk_size,
                n_epochs=n_epochs,
            )
        except Exception as exc:
            tqdm.write(f"  ERROR: {type(exc).__name__}: {exc}")
            tqdm.write("  Skipping this kernel.")
            continue
        all_results[name] = metrics
        curves_data[name] = (scores, y_test)
        tqdm.write(f"  f1={metrics['f1']:.4f}  auprc={metrics['auprc']:.4f}  auroc={metrics['auroc']:.4f}")
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps({"config": run_config, "results": all_results}, indent=2))
            tqdm.write(f"[checkpoint] Wrote {output_json}")

    # ── Results table ─────────────────────────────────────────────────────
    print("\n── Results ──")
    col_w = 12
    keys = list(all_results.keys())
    header = f"{'metric':<12}" + "".join(f"{k:>{col_w}}" for k in keys)
    print(header)
    print("-" * len(header))
    for metric in ["f1", "precision", "recall", "tnr", "accuracy", "auprc", "auroc"]:
        row = f"{metric:<12}" + "".join(f"{all_results[k][metric]:>{col_w}.4f}" for k in keys)
        print(row)

    if plot_path is not None:
        plot_curves(curves_data, plot_path)


if __name__ == "__main__":
    main()
