#!/usr/bin/env python
"""
Fit MLP and CovSeq probes to saved activation data and compare results.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import json
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm

_ANALYSIS_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _ANALYSIS_DIR.parent / "train"
_RESULTS_DIR = _ANALYSIS_DIR / "results"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from load_activ import load_activations, prepare_records  # noqa: E402
from models import CovSeqModel, MLP  # noqa: E402



# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_mlp_features(records: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten all labelled completion tokens across records into (X, y)."""
    xs, ys = [], []
    for rec in records:
        labels = torch.from_numpy(rec["token_labels"])
        keep = labels != -100
        if not keep.any():
            continue
        hs = torch.from_numpy(rec["hidden_states"]).float()
        xs.append(hs[keep])
        ys.append(labels[keep].float())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


class CovSeqDataset(Dataset):
    """On-demand window dataset — extracts each window lazily from record hidden_states.

    Builds a flat index of (rec_idx, token_pos, seq_len) at init (fast, no tensors),
    then materialises one window at a time in __getitem__.
    """

    def __init__(self, records: List[Dict[str, Any]], window_size: int) -> None:
        self.records = records
        self.window_size = window_size
        self._index: List[Tuple[int, int, int]] = []  # (rec_idx, pos, seq_len)
        for rec_idx, rec in enumerate(records):
            labels = rec["token_labels"]
            for pos, lbl in enumerate(labels):
                if lbl == -100:
                    continue
                self._index.append((rec_idx, pos, min(pos + 1, window_size)))
        print(f"[covseq] CovSeqDataset: window_size={window_size}, {len(records)} records, {len(self)} windows")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec_idx, pos, _ = self._index[i]
        rec = self.records[rec_idx]
        hs = torch.from_numpy(rec["hidden_states"]).float()
        start = max(0, pos - self.window_size + 1)
        window = hs[start : pos + 1]  # (seq_len, hidden_size)
        return window, torch.tensor(float(rec["token_labels"][pos]))

    def all_labels(self) -> torch.Tensor:
        return torch.tensor([
            float(self.records[rec_idx]["token_labels"][pos])
            for rec_idx, pos, _ in self._index
        ])


class _SeqLenBatchSampler(Sampler):
    """Yields batches of indices that all share the same effective seq_len.

    Within each seq_len bucket, indices are shuffled before batching.
    All resulting batches are then shuffled together (matching the original
    build_covseq_windows → random.shuffle(batches) behaviour).
    """

    def __init__(self, dataset: CovSeqDataset, batch_size: int, shuffle: bool = True) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._buckets: Dict[int, List[int]] = {}
        for i, (_, _, sl) in enumerate(dataset._index):
            self._buckets.setdefault(sl, []).append(i)
        self._len = sum(
            (len(idxs) + batch_size - 1) // batch_size
            for idxs in self._buckets.values()
        )

    def __iter__(self) -> Iterator[List[int]]:
        all_batches: List[List[int]] = []
        for idxs in self._buckets.values():
            idxs = list(idxs)
            if self._shuffle:
                random.shuffle(idxs)
            for start in range(0, len(idxs), self._batch_size):
                all_batches.append(idxs[start : start + self._batch_size])
        if self._shuffle:
            random.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        return self._len


def _covseq_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.stack([w for w, _ in batch]), torch.stack([elem for _, elem in batch])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    n_pos = y.sum().item()
    n_neg = (y == 0).sum().item()
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def _eval_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = (logits > 0).float()
    viol = labels == 1.0
    ok   = labels == 0.0
    tp = pred[viol].sum() if viol.any() else torch.tensor(0.0)
    fp = pred[ok].sum()   if ok.any()   else torch.tensor(0.0)
    fn = (1 - pred[viol]).sum() if viol.any() else torch.tensor(0.0)
    prec = (tp / (tp + fp + 1e-8)).item()
    rec  = (tp / (tp + fn + 1e-8)).item()
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    tnr  = ((1 - pred[ok]).mean()).item() if ok.any() else 0.0
    acc  = pred.eq(labels).float().mean().item()
    return {"f1": f1, "precision": prec, "recall": rec, "tnr": tnr, "accuracy": acc}


@torch.no_grad()
def _full_epoch_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Threshold metrics + eval_loss + AUPRC + AUROC from pre-collected logits."""
    metrics = _eval_metrics(logits, labels)
    metrics["eval_loss"] = criterion(logits, labels).item()
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    metrics["auprc"] = float(average_precision_score(labels_np, probs))
    metrics["auroc"] = float(roc_auc_score(labels_np, probs))
    return metrics


def train_mlp(
    model: MLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict[str, List]:
    """Train MLP and return per-batch loss history and per-epoch val metrics."""
    pw = _pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_train.shape[0]
    n_batches = max(1, (n + batch_size - 1) // batch_size)

    train_loss: List[Dict] = []
    val_per_epoch: List[Dict] = []
    global_step = 0

    epoch_bar = tqdm(range(epochs), desc="MLP training", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        batch_bar = tqdm(range(0, n, batch_size), desc=f"  epoch {epoch + 1}", leave=False, dynamic_ncols=True)
        for batch_idx, start in enumerate(batch_bar):
            idx = perm[start : start + batch_size]
            logits = model(X_train[idx]).squeeze(-1)
            loss = criterion(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = loss.item()
            epoch_loss += loss_val
            train_loss.append({
                "step": global_step,
                "epoch_frac": epoch + batch_idx / n_batches,
                "loss": loss_val,
            })
            global_step += 1
            batch_bar.set_postfix(loss=f"{loss_val:.4f}")

        avg = epoch_loss / n_batches
        epoch_bar.set_postfix(loss=f"{avg:.4f}")

        model.eval()
        val_logits = torch.cat([
            model(X_val[i : i + batch_size]).squeeze(-1)
            for i in range(0, X_val.shape[0], batch_size)
        ])
        ep_metrics = _full_epoch_metrics(val_logits, y_val, criterion)
        ep_metrics["epoch"] = epoch + 1
        val_per_epoch.append(ep_metrics)
        tqdm.write(f"  [MLP] epoch {epoch + 1}  eval_loss={ep_metrics['eval_loss']:.4f}  f1={ep_metrics['f1']:.4f}  auprc={ep_metrics['auprc']:.4f}")

    return {"train_loss": train_loss, "val_per_epoch": val_per_epoch}


def train_covseq(
    model: CovSeqModel,
    train_dataset: CovSeqDataset,
    val_dataset: CovSeqDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict[str, List]:
    """Train CovSeq and return per-batch loss history and per-epoch val metrics."""
    pw = _pos_weight(train_dataset.all_labels())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss: List[Dict] = []
    val_per_epoch: List[Dict] = []
    global_step = 0

    epoch_bar = tqdm(range(epochs), desc="CovSeq training", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        loader = DataLoader(
            train_dataset,
            batch_sampler=_SeqLenBatchSampler(train_dataset, batch_size, shuffle=True),
            collate_fn=_covseq_collate,
        )
        n_batches = len(loader.batch_sampler)
        epoch_loss = 0.0
        batch_bar = tqdm(enumerate(loader), total=n_batches, desc=f"  epoch {epoch + 1}", leave=False, dynamic_ncols=True)
        for batch_idx, (X_b, y_b) in batch_bar:
            logits = model(X_b).squeeze(-1)
            loss = criterion(logits, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = loss.item()
            epoch_loss += loss_val
            train_loss.append({
                "step": global_step,
                "epoch_frac": epoch + batch_idx / n_batches,
                "loss": loss_val,
            })
            global_step += 1
            batch_bar.set_postfix(loss=f"{loss_val:.4f}")

        avg = epoch_loss / n_batches
        epoch_bar.set_postfix(loss=f"{avg:.4f}")

        model.eval()
        val_logits, val_labels = _collect_covseq_logits(model, val_dataset, batch_size)
        ep_metrics = _full_epoch_metrics(val_logits, val_labels, criterion)
        ep_metrics["epoch"] = epoch + 1
        val_per_epoch.append(ep_metrics)
        tqdm.write(f"  [CovSeq] epoch {epoch + 1}  eval_loss={ep_metrics['eval_loss']:.4f}  f1={ep_metrics['f1']:.4f}  auprc={ep_metrics['auprc']:.4f}")

    return {"train_loss": train_loss, "val_per_epoch": val_per_epoch}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_mlp_logits(
    model: MLP, X: torch.Tensor, y: torch.Tensor, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits = torch.cat([model(X[i : i + batch_size]).squeeze(-1) for i in range(0, X.shape[0], batch_size)])
    return logits, y


@torch.no_grad()
def _collect_covseq_logits(
    model: CovSeqModel,
    dataset: CovSeqDataset,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_sampler=_SeqLenBatchSampler(dataset, batch_size, shuffle=False),
        collate_fn=_covseq_collate,
    )
    all_logits, all_labels = [], []
    for X_b, y_b in loader:
        all_logits.append(model(X_b).squeeze(-1))
        all_labels.append(y_b)
    return torch.cat(all_logits), torch.cat(all_labels)


def _eval_with_curves(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Return (metrics_dict, probs, labels_np) — probs and labels for curve plotting."""
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    metrics = _eval_metrics(logits, labels)
    metrics["auprc"] = float(average_precision_score(labels_np, probs))
    metrics["auroc"] = float(roc_auc_score(labels_np, probs))
    return metrics, probs, labels_np


def plot_curves(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot PR and ROC curves for each model and save to output_path."""
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))

    for name, (probs, labels) in models_data.items():
        prec, rec, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        ax_pr.plot(rec, prec, label=f"{name} (AP={ap:.3f})")

        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # No-skill baseline on PR plot (positive class rate)
    pos_rate = float(next(iter(models_data.values()))[1].mean())
    ax_pr.axhline(pos_rate, color="k", linestyle="--", linewidth=0.8, label=f"No skill ({pos_rate:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall")
    ax_pr.legend()

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC")
    ax_roc.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Curves saved to {output_path}")


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------

def _write_results(
    output_json: Optional[Path],
    config: Dict[str, Any],
    mlp_results: Dict[str, Dict],
    covseq_results: Dict[int, Dict],
) -> None:
    if output_json is None:
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({
        "config": config,
        "mlp": mlp_results,
        "covseq": {f"window_{ws}": covseq_results[ws] for ws in covseq_results},
    }, indent=2))
    print(f"[checkpoint] Wrote {output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option("--activations-dir", type=click.Path(exists=True, path_type=Path),
              default="/pool/lmawby/data/probe_ann_activations",
              help="Directory containing .npz activation files")
@click.option("--annotations-jsonl", type=click.Path(exists=True, path_type=Path),
              default="data/annotated.jsonl",
              help="Annotated JSONL to join with activations")
@click.option("--model-name", default="google/gemma-4-31B-it",
              help="Tokenizer for char→token label alignment")
@click.option("--num-to-load", type=int, default=None,
              help="Cap number of .npz files loaded (default: all)")
@click.option("--val-fraction", type=float, default=0.1,
              help="Fraction of records held out for evaluation")
@click.option("--epochs", type=int, default=5)
@click.option("--batch-size", type=int, default=512)
@click.option("--lr", type=float, default=1e-3)
@click.option("--window-sizes", default="4,8,16,32",
              help="Comma-separated CovSeq lookback window sizes to sweep")
@click.option("--compressed-size", type=int, default=64,
              help="CovSeq compression rank")
@click.option("--covseq-hidden-sizes", default="128",
              help="Hidden layer widths for each CovSeq model (e.g. '128' or '128,64')")
@click.option("--mlp-architectures", default="|128|128,64",
              help="Pipe-separated MLP architectures to sweep; empty = linear (e.g. '|128|128,64')")
@click.option("--seed", type=int, default=42)
@click.option("--output-json", type=click.Path(path_type=Path), default=_RESULTS_DIR / "fit_model.json",
              help="Write results + run config to this JSON file")
@click.option("--plot-path", type=click.Path(path_type=Path), default=_RESULTS_DIR / "fit_model_curves.png",
              help="Save PR and ROC curve plot to this path (e.g. output/curves.png)")
@click.option("--no-mlp", is_flag=True, default=False,
              help="Skip the MLP architecture sweep entirely")
@click.option("--no-covseq", is_flag=True, default=False,
              help="Skip the CovSeq window-size sweep entirely")
def main(
    activations_dir: Path,
    annotations_jsonl: Path,
    model_name: str,
    num_to_load: Optional[int],
    val_fraction: float,
    epochs: int,
    batch_size: int,
    lr: float,
    compressed_size: int,
    covseq_hidden_sizes: str,
    mlp_architectures: str,
    seed: int,
    output_json: Optional[Path],
    plot_path: Optional[Path],
    window_sizes: str,
    no_mlp: bool,
    no_covseq: bool,
):
    """Fit MLP and CovSeq probes to saved activations and compare evaluation metrics."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def _parse_arch(spec: str) -> List[int]:
        return [int(x) for x in spec.split(",") if x.strip()]

    def _arch_label(hidden: List[int]) -> str:
        return "Linear" if not hidden else f"MLP {hidden}"

    mlp_arch_list = [_parse_arch(s) for s in mlp_architectures.split("|")]
    covseq_hidden_list = _parse_arch(covseq_hidden_sizes)
    window_sizes_list = [int(x) for x in window_sizes.split(",") if x.strip()]

    # ── Load and align ──────────────────────────────────────────────────────
    data = load_activations(activations_dir, annotations_jsonl, num_to_load=num_to_load)
    if not data:
        print("No data loaded — exiting.")
        return

    records = prepare_records(data, model_name, chat_template_kwargs={"enable_thinking": False})
    print(f"Prepared {len(records)} records")

    # ── Train / test split at record level ─────────────────────────────────
    random.shuffle(records)
    split = int(len(records) * (1 - val_fraction))
    train_records, test_records = records[:split], records[split:]
    print(f"Split: {len(train_records)} train / {len(test_records)} test records")

    hidden_size = records[0]["hidden_states"].shape[1]
    print(f"Hidden size: {hidden_size}")

    run_config: Dict[str, Any] = {
        "activations_dir": str(activations_dir),
        "annotations_jsonl": str(annotations_jsonl),
        "model_name": model_name,
        "val_fraction": val_fraction,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "mlp_architectures": [_arch_label(a) for a in mlp_arch_list],
        "window_sizes": window_sizes_list,
        "compressed_size": compressed_size,
        "covseq_hidden_sizes": covseq_hidden_list,
        "seed": seed,
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "hidden_size": hidden_size,
    }

    mlp_results: Dict[str, Dict] = {}
    covseq_results: Dict[int, Dict] = {}
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ── MLP architecture sweep ───────────────────────────────────────────────
    if not no_mlp:
        print("\n── MLP sweep ──")
        X_train, y_train = build_mlp_features(train_records)
        X_test,  y_test  = build_mlp_features(test_records)
        print(f"Train tokens: {X_train.shape[0]}  test tokens: {X_test.shape[0]}")
        print(f"Violation rate  train={y_train.mean():.4f}  test={y_test.mean():.4f}")

        for arch in mlp_arch_list:
            label = _arch_label(arch)
            print(f"\n── {label} ──")
            mlp = MLP(input_size=hidden_size, hidden_sizes=arch, output_size=1)
            mlp_history = train_mlp(mlp, X_train, y_train, X_test, y_test,
                                     epochs=epochs, batch_size=batch_size, lr=lr)
            mlp_logits, mlp_labels = _collect_mlp_logits(mlp, X_test, y_test, batch_size)
            mlp_final, mlp_probs, mlp_labels_np = _eval_with_curves(mlp_logits, mlp_labels)
            mlp_results[label] = {"final_metrics": mlp_final, **mlp_history}
            curves_data[label] = (mlp_probs, mlp_labels_np)
            _write_results(output_json, run_config, mlp_results, covseq_results)

    # ── CovSeq window-size sweep ─────────────────────────────────────────────
    if not no_covseq:
        for ws in window_sizes_list:
            print(f"\n── CovSeq  window={ws} ──")
            train_ds = CovSeqDataset(train_records, ws)
            test_ds  = CovSeqDataset(test_records,  ws)
            print(f"Train windows: {len(train_ds)}  test windows: {len(test_ds)}")

            covseq = CovSeqModel(
                compressed_size=compressed_size,
                input_size=hidden_size,
                hidden_sizes=covseq_hidden_list,
                output_size=1,
            )
            cs_history = train_covseq(covseq, train_ds, test_ds,
                                       epochs=epochs, batch_size=batch_size, lr=lr)
            cs_logits, cs_labels = _collect_covseq_logits(covseq, test_ds, batch_size)
            cs_final, cs_probs, cs_labels_np = _eval_with_curves(cs_logits, cs_labels)

            covseq_results[ws] = {"final_metrics": cs_final, **cs_history}
            curves_data[f"CovSeq w={ws}"] = (cs_probs, cs_labels_np)
            _write_results(output_json, run_config, mlp_results, covseq_results)

    # ── Results table ────────────────────────────────────────────────────────
    print("\n── Results ──")
    all_model_keys = list(mlp_results.keys()) + [f"CovSeq w={w}" for w in covseq_results]
    col_w = 14
    header = f"{'metric':<12}" + "".join(f"{k:>{col_w}}" for k in all_model_keys)
    print(header)
    print("-" * len(header))
    for key in ["f1", "precision", "recall", "tnr", "accuracy", "auprc", "auroc"]:
        row = f"{key:<12}"
        row += "".join(f"{mlp_results[k]['final_metrics'][key]:>{col_w}.4f}" for k in mlp_results)
        row += "".join(f"{covseq_results[w]['final_metrics'][key]:>{col_w}.4f}" for w in covseq_results)
        print(row)

    if plot_path is not None:
        plot_curves(curves_data, plot_path)


if __name__ == "__main__":
    main()
