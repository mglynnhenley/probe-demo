#!/usr/bin/env python
"""
fit linear model on activation data. extract R^2, p-value, and other metrics.
"""

import json
import time
import click
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import Optional, Any
import random
from sklearn.metrics import r2_score
from tqdm import tqdm

# local imports
from load_activ import load_activations, prepare_records

_RESULTS_DIR = Path(__file__).resolve().parent / "results"



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
@click.option("--output-json", type=click.Path(path_type=Path), default=_RESULTS_DIR / "fit_linear.json")
def main(
    activations_dir: Path,
    annotations_jsonl: Path,
    model_name: str,
    num_to_load: Optional[int],
    val_fraction: float,
    output_json: Optional[Path],
):
    """Fit linear model on activation data."""

    # loading data
    data = load_activations(activations_dir, annotations_jsonl, num_to_load=num_to_load)
    records = prepare_records(data, model_name, chat_template_kwargs={"enable_thinking": False})

    # ── Train / test split at record level ─────────────────────────────────
    random.shuffle(records)
    split = int(len(records) * (1 - val_fraction))
    train_records, test_records = records[:split], records[split:]
    print(f"Split: {len(train_records)} train / {len(test_records)} test records")

    # extracting hidden size
    hidden_size = records[0]["hidden_states"].shape[1]
    print(f"Hidden size: {hidden_size}")

    # config to save run info
    run_config: dict[str, Any] = {
        "activations_dir": str(activations_dir),
        "annotations_jsonl": str(annotations_jsonl),
        "model_name": model_name,
        "val_fraction": val_fraction,
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "hidden_size": hidden_size,
    }

    def flatten(rec_list, desc):
        xs, ys = [], []
        for rec in tqdm(rec_list, desc=desc, dynamic_ncols=True):
            labels = rec["token_labels"]
            keep = labels != -100
            if not keep.any():
                continue
            xs.append(rec["hidden_states"][keep].astype(np.float32))
            ys.append(labels[keep].astype(np.float32))
        return np.concatenate(xs), np.concatenate(ys)

    train_X, train_y = flatten(train_records, "Flattening train")
    print(f"Train tokens: {train_X.shape[0]:,}")

    model = LinearRegression()
    print("Fitting linear model ...")
    t0 = time.time()
    model.fit(train_X, train_y)
    print(f"Fitted in {time.time() - t0:.1f}s")

    test_X, test_y = flatten(test_records, "Flattening test")
    test_preds = model.predict(test_X)

    # evaluating model
    r2 = r2_score(test_y, test_preds)
    print(f"R^2: {r2}")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"config": run_config, "results": {"r2": r2}}, indent=2))
        print(f"Results written to {output_json}")


if __name__ == "__main__":
    main()