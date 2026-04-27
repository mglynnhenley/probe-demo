#!/usr/bin/env python
"""Hugging Face Trainer subclass: vLLM prefill hidden states → probe head.

Policy-violation labels are **severely imbalanced**: token ``1`` (violation) is rare; ``0``
(non-violation) dominates. Training uses :class:`~torch.nn.BCEWithLogitsLoss` with
``pos_weight`` matching the **effective** neg/pos ratio under row sampling (see ``data.train_sampler_weights_and_effective_pos_weight``) so gradients are not double-counted.
Eval reports precision on violations and a **majority baseline** (accuracy of
always predicting non-violation) for context.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import vllm
from datasets import Dataset
from torch.utils.data import RandomSampler, WeightedRandomSampler
from transformers import Trainer, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import has_length
from vllm.lora.request import LoRARequest

from data import build_probe_feature_batches, prepare_probe_batch
from models import MultiProbeModel, ValueHeadProbe
from utils import save_training_metrics_json
from vllm_probe_plugin import extract_prefill_hidden_states


class StreamingLinearR2Reservoir:
    """Bounded-memory linear-fit diagnostic on streamed token embeddings.

    Keeps a uniform reservoir sample of labelled completion-token embeddings and
    periodically fits an ordinary least-squares linear model ``y ≈ Xw + b`` on
    that sample to estimate how linear the label map is in the underlying
    embeddings.
    """

    def __init__(self, reservoir_size: int = 8192, seed: int = 0) -> None:
        self.reservoir_size = reservoir_size
        self._rng = random.Random(seed)
        self._features: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        self._feature_dim: Optional[int] = None
        self.n_seen = 0
        self.n_stored = 0

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if self.reservoir_size <= 0 or features.numel() == 0:
            return
        if features.dim() != 2:
            raise ValueError(f"Expected features [n, d], got shape {tuple(features.shape)}")
        labels = labels.view(-1)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features rows {features.shape[0]} != labels rows {labels.shape[0]}"
            )

        features = features.detach().to("cpu", dtype=torch.float32)
        labels = labels.detach().to("cpu", dtype=torch.float32)

        if self._features is None or self._labels is None:
            self._feature_dim = int(features.shape[1])
            self._features = torch.empty(
                (self.reservoir_size, self._feature_dim), dtype=torch.float32
            )
            self._labels = torch.empty((self.reservoir_size,), dtype=torch.float32)
        elif features.shape[1] != self._feature_dim:
            raise ValueError(
                f"Feature dim changed from {self._feature_dim} to {features.shape[1]}"
            )

        assert self._features is not None
        assert self._labels is not None

        for row, label in zip(features, labels):
            self.n_seen += 1
            if self.n_stored < self.reservoir_size:
                idx = self.n_stored
                self.n_stored += 1
            else:
                idx = self._rng.randrange(self.n_seen)
                if idx >= self.reservoir_size:
                    continue
            self._features[idx].copy_(row)
            self._labels[idx] = label

    def compute_r2(self) -> Optional[float]:
        if (
            self._features is None
            or self._labels is None
            or self._feature_dim is None
            or self.n_stored <= self._feature_dim + 1
        ):
            return None

        X = self._features[: self.n_stored].to(dtype=torch.float64)
        y = self._labels[: self.n_stored].to(dtype=torch.float64)
        sst = torch.sum((y - y.mean()).square())
        if float(sst.item()) <= 1e-12:
            return None

        X_aug = torch.cat(
            [X, torch.ones((self.n_stored, 1), dtype=torch.float64)],
            dim=1,
        )
        solution = torch.linalg.lstsq(X_aug, y.unsqueeze(-1)).solution.squeeze(-1)
        y_hat = X_aug @ solution
        sse = torch.sum((y - y_hat).square())
        return float((1.0 - sse / sst).item())


class ProbeTrainer(Trainer):
    """Train a probe on completion tokens using vLLM prefill hidden states.

    After char→token mapping: ``1`` = violation (rare), ``0`` = non-violation (common),
    ``-100`` = ignore. Positive class in BCE is ``1``; use ``pos_weight`` when ``1`` is rare.
    Logits > 0 ⇒ predict violation.
    """

    def __init__(
        self,
        vllm_llm: vllm.LLM,
        tokenizer: PreTrainedTokenizerBase,
        probe: ValueHeadProbe,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: Callable[..., Dict[str, Any]],
        lora_request: LoRARequest | None,
        args: TrainingArguments,
        pos_weight: Optional[float] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler_weights: Optional[List[float]] = None,
        probe_training_metrics_interval_epochs: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=probe.model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            args=args,
            compute_metrics=self.compute_metrics,
            **kwargs,
        )
        self.vllm_llm = vllm_llm
        self.probe = probe
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
        self.lora_request = lora_request
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.train_sampler_weights = train_sampler_weights
        self.probe_training_metrics_interval_epochs = probe_training_metrics_interval_epochs
        self._next_probe_training_metrics_epoch = (
            probe_training_metrics_interval_epochs
            if probe_training_metrics_interval_epochs is not None
            and probe_training_metrics_interval_epochs > 0
            else None
        )
        self._latest_probe_metrics_batch: Optional[torch.Tensor] = None
        # Stashed in compute_loss, merged into the next training ``log()`` (loss row) so we do not double-log.
        self._pending_probe_metrics: Optional[Dict[str, float]] = None
        self._linear_r2_reservoir = StreamingLinearR2Reservoir(
            reservoir_size=8192,
            seed=int(getattr(args, "seed", 0)),
        )

    def log(self, logs: Any, start_time: Optional[float] = None) -> None:  # type: ignore[override]
        """Single ``log_history`` row per ``global_step``: merge probe metrics into the loss row; merge eval/runtime into the same step."""
        if not isinstance(logs, dict):
            return super().log(logs, start_time)

        logs = dict(logs)
        if "loss" in logs and self._pending_probe_metrics is not None:
            logs.update(self._pending_probe_metrics)
            self._pending_probe_metrics = None

        if self.state.log_history:
            last = self.state.log_history[-1]
            if last.get("step") == self.state.global_step:
                prev = self.state.log_history.pop()
                logs = {**prev, **logs}

        return super().log(logs, start_time)

    def _move_model_to_device(self, model: nn.Module, device: torch.device) -> nn.Module:
        # vLLM owns GPUs; probe stays on CPU (hidden states arrive as CPU tensors).
        return model

    def _get_hidden_states(self, token_id_lists: List[List[int]]) -> Dict[int, List[torch.Tensor]]:
        sampling_params = vllm.SamplingParams(n=1, temperature=0.0, max_tokens=1)
        outputs = self.vllm_llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids in token_id_lists],
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self.lora_request,
        )
        hidden_states_dict: Dict[int, List[torch.Tensor]] = {}
        for output in outputs:
            per_req = extract_prefill_hidden_states(output)
            for layer_id, hs in per_req.items():
                hidden_states_dict.setdefault(layer_id, []).append(hs)
        return hidden_states_dict

    def _stash_probe_metrics_batch(self, feature_batches: List[Any]) -> None:
        if feature_batches and isinstance(feature_batches[0], list):
            # multi_layer_covseq: feature_batches is List[List[ProbeFeatureBatch]]
            # stash as a list of the longest-window tensor per layer
            result = []
            for layer_batches in feature_batches:
                seq_batches = [
                    b.features for b in layer_batches
                    if isinstance(b.features, torch.Tensor) and b.features.dim() == 3
                ]
                result.append(
                    max(seq_batches, key=lambda t: int(t.shape[1])) if seq_batches else None
                )
            self._latest_probe_metrics_batch = result if any(x is not None for x in result) else None
            return
        seq_batches = [
            batch.features
            for batch in feature_batches
            if isinstance(batch.features, torch.Tensor) and batch.features.dim() == 3
        ]
        if not seq_batches:
            self._latest_probe_metrics_batch = None
            return
        self._latest_probe_metrics_batch = max(seq_batches, key=lambda batch: int(batch.shape[1]))

    def _update_linear_r2_reservoir(
        self,
        hidden_states_list: List[torch.Tensor],
        completion_token_labels: List[torch.Tensor],
    ) -> None:
        if not self.model.training:
            return
        for hidden_states, token_labels in zip(hidden_states_list, completion_token_labels):
            n_completion_tokens = int(token_labels.shape[0])
            completion_hidden_states = (
                hidden_states[-n_completion_tokens:] if n_completion_tokens else hidden_states[:0]
            )
            n_tokens = min(completion_hidden_states.shape[0], token_labels.shape[0])
            if n_tokens == 0:
                continue
            token_labels = token_labels[:n_tokens]
            keep = token_labels != -100
            if not keep.any():
                continue
            self._linear_r2_reservoir.update(
                completion_hidden_states[:n_tokens][keep].float(),
                token_labels[keep].float(),
            )

    def _maybe_print_probe_training_metrics(self) -> None:
        interval = self.probe_training_metrics_interval_epochs
        next_epoch = self._next_probe_training_metrics_epoch
        if interval is None or interval <= 0 or next_epoch is None:
            return

        epoch = self.state.epoch
        metrics_fn = getattr(self.probe.model, "training_metrics", None)
        batch = self._latest_probe_metrics_batch
        if epoch is None or epoch + 1e-12 < next_epoch or not callable(metrics_fn) or batch is None:
            return

        metrics = metrics_fn(batch)
        print("")
        print(f"[probe.training_metrics] epoch={epoch:.3f}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6g}")
        linear_r2 = self._linear_r2_reservoir.compute_r2()
        if linear_r2 is None:
            print(
                "  linear_r2_raw_embeddings: pending "
                f"(reservoir={self._linear_r2_reservoir.n_stored}/{self._linear_r2_reservoir.reservoir_size}, "
                f"seen={self._linear_r2_reservoir.n_seen})"
            )
        else:
            print(
                "  linear_r2_raw_embeddings: "
                f"{linear_r2:.6g} "
                f"(reservoir={self._linear_r2_reservoir.n_stored}, seen={self._linear_r2_reservoir.n_seen})"
            )

        while self._next_probe_training_metrics_epoch is not None and epoch + 1e-12 >= self._next_probe_training_metrics_epoch:
            self._next_probe_training_metrics_epoch += interval

    def compute_loss(
        self,
        probe_model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, List[str]]],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        prompts = inputs["prompt"]
        completions = inputs["completion"]
        annotations = inputs["annotations_val"].float().to(self.args.device)

        prepared_batch = prepare_probe_batch(
            self.processing_class,
            prompts,
            completions,
            annotations,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        hidden_states_dict = self._get_hidden_states(prepared_batch.token_id_lists)
        model_cfg = self.probe.cfg.model
        if model_cfg.model_type == "multi_layer_covseq":
            # Dict[layer_idx -> List[Tensor]] ordered by layer_indices
            hidden_states_per_layer = [
                hidden_states_dict[li] for li in model_cfg.layer_indices
            ]
            # Use first layer for R² reservoir diagnostic
            self._update_linear_r2_reservoir(
                hidden_states_per_layer[0],
                prepared_batch.completion_token_labels,
            )
        else:
            hidden_states_per_layer = hidden_states_dict[self.probe.layer_idx]
            self._update_linear_r2_reservoir(
                hidden_states_per_layer,
                prepared_batch.completion_token_labels,
            )
        feature_batches = build_probe_feature_batches(
            hidden_states_per_layer,
            prepared_batch.completion_token_labels,
            model_cfg,
        )
        self._stash_probe_metrics_batch(feature_batches)

        if not feature_batches:
            loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
            if return_outputs:
                empty_logits = torch.empty((0, 1), device=self.args.device)
                empty_labels = torch.empty((0,), device=self.args.device)
                return (loss, empty_logits, empty_labels)
            return loss

        logits = torch.cat([
            probe_model(batch.features) if not isinstance(batch.features, list)
            else probe_model([f.to(self.args.device) for f in batch.features])
            for batch in feature_batches
        ], dim=0)
        ann_tok = torch.cat([batch.labels.to(self.args.device) for batch in feature_batches], dim=0)

        loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(
            logits.squeeze(-1), ann_tok
        )

        with torch.no_grad():
            pred_viol = (logits.squeeze(-1) > 0).float()
            viol = ann_tok == 1.0
            ok = ann_tok == 0.0
            n_viol_tok = int(viol.sum().item())
            n_nonviol_tok = int(ok.sum().item())
            tp = pred_viol[viol].sum() if viol.any() else torch.tensor(0.0)
            fp = pred_viol[ok].sum() if ok.any() else torch.tensor(0.0)
            fn = (1.0 - pred_viol[viol]).sum() if viol.any() else torch.tensor(0.0)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            token_accuracy = pred_viol.eq(ann_tok).float().mean().item()
            metrics = {
                "n_violation_tokens_in_batch": float(n_viol_tok),
                "n_nonviolation_tokens_in_batch": float(n_nonviol_tok),
                "f1": f1.item(),
                "token_accuracy": token_accuracy,
                "prec_viol": prec.item(),
                "tpr": rec.item(),
                "tnr": (1.0 - pred_viol[ok]).mean().item() if ok.any() else 0.0,
                "frac_pred_viol": pred_viol.mean().item(),
            }
            if self.model.training:
                self._pending_probe_metrics = metrics

        if return_outputs:
            return (loss, logits, ann_tok)
        return loss

    def compute_metrics(self, eval_pred: Any) -> dict:
        logits_np = np.asarray(eval_pred.predictions)
        labels_np = np.asarray(eval_pred.label_ids)
        logits = torch.from_numpy(logits_np).squeeze(-1)
        labels = torch.from_numpy(labels_np).float()
        mask = labels != -100
        if not mask.any():
            return {
                "eval_f1": 0.0,
                "eval_token_accuracy": 0.0,
                "eval_precision_viol": 0.0,
                "eval_tpr": 0.0,
                "eval_tnr": 0.0,
                "eval_baseline_nonviol_acc": 0.0,
            }
        logits, labels = logits[mask], labels[mask]

        # Accuracy if we always predict non-violation (0): high when violations are rare.
        baseline_nonviol_acc = (labels == 0.0).float().mean()

        pred_viol = (logits > 0).float()
        viol, ok = labels == 1.0, labels == 0.0
        tp = pred_viol[viol].sum() if viol.any() else torch.tensor(0.0)
        fp = pred_viol[ok].sum() if ok.any() else torch.tensor(0.0)
        fn = (1.0 - pred_viol[viol]).sum() if viol.any() else torch.tensor(0.0)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8) if viol.any() else torch.tensor(0.0)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        tnr = (1.0 - pred_viol[ok]).mean() if ok.any() else torch.tensor(0.0)
        eval_token_accuracy = pred_viol.eq(labels).float().mean().item()
        return {
            "eval_f1": f1.item(),
            "eval_token_accuracy": eval_token_accuracy,
            "eval_precision_viol": prec.item(),
            "eval_tpr": rec.item(),
            "eval_tnr": tnr.item(),
            "eval_baseline_nonviol_acc": baseline_nonviol_acc.item(),
        }

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            out = self.compute_loss(model, inputs, return_outputs=True)
            loss, logits, labels = out

        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), logits.detach(), labels.detach())

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, List[str]]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._maybe_print_probe_training_metrics()
        return loss

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[Union[RandomSampler, WeightedRandomSampler]]:
        """Uniform random rows, or :class:`~torch.utils.data.WeightedRandomSampler` when ``train_sampler_weights`` is set."""
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if ds is None or not has_length(ds):
            return None

        weights = self.train_sampler_weights
        if weights is not None:
            if len(weights) != len(ds):
                raise ValueError(
                    f"train_sampler_weights length {len(weights)} != dataset length {len(ds)}."
                )
            n_zero = sum(1 for w in weights if w == 0.0)
            if n_zero:
                print(
                    f"[sampling] warning: {n_zero} examples have zero sampler weight (never drawn)"
                )
            return WeightedRandomSampler(
                weights=weights,
                num_samples=len(ds),
                replacement=True,
            )
        return RandomSampler(ds)

    def _save_checkpoint(self, model: nn.Module, trial: Any) -> None:
        super()._save_checkpoint(model, trial)
        if not self.args.should_save:
            return

        checkpoint_dir = (
            Path(self._get_output_dir(trial=trial))
            / f"checkpoint-{self.state.global_step}"
        )
        save_training_metrics_json(self, checkpoint_dir / "training_metrics.json")


class MultiProbeTrainer(ProbeTrainer):
    """ProbeTrainer variant that trains one CovSeq probe per hidden layer jointly.

    vLLM runs once per batch and returns hidden states for all requested layers.
    Each layer's probe computes a BCE loss; the mean loss is back-propagated through
    all probes simultaneously.  Eval metrics are averaged across layers.
    """

    def __init__(
        self,
        vllm_llm: vllm.LLM,
        tokenizer: PreTrainedTokenizerBase,
        probes: Dict[int, ValueHeadProbe],
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: Callable[..., Dict[str, Any]],
        lora_request: LoRARequest | None,
        args: TrainingArguments,
        pos_weight: Optional[float] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler_weights: Optional[List[float]] = None,
        probe_training_metrics_interval_epochs: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.probes = probes
        self.layer_indices = sorted(probes.keys())
        multi_model = MultiProbeModel(probes)

        # Bypass ProbeTrainer.__init__ (single-probe); call Trainer directly.
        Trainer.__init__(
            self,
            model=multi_model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            args=args,
            compute_metrics=self.compute_metrics,
            **kwargs,
        )

        self.vllm_llm = vllm_llm
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
        self.lora_request = lora_request
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.train_sampler_weights = train_sampler_weights
        self.probe_training_metrics_interval_epochs = probe_training_metrics_interval_epochs
        self._next_probe_training_metrics_epoch = (
            probe_training_metrics_interval_epochs
            if probe_training_metrics_interval_epochs is not None
            and probe_training_metrics_interval_epochs > 0
            else None
        )
        self._latest_probe_metrics_batches: Dict[int, Optional[torch.Tensor]] = {
            layer: None for layer in self.layer_indices
        }
        self._pending_probe_metrics: Optional[Dict[str, float]] = None
        self._linear_r2_reservoir = StreamingLinearR2Reservoir(
            reservoir_size=8192,
            seed=int(getattr(args, "seed", 0)),
        )

    def compute_loss(
        self,
        multi_probe_model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, List[str]]],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        prompts = inputs["prompt"]
        completions = inputs["completion"]
        annotations = inputs["annotations_val"].float().to(self.args.device)

        prepared_batch = prepare_probe_batch(
            self.processing_class,
            prompts,
            completions,
            annotations,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        hidden_states_dict = self._get_hidden_states(prepared_batch.token_id_lists)

        # Track linear separability diagnostics using the first layer only.
        self._update_linear_r2_reservoir(
            hidden_states_dict[self.layer_indices[0]],
            prepared_batch.completion_token_labels,
        )

        losses: List[torch.Tensor] = []
        all_logits_layers: List[torch.Tensor] = []
        all_ann_tok: Optional[torch.Tensor] = None

        for layer_idx in self.layer_indices:
            hs_list = hidden_states_dict[layer_idx]
            feature_batches = build_probe_feature_batches(
                hs_list,
                prepared_batch.completion_token_labels,
                self.probes[layer_idx].cfg.model,
            )

            seq_batches = [
                b.features
                for b in feature_batches
                if isinstance(b.features, torch.Tensor) and b.features.dim() == 3
            ]
            if seq_batches:
                self._latest_probe_metrics_batches[layer_idx] = max(
                    seq_batches, key=lambda b: int(b.shape[1])
                )

            if not feature_batches:
                continue

            logits = torch.cat(
                [multi_probe_model.forward_for_layer(layer_idx, b.features) for b in feature_batches],
                dim=0,
            )
            ann_tok = torch.cat([b.labels.to(self.args.device) for b in feature_batches], dim=0)

            layer_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(
                logits.squeeze(-1), ann_tok
            )
            losses.append(layer_loss)
            all_logits_layers.append(logits.detach())
            all_ann_tok = ann_tok

        if not losses:
            loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
            if return_outputs:
                return (loss, torch.empty((0, 1), device=self.args.device), torch.empty((0,), device=self.args.device))
            return loss

        loss = torch.stack(losses).mean()
        mean_logits = torch.stack(all_logits_layers).mean(dim=0)

        with torch.no_grad():
            pred_viol = (mean_logits.squeeze(-1) > 0).float()
            viol = all_ann_tok == 1.0
            ok = all_ann_tok == 0.0
            n_viol_tok = int(viol.sum().item())
            n_nonviol_tok = int(ok.sum().item())
            tp = pred_viol[viol].sum() if viol.any() else torch.tensor(0.0)
            fp = pred_viol[ok].sum() if ok.any() else torch.tensor(0.0)
            fn = (1.0 - pred_viol[viol]).sum() if viol.any() else torch.tensor(0.0)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            token_accuracy = pred_viol.eq(all_ann_tok).float().mean().item()
            metrics = {
                "n_violation_tokens_in_batch": float(n_viol_tok),
                "n_nonviolation_tokens_in_batch": float(n_nonviol_tok),
                "f1": f1.item(),
                "token_accuracy": token_accuracy,
                "prec_viol": prec.item(),
                "tpr": rec.item(),
                "tnr": (1.0 - pred_viol[ok]).mean().item() if ok.any() else 0.0,
                "frac_pred_viol": pred_viol.mean().item(),
            }
            if self.model.training:
                self._pending_probe_metrics = metrics

        if return_outputs:
            return (loss, mean_logits, all_ann_tok)
        return loss

    def _maybe_print_probe_training_metrics(self) -> None:
        interval = self.probe_training_metrics_interval_epochs
        next_epoch = self._next_probe_training_metrics_epoch
        if interval is None or interval <= 0 or next_epoch is None:
            return

        epoch = self.state.epoch
        if epoch is None or epoch + 1e-12 < next_epoch:
            return

        print(f"\n[multi_probe.training_metrics] epoch={epoch:.3f}")
        linear_r2 = self._linear_r2_reservoir.compute_r2()
        first = self.layer_indices[0]
        if linear_r2 is None:
            print(
                f"  linear_r2_raw_embeddings (layer {first}): pending "
                f"(reservoir={self._linear_r2_reservoir.n_stored}/{self._linear_r2_reservoir.reservoir_size}, "
                f"seen={self._linear_r2_reservoir.n_seen})"
            )
        else:
            print(
                f"  linear_r2_raw_embeddings (layer {first}): {linear_r2:.6g} "
                f"(reservoir={self._linear_r2_reservoir.n_stored}, seen={self._linear_r2_reservoir.n_seen})"
            )

        for layer_idx in self.layer_indices:
            probe_model = self.probes[layer_idx].model
            metrics_fn = getattr(probe_model, "training_metrics", None)
            batch = self._latest_probe_metrics_batches.get(layer_idx)
            if not callable(metrics_fn) or batch is None:
                continue
            metrics = metrics_fn(batch)
            print(f"  [layer {layer_idx}]")
            for key, value in metrics.items():
                print(f"    {key}: {value:.6g}")

        while self._next_probe_training_metrics_epoch is not None and epoch + 1e-12 >= self._next_probe_training_metrics_epoch:
            self._next_probe_training_metrics_epoch += interval

    def _save_checkpoint(self, model: nn.Module, trial: Any) -> None:
        super()._save_checkpoint(model, trial)
        if not self.args.should_save:
            return

        checkpoint_dir = (
            Path(self._get_output_dir(trial=trial))
            / f"checkpoint-{self.state.global_step}"
        )
        for layer_idx, probe in self.probes.items():
            probe.model.load_state_dict(self.model.models[str(layer_idx)].state_dict())
            probe.save(checkpoint_dir / f"probe_head_layer{layer_idx}.bin")
