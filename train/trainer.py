#!/usr/bin/env python
"""Hugging Face Trainer subclass: vLLM prefill hidden states → probe head.

Policy-violation labels are **severely imbalanced**: token ``1`` (violation) is rare; ``0``
(non-violation) dominates. Training uses :class:`~torch.nn.BCEWithLogitsLoss` with
``pos_weight`` matching the **effective** neg/pos ratio under row sampling (see ``data.train_sampler_weights_and_effective_pos_weight``) so gradients are not double-counted.
Eval reports precision on violations and a **majority baseline** (accuracy of
always predicting non-violation) for context.
"""

from __future__ import annotations

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

from models import ValueHeadProbe
from utils import (
    encode_probe_chat,
    map_char_labels_to_completion_token_labels,
    probe_completion_char_start,
)
from vllm_probe_plugin import extract_prefill_hidden_states


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
        # Stashed in compute_loss, merged into the next training ``log()`` (loss row) so we do not double-log.
        self._pending_probe_metrics: Optional[Dict[str, float]] = None

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

    def _clear_unlabelled(
        self, logits: torch.Tensor, annotations: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mask = annotations != -100
        logits = logits[mask]
        annotations = annotations[mask]
        if mask.sum() == 0:
            return None, None
        return logits, annotations

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

        encodings = [
            encode_probe_chat(
                self.processing_class,
                [{"role": "user", "content": p}, {"role": "assistant", "content": c}],
                chat_template_kwargs=self.chat_template_kwargs,
                return_offsets_mapping=True,
            )
            for p, c in zip(prompts, completions)
        ]
        token_id_lists = [enc["input_ids"] for enc in encodings]
        offset_mappings = [enc["offset_mapping"] for enc in encodings]

        comp_char_starts = [
            probe_completion_char_start(
                self.processing_class,
                p,
                c,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            for p, c in zip(prompts, completions)
        ]

        hidden_states_dict = self._get_hidden_states(token_id_lists)
        hidden_states_list = hidden_states_dict[self.probe.layer_idx]

        all_logits: List[torch.Tensor] = []
        all_ann_tok: List[torch.Tensor] = []
        for i in range(len(prompts)):
            ann_token, comp_idx = map_char_labels_to_completion_token_labels(
                annotations[i], offset_mappings[i], comp_char_starts[i]
            )
            hs_i = hidden_states_list[i][comp_idx:, :].float()
            ann_token = ann_token[: hs_i.shape[0]]

            logits_i = probe_model(hs_i)
            all_logits.append(logits_i)
            all_ann_tok.append(ann_token.to(self.args.device))

        logits = torch.cat(all_logits, dim=0)
        ann_tok = torch.cat(all_ann_tok, dim=0)

        logits, ann_tok = self._clear_unlabelled(logits, ann_tok)
        if logits is None:
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)

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
