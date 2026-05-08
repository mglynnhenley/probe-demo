#!/usr/bin/env python
"""
Probe trained on vLLM hidden states for policy-violation detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class CovSeqConfig:
    """Hyperparameters for covariance-pooled sequence probes."""

    window_size: int = 8
    compressed_size: int = 64
    training_metrics_interval_epochs: float = 1.0


@dataclass
class ProbeModelConfig:
    """Architecture configuration for the probe head."""

    probe_model_type: str = "mlp"
    hidden_size: int = 0
    hidden_sizes: List[int] = field(default_factory=list)
    output_size: int = 1
    covseq: CovSeqConfig = field(default_factory=CovSeqConfig)
    # vLLM layer indices the probe reads hidden states from. Always non-empty.
    # ``mlp`` and ``covseq`` use a single layer (``layer_indices == [layer]``);
    # ``multi_layer_covseq`` requires at least 2 layers.
    layer_indices: List[int] = field(default_factory=list)

    @property
    def model_type(self) -> str:
        return self.probe_model_type.lower()

    @property
    def layer_idx(self) -> int:
        """First layer used by this probe.

        For ``mlp`` / ``covseq`` this is *the* layer the probe reads. For
        ``multi_layer_covseq`` it is the representative tag returned alongside
        per-token scores (the probe internally consumes all ``layer_indices``).
        """
        if not self.layer_indices:
            raise ValueError("ProbeModelConfig.layer_indices is empty")
        if self.model_type != "multi_layer_covseq" and len(self.layer_indices) != 1:
            raise ValueError(
                f"Single-layer probe requires exactly one entry in layer_indices, "
                f"got {self.layer_indices!r}"
            )
        return self.layer_indices[0]


@dataclass
class ProbeConfig:
    """Everything needed to construct, save, and interpret the probe head."""

    model: ProbeModelConfig
    underlying_model: Optional[str]  # base model this probe was trained on (optional for transfer)
    path: Optional[Path] = None  # pretrained weights; if set, loaded in ValueHeadProbe.__init__
    policy: Optional[str] = None  # natural-language policy (optional metadata)
    dtype: Optional[torch.dtype] = None  # if set, model is initialised in this dtype

    @property
    def layer_indices(self) -> List[int]:
        return self.model.layer_indices

    @property
    def layer_idx(self) -> int:
        return self.model.layer_idx

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.path is not None:
            payload["path"] = str(self.path)
        payload["dtype"] = str(self.dtype) if self.dtype is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbeConfig":
        payload = dict(payload)
        # Back-compat: older checkpoints stored layer_idx at the top level.
        legacy_layer_idx = payload.pop("layer_idx", None)
        if payload.get("path") is not None:
            payload["path"] = Path(payload["path"])
        if payload.get("dtype") is not None:
            payload["dtype"] = getattr(torch, payload["dtype"].replace("torch.", ""))
        model_payload = dict(payload["model"])
        model_payload["covseq"] = CovSeqConfig(**model_payload.get("covseq", {}))
        if not model_payload.get("layer_indices") and legacy_layer_idx is not None:
            model_payload["layer_indices"] = [int(legacy_layer_idx)]
        payload["model"] = ProbeModelConfig(**model_payload)
        return cls(**payload)


class MLP(nn.Module):
    """Feed-forward probe head with GELU between hidden layers."""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers_lst: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers_lst.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers_lst.append(nn.GELU())
        self.layers = nn.Sequential(*layers_lst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CovSeqModel(nn.Module):
    """sequence model with covariant pooling across tokens"""
    def __init__(self, compressed_size: int, input_size: int, hidden_sizes: List[int], output_size: int) -> None:
        """
        input_size - d_model input to the L and R compressing MLPs
        compressed_size - output of the compressing MLPs and the rank of the compressed covariance matrix
        hidden_sizes - hidden size of the MLP acting on the compressed covariance matrix when flattened (compressed_size ** 2)
        output_size - output of the MLP, typically 1 for a classification task
        """
        super().__init__()
        # compressing MLPs into the covariance matrix
        # note that the L, R matrices act on each embedding individually, and these outputs are pooled into a cov matrix approximation
        self.L = nn.Linear(input_size, compressed_size, bias=False) # no bias entries as this should capture second moment terms only
        self.R = nn.Linear(input_size, compressed_size, bias=False)

        # MLP which operates on flattened covariance matrix
        self.mlp = MLP(
            input_size = compressed_size ** 2,
            hidden_sizes = hidden_sizes,
            output_size = output_size,
        )

    def pool(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [batch, seq_len, d_model]
        returns: M [batch, compressed_size, compressed_size]
        """
        seq_len = X.shape[1]
        XL = self.L(X) # [batch, seq_len, compressed_size]
        XR = self.R(X) # [batch, seq_len, compressed_size]
        M = XL.mT @ XR / seq_len # [batch, compressed_size, compressed_size]
        return M # this will then be flattened

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        M = self.pool(X)
        M = M.view(M.shape[0], -1)
        return self.mlp(M)

    def reconstruction_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Low-cost sequence-level proxy for anti-collapse regularization.

        This computes the RMS Frobenius norm of the sequence-level tensor
        ``XR @ XL^T / seq_len`` with shape ``[batch, seq_len, seq_len]``.
        By ``||AB||_F = ||BA||_F`` for the corresponding product, this is the
        same norm as the pooled compressed covariance ``XL^T @ XR / seq_len``,
        but avoids materializing a larger feature-space tensor. Dividing the
        Frobenius norm by ``seq_len`` normalizes by ``sqrt(seq_len * seq_len)``,
        so the metric is comparable across different window sizes.
        """
        seq_len = X.shape[1]
        XL = self.L(X)  # [batch, seq_len, compressed_size]
        XR = self.R(X)  # [batch, seq_len, compressed_size]
        sequence_proxy = XR @ XL.mT / seq_len  # [batch, seq_len, seq_len]
        # note that the right way to normalise the frobenius norm is with the sqrt of the number of elements in the matrix, i.e. sqrt(seq_len * seq_len)
        rms_frobenius_norm = torch.norm(sequence_proxy, "fro", dim=(-2, -1)) / seq_len
        return rms_frobenius_norm.mean()

    def training_metrics(self, X: torch.Tensor) -> dict[str, float]:
        """
        returns information relevant to model convergence and catching collapse early in training
        
        X: [batch, seq_len, d_model]
        returns: dict of training metrics
        """
        # looking at gradient norms or L and R matrices as well as the rank
        l_grad_norm = (
            torch.norm(self.L.weight.grad, "fro").item()
            if self.L.weight.grad is not None
            else 0.0
        )
        r_grad_norm = (
            torch.norm(self.R.weight.grad, "fro").item()
            if self.R.weight.grad is not None
            else 0.0
        )
        l_cond_number = torch.linalg.cond(self.L.weight)
        r_cond_number = torch.linalg.cond(self.R.weight)
        l_rank = torch.linalg.matrix_rank(self.L.weight)
        r_rank = torch.linalg.matrix_rank(self.R.weight)
        # getting some analytics about compressed matrix
        with torch.no_grad():
            M = self.pool(X) # [batch, compressed_size, compressed_size]
            M_rank = torch.linalg.matrix_rank(M).float().mean().item()
            M_cond_number = torch.linalg.cond(M).mean().item()
            # variance of flattened M across batch (catches a collapse in embeddings)
            M_variance = M.flatten(start_dim=1).var(dim=0).mean().item() # dim = 0 as this is across the batch

        # anti-collapse proxy norm as a training diagnostic
        with torch.no_grad():
            reconstruction_proxy_norm = self.reconstruction_loss(X).item()

        # what's the gradient norm on the MLP?
        mlp_weight_grad_norms = [
            torch.norm(p.grad, "fro")
            for p in self.mlp.parameters()
            if p.grad is not None and p.dim() == 2  # weight matrices only, skip biases
        ]
        mlp_grad_norm = (
            torch.norm(torch.stack(mlp_weight_grad_norms)).item()
            if mlp_weight_grad_norms
            else 0.0
        )

        return {
            'l_grad_norm': l_grad_norm,
            'r_grad_norm': r_grad_norm,
            'l_cond_number': l_cond_number.item(),
            'r_cond_number': r_cond_number.item(),
            'l_rank': l_rank.item(),
            'r_rank': r_rank.item(),
            'M_rank': M_rank,
            'M_cond_number': M_cond_number,
            'M_variance': M_variance,
            'reconstruction_proxy_norm': reconstruction_proxy_norm,
            'mlp_grad_norm': mlp_grad_norm,
        }


class MultiLayerCovSeqModel(nn.Module):
    """Per-layer CovSeq pooling concatenated into a single MLP.

    Each layer gets its own independent L/R compressors. Their pooled covariance
    matrices (each ``compressed_size²``-dim) are concatenated and fed through a
    shared MLP to produce a single scalar per token window.
    """

    def __init__(
        self,
        n_layers: int,
        compressed_size: int,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.compressed_size = compressed_size
        # One pair of compressors per layer; no output MLP on each (pool only)
        self.L_layers = nn.ModuleList(
            [nn.Linear(input_size, compressed_size, bias=False) for _ in range(n_layers)]
        )
        self.R_layers = nn.ModuleList(
            [nn.Linear(input_size, compressed_size, bias=False) for _ in range(n_layers)]
        )
        self.mlp = MLP(
            input_size=n_layers * compressed_size ** 2,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
        )

    def pool_layer(self, X: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Pool one layer's hidden states into a compressed covariance matrix.

        X: [batch, seq_len, hidden_size]
        returns: [batch, compressed_size * compressed_size]
        """
        seq_len = X.shape[1]
        XL = self.L_layers[layer_idx](X)
        XR = self.R_layers[layer_idx](X)
        M = XL.mT @ XR / seq_len
        return M.view(M.shape[0], -1)

    def forward(self, X_per_layer: List[torch.Tensor]) -> torch.Tensor:
        """
        X_per_layer: list of n_layers tensors each [batch, seq_len, hidden_size]
        returns: [batch, output_size]
        """
        pooled = [self.pool_layer(X, i) for i, X in enumerate(X_per_layer)]
        cat = torch.cat(pooled, dim=-1)
        return self.mlp(cat)

    def training_metrics(self, X_per_layer: List[torch.Tensor]) -> dict:
        metrics: dict = {}
        for i, X in enumerate(X_per_layer):
            with torch.no_grad():
                M = self.pool_layer(X, i).view(X.shape[0], self.compressed_size, self.compressed_size)
                metrics[f"layer{i}_M_variance"] = M.flatten(start_dim=1).var(dim=0).mean().item()
                metrics[f"layer{i}_M_rank"] = torch.linalg.matrix_rank(M).float().mean().item()
        mlp_grad_norms = [
            torch.norm(p.grad, "fro")
            for p in self.mlp.parameters()
            if p.grad is not None and p.dim() == 2
        ]
        metrics["mlp_grad_norm"] = (
            torch.norm(torch.stack(mlp_grad_norms)).item() if mlp_grad_norms else 0.0
        )
        return metrics


class ValueHeadProbe:
    """Wraps the probe MLP plus :class:`ProbeConfig` (architecture and provenance)."""

    def __init__(self, cfg: ProbeConfig) -> None:
        self.cfg = cfg
        if not cfg.model.layer_indices:
            raise ValueError("ProbeModelConfig.layer_indices must not be empty")
        self.model = self._build_model()
        if cfg.path is not None:
            self.load_from_state_dict()

    @property
    def layer_indices(self) -> List[int]:
        return self.cfg.model.layer_indices

    @property
    def layer_idx(self) -> int:
        return self.cfg.model.layer_idx

    def _build_model(self) -> nn.Module:
        model_cfg = self.cfg.model
        n_layers = len(model_cfg.layer_indices)
        if model_cfg.model_type == "mlp":
            if n_layers != 1:
                raise ValueError(
                    f"mlp probe requires exactly one layer in layer_indices, got {model_cfg.layer_indices!r}"
                )
            m = MLP(
                input_size=model_cfg.hidden_size,
                hidden_sizes=model_cfg.hidden_sizes,
                output_size=model_cfg.output_size,
            )
        elif model_cfg.model_type == "covseq":
            if n_layers != 1:
                raise ValueError(
                    f"covseq probe requires exactly one layer in layer_indices, got {model_cfg.layer_indices!r}"
                )
            if model_cfg.covseq.window_size < 2:
                raise ValueError("covseq.window_size must be at least 2")
            m = CovSeqModel(
                compressed_size=model_cfg.covseq.compressed_size,
                input_size=model_cfg.hidden_size,
                hidden_sizes=model_cfg.hidden_sizes,
                output_size=model_cfg.output_size,
            )
        elif model_cfg.model_type == "multi_layer_covseq":
            if model_cfg.covseq.window_size < 2:
                raise ValueError("covseq.window_size must be at least 2")
            if n_layers < 2:
                raise ValueError("multi_layer_covseq requires at least 2 layer_indices")
            m = MultiLayerCovSeqModel(
                n_layers=n_layers,
                compressed_size=model_cfg.covseq.compressed_size,
                input_size=model_cfg.hidden_size,
                hidden_sizes=model_cfg.hidden_sizes,
                output_size=model_cfg.output_size,
            )
        else:
            raise ValueError(f"Unknown probe model type: {model_cfg.probe_model_type!r}")
        if self.cfg.dtype is not None:
            m = m.to(self.cfg.dtype)
        return m

    def _validate_loaded_config(self, saved_cfg: ProbeConfig) -> None:
        current = self.cfg.model
        saved = saved_cfg.model
        mismatches: list[str] = []
        fields_to_check = ("probe_model_type", "hidden_size", "hidden_sizes", "output_size", "layer_indices")
        for field_name in fields_to_check:
            if getattr(current, field_name) != getattr(saved, field_name):
                mismatches.append(
                    f"{field_name}: expected {getattr(current, field_name)!r}, "
                    f"found {getattr(saved, field_name)!r}"
                )
        if current.covseq != saved.covseq:
            mismatches.append(f"covseq: expected {current.covseq!r}, found {saved.covseq!r}")
        if mismatches:
            raise ValueError(
                "Checkpoint probe config does not match the requested architecture: "
                + "; ".join(mismatches)
            )

    def save(self, path: Path | str) -> None:
        """Persist probe weights and architecture metadata."""
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "probe_config": self.cfg.to_dict(),
            },
            path,
        )

    def load_from_state_dict(self) -> None:
        if self.cfg.path is None:
            raise ValueError("ProbeConfig.path is not set")
        path = Path(self.cfg.path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file as safetensors_load
            state_dict = safetensors_load(str(path), device="cpu")
            # HF Trainer wraps the model in MultiProbeModel (_probes.<layer_idx>.*)
            # or saves bare keys for single-layer ProbeTrainer. Strip any prefix
            # that doesn't match the probe model's own parameter names.
            own_keys = set(self.model.state_dict().keys())
            if not own_keys.issubset(state_dict.keys()):
                # Try stripping one level of prefix (e.g. "_probes.45.")
                for key in list(state_dict.keys()):
                    stripped = ".".join(key.split(".")[2:]) if key.count(".") >= 2 else key
                    if stripped in own_keys and key not in own_keys:
                        state_dict[stripped] = state_dict.pop(key)
            # Drop any remaining foreign keys
            state_dict = {k: v for k, v in state_dict.items() if k in own_keys}
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict) and "state_dict" in payload:
                saved_cfg_raw = payload.get("probe_config")
                if isinstance(saved_cfg_raw, dict):
                    self._validate_loaded_config(ProbeConfig.from_dict(saved_cfg_raw))
                state_dict = payload["state_dict"]
            else:
                state_dict = payload
        self.model.load_state_dict(state_dict)


class MultiProbeModel(nn.Module):
    """Wraps multiple per-layer :class:`ValueHeadProbe` instances as a single ``nn.Module``."""

    def __init__(self, probes: Dict[int, "ValueHeadProbe"]) -> None:
        super().__init__()
        self._probes = nn.ModuleDict({str(k): v.model for k, v in probes.items()})

    def forward_for_layer(self, layer_idx: int, features: torch.Tensor) -> torch.Tensor:
        return self._probes[str(layer_idx)](features)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Use forward_for_layer instead")
