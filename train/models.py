#!/usr/bin/env python
"""
Probe trained on vLLM hidden states for policy-violation detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class ProbeConfig:
    """Everything needed to construct, save, and interpret the probe head."""

    layer_idx: int
    hidden_size: int  # d_model of the underlying LLM
    underlying_model: Optional[str]  # base model this probe was trained on (optional for transfer)
    hidden_sizes: Optional[List[int]] = None  # MLP hidden layers; None or [] = linear probe
    output_size: int = 1
    path: Optional[Path] = None  # pretrained weights; if set, loaded in ValueHeadProbe.__init__
    policy: Optional[str] = None  # natural-language policy (optional metadata)


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


class ValueHeadProbe:
    """Wraps the probe MLP plus :class:`ProbeConfig` (architecture and provenance)."""

    def __init__(self, cfg: ProbeConfig) -> None:
        self.cfg = cfg
        self.layer_idx = cfg.layer_idx
        hs = cfg.hidden_sizes or []
        self.model = MLP(
            input_size=cfg.hidden_size,
            hidden_sizes=hs,
            output_size=cfg.output_size,
        ).float()
        if cfg.path is not None:
            self.load_from_state_dict()

    def save(self, path: Path | str) -> None:
        """Persist probe weights (``state_dict`` only; metadata lives in ProbeConfig / sidecar)."""
        torch.save(self.model.state_dict(), path)

    def load_from_state_dict(self) -> None:
        if self.cfg.path is None:
            raise ValueError("ProbeConfig.path is not set")
        self.model.load_state_dict(
            torch.load(self.cfg.path, map_location="cpu", weights_only=True)
        )
