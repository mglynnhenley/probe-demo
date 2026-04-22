"""Hugging Face transformers backend for probe-training hidden-state extraction.

Replaces the vLLM prefill path: given token-id lists, run the HF model forward
with ``output_hidden_states=True`` and return per-layer completion hidden
states on CPU so they can feed the probe head unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFHiddenStateExtractor:
    def __init__(
        self,
        model_name: str,
        *,
        layers: List[int],
        dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.layers = list(layers)
        self.device = torch.device(device) if device is not None else _default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            output_hidden_states=True,
        )
        self.model.to(self.device).eval()
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def extract(self, token_id_lists: List[List[int]]) -> Dict[int, List[torch.Tensor]]:
        out: Dict[int, List[torch.Tensor]] = {layer: [] for layer in self.layers}
        for ids in token_id_lists:
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            # hidden_states is a tuple of (num_layers + 1) tensors; index 0 is embeddings,
            # index i+1 is the output of decoder layer i — match vLLM plugin convention
            # of "layer 0" meaning the first decoder block's output.
            for layer in self.layers:
                hs = outputs.hidden_states[layer + 1][0].to("cpu", dtype=torch.float32)
                out[layer].append(hs)
        return out


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
