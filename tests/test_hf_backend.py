"""TDD suite for the HF transformers backend that replaces vLLM prefill."""
from __future__ import annotations

import pytest
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def extractor():
    from train.hf_backend import HFHiddenStateExtractor

    return HFHiddenStateExtractor(
        model_name=MODEL_ID,
        layers=[0, 10, 23],
        dtype=torch.float32,
        device="cpu",
    )


def test_extract_returns_dict_keyed_by_layer(extractor):
    token_ids = extractor.tokenizer("hello world", return_tensors=None)["input_ids"]
    out = extractor.extract([token_ids])

    assert set(out.keys()) == {0, 10, 23}


def test_extract_hidden_states_shape(extractor):
    token_ids = extractor.tokenizer("hello world", return_tensors=None)["input_ids"]
    out = extractor.extract([token_ids])

    for layer_idx, tensors in out.items():
        assert len(tensors) == 1, f"layer {layer_idx}: expected one seq back"
        hs = tensors[0]
        assert hs.dim() == 2
        assert hs.shape[0] == len(token_ids)
        assert hs.shape[1] == extractor.hidden_size


def test_extract_returns_cpu_tensors(extractor):
    token_ids = extractor.tokenizer("hi", return_tensors=None)["input_ids"]
    out = extractor.extract([token_ids])
    for tensors in out.values():
        for hs in tensors:
            assert hs.device.type == "cpu"


def test_extract_handles_batch_of_varying_lengths(extractor):
    ids_short = extractor.tokenizer("hi", return_tensors=None)["input_ids"]
    ids_long = extractor.tokenizer(
        "this is a longer prompt with more tokens", return_tensors=None
    )["input_ids"]
    out = extractor.extract([ids_short, ids_long])

    for layer_idx, tensors in out.items():
        assert len(tensors) == 2
        assert tensors[0].shape[0] == len(ids_short)
        assert tensors[1].shape[0] == len(ids_long)
