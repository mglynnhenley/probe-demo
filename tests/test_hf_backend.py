"""TDD suite for the HF transformers backend that replaces vLLM prefill."""
from __future__ import annotations

import os
from pathlib import Path

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


@pytest.fixture(scope="module")
def lora_dir(tmp_path_factory):
    """Hermetic tiny PEFT adapter on MODEL_ID; override with PROBE_TEST_LORA_DIR."""
    override = os.environ.get("PROBE_TEST_LORA_DIR")
    if override:
        return Path(override)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    peft_model = get_peft_model(base, cfg)
    # PEFT inits lora_B to zero (adapter is identity until trained); we want a
    # nonzero adapter so the "hidden states differ" assertion has teeth.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.01)
    out = tmp_path_factory.mktemp("tiny_lora")
    peft_model.save_pretrained(out)
    return Path(out)


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


def test_lora_adapter_changes_hidden_states(extractor, lora_dir):
    """Loading a PEFT adapter into HFHiddenStateExtractor must shift hidden states."""
    from train.hf_backend import HFHiddenStateExtractor

    lora_extractor = HFHiddenStateExtractor(
        model_name=MODEL_ID,
        layers=[10],
        dtype=torch.float32,
        device="cpu",
        lora_path=lora_dir,
    )

    token_ids = extractor.tokenizer(
        "the quick brown fox jumps over the lazy dog", return_tensors=None
    )["input_ids"]
    base_hs = extractor.extract([token_ids])[10][0]
    lora_hs = lora_extractor.extract([token_ids])[10][0]

    assert base_hs.shape == lora_hs.shape
    diff = (base_hs - lora_hs).abs().max().item()
    assert diff > 1e-4, f"LoRA adapter did not change hidden states (max abs diff {diff})"
