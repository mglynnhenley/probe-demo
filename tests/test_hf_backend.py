"""TDD suite for the HF transformers backend that replaces vLLM prefill."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def extractor():
    from hf_backend import HFHiddenStateExtractor

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


def test_train_py_end_to_end(tmp_path):
    """train.py runs start-to-finish on a toy dataset and writes all expected artifacts."""
    import json
    import subprocess
    import sys

    jsonl = tmp_path / "annotations.jsonl"
    rows = [
        {
            "question": "give an opinion",
            "completion": "yes definitely it is",
            "annotations": [{"span": "definitely", "index": 4, "label": 1}],
        },
        {
            "question": "name a color",
            "completion": "the sky is blue",
            "annotations": [],
        },
        {
            "question": "do it",
            "completion": "no absolutely not",
            "annotations": [{"span": "absolutely", "index": 3, "label": 1}],
        },
        {
            "question": "describe weather",
            "completion": "it is raining outside",
            "annotations": [],
        },
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""\
model_name: {MODEL_ID}
dtype: float32
max_model_len: 256
layer_idx: 10
probe:
  probe_model_type: mlp
  hidden_size: 0
  hidden_sizes: [16]
  output_size: 1
annotations_jsonl: {jsonl}
val_fraction: 0.5
pos_weight: auto
epochs: 1
train_batch_size: 2
grad_accumulation_steps: 1
probe_lr: 0.001
warmup_steps: 0
val_interval: 500
log_interval: 1
checkpoint_interval: 500
output_dir: {output_dir}
seed: 0
"""
    )

    repo_root = Path(__file__).resolve().parent.parent
    train_script = repo_root / "train" / "train.py"
    result = subprocess.run(
        [sys.executable, str(train_script), str(config_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    assert (output_dir / "probe_head.bin").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "training_metrics.json").exists()


def test_probe_trainer_compute_loss_with_hf_extractor(extractor, tmp_path):
    """ProbeTrainer must accept an HFHiddenStateExtractor and produce a scalar loss."""
    from datasets import Dataset
    from transformers import TrainingArguments

    from data import collate_fn
    from models import (
        CovSeqConfig,
        ProbeConfig,
        ProbeModelConfig,
        ValueHeadProbe,
    )
    from trainer import ProbeTrainer

    rows = [
        {"prompt": "name a color", "completion": "blue", "annotations_val": [0.0] * 4},
        {
            "prompt": "give an opinion",
            "completion": "yes it is",
            "annotations_val": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
    ds = Dataset.from_list(rows)

    probe_cfg = ProbeConfig(
        layer_idx=10,
        model=ProbeModelConfig(
            probe_model_type="mlp",
            hidden_size=extractor.hidden_size,
            hidden_sizes=[16],
            output_size=1,
            covseq=CovSeqConfig(),
        ),
        underlying_model=MODEL_ID,
    )
    probe = ValueHeadProbe(probe_cfg)

    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        use_cpu=True,
        report_to=[],
    )

    trainer = ProbeTrainer(
        hf_extractor=extractor,
        probe=probe,
        train_dataset=ds,
        eval_dataset=ds,
        data_collator=collate_fn,
        args=args,
    )

    batch = collate_fn([ds[0], ds[1]])
    loss = trainer.compute_loss(probe.model, batch)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()


def test_trainer_module_has_no_vllm_imports():
    """Regression: trainer module must not import vllm or vllm_probe_plugin."""
    import trainer as trainer_mod

    src = Path(trainer_mod.__file__).read_text()
    assert "import vllm" not in src
    assert "vllm_probe_plugin" not in src
    assert "LoRARequest" not in src


def test_lora_adapter_changes_hidden_states(extractor, lora_dir):
    """Loading a PEFT adapter into HFHiddenStateExtractor must shift hidden states."""
    from hf_backend import HFHiddenStateExtractor

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
