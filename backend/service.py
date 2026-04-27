"""Inference service: HF transformers + probe scoring."""

from __future__ import annotations

import difflib
import json
import os
import sys
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

# Probe model classes live in the sibling train package.
_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from models import CovSeqModel, ValueHeadProbe, ProbeConfig  # noqa: E402  (train package)
from utils import apply_chat_template_to_text  # noqa: E402  (train package)


_KNOWN_ENV_VARS = {
    "MODEL_NAME",
    "PROBE_PATH",
    "HF_DTYPE",
    "HF_DEVICE",
    "HF_TOKEN",
    "HF_TRUST_REMOTE_CODE",
    "MAX_MODEL_LEN",
    "HOST",
    "PORT",
}


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _resolve_dtype(name: Optional[str]) -> torch.dtype:
    if not name or str(name).lower() == "auto":
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            return torch.bfloat16
        return torch.float32
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(_DTYPE_MAP) + ['auto']}")
    return _DTYPE_MAP[name]


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _warn_suspicious_env_vars() -> None:
    """Warn if any env var looks like a typo of a known backend var."""
    for key in os.environ:
        if key in _KNOWN_ENV_VARS:
            continue
        matches = difflib.get_close_matches(key, _KNOWN_ENV_VARS, n=1, cutoff=0.8)
        if matches:
            print(
                f"[WARN] Unrecognised env var {key!r} looks like a typo of {matches[0]!r}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Resolve base model + training sidecar (chat template kwargs, dtype, etc.)
# ---------------------------------------------------------------------------

def _expand_probe_checkpoint(raw: str) -> Path:
    """If *raw* is a directory, prefer ``probe_head.bin`` inside it (trainer output layout)."""
    p = Path(raw).expanduser()
    if p.is_dir():
        cand = p / "probe_head.bin"
        if cand.is_file():
            return cand
    return p


def _read_training_sidecar_next_to_probe(probe_file: Path) -> dict[str, Any]:
    cfg = probe_file.parent / "config.json"
    if not cfg.is_file():
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_inference_model_and_config(raw_probe_paths: list[str]) -> tuple[str, dict[str, Any]]:
    """Infer the Hugging Face model id from env, probe checkpoints, or training ``config.json``.

    Training saves :class:`~models.ProbeConfig` with ``underlying_model`` set to the HF base model.
    The run directory also contains ``config.json`` (asdict of TrainingConfig) with ``model_name``,
    ``chat_template_kwargs``, ``dtype``, etc.
    """
    env_model = (os.environ.get("MODEL_NAME") or "").strip()
    inferred: str | None = None
    sidecar: dict[str, Any] = {}

    for raw in raw_probe_paths:
        p = _expand_probe_checkpoint(raw)
        if p.is_file():
            sc = _read_training_sidecar_next_to_probe(p)
            if sc:
                sidecar = sc
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if isinstance(payload, dict) and "probe_config" in payload:
                pc = ProbeConfig.from_dict(payload["probe_config"])
                if pc.underlying_model:
                    inferred = pc.underlying_model
                    break
        elif p.is_dir() and (p / "config.json").is_file():
            try:
                sidecar = json.loads((p / "config.json").read_text(encoding="utf-8"))
            except Exception:
                pass

    # Directory-only PROBE_PATH: sidecar may already have model_name
    if not sidecar and raw_probe_paths:
        for raw in raw_probe_paths:
            d = Path(raw).expanduser()
            if d.is_dir() and (d / "config.json").is_file():
                try:
                    sidecar = json.loads((d / "config.json").read_text(encoding="utf-8"))
                except Exception:
                    pass
                break

    sidecar_model = (sidecar.get("model_name") if sidecar else None) or None
    model = env_model or inferred or (str(sidecar_model) if sidecar_model else "")
    if not model:
        raise RuntimeError(
            "Could not determine which Hugging Face model to load.\n"
            "  • export MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct\n"
            "  • or export PROBE_PATH to your probe checkpoint:\n"
            "      export PROBE_PATH=/path/to/output/run/probe_head.bin\n"
            "    or the training output directory containing probe_head.bin + config.json\n"
            "The checkpoint must include probe_config.underlying_model (from training), "
            "or sit next to config.json with \"model_name\"."
        )

    print(f"[INIT] HF model: {model}", flush=True)
    if env_model:
        print("[INIT] (MODEL_NAME env overrides inferred id)", flush=True)
    return model, sidecar


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probe(path: str | Path) -> ValueHeadProbe:
    """Load a ValueHeadProbe checkpoint from *path* onto CPU."""
    path = _expand_probe_checkpoint(str(path))
    if not path.is_file():
        raise FileNotFoundError(
            f"Probe weights not found at {path} (expected a file or a directory containing probe_head.bin)"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(payload, dict) and "probe_config" in payload):
        raise ValueError(f"Checkpoint at {path} has no 'probe_config' key")
    cfg = ProbeConfig.from_dict({**payload["probe_config"], "path": str(path)})
    probe = ValueHeadProbe(cfg)
    probe.model.eval()
    return probe


# ---------------------------------------------------------------------------
# Per-step probe runners
# ---------------------------------------------------------------------------

def _run_mlp_step(probe: ValueHeadProbe, hs: torch.Tensor) -> float:
    with torch.no_grad():
        logit = probe.model(hs.to(torch.float32))
        return float(torch.sigmoid(logit).squeeze())


def _run_covseq_step(
    probe: ValueHeadProbe,
    hs: torch.Tensor,
    buf: deque,
) -> float:
    vec = hs.squeeze(0).to(torch.float32)
    buf.append(vec)
    # Use only real hidden states — training used truncated windows of length 1..T-1
    # for early tokens, never zero-padded full windows. Zero-padding causes a 1/T
    # scale error in the covariance matrix that produces out-of-distribution scores.
    window = torch.stack(list(buf), dim=0).unsqueeze(0)  # [1, actual_len, d_model]
    with torch.no_grad():
        logit = probe.model(window)
        return float(torch.sigmoid(logit).squeeze())


def run_probe_step(
    probe: ValueHeadProbe,
    hs: torch.Tensor,
    buf: deque | None,
) -> tuple[float, deque | None]:
    """Run one decode step through *probe*. Returns (score, updated_buf)."""
    if isinstance(probe.model, CovSeqModel):
        if buf is None:
            buf = deque(maxlen=probe.cfg.model.covseq.window_size)
        score = _run_covseq_step(probe, hs, buf)
    else:
        score = _run_mlp_step(probe, hs)
    return score, buf


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> int:
    """Sample one token id from a 1-D logits tensor.

    Greedy when ``temperature <= 0``; otherwise applies temperature and
    nucleus (top-p) filtering before multinomial sampling.
    """
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits.float() / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Mask sorted positions whose cumulative probability has already crossed
        # top_p, but keep the first crossing token so at least one nonzero prob remains.
        remove_sorted = cumulative > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(0, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ProbeService:
    """Owns the HF transformers model and a cache of loaded probes."""

    def __init__(self, model_name: str, training_sidecar: dict[str, Any] | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tc = training_sidecar or {}
        self._chat_template_kwargs: dict[str, Any] = dict(tc.get("chat_template_kwargs") or {})

        self.model_name = model_name
        self._probe_cache: dict[str, ValueHeadProbe] = {}  # path → probe
        self._shutting_down = False
        # HF model forward isn't safe to invoke from multiple threads concurrently
        # (shared past_key_values, internal buffers). Serialize generation requests.
        self._gen_lock = threading.Lock()

        # Pre-load probe(s) configured via env so layer IDs are known at startup.
        startup_probe_paths = [
            p.strip()
            for p in os.environ.get("PROBE_PATH", "").split(",")
            if p.strip()
        ]
        for p in startup_probe_paths:
            self._load_probe(p)

        print(
            f"[INIT] Probe layer IDs: {sorted({pr.cfg.layer_idx for pr in self._probe_cache.values()})}",
            flush=True,
        )

        dtype_name = os.environ.get("HF_DTYPE") or tc.get("dtype")
        dtype = _resolve_dtype(dtype_name)
        device_name = os.environ.get("HF_DEVICE")
        self._device = torch.device(device_name) if device_name else _default_device()
        self._max_model_len = int(float(
            os.environ.get("MAX_MODEL_LEN", tc.get("max_model_len", 32768))
        ))
        trust_remote_code_env = os.environ.get("HF_TRUST_REMOTE_CODE")
        if trust_remote_code_env is not None:
            trust_remote_code = trust_remote_code_env.strip().lower() in ("1", "true", "yes")
        else:
            trust_remote_code = bool(tc.get("trust_remote_code", True))

        hf_token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        # Optional LoRA adapter recorded in the training sidecar.
        lora_path = tc.get("lora_path") if tc.get("lora_present") else None
        if lora_path:
            from peft import PeftModel

            print(f"[INIT] Loading LoRA adapter: {lora_path}", flush=True)
            model = PeftModel.from_pretrained(model, str(lora_path))

        self.model = model.to(self._device).eval()

        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos, int):
            self._eos_token_ids: set[int] = {eos}
        elif eos:
            self._eos_token_ids = {int(t) for t in eos}
        else:
            self._eos_token_ids = set()

        print(
            f"[INIT] Ready: {model_name} on {self._device} (dtype={dtype}, max_len={self._max_model_len})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Probe management
    # ------------------------------------------------------------------

    def _load_probe(self, path: str) -> ValueHeadProbe:
        if path not in self._probe_cache:
            print(f"[PROBE] Loading {path}", flush=True)
            self._probe_cache[path] = load_probe(path)
        return self._probe_cache[path]

    @staticmethod
    def _probe_name(path: str) -> str:
        """Derive a human-readable probe name from its checkpoint path.

        Uses the parent directory name when the file is probe_head.bin
        (standard trainer output layout), otherwise the file stem.
        """
        p = Path(path)
        return p.parent.name if p.name == "probe_head.bin" else p.stem

    def get_probes(self, probe_path: str | None) -> list[tuple[str, ValueHeadProbe]]:
        """Return (name, probe) pairs for a request.

        Priority: per-request *probe_path* → PROBE_PATH env var → empty list.
        """
        if probe_path:
            resolved = str(_expand_probe_checkpoint(probe_path))
            return [(self._probe_name(resolved), self._load_probe(probe_path))]
        return [(self._probe_name(path), probe) for path, probe in self._probe_cache.items()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict]) -> str:
        if messages and messages[0].get("role") != "system":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ] + list(messages)
        return apply_chat_template_to_text(
            self.tokenizer,
            messages,
            chat_template_kwargs=self._chat_template_kwargs,
            add_generation_prompt=True,
        )

    @staticmethod
    def _layer_hidden_state(hidden_states: tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
        """HF ``output_hidden_states=True`` returns ``[embeddings, layer_0_out, layer_1_out, ...]``;
        decoder layer ``i`` is at index ``i+1``. Must match :class:`HFHiddenStateExtractor`."""
        return hidden_states[layer_idx + 1]

    # ------------------------------------------------------------------
    # Streaming generation — yields (delta_text, token_index, per_layer_scores)
    # per decode step, then (None, None, None) as the termination sentinel.
    # ------------------------------------------------------------------

    def generate_streaming(
        self,
        messages: list[dict],
        probe_path: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        include_probe_scores: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[tuple[str | None, int | None, list[tuple[int, float]] | None]]:
        """Sync generator.

        Yields ``(delta_text, token_index, per_layer_scores)`` for each token.
        ``per_layer_scores`` is ``[(layer_idx, score), ...]`` — one entry per
        loaded probe. Yields ``(None, None, None)`` once generation is complete.
        """
        probes = self.get_probes(probe_path) if include_probe_scores else []
        prompt_str = self._apply_chat_template(messages)

        # Chat template adds all required special tokens; encode without extras.
        encoded = self.tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(self._device)
        prompt_len = int(input_ids.shape[1])

        budget = max(0, self._max_model_len - prompt_len)
        if max_tokens is None:
            max_tokens = budget
        else:
            max_tokens = max(0, min(int(max_tokens), budget))

        if max_tokens == 0:
            yield None, None, None
            return

        buffers: dict[int, deque | None] = {probe.cfg.layer_idx: None for _, probe in probes}
        prev_text = ""
        generated_ids: list[int] = []

        with self._gen_lock, torch.no_grad():
            # Prefill: don't capture hidden states (the probe scores completion tokens
            # only; matching training's hs[-n_completion_tokens:] convention).
            outputs = self.model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = _sample_token(
                outputs.logits[0, -1], temperature=temperature, top_p=top_p
            )

            if next_token in self._eos_token_ids:
                yield None, None, None
                return

            generated_ids.append(next_token)
            step = 0

            while step < max_tokens:
                if self._shutting_down or (cancel_event and cancel_event.is_set()):
                    break

                # Training scored state AFTER each completion token (hs[-n_completion_tokens:]),
                # so feed the just-sampled token forward and probe its post-token hidden state.
                inp = torch.tensor([[next_token]], dtype=torch.long, device=self._device)
                outputs = self.model(
                    input_ids=inp,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=bool(probes),
                )
                past_key_values = outputs.past_key_values

                per_layer: list[tuple[int, float]] = []
                if probes:
                    for _, probe in probes:
                        layer_hs = self._layer_hidden_state(
                            outputs.hidden_states, probe.cfg.layer_idx
                        )
                        hs_step = (
                            layer_hs[0, -1]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        score, buffers[probe.cfg.layer_idx] = run_probe_step(
                            probe, hs_step, buffers[probe.cfg.layer_idx]
                        )
                        per_layer.append((probe.cfg.layer_idx, score))

                current_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                delta = current_text[len(prev_text):]
                prev_text = current_text
                if delta:
                    yield delta, step, per_layer if per_layer else None

                step += 1
                if step >= max_tokens:
                    break
                sampled = _sample_token(
                    outputs.logits[0, -1], temperature=temperature, top_p=top_p
                )
                if sampled in self._eos_token_ids:
                    break
                next_token = sampled
                generated_ids.append(next_token)

        yield None, None, None

    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._shutting_down = True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: Optional[ProbeService] = None


def get_service() -> ProbeService:
    if _service is None:
        raise RuntimeError("Service not initialised")
    return _service


@asynccontextmanager
async def lifespan(app):
    global _service
    _warn_suspicious_env_vars()
    raw_probe_paths = [
        p.strip()
        for p in os.environ.get("PROBE_PATH", "").split(",")
        if p.strip()
    ]
    model, sidecar = resolve_inference_model_and_config(raw_probe_paths)
    _service = ProbeService(model, training_sidecar=sidecar)
    yield
    if _service is not None:
        _service.shutdown()
