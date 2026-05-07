"""Inference service: vLLM AsyncLLM + probe scoring."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import queue as _queue
import sys
import threading
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from api.schemas import ChatCompletionRequest

import torch
import uvloop

# Probe model classes live in the sibling train package.
_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from models import CovSeqModel, MultiLayerCovSeqModel, ValueHeadProbe, ProbeConfig  # noqa: E402  (train package)

_DONE = object()

_KNOWN_ENV_VARS = {
    "MODEL_NAME",
    "PROBE_PATH",
    "DTYPE",
    "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION",
    "TENSOR_PARALLEL_SIZE",
    "TRUST_REMOTE_CODE",
    "HF_TOKEN",
    "HOST",
    "PORT",
}


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
# Resolve base model + training sidecar (TP, max_len, chat template kwargs)
# ---------------------------------------------------------------------------

def _expand_probe_checkpoint(raw: str) -> Path:
    """If *raw* is a directory, resolve to the probe weights file inside it."""
    p = Path(raw).expanduser()
    if p.is_dir():
        for name in ("probe_head.bin", "model.safetensors"):
            cand = p / name
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

    Training saves :class:`~models.ProbeConfig` with ``underlying_model`` set to the vLLM base model.
    The run directory also contains ``config.json`` with ``model_name``, ``tensor_parallel_size``, etc.
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
            "  • export MODEL_NAME=google/gemma-4-31B-it\n"
            "  • or export PROBE_PATH to your probe checkpoint:\n"
            "      export PROBE_PATH=/path/to/output/run/probe_head.bin\n"
            "    or the training output directory containing probe_head.bin + config.json\n"
            "The checkpoint must include probe_config.underlying_model (from training), "
            "or sit next to config.json with \"model_name\"."
        )

    print(f"[INIT] HF model for vLLM: {model}", flush=True)
    if env_model:
        print("[INIT] (MODEL_NAME env overrides inferred id)", flush=True)
    return model, sidecar


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probe(path: str | Path) -> ValueHeadProbe:
    """Load a ValueHeadProbe checkpoint from *path* onto CPU.

    Supports both ``.bin`` (torch.save with embedded probe_config) and
    ``.safetensors`` (HF Trainer output — requires a ``probe_config.json``
    sidecar in the same directory).
    """
    path = _expand_probe_checkpoint(str(path))
    if not path.is_file():
        raise FileNotFoundError(
            f"Probe weights not found at {path} (expected a file or a directory containing probe_head.bin)"
        )
    if path.suffix == ".safetensors":
        cfg_path = path.parent / "probe_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"No probe_config.json sidecar found at {cfg_path}. "
                "Save one alongside model.safetensors so the architecture can be reconstructed."
            )
        cfg = ProbeConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        probe = ValueHeadProbe(cfg)  # load_from_state_dict called inside via cfg.path=None
        probe.cfg.path = path
        probe.load_from_state_dict()
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "probe_config" in payload:
            cfg = ProbeConfig.from_dict(payload["probe_config"])
        else:
            raise ValueError(f"Checkpoint at {path} has no 'probe_config' key")
        probe = ValueHeadProbe(cfg)
        if isinstance(payload, dict) and "state_dict" in payload:
            probe.model.load_state_dict(payload["state_dict"])
    probe.model.eval()
    return probe


# ---------------------------------------------------------------------------
# Per-step probe runners
# ---------------------------------------------------------------------------

def probe_layer_ids(probe: ValueHeadProbe) -> list[int]:
    """Return the vLLM layer indices this probe needs hidden states from."""
    if isinstance(probe.model, MultiLayerCovSeqModel):
        return list(probe.cfg.model.layer_indices)
    return [probe.cfg.layer_idx]


def _run_mlp_step(probe: ValueHeadProbe, hs: torch.Tensor) -> float:
    with torch.no_grad():
        logit = probe.model(hs)
        return float(torch.sigmoid(logit).squeeze())


def _run_covseq_step(
    probe: ValueHeadProbe,
    hs: torch.Tensor,
    buf: deque,
) -> float:
    # Decode path feeds [1, hidden]; analyze path feeds [1, 1, hidden] from
    # sliced prefill tensors. Reshape to a flat [hidden] so the stacked window
    # is always [1, T, hidden] regardless of caller.
    vec = hs.reshape(-1)
    buf.append(vec)
    # Use only real hidden states — training used truncated windows of length 1..T-1
    # for early tokens, never zero-padded full windows. Zero-padding causes a 1/T
    # scale error in the covariance matrix that produces out-of-distribution scores.
    window = torch.stack(list(buf), dim=0).unsqueeze(0)  # [1, actual_len, d_model]
    with torch.no_grad():
        logit = probe.model(window)
        return float(torch.sigmoid(logit).squeeze())


def _run_multilayer_covseq_step(
    probe: ValueHeadProbe,
    hs_per_layer: dict[int, torch.Tensor],
    bufs: dict[int, deque],
) -> float:
    """One decode step for MultiLayerCovSeqModel.

    *hs_per_layer* maps vLLM layer index → hidden state tensor for this step.
    *bufs* maps vLLM layer index → rolling window deque (mutated in place).
    Returns the sigmoid score for this token.
    """
    window_size = probe.cfg.model.covseq.window_size
    layer_indices = probe.cfg.model.layer_indices

    windows: list[torch.Tensor] = []
    for layer_idx in layer_indices:
        if layer_idx not in bufs:
            bufs[layer_idx] = deque(maxlen=window_size)
        # Reshape to flat [hidden] — handles both [1, hidden] (decode) and
        # [1, 1, hidden] (analyze prefill slice) inputs.
        vec = hs_per_layer[layer_idx].reshape(-1)
        bufs[layer_idx].append(vec)
        window = torch.stack(list(bufs[layer_idx]), dim=0).unsqueeze(0)  # [1, T, d]
        windows.append(window)

    with torch.no_grad():
        logit = probe.model(windows)
        return float(torch.sigmoid(logit).squeeze())


def run_probe_step(
    probe: ValueHeadProbe,
    hs: torch.Tensor | dict[int, torch.Tensor],
    buf: deque | dict[int, deque] | None,
) -> tuple[float, deque | dict[int, deque] | None]:
    """Run one decode step through *probe*. Returns (score, updated_buf).

    For MultiLayerCovSeqModel, *hs* must be a dict mapping layer index → tensor
    and *buf* must be a dict mapping layer index → deque (or None to initialise).
    For single-layer probes, *hs* is a single tensor and *buf* is a single deque or None.
    """
    probe_dtype = next(probe.model.parameters()).dtype
    if isinstance(hs, dict):
        hs = {k: v.to(probe_dtype) for k, v in hs.items()}
    else:
        hs = hs.to(probe_dtype)

    if isinstance(probe.model, MultiLayerCovSeqModel):
        if buf is None:
            buf = {}
        assert isinstance(hs, dict), "MultiLayerCovSeqModel requires hs as dict[int, Tensor]"
        score = _run_multilayer_covseq_step(probe, hs, buf)
    elif isinstance(probe.model, CovSeqModel):
        if buf is None:
            buf = deque(maxlen=probe.cfg.model.covseq.window_size)
        assert isinstance(hs, torch.Tensor)
        score = _run_covseq_step(probe, hs, buf)
    else:
        assert isinstance(hs, torch.Tensor)
        score = _run_mlp_step(probe, hs)
        buf = None
    return score, buf


# ---------------------------------------------------------------------------
# Shared tokenization helpers
# ---------------------------------------------------------------------------

def apply_chat_template_str(
    tokenizer,
    messages: list[dict],
    chat_template_kwargs: dict | None = None,
) -> str:
    """Render *messages* to a prompt string, never to token ids.

    Always passes return_dict=False and tokenize=False so the return type is
    unambiguously str regardless of the transformers version installed.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_dict=False,
        **(chat_template_kwargs or {}),
    )


def apply_chat_template_ids(
    tokenizer,
    messages: list[dict],
    chat_template_kwargs: dict | None = None,
) -> list[int]:
    """Render *messages* directly to a token-id list.

    Always passes return_dict=False and tokenize=True so the return type is
    unambiguously list[int] regardless of the transformers version installed.
    """
    return list(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **(chat_template_kwargs or {}),
    ))


def encode_text(tokenizer, text: str) -> list[int]:
    """Encode *text* to token ids without adding special tokens."""
    return list(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ProbeService:
    """Owns the vLLM AsyncLLM engine and a cache of loaded probes."""

    def __init__(self, model_name: str, training_sidecar: dict[str, Any] | None = None) -> None:
        import vllm
        import vllm_probe_plugin
        from vllm_probe_plugin import _result_bus
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from transformers import AutoTokenizer

        tc = training_sidecar or {}
        self._chat_template_kwargs: dict[str, Any] = dict(tc.get("chat_template_kwargs") or {})

        self.model_name = model_name
        self.tokenizer = None
        self._probe_cache: dict[str, ValueHeadProbe] = {}  # path → probe
        self._shutting_down = False

        # Pre-load probe(s) configured via env so layer IDs are known at startup.
        startup_probe_paths = [
            p.strip()
            for p in os.environ.get("PROBE_PATH", "").split(",")
            if p.strip()
        ]
        for p in startup_probe_paths:
            self._load_probe(p)

        all_layer_ids = sorted({
            lid
            for pr in self._probe_cache.values()
            for lid in probe_layer_ids(pr)
        })
        if not all_layer_ids:
            # No probes configured; capture nothing — plain generation still works.
            all_layer_ids = []

        print(f"[INIT] Probe layer IDs: {all_layer_ids}", flush=True)

        # Plugin env vars must be set before engine start.
        if all_layer_ids:
            os.environ["VLLM_PROBE_LAYER_IDS"] = ",".join(str(i) for i in all_layer_ids)
            os.environ["VLLM_PROBE_INCLUDE_PREFILL"] = "1"
        os.environ.setdefault("VLLM_RINGBUFFER_WARNING_INTERVAL", "180")
        os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

        torch._dynamo.config.suppress_errors = True
        vllm_probe_plugin.register()
        _result_bus.start()

        self._loop = uvloop.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="asyncllm-loop"
        )
        self._loop_thread.start()

        dtype = os.environ.get("DTYPE") or tc.get("dtype") or "auto"
        if not isinstance(dtype, str):
            dtype = str(dtype)
        max_model_len = int(float(os.environ.get("MAX_MODEL_LEN", tc.get("max_model_len", 32768))))
        gpu_mem = float(os.environ.get("GPU_MEMORY_UTILIZATION", tc.get("gpu_memory_utilization", 0.9)))
        tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", tc.get("tensor_parallel_size", 1)))
        trust_remote_code = tc.get("trust_remote_code", True)
        if os.environ.get("TRUST_REMOTE_CODE") is not None:
            trust_remote_code = os.environ["TRUST_REMOTE_CODE"].strip().lower() in (
                "1", "true", "yes",
            )

        chunked = tc.get("enable_chunked_prefill")

        async def _create_llm():
            kwargs: dict[str, Any] = dict(
                model=model_name,
                kv_transfer_config=vllm.config.KVTransferConfig(
                    kv_connector="HiddenStatesConnector",
                    kv_role="kv_producer",
                ),
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_mem,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                enable_prefix_caching=True,
                enforce_eager=True, # I'm sad about this
            )
            if chunked is not None:
                kwargs["enable_chunked_prefill"] = bool(chunked)
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))

        self._llm = asyncio.run_coroutine_threadsafe(_create_llm(), self._loop).result()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=trust_remote_code,
        )
        print(
            f"[INIT] Ready: {model_name} (tensor_parallel_size={tensor_parallel_size})",
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

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _apply_chat_template(self, messages: list[dict]) -> str:
        # Prepend a system prompt if the first message isn't one.
        if messages and messages[0].get("role") != "system":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ] + list(messages)
        return apply_chat_template_str(self.tokenizer, messages, self._chat_template_kwargs)

    # ------------------------------------------------------------------
    # SamplingParams builder
    # ------------------------------------------------------------------

    @staticmethod
    def _to_sampling_params(request: ChatCompletionRequest) -> Any:
        """Translate a ChatCompletionRequest into a vllm.SamplingParams.

        Only passes fields confirmed present in vLLM 0.18/0.19. Standard OpenAI
        fields with no vLLM equivalent (user, store, metadata, stream_options,
        response_format, top_logprobs) are accepted on the schema but not forwarded.
        """
        import vllm

        max_tokens = request.max_completion_tokens or request.max_tokens

        # OpenAI logit_bias keys are stringified token ids; vLLM wants int keys.
        logit_bias: dict[int, float] | None = None
        if request.logit_bias:
            logit_bias = {int(k): v for k, v in request.logit_bias.items()}

        stop = request.stop if isinstance(request.stop, list) else (
            [request.stop] if isinstance(request.stop, str) else None
        )

        return vllm.SamplingParams(
            n=request.n,
            temperature=request.temperature if request.temperature > 0 else 0.0,
            top_p=request.top_p if request.temperature > 0 else 1.0,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            seed=request.seed,
            logprobs=request.top_logprobs if request.top_logprobs is not None else (1 if request.logprobs else None),
            logit_bias=logit_bias,
        )

    # ------------------------------------------------------------------
    # Streaming generation — yields (delta_text, token_index, per_layer_scores)
    # per decode step, then (None, None, None) as the termination sentinel.
    # ------------------------------------------------------------------

    def generate_streaming(
        self,
        messages: list[dict],
        probe_path: str | None = None,
        sampling_params: Any = None,
        include_probe_scores: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[tuple[str | None, int | None, list[tuple[int, float]] | None]]:
        """Sync generator.

        Yields ``(delta_text, token_index, per_layer_scores)`` for each token.
        ``per_layer_scores`` is ``[(layer_idx, score), ...]`` — one entry per
        loaded probe. Yields ``(None, None, None)`` once generation is complete.
        """
        import vllm
        from vllm_probe_plugin import _result_bus

        probes = self.get_probes(probe_path) if include_probe_scores else []
        req_id = f"gen-{uuid.uuid4().hex[:12]}"
        prompt_str = self._apply_chat_template(messages)

        if sampling_params is None:
            sampling_params = vllm.SamplingParams()

        out_q: _queue.Queue = _queue.Queue()

        async def _run():
            prev_text = ""
            step = 0
            # Per-probe rolling buffers (None = uninitialised; dict for multi-layer).
            buffers: dict[str, deque | dict | None] = {path: None for path, _ in probes}
            internal_req_id = None
            try:
                async for output in self._llm.generate(
                    prompt=prompt_str,
                    sampling_params=sampling_params,
                    request_id=req_id,
                ):
                    if self._shutting_down or (cancel_event and cancel_event.is_set()):
                        await self._llm.abort(req_id)
                        return

                    current_text = output.outputs[0].text
                    delta = current_text[len(prev_text):]
                    prev_text = current_text
                    if not delta:
                        continue

                    step += 1
                    token_index = step - 1

                    per_layer: list[tuple[int, float]] = []
                    if probes:
                        internal_req_id, step_hidden = await _result_bus.read_step_async(
                            req_id, step
                        )
                        for path, probe in probes:
                            if isinstance(probe.model, MultiLayerCovSeqModel):
                                layer_ids = probe.cfg.model.layer_indices
                                if any(step_hidden.get(lid) is None for lid in layer_ids):
                                    continue
                                hs_dict = {lid: step_hidden[lid] for lid in layer_ids}
                                score, buffers[path] = run_probe_step(probe, hs_dict, buffers[path])
                            else:
                                hs = step_hidden.get(probe.cfg.layer_idx)
                                if hs is None:
                                    continue
                                score, buffers[path] = run_probe_step(probe, hs, buffers[path])
                            per_layer.append((probe.cfg.layer_idx, score))

                    out_q.put((delta, token_index, per_layer if per_layer else None))

                if internal_req_id:
                    _result_bus.clear(internal_req_id)
                    _result_bus.clear_prefill(internal_req_id)
            except Exception as exc:
                import traceback
                print(f"[GEN:{req_id}] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
                out_q.put(exc)
            finally:
                out_q.put(_DONE)

        asyncio.run_coroutine_threadsafe(_run(), self._loop)

        for item in iter(out_q.get, _DONE):
            if isinstance(item, Exception):
                raise item
            yield item
        yield None, None, None  # completion sentinel

    # ------------------------------------------------------------------
    # Closed-source streaming — same (delta, token_index, per_layer_scores) yield
    # contract as generate_streaming, but tokens come from an external model and are
    # forced through the local vLLM model to extract probe scores.
    # ------------------------------------------------------------------

    def generate_closed_source_streaming(
        self,
        messages: list[dict],
        closed_source_model: str,
        probe_path: str | None = None,
        sampling_params: Any = None,
        include_probe_scores: bool = True,
        cancel_event: threading.Event | None = None,
        block_size: int = 1,
    ) -> Iterator[tuple[str | None, int | None, list[tuple[int, float]] | None]]:
        """Sync generator with the same yield contract as generate_streaming.

        Tokens come from *closed_source_model* via OpenRouter or the Anthropic API.
        Each token is forced through the local vLLM model (allowed_token_ids=[token_id])
        to extract a hidden-state probe score. Yields (None, None, None) as sentinel.
        """
        import os as _os
        from closed_source_probe import (
            generate_closed_source_with_probe_streaming,
            is_anthropic_model,
        )

        openrouter_api_key = _os.environ.get("OPENROUTER_API_KEY", "")
        openrouter_base_url = _os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        anthropic_api_key = _os.environ.get("ANTHROPIC_API_KEY", "")

        if is_anthropic_model(closed_source_model):
            if not anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
        elif not openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        probes = self.get_probes(probe_path) if include_probe_scores else []

        # Build a run_probe_step_fn that closed_source_probe expects: (step_hidden) -> float.
        if probes:
            _probe_name, _probe = probes[0]
            _buf_ref: list[Any] = [None]

            def _run_probe(step_hidden: dict) -> float:
                if isinstance(_probe.model, MultiLayerCovSeqModel):
                    layer_ids = _probe.cfg.model.layer_indices
                    if any(step_hidden.get(lid) is None for lid in layer_ids):
                        return 0.0
                    hs: Any = {lid: step_hidden[lid] for lid in layer_ids}
                else:
                    hs = step_hidden.get(_probe.cfg.layer_idx)
                    if hs is None:
                        return 0.0
                score, _buf_ref[0] = run_probe_step(_probe, hs, _buf_ref[0])
                return score
        else:
            def _run_probe(step_hidden: dict) -> float:  # type: ignore[misc]
                return 0.0

        out_q: _queue.Queue = _queue.Queue()

        def _token_sink(item: dict) -> None:
            out_q.put(("token", item["token"], item["index"]))

        async def _run():
            try:
                async for event in generate_closed_source_with_probe_streaming(
                    messages=messages,
                    llm=self._llm,
                    tokenizer=self.tokenizer,
                    run_probe_step_fn=_run_probe,
                    closed_source_model=closed_source_model,
                    openrouter_api_key=openrouter_api_key,
                    openrouter_base_url=openrouter_base_url,
                    anthropic_api_key=anthropic_api_key,
                    max_tokens=sampling_params.max_tokens if sampling_params else None,
                    temperature=sampling_params.temperature if sampling_params else 1.0,
                    block_size=block_size,
                    token_sink=_token_sink,
                ):
                    if self._shutting_down or (cancel_event and cancel_event.is_set()):
                        return
                    if event.get("probe_batch"):
                        out_q.put(("probe_batch", event["probes"]))
                    elif event.get("done") or event.get("error"):
                        pass  # sentinel from _DONE handles termination
            except Exception as exc:
                import traceback
                print(f"[CS_GEN] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
                out_q.put(exc)
            finally:
                out_q.put(_DONE)

        asyncio.run_coroutine_threadsafe(_run(), self._loop)

        # token events arrive before their probe_batch; buffer scores and re-attach.
        # Since tokens always precede their batch, we emit tokens immediately with
        # score=None, then back-fill when the batch arrives — but the OpenAI streaming
        # protocol doesn't support back-filling. Instead we hold each token until its
        # score is known by consuming from the queue sequentially:
        #   token → held in pending
        #   probe_batch → resolves held tokens, emit all at once
        pending: list[tuple[str, int]] = []  # (delta, token_index) awaiting score
        buffered_scores: dict[int, float] = {}

        for msg in iter(out_q.get, _DONE):
            if isinstance(msg, Exception):
                raise msg
            kind = msg[0]
            if kind == "token":
                _, delta, idx = msg
                pending.append((delta, idx))
            elif kind == "probe_batch":
                _, entries = msg
                for entry in entries:
                    buffered_scores[entry["index"]] = entry["probe_prob"]
                # Flush pending tokens whose scores have arrived.
                remaining: list[tuple[str, int]] = []
                for delta, idx in pending:
                    score = buffered_scores.pop(idx, None)
                    if score is not None or not include_probe_scores:
                        per_layer: list[tuple[int, float]] | None = None
                        if score is not None and probes:
                            per_layer = [(probes[0][1].cfg.layer_idx, score)]
                        yield delta, idx, per_layer
                    else:
                        remaining.append((delta, idx))
                pending = remaining

        # Flush any remaining tokens (no probe scores — e.g. probes disabled).
        for delta, idx in pending:
            yield delta, idx, None

        yield None, None, None  # completion sentinel

    # ------------------------------------------------------------------
    # Analyze — prefill pass only, no generation.
    # Takes pre-existing text (from the last assistant message), runs it
    # through the local model as a single prefill request, and streams back
    # per-token probe scores.  Yields (token_text, token_index, per_layer_scores)
    # then (None, None, None) as the termination sentinel — same contract as
    # generate_streaming so main.py can treat them identically.
    # ------------------------------------------------------------------

    def analyze_streaming(
        self,
        text: str,
        user_prompt: str | None = None,
        probe_path: str | None = None,
        batch_size: int = 64,
    ) -> Iterator[tuple[str | None, int | None, list[tuple[int, float]] | None]]:
        """Sync generator: prefill *text* through the local model and stream probe scores.

        Thin wrapper around :func:`analyze_probe.analyze_streaming`. Delegates all
        prefill and scoring logic there; this method only bridges the async generator
        to a synchronous iterator via the shared event loop and a queue.

        Yields ``(token_text, token_index, [(layer_idx, score), ...])`` for each token
        in *text*, then ``(None, None, None)`` as the termination sentinel.
        """
        import analyze_probe

        probes = self.get_probes(probe_path)
        if not probes:
            raise RuntimeError(
                "No probes loaded — PROBE_PATH must be set for analyze mode"
            )

        out_q: _queue.Queue = _queue.Queue()

        async def _run():
            try:
                async for item in analyze_probe.analyze_streaming(
                    text=text,
                    llm=self._llm,
                    tokenizer=self.tokenizer,
                    probes=probes,
                    chat_template_kwargs=self._chat_template_kwargs,
                    user_prompt=user_prompt,
                    batch_size=batch_size,
                ):
                    out_q.put(item)
            except Exception as exc:
                import traceback
                print(f"[ANALYZE] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
                out_q.put(exc)
            finally:
                out_q.put(_DONE)

        asyncio.run_coroutine_threadsafe(_run(), self._loop)

        for item in iter(out_q.get, _DONE):
            if isinstance(item, Exception):
                raise item
            yield item


    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._shutting_down = True
        if self._llm is not None:
            try:
                self._llm.shutdown()
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)


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
