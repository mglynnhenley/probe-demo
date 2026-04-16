"""Inference service: vLLM AsyncLLM + probe scoring."""

from __future__ import annotations

import asyncio
import os
import queue as _queue
import sys
import threading
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import uvloop

# Probe model classes live in the sibling train package.
_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from models import CovSeqModel, ValueHeadProbe  # noqa: E402  (added to path above)
from config import ProbeConfig  # noqa: E402

_DONE = object()


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probe(path: str | Path) -> ValueHeadProbe:
    """Load a ValueHeadProbe checkpoint from *path* onto CPU."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
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

def _run_mlp_step(probe: ValueHeadProbe, hs: torch.Tensor) -> float:
    with torch.no_grad():
        logit = probe.model(hs.to(torch.float32))
        return float(torch.sigmoid(logit).squeeze())


def _run_covseq_step(
    probe: ValueHeadProbe,
    hs: torch.Tensor,
    buf: deque,
) -> float:
    T = probe.cfg.model.covseq.window_size
    vec = hs.squeeze(0).to(torch.float32)
    buf.append(vec)
    d = vec.shape[0]
    n_pad = T - len(buf)
    if n_pad > 0:
        pad = [torch.zeros(d, dtype=torch.float32)] * n_pad
        window = torch.stack(pad + list(buf), dim=0).unsqueeze(0)
    else:
        window = torch.stack(list(buf), dim=0).unsqueeze(0)
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
# Service
# ---------------------------------------------------------------------------

class ProbeService:
    """Owns the vLLM AsyncLLM engine and a cache of loaded probes."""

    def __init__(self, model_name: str) -> None:
        import vllm
        import vllm_probe_plugin
        from vllm_probe_plugin import _result_bus
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from transformers import AutoTokenizer

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

        probe_layer_ids = sorted({pr.cfg.layer_idx for pr in self._probe_cache.values()})
        if not probe_layer_ids:
            # No probes configured; capture nothing — plain generation still works.
            probe_layer_ids = []

        print(f"[INIT] Probe layer IDs: {probe_layer_ids}", flush=True)

        # Plugin env vars must be set before engine start.
        if probe_layer_ids:
            os.environ["VLLM_PROBE_LAYER_IDS"] = ",".join(str(i) for i in probe_layer_ids)
            os.environ["VLLM_PROBE_INCLUDE_PREFILL"] = "1"
        os.environ.setdefault("VLLM_RINGBUFFER_WARNING_INTERVAL", "180")
        os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

        vllm_probe_plugin.register()
        _result_bus.start()

        self._loop = uvloop.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="asyncllm-loop"
        )
        self._loop_thread.start()

        dtype = os.environ.get("VLLM_DTYPE", "auto")
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))
        gpu_mem = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))

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
                enable_prefix_caching=True,
            )
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))

        self._llm = asyncio.run_coroutine_threadsafe(_create_llm(), self._loop).result()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=os.environ.get("HF_TOKEN")
        )
        print(f"[INIT] Ready: {model_name}", flush=True)

    # ------------------------------------------------------------------
    # Probe management
    # ------------------------------------------------------------------

    def _load_probe(self, path: str) -> ValueHeadProbe:
        if path not in self._probe_cache:
            print(f"[PROBE] Loading {path}", flush=True)
            self._probe_cache[path] = load_probe(path)
        return self._probe_cache[path]

    def get_probes(self, probe_path: str | None) -> list[ValueHeadProbe]:
        """Return the list of probes to use for a request.

        Priority: per-request *probe_path* → PROBE_PATH env var → empty list.
        """
        if probe_path:
            return [self._load_probe(probe_path)]
        return list(self._probe_cache.values())

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
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
        import vllm
        from vllm_probe_plugin import _result_bus

        probes = self.get_probes(probe_path) if include_probe_scores else []
        req_id = f"gen-{uuid.uuid4().hex[:12]}"
        prompt_str = self._apply_chat_template(messages)

        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            top_p=top_p if temperature > 0 else 1.0,
            max_tokens=max_tokens,
        )

        out_q: _queue.Queue = _queue.Queue()

        async def _run():
            prev_text = ""
            step = 0
            # Per-probe rolling buffers for CovSeqModel (keyed by probe path).
            buffers: dict[str, deque | None] = {pr.cfg.layer_idx: None for pr in probes}
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
                        for probe in probes:
                            hs = step_hidden.get(probe.cfg.layer_idx)
                            if hs is None:
                                continue
                            score, buffers[probe.cfg.layer_idx] = run_probe_step(
                                probe, hs, buffers[probe.cfg.layer_idx]
                            )
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
    model = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    _service = ProbeService(model)
    yield
    if _service is not None:
        _service.shutdown()
