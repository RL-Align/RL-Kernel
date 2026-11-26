# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``final_distributed_h3_run``: the vLLM-Omni H3 rollout runs the actor's H3 model.

vLLM-Omni's ``MiniMaxH3DiTModel`` runs its own boundary modules (patch projections, time
embedder, token refiner, final layer: TP-sharded, with an FP32 all-reduce and fused Triton
kernels) around its blocks, so binding only the blocks (or only the projections, as
``h3_omni.install_h3_rollout`` does) cannot give the actor's bytes. Here the whole DiT call is
replaced instead (contract ``h3-omni-rollout-v1``): ``install_h3_omni_model`` swaps the
transformer's ``forward`` for :class:`H3OmniModelBinding`, which

* converts vLLM-Omni's packed DiT keyword arguments to the Diffusers ``MiniMaxH3Transformer3DModel``
  inputs that verl-omni's actor builds for the same request (``omni_to_diffusers``): drop the
  alignment-padding rows, take the video / audio rows at their positions, and rebuild the row
  timesteps' ``unique`` / ``inverse`` over the real rows. Only data movement, so the actor's
  inputs bit for bit;
* runs the actor's model (the Diffusers-layout H3 of ``h3_sharded_load``, with LoRA from
  ``h3_lora``) under ``h3_parallel_model.parallel_model_execution`` on the rollout's TP grid
  (``Grid(tp, 1)``), the same code as the actor's TP x CP grid, so both equal the single-GPU
  canonical run;
* splits a step-mode batch (``num_requests > 1``) into its requests and runs each alone, so a
  request's velocity does not depend on the others;
* runs under PyTorch's default SDPA backends (``torch_default_sdpa``). Importing vLLM switches
  cuDNN attention off for the whole process (``vllm/platforms/cuda.py``), and without the strict
  boundaries (Hopper only) the token refiner's attention is SDPA, so the engine would otherwise
  pick another kernel than the actor's and the canonical run do;
* returns ``(video, audio)`` as vLLM-Omni does: video rows at ``img_pos_for_infer_output_info``
  times ``update_mask`` (unless ``skip_mask_out_condition``), audio rows times
  ``update_audio_mask`` when given.

The engine's own blocks are never called, which also bypasses their compiled forwards, the AdaLN
cache and block hooks. What would change the result instead of being bypassed is refused: the
reference-KV cache, control hints, PDD steps, FastH3 sparse (VSA) attention, VDN attention, sequence
parallelism, a TP size other than the grid's, cache backends that skip DiT calls, quantization
and an engine LoRA (the LoRA lives on the actor's model: ``h3_lora``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

CONTRACT = "h3-omni-rollout-v1"

# vLLM-Omni 6039984e ``MiniMaxH3DiTModel.forward``'s keyword contract.
KNOWN_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "video_token_layout",
        "rope_table",
        "reference_kv_state",
        "control_rows",
        "control_context_scale",
        "update_audio_mask",
    }
)
PAD_TAG = -1


def _default_sdpa_backends() -> list:
    from torch.nn.attention import SDPBackend

    # torch's own enabled set, in torch's order; vLLM's import drops CUDNN_ATTENTION from it
    return [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
        SDPBackend.OVERRIDEABLE,
    ]


@contextmanager
def torch_default_sdpa() -> Iterator[None]:
    """Enable exactly PyTorch's default SDPA backends inside the ``with`` body, whatever the
    process switched off (vLLM disables cuDNN attention at import); restored on exit."""

    from torch.nn.attention import sdpa_kernel

    with sdpa_kernel(_default_sdpa_backends()):
        yield


def sdpa_backends_enabled() -> dict[str, bool]:
    """Which SDPA backends are enabled right now (for the reports)."""

    cuda = torch.backends.cuda
    return {
        "cudnn": cuda.cudnn_sdp_enabled(),
        "flash": cuda.flash_sdp_enabled(),
        "efficient": cuda.mem_efficient_sdp_enabled(),
        "math": cuda.math_sdp_enabled(),
    }


_ABSENT = object()


def _positions(info: Any, name: str) -> torch.Tensor:
    pos = info.get("position_ids") if isinstance(info, Mapping) else getattr(info, "position_ids")
    if pos is None:
        raise ValueError(f"{name} has no position_ids")
    return pos.view(-1).to(torch.long)


def _field(params: Any, name: str, default: Any = _ABSENT) -> Any:
    value = (
        params.get(name, _ABSENT) if isinstance(params, Mapping) else getattr(params, name, _ABSENT)
    )
    if value is _ABSENT:
        if default is _ABSENT:
            raise ValueError(f"packed_seq_params has no {name}")
        return default
    return value


def check_call(kwargs: Mapping[str, Any], pdd_step: Any = None) -> None:
    """Refuse a DiT call whose result the actor's model would not reproduce."""

    unknown = sorted(set(kwargs) - KNOWN_KWARGS)
    if unknown:
        raise TypeError(f"unknown MiniMax-H3 DiT kwargs {unknown} (validated: vLLM-Omni 6039984e)")
    if pdd_step is not None:
        raise ValueError("PDD steps run distilled output heads the actor does not have")
    if kwargs.get("reference_kv_state") is not None:
        raise ValueError("the reference-KV cache replaces rows of the DiT input; not supported")
    if (
        kwargs.get("control_rows") is not None
        and float(kwargs.get("control_context_scale", 1.0)) != 0
    ):
        raise ValueError("control hints add a ControlNet branch the actor does not have")


def split_requests(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Single-request DiT kwargs for each request of a (step-mode) batch, in request order.

    vLLM-Omni's batched layout concatenates every request's packed rows (real rows, then its
    padding tail) and gives each real span and each nonempty padding tail its own ``cu_seqlens``
    document; padding rows carry token tag -1. A batch of one is returned as it is.
    """

    psp = kwargs["packed_seq_params"]
    num_requests = int(_field(psp, "num_requests", 1))
    if num_requests == 1:
        return [dict(kwargs)]
    tags = kwargs["token_tags"].view(-1)
    bounds = [int(b) for b in _field(psp, "cu_seqlens_q").view(-1).tolist()]
    spans: list[list[int]] = []
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            raise ValueError(f"empty packed document [{start}, {end})")
        pad = bool((tags[start:end] == PAD_TAG).all())
        if pad:
            if not spans or spans[-1][1] != start:
                raise ValueError("a padding document must follow its request's rows")
            spans[-1][1] = end
        else:
            if bool((tags[start:end] == PAD_TAG).any()):
                raise ValueError("a request document holds padding rows")
            spans.append([start, end])
    if len(spans) != num_requests:
        raise ValueError(f"{len(spans)} request documents, num_requests says {num_requests}")
    refiner = [
        int(b)
        for b in _field(kwargs["refiner_packed_seq_params"], "cu_seqlens_q").view(-1).tolist()
    ]
    if len(refiner) != num_requests + 1:
        raise ValueError("the refiner bounds do not give one text span per request")
    img, aud, txt = (
        _positions(kwargs[name], name)
        for name in ("img_pos_info", "audio_pos_info", "text_pos_info")
    )
    out_pos = _positions(kwargs["img_pos_for_infer_output_info"], "img_pos_for_infer_output_info")
    if not torch.equal(out_pos, img):
        raise ValueError("output rows differ from the video rows; not a vLLM-Omni batched layout")
    update = kwargs["update_mask"].view(-1)
    audio_update = kwargs.get("update_audio_mask")
    requests = []
    img_at = aud_at = 0
    for i, (start, end) in enumerate(spans):

        def take(pos: torch.Tensor, at: int) -> tuple[torch.Tensor, int]:
            inside = (pos >= start) & (pos < end)
            n = int(inside.sum())
            if not bool(inside[at : at + n].all()):
                raise ValueError("positions are not grouped by request")
            return pos[at : at + n] - start, at + n

        img_i, img_next = take(img, img_at)
        aud_i, aud_next = take(aud, aud_at)
        txt_i = txt[refiner[i] : refiner[i + 1]] - start
        if bool((txt_i < 0).any()) or bool((txt_i >= end - start).any()):
            raise ValueError("text positions fall outside their request")
        used = end - start
        tail = tags[start:end] == PAD_TAG
        real = used - int(tail.sum())
        cu = [0, real] + ([used] if real != used else [])
        req = dict(kwargs)
        req.update(
            x=kwargs["x"][:, start:end],
            audio_x=kwargs["audio_x"][:, start:end],
            img_position_ids=kwargs["img_position_ids"][:, start:end],
            token_tags=tags[start:end],
            inverse_indices=kwargs["inverse_indices"].view(-1)[start:end],
            update_mask=update[img_at:img_next],
            prompt_embeds=kwargs["prompt_embeds"][refiner[i] : refiner[i + 1]],
            img_pos_info={"position_ids": img_i},
            audio_pos_info={"position_ids": aud_i},
            text_pos_info={"position_ids": txt_i},
            img_pos_for_infer_output_info={"position_ids": img_i},
            packed_seq_params={
                "cu_seqlens_q": torch.tensor(cu, dtype=torch.int32),
                "max_seqlen_q": max(real, used - real),
                "num_requests": 1,
            },
            refiner_packed_seq_params={
                "cu_seqlens_q": torch.tensor([0, refiner[i + 1] - refiner[i]], dtype=torch.int32),
                "max_seqlen_q": refiner[i + 1] - refiner[i],
            },
        )
        if audio_update is not None:
            req["update_audio_mask"] = audio_update.view(-1)[aud_at:aud_next]
        req.pop("rope_table", None)  # a precomputed table covers the whole batch
        requests.append(req)
        img_at, aud_at = img_next, aud_next
    if img_at != img.numel() or aud_at != aud.numel():
        raise ValueError("positions outside every request")
    return requests


def omni_to_diffusers(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The Diffusers ``MiniMaxH3Transformer3DModel`` inputs of one request's DiT kwargs.

    The same inputs verl-omni's actor (``prepare_model_inputs``) builds from what the rollout
    stores: the packed layout up to the real length, the video / audio rows in position order,
    the text embeddings, and ``torch.unique`` of the real rows' timesteps.
    """

    psp = kwargs["packed_seq_params"]
    if int(_field(psp, "num_requests", 1)) != 1:
        raise ValueError("one request at a time: split_requests first")
    x, audio_x = kwargs["x"], kwargs["audio_x"]
    if x.dim() != 3 or x.shape[0] != 1 or audio_x.shape[:2] != x.shape[:2]:
        raise ValueError(
            f"x / audio_x must be [1, S, C], got {list(x.shape)} / {list(audio_x.shape)}"
        )
    seq = x.shape[1]
    tags = kwargs["token_tags"].view(-1)
    bounds = [int(b) for b in _field(psp, "cu_seqlens_q").view(-1).tolist()]
    if bounds[0] != 0 or bounds[-1] != seq or len(bounds) not in (2, 3):
        raise ValueError(
            f"cu_seqlens {bounds} is not one request (plus a padding tail) of {seq} rows"
        )
    used = bounds[1]
    if bool((tags[:used] == PAD_TAG).any()) or not bool((tags[used:] == PAD_TAG).all()):
        raise ValueError("padding rows must be exactly the rows after the first document")
    img = _positions(kwargs["img_pos_info"], "img_pos_info")
    aud = _positions(kwargs["audio_pos_info"], "audio_pos_info")
    txt = _positions(kwargs["text_pos_info"], "text_pos_info")
    out_pos = _positions(kwargs["img_pos_for_infer_output_info"], "img_pos_for_infer_output_info")
    if not torch.equal(out_pos.cpu(), img.cpu()):
        raise ValueError("output rows differ from the video rows; the actor returns the video rows")
    for name, pos in (("video", img), ("audio", aud), ("text", txt)):
        if pos.numel() and int(pos.max()) >= used:
            raise ValueError(f"{name} positions reach the padding rows")
    text = kwargs["prompt_embeds"]
    if text.dim() != 2 or text.shape[0] != txt.numel():
        raise ValueError(f"prompt_embeds {list(text.shape)} does not match {txt.numel()} text rows")
    device = x.device
    inverse = kwargs["inverse_indices"].view(-1).to(torch.long)
    row_t = (
        kwargs["unique_timesteps"]
        .view(-1)
        .index_select(0, inverse[:used].to(kwargs["unique_timesteps"].device))
    )
    timestep, timestep_indices = torch.unique(row_t, sorted=True, return_inverse=True)
    img_d, aud_d, txt_d = img.to(device), aud.to(device), txt.to(device)
    return {
        "hidden_states": x[0].index_select(0, img_d).unsqueeze(0),
        "audio_hidden_states": audio_x[0].index_select(0, aud_d).unsqueeze(0),
        "encoder_hidden_states": text.unsqueeze(0),
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": tags[:used],
        "position_ids": kwargs["img_position_ids"][0, :used],
        "video_indices": img_d,
        "audio_indices": aud_d,
        "text_indices": txt_d,
        "return_dict": False,
    }


def diffusers_to_omni(
    video: torch.Tensor, audio: torch.Tensor, kwargs: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM-Omni's return of one request: rows without the batch dim, condition rows zeroed."""

    video, audio = video.squeeze(0), audio.squeeze(0)
    if not bool(kwargs.get("skip_mask_out_condition", False)):
        update = kwargs["update_mask"].view(-1).to(video.device)
        if update.shape[0] != video.shape[0]:
            raise ValueError(f"update_mask length {update.shape[0]} != {video.shape[0]} video rows")
        video = video * update.unsqueeze(-1)
        audio_update = kwargs.get("update_audio_mask")
        if audio_update is not None:
            audio = audio * audio_update.view(-1).to(audio.device).unsqueeze(-1)
    return video, audio


def tensor_digest(t: torch.Tensor) -> str:
    """SHA-256 of a tensor's dtype, shape and bytes (exact: any bit changes it)."""

    t = t.detach().contiguous().cpu()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def inputs_digest(inputs: Mapping[str, Any]) -> str:
    """One SHA-256 over every tensor input (by name) of a model call."""

    h = hashlib.sha256()
    for name in sorted(inputs):
        value = inputs[name]
        h.update(name.encode())
        h.update(
            tensor_digest(value).encode()
            if isinstance(value, torch.Tensor)
            else repr(value).encode()
        )
    return h.hexdigest()


@dataclass
class H3OmniModelBinding:
    """The DiT call of the rollout, on the actor's model and the rollout's TP grid.

    ``model`` is the Diffusers-layout H3 (``h3_sharded_load.load_sharded_model`` for ``grid``'s
    ``(tp, tp_rank)``), ``grid`` an ``h3_tp.Grid`` with ``cp == 1``. ``record(kwargs, inputs,
    outputs)`` (optional) sees each request's DiT kwargs, Diffusers inputs and outputs; the G3
    report uses it to replay a step on the canonical model.
    """

    model: torch.nn.Module
    grid: Any
    strict_boundaries: bool = True
    ops: Any = None
    record: Callable[[Mapping[str, Any], dict, tuple], None] | None = None
    calls: int = 0
    # per-call input / output digests, for the G3 validation only (each one copies the tensors to
    # the CPU): off for rollout, on with ``record_digests``
    record_digests: bool = False
    digests: list[dict[str, Any]] = field(default_factory=list)
    record_errors: list[str] = field(default_factory=list)  # rank 0's failed captures

    def __post_init__(self) -> None:
        if self.grid.cp != 1:
            raise ValueError("the rollout grid is TP only (cp == 1)")
        if self.grid.attention != "h3_full":
            raise ValueError("the rollout runs the actor's attention: Grid(attention='h3_full')")

    def forward(self, *, pdd_step: Any = None, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        from rl_engine.validation.models import h3_lora
        from rl_engine.validation.models import h3_parallel_model as pm
        from rl_engine.validation.models.h3_full_model import FullModelTrace

        check_call(kwargs, pdd_step)
        h3_lora.require_refiner_lora_supported(self.model, strict_boundaries=self.strict_boundaries)
        requests = split_requests(kwargs)
        videos, audios = [], []
        with (
            torch.no_grad(),
            torch_default_sdpa(),
            pm.parallel_model_execution(
                self.model,
                self.grid,
                sp=False,
                trace=FullModelTrace(),
                ops=self.ops,
                strict_boundaries=self.strict_boundaries,
            ),
        ):
            for request in requests:
                inputs = omni_to_diffusers(request)
                video, audio = self.model(**inputs)
                if self.record is not None:
                    self.record(request, inputs, (video, audio))
                if self.record_digests:
                    self.digests.append(
                        {
                            "call": self.calls,
                            "num_requests": len(requests),
                            "inputs": inputs_digest(inputs),
                            "video": tensor_digest(video),
                            "audio": tensor_digest(audio),
                        }
                    )
                v, a = diffusers_to_omni(video, audio, request)
                videos.append(v)
                audios.append(a)
        self.calls += 1
        if len(requests) == 1:
            return videos[0], audios[0]
        return torch.cat(videos), torch.cat(audios)


def check_engine_config(od_config: Any, *, tp: int) -> list[str]:
    """Why vLLM-Omni's diffusion config ``od_config`` cannot run the bound DiT (empty: it can)."""

    reasons = []
    par = getattr(od_config, "parallel_config", None)
    if par is not None:
        if int(getattr(par, "tensor_parallel_size", 1)) != tp:
            reasons.append(f"tensor_parallel_size {par.tensor_parallel_size} != grid tp {tp}")
        for name in ("ulysses_degree", "ring_degree", "allgather_degree", "cfg_parallel_size"):
            if int(getattr(par, name, 1) or 1) != 1:
                reasons.append(f"{name} {getattr(par, name)} != 1 (the rollout grid is TP only)")
        if int(getattr(par, "pipeline_parallel_size", 1) or 1) != 1:
            reasons.append("pipeline parallelism splits the DiT")
        if int(getattr(par, "data_parallel_size", 1) or 1) != 1:
            # the bound blocks index their rows by the global rank and assemble them over it
            reasons.append(
                f"data_parallel_size {par.data_parallel_size} != 1 (the rows are owned by WORLD "
                "ranks, so the rollout process group must be the TP group alone)"
            )
        if getattr(par, "use_hsdp", False):
            reasons.append("HSDP shards the DiT weights")
    cache = getattr(od_config, "cache_backend", None)
    if cache not in (None, "none"):
        reasons.append(f"cache backend {cache!r} skips or reuses DiT calls")
    if getattr(od_config, "quantization_config", None) is not None:
        reasons.append("quantization changes the DiT weights")
    if getattr(od_config, "lora_path", None):
        reasons.append(
            "an engine LoRA (lora_path) is added by vLLM's LoRA layers, not the bound model"
        )
    if getattr(od_config, "diffusion_compile_granularity", "regional") == "full" and not getattr(
        od_config, "enforce_eager", False
    ):
        reasons.append("full-graph compile traces the DiT forward before it is replaced")
    return reasons


def check_process_world(tp: int) -> list[str]:
    """Why the default process group cannot run the bound DiT (empty: it can).

    The bound blocks index and assemble their rows by global rank, so every process must be in
    the rollout's TP group: WORLD == TP. This also refuses vLLM-Omni data parallelism (more
    processes than TP) on paths that have no engine config to read it from.
    """

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != tp:
        return [
            f"world size {dist.get_world_size()} != TP {tp}: the bound blocks own rows by global "
            "rank"
        ]
    return []


def check_engine_transformer(transformer: torch.nn.Module) -> list[str]:
    """Why vLLM-Omni's DiT module cannot be replaced by the bound DiT (empty: it can)."""

    reasons = []
    if type(transformer).__name__ != "MiniMaxH3DiTModel":
        reasons.append(f"{type(transformer).__name__} is not vLLM-Omni's MiniMaxH3DiTModel")
    if getattr(transformer, "vdn_config", None) is not None:
        reasons.append("VDN-H3 hybrid attention is not the released dense H3")
    if any(name.endswith("to_gate_compress") for name, _ in transformer.named_modules()):
        # FastH3 video sparse attention: vLLM-Omni then selects the attended tiles with the
        # packed layout's vsa_prefix_segments; the actor runs full attention.
        reasons.append("FastH3 sparse attention (to_gate_compress) is not the actor's attention")
    if "forward" in transformer.__dict__:
        reasons.append(
            "its forward is already replaced on the instance (compile, offload or another binding)"
        )
    return reasons


@dataclass
class H3OmniModelHandle:
    transformer: torch.nn.Module
    binding: H3OmniModelBinding
    released: int = 0
    _previous: Any = _ABSENT

    def remove(self) -> None:
        """Give vLLM-Omni its own DiT call back; refused once its weights were released."""
        if self.released:
            raise RuntimeError(
                f"vLLM-Omni's DiT weights were released ({self.released} bytes): its own forward "
                "cannot run again; restart the engine instead"
            )
        if self._previous is _ABSENT:
            self.transformer.__dict__.pop("forward", None)
        else:
            self.transformer.forward = self._previous

    def report(self) -> dict[str, Any]:
        from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
        from rl_engine.validation.models import h3_parallel_model as pm

        grid = self.binding.grid
        return {
            "contract": CONTRACT,
            "parallel_contract": par.CONTRACT,
            "tp": grid.tp,
            "tp_rank": grid.tp_rank,
            "attention": grid.attention,
            "strict_boundaries": self.binding.strict_boundaries,
            "sdpa_backends_outside_the_call": sdpa_backends_enabled(),
            "tp_collective": type(grid.tp_collective).__name__ if grid.tp_collective else None,
            "engine_parameter_bytes_released": self.released,
            "calls": self.binding.calls,
            "lora": bool(getattr(self.binding.model.transformer_blocks[0], pm.LORA_ATTR, None)),
            "record_errors": list(self.binding.record_errors),
        }


def release_engine_weights(transformer: torch.nn.Module) -> int:
    """Move every parameter and buffer of vLLM-Omni's DiT to ``meta`` (it never runs again).

    Returns the bytes freed. Without this a rank holds the engine's DiT next to the bound model.
    """

    freed = 0
    for module in transformer.modules():
        for store in (module._parameters, module._buffers):
            for name, tensor in list(store.items()):
                if tensor is None or tensor.is_meta:
                    continue
                freed += tensor.numel() * tensor.element_size()
                empty = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
                store[name] = (
                    torch.nn.Parameter(empty, requires_grad=False)
                    if isinstance(tensor, torch.nn.Parameter)
                    else empty
                )
    return freed


def disable_engine_dit(transformer: torch.nn.Module, reason: str) -> None:
    """After a failed install that released the weights: every later DiT call raises ``reason``
    instead of running vLLM-Omni's forward on ``meta`` tensors."""

    def forward(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"this engine has no usable DiT ({reason}); restart it")

    transformer.forward = forward


def install_h3_omni_model(
    transformer: torch.nn.Module,
    binding: H3OmniModelBinding,
    *,
    od_config: Any = None,
    release_weights: bool = False,
) -> H3OmniModelHandle:
    """Replace vLLM-Omni's DiT call with ``binding``. Every refusal is raised before any change."""

    reasons = check_engine_transformer(transformer) + check_process_world(binding.grid.tp)
    if od_config is not None:
        reasons += check_engine_config(od_config, tp=binding.grid.tp)
    if reasons:
        raise RuntimeError("cannot bind the H3 rollout DiT: " + "; ".join(reasons))
    handle = H3OmniModelHandle(
        transformer, binding, _previous=transformer.__dict__.get("forward", _ABSENT)
    )
    if release_weights:
        handle.released = release_engine_weights(transformer)
    transformer.forward = binding.forward
    return handle


GRID_COLLECTIVE_BYTES = 256 << 20  # capacity of the rollout grid's DeterministicCollective


def grid_collective_staging_bytes(tp: int) -> int:
    """Device bytes the rollout grid's collective allocates on each rank (0 at TP = 1).

    Its constructor allocates them and then exchanges IPC metadata with every rank, so a rank
    that cannot allocate would leave the others waiting in that exchange; the worker allocates
    this much once in a gathered phase first.
    """

    from rl_engine.distributed.algorithms import collectives as col

    if tp == 1:
        return 0
    return col._COLLECTIVE_STAGING_FRAMES * (
        GRID_COLLECTIVE_BYTES + col._COLLECTIVE_FRAME_METADATA_BYTES
    )


def tp_grid_from_group(group: Any, tp_rank: int, tp: int, device: Any = None):
    """``h3_tp.Grid(tp, 1)`` on an existing TP process group (vLLM's), with #701's collective."""

    from rl_engine.distributed.algorithms.collectives import DeterministicCollective
    from rl_engine.validation.models import h3_tp

    collective = None
    if tp > 1:
        device = torch.cuda.current_device() if device is None else device
        collective = DeterministicCollective(group, device, max_size_bytes=GRID_COLLECTIVE_BYTES)
    return h3_tp.Grid(tp, 1, tp_rank, 0, collective, None, "h3_full", "ring")
