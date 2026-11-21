# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM-Omni worker extension that binds the H3 rollout DiT (``h3_omni_model``) in each worker.

Pass ``worker_extension_cls="rl_engine.integrations.engines.rollout.vllm.h3_omni_worker.
H3OmniWorkerExtension"`` to the diffusion engine; its methods then run on every TP worker through
``collective_rpc``:

* ``rlk_h3_install(transformer_dir, ...)`` loads this rank's shards of the Diffusers-layout H3
  (``h3_sharded_load``) and replaces the pipeline's DiT call (``install_h3_omni_model``);
* ``rlk_h3_attach_seeded_lora(...)`` / ``rlk_h3_load_lora(...)`` put a LoRA on the bound model
  (a seeded one for tests, or the actor's shards with their digests, all ranks or none) and
  return the digests; ``rlk_h3_lora_digests()`` reads them back;
* ``rlk_h3_report()`` returns the readback (grid, collective, calls, digests of every DiT output).

vLLM-Omni's diffusion ``collective_rpc`` runs a method on every rank but returns rank 0's value
only, so each method here gathers every TP rank's result (or error) over the TP group's CPU group
first: the reply is the list of all ranks' results, and an error on any rank raises on every rank
(after the gather, so no rank waits on one that failed).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import torch

_F = TypeVar("_F", bound=Callable[..., Any])


def _run(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "result": fn()}
    except Exception as e:  # reported by every rank through _gather
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _gather(mine: dict[str, Any]) -> tuple[list[Any], dict[int, str]]:
    """Every TP rank's ``_run`` outcome, in rank order, and the failed ranks' errors."""

    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    replies: list[Any] = [None] * group.world_size
    dist.all_gather_object(replies, mine, group=group.cpu_group)
    failed = {rank: r["error"] for rank, r in enumerate(replies) if not r["ok"]}
    return [r.get("result") for r in replies], failed


def _all_ranks(method: _F) -> _F:
    """Run ``method`` on this rank, then return every TP rank's result, in rank order."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        results, failed = _gather(_run(lambda: method(self, *args, **kwargs)))
        if failed:
            raise RuntimeError(f"{method.__name__} failed on TP ranks {failed}")
        return results

    return wrapper  # type: ignore[return-value]


class H3OmniWorkerExtension:
    """Mixed into vLLM-Omni's diffusion worker: ``self.model_runner.pipeline`` is the pipeline."""

    _rlk_h3: Any = None

    def _rlk_pipeline(self):
        runner = getattr(self, "model_runner", None)
        pipeline = getattr(runner, "pipeline", None)
        if pipeline is None or getattr(pipeline, "transformer", None) is None:
            raise RuntimeError("no loaded diffusion pipeline with a transformer on this worker")
        return pipeline

    @_all_ranks
    def rlk_h3_install(
        self,
        transformer_dir: str,
        *,
        strict_boundaries: bool = True,
        release_weights: bool = False,
        record_dir: str | None = None,
        record_calls: tuple[int, ...] = (0,),
    ) -> dict[str, Any]:
        from vllm.distributed.parallel_state import get_tp_group

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om
        from rl_engine.validation.models import h3_sharded_load as sl

        if self._rlk_h3 is not None:
            raise RuntimeError("the H3 rollout DiT is already bound on this worker")
        pipeline = self._rlk_pipeline()
        od_config = getattr(self, "od_config", None) or getattr(pipeline, "od_config", None)
        tp_group = get_tp_group()
        tp, tp_rank = tp_group.world_size, tp_group.rank_in_group
        grid = om.tp_grid_from_group(tp_group.device_group, tp_rank, tp)
        # Refuse before loading anything.
        reasons = om.check_engine_transformer(pipeline.transformer)
        if od_config is not None:
            reasons += om.check_engine_config(od_config, tp=tp)
        if reasons:
            raise RuntimeError("cannot bind the H3 rollout DiT: " + "; ".join(reasons))
        # The engine's DiT never runs again once bound; free it before loading the actor's model
        # (a rank may not hold both). A failed load then leaves the engine without a DiT.
        released = om.release_engine_weights(pipeline.transformer) if release_weights else 0
        torch.cuda.empty_cache()
        model = sl.load_sharded_model(transformer_dir, tp, tp_rank, "cuda").eval()
        for p in model.parameters():
            p.requires_grad_(False)
        record: Any = None
        if record_dir is not None and tp_rank == 0:
            out = Path(record_dir)
            out.mkdir(parents=True, exist_ok=True)
            wanted = set(record_calls)
            seen = {"n": 0}

            def _record(kwargs, inputs, outputs):
                n = seen["n"]
                seen["n"] += 1
                if n not in wanted:
                    return
                cpu = {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()
                }
                torch.save(
                    {
                        "omni_kwargs": _to_cpu(kwargs),
                        "inputs": cpu,
                        "video": outputs[0].detach().cpu(),
                        "audio": outputs[1].detach().cpu(),
                    },
                    out / f"call{n}.pt",
                )

        if record_dir is not None and tp_rank == 0:
            record = _record
        binding = om.H3OmniModelBinding(
            model, grid, strict_boundaries=strict_boundaries, record=record
        )
        handle = om.install_h3_omni_model(pipeline.transformer, binding, od_config=od_config)
        handle.released = released
        torch.cuda.empty_cache()
        self._rlk_h3 = handle
        return {**handle.report(), "resident_param_bytes": _resident(model)}

    @_all_ranks
    def rlk_h3_attach_seeded_lora(
        self, *, rank: int = 64, alpha: float = 128.0, seed: int = 0, b_zero: bool = False
    ) -> dict[str, str]:
        from rl_engine.validation.models import h3_lora

        handle = self._rlk_handle()
        grid = handle.binding.grid
        h3_lora.attach_lora(
            handle.binding.model,
            grid.tp,
            grid.tp_rank,
            rank=rank,
            alpha=alpha,
            seed=seed,
            b_zero=b_zero,
        )
        return h3_lora.lora_digests(handle.binding.model)

    def rlk_h3_load_lora(self, source: str, *, scale: float) -> list[dict[str, str]]:
        """Install the LoRA shards the actor's TP rank ``r`` saved for rollout rank ``r``.

        ``source`` is a path with ``{rank}``; that file holds ``{"shards": [...], "digests":
        {...}}`` (``h3_lora.shard_block_lora`` per block and the sender's ``lora_digests``). All
        ranks or none: if any rank's shards are refused (shape or digest), every rank keeps its
        previous LoRA and the call raises.
        """

        from rl_engine.validation.models import h3_lora
        from rl_engine.validation.models import h3_parallel_model as pm

        handle = self._rlk_handle()
        grid = handle.binding.grid
        blocks = list(handle.binding.model.transformer_blocks)
        previous = [block.__dict__.get(pm.LORA_ATTR) for block in blocks]

        first = next(iter(handle.binding.model.parameters()), None)
        device = first.device if first is not None else torch.device("cuda")

        def load() -> dict[str, str]:
            sent = torch.load(source.format(rank=grid.tp_rank), weights_only=True)
            return h3_lora.install_lora_shards(
                handle.binding.model,
                sent["shards"],
                grid.tp,
                grid.tp_rank,
                scale=scale,
                expected=sent["digests"],
                device=device,
            )

        results, failed = _gather(_run(load))
        if failed:
            for block, old in zip(blocks, previous):
                if old is None:
                    block.__dict__.pop(pm.LORA_ATTR, None)
                else:
                    setattr(block, pm.LORA_ATTR, old)
            raise RuntimeError(f"rlk_h3_load_lora refused on TP ranks {failed}; no rank changed")
        return results

    @_all_ranks
    def rlk_h3_record_native(self, record_dir: str, record_calls: tuple[int, ...] = (0,)) -> int:
        """Validation only, on an unbound engine: rank 0 saves the kwargs and outputs of vLLM-Omni's
        own DiT for calls ``record_calls``, to compare with the binding (``native`` stage)."""

        from vllm.distributed.parallel_state import get_tp_group

        if self._rlk_h3 is not None:
            raise RuntimeError(
                "the H3 rollout DiT is bound; record the native DiT on its own engine"
            )
        transformer = self._rlk_pipeline().transformer
        if get_tp_group().rank_in_group != 0:
            return 0
        out = Path(record_dir)
        out.mkdir(parents=True, exist_ok=True)
        native = transformer.forward
        seen = {"n": 0}

        def forward(*args, **kwargs):
            video, audio = native(*args, **kwargs)
            n = seen["n"]
            seen["n"] += 1
            if n in record_calls:
                torch.save(
                    {
                        "omni_kwargs": _to_cpu(kwargs),
                        "video": video.detach().cpu(),
                        "audio": audio.detach().cpu(),
                    },
                    out / f"call{n}.pt",
                )
            return video, audio

        transformer.forward = forward
        return len(record_calls)

    @_all_ranks
    def rlk_h3_lora_digests(self) -> dict[str, str]:
        from rl_engine.validation.models import h3_lora

        return h3_lora.lora_digests(self._rlk_handle().binding.model)

    @_all_ranks
    def rlk_h3_report(self) -> dict[str, Any]:
        handle = self._rlk_handle()
        return {**handle.report(), "digests": list(handle.binding.digests)}

    @_all_ranks
    def rlk_h3_remove(self) -> None:
        handle = self._rlk_handle()
        handle.remove()
        self._rlk_h3 = None

    def _rlk_handle(self):
        if self._rlk_h3 is None:
            raise RuntimeError("the H3 rollout DiT is not bound on this worker (rlk_h3_install)")
        return self._rlk_h3


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    return repr(value)


def _resident(model: torch.nn.Module) -> int:
    from rl_engine.validation.models import h3_parallel_model as pm

    held = sum(p.numel() * p.element_size() for p in model.parameters() if not p.is_meta) + sum(
        b.numel() * b.element_size() for b in model.buffers() if not b.is_meta
    )
    for block in model.transformer_blocks:
        for t in (getattr(block, pm.SHARDS_ATTR, None) or {}).get("tensors", {}).values():
            held += t.numel() * t.element_size()
        lora = getattr(block, pm.LORA_ATTR, None)
        for parts in (lora or {}).get("tensors", {}).values():
            held += sum(t.numel() * t.element_size() for t in parts.values())
    return held
