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

    def rlk_h3_install(
        self,
        transformer_dir: str,
        *,
        strict_boundaries: bool = True,
        release_weights: bool = False,
        record_dir: str | None = None,
        record_calls: tuple[int, ...] = (0,),
        record_digests: bool = False,
    ) -> list[dict[str, Any]]:
        """Bind the H3 rollout DiT on every TP rank, or on none.

        ``record_digests`` keeps every DiT call's input and output digests for ``rlk_h3_report``
        (the G3 validation); rollout leaves it off, as each digest copies the tensors to the CPU.

        Three phases, each gathered over the TP group before the next starts (so no rank waits on
        one that failed): (1) checks, a checkpoint preflight and rank 0's record directory: a
        refusal changes nothing; (2) release of vLLM-Omni's DiT (``release_weights``) and the
        load of this rank's shards: if a rank fails after a release, every rank's DiT is disabled
        (``disable_engine_dit``) and the engine must be restarted; without a release nothing
        changed and a retry is safe; (3) the binding, then request LoRAs are refused (the bound
        model runs the actor's LoRA only).
        """

        from vllm.distributed.parallel_state import get_tp_group

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om
        from rl_engine.validation.models import h3_sharded_load as sl

        tp_group = get_tp_group()
        tp, tp_rank = tp_group.world_size, tp_group.rank_in_group

        def check() -> None:
            if self._rlk_h3 is not None:
                raise RuntimeError("the H3 rollout DiT is already bound on this worker")
            pipeline = self._rlk_pipeline()
            od_config = getattr(self, "od_config", None) or getattr(pipeline, "od_config", None)
            reasons = om.check_engine_transformer(pipeline.transformer)
            if od_config is not None:
                reasons += om.check_engine_config(od_config, tp=tp)
            reasons += om.check_process_world(tp)
            if reasons:
                raise RuntimeError("cannot bind the H3 rollout DiT: " + "; ".join(reasons))
            _preflight_checkpoint(transformer_dir)
            if record_dir is not None and tp_rank == 0:  # here, so a failure is gathered
                Path(record_dir).mkdir(parents=True, exist_ok=True)

        _, failed = _gather(_run(check))
        if failed:
            raise RuntimeError(f"rlk_h3_install refused on TP ranks {failed}; no rank changed")

        pipeline = self._rlk_pipeline()
        od_config = getattr(self, "od_config", None) or getattr(pipeline, "od_config", None)
        loaded: dict[str, Any] = {"released": 0}

        def load() -> None:
            # The engine's DiT never runs again once bound; free it before loading the actor's
            # model (a rank may not hold both).
            if release_weights:
                loaded["released"] = om.release_engine_weights(pipeline.transformer)
            torch.cuda.empty_cache()
            model = sl.load_sharded_model(transformer_dir, tp, tp_rank, "cuda").eval()
            for p in model.parameters():
                p.requires_grad_(False)
            loaded["model"] = model
            # the grid's collective allocates this, then exchanges with every rank: a rank that
            # cannot allocate must fail here, in a gathered phase, not leave the others waiting
            probe = om.grid_collective_staging_bytes(tp)
            if probe and torch.cuda.is_available():
                torch.empty(probe, dtype=torch.uint8, device="cuda")  # allocated and freed at once

        _, failed = _gather(_run(load))
        if failed:
            loaded.pop("model", None)
            torch.cuda.empty_cache()
            if release_weights:
                om.disable_engine_dit(
                    pipeline.transformer, f"rlk_h3_install failed after the release: {failed}"
                )
                raise RuntimeError(
                    f"rlk_h3_install failed on TP ranks {failed} after vLLM-Omni's DiT was "
                    "released; every rank's DiT is disabled, restart the engine"
                )
            raise RuntimeError(f"rlk_h3_install failed on TP ranks {failed}; no rank changed")

        model = loaded["model"]
        record_errors: list[str] = []
        record: Callable[..., None] | None = None
        if record_dir is not None and tp_rank == 0:
            out = Path(record_dir)  # created in the gathered checks
            wanted = set(record_calls)
            seen = {"n": 0}

            def _record(kwargs, inputs, outputs):
                n = seen["n"]
                seen["n"] += 1
                if n not in wanted:
                    return
                # a failed copy or save must not stop rank 0 between two requests' collectives
                try:
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
                except Exception as e:  # reported by rlk_h3_report
                    record_errors.append(f"call{n}: {type(e).__name__}: {e}")

            record = _record

        grids: dict[str, Any] = {}

        def make_grid() -> None:
            grids["grid"] = om.tp_grid_from_group(tp_group.device_group, tp_rank, tp)

        def bind() -> dict[str, Any]:
            binding = om.H3OmniModelBinding(
                model,
                grids["grid"],
                strict_boundaries=strict_boundaries,
                record=record,
                record_digests=record_digests,
            )
            binding.record_errors = record_errors
            handle = om.install_h3_omni_model(pipeline.transformer, binding, od_config=od_config)
            handle.released = loaded["released"]
            self._rlk_h3 = handle
            _refuse_request_lora(self)
            torch.cuda.empty_cache()
            return {**handle.report(), "resident_param_bytes": _resident(model)}

        # the grid (and its collective) is built in a gathered phase too: a rank that fails there
        # is reported on every rank, which then all take the failed-bind path below
        results: list[Any] = []
        _, failed = _gather(_run(make_grid))
        if not failed:
            results, failed = _gather(_run(bind))
        if failed:
            handle, self._rlk_h3 = self._rlk_h3, None
            _restore_request_lora(self)
            if loaded["released"]:
                om.disable_engine_dit(pipeline.transformer, f"rlk_h3_install failed: {failed}")
                raise RuntimeError(
                    f"rlk_h3_install failed to bind on TP ranks {failed} after vLLM-Omni's DiT "
                    "was released; every rank's DiT is disabled, restart the engine"
                )
            if handle is not None:
                handle.remove()
            raise RuntimeError(
                f"rlk_h3_install failed to bind on TP ranks {failed}; every rank keeps "
                "vLLM-Omni's DiT, the install can be retried"
            )
        return results

    @_all_ranks
    def rlk_h3_attach_seeded_lora(
        self,
        *,
        rank: int = 64,
        alpha: float = 128.0,
        seed: int = 0,
        b_zero: bool = False,
        refiner: bool = False,
    ) -> dict[str, str]:
        from rl_engine.validation.models import h3_lora

        handle = self._rlk_handle()
        grid = handle.binding.grid
        if refiner:
            h3_lora.require_provider_refiner(strict_boundaries=handle.binding.strict_boundaries)
        h3_lora.attach_lora(
            handle.binding.model,
            grid.tp,
            grid.tp_rank,
            rank=rank,
            alpha=alpha,
            seed=seed,
            b_zero=b_zero,
            refiner=refiner,
        )
        return h3_lora.lora_digests(handle.binding.model)

    def rlk_h3_load_lora(self, source: str, *, scale: float) -> list[dict[str, str]]:
        """Install the LoRA shards the actor's TP rank ``r`` saved for rollout rank ``r``.

        ``source`` is a path with ``{rank}``; that file holds ``{"shards": [...], "digests":
        {...}, "scale": s}`` (``h3_lora.shard_block_lora`` per block, the sender's
        ``lora_digests`` and the scale ``alpha / r`` it trained with), and ``"refiner"`` (a whole
        LoRA per token-refiner block) when the actor trains one. ``scale`` must equal the file's:
        the digests cover the tensors only, so a different scale would pass them and still compute
        another LoRA. A refiner LoRA is refused on a strict-boundary binding, whose refiner row
        would ignore it. All ranks or none: if any rank's shards are refused (scale, refiner, shape
        or digest), every rank keeps its previous LoRA and the call raises.
        """

        from rl_engine.validation.models import h3_lora
        from rl_engine.validation.models import h3_parallel_model as pm

        handle = self._rlk_handle()
        grid = handle.binding.grid
        blocks = list(handle.binding.model.transformer_blocks)
        previous = [block.__dict__.get(pm.LORA_ATTR) for block in blocks]
        refiners = h3_lora.refiner_blocks(handle.binding.model)
        previous_refiner = [block.__dict__.get(h3_lora.REFINER_ATTR) for block in refiners]

        first = next(iter(handle.binding.model.parameters()), None)
        device = first.device if first is not None else torch.device("cuda")

        def load() -> dict[str, str]:
            sent = torch.load(source.format(rank=grid.tp_rank), weights_only=True)
            expected = sent.get("digests") if isinstance(sent, dict) else None
            if not isinstance(expected, dict) or not expected:
                raise ValueError("the LoRA file holds no sender digests: it cannot be verified")
            sent_scale = sent.get("scale")
            if sent_scale is None or float(sent_scale) != float(scale):
                raise ValueError(
                    f"the LoRA file's scale {sent_scale} differs from the scale to apply {scale}"
                )
            if sent.get("refiner"):  # the binding's forward would refuse it on the first request
                h3_lora.require_provider_refiner(strict_boundaries=handle.binding.strict_boundaries)
            return h3_lora.install_lora_shards(
                handle.binding.model,
                sent["shards"],
                grid.tp,
                grid.tp_rank,
                scale=scale,
                expected=expected,
                device=device,
                refiner=sent.get("refiner"),
            )

        results, failed = _gather(_run(load))
        if failed:
            for block, old in zip(blocks, previous):
                if old is None:
                    block.__dict__.pop(pm.LORA_ATTR, None)
                else:
                    setattr(block, pm.LORA_ATTR, old)
            for block, old in zip(refiners, previous_refiner):
                if old is None:
                    block.__dict__.pop(h3_lora.REFINER_ATTR, None)
                else:
                    setattr(block, h3_lora.REFINER_ATTR, old)
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
        handle.remove()  # refused (and nothing changed) once vLLM-Omni's DiT was released
        _restore_request_lora(self)
        self._rlk_h3 = None

    def _rlk_handle(self):
        if self._rlk_h3 is None:
            raise RuntimeError("the H3 rollout DiT is not bound on this worker (rlk_h3_install)")
        return self._rlk_h3


def _preflight_checkpoint(transformer_dir: str) -> int:
    """Before anything is released: the index and every safetensors file it names are readable
    and hold the tensors the index assigns to them. Returns the tensor count."""

    import json

    from safetensors import safe_open

    from rl_engine.validation.models import h3_sharded_load as sl

    root = Path(transformer_dir)
    weight_map: dict[str, str] = json.loads((root / sl.INDEX).read_text())["weight_map"]
    by_file: dict[str, set[str]] = {}
    for name, file in weight_map.items():
        by_file.setdefault(file, set()).add(name)
    for file, names in sorted(by_file.items()):
        with safe_open(str(root / file), framework="pt") as f:
            missing = names - set(f.keys())
        if missing:
            raise ValueError(
                f"{file} lacks {len(missing)} indexed tensors, e.g. {sorted(missing)[:3]}"
            )
    return len(weight_map)


_REQUEST_ADAPTER = "_rlk_original_set_request_adapter"


def _refuse_request_lora(worker: Any) -> None:
    """While the DiT is bound, a request LoRA would only reach vLLM-Omni's (unused) DiT: refuse it.

    vLLM-Omni activates a request's adapter through ``lora_manager.set_request_adapter`` before
    each call and re-raises the error when a LoRA was requested.
    """

    manager = getattr(worker, "lora_manager", None)
    if manager is None or _REQUEST_ADAPTER in manager.__dict__:
        return
    original = manager.set_request_adapter

    def set_request_adapter(request: Any, scale: float = 1.0) -> Any:
        if request is not None:
            raise ValueError(
                "the H3 rollout DiT is bound: it runs the actor's LoRA (rlk_h3_load_lora), not a "
                "request LoRA, which would only reach vLLM-Omni's own DiT"
            )
        return original(request, scale)

    # (whether the manager had its own instance attribute, and that attribute)
    manager.__dict__[_REQUEST_ADAPTER] = ("set_request_adapter" in manager.__dict__, original)
    manager.set_request_adapter = set_request_adapter


def _restore_request_lora(worker: Any) -> None:
    manager = getattr(worker, "lora_manager", None)
    if manager is None or _REQUEST_ADAPTER not in manager.__dict__:
        return
    own, original = manager.__dict__.pop(_REQUEST_ADAPTER)
    if own:
        manager.set_request_adapter = original
    else:
        del manager.__dict__["set_request_adapter"]  # the class's method again


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
