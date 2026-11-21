# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM-Omni worker extension that binds the H3 rollout DiT (``h3_omni_model``) in each worker.

Pass ``worker_extension_cls="rl_engine.integrations.engines.rollout.vllm.h3_omni_worker.
H3OmniWorkerExtension"`` to the diffusion engine; its methods then run on every TP worker through
``collective_rpc``:

* ``rlk_h3_install(transformer_dir, ...)`` loads this rank's shards of the Diffusers-layout H3
  (``h3_sharded_load``) and replaces the pipeline's DiT call (``install_h3_omni_model``);
* ``rlk_h3_attach_seeded_lora(...)`` / ``rlk_h3_load_lora(...)`` put a LoRA on the bound model
  (a seeded one for tests, or the actor's shards with their digests) and return the digests;
* ``rlk_h3_report()`` returns the readback (grid, collective, calls, digests of every DiT output).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


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
        handle = om.install_h3_omni_model(
            pipeline.transformer, binding, od_config=od_config, release_weights=release_weights
        )
        torch.cuda.empty_cache()
        self._rlk_h3 = handle
        return {**handle.report(), "resident_param_bytes": _resident(model)}

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

    def rlk_h3_load_lora(
        self, shards: list, *, scale: float, expected: dict[str, str]
    ) -> dict[str, str]:
        from rl_engine.validation.models import h3_lora

        handle = self._rlk_handle()
        grid = handle.binding.grid
        return h3_lora.install_lora_shards(
            handle.binding.model, shards, grid.tp, grid.tp_rank, scale=scale, expected=expected
        )

    def rlk_h3_report(self) -> dict[str, Any]:
        handle = self._rlk_handle()
        return {**handle.report(), "digests": list(handle.binding.digests)}

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
