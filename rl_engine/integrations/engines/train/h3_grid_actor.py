# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 LoRA training on the TP x CP grid (``h3-grid-train-v1``, RFC #420
``final_distributed_h3_run``).

``H3GridActor`` turns the weight-sharded model of ``h3_sharded_load`` with the LoRA of
``h3-parallel-lora-v1`` into something a training engine can step:

* **Parameters.** Every LoRA shard a block holds becomes an ``nn.Parameter`` in place (the block
  keeps reading the same storage), in a fixed order: block, target (``parallel_lora.TARGETS``),
  part (sorted). A whole LoRA on the token-refiner blocks (``h3_lora.attach_refiner_lora``) is
  added after them. The base stays frozen.
* **Gradients.** The grid makes a fresh gradient leaf of every shard on each call. Each leaf's
  gradient (already rounded once to the adapter dtype, BF16 for the recipe) is added to its
  parameter's ``.grad`` in the order passes run; base-shard gradients are dropped at once. That
  is the arithmetic of the recipe's FSDP2 actor, whose BF16 gradients take each backward's
  reduced gradient in turn. A pass in which a block LoRA shard gets no gradient is refused.
* **Refiner.** The refiner runs replicated, with the provider attention (``strict_boundaries=
  False``), whose backward need not repeat bit for bit across ranks; ``finish_grads`` therefore
  broadcasts global rank 0's refiner gradients, so every rank steps the same replicated tensors.
* **Clip and optimizer.** ``clip_grad_norm_`` is ``clip_grad_norm_exact_`` (integer-binned, the
  same float on every rank) with each replicated tensor counted once: ``parallel_lora.
  lora_counted`` for the blocks, global rank 0 for the refiner. ``make_optimizer`` is AdamW with
  the single-tensor kernels, elementwise, so a shard of the stepped full tensor equals the
  stepped shard and every replica stays equal.
* **Export.** ``export`` writes this rank's shards as ``<dir>/cp<c>/rank<t>.pt`` in the format
  the rollout's ``rlk_h3_load_lora`` installs (one-to-one, rollout TP rank ``t`` takes actor TP
  rank ``t``'s file), and checks that both CP replicas of every TP rank hold the same digests.

``Grid(1, 1)`` with a ``tp=1`` model is the canonical single-GPU run that every grid must equal.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl
from rl_engine.integrations.engines.rollout.vllm.h3_omni_model import (
    sdpa_backends_enabled,
    torch_default_sdpa,
)
from rl_engine.ops.norm.reproducible import clip_grad_norm_exact_
from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_lora
from rl_engine.validation.models import h3_parallel_model as pm
from rl_engine.validation.models import h3_tp
from rl_engine.validation.models.h3_full_model import FullModelTrace

CONTRACT = "h3-grid-train-v1"


def _world(group) -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group), dist.get_world_size(group)
    return 0, 1


def _as_param(tensor: torch.Tensor, *, requires_grad: bool) -> nn.Parameter:
    """``tensor`` as a parameter over the same storage."""

    if isinstance(tensor, nn.Parameter):
        tensor.requires_grad_(requires_grad)
        return tensor
    return nn.Parameter(tensor.detach(), requires_grad=requires_grad)


class H3GridActor:
    """One rank's trainable view of an H3 model whose blocks run on ``grid`` (see module doc)."""

    def __init__(
        self,
        model,
        grid: h3_tp.Grid,
        *,
        sp: bool,
        strict_boundaries: bool,
        ops: hb.CandidateOps | None = None,
        group: dist.ProcessGroup | None = None,
    ):
        """``group`` holds every rank of the grid (default: the whole world). The canonical
        ``Grid(1, 1)`` run of each rank passes a group of that rank alone."""

        h3_lora.require_refiner_lora_supported(model, strict_boundaries=strict_boundaries)
        self.model, self.grid, self.sp = model, grid, sp
        self.strict_boundaries = strict_boundaries
        self._ops = ops  # resolved on the first pass (the CUDA kernel rows)
        self.group = group
        self.rank, self.world = _world(group)
        self.names: list[str] = []  # lora_digests keys, in parameter order
        self._params: list[nn.Parameter] = []
        self._by_leaf: dict[tuple[int, str], nn.Parameter] = {}
        self._counted: dict[int, bool] = {}
        self._refiner: list[nn.Parameter] = []
        self.scale: float | None = None
        for index, block in enumerate(model.transformer_blocks):
            lora = block.__dict__.get(pm.LORA_ATTR)
            if lora is None:
                raise ValueError(f"block {index} holds no LoRA: attach or install one first")
            if (lora["tp"], lora["tp_rank"]) != (grid.tp, grid.tp_rank):
                raise ValueError(
                    f"block {index} holds the LoRA of tp={lora['tp']} rank {lora['tp_rank']}, "
                    f"the grid is tp={grid.tp} rank {grid.tp_rank}"
                )
            self._check_scale(lora["scale"])
            for target in pl.TARGETS:
                parts = lora["tensors"][target]
                for part in sorted(parts):
                    p = parts[part] = _as_param(parts[part], requires_grad=False)
                    self._add(f"{index}.{target}.{part}", p)
                    self._by_leaf[(index, f"lora.{target}.{part}")] = p
                    self._counted[id(p)] = pl.lora_counted(target, part, grid.tp_rank, grid.cp_rank)
        refiners = h3_lora.refiner_blocks(model)
        bare = [i for i, b in enumerate(refiners) if h3_lora.REFINER_ATTR not in b.__dict__]
        if 0 < len(bare) < len(refiners):  # state and names index the refiner blocks by position
            raise ValueError(f"refiner blocks {bare} hold no LoRA, the others do")
        for index, block in enumerate(refiners):
            lora = block.__dict__.get(h3_lora.REFINER_ATTR)
            if lora is None:
                continue
            self._check_scale(lora["scale"])
            for target in pl.TARGETS:
                parts = lora["tensors"][target]
                for part in sorted(parts):
                    p = parts[part] = _as_param(parts[part], requires_grad=True)
                    self._add(f"refiner.{index}.{target}.{part}", p)
                    self._refiner.append(p)
                    self._counted[id(p)] = self.rank == 0
        self._received: set[tuple[int, str]] = set()

    @property
    def ops(self) -> hb.CandidateOps:
        if self._ops is None:
            self._ops = hb.CandidateOps()
        return self._ops

    def _check_scale(self, scale: float) -> None:
        if self.scale is not None and scale != self.scale:
            raise ValueError(f"LoRA scales differ across blocks: {self.scale} and {scale}")
        self.scale = scale

    def _add(self, name: str, p: nn.Parameter) -> None:
        self.names.append(name)
        self._params.append(p)

    # --- passes -------------------------------------------------------------------------------

    def parameters(self) -> list[nn.Parameter]:
        """Every trainable tensor of this rank, in the fixed order of ``names``."""

        return list(self._params)

    def counted(self, p: torch.Tensor) -> bool:
        """Whether this rank counts ``p`` in the global gradient norm."""

        return self._counted[id(p)]

    def _sink(self, index: int, name: str, grad: torch.Tensor) -> None:
        p = self._by_leaf.get((index, name))
        if p is None:
            return  # a frozen base shard
        if grad.dtype != p.dtype:
            raise TypeError(f"{index}.{name}: gradient {grad.dtype} for a {p.dtype} shard")
        if p.grad is None:
            p.grad = grad.contiguous().clone()
        else:
            p.grad.add_(grad)
        self._received.add((index, name))  # distinct shards: a shard may sink more than once

    def _execution(self):
        return pm.parallel_model_execution(
            self.model,
            self.grid,
            sp=self.sp,
            trace=FullModelTrace(),
            ops=self.ops,
            strict_boundaries=self.strict_boundaries,
        )

    def forward(self, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """A forward-only pass (the old log-prob replay): ``(video, audio)``, the same on every
        rank."""

        with torch.no_grad(), torch_default_sdpa(), self._execution():
            return self.model(**inputs)

    def forward_backward(
        self,
        inputs: dict[str, Any],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One pass: ``loss = loss_fn(video, audio)`` (already divided by the accumulation count),
        its backward, and each LoRA gradient added to its parameter. Returns the detached loss
        and outputs."""

        self._received = set()
        with torch_default_sdpa(), self._execution() as state:
            state.grad_sink = self._sink
            video, audio = self.model(**inputs)
            loss = loss_fn(video, audio)
            if loss.dim() != 0:
                raise ValueError("loss_fn must return a scalar")
            loss.backward()
        if len(self._received) != len(self._by_leaf):
            raise RuntimeError(
                f"{len(self._received)} of {len(self._by_leaf)} block LoRA shards got a gradient"
            )
        return loss.detach(), (video.detach(), audio.detach())

    def finish_grads(self) -> None:
        """Before the clip: every rank takes global rank 0's refiner gradients (see module doc)."""

        if self.world > 1 and self._refiner:
            src = dist.get_global_rank(self.group, 0) if self.group is not None else 0
            # every rank must enter the same broadcasts: agree first on which refiner tensors
            # have a gradient, and refuse a disagreement instead of hanging in a collective
            device = self._refiner[0].device
            has = torch.tensor(
                [p.grad is not None for p in self._refiner], dtype=torch.int32, device=device
            )
            low, high = has.clone(), has.clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN, group=self.group)
            dist.all_reduce(high, op=dist.ReduceOp.MAX, group=self.group)
            if not torch.equal(low, high):
                split = [i for i, (a, b) in enumerate(zip(low.tolist(), high.tolist())) if a != b]
                raise RuntimeError(
                    f"refiner LoRA tensors {split} have a gradient on some ranks only"
                )
            for p, present in zip(self._refiner, has.tolist()):
                if present:
                    dist.broadcast(p.grad, src=src, group=self.group)

    def clip_grad_norm_(self, max_norm: float) -> float:
        """The exact global gradient norm, the same float on every rank; gradients are scaled to
        ``max_norm`` like ``torch.nn.utils.clip_grad_norm_``. A non-finite norm is returned with
        the gradients unchanged (skip the step)."""

        return clip_grad_norm_exact_(self._params, max_norm, counted=self.counted, group=self.group)

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    # --- state ---------------------------------------------------------------------------------

    def digests(self) -> dict[str, str]:
        """SHA-256 of every LoRA tensor this rank holds (``h3_lora.lora_digests``)."""

        return h3_lora.lora_digests(self.model)

    def lora_state(self) -> dict[str, Any]:
        """This rank's LoRA on the CPU, in the rollout's file format."""

        def cpu(t: torch.Tensor) -> torch.Tensor:
            return t.detach().to("cpu", copy=True)

        shards = [
            {
                t: {p: cpu(v) for p, v in parts.items()}
                for t, parts in b.__dict__[pm.LORA_ATTR]["tensors"].items()
            }
            for b in self.model.transformer_blocks
        ]
        refiner = [
            {t: {p: cpu(v) for p, v in parts.items()} for t, parts in lo["tensors"].items()}
            for lo in (
                b.__dict__.get(h3_lora.REFINER_ATTR) for b in h3_lora.refiner_blocks(self.model)
            )
            if lo is not None
        ]
        state: dict[str, Any] = {
            "tp": self.grid.tp,
            "tp_rank": self.grid.tp_rank,
            "shards": shards,
            "digests": self.digests(),
            # alpha / r: the rollout's rlk_h3_load_lora refuses a file whose scale is not the one
            # it is asked to apply (the digests cover the tensors only)
            "scale": self.scale,
        }
        if refiner:
            state["refiner"] = refiner
        return state

    def load_lora_state(self, state: dict[str, Any]) -> None:
        """Copy a saved ``lora_state`` of this rank into the parameters in place (resume)."""

        mine = (self.grid.tp, self.grid.tp_rank)
        if (state.get("tp"), state.get("tp_rank")) != mine or set(state["digests"]) != set(
            self.names
        ):
            raise ValueError(
                f"the saved LoRA is tp={state.get('tp')} rank {state.get('tp_rank')}'s, "
                f"this rank is tp={mine[0]} rank {mine[1]}"
            )
        if "scale" in state and state["scale"] != self.scale:
            raise ValueError(f"the saved LoRA has scale {state['scale']}, this actor {self.scale}")
        sources: dict[str, torch.Tensor] = {}
        for index, parts_by_target in enumerate(state["shards"]):
            for target, parts in parts_by_target.items():
                for part, t in parts.items():
                    sources[f"{index}.{target}.{part}"] = t
        for index, parts_by_target in enumerate(state.get("refiner", [])):
            for target, parts in parts_by_target.items():
                for part, t in parts.items():
                    sources[f"refiner.{index}.{target}.{part}"] = t
        for name, p in zip(self.names, self._params):
            src = sources[name]
            if src.shape != p.shape or src.dtype != p.dtype:
                raise ValueError(
                    f"{name}: saved {tuple(src.shape)} {src.dtype}, held {tuple(p.shape)} {p.dtype}"
                )
            with torch.no_grad():
                p.copy_(src)
        if self.digests() != state["digests"]:
            raise RuntimeError("the LoRA after loading differs from the saved digests")

    def export(self, directory: str) -> dict[str, Any]:
        """Write this rank's LoRA to ``<directory>/cp<c>/rank<t>.pt`` (atomically) and check that
        every TP rank's CP replicas hold the same digests and every rank the same refiner LoRA.

        The rollout engine fed by CP group ``c`` loads ``<directory>/cp<c>/rank{rank}.pt``.
        """

        folder = os.path.join(directory, f"cp{self.grid.cp_rank}")
        path = os.path.join(folder, f"rank{self.grid.tp_rank}.pt")
        # A rank whose write fails still joins the gather below, so every rank learns of it and
        # raises, instead of the others waiting in the collective for a rank that left.
        error = None
        try:
            state = self.lora_state()
            os.makedirs(folder, exist_ok=True)
            torch.save(state, path + ".tmp")
            os.replace(path + ".tmp", path)
            digests_here = state["digests"]
        except Exception as exc:  # reported to every rank, then raised there
            error = f"{type(exc).__name__}: {exc}"
            digests_here = {}
        mine = (self.grid.tp_rank, digests_here, error)
        everyone: list[Any] = [mine]
        if self.world > 1:
            everyone = [None] * self.world
            dist.all_gather_object(everyone, mine, group=self.group)
        failed = [(rank, err) for rank, (_, _, err) in enumerate(everyone) if err is not None]
        if failed:
            raise RuntimeError(f"the LoRA export failed on ranks {failed}")
        by_tp: dict[int, dict[str, str]] = {}
        refiner: dict[str, str] | None = None
        for tp_rank, digests, _ in everyone:
            blocks = {k: v for k, v in digests.items() if not k.startswith("refiner.")}
            ref = {k: v for k, v in digests.items() if k.startswith("refiner.")}
            if by_tp.setdefault(tp_rank, blocks) != blocks:
                raise RuntimeError(f"the CP replicas of TP rank {tp_rank} hold different LoRA")
            if refiner is None:
                refiner = ref
            elif refiner != ref:
                raise RuntimeError("the ranks hold different refiner LoRA")
        return {
            "path": path,
            "rollout_source": os.path.join(folder, "rank{rank}.pt"),
            "digests": digests_here,
            "scale": self.scale,
        }

    def checkpoint_state(self, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
        """This rank's LoRA, optimizer and RNG state (the engine saves one file per rank)."""

        out: dict[str, Any] = {
            "contract": CONTRACT,
            "grid": (self.grid.tp, self.grid.cp, self.grid.tp_rank, self.grid.cp_rank),
            "lora": self.lora_state(),
            "rng": {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        if optimizer is not None:
            out["optimizer"] = optimizer.state_dict()
        return out

    def load_checkpoint_state(
        self, state: dict[str, Any], optimizer: torch.optim.Optimizer | None = None
    ) -> None:
        grid = (self.grid.tp, self.grid.cp, self.grid.tp_rank, self.grid.cp_rank)
        if state.get("contract") != CONTRACT or tuple(state["grid"]) != grid:
            raise ValueError(
                f"checkpoint of {state.get('contract')} {state.get('grid')}, not {grid}"
            )
        self.load_lora_state(state["lora"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"]["cpu"])
        if state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(state["rng"]["cuda"])

    def readback(self) -> dict[str, Any]:
        """What this actor runs, for the per-step evidence."""

        return {
            "contract": CONTRACT,
            "parallel_lora": pl.CONTRACT,
            "parallel_model": pm.CONTRACT,
            "grid": pm.grid_readback(self.grid, sp=self.sp, ops=self.ops),
            "strict_boundaries": self.strict_boundaries,
            "lora_tensors": len(self._params),
            "refiner_lora_tensors": len(self._refiner),
            "lora_scale": self.scale,
            "sdpa_backends_outside_the_call": sdpa_backends_enabled(),
        }


def make_optimizer(
    params: Iterable[nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """AdamW with the single-tensor (elementwise) kernels; states take the parameters' dtype."""

    return torch.optim.AdamW(
        list(params),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )
