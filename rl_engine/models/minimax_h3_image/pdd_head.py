# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3-Image PDD output heads (RFC #738 ``h3_image_pdd_head_contract``).

The checkpoint is distilled to six model evaluations and carries one FP32 output head per
evaluation. Which head runs is part of the model input: head ``i`` belongs to denoising step
``i``, i.e. to the interval ``sigma_i -> sigma_{i+1}`` of the sealed grid, and to nothing else.

* **Artifact.** ``pdd_fused_heads.safetensors`` holds ``video_out.weight`` ``[6, 96, 5376]`` and
  ``video_out.bias`` ``[6, 96]``, FP32. :func:`from_fused_file` / :func:`from_fused` /
  :func:`from_diffusers` check names, shapes, dtype and the per-head sha256 of the pinned
  release; a missing, reordered, cast or altered head fails (probe I2).
* **Selection.** :meth:`PDDHeads.select` takes the step as a plain ``int``. A timestep or sigma
  pair, if the caller has one, must be the step's own value bit for bit (probe I1); nothing is
  ever looked up by a nearest value.
* **Batch.** One packed forward runs one head. :func:`common_step` refuses a batch whose requests
  are at different steps.
* **Checkpointing.** A :class:`BoundHead` holds the selected tensors and their version counters.
  A recomputation runs the head that was bound, whatever the loop variable is by then, and
  refuses a head whose values were changed in between.
* **Rank.** :func:`check_ranks` compares step, head digest and interval across a process group
  before the head runs.
* **Loop state.** :class:`HeadCursor` hands out steps 0..5 once each, in order.

The projection itself is ``video_output_projection_fp32`` of RFC #420: same shape, same FP32
contract. :func:`head_projection` resolves the operator and says which one it is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import pdd_schedule as ps
from rl_engine.reference.minimax_h3.sigma_schedule import H3SigmaSchedule

CONTRACT_ID = "h3-image-pdd-head-v1"
PROJECTION_ROW = "video_output_projection_fp32"
NUM_HEADS = fp.NUM_PDD_HEADS
HEAD_WIDTH = 96  # 24 latent channels x patch (1, 2, 2)
HIDDEN = 5376
WEIGHT_SHAPE = (NUM_HEADS, HEAD_WIDTH, HIDDEN)
BIAS_SHAPE = (NUM_HEADS, HEAD_WIDTH)


class HeadArtifactError(ValueError):
    """The six heads are not the released ones (names, shape, dtype, order or bytes)."""


class HeadDrift(RuntimeError):
    """A head other than the one selected for the step would run."""


def head_sha256(tensor: torch.Tensor) -> str:
    """sha256 of a head tensor's raw FP32 bytes (the definition ``pinned_checkpoint`` uses)."""

    if tensor.dtype != torch.float32:
        raise HeadArtifactError(f"head tensors are FP32, got {tensor.dtype}")
    data = tensor.detach().cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def released_digests() -> tuple[tuple[str, str], ...]:
    """``(weight, bias)`` sha256 of every released head, in step order."""

    return tuple((h["weight"], h["bias"]) for h in fp.pdd_head_digests())


@dataclass(frozen=True)
class HeadSelection:
    """Head ``head`` for step ``step``; ``head == step`` by construction."""

    step: int
    head: int
    sigma: float
    sigma_next: float
    timestep: float
    weight: torch.Tensor
    bias: torch.Tensor
    weight_sha256: str
    bias_sha256: str
    sealed: bool

    def artifact(self) -> dict[str, Any]:
        """JSON record of the selection (no tensors)."""

        return {
            "contract_id": CONTRACT_ID,
            "step": self.step,
            "head": self.head,
            "sigma": self.sigma.hex(),
            "sigma_next": self.sigma_next.hex(),
            "timestep": self.timestep.hex(),
            "weight_sha256": self.weight_sha256,
            "bias_sha256": self.bias_sha256,
            "sealed": self.sealed,
        }


def _check_head(index: int, weight: Any, bias: Any) -> None:
    for name, tensor, shape in (
        ("weight", weight, WEIGHT_SHAPE[1:]),
        ("bias", bias, BIAS_SHAPE[1:]),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise HeadArtifactError(f"head {index} {name} is missing")
        if tensor.dtype != torch.float32:
            raise HeadArtifactError(f"head {index} {name} must be FP32, got {tensor.dtype}")
        if tuple(tensor.shape) != shape:
            raise HeadArtifactError(
                f"head {index} {name} must have shape {list(shape)}, got {list(tensor.shape)}"
            )


class PDDHeads:
    """The six output heads, in step order.

    ``sealed=True`` (the default) requires every head to be the released one. ``sealed=False``
    is for synthetic weights and for heads under training; its selections say so.
    """

    def __init__(
        self,
        weights: Sequence[torch.Tensor],
        biases: Sequence[torch.Tensor],
        *,
        sealed: bool = True,
        schedule: H3SigmaSchedule | None = None,
    ) -> None:
        if len(weights) != NUM_HEADS or len(biases) != NUM_HEADS:
            raise HeadArtifactError(
                f"{NUM_HEADS} heads are required, got {len(weights)} weights and "
                f"{len(biases)} biases"
            )
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            _check_head(index, weight, bias)
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        self.sealed = bool(sealed)
        self.schedule = schedule if schedule is not None else ps.released_schedule("cpu")
        if self.schedule.num_model_evals != NUM_HEADS:
            raise HeadArtifactError("the schedule must have one interval per head")
        self.digests = tuple(
            (head_sha256(weight), head_sha256(bias)) for weight, bias in zip(weights, biases)
        )
        self._versions = self._current_versions()
        if self.sealed:
            self._check_released()

    def _check_released(self) -> None:
        released = released_digests()
        for index, digest in enumerate(self.digests):
            if digest == released[index]:
                continue
            where = [i for i, other in enumerate(released) if other == digest]
            hint = f" (it is released head {where[0]})" if where else ""
            raise HeadArtifactError(f"head {index} is not released head {index}{hint}")

    def _current_versions(self) -> tuple[int, ...]:
        return tuple(t._version for t in (*self.weights, *self.biases))

    def verify(self) -> None:
        """Re-hash every head; raise if one changed since construction (any in-place write)."""

        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if (head_sha256(weight), head_sha256(bias)) != self.digests[index]:
                raise HeadDrift(f"head {index} changed since it was loaded")

    def to(self, device: torch.device | str) -> PDDHeads:
        """The same heads on ``device``; digests are recomputed there (a round-trip check)."""

        return PDDHeads(
            [w.to(device) for w in self.weights],
            [b.to(device) for b in self.biases],
            sealed=self.sealed,
            schedule=self.schedule,
        )

    def fused_state_dict(self) -> dict[str, torch.Tensor]:
        """The heads in the released file's layout (``[6, 96, 5376]`` and ``[6, 96]``)."""

        return {
            fp.PDD_WEIGHT: torch.stack([w.detach() for w in self.weights]).contiguous(),
            fp.PDD_BIAS: torch.stack([b.detach() for b in self.biases]).contiguous(),
        }

    def select(
        self,
        step: int,
        *,
        timestep: float | torch.Tensor | None = None,
        sigmas: tuple[float | torch.Tensor, float | torch.Tensor] | None = None,
    ) -> HeadSelection:
        """The head of denoising step ``step``.

        ``step`` is a plain ``int``. ``timestep`` and ``sigmas`` are cross-checks: each must be
        the step's own value, bit for bit, or the call fails. Neither ever chooses the head.
        """

        head = ps.head_for_step(step)
        if timestep is not None:
            by_time = ps.head_for_timestep(self.schedule, timestep)
            if by_time != head:
                raise HeadDrift(f"step {step} got the timestep of step {by_time}")
        interval = (float(self.schedule.sigmas[step]), float(self.schedule.sigmas[step + 1]))
        if sigmas is not None:
            got = (float(sigmas[0]), float(sigmas[1]))
            if got != interval:
                raise HeadDrift(f"step {step} covers sigmas {interval}, got {got}")
        if self._current_versions() != self._versions:
            # Cheap when nothing was written; a sealed bank must still hold the released bytes.
            if self.sealed:
                self.verify()
            else:
                self.digests = tuple(
                    (head_sha256(w), head_sha256(b)) for w, b in zip(self.weights, self.biases)
                )
            self._versions = self._current_versions()
        weight_digest, bias_digest = self.digests[head]
        return HeadSelection(
            step=step,
            head=head,
            sigma=interval[0],
            sigma_next=interval[1],
            timestep=float(self.schedule.timesteps[step]),
            weight=self.weights[head],
            bias=self.biases[head],
            weight_sha256=weight_digest,
            bias_sha256=bias_digest,
            sealed=self.sealed,
        )

    def bind(
        self,
        step: int,
        projection: HeadProjection | None = None,
        *,
        timestep: float | torch.Tensor | None = None,
    ) -> BoundHead:
        """Select the head of ``step`` and hold it as a callable (see :class:`BoundHead`)."""

        selection = self.select(step, timestep=timestep)
        if projection is None:
            projection = head_projection(selection.weight.device)
        return BoundHead(selection, projection)

    def artifact(self) -> dict[str, Any]:
        """Sealed description: shapes, per-head digests and the step-to-interval map."""

        body: dict[str, Any] = {
            "contract_id": CONTRACT_ID,
            "weight_shape": list(WEIGHT_SHAPE),
            "bias_shape": list(BIAS_SHAPE),
            "dtype": "float32",
            "sealed": self.sealed,
            "schedule_sha256": ps.artifact(self.schedule)["sha256"],
            "heads": [self.select(step).artifact() for step in range(NUM_HEADS)],
        }
        body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return body


# --- Loading -----------------------------------------------------------------------------------


def from_fused(tensors: Mapping[str, torch.Tensor], *, sealed: bool = True) -> PDDHeads:
    """Heads from the released fused layout; exactly its two tensors, FP32, in step order."""

    expected = {fp.PDD_WEIGHT: WEIGHT_SHAPE, fp.PDD_BIAS: BIAS_SHAPE}
    if set(tensors) != set(expected):
        raise HeadArtifactError(
            f"the fused head file holds {sorted(expected)}, got {sorted(tensors)}"
        )
    for name, shape in expected.items():
        tensor = tensors[name]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise HeadArtifactError(
                f"{name} must be FP32 {list(shape)}, got {tensor.dtype} {list(tensor.shape)}"
            )
    weights = [w.contiguous() for w in tensors[fp.PDD_WEIGHT].unbind(0)]
    biases = [b.contiguous() for b in tensors[fp.PDD_BIAS].unbind(0)]
    return PDDHeads(weights, biases, sealed=sealed)


def from_fused_file(path: str | Path, *, sealed: bool = True) -> PDDHeads:
    """Heads from ``pdd_fused_heads.safetensors`` (a file, or the release directory)."""

    from safetensors.torch import load_file

    path = Path(path)
    if path.is_dir():
        path = path / fp.PDD_FILE
    if not path.is_file():
        raise HeadArtifactError(f"{path.name} not found")
    return from_fused(load_file(str(path)), sealed=sealed)


def from_diffusers(source: Any, *, prefix: str = "pdd_proj_out", sealed: bool = True) -> PDDHeads:
    """Heads of a Diffusers transformer, or of its ``name -> tensor`` mapping (no copies).

    The mapping must hold exactly heads ``0..5``: a seventh head, a gap or a missing bias fails.
    """

    tensors = source if isinstance(source, Mapping) else dict(source.named_parameters())
    found = {name for name in tensors if name.startswith(prefix + ".")}
    expected = {f"{prefix}.{i}.{part}" for i in range(NUM_HEADS) for part in ("weight", "bias")}
    if found != expected:
        missing, extra = sorted(expected - found), sorted(found - expected)
        raise HeadArtifactError(f"PDD heads: missing {missing}, unexpected {extra}")
    return PDDHeads(
        [tensors[f"{prefix}.{i}.weight"] for i in range(NUM_HEADS)],
        [tensors[f"{prefix}.{i}.bias"] for i in range(NUM_HEADS)],
        sealed=sealed,
    )


# --- Projection --------------------------------------------------------------------------------


class _InterimHead(torch.autograd.Function):
    """``x @ W^T + b`` on the Triton FP32 K loop, both directions (rows independent).

    Stands in for ``video_output_projection_fp32`` where that row is not qualified. ``dX``
    reduces the 96 output channels; ``dW`` and ``db`` reduce the rows in call order.
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

        ctx.save_for_backward(x, weight)
        return _triton_gemm_fp32(x, weight.t()) + bias

    @staticmethod
    def backward(ctx, grad):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

        x, weight = ctx.saved_tensors
        grad = grad.contiguous()
        needs_x, needs_weight, needs_bias = ctx.needs_input_grad
        dx = _triton_gemm_fp32(grad, weight) if needs_x else None
        dw = _triton_gemm_fp32(grad.t(), x) if needs_weight else None
        ones = torch.ones(1, grad.shape[0], dtype=torch.float32, device=grad.device)
        db = _triton_gemm_fp32(ones, grad)[0] if needs_bias else None
        return dx, dw, db


def _interim_head(x, weight, bias, *, logical_keys=None, parameter_id=None):
    """The interim projection with the row's call signature (keys are not used)."""

    flat = x.reshape(-1, x.shape[-1]).contiguous()
    return _InterimHead.apply(flat, weight, bias).view(*x.shape[:-1], weight.shape[0])


@dataclass(frozen=True)
class HeadProjection:
    """The operator a head runs on, and how it is bound: ``row``, ``interim`` or ``reference``."""

    call: Callable[..., torch.Tensor]
    status: str
    backend: str

    def provenance(self) -> dict[str, str]:
        return {"rfc_row": PROJECTION_ROW, "status": self.status, "backend": self.backend}


def head_projection(device: torch.device | str, binding: str = "auto") -> HeadProjection:
    """Resolve the head's projection on ``device``.

    ``row`` is the RFC #420 operator (CUDA SM90 only; raises elsewhere). ``interim`` is the fixed
    FP32 K loop it is built on, for other CUDA devices. ``reference`` is the CPU FP32 reference.
    ``auto`` takes ``row`` where it is qualified, then ``interim`` on CUDA, else ``reference``.
    """

    if binding not in ("auto", "row", "interim", "reference"):
        raise ValueError("binding must be 'auto', 'row', 'interim' or 'reference'")
    device = torch.device(device)
    if binding == "auto":
        if device.type != "cuda":
            binding = "reference"
        else:
            sm90 = torch.cuda.get_device_capability(device) == (9, 0)
            binding = "row" if sm90 else "interim"
    if binding == "reference":
        from rl_engine.reference.gemm.video_output_projection_fp32 import (
            NativeVideoOutputProjectionFP32Op,
        )

        reference = NativeVideoOutputProjectionFP32Op()
        return HeadProjection(reference, "reference", reference.reference_id)
    if device.type != "cuda":
        raise RuntimeError(f"the {binding} head projection needs CUDA")
    if binding == "interim":
        from rl_engine.backends.shared.triton.gemm import det_gemm

        if not det_gemm._TRITON_AVAILABLE:
            raise RuntimeError("the interim head projection needs Triton")
        return HeadProjection(_interim_head, "interim", "triton _det_gemm_fp32_kernel (FP32)")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(f"{PROJECTION_ROW} is qualified on CUDA SM90 only")
    from rl_engine.backends.shared.triton.gemm.video_output_projection_fp32 import (
        TritonVideoOutputProjectionFP32Op,
    )

    op = TritonVideoOutputProjectionFP32Op()
    return HeadProjection(op, "row", op.backend_id)


class BoundHead:
    """One selected head as a callable, safe under activation checkpointing.

    It holds the selected tensors, not an index into the bank and not the loop variable, so a
    recomputation after the loop has moved on still runs this head. If the tensors were written
    in place since they were bound, the call raises instead of recomputing with other values.
    """

    def __init__(self, selection: HeadSelection, projection: HeadProjection) -> None:
        self.selection = selection
        self.projection = projection
        self._versions = (selection.weight._version, selection.bias._version)
        self.calls = 0

    def __call__(
        self,
        hidden: torch.Tensor,
        *,
        logical_keys: torch.Tensor | None = None,
        parameter_id: str | None = None,
    ) -> torch.Tensor:
        """Project every packed row: FP32 ``[M, 5376]`` or ``[B, S, 5376]`` -> ``[..., 96]``."""

        s = self.selection
        if (s.weight._version, s.bias._version) != self._versions:
            raise HeadDrift(f"head {s.head} was modified after it was bound to step {s.step}")
        if hidden.dtype != torch.float32:
            raise TypeError(f"the head input is cast to FP32 by the caller, got {hidden.dtype}")
        self.calls += 1
        return self.projection.call(
            hidden,
            s.weight,
            s.bias,
            logical_keys=logical_keys,
            parameter_id=parameter_id or f"pdd_proj_out.{s.head}",
        )

    def artifact(self) -> dict[str, Any]:
        return {**self.selection.artifact(), "projection": self.projection.provenance()}


# --- Drift guards ------------------------------------------------------------------------------


def common_step(steps: Sequence[int]) -> int:
    """The one step every request of a packed forward is at; a mixed batch is refused."""

    checked = {ps.head_for_step(step) for step in steps}
    if not checked:
        raise HeadDrift("an empty batch has no step")
    if len(checked) != 1:
        raise HeadDrift(f"one forward runs one PDD head, got steps {sorted(checked)}")
    return checked.pop()


class HeadCursor:
    """Hands out steps ``0..5`` once each, in order."""

    def __init__(self, heads: PDDHeads) -> None:
        self.heads = heads
        self.next_step = 0

    def take(self, step: int, *, timestep: float | torch.Tensor | None = None) -> HeadSelection:
        """The selection of ``step``, which must be the next one."""

        ps.head_for_step(step)
        if step != self.next_step:
            raise HeadDrift(f"expected step {self.next_step}, got {step}")
        selection = self.heads.select(step, timestep=timestep)
        self.next_step += 1
        return selection

    def finish(self) -> None:
        """Raise unless all six steps were taken."""

        if self.next_step != NUM_HEADS:
            raise HeadDrift(f"{self.next_step} of {NUM_HEADS} steps were taken")


def _digest_words(selection: HeadSelection) -> list[int]:
    """The selection as int64 words (step, head, interval bits, digests), for an all-gather."""

    text = json.dumps(selection.artifact(), sort_keys=True).encode()
    raw = hashlib.sha256(text).digest()
    words = [int.from_bytes(raw[i : i + 8], "little", signed=True) for i in range(0, 32, 8)]
    return [selection.step, selection.head, *words]


def check_ranks(selection: HeadSelection, group: Any | None = None) -> list[tuple[int, int]]:
    """Every rank's ``(step, head)``; raises ``HeadDrift`` on every rank if one differs.

    The record compared covers the step, the interval, the head digests and the seal, so a rank
    with the right index and another head's weights is caught as well.
    """

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return [(selection.step, selection.head)]
    # NCCL needs the tensors on the GPU; every other backend takes host tensors
    device = selection.weight.device if dist.get_backend(group) == "nccl" else torch.device("cpu")
    local = torch.tensor(_digest_words(selection), dtype=torch.int64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    rows = [[int(v) for v in t.cpu()] for t in gathered]
    pairs = [(row[0], row[1]) for row in rows]
    if any(row != rows[0] for row in rows):
        detail = ", ".join(
            f"rank {rank}: step {row[0]} head {row[1]} record {row[2] & 0xFFFFFFFFFFFFFFFF:016x}"
            for rank, row in enumerate(rows)
        )
        raise HeadDrift(f"ranks would run different PDD heads ({detail})")
    return pairs


class HeadReadback:
    """Records which of a Diffusers transformer's six heads really ran."""

    def __init__(self, model: Any) -> None:
        heads = getattr(model, "pdd_proj_out", None)
        if heads is None or len(heads) != NUM_HEADS:
            raise HeadArtifactError("the model has no six pdd_proj_out heads")
        self._heads = list(heads)
        self._handles: list[Any] = []
        self.calls = [0] * NUM_HEADS

    def __enter__(self) -> HeadReadback:
        for index, module in enumerate(self._heads):

            def hook(_module, _args, _output, index=index):
                self.calls[index] += 1

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def check(self, step: int, calls: int = 1) -> None:
        """Raise unless head ``step`` ran ``calls`` times and no other head ran; then reset."""

        head = ps.head_for_step(step)
        expected = [calls if i == head else 0 for i in range(NUM_HEADS)]
        seen, self.calls = self.calls, [0] * NUM_HEADS
        if seen != expected:
            raise HeadDrift(f"step {step} must run head {head} {calls}x, ran {seen}")
