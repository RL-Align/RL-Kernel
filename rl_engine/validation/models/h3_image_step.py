# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""One real MiniMax-H3-Image step, layer by layer (RFC #738 ``ws1_one_h3_image_step``).

A step is one transformer call of the denoising loop: text features and image rows in, the
velocity of every image row out, with the PDD head of that step. Two arms run the same step and
record the same nodes in the same order:

* strict: :class:`StrictImageTransformer`, the RL-Kernel graph on the checkpoint's tensors. It
  uses the RFC #420 rows (the 17-node block graph of ``h3_block``, the refiner graph of
  ``h3_refiner``, the time, projection, packing and final-norm rows) and the PDD head contract.
  It does not import Diffusers.
* provider: :func:`run_provider`, the unmodified pinned Diffusers transformer, observed with
  forward hooks.

Node names, in execution order: ``time_proj``, ``time_embedder``, ``proj_in``,
``context_embedder``, ``refiner.<i>``, ``token_refiner``, ``packed``, ``block.<i>``,
``norm_out``, ``pdd_proj_out`` (every packed row) and ``velocity`` (the image rows). A trace may
also record the 17 nodes inside chosen blocks (``block.<i>.<node>``).

**Canonical rows.** The strict arm packs rows in the canonical order ``[text | image rows in
selection order]`` whatever their physical position is. Reductions over the packed sequence
(attention) then never see the physical order, so a fully consistent row permutation (I-C3)
gives the same bytes. The released layout already is canonical.

**What is equal to what.** Strict executions are bitwise equal to each other: repeated, batched
with another request, physically permuted, under autograd, under activation checkpointing. The
strict arm is *not* bitwise equal to Diffusers, whose GEMMs and attention are row-count
dependent; :func:`compare` reports the distance per node instead, next to the distance between
Diffusers kernel choices, and ``isolated_blocks`` the distance of one block on the same input.

:func:`first_drift` names the first node where two traces differ, the row group (text, reference
or target) and the first packed row.
"""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch

from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import pdd_head as ph
from rl_engine.models.minimax_h3_image import pdd_schedule as ps
from rl_engine.validation.models import h3_block
from rl_engine.validation.models import h3_full_model as hfm
from rl_engine.validation.models import h3_provider as prov
from rl_engine.validation.models import h3_refiner

CONTRACT_ID = "h3-image-one-step-v1"
BLOCKS = "transformer_blocks"
NORM_EPS = 1e-5
ROW_GROUPS = ("text", "reference", "target")
# Nodes whose rows are the packed rows of the sequence (so a drift has a row group).
_STEM_ROWS = {"proj_in": "image", "context_embedder": "text", "token_refiner": "text"}


class StepInputError(ValueError):
    """The step's inputs do not describe one packed forward."""


@dataclass(frozen=True)
class StepInputs:
    """Inputs of one transformer call; the batch shares the layout and the step."""

    image_rows: torch.Tensor  # FP32 (B, N_image, 96), references first
    prompt_embeds: torch.Tensor  # (B, N_text, 5120)
    layout: pl.PackedLayout
    unique_timesteps: torch.Tensor  # FP32 (T,), distinct values
    timestep_indices: torch.Tensor  # int64 (S,), by physical row
    step: int

    def __post_init__(self) -> None:
        ps.head_for_step(self.step)
        if self.image_rows.ndim != 3 or self.image_rows.shape[-1] != ph.HEAD_WIDTH:
            raise StepInputError("image_rows must be (B, N, 96)")
        if self.image_rows.dtype != torch.float32:
            raise StepInputError(f"image_rows must be FP32, got {self.image_rows.dtype}")
        if self.prompt_embeds.ndim != 3 or len(self.prompt_embeds) != len(self.image_rows):
            raise StepInputError("prompt_embeds must be (B, T, 5120) with the same batch")
        if self.image_rows.shape[1] != self.layout.image_indices.numel():
            raise StepInputError("one image row per image index is required")
        if self.prompt_embeds.shape[1] != self.layout.text_indices.numel():
            raise StepInputError("one prompt row per text index is required")
        if self.timestep_indices.shape != (self.layout.sequence_length,):
            raise StepInputError("one timestep index per packed row is required")

    @property
    def batch(self) -> int:
        return int(self.image_rows.shape[0])

    def digest(self) -> dict[str, Any]:
        """sha256 of every input tensor and the step."""

        return {
            "image_rows": digest(self.image_rows),
            "prompt_embeds": digest(self.prompt_embeds),
            "layout_sha256": pl.layout_digest(self.layout),
            "unique_timesteps": digest(self.unique_timesteps),
            "timestep_indices": digest(self.timestep_indices),
            "step": self.step,
        }


def digest(tensor: torch.Tensor) -> str:
    """sha256 over dtype, shape and bytes."""

    t = tensor.detach().cpu().contiguous()
    header = f"{t.dtype}|{tuple(t.shape)}|".encode()
    return hashlib.sha256(header + t.view(torch.uint8).numpy().tobytes()).hexdigest()


def stack(inputs: Iterable[StepInputs]) -> StepInputs:
    """Batch requests that share layout, timesteps and step (one head runs for all of them)."""

    items = list(inputs)
    step = ph.common_step([i.step for i in items])
    first = items[0]
    reference = pl.layout_digest(first.layout)
    for other in items[1:]:
        if pl.layout_digest(other.layout) != reference:
            raise StepInputError("batched requests must share one packed layout")
        if not torch.equal(other.unique_timesteps, first.unique_timesteps):
            raise StepInputError("batched requests must share the timestep table")
        if not torch.equal(other.timestep_indices, first.timestep_indices):
            raise StepInputError("batched requests must share the timestep indices")
    return StepInputs(
        torch.cat([i.image_rows for i in items]),
        torch.cat([i.prompt_embeds for i in items]),
        first.layout,
        first.unique_timesteps,
        first.timestep_indices,
        step,
    )


def permute_inputs(inputs: StepInputs, order: torch.Tensor) -> StepInputs:
    """A fully consistent physical row permutation of the step (control I-C3)."""

    return StepInputs(
        inputs.image_rows,
        inputs.prompt_embeds,
        pl.permute(inputs.layout, order),
        inputs.unique_timesteps,
        inputs.timestep_indices.cpu()[order],
        inputs.step,
    )


# --- Trace -------------------------------------------------------------------------------------


@dataclass
class StepTrace:
    """Ordered node records of one step: digests, and the tensors (on the CPU) on request."""

    label: str = ""
    keep_tensors: bool = False
    detail_blocks: frozenset[int] = frozenset()
    order: list[str] = field(default_factory=list)
    digests: dict[str, str] = field(default_factory=dict)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    bindings: dict[str, dict[str, str]] = field(default_factory=dict)
    head: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        """Record node ``name``; each node runs once per step."""

        if name in self.order:
            raise RuntimeError(f"node {name} ran twice in one step")
        # a tuple (the AdaLN tables) is recorded as one tensor, so a batch row of it compares
        flat = h3_block._flat(value)
        tensor = flat[0] if len(flat) == 1 else torch.cat([t.contiguous() for t in flat], dim=-1)
        self.order.append(name)
        self.digests[name] = digest(tensor)
        if self.keep_tensors:
            self.tensors[name] = tensor.detach().cpu().clone()

    def row(self, index: int) -> StepTrace:
        """Batch row ``index`` of a trace kept with tensors, as a trace of its own."""

        out = StepTrace(label=f"{self.label}[{index}]", keep_tensors=True)
        for name in self.order:
            value = self.tensors[name]
            shared = name in ("time_proj", "time_embedder") or name.endswith(".adaln_projection")
            out.add(name, value if shared else value[index : index + 1].contiguous())
        return out


def row_group(layout: pl.PackedLayout, node: str, row: int) -> str | None:
    """``text`` / ``reference`` / ``target`` of canonical row ``row`` at ``node``."""

    text = int(layout.text_indices.numel())
    if node in ("time_proj", "time_embedder") or node.endswith(".adaln_projection"):
        return None
    if _STEM_ROWS.get(node) == "text" or node.startswith("refiner."):
        return "text"
    image_row = row if _STEM_ROWS.get(node) == "image" or node == "velocity" else row - text
    if image_row < 0:
        return "text"
    return "reference" if image_row < layout.num_reference_rows else "target"


def first_drift(
    lhs: StepTrace, rhs: StepTrace, layout: pl.PackedLayout | None = None
) -> dict[str, Any] | None:
    """First node where two traces differ, or ``None``.

    With tensors kept on both sides it also names the batch row, the first packed row, its group
    and the largest difference at that node.
    """

    if lhs.order != rhs.order:
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(lhs.order, rhs.order)) if a != b),
            min(len(lhs.order), len(rhs.order)),
        )
        return {"node": None, "reason": "different node sequences", "position": mismatch}
    for name in lhs.order:
        if lhs.digests[name] == rhs.digests[name]:
            continue
        out: dict[str, Any] = {"node": name, "index": lhs.order.index(name)}
        a, b = lhs.tensors.get(name), rhs.tensors.get(name)
        if a is None or b is None:
            return out
        if a.shape != b.shape or a.dtype != b.dtype:
            return {**out, "reason": f"{a.dtype}{list(a.shape)} vs {b.dtype}{list(b.shape)}"}
        out["max_abs"] = float((a.double() - b.double()).abs().max())
        out["differing_values"] = int((a != b).sum())
        if a.ndim >= 3:
            rows = (a != b).flatten(2).any(-1)
            hits = rows.nonzero()
            if len(hits):  # -0.0 vs 0.0 differs in bytes only
                out["batch_row"], out["row"] = int(hits[0][0]), int(hits[0][1])
                out["differing_rows"] = int(rows.sum())
                if layout is not None:
                    out["row_group"] = row_group(layout, name, out["row"])
        return out
    return None


def _relative(error: torch.Tensor, base: torch.Tensor) -> float:
    norm = base.norm()
    return float(error.norm() / norm) if float(norm) else float(error.norm())


def distance(
    node: str, value: torch.Tensor, base: torch.Tensor, layout: pl.PackedLayout | None = None
) -> dict[str, Any]:
    """Distance of one node's output from ``base``.

    ``rel_frobenius`` weighs every value alike, so one row with huge activations can decide it;
    ``by_group`` splits it over the row groups, and ``largest_error`` says where the largest
    difference is and how large the reference value is there.
    """

    a, b = value.double().cpu(), base.double().cpu()
    error = a - b
    out: dict[str, Any] = {
        "node": node,
        "bitwise_equal": value.dtype == base.dtype and torch.equal(a, b),
        "max_abs": float(error.abs().max()),
        "rel_frobenius": _relative(error, b),
    }
    if layout is None or a.ndim != 3 or a.shape[1] == 0:
        return out
    groups = [row_group(layout, node, row) for row in range(a.shape[1])]
    out["by_group"] = {
        group: _relative(error[:, rows], b[:, rows])
        for group in ROW_GROUPS
        if (rows := [i for i, g in enumerate(groups) if g == group])
    }
    batch, row, channel = (int(i) for i in torch.unravel_index(error.abs().argmax(), error.shape))
    out["largest_error"] = {
        "batch_row": batch,
        "row": row,
        "row_group": groups[row],
        "channel": channel,
        "reference": float(b[batch, row, channel]),
        "value": float(a[batch, row, channel]),
    }
    return out


def compare(
    lhs: StepTrace, rhs: StepTrace, layout: pl.PackedLayout | None = None
) -> list[dict[str, Any]]:
    """Per-node :func:`distance` of two kept traces; ``rhs`` is the base."""

    return [distance(n, lhs.tensors[n], rhs.tensors[n], layout) for n in lhs.order]


# --- Strict arm --------------------------------------------------------------------------------


class _InterimInput(torch.autograd.Function):
    """``x @ W^T + b`` (FP32, ``proj_in``) on the Triton FP32 K loop, rows independent."""

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
        ones = torch.ones(1, grad.shape[0], dtype=torch.float32, device=grad.device)
        return (
            _triton_gemm_fp32(grad, weight) if needs_x else None,
            _triton_gemm_fp32(grad.t(), x) if needs_weight else None,
            _triton_gemm_fp32(ones, grad)[0] if needs_bias else None,
        )


def _interim_input(x, weight, bias, *, logical_keys=None, parameter_id=None):
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    return _InterimInput.apply(flat, weight, bias).view(*x.shape[:-1], weight.shape[0])


def _input_projection(device: torch.device) -> tuple[Callable[..., torch.Tensor], dict[str, str]]:
    """``proj_in``: its RFC #420 row on SM90, the FP32 K loop it is built on elsewhere."""

    row = "video_input_projection_fp32"
    if torch.cuda.get_device_capability(device) == (9, 0):
        from rl_engine.backends.shared.triton.gemm.video_input_projection_fp32 import (
            TritonVideoInputProjectionFP32Op,
        )

        op = TritonVideoInputProjectionFP32Op()
        return op, {"rfc_row": row, "status": "row", "backend": op.backend_id}
    from rl_engine.backends.shared.triton.gemm import det_gemm

    if not det_gemm._TRITON_AVAILABLE:
        raise RuntimeError("the interim input projection needs Triton")
    backend = "triton _det_gemm_fp32_kernel (FP32)"
    return _interim_input, {"rfc_row": row, "status": "interim", "backend": backend}


def _class_path(op: Any) -> str:
    return f"{type(op).__module__}.{type(op).__qualname__}"


class RowSlicedGemm:
    """A GEMM row run in slices of at most ``max_rows`` rows when it computes no weight gradient.

    ``h3_ffn_down_gemm`` declares at most 32,768 rows (``H3_MAX_ROWS``); two edit requests batched
    are 34,406. Its forward has fixed 64x64x32 tiles, no split-K and a fixed K order, so each output
    row depends on its own input row only, and the rows are run in consecutive slices with the
    matching logical keys. The input gradient is per row as well. A weight gradient is one
    reduction over every row in one canonical backward session, so with ``weight.requires_grad``
    the row is called once and its own limit applies. ``slices`` counts the sliced calls.
    """

    def __init__(self, op: Any, max_rows: int) -> None:
        self.op = op
        self.max_rows = int(max_rows)
        self.slices = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.op, name)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        rows = flat.shape[0]
        if rows <= self.max_rows or (torch.is_grad_enabled() and weight.requires_grad):
            return self.op(x, weight, **kwargs)
        keys = kwargs.pop("logical_keys", None)
        parts = []
        for start in range(0, rows, self.max_rows):
            stop = min(start + self.max_rows, rows)
            part_keys = None if keys is None else keys[start:stop]
            parts.append(self.op(flat[start:stop], weight, logical_keys=part_keys, **kwargs))
        self.slices += 1
        return torch.cat(parts).view(*x.shape[:-1], -1)


def _row_limited(ops: h3_block.CandidateOps) -> h3_block.CandidateOps:
    """``ops`` with the ``ffn_down`` row in ``RowSlicedGemm``; a shallow copy, ``ops`` is kept."""

    down = getattr(ops, "down_gemm", None)
    if down is None or isinstance(down, RowSlicedGemm):
        return ops
    from rl_engine.ops.gemm.h3_ffn_down import H3_MAX_ROWS

    ops = copy.copy(ops)
    setattr(ops, "down_gemm", RowSlicedGemm(down, H3_MAX_ROWS))  # typed None on CandidateOps
    return ops


class StrictImageTransformer:
    """The strict MiniMax-H3-Image transformer on a ``name -> tensor`` mapping (Diffusers names).

    Every operator is resolved at construction, and a CUDA row that resolves to its PyTorch
    reference is refused. ``bindings`` says, per node, which RFC row owns the arithmetic and
    whether the row itself, an interim operator or the reference runs.
    """

    def __init__(
        self,
        weights: Mapping[str, torch.Tensor],
        heads: ph.PDDHeads,
        *,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ops: h3_block.CandidateOps | None = None,
        require_attention_row: bool = False,
    ) -> None:
        from rl_engine.runtime.registry import KernelRegistry

        self.weights = dict(weights)
        self.heads = heads
        self.num_layers = int(num_layers)
        self.num_refiner_layers = int(num_refiner_layers)
        self.device = self.weights["proj_in.weight"].device
        if self.device.type != "cuda":
            raise RuntimeError("the strict transformer runs on CUDA")
        self.ops = _row_limited(
            ops if ops is not None else h3_block.CandidateOps(device=self.device)
        )
        hfm.require_candidate_cuda(self.ops)
        if require_attention_row and self.ops.attention_h3 is None:
            # the interim operator holds an O(S^2) workspace: not usable at released image sizes
            raise hfm.CandidateBindingError(
                "h3_full_attention is not available; build with RL_KERNEL_H3_STRICT=1"
            )
        registry = KernelRegistry()
        self.time_proj = registry.get_op("timestep_sinusoid_h3", device="cuda")
        if type(self.time_proj).__name__ != "H3TimestepSinusoidCudaOp":
            raise hfm.CandidateBindingError("timestep_sinusoid_h3 fell back from its CUDA op")
        self.time_mlp = hfm._cuda_row(registry, "timestep_mlp_fp32")
        self.context = hfm._cuda_row(registry, "text_context_projection")
        self.final_norm = hfm._cuda_row(registry, "final_adaln_out")
        self.pack = registry.get_op("packed_index_copy", device="cuda")
        self.input_projection, input_binding = _input_projection(self.device)
        self.head_projection = ph.head_projection(self.device)
        self._block_weights = [self._block(i) for i in range(self.num_layers)]
        self.bindings = self._bindings(input_binding)

    def _block(self, index: int) -> dict[str, torch.Tensor]:
        prefix = f"{BLOCKS}.{index}."
        found = {
            f"{h3_block.BLOCK}.{name[len(prefix):]}": tensor
            for name, tensor in self.weights.items()
            if name.startswith(prefix)
        }
        missing = [p for p in h3_block.PARAM_LEAVES if f"{h3_block.BLOCK}.{p}" not in found]
        if missing:
            raise ValueError(f"block {index} is missing {missing}")
        return found

    def _bindings(self, input_binding: dict[str, str]) -> dict[str, dict[str, str]]:
        def bound(row, op):
            return {"rfc_row": row, "status": "row", "backend": _class_path(op)}

        out = {
            "time_proj": bound("timestep_sinusoid_h3", self.time_proj),
            "time_embedder": bound("timestep_mlp_fp32", self.time_mlp),
            "proj_in": input_binding,
            "context_embedder": bound("text_context_projection", self.context),
            "packed": bound("packed_index_copy", self.pack),
        }
        graphs = (("block", h3_block.NODES), ("refiner", h3_refiner.NODES))
        for name, nodes in graphs:
            for node, b in h3_block.bindings(self.ops, nodes).items():
                out[f"{name}.{node}"] = {
                    "rfc_row": b.rfc_row,
                    "status": b.status,
                    "backend": b.backend,
                }
        out["norm_out"] = bound("final_adaln_out", self.final_norm)
        out["pdd_proj_out"] = self.head_projection.provenance()
        out["velocity"] = {
            "rfc_row": "image_transformer_output_gather",
            "status": "reference",
            "backend": "torch.Tensor.index_select on canonical image rows",
        }
        return out

    def _w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def _refine(self, hidden: torch.Tensor, trace: StepTrace | None) -> torch.Tensor:
        return h3_refiner.run_refiner(
            "candidate",
            self.weights,
            hidden,
            self.ops,
            num_blocks=self.num_refiner_layers,
            on_block=None if trace is None else lambda i, h: trace.add(f"refiner.{i}", h),
        )

    def _run_block(
        self, index: int, hidden, temb, adaln, rope, trace: StepTrace | None
    ) -> torch.Tensor:
        ctx = dict(
            weights=self._block_weights[index],
            hidden=hidden,
            temb=temb,
            index=adaln,
            rope=rope,
            parameter_prefix=f"{BLOCKS}.{index}",
        )
        values = h3_block.run_graph("candidate", ctx, self.ops)
        if trace is not None and index in trace.detail_blocks:
            for node in h3_block.NODES:
                trace.add(f"block.{index}.{node.name}", values[node.name])
        return values[h3_block.OUTPUT]

    def forward(
        self,
        inputs: StepInputs,
        *,
        trace: StepTrace | None = None,
        checkpoint_blocks: bool = False,
    ) -> torch.Tensor:
        """Velocity ``(B, N_image, 96)`` of every image row, FP32, in selection order.

        Trainable ``proj_in`` / head / block weights need the caller's
        ``h3_full_model._canonical_session`` around forward and backward (see
        :func:`training_step`).
        """

        def record(name: str, value: Any) -> None:
            if trace is not None:
                trace.add(name, value)

        device = self.device
        layout = inputs.layout
        text_indices, image_indices = layout.text_indices.cpu(), layout.image_indices.cpu()
        canonical = torch.cat([text_indices, image_indices])
        seq, num_text = layout.sequence_length, int(text_indices.numel())
        if canonical.numel() != seq:
            raise StepInputError("every packed row must be a text or an image row")
        position_ids = layout.position_ids[canonical].to(device)
        tags = layout.token_tags[canonical].to(device)
        timestep_indices = inputs.timestep_indices.cpu()[canonical].to(device)

        features = self.time_proj(inputs.unique_timesteps.to(device))
        record("time_proj", features)
        temb = self.time_mlp(
            features,
            self._w("time_embedder.linear_1.weight"),
            self._w("time_embedder.linear_1.bias"),
            self._w("time_embedder.linear_2.weight"),
            self._w("time_embedder.linear_2.bias"),
        )
        record("time_embedder", temb)

        rows = inputs.image_rows.to(device)
        image = self.input_projection(
            rows.reshape(-1, rows.shape[-1]).contiguous(),
            self._w("proj_in.weight"),
            self._w("proj_in.bias"),
            logical_keys=h3_block._row_keys(rows),
            parameter_id="proj_in",
        ).view(*rows.shape[:-1], -1)
        record("proj_in", image)
        context_weight = self._w("context_embedder.weight")
        embeds = inputs.prompt_embeds.to(device=device, dtype=context_weight.dtype)
        text = self.context.forward(
            embeds.reshape(-1, embeds.shape[-1]).contiguous(),
            context_weight,
            self._w("context_embedder.bias"),
        ).view(*embeds.shape[:-1], -1)
        record("context_embedder", text)
        text = self._refine(text, trace)
        record("token_refiner", text)

        arange = torch.arange(seq, device=device)
        hidden = self.pack(
            text.contiguous(),
            image.contiguous(),
            image.new_zeros((len(rows), 0, image.shape[-1])),
            arange[:num_text],
            arange[num_text:],
            arange[:0],
            seq,
        )
        record("packed", hidden)

        rope = prov.provider_rope(position_ids)
        adaln = timestep_indices * ps.MODALITY_NUM + tags
        for index in range(self.num_layers):
            if checkpoint_blocks and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                hidden = checkpoint(
                    self._run_block, index, hidden, temb, adaln, rope, None, use_reentrant=False
                )
            else:
                hidden = self._run_block(index, hidden, temb, adaln, rope, trace)
            record(f"block.{index}", hidden)

        hidden = self.final_norm(
            hidden,
            self._w("norm_out.norm.weight"),
            temb,
            self._w("norm_out.linear.weight"),
            self._w("norm_out.linear.bias"),
            timestep_indices,
            NORM_EPS,
        )
        record("norm_out", hidden)
        bound = self.heads.bind(inputs.step, self.head_projection)
        out = bound(
            hidden.float().reshape(-1, hidden.shape[-1]).contiguous(),
            logical_keys=h3_block._row_keys(hidden),
        ).view(*hidden.shape[:-1], ph.HEAD_WIDTH)
        record("pdd_proj_out", out)
        velocity = out.index_select(1, arange[num_text:])
        record("velocity", velocity)
        if trace is not None:
            trace.bindings = self.bindings
            trace.head = bound.artifact()
        return velocity

    def isolated_blocks(
        self, inputs: StepInputs, provider: StepTrace
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Each strict block run on the provider's input to that block (no accumulation)."""

        hidden, temb, adaln, rope = block_context(inputs, provider, self.device)
        with torch.no_grad():
            for index in range(self.num_layers):
                yield f"block.{index}", self._run_block(index, hidden, temb, adaln, rope, None)
                hidden = provider.tensors[f"block.{index}"].to(self.device)

    __call__ = forward


def block_context(inputs: StepInputs, trace: StepTrace, device: torch.device) -> tuple[Any, ...]:
    """``(packed hidden, temb, AdaLN row index, rope)`` of the block stack from a kept trace."""

    layout = inputs.layout
    canonical = torch.cat([layout.text_indices.cpu(), layout.image_indices.cpu()])
    if not torch.equal(canonical, torch.arange(layout.sequence_length)):
        raise StepInputError("the provider's blocks are in physical order: use a canonical layout")
    tags = layout.token_tags.to(device)
    adaln = inputs.timestep_indices.to(device) * ps.MODALITY_NUM + tags
    rope = prov.provider_rope(layout.position_ids.to(device))
    return (
        trace.tensors["packed"].to(device),
        trace.tensors["time_embedder"].to(device),
        adaln,
        rope,
    )


def provider_isolated_blocks(
    model: Any, inputs: StepInputs, provider: StepTrace
) -> Iterable[tuple[str, torch.Tensor]]:
    """Each provider block on the same inputs (run it under another kernel choice to compare)."""

    device = next(model.parameters()).device
    hidden, temb, adaln, rope = block_context(inputs, provider, device)
    with torch.no_grad():
        for index, block in enumerate(model.transformer_blocks):
            yield f"block.{index}", block(hidden, temb, adaln, rope)
            hidden = provider.tensors[f"block.{index}"].to(device)


class StrictImageDenoiser:
    """:class:`StrictImageTransformer` behind ``full_chain.ImageDenoiser``."""

    backend_id = "rl-kernel:StrictImageTransformer"

    def __init__(self, transformer: StrictImageTransformer) -> None:
        self.transformer = transformer

    def __call__(self, image_rows, prompt_embeds, layout, unique_timesteps, timestep_indices, step):
        inputs = StepInputs(
            image_rows[None], prompt_embeds, layout, unique_timesteps, timestep_indices, step
        )
        with torch.no_grad():
            return self.transformer(inputs)


def run_strict(
    transformer: StrictImageTransformer, inputs: StepInputs, **trace_options: Any
) -> tuple[torch.Tensor, StepTrace]:
    """One strict step without autograd, and its trace."""

    trace = StepTrace(**trace_options)
    with torch.no_grad():
        return transformer(inputs, trace=trace), trace


def training_step(
    transformer: StrictImageTransformer,
    inputs: StepInputs,
    trainable: Iterable[str],
    upstream: torch.Tensor,
    *,
    checkpoint_blocks: bool,
    trace: StepTrace | None = None,
) -> dict[str, Any]:
    """One forward and backward of the step.

    ``trainable`` names parameters (``pdd_proj_out.<i>.weight`` included) that get gradients,
    plus ``image_rows`` / ``prompt_embeds`` for the inputs. Returns the velocity and the gradients
    (``None`` where a parameter did not take part).
    """

    trainable = list(trainable)
    leaves: dict[str, torch.Tensor] = {}
    # frozen unless named: a model's parameters may all ask for gradients
    weights = {name: tensor.detach() for name, tensor in transformer.weights.items()}
    for name in trainable:
        if name in ("image_rows", "prompt_embeds"):
            continue
        leaves[name] = weights[name].detach().clone().requires_grad_()
        weights[name] = leaves[name]
    bank = transformer.heads
    heads = ph.PDDHeads(
        [leaves.get(f"pdd_proj_out.{i}.weight", w.detach()) for i, w in enumerate(bank.weights)],
        [leaves.get(f"pdd_proj_out.{i}.bias", b.detach()) for i, b in enumerate(bank.biases)],
        sealed=bank.sealed,
        schedule=bank.schedule,
    )
    rows, embeds = inputs.image_rows, inputs.prompt_embeds
    if "image_rows" in trainable:
        rows = leaves["image_rows"] = rows.detach().clone().requires_grad_()
    if "prompt_embeds" in trainable:
        embeds = leaves["prompt_embeds"] = embeds.detach().float().clone().requires_grad_()
    step_inputs = StepInputs(
        rows,
        embeds,
        inputs.layout,
        inputs.unique_timesteps,
        inputs.timestep_indices,
        inputs.step,
    )
    bound = StrictImageTransformer(
        weights,
        heads,
        num_layers=transformer.num_layers,
        num_refiner_layers=transformer.num_refiner_layers,
        ops=transformer.ops,
    )
    # the keyed FP32 rows reduce their parameter gradients in a canonical session
    with hfm._canonical_session(True):
        velocity = bound(step_inputs, trace=trace, checkpoint_blocks=checkpoint_blocks)
        velocity.backward(upstream.to(velocity))
    return {
        "velocity": velocity.detach(),
        "grads": {name: leaf.grad for name, leaf in leaves.items()},
        "head": bound.heads.select(inputs.step).artifact(),
    }


# --- Provider arm ------------------------------------------------------------------------------


@contextmanager
def _observed(model: Any, trace: StepTrace):
    """Forward hooks that record the provider's nodes under the strict arm's names."""

    handles = []
    started = [False]

    def watch(module, name):
        handles.append(module.register_forward_hook(lambda _m, _a, out: trace.add(name, out)))

    watch(model.time_proj, "time_proj")
    watch(model.time_embedder, "time_embedder")
    watch(model.proj_in, "proj_in")
    watch(model.context_embedder, "context_embedder")
    for i, block in enumerate(model.token_refiner.refiner_blocks):
        watch(block, f"refiner.{i}")
    watch(model.token_refiner, "token_refiner")

    def packed(_module, args):
        if not started[0]:
            started[0] = True
            trace.add("packed", args[0])

    handles.append(model.transformer_blocks[0].register_forward_pre_hook(packed))
    for i, block in enumerate(model.transformer_blocks):
        watch(block, f"block.{i}")
    watch(model.norm_out, "norm_out")
    for head in model.pdd_proj_out:
        watch(head, "pdd_proj_out")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def run_provider(
    model: Any, inputs: StepInputs, **trace_options: Any
) -> tuple[torch.Tensor, StepTrace]:
    """One step of the unmodified Diffusers transformer, with the head that ran read back."""

    device = next(model.parameters()).device
    trace = StepTrace(**trace_options)
    layout = inputs.layout
    readback = ph.HeadReadback(model)
    with torch.no_grad(), _observed(model, trace), readback:
        velocity = model(
            hidden_states=inputs.image_rows.to(device),
            encoder_hidden_states=inputs.prompt_embeds.to(device),
            timestep=inputs.unique_timesteps.to(device),
            timestep_indices=inputs.timestep_indices.to(device),
            token_tags=layout.token_tags.to(device),
            position_ids=layout.position_ids.to(device),
            image_indices=layout.image_indices.to(device),
            text_indices=layout.text_indices.to(device),
            pdd_step_index=inputs.step,
            return_dict=False,
        )[0]
    readback.check(inputs.step)
    trace.add("velocity", velocity)
    # the released forward computes rotary, image, text, then time: bring it to the strict order
    trace.order = _execution_order(trace.order)
    trace.bindings = {name: {"status": "provider"} for name in trace.order}
    trace.head = {"head": inputs.step, "step": inputs.step}
    return velocity, trace


def _execution_order(names: list[str]) -> list[str]:
    """The strict arm's node order (the time nodes first; everything else unchanged)."""

    first = [n for n in ("time_proj", "time_embedder") if n in names]
    return first + [n for n in names if n not in first]


__all__ = [
    "CONTRACT_ID",
    "StepInputError",
    "StepInputs",
    "StepTrace",
    "StrictImageDenoiser",
    "StrictImageTransformer",
    "compare",
    "digest",
    "distance",
    "first_drift",
    "permute_inputs",
    "provider_isolated_blocks",
    "row_group",
    "run_provider",
    "run_strict",
    "stack",
    "training_step",
]
