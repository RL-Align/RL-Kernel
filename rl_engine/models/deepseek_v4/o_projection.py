# SPDX-License-Identifier: Apache-2.0
"""Contract-bound output projection; numerical kernels are upstream library calls."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from rl_engine.contracts.operators.o_projection import OProjectionContract, require
from rl_engine.models.deepseek_v4.indexer_projection import (
    IndexerLogicalRows,
    PreparedIndexerFP8Q,
    _LibraryLinear,
    _register_canonical,
    eager_only,
    tensor_stamp,
)
from rl_engine.ops.autograd.canonical_backward import active_session


@dataclass(frozen=True)
class OProjectionRows:
    logical: IndexerLogicalRows
    cos: torch.Tensor = field(repr=False)
    sin: torch.Tensor = field(repr=False)
    stamps: tuple = field(repr=False)
    head_ids: tuple
    version: str
    contract_fingerprint: str

    @classmethod
    def bind(cls, contract, keys, cos_cache, sin_cache, *, head_ids, version):
        """Admission boundary: own keys and selected FP32 angles, with host validation."""
        require(isinstance(contract, OProjectionContract), "missing output projection contract")
        require(isinstance(version, str) and bool(version), "model version required")
        require(tuple(head_ids) == contract.head_ids, "noncanonical global head order")
        require(
            keys.ndim == 2 and keys.shape[1] == 2 and keys.shape[0] <= contract.max_rows,
            "keys must be [request, global_position] within capacity",
        )
        logical = IndexerLogicalRows.bind(keys)
        host = logical.keys.cpu().tolist()
        require(
            all((r >= 0 and p >= 0) or (r == -1 and p == -1) for r, p in host),
            "padding must be exactly (-1,-1)",
        )
        require(
            cos_cache.ndim == 2
            and cos_cache.shape == sin_cache.shape
            and cos_cache.shape[1] == contract.rope_dim // 2
            and cos_cache.shape[0] > 0,
            "cos/sin cache must be [positions, rope_dim/2]",
        )
        for cache in (cos_cache, sin_cache):
            require(
                cache.dtype == torch.float32
                and cache.device == keys.device
                and not cache.requires_grad,
                "angles must be frozen FP32 on the row device",
            )
        require(all(p < cos_cache.shape[0] for _, p in host), "position exceeds RoPE cache")
        positions = logical.keys[:, 1].clamp_min(0)
        with torch.inference_mode(False):
            cos = (
                cos_cache.detach()
                .index_select(0, positions)
                .clone()
                .view(keys.shape[0], 1, 1, contract.rope_dim // 2)
            )
            sin = sin_cache.detach().index_select(0, positions).clone().view_as(cos)
        require(
            bool(torch.isfinite(cos).all() & torch.isfinite(sin).all()),
            "non-finite selected RoPE angles",
        )
        require(
            bool(
                torch.allclose(
                    cos.square() + sin.square(), torch.ones_like(cos), rtol=1e-5, atol=1e-5
                )
            ),
            "RoPE angles must be unit rotations",
        )
        return cls(
            logical,
            cos,
            sin,
            (tensor_stamp(cos), tensor_stamp(sin)),
            tuple(head_ids),
            version,
            contract.fingerprint,
        )

    def validate(self, contract, rows, device, version):
        require(
            self.contract_fingerprint == contract.fingerprint
            and self.head_ids == contract.head_ids,
            "row/head contract mismatch",
        )
        require(self.version == version, "row/weight model version barrier mismatch")
        self.logical.validate(rows, device)
        for angles in (self.cos, self.sin):
            require(
                angles.shape == (rows, 1, 1, contract.rope_dim // 2)
                and angles.dtype == torch.float32
                and angles.device == device
                and angles.is_contiguous()
                and not angles.requires_grad,
                "selected RoPE angle ABI mismatch",
            )
        require(
            (tensor_stamp(self.cos), tensor_stamp(self.sin)) == self.stamps,
            "bound RoPE angles were mutated",
        )


@dataclass(frozen=True)
class OProjectionWeights:
    a: torch.Tensor | None = field(repr=False)
    b: torch.Tensor | None = field(repr=False)
    packed_a: tuple = field(repr=False)
    packed_b: PreparedIndexerFP8Q | None = field(repr=False)
    a_masters: tuple = field(repr=False)
    version: str
    contract_fingerprint: str
    device: torch.device
    stamps: tuple = field(repr=False)

    def validate(self, contract):
        require(self.contract_fingerprint == contract.fingerprint, "weight contract mismatch")
        require(
            tuple(None if w is None else tensor_stamp(w) for w in (self.a, self.b)) == self.stamps,
            "master changed; prepare a new weight version",
        )
        for packed, master in zip(self.packed_a, self.a_masters, strict=True):
            packed.validate(
                SimpleNamespace(q_width=contract.o_lora_rank, q_lora_rank=contract.group_width),
                master,
                self.version,
                self.device,
            )
        if self.packed_b is not None:
            self.packed_b.validate(
                SimpleNamespace(
                    q_width=contract.hidden_size,
                    q_lora_rank=contract.local_groups * contract.o_lora_rank,
                ),
                self.b,
                self.version,
                self.device,
            )


class _InverseRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, backend):
        ctx.save_for_backward(cos, sin)
        ctx.backend = backend
        return backend.inverse_rope(x, cos, sin)

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        return (
            ctx.backend.inverse_rope_gradient(grad.to(torch.bfloat16).contiguous(), cos, sin),
            None,
            None,
            None,
        )


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, backend, logical, prefix, packed):
        # x [G,M,K], weight [G,R,K]. A single stable parent owns all group IDs.
        ctx.save_for_backward(x, weight)
        ctx.backend, ctx.logical, ctx.prefix = backend, logical, prefix
        ctx.session, ctx.slots = None, ()
        if weight.requires_grad:
            ctx.session = active_session()
            require(ctx.session is not None, "trainable weights require canonical_backward_session")
            ctx.slots = tuple(
                _register_canonical(ctx.session, f"{prefix}.{g}", weight, logical)
                for g in range(x.shape[0])
            )
        if packed:
            return torch.stack(
                [
                    backend.linear_fp8(x[g], p, output_dtype=torch.bfloat16)
                    for g, p in enumerate(packed)
                ]
            )
        return backend.grouped(x, weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        ctx.logical.validate(x.shape[1], x.device)
        grad = torch.where(
            ctx.logical.keys[None, :, :1] >= 0, grad.to(torch.bfloat16), 0
        ).contiguous()
        dx = ctx.backend.grouped(grad, weight.transpose(1, 2)) if ctx.needs_input_grad[0] else None
        dw = None
        if ctx.needs_input_grad[1]:
            parts = [
                torch.zeros_like(weight[g])
                if slot is None
                else ctx.session.submit_linear(
                    f"{ctx.prefix}.{g}", slot, x[g], grad[g], ctx.backend.weight_gradient
                )
                for g, slot in enumerate(ctx.slots)
            ]
            require(
                all(p is None for p in parts) or all(p is not None for p in parts),
                "inconsistent canonical group completion",
            )
            if parts[0] is not None:
                dw = torch.stack(parts)
        return dx, dw, None, None, None, None


@dataclass(frozen=True)
class OProjectionResult:
    partial: torch.Tensor
    inverse_rope: torch.Tensor
    groups: torch.Tensor
    rows: OProjectionRows
    contract: OProjectionContract
    implementation_fingerprint: str
    layer_id: str


class OProjection:
    def __init__(self, contract, *, layer_id, backend=None, expected_implementation=None):
        require(isinstance(contract, OProjectionContract), "missing output projection contract")
        require(isinstance(layer_id, str) and bool(layer_id), "stable layer ID required")
        if backend is None:
            from rl_engine.backends.rocm.gemm.o_projection import AiterOProjectionBackend

            backend = AiterOProjectionBackend()
        require(backend.backend_id == contract.backend_id, "backend identity mismatch")
        self.contract, self.layer_id, self.backend = contract, layer_id, backend
        identity = {
            "contract": contract.family_fingerprint,
            "provider": backend.provenance,
            "bridge": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_vjp": hashlib.sha256(
                Path(_LibraryLinear.forward.__code__.co_filename).read_bytes()
            ).hexdigest(),
            "canonical": hashlib.sha256(
                Path(active_session.__code__.co_filename).read_bytes()
            ).hexdigest(),
        }
        self.implementation_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        require(
            expected_implementation is None
            or expected_implementation == self.implementation_fingerprint,
            "train/rollout implementation fingerprint mismatch",
        )

    def prepare_weights(self, a, b, *, version, a_checkpoint=None, b_checkpoint=None):
        """Bind optimizer-version masters and/or raw FP8 checkpoint data/scales.

        Each checkpoint is (data, FP32 block scales). A is grouped [G,R,K];
        B is the local column shard [hidden,G*R]. No quantization of masters.
        """
        c = self.contract
        require(isinstance(version, str) and bool(version), "weight version required")
        values = [w for w in (a, b) if w is not None]
        for checkpoint in (a_checkpoint, b_checkpoint):
            if checkpoint is not None:
                require(
                    isinstance(checkpoint, tuple) and len(checkpoint) == 2,
                    "checkpoint must be (FP8 data, FP32 scales)",
                )
                values.extend(checkpoint)
        require(bool(values), "weights are missing")
        device = values[0].device
        require(
            device.type == self.backend.device_type
            and device == getattr(self.backend, "device", device),
            "backend device mismatch",
        )
        eager_only(device)
        for w in values:
            require(w.device == device, "weight device mismatch")
            require(not w.is_inference(), "bind versioned tensors, not inference-mode tensors")
        for name, w, shape, precision, checkpoint in (
            (
                "wo_a",
                a,
                (c.local_groups, c.o_lora_rank, c.group_width),
                c.a_precision,
                a_checkpoint,
            ),
            (
                "wo_b",
                b,
                (c.hidden_size, c.local_groups * c.o_lora_rank),
                c.b_precision,
                b_checkpoint,
            ),
        ):
            if w is not None:
                require(
                    w.shape == shape and w.dtype == torch.bfloat16 and w.is_contiguous(),
                    f"{name} master must be contiguous BF16 {shape}",
                )
                require(
                    getattr(w, "_fp8_source_scales", None) is None,
                    "FP8 source cannot be silently treated as BF16",
                )
            if precision == "bf16":
                require(w is not None and checkpoint is None, f"{name} BF16 weights required")
            else:
                require(
                    checkpoint is not None and checkpoint[0].shape == shape,
                    f"{name} native FP8 checkpoint required",
                )
                expected_scales = tuple(shape[:-2]) + (shape[-2] // 128, shape[-1] // 128)
                require(checkpoint[1].shape == expected_scales, f"{name} scale geometry mismatch")
        masters, packs = (), ()
        if a_checkpoint is not None:
            # Stable views are owned for the lifetime of this optimizer version.
            masters = tuple(None if a is None else a[g] for g in range(c.local_groups))
            packs = tuple(
                PreparedIndexerFP8Q.from_checkpoint(
                    a_checkpoint[0][g],
                    a_checkpoint[1][g],
                    version=version,
                    backend=self.backend,
                    master=masters[g],
                )
                for g in range(c.local_groups)
            )
        packed_b = (
            None
            if b_checkpoint is None
            else PreparedIndexerFP8Q.from_checkpoint(
                *b_checkpoint, version=version, backend=self.backend, master=b
            )
        )
        return OProjectionWeights(
            a,
            b,
            packs,
            packed_b,
            masters,
            version,
            c.fingerprint,
            device,
            tuple(None if w is None else tensor_stamp(w) for w in (a, b)),
        )

    def __call__(self, x, weights, *, rows):
        c, backend = self.contract, self.backend
        require(
            isinstance(weights, OProjectionWeights) and isinstance(rows, OProjectionRows),
            "prepared weights and bound rows required",
        )
        eager_only(x.device)
        require(
            x.ndim == 3
            and x.shape[1:] == (c.local_heads, c.head_dim)
            and x.shape[0] <= c.max_rows
            and x.dtype == torch.bfloat16
            and x.is_contiguous(),
            "attention output must be contiguous BF16 [M,local_heads,head_dim]",
        )
        require(
            x.device == weights.device
            and x.device.type == backend.device_type
            and x.device == getattr(backend, "device", x.device),
            "input/backend device mismatch",
        )
        weights.validate(c)
        rows.validate(c, x.shape[0], x.device, weights.version)
        train = torch.is_grad_enabled()
        for w in (weights.a, weights.b):
            require(
                not (train and w is not None and w.requires_grad) or active_session() is not None,
                "trainable weights require canonical_backward_session",
            )
        a_grad = train and (x.requires_grad or (weights.a is not None and weights.a.requires_grad))
        b_grad = train and (a_grad or (weights.b is not None and weights.b.requires_grad))
        require(not a_grad or weights.a is not None, "wo_a VJP requires BF16 master")
        require(not b_grad or weights.b is not None, "wo_b VJP requires BF16 master")
        mask = rows.logical.keys[:, 0] >= 0
        x = torch.where(mask[:, None, None], x, 0)
        inverse = _InverseRoPE.apply(x, rows.cos, rows.sin, backend)
        grouped = (
            inverse.reshape(x.shape[0], c.local_groups, c.group_width).transpose(0, 1).contiguous()
        )
        if a_grad:
            z = _GroupedLinear.apply(
                grouped,
                weights.a,
                backend,
                rows.logical,
                f"{self.layer_id}.o.tp{c.tp_rank}.a",
                weights.packed_a,
            )
        elif weights.packed_a:
            z = torch.stack(
                [
                    backend.linear_fp8(grouped[g], p, output_dtype=torch.bfloat16)
                    for g, p in enumerate(weights.packed_a)
                ]
            )
        else:
            z = backend.grouped(grouped, weights.a)
        z = z.transpose(0, 1).contiguous()
        flat = z.reshape(x.shape[0], c.local_groups * c.o_lora_rank)
        if b_grad:
            partial = _LibraryLinear.apply(
                flat,
                weights.b,
                backend,
                "o_b",
                torch.float32,
                rows.logical,
                f"{self.layer_id}.o.tp{c.tp_rank}.b",
                weights.packed_b,
            )
        elif weights.packed_b is not None:
            partial = backend.linear_fp8(flat, weights.packed_b, output_dtype=torch.float32)
        else:
            partial = backend.linear(flat, weights.b, output_dtype=torch.float32, role="o_b")
        require(
            inverse.dtype == z.dtype == torch.bfloat16 and partial.dtype == torch.float32,
            "provider output dtype mismatch",
        )
        partial = torch.where(mask[:, None], partial, 0)
        return OProjectionResult(
            partial, inverse, z, rows, c, self.implementation_fingerprint, self.layer_id
        )


def merge_tp_partials(partials):
    """Local P4 boundary/reference: increasing-rank FP32 sum, not RCCL transport.

    Fixed TP partition is part of the arithmetic contract. This does not promise
    equal bytes across different TP sizes or an arbitrary all-reduce tree.
    """
    require(bool(partials), "TP partials are missing")
    first = partials[0]
    require(isinstance(first, OProjectionResult), "expected output projection results")
    c = first.contract
    require(len(partials) == c.tp_size, "incomplete TP group")
    require(all(isinstance(p, OProjectionResult) for p in partials), "invalid TP result")
    require(
        sorted(p.contract.tp_rank for p in partials) == list(range(c.tp_size)),
        "duplicate/missing TP rank",
    )
    eager_only(first.partial.device)
    for p in partials:
        require(
            p.contract.family_fingerprint == c.family_fingerprint
            and p.implementation_fingerprint == first.implementation_fingerprint
            and p.layer_id == first.layer_id,
            "TP arithmetic/layer mismatch",
        )
        require(
            p.partial.ndim == 2
            and p.partial.shape[1] == c.hidden_size
            and p.partial.shape == first.partial.shape
            and p.partial.dtype == torch.float32
            and p.partial.device == first.partial.device,
            "TP output ABI mismatch",
        )
        p.rows.validate(p.contract, p.partial.shape[0], p.partial.device, first.rows.version)
        require(
            torch.equal(p.rows.logical.keys, first.rows.logical.keys), "TP logical row mismatch"
        )
        require(
            torch.equal(p.rows.cos, first.rows.cos) and torch.equal(p.rows.sin, first.rows.sin),
            "TP RoPE mismatch",
        )
    ordered = sorted(partials, key=lambda p: p.contract.tp_rank)
    total = ordered[0].partial.clone()
    for p in ordered[1:]:
        total = torch.add(total, p.partial)
    return total
