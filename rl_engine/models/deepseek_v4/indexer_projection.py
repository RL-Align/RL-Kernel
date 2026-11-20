# SPDX-License-Identifier: Apache-2.0
"""Item 13: reuse aiter/vLLM arithmetic and RLK's canonical backward session."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.indexer_projection import (
    IndexerProjectionContract,
    IndexerProjectionError,
)
from rl_engine.ops.autograd.canonical_backward import active_session


def require(value, message):
    if not value:
        raise IndexerProjectionError(message)


def tensor_stamp(value):
    try:
        version = value._version
    except RuntimeError:  # Framework inference parameters may lack version counters.
        version = None
    return (value.data_ptr(), version, tuple(value.shape), tuple(value.stride()), value.dtype)


def eager_only(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            require(not torch.cuda.is_current_stream_capturing(), "Graph capture is not validated")


@dataclass(frozen=True)
class IndexerLogicalRows:
    """Bind once at the scheduler boundary; numerical forward performs no readback."""

    keys: torch.Tensor = field(repr=False)
    active_keys: tuple
    stamp: tuple = field(repr=False)

    @classmethod
    def bind(cls, keys):
        require(
            keys.dtype == torch.int64 and keys.ndim == 2 and keys.shape[1] in (2, 3),
            "logical keys must be int64 [M,2 or 3]",
        )
        eager_only(keys.device)
        with torch.inference_mode(False):
            snapshot = keys.detach().clone().contiguous()
        active = tuple(tuple(row) for row in snapshot.cpu().tolist() if row[0] >= 0)
        require(all(all(v >= 0 for v in row) for row in active), "negative active logical key")
        require(len(set(active)) == len(active), "duplicate logical token identity")
        return cls(snapshot, active, tensor_stamp(snapshot))

    def validate(self, rows, device):
        require(self.keys.shape[0] == rows and self.keys.device == device, "logical row mismatch")
        require(tensor_stamp(self.keys) == self.stamp, "bound logical keys were mutated")


@dataclass(frozen=True)
class PreparedIndexerFP8Q:
    """Native loader-owned 128x128 FP8 blocks, optionally with a BF16 training master.

    Preparation clones checkpoint storage before vLLM's in-place zero normalization.
    It does not quantize/reinterpret a BF16 master as if it were the checkpoint.
    """

    data: torch.Tensor = field(repr=False)
    scales: torch.Tensor = field(repr=False)
    version: str
    source_encoding: str
    master: torch.Tensor | None = field(repr=False)
    source: torch.Tensor = field(repr=False)
    source_scales: torch.Tensor = field(repr=False)
    stamps: tuple = field(repr=False)

    @classmethod
    def from_checkpoint(cls, data, scales, *, version, backend, master=None):
        require(
            isinstance(version, str) and bool(version), "explicit packed-weight version required"
        )
        require(
            data.ndim == 2
            and data.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            and data.shape[0] % 128 == 0
            and data.shape[1] % 128 == 0,
            "packed Q must be an unshuffled OCP/FNUZ [N,K] matrix with 128-aligned dimensions",
        )
        require(
            scales.dtype == torch.float32
            and scales.shape == (data.shape[0] // 128, data.shape[1] // 128)
            and scales.device == data.device,
            "scale ABI is unpacked FP32 [N/128,K/128], not packed exponent bytes",
        )
        require(
            not data.requires_grad and not scales.requires_grad,
            "packed data/scales must be frozen; train through the explicit BF16 master",
        )
        eager_only(data.device)
        require(bool(torch.isfinite(data.float()).all()), "non-finite checkpoint FP8 values")
        require(bool((torch.isfinite(scales) & (scales > 0)).all()), "invalid checkpoint scales")
        exponents = scales.log2()
        require(bool((exponents == exponents.round()).all()), "expected power-of-two UE8M0 scales")
        if master is not None:
            require(
                master.dtype == torch.bfloat16
                and master.shape == data.shape
                and master.device == data.device,
                "training master must be BF16 and match the packed checkpoint geometry",
            )
        source_encoding = str(data.dtype)
        with torch.inference_mode(False), torch.no_grad():
            packed, scale = data.detach().clone().contiguous(), scales.detach().clone().contiguous()
            if packed.dtype == torch.float8_e4m3fn:
                packed, scale, _ = backend.normalize_checkpoint(packed, scale)
            require(packed.dtype == torch.float8_e4m3fnuz, "normalizer returned the wrong encoding")
            require(bool(torch.isfinite(scale).all()), "normalized scale overflow")
        stamps = (
            tensor_stamp(packed),
            tensor_stamp(scale),
            None if master is None else tensor_stamp(master),
            tensor_stamp(data),
            tensor_stamp(scales),
        )
        return cls(packed, scale, version, source_encoding, master, data, scales, stamps)

    def validate(self, contract, master, version, device):
        require(version == self.version, "packed Q weight/scale version barrier mismatch")
        require(master is self.master, "packed Q is bound to a different training master")
        require(
            self.data.shape == (contract.q_width, contract.q_lora_rank), "packed Q shape mismatch"
        )
        require(self.data.device == device, "packed Q device mismatch")
        current = (
            tensor_stamp(self.data),
            tensor_stamp(self.scales),
            None if master is None else tensor_stamp(master),
            tensor_stamp(self.source),
            tensor_stamp(self.source_scales),
        )
        require(current == self.stamps, "packed Q/master changed; prepare a new version")


def _register_canonical(session, prefix, weight, logical_rows):
    # Reuse the existing aggregation implementation. Add only identity validation.
    bindings = getattr(session, "_indexer_projection_bindings", None)
    if bindings is None:
        bindings = {}
        session._indexer_projection_bindings = bindings
    entry = bindings.setdefault(prefix, (weight, set()))
    require(entry[0] is weight, "canonical parameter ID is bound to another weight")
    require(
        not entry[1].intersection(logical_rows.active_keys),
        "logical token repeated across microbatches for the same parameter",
    )
    entry[1].update(logical_rows.active_keys)
    if not logical_rows.active_keys:
        return None
    return session.register(prefix, logical_rows.keys)


class _LibraryLinear(torch.autograd.Function):
    """VJP plumbing only: all three GEMMs are upstream operators.

    Match mLite's BF16-master rule: cast dY to BF16 before the GEMMs, then
    return BF16 input/master gradients. This is explicit even for FP32 outputs.
    Parameter gradients reuse the native canonical session, not per-chunk sums.
    """

    @staticmethod
    def forward(ctx, x, weight, backend, role, output_dtype, logical_rows, parameter_id, packed):
        ctx.save_for_backward(x, weight)
        ctx.backend, ctx.logical_rows = backend, logical_rows
        ctx.parameter_id, ctx.slot, ctx.session = parameter_id, None, None
        if weight.requires_grad:
            ctx.session = active_session()
            require(ctx.session is not None, "trainable weights require canonical_backward_session")
            require(logical_rows is not None, "trainable weights require bound logical rows")
            ctx.slot = _register_canonical(ctx.session, parameter_id, weight, logical_rows)
        if packed is not None:
            return backend.linear_fp8(x, packed, output_dtype=output_dtype)
        return backend.linear(x, weight, output_dtype=output_dtype, role=role)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.to(torch.bfloat16).contiguous()
        if ctx.logical_rows is not None:
            ctx.logical_rows.validate(x.shape[0], x.device)
            grad = torch.where(ctx.logical_rows.keys[:, :1] >= 0, grad, 0)
        dx = ctx.backend.input_gradient(grad, weight) if ctx.needs_input_grad[0] else None
        dw = None
        if ctx.needs_input_grad[1]:
            if ctx.slot is None:
                dw = torch.zeros_like(weight)
            else:
                dw = ctx.session.submit_linear(
                    ctx.parameter_id,
                    ctx.slot,
                    x,
                    grad,
                    ctx.backend.weight_gradient,
                )
        return dx, dw, None, None, None, None, None, None


@dataclass(frozen=True)
class IndexerProjectionResult:
    q: torch.Tensor
    weights: torch.Tensor
    contract_fingerprint: str
    implementation_fingerprint: str


class IndexerProjection:
    def __init__(
        self,
        contract: IndexerProjectionContract,
        *,
        layer_id: str,
        backend=None,
        expected_implementation: str | None = None,
    ):
        require(isinstance(contract, IndexerProjectionContract), "missing projection contract")
        require(isinstance(layer_id, str) and bool(layer_id), "layer_id must be explicit")
        if backend is None:
            from rl_engine.backends.rocm.gemm.indexer_projection import (
                AiterIndexerProjectionBackend,
            )

            backend = AiterIndexerProjectionBackend()
        require(
            backend.backend_id == contract.backend_id, "backend identity does not match contract"
        )
        self.contract, self.layer_id, self.backend = contract, layer_id, backend
        self._fingerprint = contract.fingerprint
        identity = {
            "contract": self._fingerprint,
            "provider": getattr(
                backend, "provenance", {"test_provider": type(backend).__qualname__}
            ),
            "bridge_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "canonical_source_sha256": hashlib.sha256(
                Path(active_session.__code__.co_filename).read_bytes(),
            ).hexdigest(),
        }
        if hasattr(backend, "plan"):
            identity["q_plan"] = backend.plan(
                contract.q_lora_rank,
                contract.q_width,
                "q",
                contract.q_precision,
            ).fingerprint
            identity["weights_plan"] = backend.plan(
                contract.hidden_size,
                contract.num_heads,
                "weights",
            ).fingerprint
        self.implementation_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode(),
        ).hexdigest()
        if expected_implementation is not None:
            require(
                self.implementation_fingerprint == expected_implementation,
                "train/rollout implementation fingerprint mismatch",
            )

    @staticmethod
    def _validate_matrix(value, shape, device, name):
        require(
            isinstance(value, torch.Tensor)
            and tuple(value.shape) == shape
            and value.dtype == torch.bfloat16
            and value.device == device
            and value.is_contiguous(),
            f"{name}: require contiguous BF16 {shape} on {device}",
        )
        require(
            getattr(value, "_fp8_source_scales", None) is None,
            f"{name}: FP8 source metadata cannot be silently treated as plain BF16",
        )

    def _validate_rows(self, x, width, name, logical_rows):
        require(x.ndim == 2, "inputs must be packed 2-D rows")
        require(x.shape[0] <= self.contract.max_rows, "batch exceeds contract capacity")
        require(x.device.type == self.backend.device_type, "input device does not match backend")
        if hasattr(self.backend, "device"):
            require(x.device == self.backend.device, "create a backend for the input device")
        eager_only(x.device)
        self._validate_matrix(x, (x.shape[0], width), x.device, name)
        if logical_rows is not None:
            logical_rows.validate(x.shape[0], x.device)

    def _validate_q(self, qr, q_weight, logical_rows, packed_q, q_weight_version):
        c = self.contract
        self._validate_rows(qr, c.q_lora_rank, "qr", logical_rows)
        if c.q_precision == "bf16":
            require(packed_q is None, "packed Q supplied to a BF16-only contract")
            self._validate_matrix(q_weight, (c.q_width, c.q_lora_rank), qr.device, "q_weight")
        else:
            require(
                isinstance(packed_q, PreparedIndexerFP8Q), "block-FP8 Q requires prepared weights"
            )
            packed_q.validate(c, q_weight, q_weight_version, qr.device)
            if q_weight is not None:
                require(
                    q_weight.dtype == torch.bfloat16 and q_weight.is_contiguous(),
                    "FP8 training master must be contiguous BF16",
                )
        needs_q_grad = torch.is_grad_enabled() and (
            qr.requires_grad or (q_weight is not None and q_weight.requires_grad)
        )
        require(
            not needs_q_grad or q_weight is not None, "FP8 VJP requires the BF16 training master"
        )
        self._validate_trainable(q_weight, logical_rows)

    @staticmethod
    def _validate_trainable(weight, logical_rows):
        if torch.is_grad_enabled() and weight is not None and weight.requires_grad:
            require(
                logical_rows is not None and active_session() is not None,
                "trainable weights require logical rows and canonical_backward_session",
            )

    def _validate_weights(self, hidden, weight, logical_rows):
        c = self.contract
        self._validate_rows(hidden, c.hidden_size, "hidden", logical_rows)
        self._validate_matrix(weight, (c.num_heads, c.hidden_size), hidden.device, "weights_weight")
        self._validate_trainable(weight, logical_rows)

    def _project(self, x, weight, role, output, logical_rows, pack=None):
        dtype = torch.bfloat16 if output == "bf16" else torch.float32
        needs_grad = torch.is_grad_enabled() and (
            x.requires_grad or (weight is not None and weight.requires_grad)
        )
        if needs_grad:
            out = _LibraryLinear.apply(
                x,
                weight,
                self.backend,
                role,
                dtype,
                logical_rows,
                f"{self.layer_id}.indexer.{role}",
                pack,
            )
        elif pack is not None:
            out = self.backend.linear_fp8(x, pack, output_dtype=dtype)
        else:
            out = self.backend.linear(x, weight, output_dtype=dtype, role=role)
        require(out.dtype == dtype, f"{role} backend output dtype violates contract")
        return out

    def project_q(self, qr, weight, *, logical_rows=None, packed_q=None, q_weight_version=None):
        """Independent native-stream hook; does not serialize weights_proj behind Q."""
        self._validate_q(qr, weight, logical_rows, packed_q, q_weight_version)
        q = self._project(qr, weight, "q", self.contract.q_output, logical_rows, packed_q)
        return q.view(qr.shape[0], self.contract.num_heads, self.contract.head_dim)

    def project_weights(self, hidden, weight, *, logical_rows=None):
        self._validate_weights(hidden, weight, logical_rows)
        return self._project(hidden, weight, "weights", self.contract.weights_output, logical_rows)

    def __call__(
        self,
        qr,
        hidden,
        q_weight,
        weights_weight,
        *,
        logical_rows=None,
        packed_q=None,
        q_weight_version=None,
    ):
        self._validate_q(qr, q_weight, logical_rows, packed_q, q_weight_version)
        self._validate_weights(hidden, weights_weight, logical_rows)
        require(
            qr.shape[0] == hidden.shape[0] and qr.device == hidden.device,
            "Q and weight projection rows/devices must agree",
        )
        q = self._project(qr, q_weight, "q", self.contract.q_output, logical_rows, packed_q)
        weights = self._project(
            hidden,
            weights_weight,
            "weights",
            self.contract.weights_output,
            logical_rows,
        )
        return IndexerProjectionResult(
            q.view(qr.shape[0], self.contract.num_heads, self.contract.head_dim),
            weights,
            self._fingerprint,
            self.implementation_fingerprint,
        )
