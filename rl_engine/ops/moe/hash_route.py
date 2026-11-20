# SPDX-License-Identifier: Apache-2.0
"""Absolute-layer Hash lookup with fixed table identity and preserved slot order."""

from dataclasses import dataclass
import hashlib
import math

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.hash_route import execution_record, implementation
from rl_engine.contracts.operators.hash_route import EXPERTS, HASH_LAYERS, SLOTS


def tensor_sha256(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(raw).hexdigest()


def _version(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Hash route requires a nonempty {label}")


@dataclass(frozen=True)
class HashRouteTable:
    absolute_layer: int
    hash_layers: tuple
    model_version: str
    table_version: str
    source_dtype: str
    source_sha256: str
    prepared_sha256: str
    vocabulary_size: int
    _prepared: torch.Tensor

    def verify(self):
        # A data.copy_ reload can bypass Tensor._version. Verify content, not
        # just a version counter or data_ptr, before trusting any lookup.
        if (
            self._prepared.dtype != torch.int32
            or (tuple(self._prepared.shape) != (self.vocabulary_size, SLOTS))
            or not self._prepared.is_contiguous()
        ):
            raise ValueError("Hash route prepared table layout changed; prepare a new version")
        if tensor_sha256(self._prepared) != self.prepared_sha256:
            raise ValueError("Hash route table content changed; prepare a new version")


def prepare_hash_route_table(
    tid2eid,
    *,
    absolute_layer,
    model_version,
    table_version,
    expected_sha256,
    hash_layers=HASH_LAYERS,
):
    """Own a detached I32 device snapshot; preserve I64 checkpoint source identity."""
    _version(model_version, "model_version")
    _version(table_version, "table_version")
    if (
        not isinstance(hash_layers, tuple)
        or not hash_layers
        or any(type(layer) is not int or layer < 0 for layer in hash_layers)
        or len(set(hash_layers)) != len(hash_layers)
    ):
        raise ValueError("hash_layers must be distinct nonnegative absolute layer numbers")
    if type(absolute_layer) is not int or absolute_layer not in hash_layers:
        raise ValueError("absolute layer is not an explicitly configured Hash layer")
    if not isinstance(tid2eid, torch.Tensor) or tid2eid.dtype not in (torch.int32, torch.int64):
        raise TypeError("tid2eid must be an integer I32/I64 tensor")
    if tid2eid.ndim != 2 or tid2eid.shape[1] != SLOTS or not 0 < tid2eid.shape[0] < 2**31:
        raise ValueError("tid2eid must have shape [positive vocabulary,6]")
    if not tid2eid.is_cuda or torch.version.hip is None:
        raise RuntimeError("tid2eid must be prepared on a ROCm GPU")
    with torch.cuda.device(tid2eid.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Hash table preparation does not support Graph capture")
    # Snapshot before validating so caller mutations after preparation cannot
    # change the derived table. Range validation precedes narrowing to I32.
    source = tid2eid.detach().clone().contiguous()
    if tensor_sha256(source) != expected_sha256:
        raise ValueError("tid2eid checksum does not match the expected table identity")
    if ((source < 0) | (source >= EXPERTS)).any().item():
        raise ValueError("tid2eid global expert IDs must be in [0,256)")
    sorted_rows = source.sort(dim=1).values
    if (sorted_rows[:, 1:] == sorted_rows[:, :-1]).any().item():
        raise ValueError("tid2eid rows must contain six distinct global experts")
    prepared = source.to(torch.int32).clone()
    return HashRouteTable(
        absolute_layer,
        hash_layers,
        model_version,
        table_version,
        str(source.dtype),
        expected_sha256,
        tensor_sha256(prepared),
        source.shape[0],
        prepared,
    )


class _HashRouteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, token_ids, table, renormalize, scale):
        api, _ = implementation()
        with torch.cuda.device(logits.device):
            if logits.shape[0]:
                weights, ids = api.hash_route_fwd(logits, table, token_ids, renormalize, scale)
            else:
                weights = torch.empty((0, SLOTS), dtype=torch.float32, device=logits.device)
                ids = torch.empty((0, SLOTS), dtype=torch.int64, device=logits.device)
        if not torch.isfinite(weights).all().item():
            raise ValueError("Hash route weights overflowed FP32")
        ctx.save_for_backward(logits, ids)
        ctx.renormalize, ctx.scale = renormalize, scale
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_weights, _grad_ids):
        logits, ids = ctx.saved_tensors
        if not torch.isfinite(grad_weights).all().item():
            raise ValueError("Hash route requires finite upstream weight gradients")
        api, _ = implementation()
        with torch.cuda.device(logits.device):
            dx = api.hash_route_bwd(logits, ids, grad_weights, ctx.renormalize, ctx.scale)
        if not torch.isfinite(dx).all().item():
            raise ValueError("Hash route VJP overflowed FP32")
        return dx, None, None, None, None


@dataclass(frozen=True)
class HashRouteResult:
    weights: torch.Tensor
    expert_ids: torch.Tensor
    execution: dict


def hash_route(
    logits,
    token_ids,
    table,
    *,
    absolute_layer,
    model_version,
    table_version,
    renormalize=True,
    routed_scaling_factor=1.5,
):
    """Hash lookup, task-29 scores, ordered normalization, scale and local VJP.

    Slots are tid2eid columns. No sorting, Top-K, bias, capacity or communication.
    """
    if not isinstance(table, HashRouteTable):
        raise TypeError("Hash route requires a prepared HashRouteTable")
    if type(absolute_layer) is not int or absolute_layer != table.absolute_layer:
        raise ValueError("Hash route absolute layer does not match the prepared table")
    if model_version != table.model_version or table_version != table.table_version:
        raise ValueError("Hash route model/table version barrier mismatch")
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32:
        raise TypeError("Hash route requires FP32 gate logits")
    if logits.ndim != 2 or logits.shape[1] != EXPERTS:
        raise ValueError("Hash route logits must have shape [M,256]")
    if not isinstance(token_ids, torch.Tensor) or token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Hash route token_ids must be I32/I64")
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("Hash route needs one logical token ID per logit row")
    if (
        not logits.is_cuda
        or torch.version.hip is None
        or (token_ids.device != logits.device or table._prepared.device != logits.device)
    ):
        raise RuntimeError("Hash route logits, token IDs and table must share one ROCm GPU")
    with torch.cuda.device(logits.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Hash route does not support Graph capture")
    arch = torch.cuda.get_device_properties(logits.device).gcnArchName.partition(":")[0]
    if arch not in {"gfx942", "gfx950"}:
        raise RuntimeError(f"unsupported Hash route architecture: {arch}")
    if any(s <= 0 for tensor in (logits, token_ids) for s in tensor.stride()):
        raise ValueError("Hash route requires positive input strides")
    if type(renormalize) is not bool:
        raise TypeError("renormalize must be a bool")
    if type(routed_scaling_factor) not in (int, float) or not (
        math.isfinite(routed_scaling_factor)
        and 0 <= routed_scaling_factor <= torch.finfo(torch.float32).max
    ):
        raise ValueError("routing scale must be finite, nonnegative and representable in FP32")
    if not torch.isfinite(logits).all().item():
        raise ValueError("Hash route requires finite logits")
    if ((token_ids < 0) | (token_ids >= table.vocabulary_size)).any().item():
        raise ValueError("Hash route token ID outside table vocabulary; IDs are never clamped")
    table.verify()
    record = execution_record(logits)
    record.update(
        {
            "absolute_layer": absolute_layer,
            "hash_layers": list(table.hash_layers),
            "model_version": model_version,
            "table_version": table_version,
            "table_source_sha256": table.source_sha256,
            "table_source_dtype": table.source_dtype,
            "prepared_table_sha256": table.prepared_sha256,
            "table_shape": list(table._prepared.shape),
            "token_ids_dtype": str(token_ids.dtype),
            "token_ids_stride": list(token_ids.stride()),
            "renormalize": renormalize,
            "routed_scaling_factor": float(routed_scaling_factor),
            "input_materialized_contiguous": not logits.is_contiguous(),
        }
    )
    weights, ids = _HashRouteFn.apply(
        logits.contiguous(),
        token_ids.to(torch.int32).contiguous(),
        table._prepared,
        renormalize,
        float(routed_scaling_factor),
    )
    return HashRouteResult(weights, ids, record)


class HashRouteOp:
    def __call__(self, *args, **kwargs):
        return hash_route(*args, **kwargs)
