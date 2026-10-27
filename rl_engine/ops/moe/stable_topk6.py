# SPDX-License-Identifier: Apache-2.0
"""Stable Top-6 selection: score descending, tied score global expert ID ascending."""

from dataclasses import dataclass
import hashlib

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.stable_topk6 import execution_record, implementation
from rl_engine.contracts.operators.stable_topk6 import MAX_EXPERTS, MIN_EXPERTS


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def _validate(scores, global_expert_ids, global_expert_count):
    if not isinstance(scores, torch.Tensor) or scores.dtype != torch.float32:
        raise TypeError("stable Top-6 requires FP32 selection scores; no implicit casts")
    if (
        type(global_expert_count) is not int
        or not MIN_EXPERTS <= global_expert_count <= MAX_EXPERTS
    ):
        raise ValueError("global_expert_count must be an integer in [6,32767]")
    if scores.ndim != 2 or scores.shape[1] != global_expert_count:
        raise ValueError("stable Top-6 requires complete global scores [M,global_expert_count]")
    if not scores.is_cuda or torch.version.hip is None:
        raise RuntimeError("stable Top-6 requires a ROCm GPU")
    with torch.cuda.device(scores.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("stable Top-6 does not support Graph capture")
    arch = torch.cuda.get_device_properties(scores.device).gcnArchName.partition(":")[0]
    if arch not in {"gfx942", "gfx950"}:
        raise RuntimeError(f"unsupported stable Top-6 architecture: {arch}")
    if any(stride <= 0 for stride in scores.stride()):
        raise ValueError("stable Top-6 requires positive score strides")
    if not torch.isfinite(scores).all().item():
        raise ValueError("stable Top-6 selection scores must be finite")
    if global_expert_ids is None:
        ids = torch.arange(global_expert_count, dtype=torch.int64, device=scores.device)
        source_dtype = "canonical"
    else:
        if not isinstance(global_expert_ids, torch.Tensor) or global_expert_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("global_expert_ids must be I32/I64")
        if global_expert_ids.device != scores.device:
            raise RuntimeError("selection scores and global IDs must share one ROCm GPU")
        if global_expert_ids.ndim != 1 or global_expert_ids.shape[0] != global_expert_count:
            raise ValueError("one global expert ID is required per complete score column")
        if global_expert_ids.stride(0) <= 0:
            raise ValueError("global expert IDs require a positive stride")
        # Snapshot the caller-owned map before validation, range checks and narrowing.
        ids = global_expert_ids.detach().to(torch.int64).clone().contiguous()
        source_dtype = str(global_expert_ids.dtype)
        if not torch.equal(
            ids.sort().values, torch.arange(global_expert_count, device=scores.device)
        ):
            raise ValueError("global IDs must cover [0,global_expert_count) exactly once")
    return ids, source_dtype


class _StableTopK6Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, ids):
        api, _, _, _ = implementation()
        with torch.cuda.device(scores.device):
            values, global_ids, columns = api.stable_topk6_fwd(scores, ids)
        ctx.save_for_backward(scores, values, global_ids, columns)
        ctx.mark_non_differentiable(global_ids, columns)
        return values, global_ids, columns

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_values, _grad_ids, _grad_columns):
        scores, _values, _ids, columns = ctx.saved_tensors
        if not torch.isfinite(grad_values).all().item():
            raise ValueError("stable Top-6 requires finite upstream value gradients")
        api, _, _, _ = implementation()
        with torch.cuda.device(scores.device):
            return api.stable_topk6_bwd(columns, grad_values, scores.shape[1]), None


@dataclass(frozen=True)
class StableTopK6Result:
    selected_scores: torch.Tensor
    expert_ids: torch.Tensor
    column_indices: torch.Tensor
    execution: dict


def stable_topk6(
    selection_scores, *, model_version, global_expert_ids=None, global_expert_count=256
):
    """Select complete global scores; preserve order as routing slots.

    The caller owns score transforms, bias, transport and version barriers.
    A local shard is not a global selection and is explicitly rejected.
    """
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("stable Top-6 requires a synchronized model_version")
    ids, source_dtype = _validate(selection_scores, global_expert_ids, global_expert_count)
    record = execution_record(selection_scores)
    record.update(
        {
            "model_version": model_version,
            "global_expert_count": global_expert_count,
            "global_expert_ids_sha256": tensor_sha256(ids),
            "global_ids_source_dtype": source_dtype,
            "complete_global_coverage": True,
            "grad_enabled": torch.is_grad_enabled(),
            "input_requires_grad": selection_scores.requires_grad,
            "input_materialized_contiguous": not selection_scores.is_contiguous(),
            "slot_order": "selection_score_order",
        }
    )
    values, selected_ids, columns = _StableTopK6Fn.apply(selection_scores.contiguous(), ids)
    return StableTopK6Result(values, selected_ids, columns, record)


class StableTopK6Op:
    def __call__(self, selection_scores, **kwargs):
        return stable_topk6(selection_scores, **kwargs)
