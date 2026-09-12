from __future__ import annotations

import os
import pathlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.integrations.ablation import operator_ablation_case
from rl_engine.kernels.logprob_contract import (
    LogprobContract,
    LogprobDType,
    LogprobRole,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import DEFAULT_NUM_VOCAB_TILES


_ALIGNMENT_DIAGNOSTIC_LOCK = threading.Lock()
_ALIGNMENT_DIAGNOSTIC_CALLS = 0


def _alignment_diagnostics_enabled() -> bool:
    return os.getenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _alignment_diagnostic_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except (RuntimeError, ValueError):
        pass
    try:
        return int(os.getenv("RANK", "0"))
    except ValueError:
        return 0


def _alignment_tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    flat = detached.reshape(-1)
    first = flat[:16].to(device="cpu")
    last = flat[-16:].to(device="cpu") if flat.numel() else first
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "first16": first.tolist(),
        "last16": last.tolist(),
    }
    if detached.numel():
        values = detached.float()
        summary.update(
            {
                "sum_fp32": float(values.sum().item()),
                "abs_sum_fp32": float(values.abs().sum().item()),
                "min_fp32": float(values.min().item()),
                "max_fp32": float(values.max().item()),
            }
        )
    return summary


def _record_alignment_diagnostics(
    *,
    target: str,
    local_logits: torch.Tensor,
    target_ids: torch.Tensor,
    result: torch.Tensor,
    lse: torch.Tensor,
    vocab_start_index: int,
    hidden: torch.Tensor | None = None,
    lm_head_weight: torch.Tensor | None = None,
) -> None:
    """Persist bounded row diagnostics only when explicitly enabled.

    This branch performs device-to-host copies and reductions for debugging;
    production runs leave it cold and do no additional tensor work.
    """

    if not _alignment_diagnostics_enabled():
        return
    root = os.getenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR", "").strip()
    if not root:
        return
    global _ALIGNMENT_DIAGNOSTIC_CALLS
    with _ALIGNMENT_DIAGNOSTIC_LOCK:
        call_index = _ALIGNMENT_DIAGNOSTIC_CALLS
        _ALIGNMENT_DIAGNOSTIC_CALLS += 1
    detached_logits = local_logits.detach()
    ids = target_ids.detach().to(device="cpu", dtype=torch.int64)
    local_ids = target_ids.detach().to(dtype=torch.long) - int(vocab_start_index)
    local_mask = (local_ids >= 0) & (local_ids < detached_logits.size(1))
    safe_ids = local_ids.clamp(0, max(int(detached_logits.size(1)) - 1, 0))
    selected_local = detached_logits.gather(1, safe_ids.reshape(-1, 1)).reshape(-1)
    selected_local = torch.where(
        local_mask,
        selected_local,
        torch.full_like(selected_local, float("nan")),
    )
    row_sum = detached_logits.float().sum(dim=1)
    row_abs_sum = detached_logits.float().abs().sum(dim=1)
    payload = {
        "schema_version": "rlkernel.linear_logp_alignment_diagnostic.v1",
        "target": target,
        "rank": _alignment_diagnostic_rank(),
        "pid": os.getpid(),
        "call_index": call_index,
        "vocab_start_index": int(vocab_start_index),
        "target_ids": ids,
        "selected_local_logits": selected_local.cpu(),
        "result": result.detach().cpu(),
        "lse": lse.detach().cpu(),
        "row_sum_fp32": row_sum.cpu(),
        "row_abs_sum_fp32": row_abs_sum.cpu(),
        "logits_head": detached_logits[:, :8].cpu(),
        "logits_tail": detached_logits[:, -8:].cpu(),
        "logits_summary": _alignment_tensor_summary(detached_logits),
    }
    if hidden is not None:
        payload["hidden"] = hidden.detach().cpu()
        payload["hidden_summary"] = _alignment_tensor_summary(hidden)
    if lm_head_weight is not None:
        detached_weight = lm_head_weight.detach()
        payload["lm_head_weight_head"] = detached_weight[:8].cpu()
        payload["lm_head_weight_tail"] = detached_weight[-8:].cpu()
        payload["lm_head_weight_summary"] = _alignment_tensor_summary(detached_weight)
    output_dir = pathlib.Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"{target}-pid{os.getpid()}-rank{_alignment_diagnostic_rank():05d}-"
        f"call{call_index:08d}.pt"
    )
    torch.save(payload, path)


@dataclass(frozen=True)
class RolloutLinearLogpContext:
    """The exact hidden/LM-head contract for the immediately following sample."""

    hidden: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_bias: torch.Tensor | None
    tp_group: Any
    vocab_start_index: int
    global_vocab_size: int
    real_vocab_size: int


_ROLLOUT_CONTEXT: RolloutLinearLogpContext | None = None


def publish_rollout_linear_logp_context(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    *,
    tp_group: Any,
    vocab_start_index: int,
    global_vocab_size: int,
    real_vocab_size: int,
) -> None:
    """Publish one model-output context for the next vLLM sampler invocation."""

    global _ROLLOUT_CONTEXT
    if hidden.ndim != 2:
        raise ValueError(
            f"rollout linear_logp hidden must be [tokens, hidden], got {tuple(hidden.shape)}"
        )
    if lm_head_weight.ndim != 2:
        raise ValueError("rollout linear_logp LM-head weight must be [vocab_local, hidden]")
    if hidden.device != lm_head_weight.device:
        raise ValueError("rollout linear_logp hidden and LM-head must share a device")
    if hidden.size(1) != lm_head_weight.size(1):
        raise ValueError("rollout linear_logp hidden and LM-head hidden widths must match")
    if lm_head_bias is not None and (
        lm_head_bias.ndim != 1
        or lm_head_bias.size(0) != lm_head_weight.size(0)
        or lm_head_bias.device != hidden.device
    ):
        raise ValueError("rollout linear_logp bias must match the complete local LM-head shard")
    if int(global_vocab_size) <= 0 or int(real_vocab_size) <= 0:
        raise ValueError("rollout linear_logp vocab sizes must be positive")
    if int(real_vocab_size) > int(global_vocab_size):
        raise ValueError("rollout linear_logp real vocab cannot exceed padded global vocab")
    _ROLLOUT_CONTEXT = RolloutLinearLogpContext(
        hidden=hidden,
        lm_head_weight=lm_head_weight,
        lm_head_bias=lm_head_bias,
        tp_group=tp_group,
        vocab_start_index=int(vocab_start_index),
        global_vocab_size=int(global_vocab_size),
        real_vocab_size=int(real_vocab_size),
    )


def clear_rollout_linear_logp_context() -> None:
    """Discard a context when vLLM computes logits without sampling from them."""

    global _ROLLOUT_CONTEXT
    _ROLLOUT_CONTEXT = None


def take_rollout_linear_logp_context() -> RolloutLinearLogpContext:
    """Consume the context published by the matching ``compute_logits`` call."""

    global _ROLLOUT_CONTEXT
    context = _ROLLOUT_CONTEXT
    _ROLLOUT_CONTEXT = None
    if context is None:
        raise RuntimeError(
            "strict rollout linear_logp sampler has no matching hidden/LM-head context"
        )
    return context


class LinearLogpWrapper:
    """PR230 integration adapter for the existing TP-aware linear_logp op."""

    # This is deliberately a different backend from the existing high-
    # performance linear_logp op.  PR1 exposes these strict entry points
    # without changing the old registry route.
    backend_id = "rlkernel.linear_logp.bitwise.v1"

    def __init__(self) -> None:
        self._op: Any | None = None
        self._tp_op: Any | None = None
        self._rocm_op: Any | None = None
        self._rocm_linear: Any | None = None
        self._last_provenance: dict[str, Any] = {}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self._last_provenance)

    def _resolve(self, *, tensor_parallel: bool) -> Any:
        if tensor_parallel and self._tp_op is not None:
            return self._tp_op
        if not tensor_parallel and self._op is not None:
            return self._op
        try:
            from rl_engine.kernels.ops.cuda.loss.linear_logp import (
                sm90_deterministic_linear_logp,
                sm90_deterministic_linear_logp_tp,
            )
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "strict bitwise linear_logp requires PR1's separate "
                "sm90_deterministic_linear_logp entry points"
            ) from exc
        op = (
            sm90_deterministic_linear_logp_tp if tensor_parallel else sm90_deterministic_linear_logp
        )
        # This wrapper is the strict boundary. Fail closed if a refactor ever
        # swaps in the native/provider logp path under the same API.
        expected_name = (
            "sm90_deterministic_linear_logp_tp"
            if tensor_parallel
            else "sm90_deterministic_linear_logp"
        )
        if getattr(op, "__name__", "") != expected_name:
            raise RuntimeError(
                "strict linear_logp resolved a non-deterministic implementation: "
                f"{getattr(op, '__name__', op)!r}"
            )
        if tensor_parallel:
            self._tp_op = op
        else:
            self._op = op
        return op

    def _resolve_rocm(self) -> Any:
        if self._rocm_op is None:
            from rl_engine.kernels.ops.rocm.loss.vocab_parallel_logp import (
                RocmVocabParallelLogprobOp,
            )

            self._rocm_op = RocmVocabParallelLogprobOp()
        return self._rocm_op

    def _resolve_rocm_linear(self) -> Any:
        if self._rocm_linear is None:
            from rl_engine.kernels.ops.matmul.det_gemm import DetGemmOp

            self._rocm_linear = DetGemmOp()
        return self._rocm_linear

    @staticmethod
    def _tp_coordinates(tp_group: Any) -> tuple[int, int]:
        if tp_group is None:
            return 0, 1
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("strict TP linear_logp requires initialized torch.distributed")
        return (
            int(torch.distributed.get_rank(group=tp_group)),
            int(torch.distributed.get_world_size(group=tp_group)),
        )

    @staticmethod
    def _temperature_tensor(
        temperature: float | torch.Tensor | None, *, rows: int, device: torch.device
    ) -> torch.Tensor | None:
        if temperature is None:
            return None
        if isinstance(temperature, torch.Tensor):
            value = temperature.to(device=device, dtype=torch.float32).reshape(-1)
            if value.numel() == 1:
                value = value.expand(rows).contiguous()
            elif value.numel() == rows:
                value = value.contiguous()
            else:
                raise ValueError(
                    f"temperature must be scalar or one value per row, got {value.numel()}"
                )
            positive = torch.all(value > 0)
            if positive.is_cuda:
                torch._assert_async(positive, "temperature must be positive")
            elif not bool(positive):
                raise ValueError("temperature must be positive")
            return value
        scalar = float(temperature)
        if scalar <= 0.0:
            raise ValueError(f"temperature must be positive, got {scalar}")
        return torch.full((rows,), scalar, device=device, dtype=torch.float32)

    @staticmethod
    def _validate_targets(target_ids: torch.Tensor, *, rows: int, real_vocab_size: int) -> None:
        if target_ids.ndim != 1 or target_ids.numel() != rows:
            raise ValueError(
                f"linear_logp target_ids must be [rows]={rows}, got {tuple(target_ids.shape)}"
            )
        if target_ids.is_floating_point() or target_ids.is_complex():
            raise TypeError("linear_logp target_ids must use an integer dtype")
        if target_ids.numel():
            target_long = target_ids.to(dtype=torch.long)
            valid = torch.all((target_long >= 0) & (target_long < real_vocab_size))
            if valid.is_cuda:
                torch._assert_async(
                    valid,
                    f"linear_logp target_ids must be in [0, {real_vocab_size})",
                )
            elif not bool(valid):
                raise ValueError(
                    f"linear_logp target_ids must be in [0, {real_vocab_size}), got invalid ids"
                )

    @classmethod
    def _validate_contract(
        cls,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        tp_group: Any,
        vocab_start_index: int,
        global_vocab_size: int | None,
        real_vocab_size: int | None,
    ) -> tuple[bool, int, int]:
        if hidden.ndim != 2:
            raise ValueError(
                f"linear_logp wrapper expects hidden [tokens, hidden], got {tuple(hidden.shape)}"
            )
        if not hidden.is_cuda:
            raise RuntimeError("strict linear_logp requires CUDA/ROCm GPU tensors")
        if lm_head_weight.ndim != 2:
            raise ValueError("linear_logp LM-head weight must be [vocab_local, hidden]")
        if hidden.dtype != torch.bfloat16 or lm_head_weight.dtype != torch.bfloat16:
            raise TypeError("strict linear_logp requires bfloat16 hidden and LM-head")
        if lm_head_weight.device != hidden.device:
            raise ValueError("linear_logp hidden and LM-head must share a device")
        if hidden.size(1) != lm_head_weight.size(1):
            raise ValueError("linear_logp hidden width must match LM-head width")
        if bias is not None and (
            bias.ndim != 1 or bias.size(0) != lm_head_weight.size(0) or bias.device != hidden.device
        ):
            raise ValueError("linear_logp bias must match the complete local LM-head shard")

        local_vocab = int(lm_head_weight.size(0))
        rank, world = cls._tp_coordinates(tp_group)
        tensor_parallel = tp_group is not None or int(vocab_start_index) != 0
        requested_global = (
            local_vocab * world
            if global_vocab_size is None and tensor_parallel
            else local_vocab if global_vocab_size is None else int(global_vocab_size)
        )
        if requested_global <= 0:
            raise ValueError("linear_logp global_vocab_size must be positive")
        if tensor_parallel:
            if tp_group is None:
                raise ValueError("TP linear_logp requires an explicit TP process group")
            if requested_global != local_vocab * world:
                raise ValueError(
                    "TP linear_logp requires complete equal shards: "
                    f"local={local_vocab}, world={world}, global={requested_global}"
                )
            expected_start = rank * local_vocab
            if int(vocab_start_index) != expected_start:
                raise ValueError(
                    "TP linear_logp vocab_start_index does not match rank-local shard: "
                    f"got {vocab_start_index}, expected {expected_start}"
                )
        elif requested_global != local_vocab or int(vocab_start_index) != 0:
            raise ValueError("non-TP linear_logp must use the complete local vocab at offset zero")
        real = requested_global if real_vocab_size is None else int(real_vocab_size)
        if real <= 0 or real > requested_global:
            raise ValueError(
                f"linear_logp real_vocab_size must be in [1, {requested_global}], got {real}"
            )
        cls._validate_targets(target_ids, rows=hidden.size(0), real_vocab_size=real)
        return tensor_parallel, requested_global, real

    @staticmethod
    def _rocm_contract(
        local_logits: torch.Tensor,
        *,
        rank: int,
        world: int,
        global_vocab_size: int,
        real_vocab_size: int,
        target: str,
    ) -> LogprobContract:
        local_vocab = int(local_logits.size(1))
        dtype = {
            torch.bfloat16: LogprobDType.BF16,
            torch.float16: LogprobDType.FP16,
            torch.float32: LogprobDType.FP32,
        }.get(local_logits.dtype)
        if dtype is None:
            raise TypeError(f"unsupported ROCm linear_logp dtype {local_logits.dtype}")
        role = LogprobRole.TRAIN if target == "training" else LogprobRole.INFER
        return LogprobContract(
            role=role,
            dtype=dtype,
            mask=MaskSpec(
                num_tokens=int(local_logits.size(0)),
                active_mask=(True,) * int(local_logits.size(0)),
            ),
            sharding=ShardingSpec(
                tp_rank=rank,
                tp_world_size=world,
                vocab_shard_bounds=tuple(
                    (index * local_vocab, (index + 1) * local_vocab) for index in range(world)
                ),
                real_vocab_size=real_vocab_size,
                padded_vocab_size=global_vocab_size,
            ),
            reduction=ReductionSpec(),
        )

    def _rocm_from_local_logits(
        self,
        local_logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        tp_group: Any,
        rank: int,
        world: int,
        global_vocab_size: int,
        real_vocab_size: int,
        target: str,
        temperature: float | torch.Tensor | None,
        diagnostics_hidden: torch.Tensor | None = None,
        diagnostics_lm_head_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        if temperature is None or (
            not isinstance(temperature, torch.Tensor) and float(temperature) == 1.0
        ):
            effective_logits = local_logits.contiguous()
            # Megatron exposes the materialized TP LM-head through a squeezed
            # view.  The loss path may reuse that storage while autograd still
            # needs it for the logp backward.  Keep one gradient-preserving
            # snapshot for the strict training kernel; rollout is inference
            # only and remains zero-copy.
            if target == "training" and effective_logits.requires_grad:
                effective_logits = effective_logits.clone()
        else:
            temperature_tensor = self._temperature_tensor(
                temperature,
                rows=local_logits.size(0),
                device=local_logits.device,
            )
            assert temperature_tensor is not None
            effective_logits = local_logits.float() / temperature_tensor.unsqueeze(1)
        contract = self._rocm_contract(
            effective_logits,
            rank=rank,
            world=world,
            global_vocab_size=global_vocab_size,
            real_vocab_size=real_vocab_size,
            target=target,
        )
        op = self._resolve_rocm()
        result, lse = op.apply(
            effective_logits,
            target_ids,
            contract=contract,
            tp_group=tp_group,
            num_vocab_tiles=DEFAULT_NUM_VOCAB_TILES,
            deterministic=True,
        )
        _record_alignment_diagnostics(
            target=target,
            local_logits=effective_logits,
            target_ids=target_ids,
            result=result,
            lse=lse,
            vocab_start_index=rank * int(effective_logits.size(1)),
            hidden=diagnostics_hidden,
            lm_head_weight=diagnostics_lm_head_weight,
        )
        return result, lse, str(op.backend_id)

    def from_local_logits(
        self,
        local_logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        tp_group: Any,
        vocab_start_index: int,
        global_vocab_size: int,
        real_vocab_size: int,
        target: str = "rollout",
        temperature: float | torch.Tensor | None = None,
        return_lse: bool = False,
        diagnostics_hidden: torch.Tensor | None = None,
        diagnostics_lm_head_weight: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Score deterministic local LM-head logits without a duplicate GEMM."""

        if local_logits.ndim != 2 or local_logits.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise TypeError("strict reused LM-head logits must be 2-D bf16/fp16/fp32")
        if not local_logits.is_cuda:
            raise RuntimeError("strict reused LM-head logits require CUDA/ROCm")
        if target_ids.device != local_logits.device:
            raise ValueError("linear_logp target_ids must share the logits device")
        rank, world = self._tp_coordinates(tp_group)
        local_vocab = int(local_logits.size(1))
        requested_global = int(global_vocab_size)
        expected_global = local_vocab * world
        expected_start = rank * local_vocab
        if requested_global != expected_global or int(vocab_start_index) != expected_start:
            raise ValueError("reused rollout LM-head logits must use equal rank-ordered TP shards")
        real = int(real_vocab_size)
        if not 0 < real <= requested_global:
            raise ValueError("real_vocab_size must be within the padded global vocabulary")
        self._validate_targets(
            target_ids,
            rows=local_logits.size(0),
            real_vocab_size=real,
        )
        if torch.version.hip is not None:
            result, lse, kernel_backend = self._rocm_from_local_logits(
                local_logits,
                target_ids,
                tp_group=tp_group,
                rank=rank,
                world=world,
                global_vocab_size=requested_global,
                real_vocab_size=real,
                target=target,
                temperature=temperature,
                diagnostics_hidden=diagnostics_hidden,
                diagnostics_lm_head_weight=diagnostics_lm_head_weight,
            )
            runtime_platform = "rocm"
            strict_entrypoint = (
                "rocm_deterministic_linear_logp_tp"
                if target == "training"
                else "rocm_vocab_parallel_logp_from_local_logits_tp"
            )
            contract_version = "rocm-det-gemm-vocab-parallel-logp-ws2-v1"
        else:
            temperature_tensor = self._temperature_tensor(
                temperature,
                rows=local_logits.size(0),
                device=local_logits.device,
            )
            from rl_engine.kernels.ops.cuda.loss.linear_logp import (
                sm90_deterministic_logp_from_local_logits_tp,
            )

            result, lse = sm90_deterministic_logp_from_local_logits_tp(
                local_logits.contiguous(),
                target_ids,
                tp_group=tp_group,
                vocab_start_index=int(vocab_start_index),
                global_vocab_size=requested_global,
                real_vocab_size=real,
                temperature=temperature_tensor,
            )
            kernel_backend = self.backend_id
            runtime_platform = "cuda"
            strict_entrypoint = "sm90_deterministic_logp_from_local_logits_tp"
            contract_version = "cuda-det-gemm-linear-logp-sm90-contract-v2"
        self._last_provenance = {
            **self._mismatch_provenance(),
            "target": target,
            "runtime_platform": runtime_platform,
            "triton_used": runtime_platform == "rocm",
            "actual_backend": self.backend_id,
            "logprob_kernel_backend": kernel_backend,
            "deterministic_linear_logp": True,
            "strict_entrypoint": strict_entrypoint,
            "local_logits_shape": list(local_logits.shape),
            "target_shape": list(target_ids.shape),
            "tp_group_present": True,
            "vocab_start_index": int(vocab_start_index),
            "global_vocab_size": requested_global,
            "real_vocab_size": real,
            "temperature": None if temperature is None else "provided",
            "contract_version": contract_version,
            "logits_materialized": True,
            "lm_head_result_reused": True,
        }
        return (result, lse) if return_lse else result

    @staticmethod
    def _mismatch_provenance() -> dict[str, Any]:
        case_id = os.getenv("RL_KERNEL_LOGP_CASE", "P/P").strip().upper()
        case = operator_ablation_case("logp", case_id)
        cross_engine_mismatch = case.training is not case.rollout
        return {
            "module": "logp",
            "case_id": case.case_id,
            "training_implementation": case.training.value,
            "rollout_implementation": case.rollout.value,
            "mismatch_axes": {
                # PR230 L1-L3 axes. R/R closes all of them by sharing the
                # padded TP LM-head and the rank-ordered LSE merge.
                "vocab_shard_ownership": cross_engine_mismatch,
                "selected_token_identity": cross_engine_mismatch,
                "vocab_lse_reduction": cross_engine_mismatch,
            },
            "arithmetic_contract": {
                "batch_invariant": True,
                "fixed_k_reduction": True,
                "vocab_group_width": 64,
                "summary_merge_tree": "rank_ordered_fixed",
                "vocab_reduction_axis": "TP",
                "cp_is_merge_axis": False,
                "output_dtype": "fp32",
            },
        }

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: int | None = None,
        real_vocab_size: int | None = None,
        target: str = "training",
        temperature: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_ids.device != hidden.device:
            raise ValueError("linear_logp target_ids must share hidden device")
        tensor_parallel, requested_global_vocab, real = self._validate_contract(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            tp_group=tp_group,
            vocab_start_index=int(vocab_start_index),
            global_vocab_size=global_vocab_size,
            real_vocab_size=real_vocab_size,
        )
        effective_weight = lm_head_weight
        effective_bias = bias
        if torch.version.hip is not None:
            local_logits = self._resolve_rocm_linear().linear(hidden, effective_weight)
            if effective_bias is not None:
                local_logits = local_logits + effective_bias
            result, _lse, kernel_backend = self._rocm_from_local_logits(
                local_logits,
                target_ids,
                tp_group=tp_group,
                rank=self._tp_coordinates(tp_group)[0],
                world=self._tp_coordinates(tp_group)[1],
                global_vocab_size=requested_global_vocab,
                real_vocab_size=real,
                target=target,
                temperature=temperature,
                diagnostics_hidden=hidden,
                diagnostics_lm_head_weight=effective_weight,
            )
            runtime_platform = "rocm"
            strict_entrypoint = "rocm_deterministic_linear_logp_tp"
            contract_version = "rocm-det-gemm-vocab-parallel-logp-ws2-v1"
        else:
            temperature_tensor = self._temperature_tensor(
                temperature, rows=hidden.size(0), device=hidden.device
            )
            op = self._resolve(tensor_parallel=tensor_parallel)
            if tensor_parallel:
                result, _lse = op(
                    hidden,
                    effective_weight,
                    target_ids,
                    effective_bias,
                    tp_group=tp_group,
                    vocab_start_index=int(vocab_start_index),
                    global_vocab_size=requested_global_vocab,
                    real_vocab_size=real,
                    temperature=temperature_tensor,
                )
            else:
                result, _lse = op(
                    hidden,
                    effective_weight,
                    target_ids,
                    effective_bias,
                    real_vocab_size=real,
                    temperature=temperature_tensor,
                )
            kernel_backend = self.backend_id
            runtime_platform = "cuda"
            strict_entrypoint = (
                "sm90_deterministic_linear_logp_tp"
                if tensor_parallel
                else "sm90_deterministic_linear_logp"
            )
            contract_version = "cuda-det-gemm-linear-logp-sm90-contract-v2"
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("linear_logp backend returned a non-tensor result")
        if result.numel() != hidden.size(0):
            raise RuntimeError(
                "linear_logp backend returned an unexpected token count: "
                f"{result.numel()} != {hidden.size(0)}"
            )

        actual_backend = self.backend_id
        self._last_provenance = {
            **self._mismatch_provenance(),
            "target": target,
            "runtime_platform": runtime_platform,
            "triton_used": runtime_platform == "rocm",
            "actual_backend": actual_backend,
            "logprob_kernel_backend": kernel_backend,
            "deterministic_linear_logp": True,
            "strict_entrypoint": strict_entrypoint,
            "hidden_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype).replace("torch.", ""),
            "lm_head_weight_shape": list(lm_head_weight.shape),
            "target_shape": list(target_ids.shape),
            "tp_group_present": tp_group is not None,
            "vocab_start_index": int(vocab_start_index),
            "global_vocab_size": requested_global_vocab,
            "requested_global_vocab_size": requested_global_vocab,
            "real_vocab_size": real,
            "requested_real_vocab_size": None if real_vocab_size is None else int(real_vocab_size),
            "temperature": None if temperature is None else "provided",
            "contract_version": contract_version,
            "logits_materialized": True,
        }
        return result


__all__ = [
    "LinearLogpWrapper",
    "RolloutLinearLogpContext",
    "clear_rollout_linear_logp_context",
    "publish_rollout_linear_logp_context",
    "take_rollout_linear_logp_context",
]
