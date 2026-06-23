# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import inspect
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Literal, Mapping, Optional

import torch

from rl_engine.testing.reference_ops import selected_logprobs_reference

StatelessForwardMode = Literal["reference", "reward", "both"]
StatelessAttentionBackend = Literal["flash_attention_2", "sdpa", "eager", "model_default"]
RewardAdapter = Callable[["StatelessForwardOutputs", "StatelessForwardInputs"], torch.Tensor]
_MISSING = object()


@dataclass(frozen=True)
class StatelessForwardConfig:
    """Configuration for no-cache reference/reward model scoring."""

    mode: StatelessForwardMode = "both"
    use_cache: bool = False
    attention_backend: StatelessAttentionBackend = "flash_attention_2"
    reject_kv_cache_outputs: bool = True
    detach_outputs: bool = True
    return_token_scores: bool = False
    max_batch_size: Optional[int] = None
    temperature: float = 1.0
    output_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.mode not in {"reference", "reward", "both"}:
            raise ValueError("mode must be 'reference', 'reward', or 'both'")
        if self.use_cache:
            raise ValueError("StatelessForwardConfig.use_cache must be False")
        if self.attention_backend not in {"flash_attention_2", "sdpa", "eager", "model_default"}:
            raise ValueError(
                "attention_backend must be 'flash_attention_2', 'sdpa', 'eager', "
                "or 'model_default'"
            )
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")


@dataclass(frozen=True)
class StatelessForwardInputs:
    """Dense full-sequence batch for stateless scoring."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class StatelessForwardOutputs:
    """Normalized model outputs consumed by scoring adapters."""

    raw: Any
    logits: Optional[torch.Tensor]
    kv_cache: Optional[Any] = None


@dataclass(frozen=True)
class StatelessForwardResult:
    """Scoring tensors and scalar metrics produced by the stateless executor."""

    reference_logps: Optional[torch.Tensor]
    rewards: Optional[torch.Tensor]
    token_scores: Optional[torch.Tensor]
    metrics: Mapping[str, float | int | str | bool]


@dataclass(frozen=True)
class TensorTreeSummary:
    """Count tensor payloads in nested optional runtime outputs."""

    tensor_count: int
    total_bytes: int

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1_048_576.0


class StatelessForwardExecutor:
    """
    Lightweight Reference/Reward scoring wrapper.

    The executor runs one full-sequence forward pass with ``use_cache=False``
    whenever the wrapped model accepts that argument. It never calls a
    generation loop and it does not instantiate a paged-KV serving runtime.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[StatelessForwardConfig] = None,
        *,
        reward_adapter: Optional[RewardAdapter] = None,
    ):
        self.model = model
        self.config = config or StatelessForwardConfig()
        self.reward_adapter = reward_adapter or default_reward_adapter

    def score(self, inputs: StatelessForwardInputs) -> StatelessForwardResult:
        _validate_inputs(inputs, self.config)

        device = inputs.input_ids.device
        cuda_tracking = device.type == "cuda" and torch.cuda.is_available()
        if cuda_tracking:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        started_at = time.perf_counter()
        with _temporarily_configure_stateless_model(self.model, self.config) as no_cache_policy:
            with torch.no_grad():
                try:
                    raw_outputs, use_cache_passed = _run_no_cache_forward(self.model, inputs)
                    no_cache_policy["attention_backend_fallback"] = False
                except Exception as exc:
                    if not _should_fallback_attention_backend(exc, self.config):
                        raise
                    fallback_config = replace(self.config, attention_backend="eager")
                    with _temporarily_configure_stateless_model(
                        self.model,
                        fallback_config,
                    ) as fallback_policy:
                        raw_outputs, use_cache_passed = _run_no_cache_forward(self.model, inputs)
                    no_cache_policy.update(fallback_policy)
                    no_cache_policy.update(
                        {
                            "attention_backend_requested": self.config.attention_backend,
                            "attention_backend_fallback": True,
                            "attention_backend_fallback_reason": (
                                f"{type(exc).__name__}: {str(exc)[:200]}"
                            ),
                        }
                    )
                kv_cache = extract_kv_cache_outputs(raw_outputs)
                kv_cache_summary = summarize_tensor_tree(kv_cache)
                if self.config.reject_kv_cache_outputs and kv_cache is not None:
                    raise ValueError(
                        "stateless scoring received KV-cache outputs despite use_cache=False "
                        f"({kv_cache_summary.tensor_count} tensors, "
                        f"{kv_cache_summary.total_mb:.4f} MiB)."
                    )
                outputs = StatelessForwardOutputs(
                    raw=raw_outputs,
                    logits=_extract_logits(raw_outputs),
                    kv_cache=kv_cache,
                )

                reference_logps: Optional[torch.Tensor] = None
                rewards: Optional[torch.Tensor] = None
                token_scores: Optional[torch.Tensor] = None

                if self.config.mode in {"reference", "both"}:
                    if outputs.logits is None:
                        raise ValueError("reference mode requires model outputs to expose logits")
                    reference_logps = score_reference_logprobs(
                        outputs.logits,
                        inputs,
                        temperature=self.config.temperature,
                        output_dtype=self.config.output_dtype,
                    )
                    if self.config.return_token_scores:
                        token_scores = reference_logps

                if self.config.mode in {"reward", "both"}:
                    rewards = score_rewards(
                        outputs,
                        inputs,
                        reward_adapter=self.reward_adapter,
                        output_dtype=self.config.output_dtype,
                    )

                if self.config.detach_outputs:
                    reference_logps = _detach_optional(reference_logps)
                    rewards = _detach_optional(rewards)
                    token_scores = _detach_optional(token_scores)

        if cuda_tracking:
            torch.cuda.synchronize(device)
        finished_at = time.perf_counter()

        metrics = collect_stateless_metrics(
            inputs,
            mode=self.config.mode,
            elapsed_seconds=finished_at - started_at,
            use_cache_passed=use_cache_passed,
            detached_outputs=self.config.detach_outputs,
            cuda_tracking=cuda_tracking,
            no_cache_policy=no_cache_policy,
            kv_cache_summary=kv_cache_summary,
        )
        return StatelessForwardResult(
            reference_logps=reference_logps,
            rewards=rewards,
            token_scores=token_scores,
            metrics=metrics,
        )


def score_reference_logprobs(
    logits: torch.Tensor,
    inputs: StatelessForwardInputs,
    *,
    temperature: float = 1.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute causal next-token selected logprobs aligned to ``[B, S]`` masks."""

    if logits.ndim != 3:
        raise ValueError(f"reference logits must have shape [B, S, V], got {tuple(logits.shape)}")
    if logits.shape[:2] != inputs.input_ids.shape:
        raise ValueError(
            "reference logits leading shape must match input_ids shape, got "
            f"{tuple(logits.shape[:2])} and {tuple(inputs.input_ids.shape)}"
        )

    labels = inputs.labels if inputs.labels is not None else inputs.input_ids
    if labels.shape != inputs.input_ids.shape:
        raise ValueError("labels shape must match input_ids shape")
    if _completion_mask_starts_at_position_zero(inputs.completion_mask, device=logits.device):
        raise ValueError(
            "completion_mask[:, 0] must be False; completions cannot start at position 0"
        )

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_mask = _bool_mask(inputs.completion_mask[:, 1:], device=logits.device)
    shifted_logps = selected_logprobs_reference(
        shifted_logits,
        shifted_labels.to(device=logits.device),
        mask=shifted_mask,
        temperature=temperature,
        output_dtype=output_dtype,
    )
    result = torch.zeros(
        inputs.input_ids.shape,
        device=logits.device,
        dtype=output_dtype,
    )
    result[:, 1:] = shifted_logps
    return result.masked_fill(~_bool_mask(inputs.completion_mask, device=result.device), 0.0)


def score_rewards(
    outputs: StatelessForwardOutputs,
    inputs: StatelessForwardInputs,
    *,
    reward_adapter: RewardAdapter,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run a reward adapter and validate that it returns one scalar per sequence."""

    rewards = reward_adapter(outputs, inputs)
    if not isinstance(rewards, torch.Tensor):
        raise TypeError("reward_adapter must return a torch.Tensor")
    if rewards.ndim != 1:
        raise ValueError(f"reward_adapter must return shape [B], got {tuple(rewards.shape)}")
    if rewards.shape[0] != inputs.input_ids.shape[0]:
        raise ValueError(
            f"reward_adapter batch size {rewards.shape[0]} must match input batch size "
            f"{inputs.input_ids.shape[0]}"
        )
    return rewards.to(device=inputs.input_ids.device, dtype=output_dtype)


def default_reward_adapter(
    outputs: StatelessForwardOutputs,
    inputs: StatelessForwardInputs,
) -> torch.Tensor:
    """Default adapter for common scalar reward model outputs."""

    del inputs
    reward = _extract_named_tensor(outputs.raw, ("rewards", "reward", "scores", "score"))
    if reward is not None:
        return _squeeze_reward_tensor(reward)

    logits = outputs.logits
    if logits is None:
        raise ValueError("reward mode requires scalar logits or a reward_adapter")
    return _squeeze_reward_tensor(logits)


def collect_stateless_metrics(
    inputs: StatelessForwardInputs,
    *,
    mode: StatelessForwardMode,
    elapsed_seconds: float,
    use_cache_passed: bool,
    detached_outputs: bool,
    cuda_tracking: bool,
    no_cache_policy: Mapping[str, float | int | str | bool],
    kv_cache_summary: TensorTreeSummary,
) -> dict[str, float | int | str | bool]:
    """Collect common scoring metrics without importing optional runtimes."""

    input_ids = inputs.input_ids
    active_tokens = int(_bool_mask(inputs.completion_mask, device=input_ids.device).sum().item())
    metrics: dict[str, float | int | str | bool] = {
        "mode": mode,
        "batch_size": int(input_ids.shape[0]),
        "sequence_len": int(input_ids.shape[1]),
        "active_completion_tokens": active_tokens,
        "device": str(input_ids.device),
        "dtype": str(input_ids.dtype).replace("torch.", ""),
        "elapsed_ms": elapsed_seconds * 1000.0,
        "use_cache": False,
        "use_cache_passed": bool(use_cache_passed),
        "detached_outputs": bool(detached_outputs),
        "zero_kv_cache": kv_cache_summary.tensor_count == 0,
        "kv_cache_output_present": kv_cache_summary.tensor_count > 0,
        "kv_cache_output_tensors": kv_cache_summary.tensor_count,
        "kv_cache_output_bytes": kv_cache_summary.total_bytes,
        "kv_cache_output_mb": kv_cache_summary.total_mb,
        **dict(no_cache_policy),
    }
    if cuda_tracking:
        device = input_ids.device
        metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1_048_576.0
        metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1_048_576.0
    return metrics


def configure_stateless_model(
    model: torch.nn.Module,
    config: StatelessForwardConfig,
) -> dict[str, float | int | str | bool]:
    """
    Apply best-effort no-cache scoring knobs without importing optional runtimes.

    Hugging Face-style models decide attention kernels from their config rather
    than forward kwargs. Setting these attributes keeps the wrapper explicit
    while still allowing plain PyTorch modules to run unchanged.
    """

    use_cache_targets = 0
    attention_targets = 0

    for target in _model_config_targets(model):
        if hasattr(target, "use_cache"):
            target.use_cache = False
            use_cache_targets += 1
        if config.attention_backend != "model_default":
            target.attn_implementation = config.attention_backend
            target._attn_implementation = config.attention_backend
            attention_targets += 1

    return {
        "attention_backend": config.attention_backend,
        "attention_backend_configured": attention_targets > 0,
        "attention_backend_config_targets": attention_targets,
        "model_config_use_cache_disabled": use_cache_targets > 0,
        "model_config_use_cache_targets": use_cache_targets,
        "reject_kv_cache_outputs": config.reject_kv_cache_outputs,
    }


@contextmanager
def _temporarily_configure_stateless_model(
    model: torch.nn.Module,
    config: StatelessForwardConfig,
) -> Iterator[dict[str, float | int | str | bool]]:
    saved = _model_config_snapshot(model, config)
    policy = configure_stateless_model(model, config)
    try:
        yield policy
    finally:
        _restore_model_config_snapshot(saved)


def extract_kv_cache_outputs(raw_outputs: Any) -> Optional[Any]:
    """Extract common cache-bearing output fields from model outputs."""

    names = (
        "past_key_values",
        "past_key_value",
        "presents",
        "present",
        "kv_cache",
        "cache",
    )
    if isinstance(raw_outputs, Mapping):
        for name in names:
            if name in raw_outputs and raw_outputs[name] is not None:
                return raw_outputs[name]
        return None

    for name in names:
        value = getattr(raw_outputs, name, None)
        if value is not None:
            return value

    if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) > 1:
        candidate = raw_outputs[1]
        if _looks_like_kv_cache(candidate):
            return candidate
    return None


def summarize_tensor_tree(value: Any) -> TensorTreeSummary:
    """Count tensors and bytes in nested tuples/lists/mappings/dataclass-like outputs."""

    seen: set[int] = set()

    def visit(node: Any) -> tuple[int, int]:
        if node is None:
            return 0, 0
        node_id = id(node)
        if node_id in seen:
            return 0, 0
        seen.add(node_id)

        if isinstance(node, torch.Tensor):
            return 1, _tensor_nbytes(node)
        if isinstance(node, Mapping):
            counts = [visit(item) for item in node.values()]
        elif isinstance(node, (tuple, list)):
            counts = [visit(item) for item in node]
        elif hasattr(node, "__dict__"):
            counts = [visit(item) for item in vars(node).values()]
        else:
            return 0, 0
        return sum(count for count, _bytes in counts), sum(_bytes for _count, _bytes in counts)

    tensor_count, total_bytes = visit(value)
    return TensorTreeSummary(tensor_count=tensor_count, total_bytes=total_bytes)


def _validate_inputs(inputs: StatelessForwardInputs, config: StatelessForwardConfig) -> None:
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    completion_mask = inputs.completion_mask

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, S], got {tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask shape must match input_ids shape")
    if completion_mask.shape != input_ids.shape:
        raise ValueError("completion_mask shape must match input_ids shape")
    if inputs.labels is not None and inputs.labels.shape != input_ids.shape:
        raise ValueError("labels shape must match input_ids shape")
    if attention_mask.device != input_ids.device:
        raise ValueError("attention_mask device must match input_ids device")
    if completion_mask.device != input_ids.device:
        raise ValueError("completion_mask device must match input_ids device")
    if inputs.labels is not None and inputs.labels.device != input_ids.device:
        raise ValueError("labels device must match input_ids device")
    if config.max_batch_size is not None and input_ids.shape[0] > config.max_batch_size:
        raise ValueError(
            f"batch size {input_ids.shape[0]} exceeds max_batch_size {config.max_batch_size}"
        )
    if config.mode in {"reference", "both"}:
        if input_ids.shape[1] < 2:
            raise ValueError("reference scoring requires sequence_len >= 2")
        if _completion_mask_starts_at_position_zero(completion_mask, device=input_ids.device):
            raise ValueError(
                "completion_mask[:, 0] must be False; completions cannot start at position 0"
            )
    if not bool(_bool_mask(completion_mask, device=input_ids.device).any().item()):
        raise ValueError("completion_mask must contain at least one active token")


def _run_no_cache_forward(
    model: torch.nn.Module,
    inputs: StatelessForwardInputs,
) -> tuple[Any, bool]:
    kwargs: dict[str, Any] = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }
    if _call_accepts_keyword(model, "use_cache"):
        kwargs["use_cache"] = False
        return model(**kwargs), True

    try:
        kwargs["use_cache"] = False
        return model(**kwargs), True
    except TypeError as exc:
        if "use_cache" not in str(exc):
            raise
        kwargs.pop("use_cache", None)
        return model(**kwargs), False


def _should_fallback_attention_backend(
    exc: Exception,
    config: StatelessForwardConfig,
) -> bool:
    if config.attention_backend != "flash_attention_2":
        return False
    if isinstance(exc, (ImportError, ModuleNotFoundError, StopIteration)):
        return True
    message = str(exc).lower()
    markers = (
        "flash_attention",
        "flash attention",
        "flash-attn",
        "flash_attn",
        "flash attention kernels",
        "attn_implementation",
    )
    return any(marker in message for marker in markers)


def _call_accepts_keyword(model: torch.nn.Module, keyword: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True
    return False


def _extract_logits(raw_outputs: Any) -> Optional[torch.Tensor]:
    if isinstance(raw_outputs, torch.Tensor):
        return raw_outputs
    if isinstance(raw_outputs, Mapping):
        value = raw_outputs.get("logits")
        return value if isinstance(value, torch.Tensor) else None
    logits = getattr(raw_outputs, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(raw_outputs, (tuple, list)) and raw_outputs:
        first = raw_outputs[0]
        return first if isinstance(first, torch.Tensor) else None
    return None


def _extract_named_tensor(raw_outputs: Any, names: tuple[str, ...]) -> Optional[torch.Tensor]:
    if isinstance(raw_outputs, Mapping):
        for name in names:
            value = raw_outputs.get(name)
            if isinstance(value, torch.Tensor):
                return value
    for name in names:
        value = getattr(raw_outputs, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _squeeze_reward_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value
    if value.ndim == 2 and value.shape[1] == 1:
        return value[:, 0]
    raise ValueError(f"reward tensor must have shape [B] or [B, 1], got {tuple(value.shape)}")


def _bool_mask(mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    return mask.to(device=device, dtype=torch.bool)


def _completion_mask_starts_at_position_zero(
    completion_mask: torch.Tensor,
    *,
    device: torch.device,
) -> bool:
    if completion_mask.shape[1] == 0:
        return False
    return bool(_bool_mask(completion_mask, device=device)[:, 0].any().item())


def _detach_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return tensor.detach() if tensor is not None else None


def _model_config_snapshot(
    model: torch.nn.Module,
    config: StatelessForwardConfig,
) -> list[tuple[Any, str, Any]]:
    saved: list[tuple[Any, str, Any]] = []
    for target in _model_config_targets(model):
        if hasattr(target, "use_cache"):
            saved.append((target, "use_cache", target.use_cache))
        if config.attention_backend != "model_default":
            saved.append(
                (
                    target,
                    "attn_implementation",
                    getattr(target, "attn_implementation", _MISSING),
                )
            )
            saved.append(
                (
                    target,
                    "_attn_implementation",
                    getattr(target, "_attn_implementation", _MISSING),
                )
            )
    return saved


def _restore_model_config_snapshot(saved: list[tuple[Any, str, Any]]) -> None:
    for target, attribute, previous in reversed(saved):
        if previous is _MISSING:
            if hasattr(target, attribute):
                delattr(target, attribute)
        else:
            setattr(target, attribute, previous)


def _model_config_targets(model: torch.nn.Module) -> list[Any]:
    targets: list[Any] = []
    seen: set[int] = set()
    for name in ("config", "generation_config"):
        target = getattr(model, name, None)
        if target is None or id(target) in seen:
            continue
        targets.append(target)
        seen.add(id(target))
    return targets


def _looks_like_kv_cache(value: Any) -> bool:
    summary = summarize_tensor_tree(value)
    return summary.tensor_count > 0


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())
