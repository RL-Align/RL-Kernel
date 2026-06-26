# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import os
import sysconfig
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, TypeVar, overload

import torch

from rl_engine.executors.bridge import (
    WeightBridgeUnavailableError,
    WeightPublisher,
    WeightUpdateManifest,
    make_weight_bridge,
)
from rl_engine.executors.training_contract import (
    RolloutBatchMixin,
    RolloutStageResult,
    TorchRLTrainingConfig,
    TrainingStageResult,
    objective_reference_logps,
)
from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp
from rl_engine.kernels.registry import kernel_registry
from rl_engine.testing import compute_policy_ratio, compute_reference_kl, masked_mean

_TDestination = TypeVar("_TDestination", bound=dict[str, Any])


class DeepSpeedUnavailableError(RuntimeError):
    """Raised when the optional DeepSpeed runtime cannot be imported."""


@dataclass(frozen=True)
class DeepSpeedTrainingConfig(TorchRLTrainingConfig):
    """Configuration for the optional DeepSpeed training worker."""

    zero_stage: int = 0
    deepspeed_config: Mapping[str, Any] = field(default_factory=dict)
    initialize_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.zero_stage < 0:
            raise ValueError("zero_stage must be >= 0")


class _EmbeddingLMHeadModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        *,
        bias: bool = True,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=bias)
        if tie_weights:
            self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids.long())


class DeepSpeedTrainingWorker(RolloutBatchMixin):
    """
    Training worker implementation backed by a real DeepSpeed engine contract.

    DeepSpeed is optional for RL-Kernel, so importing this module never imports
    DeepSpeed. The runtime is loaded only when a worker is constructed.
    """

    config: DeepSpeedTrainingConfig

    def __init__(
        self,
        config: Optional[DeepSpeedTrainingConfig] = None,
        *,
        weight_bridge: Optional[WeightPublisher] = None,
        weight_transport: str = "local-clone",
    ):
        self.config = config or DeepSpeedTrainingConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")
        self.weight_bridge = weight_bridge or make_weight_bridge(
            weight_transport,
            source_worker="deepspeed-training",
            source_rank=0,
        )
        self._latest_published_weight_version = -1

        deepspeed = _load_deepspeed()
        self._deepspeed = deepspeed
        torch.manual_seed(self.config.seed)
        self.model = _EmbeddingLMHeadModel(
            self.config.vocab_size,
            self.config.hidden_dim,
        ).to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self._deepspeed_config = self._resolved_deepspeed_config()
        self._deepspeed_zero_stage = _resolved_zero_stage(
            self._deepspeed_config,
            fallback=self.config.zero_stage,
        )

        init_result = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer,
            config=self._deepspeed_config,
            **dict(self.config.initialize_kwargs),
        )
        self.engine = _first_initialize_result(init_result)
        engine_device = getattr(self.engine, "device", None)
        if engine_device is not None:
            self.device = torch.device(engine_device)
        self._linear_logp = _linear_logp_op_for_device(self.device)

    def train(self, rollout: RolloutStageResult) -> TrainingStageResult:
        started_at = time.perf_counter()
        batch, payload_metrics = self._batch_from_rollout_or_synthetic(rollout)
        training_model = _unwrap_training_model(self.engine, self.model)
        training_embedding = _embedding_layer(training_model)
        _validate_model_input_token_ids(
            batch.token_ids,
            vocab_size=training_embedding.num_embeddings,
        )

        if hasattr(self.engine, "zero_grad"):
            try:
                self.engine.zero_grad(set_to_none=True)
            except TypeError:
                self.engine.zero_grad()
        elif hasattr(self.optimizer, "zero_grad"):
            self.optimizer.zero_grad(set_to_none=True)

        with _linear_logp_parameter_context(
            self._deepspeed,
            training_model,
            zero_stage=self._deepspeed_zero_stage,
            world_size=self._engine_world_size(),
        ):
            current_logps = _extract_logps(
                self.engine(batch.token_ids.long()),
                training_model,
                batch.token_ids,
                batch.completion_mask,
                self._linear_logp,
                output_dtype=torch.float32,
            )
            old_logps = current_logps.detach() - 0.01
            ref_logps = objective_reference_logps(current_logps, batch)
            ratio = compute_policy_ratio(current_logps, old_logps, batch.completion_mask)
            unclipped = ratio * batch.advantages.float()
            clipped = torch.clamp(ratio, 0.8, 1.2) * batch.advantages.float()
            policy_loss = -torch.minimum(unclipped, clipped)
            kl = compute_reference_kl(current_logps, ref_logps, batch.completion_mask)
            loss = masked_mean(policy_loss + 0.01 * kl, batch.completion_mask)
            self.engine.backward(loss)
        self.engine.step()

        finished_at = time.perf_counter()
        published = self._next_published_weight_version(rollout.weight_version)
        active_advantages = batch.advantages.float()[batch.completion_mask]
        return TrainingStageResult(
            iteration=rollout.iteration,
            consumed_weight_version=rollout.weight_version,
            published_weight_version=published,
            metrics={
                "loss": float(loss.detach().cpu().item()),
                "active_tokens": int(batch.completion_mask.sum().item()),
                "payload_type": type(rollout.payload).__name__,
                "training_backend": "deepspeed",
                "training_device": str(self.device),
                "deepspeed_engine": type(self.engine).__name__,
                "deepspeed_zero_stage": self._deepspeed_zero_stage,
                "current_logp_path": "linear_logp",
                "current_logp_backend": type(self._linear_logp).__name__,
                "active_advantage_mean_global": (
                    float(active_advantages.mean().detach().cpu().item())
                    if active_advantages.numel()
                    else 0.0
                ),
                "active_advantage_std_global": (
                    float(active_advantages.std(unbiased=False).detach().cpu().item())
                    if active_advantages.numel()
                    else 0.0
                ),
                **payload_metrics,
            },
            started_at=started_at,
            finished_at=finished_at,
        )

    def publish_weights(
        self,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest:
        """
        Publish the current DeepSpeed model state as a complete weight manifest.

        ZeRO-3 partitions parameters across ranks, so publication first exports
        a gathered full-state view through DeepSpeed's gather context. If the
        runtime cannot provide a safe full-state view, the worker fails
        explicitly instead of publishing a shard.
        """
        manifest_metadata = dict(metadata or {})
        layout = {
            "kind": "full-state",
            "zero_stage": self._deepspeed_zero_stage,
            "world_size": self._engine_world_size(),
            "rank": self._engine_rank(),
        }
        layout.update(dict(manifest_metadata.get("layout", {})))
        manifest_metadata["layout"] = layout
        publish_model: torch.nn.Module = self.model
        if self._deepspeed_zero_stage >= 3:
            publish_model, export_metadata = self._export_zero3_full_state_model()
            manifest_metadata["deepspeed_zero3_full_state_export"] = export_metadata
        return self.weight_bridge.publish(
            publish_model,
            weight_version=weight_version,
            metadata=manifest_metadata,
        )

    def release_weights(self, update_id: str) -> None:
        self.weight_bridge.release(update_id)

    def _next_published_weight_version(self, consumed_weight_version: int) -> int:
        published = max(
            int(consumed_weight_version) + 1,
            self._latest_published_weight_version + 1,
        )
        self._latest_published_weight_version = published
        return published

    def _export_zero3_full_state_model(self) -> tuple[torch.nn.Module, Mapping[str, Any]]:
        model = getattr(self.engine, "module", self.model)
        rank = self._engine_rank()
        if rank != 0:
            raise WeightBridgeUnavailableError(
                "DeepSpeed ZeRO-3 full-state publish is only supported from rank 0 "
                "in this worker contract."
            )

        gathered_parameters = getattr(
            getattr(self._deepspeed, "zero", None),
            "GatheredParameters",
            None,
        )
        parameters = list(model.parameters())
        if callable(gathered_parameters):
            with gathered_parameters(parameters, modifier_rank=0):
                state = _clone_state_dict(model.state_dict())
            method = "deepspeed.zero.GatheredParameters"
        elif self._engine_world_size() == 1:
            state = _clone_state_dict(model.state_dict())
            method = "single-rank-state-dict"
        else:
            raise WeightBridgeUnavailableError(
                "DeepSpeed ZeRO-3 publish requires deepspeed.zero.GatheredParameters "
                "or an equivalent full-state export API before rollout workers can "
                "consume the weight manifest."
            )

        if not state:
            raise WeightBridgeUnavailableError(
                "DeepSpeed ZeRO-3 full-state export produced no tensors"
            )
        return _StateDictModule(state), {
            "method": method,
            "rank": rank,
            "world_size": self._engine_world_size(),
            "tensor_count": len(state),
        }

    def _engine_world_size(self) -> int:
        for attr in ("world_size", "dp_world_size"):
            value = getattr(self.engine, attr, None)
            if value is not None:
                return int(value)
        return 1

    def _engine_rank(self) -> int:
        for attr in ("global_rank", "rank", "local_rank"):
            value = getattr(self.engine, attr, None)
            if value is not None:
                return int(value)
        return 0

    def _resolved_deepspeed_config(self) -> dict[str, Any]:
        batch_size = max(1, self.config.num_prompts * self.config.samples_per_prompt)
        base = {
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": self.config.zero_stage},
            "fp16": {"enabled": self.config.dtype == torch.float16},
            "bf16": {"enabled": self.config.dtype == torch.bfloat16},
        }
        return _deep_merge(base, dict(self.config.deepspeed_config))


def _load_deepspeed() -> Any:
    _configure_cuda_home_from_python_packages()
    try:
        return importlib.import_module("deepspeed")
    except ImportError as exc:
        raise DeepSpeedUnavailableError(
            "DeepSpeed is not installed or cannot be imported. Install a DeepSpeed "
            "runtime supported by the active Python/PyTorch/CUDA environment before "
            "running DeepSpeedTrainingWorker."
        ) from exc
    except Exception as exc:
        raise DeepSpeedUnavailableError(
            "DeepSpeed is installed but failed to import in this Python/PyTorch/CUDA "
            "environment. If CUDA is provided by Python NVIDIA wheels, ensure CUDA_HOME "
            "points to the wheel toolkit root that contains bin/nvcc and include/cuda.h."
        ) from exc


def _configure_cuda_home_from_python_packages() -> None:
    explicit_cuda_home = os.environ.get("CUDA_HOME")
    if explicit_cuda_home:
        _sync_torch_cuda_home(Path(explicit_cuda_home))
        return
    for candidate in _python_cuda_home_candidates():
        if _looks_like_cuda_home(candidate):
            os.environ["CUDA_HOME"] = str(candidate)
            _sync_torch_cuda_home(candidate)
            _prepend_env_path("PATH", candidate / "bin")
            for lib_dir_name in ("lib64", "lib"):
                lib_dir = candidate / lib_dir_name
                if lib_dir.is_dir():
                    _prepend_env_path("LD_LIBRARY_PATH", lib_dir)
            return


def _python_cuda_home_candidates() -> list[Path]:
    candidates: list[Path] = []

    site_roots: set[Path] = set()
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            site_roots.add(Path(value))
    for site_root in site_roots:
        nvidia_root = site_root / "nvidia"
        if nvidia_root.is_dir():
            candidates.extend(sorted(nvidia_root.glob("cu*"), reverse=True))

    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        CUDA_HOME = None
    if CUDA_HOME:
        candidates.append(Path(str(CUDA_HOME)))
    return candidates


def _looks_like_cuda_home(path: Path) -> bool:
    return (path / "bin" / "nvcc").is_file() and (path / "include" / "cuda.h").is_file()


def _sync_torch_cuda_home(path: Path) -> None:
    try:
        import torch.utils.cpp_extension as cpp_extension
    except Exception:
        return
    cpp_extension.CUDA_HOME = str(path)


def _prepend_env_path(key: str, path: Path) -> None:
    value = str(path)
    current = os.environ.get(key)
    if not current:
        os.environ[key] = value
        return
    entries = current.split(os.pathsep)
    if value not in entries:
        os.environ[key] = os.pathsep.join([value, *entries])


def _first_initialize_result(init_result: Any) -> Any:
    if isinstance(init_result, tuple):
        if not init_result:
            raise RuntimeError("deepspeed.initialize returned an empty tuple")
        return init_result[0]
    return init_result


def _resolved_zero_stage(config: Mapping[str, Any], *, fallback: int) -> int:
    zero_config = config.get("zero_optimization")
    if isinstance(zero_config, Mapping):
        return int(zero_config.get("stage", fallback))
    if zero_config is False:
        return 0
    return int(fallback)


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, Mapping) and "logits" in model_output:
        return model_output["logits"]
    logits = getattr(model_output, "logits", None)
    if logits is not None:
        return logits
    if isinstance(model_output, (tuple, list)) and model_output:
        return _extract_logits(model_output[0])
    raise TypeError(f"DeepSpeed model output does not expose logits: {type(model_output)!r}")


def _extract_hidden_states(
    model_output: Any,
    *,
    expected_hidden_dim: Optional[int] = None,
) -> torch.Tensor:
    hidden = _coerce_hidden_tensor(model_output, expected_hidden_dim=expected_hidden_dim)
    if hidden is None:
        raise TypeError(
            f"DeepSpeed model output does not expose a hidden-state tensor: {type(model_output)!r}"
        )
    return hidden


def _linear_logp_op_for_device(device: torch.device | str) -> Any:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return NativeLinearLogpOp()
    return kernel_registry.get_op("linear_logp")


def _unwrap_training_model(engine: Any, fallback_model: torch.nn.Module) -> torch.nn.Module:
    model = getattr(engine, "module", None)
    if isinstance(model, torch.nn.Module):
        return model
    return fallback_model


def _embedding_layer(model: torch.nn.Module) -> torch.nn.Embedding:
    embedding = getattr(model, "embedding", None)
    if not isinstance(embedding, torch.nn.Embedding):
        raise TypeError(
            "DeepSpeed training model must expose an embedding torch.nn.Embedding for "
            "model-input validation"
        )
    return embedding


def _coerce_hidden_tensor(
    candidate: Any,
    *,
    expected_hidden_dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if isinstance(candidate, torch.Tensor):
        return candidate if _looks_like_hidden_tensor(candidate, expected_hidden_dim) else None
    if isinstance(candidate, Mapping):
        for key in ("last_hidden_state", "hidden"):
            value = candidate.get(key)
            hidden = _coerce_hidden_tensor(value, expected_hidden_dim=expected_hidden_dim)
            if hidden is not None:
                return hidden
        hidden_states = candidate.get("hidden_states")
        hidden = _last_hidden_state_tensor(
            hidden_states,
            expected_hidden_dim=expected_hidden_dim,
        )
        if hidden is not None:
            return hidden
        return None
    for attr in ("last_hidden_state", "hidden"):
        if hasattr(candidate, attr):
            hidden = _coerce_hidden_tensor(
                getattr(candidate, attr),
                expected_hidden_dim=expected_hidden_dim,
            )
            if hidden is not None:
                return hidden
    if hasattr(candidate, "hidden_states"):
        hidden = _last_hidden_state_tensor(
            getattr(candidate, "hidden_states"),
            expected_hidden_dim=expected_hidden_dim,
        )
        if hidden is not None:
            return hidden
    if isinstance(candidate, (tuple, list)):
        for item in candidate:
            if _has_hidden_state_metadata(item):
                hidden = _coerce_hidden_tensor(item, expected_hidden_dim=expected_hidden_dim)
                if hidden is not None:
                    return hidden
        tensor_candidates = [
            item
            for item in candidate
            if isinstance(item, torch.Tensor)
            and _looks_like_hidden_tensor(item, expected_hidden_dim)
        ]
        if len(tensor_candidates) == 1:
            return tensor_candidates[0]
        if tensor_candidates:
            max_ndim = max(tensor.ndim for tensor in tensor_candidates)
            deepest = [tensor for tensor in tensor_candidates if tensor.ndim == max_ndim]
            if len(deepest) == 1:
                return deepest[0]
        for item in candidate:
            if isinstance(item, torch.Tensor):
                continue
            hidden = _coerce_hidden_tensor(item, expected_hidden_dim=expected_hidden_dim)
            if hidden is not None:
                return hidden
    return None


def _has_hidden_state_metadata(candidate: Any) -> bool:
    if isinstance(candidate, Mapping):
        return any(key in candidate for key in ("last_hidden_state", "hidden", "hidden_states"))
    return any(
        hasattr(candidate, attr) for attr in ("last_hidden_state", "hidden", "hidden_states")
    )


def _looks_like_hidden_tensor(
    tensor: torch.Tensor,
    expected_hidden_dim: Optional[int],
) -> bool:
    if tensor.ndim < 2:
        return False
    if expected_hidden_dim is not None and int(tensor.size(-1)) != int(expected_hidden_dim):
        return False
    return True


def _last_hidden_state_tensor(
    candidate: Any,
    *,
    expected_hidden_dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if isinstance(candidate, torch.Tensor):
        return candidate if _looks_like_hidden_tensor(candidate, expected_hidden_dim) else None
    if isinstance(candidate, (tuple, list)):
        for item in reversed(candidate):
            hidden = _coerce_hidden_tensor(item, expected_hidden_dim=expected_hidden_dim)
            if hidden is not None:
                return hidden
    return None


def _safe_token_ids(token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    safe_token_ids = token_ids.long()
    if mask is None:
        return safe_token_ids
    active_mask = mask.to(device=safe_token_ids.device, dtype=torch.bool)
    if active_mask.shape != safe_token_ids.shape:
        raise ValueError(
            f"mask shape {tuple(active_mask.shape)} must match token_ids shape "
            f"{tuple(safe_token_ids.shape)}"
        )
    return safe_token_ids.masked_fill(~active_mask, 0)


def _validate_model_input_token_ids(token_ids: torch.Tensor, *, vocab_size: int) -> None:
    invalid = (token_ids < 0) | (token_ids >= int(vocab_size))
    if bool(invalid.any().item()):
        t_min = int(token_ids.min().item())
        t_max = int(token_ids.max().item())
        raise ValueError(
            f"model input token_ids must be in [0, {int(vocab_size) - 1}], got "
            f"[{t_min}, {t_max}]. Keep ignore-index / padding sentinels out of the model "
            "input path and apply masking only at the logprob/loss stage."
        )


def _linear_logp_parameter_context(
    deepspeed_runtime: Any,
    model: torch.nn.Module,
    *,
    zero_stage: int,
    world_size: int,
) -> Any:
    if int(zero_stage) < 3 or int(world_size) <= 1:
        return nullcontext()

    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, torch.nn.Linear):
        raise TypeError(
            "DeepSpeed training model must expose an lm_head torch.nn.Linear for ZeRO-3 "
            "linear_logp gathering"
        )

    gathered_parameters = getattr(
        getattr(deepspeed_runtime, "zero", None),
        "GatheredParameters",
        None,
    )
    if not callable(gathered_parameters):
        raise WeightBridgeUnavailableError(
            "DeepSpeed ZeRO-3 linear_logp training requires deepspeed.zero.GatheredParameters "
            "or an equivalent full-parameter gather API."
        )

    parameters = [lm_head.weight]
    if lm_head.bias is not None:
        parameters.append(lm_head.bias)
    return gathered_parameters(parameters, modifier_rank=None)


def _extract_logps(
    model_output: Any,
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    completion_mask: Optional[torch.Tensor],
    linear_logp_op: Any,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, torch.nn.Linear):
        raise TypeError(
            "DeepSpeed training model must expose an lm_head torch.nn.Linear for linear_logp"
        )

    hidden = _extract_hidden_states(
        model_output,
        expected_hidden_dim=int(lm_head.in_features),
    )
    targets = _safe_token_ids(token_ids.to(device=hidden.device), completion_mask)
    logps = linear_logp_op(hidden, lm_head.weight, targets, lm_head.bias)
    if completion_mask is not None:
        logps = logps.masked_fill(~completion_mask.to(device=logps.device, dtype=torch.bool), 0.0)
    return logps.to(dtype=output_dtype)


class _StateDictModule(torch.nn.Module):
    def __init__(self, state_dict: Mapping[str, torch.Tensor]):
        super().__init__()
        self._state_dict = dict(state_dict)

    @overload
    def state_dict(
        self,
        *,
        destination: _TDestination,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> _TDestination: ...

    @overload
    def state_dict(
        self,
        *,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]: ...

    def state_dict(
        self,
        destination: Optional[dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        target = destination if destination is not None else {}
        for name, tensor in self._state_dict.items():
            target[f"{prefix}{name}"] = tensor if keep_vars else tensor.detach()
        return target


def _clone_state_dict(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone(memory_format=torch.preserve_format)
        for name, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor)
    }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged
