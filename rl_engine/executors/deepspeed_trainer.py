# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import os
import sysconfig
import time
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
from rl_engine.testing import (
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    selected_logprobs_reference,
)

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
        self.model = torch.nn.Sequential(
            torch.nn.Embedding(self.config.vocab_size, self.config.hidden_dim),
            torch.nn.Linear(self.config.hidden_dim, self.config.vocab_size),
        ).to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        init_result = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer,
            config=self._resolved_deepspeed_config(),
            **dict(self.config.initialize_kwargs),
        )
        self.engine = _first_initialize_result(init_result)
        engine_device = getattr(self.engine, "device", None)
        if engine_device is not None:
            self.device = torch.device(engine_device)

    def train(self, rollout: RolloutStageResult) -> TrainingStageResult:
        started_at = time.perf_counter()
        batch, payload_metrics = self._batch_from_rollout_or_synthetic(rollout)

        logits = _extract_logits(self.engine(batch.token_ids.long()))
        current_logps = selected_logprobs_reference(
            logits,
            batch.token_ids,
            mask=batch.completion_mask,
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

        if hasattr(self.engine, "zero_grad"):
            try:
                self.engine.zero_grad(set_to_none=True)
            except TypeError:
                self.engine.zero_grad()
        elif hasattr(self.optimizer, "zero_grad"):
            self.optimizer.zero_grad(set_to_none=True)
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
                "deepspeed_zero_stage": self.config.zero_stage,
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
            "zero_stage": self.config.zero_stage,
            "world_size": self._engine_world_size(),
            "rank": self._engine_rank(),
        }
        layout.update(dict(manifest_metadata.get("layout", {})))
        manifest_metadata["layout"] = layout
        publish_model: torch.nn.Module = self.model
        if self.config.zero_stage >= 3:
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
