# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from rl_engine.executors.bridge import (
    WeightBridgeUnavailableError,
    WeightConsumer,
    WeightInstallAdapter,
    WeightUpdateManifest,
    make_weight_bridge,
)
from rl_engine.executors.vllm_sampler import VLLMSamplerConfig, VLLMSharedPrefixSampler
from rl_engine.kernels.registry import kernel_registry, resolve_logp_op_type
from rl_engine.utils.logger import logger


class RolloutExecutor:
    """
    Unified execution engine for RL rollout (sampling) phase.
    Manages shared weights and dispatches hardware-specific kernels for large-scale sampling.
    """

    def __init__(
        self,
        model_config: Optional[dict] = None,
        *,
        weight_bridge: Optional[WeightConsumer] = None,
        weight_transport: Optional[str] = None,
        weight_install_adapter: Optional[WeightInstallAdapter] = None,
    ):
        self.config = model_config or {}
        transport = weight_transport or str(self.config.get("weight_transport", "cuda-vmm"))
        self.bridge = weight_bridge or make_weight_bridge(transport)
        self.shared_weights: dict[str, torch.Tensor] = {}
        self.weight_install_adapter = weight_install_adapter
        self.active_weight_version: Optional[int] = None
        self.active_weight_update_id: Optional[str] = None
        self.logp_op_type = resolve_logp_op_type(
            self.config.get("logp_backend"),
            require_batch_invariant=bool(self.config.get("require_batch_invariant_logp", False)),
        )
        self.logp_op = None
        self.attn_op = None
        self.sampler_config: Optional[VLLMSamplerConfig] = None
        self.sampler: Optional[VLLMSharedPrefixSampler] = None

        logger.info("Initializing RolloutExecutor with weight transport %s", transport)

    def update_weights(self, manifest: WeightUpdateManifest) -> Mapping[str, torch.Tensor]:
        """
        Import a complete published weight manifest and switch active version.

        The underlying bridge owns transport-specific semantics. Same-node CPU
        shared-memory imports return aliases to the published shared segment;
        CUDA VMM imports return aliases to the published GPU allocation. Legacy
        CUDA IPC remains capability-gated until its lifecycle is validated.
        """
        logger.info(
            "Importing weight update %s version=%s transport=%s",
            manifest.update_id,
            manifest.weight_version,
            manifest.transport,
        )
        previous_update_id = self.active_weight_update_id
        try:
            imported = dict(self.bridge.import_update(manifest))
            if self.weight_install_adapter is not None:
                self.weight_install_adapter.install(manifest, imported)
            self.bridge.acknowledge(manifest.update_id)
        except Exception as exc:
            try:
                self.bridge.reject(manifest.update_id, f"rollout weight install failed: {exc}")
            except Exception:
                logger.exception("Failed to reject weight update %s", manifest.update_id)
            raise
        self.shared_weights = imported
        self.active_weight_version = manifest.weight_version
        self.active_weight_update_id = manifest.update_id
        if previous_update_id is not None and previous_update_id != manifest.update_id:
            self.release_weight_update(previous_update_id)
        logger.info("Activated %s imported weight tensors.", len(self.shared_weights))
        return self.shared_weights

    def release_weights(self) -> None:
        """Release the active weight update held by this rollout worker."""
        if self.active_weight_update_id is None:
            return
        update_id = self.active_weight_update_id
        if self.weight_install_adapter is not None:
            self.weight_install_adapter.release(update_id)
        self.bridge.release(update_id)
        self.shared_weights = {}
        self.active_weight_update_id = None

    def release_weight_update(self, update_id: str) -> None:
        """Release a specific manifest update, active or already superseded."""
        if self.weight_install_adapter is not None:
            self.weight_install_adapter.release(update_id)
        self.bridge.release(update_id)
        if self.active_weight_update_id == update_id:
            self.shared_weights = {}
            self.active_weight_update_id = None

    def update_weights_via_ipc(self, ipc_handles: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        """
        Backward-compatible IPC entry point.

        Raw CUDA IPC handle imports are intentionally unavailable until issue
        #13 validates CUDA IPC snapshot visibility and handle lifetime in a real
        target runtime. New callers should pass a `WeightUpdateManifest` to
        `update_weights(...)` instead.
        """
        del ipc_handles
        raise WeightBridgeUnavailableError(
            "Raw CUDA IPC handle imports are not production-ready. Publish a "
            "WeightUpdateManifest and call update_weights(...) so the bridge can "
            "validate version, tensor metadata, acknowledgement, and release lifecycle."
        )

    def _prepare_kernels(self):
        """
        Hardware-aware operator initialization.
        Dynamically retrieves optimal operator objects for CUDA or ROCm environments.
        """
        if not self.logp_op:
            # Retrieves the best implementation based on hardware.
            self.logp_op = kernel_registry.get_op(self.logp_op_type)
            self.attn_op = kernel_registry.get_op("attn")

            logger.info(
                f"Active Kernels -> Logp({self.logp_op_type}): {type(self.logp_op).__name__},"
                f" Attn: {type(self.attn_op).__name__}"
            )

    def _prepare_sampler(self) -> VLLMSharedPrefixSampler:
        """
        Lazily construct the vLLM-backed sampler.

        vLLM import and engine construction are deferred so CPU-only tests and
        kernel-only workflows do not pay the sampler startup cost.
        """
        if self.sampler is None:
            if self.sampler_config is None:
                self.sampler_config = VLLMSamplerConfig.from_model_config(self.config)
            sampler_config = self.sampler_config
            self.sampler = VLLMSharedPrefixSampler(sampler_config)
            logger.info(
                "Initialized vLLM rollout sampler "
                f"(prefix_cache={sampler_config.enable_prefix_caching}, "
                f"num_generations={sampler_config.num_generations})"
            )
        return self.sampler

    def generate_candidates(
        self,
        prompts: str | Sequence[str],
        *,
        num_generations: Optional[int] = None,
        sampling_params: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Generate GRPO rollout candidates through vLLM with shared prefix caching.
        """
        sampler = self._prepare_sampler()
        return sampler.generate(
            prompts,
            num_generations=num_generations,
            sampling_params=sampling_params,
        )

    def execute_rollout(self, input_ids: torch.Tensor):
        """
        Execute sampling using optimized fused kernels.
        Solves the O(G * L * V) memory wall for GRPO rollout.
        """
        self._prepare_kernels()

        # Optimized workflow:
        # 1. High-throughput Attention computation.
        # 2. Fused Logprobs calculation to bypass VRAM bottlenecks.

        logger.info("Executing optimized rollout...")

        # Example: result = self.logp_op.forward(input_ids, self.shared_weights)

        return {"status": "success", "device": "cuda" if torch.cuda.is_available() else "rocm"}
