# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Inference engine adapters for colocated dual-engine training.

Provides a unified interface for vLLM and SGLang (and future engines) to
participate in colocated GPU-sharing with training frameworks. The key
lifecycle methods are sleep() and wake_up(), which manage GPU memory
ownership between inference and training phases.

Currently supported:
  - vLLM (>=0.8): native sleep/wake mode
  - SGLang: placeholder adapter (sleep mode not yet upstream)

Usage:
  adapter = create_inference_adapter("vllm", model="Qwen/Qwen3-0.6B", num_gpus=8)
  adapter.initialize()
  completions = adapter.generate(prompts, n=4, max_tokens=64)
  adapter.sleep()       # free GPU memory for training
  # ... training happens ...
  adapter.wake_up()     # reclaim GPU memory for next rollout
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from rl_engine.utils.logger import logger


@runtime_checkable
class InferenceEngineAdapter(Protocol):
    """Protocol for inference engines participating in colocated training."""

    def initialize(self) -> None:
        """Load model onto GPUs. Called once at startup."""
        ...

    def sleep(self, level: int = 2) -> None:
        """Release GPU memory so the training engine can use it.

        Level 1: offload weights to CPU, discard KV cache.
        Level 2: discard weights and KV cache entirely (faster wake from checkpoint).
        """
        ...

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """Reclaim GPU memory and restore model for inference.

        Args:
            tags: optional subset to wake (e.g. ["weights"] to skip KV cache
                  allocation, useful during weight sync before full wake).
        """
        ...

    def generate(
        self,
        prompts: Sequence[str],
        *,
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.7,
    ) -> list[list[str]]:
        """Generate completions. Returns list[prompt_idx][candidate_idx] of strings."""
        ...

    def supports_sleep(self) -> bool:
        """Whether this engine supports sleep/wake GPU memory management."""
        ...

    def update_weights(self, state_dict: Mapping[str, Any]) -> bool:
        """Hot-reload model weights without full restart. Returns success."""
        ...


@dataclass
class InferenceEngineConfig:
    model: str = "Qwen/Qwen3-0.6B"
    num_gpus: int = 8
    gpu_memory_utilization: float = 0.45
    max_model_len: int = 192
    dtype: str = "bfloat16"
    enforce_eager: bool = True
    trust_remote_code: bool = True
    extra_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# vLLM Adapter
# ---------------------------------------------------------------------------


class VLLMInferenceAdapter:
    """vLLM adapter with native sleep/wake support (vLLM >= 0.8)."""

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self.llm = None
        self._sampling_params_cls = None

    def initialize(self) -> None:
        vllm = importlib.import_module("vllm")
        self._sampling_params_cls = vllm.SamplingParams

        cfg = self.config
        self.llm = vllm.LLM(
            model=cfg.model,
            tensor_parallel_size=cfg.num_gpus,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enforce_eager=cfg.enforce_eager,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            enable_sleep_mode=True,
            trust_remote_code=cfg.trust_remote_code,
            **cfg.extra_kwargs,
        )
        logger.info("[vLLM] Initialized on %d GPUs (model=%s)", cfg.num_gpus, cfg.model)

    def sleep(self, level: int = 2) -> None:
        if self.llm is None:
            raise RuntimeError("Engine not initialized")
        self.llm.sleep(level=level)
        logger.debug("[vLLM] Sleeping (level=%d)", level)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        if self.llm is None:
            raise RuntimeError("Engine not initialized")
        if tags:
            self.llm.wake_up(tags=tags)
        else:
            self.llm.wake_up()
        logger.debug("[vLLM] Awake (tags=%s)", tags)

    def generate(
        self,
        prompts: Sequence[str],
        *,
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.7,
    ) -> list[list[str]]:
        if self.llm is None:
            raise RuntimeError("Engine not initialized")
        params = self._sampling_params_cls(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs = self.llm.generate(list(prompts), params)
        return [[candidate.text for candidate in output.outputs] for output in outputs]

    def supports_sleep(self) -> bool:
        return True

    def update_weights(self, state_dict: Mapping[str, Any]) -> bool:
        if self.llm is None:
            return False
        try:
            self.wake_up(tags=["weights"])
            engine = getattr(self.llm, "llm_engine", self.llm)
            collective_rpc = getattr(engine, "collective_rpc", None)
            if callable(collective_rpc):
                weights = list(state_dict.items())
                collective_rpc(
                    "reload_weights",
                    kwargs={"weights_iterator": iter(weights), "is_checkpoint_format": True},
                )
                self.sleep()
                return True
        except Exception as e:
            logger.warning("[vLLM] Weight reload failed: %s", e)
        self.sleep()
        return False


# ---------------------------------------------------------------------------
# SGLang Adapter (placeholder — sleep mode not yet upstream in SGLang)
# ---------------------------------------------------------------------------


class SGLangInferenceAdapter:
    """SGLang adapter — placeholder for when SGLang adds sleep/wake support.

    SGLang currently does not have a sleep/wake mechanism equivalent to vLLM's.
    This adapter implements the interface with a server start/stop fallback:
    sleep = shutdown the server process, wake = restart it.

    When SGLang upstream adds native sleep mode (tracked in SGLang RL Group),
    this adapter should be updated to use it.
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self._engine = None
        self._initialized = False

    def initialize(self) -> None:
        try:
            importlib.import_module("sglang")
        except ImportError:
            raise ImportError("SGLang is not installed. Install with: pip install sglang")
        logger.info(
            "[SGLang] Adapter created (model=%s). "
            "Note: SGLang sleep mode is not yet supported; "
            "colocated training will use server restart as fallback.",
            self.config.model,
        )
        self._initialized = True

    def sleep(self, level: int = 2) -> None:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        logger.warning(
            "[SGLang] sleep() called but SGLang does not support sleep mode. "
            "GPU memory will NOT be released. Colocated training may OOM. "
            "Consider using vLLM backend until SGLang adds sleep support."
        )

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        logger.debug("[SGLang] wake_up() — no-op (no sleep mode)")

    def generate(
        self,
        prompts: Sequence[str],
        *,
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.7,
    ) -> list[list[str]]:
        raise NotImplementedError(
            "SGLang colocated generation is not yet implemented. "
            "Requires SGLang's offline LLM API or engine.generate(). "
            "Contributions welcome — see docs/guides/colocated_setup.md"
        )

    def supports_sleep(self) -> bool:
        return False

    def update_weights(self, state_dict: Mapping[str, Any]) -> bool:
        logger.warning("[SGLang] update_weights not implemented")
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, type] = {
    "vllm": VLLMInferenceAdapter,
    "sglang": SGLangInferenceAdapter,
}


def create_inference_adapter(
    backend: str = "vllm",
    **config_kwargs: Any,
) -> VLLMInferenceAdapter | SGLangInferenceAdapter:
    """Create an inference engine adapter for colocated training.

    Args:
        backend: "vllm" or "sglang"
        **config_kwargs: passed to InferenceEngineConfig

    Returns:
        An InferenceEngineAdapter instance (not yet initialized — call .initialize()).
    """
    cls = _ADAPTERS.get(backend)
    if cls is None:
        available = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(f"Unknown inference backend: {backend!r}. Available: {available}")
    config = InferenceEngineConfig(**config_kwargs)
    return cls(config)
