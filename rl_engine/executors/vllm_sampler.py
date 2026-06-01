# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class VLLMSamplerConfig:
    """Configuration for vLLM-backed GRPO rollout sampling."""

    model: Optional[str] = None
    num_generations: int = 1
    enable_prefix_caching: bool = True
    sampling_params: Mapping[str, Any] = field(default_factory=dict)
    engine_kwargs: Mapping[str, Any] = field(default_factory=dict)
    backend: str = "vllm"

    def __post_init__(self):
        if self.num_generations < 1:
            raise ValueError("num_generations must be >= 1")
        if self.backend != "vllm":
            raise ValueError(f"Unsupported rollout sampler backend: {self.backend}")

    @classmethod
    def from_model_config(cls, model_config: Optional[Mapping[str, Any]]) -> "VLLMSamplerConfig":
        """Build a sampler config from the executor's loose model_config dictionary."""
        config = dict(model_config or {})
        sampler_config = dict(config.get("sampler", {}))
        vllm_config = dict(config.get("vllm", {}))

        merged = {}
        merged.update(vllm_config)
        merged.update(sampler_config)
        merged.update(
            {key: value for key, value in config.items() if key not in {"sampler", "vllm"}}
        )

        sampling_params = dict(merged.get("sampling_params", {}))
        engine_kwargs = dict(merged.get("engine_kwargs", {}))

        return cls(
            model=merged.get("model"),
            num_generations=int(merged.get("num_generations", 1)),
            enable_prefix_caching=bool(merged.get("enable_prefix_caching", True)),
            sampling_params=sampling_params,
            engine_kwargs=engine_kwargs,
            backend=merged.get("backend", "vllm"),
        )


@dataclass(frozen=True)
class NormalizedRolloutCandidate:
    """Stable RL-Kernel view over a vLLM request output candidate."""

    prompt_index: int
    candidate_index: int
    request_id: Optional[str]
    prompt_token_ids: Optional[list[int]]
    token_ids: list[int]
    text: str
    finish_reason: Optional[str]
    cumulative_logprob: Optional[float]
    logprobs: Optional[Any] = None
    raw_output: Optional[Any] = field(default=None, repr=False, compare=False)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_raw:
            data.pop("raw_output", None)
        return data


class VLLMSharedPrefixSampler:
    """Lazy vLLM wrapper that preserves shared prompt prefixes across candidates."""

    def __init__(
        self,
        config: VLLMSamplerConfig,
        *,
        engine: Optional[Any] = None,
        llm_cls: Optional[type] = None,
        sampling_params_cls: Optional[type] = None,
    ):
        self.config = config
        self._engine = engine
        self._llm_cls = llm_cls
        self._sampling_params_cls = sampling_params_cls

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def generate(
        self,
        prompts: str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]],
        *,
        num_generations: Optional[int] = None,
        sampling_params: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate grouped candidates while keeping each prompt prefix byte-identical."""
        prompt_list = _normalize_prompts(prompts)
        generations = num_generations or self.config.num_generations
        if generations < 1:
            raise ValueError("num_generations must be >= 1")

        expanded_prompts = _expand_prompts(prompt_list, generations)
        params = self._build_sampling_params(sampling_params)
        outputs = self.engine.generate(expanded_prompts, params)
        grouped_outputs = _group_outputs(outputs, len(prompt_list), generations)

        return {
            "backend": self.config.backend,
            "prefix_cache_enabled": self.config.enable_prefix_caching,
            "num_prompts": len(prompt_list),
            "num_generations": generations,
            "outputs": grouped_outputs,
            "normalized_outputs": normalize_grouped_outputs(grouped_outputs),
        }

    def _build_engine(self) -> Any:
        if not self.config.model:
            raise ValueError("A model path or model name is required for vLLM rollout sampling.")

        llm_cls, _ = self._load_vllm_classes()
        kwargs = dict(self.config.engine_kwargs)
        kwargs["model"] = self.config.model
        kwargs["enable_prefix_caching"] = self.config.enable_prefix_caching

        try:
            return llm_cls(**kwargs)
        except TypeError as exc:
            raise TypeError(
                "Failed to construct vLLM LLM with enable_prefix_caching. "
                "Verify the installed vLLM version supports this engine argument."
            ) from exc

    def _build_sampling_params(self, overrides: Optional[Mapping[str, Any]]) -> Any:
        _, sampling_params_cls = self._load_vllm_classes()
        kwargs = dict(self.config.sampling_params)
        if overrides:
            kwargs.update(overrides)
        return sampling_params_cls(**kwargs)

    def _load_vllm_classes(self) -> tuple[type, type]:
        if self._llm_cls is not None and self._sampling_params_cls is not None:
            return self._llm_cls, self._sampling_params_cls

        vllm = importlib.import_module("vllm")
        self._llm_cls = self._llm_cls or vllm.LLM
        self._sampling_params_cls = self._sampling_params_cls or vllm.SamplingParams
        return self._llm_cls, self._sampling_params_cls


def _normalize_prompts(
    prompts: str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]],
) -> list[str | Mapping[str, Any]]:
    prompt_list: list[str | Mapping[str, Any]]
    if isinstance(prompts, str):
        prompt_list = [prompts]
    elif isinstance(prompts, Mapping):
        prompt_list = [dict(prompts)]
    else:
        prompt_list = list(prompts)

    if not prompt_list:
        raise ValueError("At least one prompt is required for rollout sampling.")
    if not all(isinstance(prompt, (str, Mapping)) for prompt in prompt_list):
        raise TypeError("vLLM rollout sampling expects text prompts or token prompt mappings.")

    return prompt_list


def _expand_prompts(prompts: Sequence[str | Mapping[str, Any]], num_generations: int) -> list[Any]:
    return [
        dict(prompt) if isinstance(prompt, Mapping) else prompt
        for prompt in prompts
        for _ in range(num_generations)
    ]


def _group_outputs(
    outputs: Sequence[Any],
    batch_size: int,
    num_generations: int,
) -> list[list[Any]]:
    expected = batch_size * num_generations
    output_list = list(outputs)
    if len(output_list) != expected:
        raise ValueError(f"Expected {expected} vLLM outputs, received {len(output_list)}")

    return [
        output_list[index : index + num_generations]
        for index in range(0, expected, num_generations)
    ]


def normalize_grouped_outputs(
    grouped_outputs: Sequence[Sequence[Any]],
) -> list[list[NormalizedRolloutCandidate]]:
    return [
        [
            normalize_output_candidate(
                raw_output,
                prompt_index=prompt_index,
                candidate_index=candidate_index,
            )
            for candidate_index, raw_output in enumerate(candidate_group)
        ]
        for prompt_index, candidate_group in enumerate(grouped_outputs)
    ]


def normalize_output_candidate(
    raw_output: Any,
    *,
    prompt_index: int,
    candidate_index: int,
) -> NormalizedRolloutCandidate:
    if isinstance(raw_output, Mapping):
        return _normalize_mapping_output(
            raw_output,
            prompt_index=prompt_index,
            candidate_index=candidate_index,
        )

    request_id = _safe_getattr(raw_output, "request_id")
    prompt_token_ids = _copy_int_list(_safe_getattr(raw_output, "prompt_token_ids"))
    candidate_payload = _first_sequence_item(_safe_getattr(raw_output, "outputs"))

    return NormalizedRolloutCandidate(
        prompt_index=prompt_index,
        candidate_index=candidate_index,
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        token_ids=_copy_int_list(_safe_getattr(candidate_payload, "token_ids")) or [],
        text=str(_safe_getattr(candidate_payload, "text") or ""),
        finish_reason=_safe_getattr(candidate_payload, "finish_reason"),
        cumulative_logprob=_safe_float(_safe_getattr(candidate_payload, "cumulative_logprob")),
        logprobs=_safe_getattr(candidate_payload, "logprobs"),
        raw_output=raw_output,
    )


def _normalize_mapping_output(
    raw_output: Mapping[str, Any],
    *,
    prompt_index: int,
    candidate_index: int,
) -> NormalizedRolloutCandidate:
    output_payload = raw_output
    nested_outputs = raw_output.get("outputs")
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        first_nested = _first_sequence_item(nested_outputs)
        if isinstance(first_nested, Mapping):
            output_payload = first_nested

    return NormalizedRolloutCandidate(
        prompt_index=prompt_index,
        candidate_index=candidate_index,
        request_id=raw_output.get("request_id"),
        prompt_token_ids=_copy_int_list(raw_output.get("prompt_token_ids")),
        token_ids=_copy_int_list(output_payload.get("token_ids")) or [],
        text=str(output_payload.get("text") or raw_output.get("text") or ""),
        finish_reason=output_payload.get("finish_reason") or raw_output.get("finish_reason"),
        cumulative_logprob=_safe_float(
            output_payload.get("cumulative_logprob", raw_output.get("cumulative_logprob"))
        ),
        logprobs=output_payload.get("logprobs", raw_output.get("logprobs")),
        raw_output=raw_output,
    )


def _safe_getattr(value: Any, attr: str) -> Any:
    if value is None:
        return None
    return getattr(value, attr, None)


def _first_sequence_item(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return value[0]
    return None


def _copy_int_list(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
