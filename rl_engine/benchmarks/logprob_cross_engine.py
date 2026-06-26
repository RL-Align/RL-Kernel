# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from rl_engine.testing import selected_logprobs_reference

SCHEMA_VERSION = "1.0"
DEFAULT_POLICY_CHANNEL = "policy"
MISMATCH_CHOICES = ("none", "token_shift", "logprob_rounding")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogprobSequence:
    """One logical rollout sequence with selected logprobs for completion tokens."""

    sequence_id: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    rollout_logprobs: dict[str, list[float]]
    completion_mask: list[bool]
    prompt_text: str | None = None
    completion_text: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id must be non-empty")
        if not self.prompt_token_ids:
            raise ValueError(f"{self.sequence_id}: prompt_token_ids must be non-empty")
        if not self.completion_token_ids:
            raise ValueError(f"{self.sequence_id}: completion_token_ids must be non-empty")
        if len(self.completion_mask) != len(self.completion_token_ids):
            raise ValueError(
                f"{self.sequence_id}: completion_mask length must match completion_token_ids"
            )
        for channel, values in self.rollout_logprobs.items():
            if len(values) != len(self.completion_token_ids):
                raise ValueError(
                    f"{self.sequence_id}: rollout_logprobs[{channel!r}] length must match "
                    "completion_token_ids"
                )

    @property
    def input_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.completion_token_ids]

    @property
    def completion_len(self) -> int:
        return len(self.completion_token_ids)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "prompt_token_ids": self.prompt_token_ids,
            "completion_token_ids": self.completion_token_ids,
            "rollout_logprobs": self.rollout_logprobs,
            "completion_mask": self.completion_mask,
            "prompt_text": self.prompt_text,
            "completion_text": self.completion_text,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        default_sequence_id: str,
    ) -> "LogprobSequence":
        prompt_token_ids = _coerce_int_list(
            payload.get("prompt_token_ids") or payload.get("prompt_tokens"),
            field_name="prompt_token_ids",
        )
        completion_token_ids = _coerce_int_list(
            payload.get("completion_token_ids")
            or payload.get("output_token_ids")
            or payload.get("token_ids")
            or payload.get("tokens"),
            field_name="completion_token_ids",
        )
        completion_mask_value = payload.get("completion_mask")
        if completion_mask_value is None:
            completion_mask = [True] * len(completion_token_ids)
        else:
            completion_mask = [bool(item) for item in completion_mask_value]

        rollout_logprobs = _normalize_rollout_logprobs(payload, len(completion_token_ids))
        return cls(
            sequence_id=str(payload.get("sequence_id") or payload.get("id") or default_sequence_id),
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            rollout_logprobs=rollout_logprobs,
            completion_mask=completion_mask,
            prompt_text=_optional_str(payload.get("prompt_text") or payload.get("prompt")),
            completion_text=_optional_str(
                payload.get("completion_text") or payload.get("text") or payload.get("output_text")
            ),
            request_id=_optional_str(payload.get("request_id")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LogprobBenchmarkFixture:
    """Portable rollout fixture consumed by the training-side replay path."""

    schema_version: str
    created_at: str
    rollout_engine: str
    model: str
    tokenizer: str | None
    dtype: str
    sequences: list[LogprobSequence]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported fixture schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        if not self.sequences:
            raise ValueError("fixture must contain at least one sequence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "rollout_engine": self.rollout_engine,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dtype": self.dtype,
            "metadata": self.metadata,
            "sequences": [sequence.to_dict() for sequence in self.sequences],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogprobBenchmarkFixture":
        sequences_payload = payload.get("sequences")
        if not isinstance(sequences_payload, Sequence) or isinstance(
            sequences_payload, (str, bytes)
        ):
            raise ValueError("fixture JSON must contain a 'sequences' list")
        return cls(
            schema_version=str(payload.get("schema_version") or SCHEMA_VERSION),
            created_at=str(payload.get("created_at") or _utc_now()),
            rollout_engine=str(payload.get("rollout_engine") or "fixture"),
            model=str(payload.get("model") or payload.get("model_name") or "unknown"),
            tokenizer=_optional_str(payload.get("tokenizer")),
            dtype=str(payload.get("dtype") or "unknown"),
            metadata=dict(payload.get("metadata") or {}),
            sequences=[
                LogprobSequence.from_dict(item, default_sequence_id=f"seq-{index}")
                for index, item in enumerate(sequences_payload)
                if isinstance(item, Mapping)
            ],
        )


@dataclass(frozen=True)
class DriftThresholds:
    max_abs_error: float
    mean_abs_error: float
    max_relative_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LogprobCrossBenchmarkConfig:
    rollout_engine: str = "synthetic"
    training_engine: str = "torch"
    model: str = "synthetic"
    old_model: str | None = None
    reference_model: str | None = None
    tokenizer: str | None = None
    prompts: Path | None = None
    rollout_fixture: Path | None = None
    output_dir: Path = Path("artifacts/logprob_cross_engine/latest")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() else "float32"
    model_revision: str | None = None
    trust_remote_code: bool = False
    seed: int = 0
    num_prompts: int = 2
    prompt_len: int = 8
    max_new_tokens: int = 16
    vocab_size: int = 256
    hidden_size: int = 64
    num_generations: int = 1
    rollout_batch_size: int = 4
    training_micro_batch_size: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    max_abs_error: float = 1e-4
    mean_abs_error: float = 1e-5
    max_relative_error: float = 1e-4
    fail_on_drift: bool = True
    inject_mismatch: str = "none"
    include_token_drifts_in_report: bool = False
    summary_top_k: int = 10

    @property
    def thresholds(self) -> DriftThresholds:
        return DriftThresholds(
            max_abs_error=self.max_abs_error,
            mean_abs_error=self.mean_abs_error,
            max_relative_error=self.max_relative_error,
        )


@dataclass(frozen=True)
class _BatchLayout:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_token_ids: torch.Tensor
    target_mask: torch.Tensor
    logit_positions: torch.Tensor


class _TinyCausalLM(torch.nn.Module):
    """Small deterministic causal LM used for dependency-free smoke validation."""

    def __init__(self, *, vocab_size: int, hidden_size: int, seed: int):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, pad_token_id=0)
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.randn(self.embed.weight.shape, generator=generator) * 0.05
            )
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()
            self.lm_head.weight.copy_(
                torch.randn(self.lm_head.weight.shape, generator=generator) * 0.05
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        hidden = self.norm(self.embed(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def run_cross_engine_benchmark(config: LogprobCrossBenchmarkConfig) -> dict[str, Any]:
    """Run rollout fixture capture, training replay, comparison, and report writing."""

    _validate_config(config)
    started_at = time.perf_counter()
    fixture = build_rollout_fixture(config)
    fixture = inject_rollout_mismatch(fixture, config.inject_mismatch)

    training_scores = score_training_logprobs(fixture, config)
    report, token_drifts = compare_logprobs(
        fixture=fixture,
        training_scores=training_scores,
        config=config,
        duration_seconds=time.perf_counter() - started_at,
    )
    write_benchmark_outputs(fixture, report, token_drifts, config.output_dir)
    return report


def build_rollout_fixture(config: LogprobCrossBenchmarkConfig) -> LogprobBenchmarkFixture:
    engine = config.rollout_engine.lower()
    if engine == "synthetic":
        return build_synthetic_rollout_fixture(config)
    if engine in {"fixture", "ingest"}:
        if config.rollout_fixture is None:
            raise ValueError("--rollout-fixture is required when --rollout-engine=fixture")
        return load_rollout_fixture(config.rollout_fixture)
    if engine in {"hf", "transformers"}:
        return build_hf_rollout_fixture(config)
    if engine == "vllm":
        return build_vllm_rollout_fixture(config)
    raise ValueError(f"unsupported rollout engine: {config.rollout_engine}")


def build_synthetic_rollout_fixture(
    config: LogprobCrossBenchmarkConfig,
) -> LogprobBenchmarkFixture:
    dtype = _torch_dtype_from_string(config.dtype)
    device = torch.device(config.device)
    model = _build_synthetic_model(config).to(device=device)
    if _model_dtype_supported_on_device(dtype, device):
        model = model.to(dtype=dtype)
    model.eval()

    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(config.seed + 17)
    prompt_ids = torch.randint(
        low=1,
        high=config.vocab_size,
        size=(config.num_prompts, config.prompt_len),
        generator=generator,
        device=device,
        dtype=torch.long,
    )

    sequences: list[LogprobSequence] = []
    for prompt_index in range(config.num_prompts):
        for candidate_index in range(config.num_generations):
            current = prompt_ids[prompt_index : prompt_index + 1].clone()
            completion: list[int] = []
            rollout_logprobs: list[float] = []
            for _ in range(config.max_new_tokens):
                with torch.no_grad():
                    logits = _extract_logits(_call_model(model, current))[:, -1, :].float()
                    log_probs = torch.log_softmax(logits / config.temperature, dim=-1)
                    if config.do_sample:
                        probs = torch.exp(log_probs)
                        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
                    else:
                        next_token = torch.argmax(log_probs, dim=-1, keepdim=True)
                    selected = log_probs.gather(-1, next_token)
                token_id = int(next_token.item())
                completion.append(token_id)
                rollout_logprobs.append(float(selected.item()))
                current = torch.cat([current, next_token.to(dtype=torch.long)], dim=1)

            sequences.append(
                LogprobSequence(
                    sequence_id=f"synthetic-{prompt_index}-{candidate_index}",
                    prompt_token_ids=[int(item) for item in prompt_ids[prompt_index].tolist()],
                    completion_token_ids=completion,
                    rollout_logprobs={DEFAULT_POLICY_CHANNEL: rollout_logprobs},
                    completion_mask=[True] * len(completion),
                    prompt_text=None,
                    completion_text=None,
                    metadata={
                        "prompt_index": prompt_index,
                        "candidate_index": candidate_index,
                        "generation_mode": "sample" if config.do_sample else "greedy",
                    },
                )
            )

    return LogprobBenchmarkFixture(
        schema_version=SCHEMA_VERSION,
        created_at=_utc_now(),
        rollout_engine="synthetic",
        model="synthetic",
        tokenizer=None,
        dtype=str(dtype).replace("torch.", ""),
        sequences=sequences,
        metadata={
            "seed": config.seed,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "prompt_len": config.prompt_len,
            "max_new_tokens": config.max_new_tokens,
            "num_prompts": config.num_prompts,
            "num_generations": config.num_generations,
            "temperature": config.temperature,
            "do_sample": config.do_sample,
            "rollout_batch_size": config.rollout_batch_size,
            "model_kind": "tiny_causal_lm",
        },
    )


def build_hf_rollout_fixture(config: LogprobCrossBenchmarkConfig) -> LogprobBenchmarkFixture:
    prompts = _load_prompt_records(config.prompts)
    if not prompts:
        raise ValueError("--prompts must contain at least one prompt for --rollout-engine=hf")
    transformers = _import_transformers()
    tokenizer_name = config.tokenizer or config.model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = _torch_dtype_from_string(config.dtype)
    device = torch.device(config.device)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=dtype if device.type == "cuda" else None,
    ).to(device)
    model.eval()

    sequences: list[LogprobSequence] = []
    for batch_start in range(0, len(prompts), config.rollout_batch_size):
        batch_records = prompts[batch_start : batch_start + config.rollout_batch_size]
        encoded = _encode_prompt_batch(tokenizer, batch_records, device)
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "temperature": config.temperature if config.do_sample else None,
            "top_p": config.top_p if config.do_sample else None,
            "top_k": config.top_k if config.do_sample and config.top_k > 0 else None,
            "num_return_sequences": config.num_generations,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        generation_kwargs = {
            key: value for key, value in generation_kwargs.items() if value is not None
        }
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                **generation_kwargs,
            )
            transition_scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )

        input_width = int(encoded["input_ids"].shape[1])
        for output_index, output_ids in enumerate(outputs.sequences):
            source_index = output_index // config.num_generations
            candidate_index = output_index % config.num_generations
            prompt_record = batch_records[source_index]
            prompt_token_ids = _prompt_tokens_from_record_or_batch(
                tokenizer,
                prompt_record,
                encoded["input_ids"][source_index],
                encoded["attention_mask"][source_index],
            )
            generated = output_ids[input_width:].detach().cpu().tolist()
            completion_token_ids = _trim_generated_tokens(
                generated,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            if not completion_token_ids:
                continue
            logprobs = [
                float(item)
                for item in transition_scores[output_index, : len(completion_token_ids)]
                .detach()
                .cpu()
                .tolist()
            ]
            sequence_id = f"hf-{batch_start + source_index}-{candidate_index}"
            sequences.append(
                LogprobSequence(
                    sequence_id=sequence_id,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    rollout_logprobs={DEFAULT_POLICY_CHANNEL: logprobs},
                    completion_mask=[True] * len(completion_token_ids),
                    prompt_text=_prompt_text(prompt_record),
                    completion_text=tokenizer.decode(
                        completion_token_ids,
                        skip_special_tokens=True,
                    ),
                    metadata={
                        "prompt_index": batch_start + source_index,
                        "candidate_index": candidate_index,
                        "generation_mode": "sample" if config.do_sample else "greedy",
                    },
                )
            )

    if not sequences:
        raise RuntimeError("HF rollout produced no completion tokens to compare")
    return LogprobBenchmarkFixture(
        schema_version=SCHEMA_VERSION,
        created_at=_utc_now(),
        rollout_engine="hf",
        model=config.model,
        tokenizer=tokenizer_name,
        dtype=str(dtype).replace("torch.", ""),
        sequences=sequences,
        metadata={
            "model_revision": config.model_revision,
            "transformers_version": _safe_package_version("transformers"),
            "seed": config.seed,
            "max_new_tokens": config.max_new_tokens,
            "num_generations": config.num_generations,
            "rollout_batch_size": config.rollout_batch_size,
        },
    )


def build_vllm_rollout_fixture(config: LogprobCrossBenchmarkConfig) -> LogprobBenchmarkFixture:
    prompts = _load_prompt_records(config.prompts)
    if not prompts:
        raise ValueError("--prompts must contain at least one prompt for --rollout-engine=vllm")
    try:
        from rl_engine.executors.vllm_sampler import VLLMSamplerConfig, VLLMSharedPrefixSampler
    except Exception as exc:
        raise RuntimeError("vLLM rollout adapter could not import RL-Kernel vLLM wrapper") from exc

    sampling_params = {
        "max_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "logprobs": 1,
    }
    if config.top_k > 0:
        sampling_params["top_k"] = config.top_k

    sampler = VLLMSharedPrefixSampler(
        VLLMSamplerConfig(
            model=config.model,
            num_generations=config.num_generations,
            sampling_params=sampling_params,
            engine_kwargs={
                "dtype": config.dtype,
                "trust_remote_code": config.trust_remote_code,
            },
        )
    )
    prompt_values = [_prompt_payload_for_vllm(record) for record in prompts]
    result = sampler.generate(prompt_values)

    tokenizer = None
    if config.tokenizer or config.model:
        try:
            transformers = _import_transformers()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.tokenizer or config.model,
                revision=config.model_revision,
                trust_remote_code=config.trust_remote_code,
            )
        except Exception:
            logger.debug("Optional vLLM tokenizer fallback failed", exc_info=True)
            tokenizer = None

    sequences: list[LogprobSequence] = []
    for group in result["normalized_outputs"]:
        for candidate in group:
            completion_token_ids = [int(item) for item in candidate.token_ids]
            if not completion_token_ids:
                continue
            prompt_record = prompts[candidate.prompt_index]
            prompt_token_ids = candidate.prompt_token_ids
            if prompt_token_ids is None:
                prompt_token_ids = _prompt_token_ids_from_record(prompt_record, tokenizer)
            if not prompt_token_ids:
                raise ValueError(
                    "vLLM output did not include prompt_token_ids and tokenizer fallback failed"
                )
            logprobs = _selected_logprobs_from_rollout_payload(
                candidate.logprobs,
                completion_token_ids,
            )
            sequence_id = f"vllm-{candidate.prompt_index}-{candidate.candidate_index}"
            sequences.append(
                LogprobSequence(
                    sequence_id=sequence_id,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    rollout_logprobs={DEFAULT_POLICY_CHANNEL: logprobs},
                    completion_mask=[True] * len(completion_token_ids),
                    prompt_text=_prompt_text(prompt_record),
                    completion_text=candidate.text,
                    request_id=candidate.request_id,
                    metadata={
                        "prompt_index": candidate.prompt_index,
                        "candidate_index": candidate.candidate_index,
                        "finish_reason": candidate.finish_reason,
                        "cumulative_logprob": candidate.cumulative_logprob,
                    },
                )
            )

    if not sequences:
        raise RuntimeError("vLLM rollout produced no completion tokens to compare")
    return LogprobBenchmarkFixture(
        schema_version=SCHEMA_VERSION,
        created_at=_utc_now(),
        rollout_engine="vllm",
        model=config.model,
        tokenizer=config.tokenizer or config.model,
        dtype=config.dtype,
        sequences=sequences,
        metadata={
            "vllm_version": _safe_package_version("vllm"),
            "max_new_tokens": config.max_new_tokens,
            "num_generations": config.num_generations,
            "sampling_params": sampling_params,
        },
    )


def score_training_logprobs(
    fixture: LogprobBenchmarkFixture,
    config: LogprobCrossBenchmarkConfig,
) -> dict[str, dict[str, list[float]]]:
    engine = config.training_engine.lower()
    if engine not in {"torch", "hf", "transformers"}:
        raise ValueError(f"unsupported training engine: {config.training_engine}")
    results: dict[str, dict[str, list[float]]] = {}
    scored_model_cache: dict[str, dict[str, list[float]]] = {}
    for channel in _rollout_channels(fixture):
        model_name = _model_name_for_channel(channel, config)
        if model_name is None:
            continue
        shared_cache_key = f"model:{model_name}"
        if shared_cache_key not in scored_model_cache:
            scored_model_cache[shared_cache_key] = _score_training_channel(
                fixture,
                config,
                model_name=model_name,
            )
        results[channel] = scored_model_cache[shared_cache_key]
    return results


def _rollout_channels(fixture: LogprobBenchmarkFixture) -> list[str]:
    channels: set[str] = set()
    for sequence in fixture.sequences:
        channels.update(sequence.rollout_logprobs)
    return sorted(channels)


def _model_name_for_channel(
    channel: str,
    config: LogprobCrossBenchmarkConfig,
) -> str | None:
    normalized = channel.lower().replace("-", "_")
    if normalized in {"policy", "current", "current_policy"}:
        return config.model
    if normalized in {"old", "old_policy", "old_policy_logprobs"}:
        return config.old_model
    if normalized in {"ref", "reference", "reference_policy", "ref_policy"}:
        return config.reference_model
    return None


def load_training_model(
    fixture: LogprobBenchmarkFixture,
    config: LogprobCrossBenchmarkConfig,
    *,
    model_name: str | None = None,
) -> torch.nn.Module:
    model_name = model_name or config.model
    if model_name == "synthetic" and fixture.model not in {"synthetic", "unknown", "fixture"}:
        model_name = fixture.model
    if model_name == "synthetic" or fixture.metadata.get("model_kind") == "tiny_causal_lm":
        seed = int(fixture.metadata.get("seed", config.seed))
        vocab_size = int(fixture.metadata.get("vocab_size", config.vocab_size))
        hidden_size = int(fixture.metadata.get("hidden_size", config.hidden_size))
        return _TinyCausalLM(vocab_size=vocab_size, hidden_size=hidden_size, seed=seed)

    transformers = _import_transformers()
    dtype = _torch_dtype_from_string(config.dtype)
    device = torch.device(config.device)
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=dtype if device.type == "cuda" else None,
    )


def _score_training_channel(
    fixture: LogprobBenchmarkFixture,
    config: LogprobCrossBenchmarkConfig,
    *,
    model_name: str,
) -> dict[str, list[float]]:
    model = load_training_model(fixture, config, model_name=model_name)
    model.eval()
    dtype = _torch_dtype_from_string(config.dtype)
    device = torch.device(config.device)
    if _model_dtype_supported_on_device(dtype, device):
        model = model.to(dtype=dtype)
    model = model.to(device)
    return score_sequences_with_model(
        model=model,
        sequences=fixture.sequences,
        device=device,
        micro_batch_size=config.training_micro_batch_size,
        pad_token_id=_pad_token_id(model),
        temperature=config.temperature,
    )


def score_sequences_with_model(
    *,
    model: torch.nn.Module,
    sequences: Sequence[LogprobSequence],
    device: torch.device,
    micro_batch_size: int,
    pad_token_id: int,
    temperature: float,
) -> dict[str, list[float]]:
    if micro_batch_size <= 0:
        raise ValueError("training_micro_batch_size must be greater than zero")
    scores: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(sequences), micro_batch_size):
            batch_sequences = list(sequences[start : start + micro_batch_size])
            layout = _build_batch_layout(batch_sequences, pad_token_id=pad_token_id, device=device)
            logits = _extract_logits(
                _call_model(
                    model,
                    layout.input_ids,
                    attention_mask=layout.attention_mask,
                    use_cache=False,
                )
            )
            selected_logits = _gather_position_logits(logits, layout.logit_positions)
            selected = selected_logprobs_reference(
                selected_logits,
                layout.target_token_ids,
                mask=layout.target_mask,
                temperature=temperature,
                output_dtype=torch.float32,
            )
            selected_cpu = selected.detach().cpu()
            for row, sequence in enumerate(batch_sequences):
                length = sequence.completion_len
                scores[sequence.sequence_id] = [
                    float(item) for item in selected_cpu[row, :length].tolist()
                ]
    return scores


def compare_logprobs(
    *,
    fixture: LogprobBenchmarkFixture,
    training_scores: dict[str, dict[str, list[float]]],
    config: LogprobCrossBenchmarkConfig,
    duration_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    thresholds = config.thresholds
    token_drifts: list[dict[str, Any]] = []
    sequence_summaries: list[dict[str, Any]] = []
    compared_channels: set[str] = set()
    skipped_channels: set[str] = set()

    for sequence_index, sequence in enumerate(fixture.sequences):
        sequence_errors: list[float] = []
        for channel, rollout_values in sequence.rollout_logprobs.items():
            channel_scores = training_scores.get(channel)
            if channel_scores is None or sequence.sequence_id not in channel_scores:
                skipped_channels.add(channel)
                continue
            compared_channels.add(channel)
            training_values = channel_scores[sequence.sequence_id]
            if len(training_values) != sequence.completion_len:
                raise ValueError(
                    f"{sequence.sequence_id}: training score length for {channel!r} "
                    "does not match completion length"
                )
            for token_index, active in enumerate(sequence.completion_mask):
                if not active:
                    continue
                rollout_value = float(rollout_values[token_index])
                training_value = float(training_values[token_index])
                abs_error = abs(rollout_value - training_value)
                rel_error = abs_error / max(abs(training_value), 1e-8)
                token_drifts.append(
                    {
                        "channel": channel,
                        "sequence_id": sequence.sequence_id,
                        "sequence_index": sequence_index,
                        "completion_index": token_index,
                        "absolute_position": sequence.prompt_len + token_index,
                        "region": "completion",
                        "target_token_id": int(sequence.completion_token_ids[token_index]),
                        "rollout_logprob": rollout_value,
                        "training_logprob": training_value,
                        "abs_error": abs_error,
                        "relative_error": rel_error,
                        "prompt_len": sequence.prompt_len,
                        "completion_len": sequence.completion_len,
                    }
                )
                sequence_errors.append(abs_error)
        if sequence_errors:
            sequence_summaries.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "active_tokens": len(sequence_errors),
                    "max_abs_error": max(sequence_errors),
                    "mean_abs_error": sum(sequence_errors) / len(sequence_errors),
                    "sum_abs_error": sum(sequence_errors),
                }
            )

    if not token_drifts:
        raise ValueError("no overlapping active rollout/training logprob channels were compared")

    max_abs_error = max(item["abs_error"] for item in token_drifts)
    mean_abs_error = sum(item["abs_error"] for item in token_drifts) / len(token_drifts)
    max_relative_error = max(item["relative_error"] for item in token_drifts)
    worst_drift = max(token_drifts, key=lambda item: item["abs_error"])
    status = (
        "pass"
        if max_abs_error <= thresholds.max_abs_error
        and mean_abs_error <= thresholds.mean_abs_error
        and max_relative_error <= thresholds.max_relative_error
        else "fail"
    )

    report = {
        "report_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "rollout_engine": fixture.rollout_engine,
        "training_engine": config.training_engine,
        "model": fixture.model,
        "tokenizer": fixture.tokenizer,
        "dtype": config.dtype,
        "device": config.device,
        "thresholds": thresholds.to_dict(),
        "summary": {
            "sequence_count": len(fixture.sequences),
            "active_tokens": len(token_drifts),
            "compared_channels": sorted(compared_channels),
            "skipped_channels": sorted(skipped_channels),
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "max_relative_error": max_relative_error,
            "duration_seconds": duration_seconds,
        },
        "worst_drift": worst_drift,
        "top_token_drifts": sorted(
            token_drifts,
            key=lambda item: item["abs_error"],
            reverse=True,
        )[: max(1, config.summary_top_k)],
        "sequence_summaries": sorted(
            sequence_summaries,
            key=lambda item: item["max_abs_error"],
            reverse=True,
        ),
        "metadata": {
            "fixture": fixture.metadata,
            "runtime": _runtime_metadata(),
            "config": _public_config_dict(config),
        },
    }
    if config.include_token_drifts_in_report:
        report["token_drifts"] = token_drifts
    return report, token_drifts


def write_benchmark_outputs(
    fixture: LogprobBenchmarkFixture,
    report: Mapping[str, Any],
    token_drifts: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "rollout_fixture.json", fixture.to_dict())
    _write_json(output_path / "report.json", report)
    with (output_path / "token_drifts.jsonl").open("w", encoding="utf-8") as handle:
        for item in token_drifts:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    (output_path / "summary.md").write_text(_format_markdown_summary(report), encoding="utf-8")


def load_rollout_fixture(path: Path | str) -> LogprobBenchmarkFixture:
    fixture_path = Path(path)
    if not fixture_path.exists():
        raise FileNotFoundError(f"rollout fixture does not exist: {fixture_path}")
    if fixture_path.suffix.lower() == ".jsonl":
        sequences: list[LogprobSequence] = []
        with fixture_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if "sequences" in payload:
                    nested = LogprobBenchmarkFixture.from_dict(payload)
                    sequences.extend(nested.sequences)
                    continue
                sequences.append(
                    LogprobSequence.from_dict(payload, default_sequence_id=f"jsonl-{index}")
                )
        return LogprobBenchmarkFixture(
            schema_version=SCHEMA_VERSION,
            created_at=_utc_now(),
            rollout_engine="fixture",
            model="fixture",
            tokenizer=None,
            dtype="unknown",
            sequences=sequences,
            metadata={"source_path": str(fixture_path)},
        )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return LogprobBenchmarkFixture.from_dict(payload)


def inject_rollout_mismatch(
    fixture: LogprobBenchmarkFixture,
    mismatch: str,
) -> LogprobBenchmarkFixture:
    if mismatch == "none":
        return fixture
    if mismatch not in MISMATCH_CHOICES:
        raise ValueError(f"unsupported mismatch injection: {mismatch}")

    sequences = list(fixture.sequences)
    first = sequences[0]
    mismatch_channel = next(iter(first.rollout_logprobs))
    values = list(first.rollout_logprobs[mismatch_channel])
    if mismatch == "token_shift" and len(values) > 1:
        values = values[1:] + values[:1]
        original_values = first.rollout_logprobs[mismatch_channel]
        unchanged = all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(values, original_values, strict=True)
        )
        if unchanged:
            values[0] += 0.25
    elif mismatch == "logprob_rounding":
        values = [float(torch.tensor(value, dtype=torch.float16).item()) for value in values]

    updated_logprobs = dict(first.rollout_logprobs)
    updated_logprobs[mismatch_channel] = values
    sequences[0] = LogprobSequence(
        sequence_id=first.sequence_id,
        prompt_token_ids=first.prompt_token_ids,
        completion_token_ids=first.completion_token_ids,
        rollout_logprobs=updated_logprobs,
        completion_mask=first.completion_mask,
        prompt_text=first.prompt_text,
        completion_text=first.completion_text,
        request_id=first.request_id,
        metadata={**first.metadata, "mismatch_injected": mismatch},
    )
    return LogprobBenchmarkFixture(
        schema_version=fixture.schema_version,
        created_at=fixture.created_at,
        rollout_engine=fixture.rollout_engine,
        model=fixture.model,
        tokenizer=fixture.tokenizer,
        dtype=fixture.dtype,
        sequences=sequences,
        metadata={**fixture.metadata, "mismatch_injected": mismatch},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end selected-logprob cross-benchmark for rollout vs training engines"
    )
    parser.add_argument(
        "--rollout-engine",
        default="synthetic",
        choices=["synthetic", "hf", "vllm", "fixture"],
    )
    parser.add_argument(
        "--training-engine",
        default="torch",
        choices=["torch", "hf", "transformers"],
    )
    parser.add_argument("--model", default="synthetic")
    parser.add_argument("--old-model", default=None)
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--rollout-fixture", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/logprob_cross_engine/latest"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--training-micro-batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--max-abs-error", type=float, default=1e-4)
    parser.add_argument("--mean-abs-error", type=float, default=1e-5)
    parser.add_argument("--max-relative-error", type=float, default=1e-4)
    parser.add_argument("--no-fail-on-drift", action="store_true")
    parser.add_argument("--inject-mismatch", choices=MISMATCH_CHOICES, default="none")
    parser.add_argument("--include-token-drifts-in-report", action="store_true")
    parser.add_argument("--summary-top-k", type=int, default=10)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny deterministic local model and short shapes for CI/local validation.",
    )
    parser.add_argument("--no-summary", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> LogprobCrossBenchmarkConfig:
    if args.smoke:
        args.rollout_engine = "synthetic"
        args.training_engine = "torch"
        args.model = "synthetic"
        args.dtype = "float32"
        args.num_prompts = min(args.num_prompts, 2)
        args.prompt_len = min(args.prompt_len, 4)
        args.max_new_tokens = min(args.max_new_tokens, 6)
        args.vocab_size = min(args.vocab_size, 64)
        args.hidden_size = min(args.hidden_size, 32)
        args.rollout_batch_size = min(args.rollout_batch_size, 2)
        args.training_micro_batch_size = min(args.training_micro_batch_size, 2)

    return LogprobCrossBenchmarkConfig(
        rollout_engine=args.rollout_engine,
        training_engine=args.training_engine,
        model=args.model,
        old_model=args.old_model,
        reference_model=args.reference_model,
        tokenizer=args.tokenizer,
        prompts=args.prompts,
        rollout_fixture=args.rollout_fixture,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        model_revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        num_prompts=args.num_prompts,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_generations=args.num_generations,
        rollout_batch_size=args.rollout_batch_size,
        training_micro_batch_size=args.training_micro_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        max_abs_error=args.max_abs_error,
        mean_abs_error=args.mean_abs_error,
        max_relative_error=args.max_relative_error,
        fail_on_drift=not args.no_fail_on_drift,
        inject_mismatch=args.inject_mismatch,
        include_token_drifts_in_report=args.include_token_drifts_in_report,
        summary_top_k=args.summary_top_k,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    report = run_cross_engine_benchmark(config)
    if not args.no_summary:
        print(_format_console_summary(report, config.output_dir))
    if report["status"] != "pass" and config.fail_on_drift:
        return 1
    return 0


def _validate_config(config: LogprobCrossBenchmarkConfig) -> None:
    if config.temperature <= 0.0:
        raise ValueError("temperature must be greater than zero")
    if config.num_prompts <= 0:
        raise ValueError("num_prompts must be greater than zero")
    if config.prompt_len <= 0:
        raise ValueError("prompt_len must be greater than zero")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if config.rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be greater than zero")
    if config.training_micro_batch_size <= 0:
        raise ValueError("training_micro_batch_size must be greater than zero")
    if config.rollout_engine in {"hf", "vllm"} and config.model == "synthetic":
        raise ValueError("--model must name a real model for hf/vllm rollout")


def _build_synthetic_model(config: LogprobCrossBenchmarkConfig) -> _TinyCausalLM:
    return _TinyCausalLM(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        seed=config.seed,
    )


def _build_batch_layout(
    sequences: Sequence[LogprobSequence],
    *,
    pad_token_id: int,
    device: torch.device,
) -> _BatchLayout:
    max_input_len = max(len(sequence.input_token_ids) for sequence in sequences)
    max_completion_len = max(sequence.completion_len for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_input_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    target_token_ids = torch.zeros(
        (len(sequences), max_completion_len),
        dtype=torch.long,
        device=device,
    )
    target_mask = torch.zeros_like(target_token_ids, dtype=torch.bool)
    logit_positions = torch.zeros_like(target_token_ids, dtype=torch.long)

    for row, sequence in enumerate(sequences):
        input_values = torch.tensor(sequence.input_token_ids, dtype=torch.long, device=device)
        input_ids[row, : input_values.numel()] = input_values
        attention_mask[row, : input_values.numel()] = True
        completion_values = torch.tensor(
            sequence.completion_token_ids,
            dtype=torch.long,
            device=device,
        )
        target_token_ids[row, : completion_values.numel()] = completion_values
        mask_values = torch.tensor(sequence.completion_mask, dtype=torch.bool, device=device)
        target_mask[row, : mask_values.numel()] = mask_values
        positions = torch.arange(
            sequence.prompt_len - 1,
            sequence.prompt_len + sequence.completion_len - 1,
            dtype=torch.long,
            device=device,
        )
        logit_positions[row, : positions.numel()] = positions

    return _BatchLayout(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_token_ids=target_token_ids,
        target_mask=target_mask,
        logit_positions=logit_positions,
    )


def _gather_position_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"expected logits with shape [B, S, V], got {tuple(logits.shape)}")
    gather_index = positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    return torch.gather(logits, dim=1, index=gather_index)


def _call_model(model: torch.nn.Module, input_ids: torch.Tensor, **kwargs: Any) -> Any:
    try:
        return model(input_ids=input_ids, **kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        try:
            return model(input_ids=input_ids, **kwargs)
        except TypeError:
            kwargs.pop("attention_mask", None)
            return model(input_ids)


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, Mapping):
        logits = model_output.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
    logits = getattr(model_output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(model_output, Sequence) and not isinstance(model_output, (str, bytes)):
        for item in model_output:
            try:
                return _extract_logits(item)
            except TypeError:
                continue
    raise TypeError(f"model output does not expose logits: {type(model_output)!r}")


def _pad_token_id(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "pad_token_id", None)
    if value is None:
        value = getattr(config, "eos_token_id", 0)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = value[0] if value else 0
    return int(value or 0)


def _torch_dtype_from_string(value: str) -> torch.dtype:
    normalized = value.lower().replace("torch.", "")
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _model_dtype_supported_on_device(dtype: torch.dtype, device: torch.device) -> bool:
    if device.type == "cpu" and dtype == torch.float16:
        return False
    return True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_markdown_summary(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    worst = report["worst_drift"]
    lines = [
        "# Logprob Cross-Engine Benchmark",
        "",
        f"- status: `{report['status']}`",
        f"- rollout_engine: `{report['rollout_engine']}`",
        f"- training_engine: `{report['training_engine']}`",
        f"- model: `{report['model']}`",
        f"- dtype: `{report['dtype']}`",
        f"- active_tokens: `{summary['active_tokens']}`",
        f"- max_abs_error: `{summary['max_abs_error']:.8g}`",
        f"- mean_abs_error: `{summary['mean_abs_error']:.8g}`",
        f"- max_relative_error: `{summary['max_relative_error']:.8g}`",
        "",
        "## Worst Drift",
        "",
        f"- sequence_id: `{worst['sequence_id']}`",
        f"- channel: `{worst['channel']}`",
        f"- completion_index: `{worst['completion_index']}`",
        f"- absolute_position: `{worst['absolute_position']}`",
        f"- target_token_id: `{worst['target_token_id']}`",
        f"- rollout_logprob: `{worst['rollout_logprob']:.8g}`",
        f"- training_logprob: `{worst['training_logprob']:.8g}`",
        f"- abs_error: `{worst['abs_error']:.8g}`",
    ]
    return "\n".join(lines) + "\n"


def _format_console_summary(report: Mapping[str, Any], output_dir: Path) -> str:
    summary = report["summary"]
    worst = report["worst_drift"]
    return (
        "RL-Kernel logprob cross-engine benchmark\n"
        f"status={report['status']} "
        f"rollout={report['rollout_engine']} training={report['training_engine']} "
        f"active_tokens={summary['active_tokens']} "
        f"max_abs={summary['max_abs_error']:.8g} "
        f"mean_abs={summary['mean_abs_error']:.8g} "
        f"worst={worst['sequence_id']}:{worst['completion_index']} "
        f"output_dir={output_dir}"
    )


def _coerce_int_list(value: Any, *, field_name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    raise ValueError(f"{field_name} must be a sequence of integers")


def _normalize_rollout_logprobs(
    payload: Mapping[str, Any],
    expected_len: int,
) -> dict[str, list[float]]:
    raw = (
        payload.get("rollout_logprobs")
        or payload.get("completion_logprobs")
        or payload.get("selected_logprobs")
        or payload.get("logprobs")
    )
    if raw is None:
        raise ValueError("sequence must include rollout_logprobs/logprobs")
    if isinstance(raw, Mapping):
        normalized = {str(key): _coerce_float_list(value) for key, value in raw.items()}
    else:
        normalized = {DEFAULT_POLICY_CHANNEL: _coerce_float_list(raw)}
    for channel, values in normalized.items():
        if len(values) != expected_len:
            raise ValueError(
                f"rollout_logprobs[{channel!r}] length {len(values)} does not match "
                f"completion length {expected_len}"
            )
    return normalized


def _coerce_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    raise ValueError("expected a sequence of floats")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "transformers_version": _safe_package_version("transformers"),
        "vllm_version": _safe_package_version("vllm"),
        "argv": sys.argv,
    }


def _safe_package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _public_config_dict(config: LogprobCrossBenchmarkConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for HF rollout/training paths. "
            "Install the project dependencies or use --rollout-engine=synthetic/fixture."
        ) from exc
    return transformers


def _load_prompt_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"prompts file does not exist: {path}")
    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, str):
                    records.append({"prompt": payload, "prompt_index": index})
                elif isinstance(payload, Mapping):
                    records.append({**payload, "prompt_index": payload.get("prompt_index", index)})
                else:
                    raise ValueError(f"unsupported prompt JSONL record at line {index + 1}")
        return records
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.rstrip("\n")
            if text:
                records.append({"prompt": text, "prompt_index": index})
    return records


def _encode_prompt_batch(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if all("prompt_token_ids" in record for record in records):
        token_lists = [
            _coerce_int_list(record["prompt_token_ids"], field_name="prompt_token_ids")
            for record in records
        ]
        max_len = max(len(tokens) for tokens in token_lists)
        pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
        input_ids = torch.full((len(token_lists), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, tokens in enumerate(token_lists):
            values = torch.tensor(tokens, dtype=torch.long)
            input_ids[row, -values.numel() :] = values
            attention_mask[row, -values.numel() :] = 1
        return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}

    texts = [_prompt_text(record) for record in records]
    if any(text is None for text in texts):
        raise ValueError("prompt records must contain either prompt text or prompt_token_ids")
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _prompt_tokens_from_record_or_batch(
    tokenizer: Any,
    record: Mapping[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[int]:
    if "prompt_token_ids" in record:
        return _coerce_int_list(record["prompt_token_ids"], field_name="prompt_token_ids")
    active = input_ids[attention_mask.bool()].detach().cpu().tolist()
    return [int(item) for item in active]


def _prompt_text(record: Mapping[str, Any]) -> str | None:
    value = record.get("prompt") or record.get("prompt_text") or record.get("text")
    return _optional_str(value)


def _prompt_payload_for_vllm(record: Mapping[str, Any]) -> str | dict[str, Any]:
    if "prompt_token_ids" in record:
        return {
            "prompt_token_ids": _coerce_int_list(
                record["prompt_token_ids"],
                field_name="prompt_token_ids",
            )
        }
    text = _prompt_text(record)
    if text is None:
        raise ValueError("vLLM prompt records must contain prompt text or prompt_token_ids")
    return text


def _prompt_token_ids_from_record(record: Mapping[str, Any], tokenizer: Any | None) -> list[int]:
    if "prompt_token_ids" in record:
        return _coerce_int_list(record["prompt_token_ids"], field_name="prompt_token_ids")
    text = _prompt_text(record)
    if text is None or tokenizer is None:
        return []
    return [int(item) for item in tokenizer(text, add_special_tokens=True)["input_ids"]]


def _trim_generated_tokens(
    token_ids: Sequence[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> list[int]:
    trimmed: list[int] = []
    for token in token_ids:
        token_int = int(token)
        if pad_token_id is not None and token_int == int(pad_token_id):
            if eos_token_id is None or trimmed:
                break
        trimmed.append(token_int)
        if eos_token_id is not None and token_int == int(eos_token_id):
            break
    return trimmed


def _selected_logprobs_from_rollout_payload(
    raw_logprobs: Any,
    token_ids: Sequence[int],
) -> list[float]:
    if raw_logprobs is None:
        raise ValueError("rollout output did not include per-token logprobs")
    if len(raw_logprobs) < len(token_ids):
        raise ValueError("rollout logprobs are shorter than generated token_ids")
    selected: list[float] = []
    for index, token_id in enumerate(token_ids):
        item = raw_logprobs[index]
        selected.append(_selected_logprob_from_item(item, int(token_id)))
    return selected


def _selected_logprob_from_item(item: Any, token_id: int) -> float:
    if isinstance(item, (float, int)):
        return float(item)
    if isinstance(item, Mapping):
        for key in (token_id, str(token_id)):
            if key in item:
                return _logprob_value(item[key])
        token_id_value = item.get("token_id")
        if "logprob" in item and (token_id_value is None or int(token_id_value) == token_id):
            return float(item["logprob"])
    value = getattr(item, "logprob", None)
    if value is not None:
        return float(value)
    raise ValueError(f"could not extract selected logprob for token id {token_id}")


def _logprob_value(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, Mapping) and "logprob" in value:
        return float(value["logprob"])
    attr = getattr(value, "logprob", None)
    if attr is not None:
        return float(attr)
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
