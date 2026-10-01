# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Explicit runtime adapters and replacement capabilities, independent of diagnosis."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Replacement:
    name: str
    group: str
    modules: tuple[str, ...]
    eligible: bool
    reasons: tuple[str, ...] = ()
    validation: str = "requires_same_replay_verification"


class RuntimeAdapter(Protocol):
    """Installed entry points return an instance implementing this protocol.

    run owns faithful capture/launch and must return CLI exit code 0/1/2.
    Merely recognizing an architecture must not imply replacement eligibility.
    """

    name: str

    def matches(self, model: dict[str, Any], backend: str) -> bool: ...

    def replacements(self, model: dict[str, Any], args: Any) -> list[Replacement]: ...

    def run(self, paths: Any, profile: dict[str, Any], args: Any) -> int: ...


class Qwen3Adapter:
    name = "qwen3-dense-vime"

    def matches(self, model: dict[str, Any], backend: str) -> bool:
        # These Dense shapes have explicit launcher contracts; recognition is
        # only eligibility, and never substitutes for verification on a replay.
        dimensions = {
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        }
        semantics = {
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "use_sliding_window": False,
            "rope_scaling": None,
        }
        return (
            backend in {"cuda", "rocm"}
            and model.get("model_type") == "qwen3"
            and tuple(
                model.get(k)
                for k in (
                    "num_hidden_layers",
                    "hidden_size",
                    "intermediate_size",
                    "num_attention_heads",
                )
            )
            + (model.get("tie_word_embeddings", False),)
            in {(36, 4096, 12288, 32, False), (28, 1024, 3072, 16, True)}
            and all(model.get(k) == v for k, v in dimensions.items())
            and all(model.get(k, v) == v for k, v in semantics.items())
            and not model.get("quantization_config")
            and model.get("dtype", model.get("torch_dtype")) in {"bfloat16", "bf16"}
        )

    def replacements(self, model: dict[str, Any], args: Any) -> list[Replacement]:
        reasons = []
        if not self.matches(model, args.backend):
            reasons.append(
                "model/backend/dtype does not match a supported Qwen3 Dense BF16 contract"
            )
        if model.get("tie_word_embeddings"):
            reasons.append(
                "tied embedding/output weights have no supported strict replacement; "
                "native capture remains available"
            )
        from rl_engine import repro

        try:
            repro._validate_topology_args(args)
        except repro.ReproError as exc:
            reasons.append(str(exc))
        for tp in (args.tp_size, args.rollout_tp_size):
            if any(
                model.get(k, 0) % tp
                for k in (
                    "num_attention_heads",
                    "num_key_value_heads",
                    "intermediate_size",
                )
            ):
                reasons.append(f"model dimensions are not divisible by TP={tp}")
        if not math.isfinite(args.rollout_temperature) or args.rollout_temperature <= 0:
            reasons.append("this adapter requires finite temperature > 0")
        if not 0 < args.rollout_top_p <= 1:
            reasons.append("top-p must be in (0, 1]")
        if args.rollout_top_k != -1:
            reasons.append("strict replay replacement supports top-k=-1 only")
        return [
            Replacement(
                "rl-kernel/qwen3-dense",
                "M111",
                ("attention", "ffn", "logp"),
                not reasons,
                tuple(reasons),
            )
        ]

    def run(self, paths: Any, profile: dict[str, Any], args: Any) -> int:
        from .session import run_debug

        return run_debug(paths, profile, args, adapter=self)


def megatron_model_args(model: dict[str, Any]) -> list[str]:
    """Derive the diagnostic training architecture from the actual HF checkpoint."""
    if not Qwen3Adapter().matches(model, "rocm"):
        raise ValueError("unsupported Qwen3 diagnostic model configuration")
    args = [
        "--swiglu",
        "--group-query-attention",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--normalization",
        "RMSNorm",
        "--qk-layernorm",
    ]
    for flag, key in (
        ("num-layers", "num_hidden_layers"),
        ("hidden-size", "hidden_size"),
        ("ffn-hidden-size", "intermediate_size"),
        ("num-attention-heads", "num_attention_heads"),
        ("num-query-groups", "num_key_value_heads"),
        ("kv-channels", "head_dim"),
        ("vocab-size", "vocab_size"),
        ("norm-epsilon", "rms_norm_eps"),
        ("rotary-base", "rope_theta"),
    ):
        args.extend((f"--{flag}", str(model[key])))
    if not model["tie_word_embeddings"]:
        args.append("--untie-embeddings-and-output-weights")
    return args


def select_adapter(model: dict[str, Any], backend: str) -> RuntimeAdapter:
    adapters: list[RuntimeAdapter] = [Qwen3Adapter()]
    for point in entry_points(group="rlkernel.debug_adapters"):
        adapters.append(point.load()())
    names = [adapter.name for adapter in adapters]
    if len(names) != len(set(names)):
        raise ValueError("duplicate installed diagnostic adapter names")
    matching = [adapter for adapter in adapters if adapter.matches(model, backend)]
    if len(matching) != 1:
        raise ValueError(
            f"{'Ambiguous' if matching else 'Unsupported'} live adapter for "
            f"model_type={model.get('model_type', 'unknown')}, backend={backend}. "
            "Install a matching rlkernel.debug_adapters plugin, or export a portable "
            "Capture bundle and run rlk debug on it. Existing evidence remains diagnosable."
        )
    return matching[0]


def run_debug(paths: Any, profile: dict[str, Any], args: Any) -> int:
    from rl_engine import repro
    from .replay import source_configuration

    try:
        source = source_configuration(args.source)
        if source.get("model_root") and "--model-root" not in args.explicit_flags:
            paths = replace(paths, model_root=Path(source["model_root"]))
        config = paths.model_root / "config.json"
        if not config.is_file():
            raise ValueError(f"Cannot identify model architecture: missing {config}")
        model = json.loads(config.read_text(encoding="utf-8"))
        adapter = select_adapter(model, args.backend)
        args.debug_model = model
        print(
            f"[adapter] {adapter.name}; model={model.get('model_type')}",
            file=sys.stderr if getattr(args, "as_json", False) else sys.stdout,
            flush=True,
        )
        return adapter.run(paths, profile, args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise repro.ReproError(str(exc)) from exc
