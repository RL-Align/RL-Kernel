# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Opt-in fixed-token replay and observational Qwen3 diagnostic hooks.

No production forward is reimplemented here. Sampling still computes the
requested distribution; only the selected token is substituted, before the
framework gathers its logprob. This is teacher-forced diagnosis, not generation.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

from .core import atomic_save, fingerprint, snapshot

_BATCH: dict[str, Any] | None = None
_ROWS: dict[str, torch.Tensor] = {}
_CALLS: dict[tuple[str, int], int] = {}
_WEIGHTS: dict[tuple[str, int], dict[str, str]] = {}
_LAYER_COUNTS: dict[str, int] = {}
_HEAD_CALLS: dict[str, int] = {}
_MARKER = "_rlk_frozen_diagnostic_original"


def capture_guard(side: str, stage: str):
    """Keep worker exceptions visible even if the framework drops an RPC error."""

    def decorate(function):
        @functools.wraps(function)
        def guarded(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as error:
                try:
                    root = Path(os.environ["RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR"])
                    atomic_save(
                        root / "errors" / f"{side}-pid{os.getpid()}-{stage}.pt",
                        {"side": side, "stage": stage, "error": f"{type(error).__name__}: {error}"},
                    )
                except Exception as recording_error:
                    error.add_note(f"Could not persist capture failure: {recording_error}")
                raise

        return guarded

    return decorate


def topology_coordinates(side: str) -> tuple[int, int, int, int]:
    value = topology(side)
    if len(value) == 2:  # compatibility with lightweight unit-test shims
        return value[0], value[1], 0, 1
    return value


def is_framework_warmup() -> bool:
    """Identify real vLLM warmup calls, never infer them from zero-valued data."""
    frame = sys._getframe(1)
    while frame is not None:
        if str(frame.f_globals.get("__name__", "")).startswith(
            "vllm."
        ) and frame.f_code.co_name in {
            "_dummy_run",
            "profile_run",
            "_dummy_sampler_run",
            "warmup_kernels",
        }:
            return True
        frame = frame.f_back
    return False


def install_layers(side: str) -> None:
    """Observe every layer without replacing its numerical implementation."""
    if side == "training":
        from megatron.core.transformer.transformer_layer import TransformerLayer as Layer
    else:
        from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer as Layer
    if hasattr(Layer.forward, _MARKER):
        return
    original = Layer.forward
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if side not in _ROWS:
            return original(instance, *args, **kwargs)
        if side == "training":
            layer = int(instance.layer_number) - 1
        else:
            attention = instance.self_attn
            prefix = getattr(attention.attn, "layer_name", "")
            match = re.search(r"layers\.(\d+)", prefix)
            if match is None:
                raise RuntimeError(f"cannot identify Qwen3 layer: {prefix!r}")
            layer = int(match.group(1))
        if side == "training" and _CALLS.get((side, layer), 0):
            return original(instance, *args, **kwargs)
        bound = signature.bind(instance, *args, **kwargs).arguments
        # Megatron's instrumentation can wrap forward as (*args, **kwargs)
        # without preserving its signature. Bind its public first tensor
        # argument rather than treating wrapper parameter names as semantics.
        hidden = bound.get("hidden_states", kwargs.get("hidden_states"))
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("decoder layer did not expose its actual hidden-state input")
        residual = bound.get("residual")
        tensors: dict[str, torch.Tensor] = {
            "input": snapshot(hidden if residual is None else hidden + residual)
        }
        handles = []

        def result_tensor(result: Any) -> torch.Tensor | None:
            candidate = result[0] if isinstance(result, tuple) else result
            return candidate if isinstance(candidate, torch.Tensor) else None

        def observe(module: Any, name: str, input_names: tuple[str, ...] = ()) -> None:
            def before(_module: Any, values: tuple[Any, ...], keywords: dict[str, Any]) -> None:
                for i, key in enumerate(input_names):
                    value = values[i] if i < len(values) else keywords.get(key)
                    if isinstance(value, torch.Tensor):
                        tensors[key] = snapshot(value)

            def after(_module: Any, _args: Any, result: Any) -> None:
                value = result_tensor(result)
                if value is not None:
                    tensors[name] = snapshot(value)

            handles.extend(
                (
                    module.register_forward_pre_hook(before, with_kwargs=True),
                    module.register_forward_hook(after),
                )
            )

        if side == "training":
            attention = instance.self_attention
            observe(attention.linear_qkv, "qkv")
            observe(attention.q_layernorm, "q_norm")
            observe(attention.k_layernorm, "k_norm")
            observe(attention.core_attention, "attention_core", ("query", "key", "value"))
            observe(attention, "attention_output")
            observe(instance.mlp.linear_fc1, "mlp_gate_up")
            observe(instance.mlp.linear_fc2, "mlp_projection", ("mlp_activation",))
            observe(instance.mlp, "mlp_output")
        else:
            attention = instance.self_attn
            observe(instance.input_layernorm, "attention_norm")
            observe(attention.qkv_proj, "qkv")
            observe(attention.q_norm, "q_norm")
            observe(attention.k_norm, "k_norm")
            observe(attention.attn, "attention_core", ("query", "key", "value"))
            observe(attention, "attention_output")
            observe(instance.post_attention_layernorm, "mlp_norm")
            observe(instance.mlp.gate_up_proj, "mlp_gate_up")
            observe(instance.mlp.act_fn, "mlp_activation")
            observe(instance.mlp, "mlp_output")
        try:
            result = original(instance, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        if side == "training":
            tensors["output"] = snapshot(result_tensor(result))
        else:
            tensors["output"] = snapshot(result[0] + result[1])
            tensors["attention_residual"] = snapshot(result[1])
        # Metadata is observed, not a second source of truth for execution.
        config = getattr(instance, "config", None)
        metadata = {
            "side": side,
            "grad_enabled": torch.is_grad_enabled(),
            "head_dim": getattr(attention, "head_dim", getattr(config, "kv_channels", None)),
            "scale": getattr(attention, "scaling", None),
            "capture_mode": "eager_observational",
            "expected_layers": _LAYER_COUNTS.get(side),
        }
        if side == "training":
            metadata["contract"] = {
                "scale": float(
                    getattr(attention.core_attention, "softmax_scale", config.kv_channels**-0.5)
                ),
                "q_norm_eps": float(
                    getattr(attention.q_layernorm, "eps", config.layernorm_epsilon)
                ),
                "k_norm_eps": float(
                    getattr(attention.k_layernorm, "eps", config.layernorm_epsilon)
                ),
                "attention_dropout": float(config.attention_dropout),
            }
            packed = bound.get("packed_seq_params")
            metadata["sequence_boundaries"] = snapshot(getattr(packed, "cu_seqlens_q", None))
            metadata["attention_mask"] = snapshot(bound.get("attention_mask"))
        else:
            metadata["contract"] = {
                "scale": float(attention.scaling),
                "q_norm_eps": float(attention.q_norm.variance_epsilon),
                "k_norm_eps": float(attention.k_norm.variance_epsilon),
                "attention_dropout": 0.0,
            }
        tensors, layout = canonical_tensors(instance, side, tensors)
        metadata.update(layout)
        if (side, layer) not in _WEIGHTS:
            _WEIGHTS[side, layer] = weight_tiles(instance, side)
            root = Path(os.environ["RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR"])
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            path = root / "weights" / f"{side}-pid{os.getpid()}-rank{rank}-layer{layer}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_WEIGHTS[side, layer], sort_keys=True), encoding="utf-8")
        save_layer(side, layer, tensors, metadata=metadata)
        return result

    setattr(forward, _MARKER, original)
    Layer.forward = capture_guard(side, "layer")(forward)


def install(side: str) -> None:
    if frozen_batch() is None:
        return
    install_root_context(side)
    install_layers(side)
    if side == "rollout":
        install_fixed_sampler()
        install_rollout_head()
    else:
        install_training_identity()


def write_identity(side: str, tokens: Any, mask: Any, sampling: dict[str, Any]) -> None:
    batch = frozen_batch()
    actual_tokens = torch.as_tensor(tokens).reshape(-1).long()
    actual_mask = torch.as_tensor(mask).reshape(-1).long()
    if not torch.equal(actual_tokens.cpu(), torch.tensor(batch["tokens"])):
        raise RuntimeError(f"{side} batch tokens differ from frozen replay")
    if not torch.equal(actual_mask.cpu(), torch.tensor(batch["mask"])):
        raise RuntimeError(f"{side} active mask differs from frozen replay")
    if sampling != batch["sampling"]:
        raise RuntimeError(f"{side} sampling parameters differ from frozen replay")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = Path(os.environ["RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR"]) / "identity"
    path.mkdir(parents=True, exist_ok=True)
    value = {
        "tokens": fingerprint(actual_tokens),
        "mask": fingerprint(actual_mask),
        "sampling": sampling,
        "side": side,
    }
    (path / f"{side}-pid{os.getpid()}-rank{rank}.json").write_text(
        json.dumps(value), encoding="utf-8"
    )


def install_training_identity() -> None:
    from vime.backends.megatron_utils import actor

    # Ray may deserialize an actor class independently of the module's class
    # object. Observe its module-level data boundary, not a class copy.
    original = actor.process_rollout_data
    if hasattr(original, _MARKER):
        return

    @functools.wraps(original)
    def get_data(run_args: Any, *args: Any, **kwargs: Any):
        result = original(run_args, *args, **kwargs)
        sampling = {
            key: getattr(run_args, f"rollout_{key}") for key in ("temperature", "top_p", "top_k")
        }
        for tokens, mask in zip(result["tokens"], result["loss_masks"], strict=True):
            write_identity("training", tokens, mask, sampling)
        return result

    setattr(get_data, _MARKER, original)
    actor.process_rollout_data = get_data


def topology(side: str) -> tuple[int, int, int, int]:
    if side == "training":
        from megatron.core import mpu

        return (
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_tensor_model_parallel_world_size(),
            mpu.get_context_parallel_rank() if mpu.get_context_parallel_world_size() > 1 else 0,
            mpu.get_context_parallel_world_size(),
        )
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    # vLLM diagnostic runs normally use rollout CP=1. Keep the coordinate
    # explicit so a future CP>1 run cannot merge distinct partitions.
    cp_rank, cp_world = 0, 1
    try:
        from vllm.distributed import get_pcp_group

        cp_group = get_pcp_group()
        if cp_group is not None:
            cp_rank, cp_world = cp_group.rank_in_group, cp_group.world_size
    except (ImportError, AttributeError):
        pass
    return group.rank_in_group, group.world_size, cp_rank, cp_world


def canonical_tensors(instance: Any, side: str, tensors: dict[str, torch.Tensor]):
    """Normalize Qwen3 grouped QKV and gate/up ownership, not its arithmetic."""
    rank, world, cp_rank, cp_world = topology_coordinates(side)
    rows = _ROWS[side]
    config = getattr(instance, "config", None)
    attn = instance.self_attention if side == "training" else instance.self_attn
    head = int(config.kv_channels if side == "training" else attn.head_dim)
    groups = int(config.num_query_groups if side == "training" else attn.total_num_kv_heads)
    total_heads = int(config.num_attention_heads if side == "training" else attn.total_num_heads)
    local_heads, local_groups = total_heads // world, max(1, groups // world)
    group_rank = rank if groups >= world else rank // (world // groups)
    feature_offsets: dict[str, int] = {}
    for name in ("query", "q_norm", "attention_core", "q_projection"):
        feature_offsets[name] = rank * local_heads * head
    for name in ("key", "k_norm", "value", "k_projection", "v_projection"):
        feature_offsets[name] = group_rank * local_groups * head
    packed = tensors.pop("qkv", None)
    if packed is not None:
        packed = packed.reshape(packed.shape[0], -1)
        if side == "training":
            grouped = packed.reshape(packed.shape[0], local_groups, -1, head)
            q = grouped[:, :, :-2].reshape(packed.shape[0], -1)
            k = grouped[:, :, -2].reshape(packed.shape[0], -1)
            v = grouped[:, :, -1].reshape(packed.shape[0], -1)
        else:
            q, k, v = packed.split(
                (local_heads * head, local_groups * head, local_groups * head), dim=-1
            )
        tensors.update(q_projection=q, k_projection=k, v_projection=v)
    gate_up = tensors.pop("mlp_gate_up", None)
    if gate_up is not None:
        gate_up = gate_up.reshape(gate_up.shape[0], -1)
        tensors["mlp_gate"], tensors["mlp_up"] = gate_up.chunk(2, dim=-1)
    positions = {}
    normalized = {}
    feature_sizes = {}
    for name, value in tensors.items():
        # Megatron's packed shape is [S,1,H], serving uses [T,H].
        value = value.reshape(value.shape[0], -1)
        if value.shape[0] == rows.numel():
            owned = rows
        elif side == "training" and value.shape[0] * world == rows.numel():
            owned = rows.chunk(world)[rank]  # sequence-parallel token partition
        else:
            raise RuntimeError(
                f"cannot prove token ownership for {side}/{name}: {value.shape} vs {rows.shape}"
            )
        positions[name] = owned
        normalized[name] = value
        feature_sizes[name] = value.shape[-1]
        if name in {"mlp_gate", "mlp_up", "mlp_activation"}:
            feature_offsets[name] = rank * value.shape[-1]
            feature_sizes[name] *= world
        elif name in {"query", "q_norm", "attention_core", "q_projection"}:
            feature_sizes[name] = total_heads * head
        elif name in {"key", "k_norm", "value", "k_projection", "v_projection"}:
            feature_sizes[name] = groups * head
    return normalized, {
        "stage_positions": positions,
        "feature_offsets": feature_offsets,
        "feature_sizes": feature_sizes,
        "tp_rank": rank,
        "tp_size": world,
        "cp_rank": cp_rank,
        "cp_size": cp_world,
        "head_dim": head,
        "total_heads": total_heads,
        "total_kv_heads": groups,
    }


def tensor_tiles(name: str, weight: torch.Tensor, row: int = 0, col: int = 0):
    result = {}
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(f"cannot inspect weight {name}")
    value = snapshot(weight)
    if value.ndim == 1:
        result[name] = fingerprint(value)
        return result
    # Vocabulary padding can differ between frameworks (e.g. 37984 vs
    # 38016 rows per TP shard). Each complete vocabulary row is canonical;
    # a 64-row tile spanning such a boundary is not.
    vocab_rows = name == "lm_head"
    if value.ndim != 2 or (col != 0 if vocab_rows else row % 64 or col % 64):
        raise RuntimeError(f"unsupported weight tile ownership for {name}")
    row_step, col_step = (1, value.shape[1]) if vocab_rows else (64, 64)
    for r in range(0, value.shape[0], row_step):
        for c in range(0, value.shape[1], col_step):
            tile = value[r : r + row_step, c : c + col_step].contiguous()
            raw = tile.view(torch.uint8).numpy().tobytes()
            result[f"{name}:{row + r}:{col + c}:{tuple(tile.shape)}:{tile.dtype}"] = hashlib.sha256(
                raw
            ).hexdigest()
    return result


def weight_tiles(instance: Any, side: str) -> dict[str, str]:
    """Hash logical weight tiles, so different physical TP splits compare exactly."""
    rank, world, _cp_rank, _cp_world = topology_coordinates(side)
    result: dict[str, str] = {}

    def add(name: str, weight: torch.Tensor, row: int = 0, col: int = 0):
        result.update(tensor_tiles(name, weight, row, col))

    config = getattr(instance, "config", None)
    attn = instance.self_attention if side == "training" else instance.self_attn
    mlp = instance.mlp
    head = int(config.kv_channels if side == "training" else attn.head_dim)
    groups = int(config.num_query_groups if side == "training" else attn.total_num_kv_heads)
    heads = int(config.num_attention_heads if side == "training" else attn.total_num_heads)
    local_groups, local_heads = max(1, groups // world), heads // world
    group_rank = rank if groups >= world else rank // (world // groups)
    if side == "training":
        qkv = attn.linear_qkv.weight
        packed = qkv.reshape(local_groups, -1, head, qkv.shape[-1])
        q = packed[:, :-2].reshape(-1, qkv.shape[-1])
        k, v = packed[:, -2].reshape(-1, qkv.shape[-1]), packed[:, -1].reshape(-1, qkv.shape[-1])
        gate, up = mlp.linear_fc1.weight.chunk(2, dim=0)
        proj, down = attn.linear_proj.weight, mlp.linear_fc2.weight
        add("attention_norm", attn.linear_qkv.layer_norm_weight)
        add("mlp_norm", mlp.linear_fc1.layer_norm_weight)
        add("q_norm", attn.q_layernorm.weight)
        add("k_norm", attn.k_layernorm.weight)
    else:
        q, k, v = attn.qkv_proj.weight.split(
            (local_heads * head, local_groups * head, local_groups * head), dim=0
        )
        gate, up = mlp.gate_up_proj.weight.chunk(2, dim=0)
        proj, down = attn.o_proj.weight, mlp.down_proj.weight
        add("attention_norm", instance.input_layernorm.weight)
        add("mlp_norm", instance.post_attention_layernorm.weight)
        add("q_norm", attn.q_norm.weight)
        add("k_norm", attn.k_norm.weight)
    add("q", q, row=rank * q.shape[0])
    add("k", k, row=group_rank * k.shape[0])
    add("v", v, row=group_rank * v.shape[0])
    add("o", proj, col=rank * proj.shape[1])
    add("gate", gate, row=rank * gate.shape[0])
    add("up", up, row=rank * up.shape[0])
    add("down", down, col=rank * down.shape[1])
    return result


def frozen_batch() -> dict[str, Any] | None:
    global _BATCH
    path = os.getenv("RL_KERNEL_ALIGNMENT_REPLAY", "")
    if not path:
        return None
    if _BATCH is None:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        tokens = value.get("tokens")
        prompt = value.get("prompt_length")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or type(prompt) is not int
            or not 0 < prompt < len(tokens)
        ):
            raise ValueError("invalid frozen tokens/prompt_length")
        if len(value.get("mask", [])) != len(tokens) - prompt:
            raise ValueError("frozen response mask length differs from tokens")
        _BATCH = value
    return _BATCH


def validate_rows(tokens: torch.Tensor, positions: torch.Tensor) -> None:
    batch = frozen_batch()
    if batch is None:
        return
    tokens, positions = snapshot(tokens).reshape(-1), snapshot(positions).reshape(-1).long()
    active = positions >= 0
    if positions.numel() != tokens.numel() or bool(
        (positions[active] >= len(batch["tokens"])).any()
    ):
        raise RuntimeError("diagnostic token/position shape or bounds mismatch")
    expected = torch.tensor(batch["tokens"], dtype=torch.long)[positions[active]]
    if not torch.equal(tokens[active].long(), expected):
        raise RuntimeError("runtime tokens differ from the frozen replay at logical positions")


def install_root_context(side: str) -> None:
    if frozen_batch() is None:
        return
    if side == "training":
        from megatron.core.models.gpt.gpt_model import GPTModel as Model
    else:
        from vllm.model_executor.models.qwen3 import Qwen3Model as Model
    if hasattr(Model.forward, _MARKER):
        return
    original = Model.forward
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(instance, *args, **kwargs).arguments
        tokens = bound.get("input_ids")
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError("fixed replay requires token IDs, not inputs_embeds")
        if side == "rollout":
            from vllm.forward_context import get_forward_context

            if get_forward_context().attn_metadata is None or is_framework_warmup():
                _ROWS.pop(side, None)
                return original(instance, *args, **kwargs)
            positions = bound.get("positions")
            if not isinstance(positions, torch.Tensor):
                raise RuntimeError("rollout did not expose position IDs")
        else:
            from vime.backends.megatron_utils.cp_utils import slice_with_cp

            batch = frozen_batch()
            positions = slice_with_cp(
                torch.arange(len(batch["tokens"]), device=tokens.device), -1
            ).reshape(-1)
            if positions.numel() > tokens.numel():
                raise RuntimeError("training replay unexpectedly truncated the frozen sequence")
            positions = torch.nn.functional.pad(
                positions, (0, tokens.numel() - positions.numel()), value=-1
            )
        validate_rows(tokens, positions)
        _ROWS[side] = snapshot(positions).reshape(-1).long()
        config = getattr(instance, "config", None)
        count = getattr(config, "num_layers", getattr(config, "num_hidden_layers", None))
        if count is None:
            container = getattr(instance, "decoder", instance)
            count = len(container.layers)
        _LAYER_COUNTS[side] = int(count)
        handles = []
        head_input = {}
        if side == "training" and not _HEAD_CALLS.get(side, 0):

            def before(module: Any, values: tuple[Any, ...], keywords: dict[str, Any]):
                head_input["hidden"] = snapshot(values[0])
                weight = keywords.get("weight", values[1] if len(values) > 1 else None)
                head_input["weight"] = module.weight if weight is None else weight

            def after(module: Any, _args: Any, result: Any):
                logits = result[0] if isinstance(result, tuple) else result
                save_head(side, head_input["hidden"], logits, head_input["weight"])

            handles = [
                instance.output_layer.register_forward_pre_hook(before, with_kwargs=True),
                instance.output_layer.register_forward_hook(after),
            ]
        try:
            return original(instance, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

    setattr(forward, _MARKER, original)
    Model.forward = capture_guard(side, "model")(forward)


def install_rollout_head() -> None:
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    original = Qwen3ForCausalLM.compute_logits
    if hasattr(original, _MARKER):
        return

    @functools.wraps(original)
    def compute(instance: Any, hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        logits = original(instance, hidden_states, *args, **kwargs)
        if "rollout" in _ROWS:
            # Logits may be gathered only on the engine's output rank. All TP
            # ranks still contribute actual LM-head weight tiles.
            save_head("rollout", hidden_states, logits, instance.lm_head.weight)
        return logits

    setattr(compute, _MARKER, original)
    Qwen3ForCausalLM.compute_logits = capture_guard("rollout", "head")(compute)


def save_head(side: str, hidden: torch.Tensor, logits: Any, weight: torch.Tensor) -> None:
    rank, world, cp_rank, cp_world = topology_coordinates(side)
    rows = _ROWS[side]
    real_vocab = int(os.environ["RL_KERNEL_VLLM_REAL_VOCAB_SIZE"])
    offset = rank * weight.shape[0]
    call = _HEAD_CALLS.get(side, 0)
    root = Path(os.environ["RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR"])
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if call == 0:
        tiles = tensor_tiles("lm_head", weight[: max(0, real_vocab - offset)], row=offset)
        path = root / "weights" / f"{side}-pid{os.getpid()}-rank{global_rank}-head.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tiles), encoding="utf-8")
    _HEAD_CALLS[side] = call + 1
    if logits is None:
        return
    hidden = snapshot(hidden).reshape(hidden.shape[0], -1)
    logits = snapshot(logits).reshape(logits.shape[0], -1)
    if side == "rollout":
        # One last query per packed request is passed to compute_logits.
        ends = torch.cat(
            ((rows[1:] <= rows[:-1]).nonzero().flatten(), torch.tensor([rows.numel() - 1]))
        )
        owned = rows[ends]
        if owned.numel() != logits.shape[0]:
            raise RuntimeError("cannot align rollout LM-head rows with request positions")
        positions = {"head_input": owned, "logits": owned}
        offset = 0  # serving compute_logits returns the gathered real vocabulary
    else:
        positions = {"logits": rows}
        positions["head_input"] = (
            rows if hidden.shape[0] == rows.numel() else rows.chunk(world)[rank]
        )
    logits = logits[:, : max(0, real_vocab - offset)]
    metadata = {
        "stage_positions": positions,
        "feature_offsets": {"logits": offset},
        "feature_sizes": {"head_input": hidden.shape[-1], "logits": real_vocab},
        "tp_rank": rank,
        "tp_size": world,
        "cp_rank": cp_rank,
        "cp_size": cp_world,
    }
    atomic_save(
        root / "heads" / f"{side}-pid{os.getpid()}-rank{global_rank}-call{call}.pt",
        {
            "side": side,
            "call": call,
            "metadata": metadata,
            "tensors": {"head_input": hidden, "logits": logits},
        },
    )


def install_fixed_sampler() -> None:
    if frozen_batch() is None:
        return
    from vllm.v1.sample.sampler import Sampler

    classes = [Sampler]
    try:
        from vllm.v1.worker.gpu.sample.sampler import Sampler as WorkerSampler

        classes.append(WorkerSampler)
    except ImportError:
        pass
    for cls in classes:
        if hasattr(cls.sample, _MARKER):
            continue
        original = cls.sample
        signature = inspect.signature(original)

        def wrapped(
            instance: Any,
            *args: Any,
            _original: Any = original,
            _signature: Any = signature,
            **kwargs: Any,
        ) -> Any:
            result = _original(instance, *args, **kwargs)
            rows = _ROWS.get("rollout")
            if rows is None:
                return result  # allocator/profile invocation, not a real request
            bound = _signature.bind(instance, *args, **kwargs).arguments
            batch = frozen_batch()
            for key, expected in batch.get("sampling", {}).items():
                metadata = bound.get("sampling_metadata")
                if metadata is not None:
                    actual = getattr(metadata, key, None)
                    if actual is None:
                        actual = 0.0 if key == "temperature" else (1.0 if key == "top_p" else -1)
                else:
                    states = getattr(instance, "sampling_states", None)
                    state = getattr(states, key, None)
                    if state is None or not hasattr(state, "np"):
                        raise RuntimeError(f"worker sampler did not expose actual {key}")
                    actual = state.np[bound["idx_mapping_np"]]
                actual = torch.as_tensor(actual).detach().cpu()
                if key == "top_k" and expected == -1:
                    # vLLM GPU SamplingStates stores disabled top-k as vocab_size.
                    # Compare equivalent domains, not framework sentinel spelling.
                    vocab = getattr(getattr(instance, "sampling_states", None), "vocab_size", None)
                    if vocab is None and isinstance(bound.get("logits"), torch.Tensor):
                        vocab = bound["logits"].shape[-1]
                    if vocab is not None:
                        actual = torch.where(actual == vocab, -1, actual)
                expected_tensor = torch.full_like(actual, expected)
                if not torch.equal(actual, expected_tensor):
                    raise RuntimeError(f"sampler {key} differs from frozen replay")
            positions = bound.get("pos")
            if positions is None:
                metadata = bound.get("sampling_metadata")
                histories = getattr(metadata, "output_token_ids", None)
                if histories is not None:
                    prompt_length = frozen_batch()["prompt_length"]
                    positions = torch.tensor([prompt_length + len(ids) - 1 for ids in histories])
                else:
                    positions = rows[-1:]
            positions = snapshot(positions).reshape(-1).long()
            sampled, processed = result
            if sampled.numel() != positions.numel():
                raise RuntimeError(
                    "fixed diagnostic replay cannot align sampler rows; "
                    "disable speculative decoding"
                )
            batch = frozen_batch()
            next_positions = positions + 1
            if bool((next_positions >= len(batch["tokens"])).any()):
                raise RuntimeError("sampler ran beyond the frozen response")
            forced = torch.tensor(batch["tokens"], dtype=sampled.dtype)[next_positions]
            return forced.to(sampled.device).reshape_as(sampled), processed

        setattr(wrapped, _MARKER, original)
        cls.sample = capture_guard("rollout", "sampler")(wrapped)


def generate_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False):
    """Vime's normal rollout with one explicitly frozen sample."""
    from vime.rollout import vllm_rollout

    batch = frozen_batch()
    if batch is None:
        raise RuntimeError("missing frozen replay file")
    for key, value in batch["sampling"].items():
        actual = getattr(args, f"rollout_{key}")
        if actual != value:
            raise RuntimeError(f"replay sampling {key}: expected {value}, got {actual}")
    original = vllm_rollout.generate
    original_builder = vllm_rollout._build_inference_sampling_params

    def build_parameters(params: dict[str, Any]) -> dict[str, Any]:
        value = original_builder(params)
        value.update(ignore_eos=True, stop=[], stop_token_ids=[])
        return value

    async def generate(run_args: Any, sample: Any, sampling_params: dict[str, Any]):
        sample.tokens = list(batch["tokens"][: batch["prompt_length"]])
        params = dict(sampling_params)
        params.update(max_new_tokens=len(batch["tokens"]) - batch["prompt_length"], ignore_eos=True)
        result = await original(run_args, sample, params)
        if list(result.tokens) != batch["tokens"]:
            raise RuntimeError("rollout did not replay the exact frozen token sequence")
        result.loss_mask = list(batch["mask"])
        write_identity(
            "rollout",
            result.tokens,
            result.loss_mask,
            {key: sampling_params[key] for key in ("temperature", "top_p", "top_k")},
        )
        return result

    vllm_rollout.generate = generate
    vllm_rollout._build_inference_sampling_params = build_parameters
    try:
        return vllm_rollout.generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    finally:
        vllm_rollout.generate = original
        vllm_rollout._build_inference_sampling_params = original_builder


def save_layer(
    side: str,
    layer: int,
    tensors: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist actual local tensors with explicit logical token coordinates."""
    batch = frozen_batch()
    rows = _ROWS.get(side)
    if batch is None or rows is None:
        return
    root = Path(os.environ["RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR"])
    key = (side, layer)
    call = _CALLS.get(key, 0)
    _CALLS[key] = call + 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    payload = {
        "schema_version": "rlkernel.layer_replay.v1",
        "side": side,
        "rank": rank,
        "layer": layer,
        "call": call,
        "positions": rows,
        "sequence_length": len(batch["tokens"]),
        "frozen_tokens_sha256": fingerprint(torch.tensor(batch["tokens"])),
        "sampling": batch["sampling"],
        "tensors": tensors,
        "metadata": metadata or {},
    }
    atomic_save(
        root / "layers" / f"{side}-pid{os.getpid()}-rank{rank}-layer{layer}-call{call}.pt",
        payload,
    )
