# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
from typing import Any

import torch

from rl_engine.kernels.ops.pytorch.mhc import MHC_PRE_HIDDEN_SIZE

DEFAULT_HIDDEN = 4096
DEFAULT_N_HEADS = 32
DEFAULT_N_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_INTERMEDIATE = 12288
DEFAULT_VOCAB = 151936
DEFAULT_ROPE_THETA = 1.0e6
DEFAULT_RMS_EPS = 1.0e-6


def make_operator_inputs(
    op_name: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    builders = {
        "mhc_pre_h_aggregate": _make_mhc_pre_h_aggregate_inputs,
        "rms_norm": _make_rms_norm_inputs,
        "qk_norm": _make_qk_norm_inputs,
        "pack": _make_pack_inputs,
        "matmul": _make_matmul_inputs,
        "det_gemm": _make_det_gemm_inputs,
        "attention": _make_attention_inputs,
        "cp_attention": _make_cp_attention_inputs,
        "logp": _make_logp_inputs,
        "linear_logp": _make_linear_logp_inputs,
        "batch_invariant_logp": _make_batch_invariant_logp_inputs,
        "rope": _make_rope_inputs,
        "silu": _make_silu_inputs,
        "swiglu": _make_swiglu_inputs,
        "embedding": _make_embedding_inputs,
        "lm_head": _make_lm_head_inputs,
        "kv_cache_attention": _make_kv_cache_attention_inputs,
    }
    try:
        return builders[op_name](args, dtype, device)
    except KeyError as exc:
        raise ValueError(f"unsupported operator inputs: {op_name}") from exc


def operator_shape_name(op_name: str, args: argparse.Namespace) -> str:
    batch, seq = _batch_seq(args)
    vocab = _arg_int(args, "vocab", DEFAULT_VOCAB)
    names = {
        "mhc_pre_h_aggregate": f"{batch * seq}x4x{MHC_PRE_HIDDEN_SIZE}",
        "rms_norm": f"{batch}x{seq}x{_normalized_dim(args)}",
        "qk_norm": f"{batch}x{seq}x{_arg_int(args, 'n_heads', DEFAULT_N_HEADS)}x"
        f"{_arg_int(args, 'head_dim', DEFAULT_HEAD_DIM)}",
        "pack": f"{batch}x{seq}x{_normalized_dim(args)}",
        "matmul": f"{batch}x{seq}x{_matmul_k(args)}x{_matmul_n(args)}",
        "det_gemm": f"{batch}x{seq}x{_matmul_k(args)}x{_matmul_n(args)}",
        "attention": f"{batch}x{DEFAULT_N_HEADS}x{seq}x{DEFAULT_HEAD_DIM}",
        "cp_attention": f"{batch}x{DEFAULT_N_HEADS}x{seq}x{DEFAULT_HEAD_DIM}xcp2",
        "logp": f"{batch}x{seq}x{vocab}",
        "linear_logp": f"{batch}x{seq}x{_normalized_dim(args)}x{vocab}",
        "batch_invariant_logp": f"{batch}x{seq}x{vocab}",
        "rope": f"{batch}x{DEFAULT_N_HEADS}x{seq}x{DEFAULT_HEAD_DIM}",
        "silu": f"{batch}x{seq}x{DEFAULT_INTERMEDIATE}",
        "swiglu": f"{batch}x{seq}x{DEFAULT_INTERMEDIATE}",
        "embedding": f"{batch}x{seq}x{vocab}x{_normalized_dim(args)}",
        "lm_head": f"{batch}x{seq}x{_normalized_dim(args)}x{vocab}",
        "kv_cache_attention": f"{batch}x{DEFAULT_N_HEADS}x1x{seq + 1}x{DEFAULT_HEAD_DIM}",
    }
    try:
        return names[op_name]
    except KeyError as exc:
        raise ValueError(f"unsupported operator shape: {op_name}") from exc


def _make_rms_norm_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    normalized_dim = _normalized_dim(args)
    return {
        "x": _floating_tensor((batch, seq, normalized_dim), args, dtype, device, offset=0),
        "weight": _floating_tensor((normalized_dim,), args, dtype, device, offset=1),
        "eps": _arg_float(args, "eps", DEFAULT_RMS_EPS),
    }


def _make_mhc_pre_h_aggregate_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    num_tokens = batch * seq
    return {
        "residual": _floating_tensor(
            (num_tokens, 4, MHC_PRE_HIDDEN_SIZE), args, dtype, device, offset=0
        ),
        "pre": _floating_tensor((num_tokens, 4), args, torch.float32, device, offset=1),
    }


def _make_qk_norm_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    """Per-head RMSNorm: last dim is head_dim, not the full hidden width."""
    batch, seq = _batch_seq(args)
    n_heads = _arg_int(args, "n_heads", DEFAULT_N_HEADS)
    head_dim = _arg_int(args, "head_dim", DEFAULT_HEAD_DIM)
    return {
        "x": _floating_tensor((batch, seq * n_heads, head_dim), args, dtype, device, offset=0),
        "weight": _floating_tensor((head_dim,), args, dtype, device, offset=1),
        "eps": _arg_float(args, "eps", DEFAULT_RMS_EPS),
    }


def _make_pack_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    hidden = _normalized_dim(args)
    x = _floating_tensor((batch, seq, hidden), args, dtype, device, offset=0)
    mode = _arg_str(args, "input_mode", "random")
    if mode == "constant":
        mask = torch.zeros(batch, seq, device=device, dtype=torch.bool)
        mask[:, : max(1, seq // 2)] = True
    else:
        generator = _generator(args, device, offset=17)
        mask = torch.randint(0, 2, (batch, seq), generator=generator, device=device) > 0
        mask[:, 0] = True
    return {"x": x, "mask": mask}


def _make_matmul_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    k_dim = _matmul_k(args)
    n_dim = _matmul_n(args)
    return {
        "a": _floating_tensor((batch, seq, k_dim), args, dtype, device, offset=0),
        "b": _floating_tensor((k_dim, n_dim), args, dtype, device, offset=1),
    }


def _make_det_gemm_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    k_dim = _matmul_k(args)
    n_dim = _matmul_n(args)
    m_dim = batch * seq
    return {
        "a": _floating_tensor((m_dim, k_dim), args, dtype, device, offset=0),
        "b": _floating_tensor((k_dim, n_dim), args, dtype, device, offset=1),
    }


def _make_attention_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    skv = _arg_int(args, "skv", seq)
    n_heads = _arg_int(args, "n_heads", DEFAULT_N_HEADS)
    n_kv_heads = _arg_int(args, "n_kv_heads", DEFAULT_N_KV_HEADS)
    causal = bool(_arg_int(args, "causal", 1))
    use_padding = bool(_arg_int(args, "use_padding", 0))
    scale_mode = _arg_str(args, "scale_mode", "default")

    inputs: dict[str, Any] = {
        "q": _floating_tensor((batch, n_heads, seq, DEFAULT_HEAD_DIM), args, dtype, device, 0),
        "k": _floating_tensor((batch, n_kv_heads, skv, DEFAULT_HEAD_DIM), args, dtype, device, 1),
        "v": _floating_tensor((batch, n_kv_heads, skv, DEFAULT_HEAD_DIM), args, dtype, device, 2),
        "causal": causal,
    }

    if scale_mode == "zero":
        inputs["scale"] = 0.0
    elif scale_mode == "custom":
        inputs["scale"] = 0.05
    # else: scale_mode == "default" -> no scale kwarg (uses 1/sqrt(D))

    if use_padding:
        generator = _generator(args, device, offset=42)
        key_padding_mask = torch.rand((batch, skv), generator=generator, device=device) > 0.3
        key_padding_mask[:, 0] = True
        inputs["key_padding_mask"] = key_padding_mask

    return inputs


def _make_cp_attention_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    return {
        "q": _floating_tensor(
            (batch, DEFAULT_N_HEADS, seq, DEFAULT_HEAD_DIM), args, dtype, device, 0
        ),
        "k": _floating_tensor(
            (batch, DEFAULT_N_KV_HEADS, seq, DEFAULT_HEAD_DIM), args, dtype, device, 1
        ),
        "v": _floating_tensor(
            (batch, DEFAULT_N_KV_HEADS, seq, DEFAULT_HEAD_DIM), args, dtype, device, 2
        ),
        "causal": True,
        "cp_world_size": 2,
        "kv_chunk_size": max(1, seq // 2),
    }


def _make_logp_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    vocab = _arg_int(args, "vocab", DEFAULT_VOCAB)
    return {
        "logits": _floating_tensor((batch, seq, vocab), args, dtype, device, offset=0),
        "token_ids": _token_ids((batch, seq), vocab, args, device),
    }


def _make_linear_logp_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    hidden_dim = _normalized_dim(args)
    vocab = _arg_int(args, "vocab", DEFAULT_VOCAB)
    return {
        "hidden": _floating_tensor((batch, seq, hidden_dim), args, dtype, device, offset=0),
        "lm_head_weight": _floating_tensor((vocab, hidden_dim), args, dtype, device, offset=1),
        "target_ids": _token_ids((batch, seq), vocab, args, device),
        "bias": None,
    }


def _make_batch_invariant_logp_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    vocab = _arg_int(args, "vocab", DEFAULT_VOCAB)
    return {
        "logits": _floating_tensor((batch, seq, vocab), args, dtype, device, offset=0),
        "target_ids": _token_ids((batch, seq), vocab, args, device),
    }


def _make_rope_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    return {
        "x": _floating_tensor(
            (batch, DEFAULT_N_HEADS, seq, DEFAULT_HEAD_DIM), args, dtype, device, 0
        ),
        "positions": torch.arange(seq, device=device, dtype=torch.long),
        "theta": _arg_float(args, "theta", DEFAULT_ROPE_THETA),
    }


def _make_silu_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    return {
        "x": _floating_tensor((batch, seq, DEFAULT_INTERMEDIATE), args, dtype, device, 0),
    }


def _make_swiglu_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    return {
        "gate": _floating_tensor((batch, seq, DEFAULT_INTERMEDIATE), args, dtype, device, 0),
        "up": _floating_tensor((batch, seq, DEFAULT_INTERMEDIATE), args, dtype, device, 1),
    }


def _make_embedding_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    vocab = _arg_int(args, "vocab", DEFAULT_VOCAB)
    hidden_dim = _normalized_dim(args)
    return {
        "token_ids": _token_ids((batch, seq), vocab, args, device),
        "weight": _floating_tensor((vocab, hidden_dim), args, dtype, device, 0),
    }


def _make_lm_head_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    vocab = _arg_int(args, "vocab", DEFAULT_VOCAB)
    hidden_dim = _normalized_dim(args)
    return {
        "hidden": _floating_tensor((batch, seq, hidden_dim), args, dtype, device, 0),
        "weight": _floating_tensor((vocab, hidden_dim), args, dtype, device, 1),
        "bias": None,
    }


def _make_kv_cache_attention_inputs(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    batch, seq = _batch_seq(args)
    return {
        "q": _floating_tensor(
            (batch, DEFAULT_N_HEADS, 1, DEFAULT_HEAD_DIM), args, dtype, device, 0
        ),
        "k_cache": _floating_tensor(
            (batch, DEFAULT_N_KV_HEADS, seq, DEFAULT_HEAD_DIM), args, dtype, device, 1
        ),
        "v_cache": _floating_tensor(
            (batch, DEFAULT_N_KV_HEADS, seq, DEFAULT_HEAD_DIM), args, dtype, device, 2
        ),
        "k_new": _floating_tensor(
            (batch, DEFAULT_N_KV_HEADS, 1, DEFAULT_HEAD_DIM), args, dtype, device, 3
        ),
        "v_new": _floating_tensor(
            (batch, DEFAULT_N_KV_HEADS, 1, DEFAULT_HEAD_DIM), args, dtype, device, 4
        ),
        "causal": True,
    }


def _floating_tensor(
    shape: tuple[int, ...],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    offset: int,
) -> torch.Tensor:
    # Example: torch.randn((B, S, V), device="cuda", dtype=torch.bfloat16)
    mode = _arg_str(args, "input_mode", "random")
    if mode == "constant":
        value = _arg_float(args, "constant_value", 0.25) + float(offset) * 0.01
        return torch.full(shape, value, device=device, dtype=dtype)
    if mode != "random":
        raise ValueError(f"unsupported input_mode: {mode}")
    generator = _generator(args, device, offset)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def _token_ids(
    shape: tuple[int, ...],
    vocab: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    mode = _arg_str(args, "input_mode", "random")
    if mode == "constant":
        value = _arg_int(args, "token_value", 0) % vocab
        return torch.full(shape, value, device=device, dtype=torch.long)
    generator = _generator(args, device, offset=13)
    return torch.randint(0, vocab, shape, generator=generator, device=device, dtype=torch.long)


def _generator(args: argparse.Namespace, device: torch.device, offset: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(_arg_int(args, "seed", 123) + offset)
    return generator


def _batch_seq(args: argparse.Namespace) -> tuple[int, int]:
    return _arg_int(args, "batch", 2), _arg_int(args, "seq", 16)


def _normalized_dim(args: argparse.Namespace) -> int:
    return _arg_int(args, "normalized_dim", DEFAULT_HIDDEN)


def _matmul_k(args: argparse.Namespace) -> int:
    return _arg_int(args, "k_dim", DEFAULT_HIDDEN)


def _matmul_n(args: argparse.Namespace) -> int:
    return _arg_int(args, "n_dim", DEFAULT_HIDDEN)


def _arg_float(args: argparse.Namespace, name: str, default: float) -> float:
    return float(getattr(args, name, default))


def _arg_int(args: argparse.Namespace, name: str, default: int) -> int:
    return int(getattr(args, name, default))


def _arg_str(args: argparse.Namespace, name: str, default: str) -> str:
    return str(getattr(args, name, default))
