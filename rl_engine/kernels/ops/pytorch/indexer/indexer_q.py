# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 indexer query: RoPE + per-(token, head) FP8 quantization (golden).

Reproduces vLLM's ``IndexerQFp8Kernel`` (CuTe DSL,
``vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py`` at
``ISEEKYAN/vllm@1a6542c``), which the CUDA rollout runs whenever
``nvidia-cutlass-dsl`` is installed (a hard dependency of the vLLM CUDA build).
Its Triton fallback ``_fused_indexer_q_rope_quant_kernel`` rounds differently
and is not reproduced. For each (token, head) row of ``head_dim = 128``:

* The first ``head_dim - rope_dim`` values pass through. The last ``rope_dim``
  are rotated GPT-J style (interleaved pairs ``(q0, q1)``) with
  ``cos, sin = cos_sin_cache[position]`` (cos first, then sin; FP32 or BF16),
  using the contraction the compiled kernel uses (established by running it)::

      rot0 = fma(q0, cos, -(q1 * sin))
      rot1 = fma(q0, sin,  (q1 * cos))

  and rounded to BF16 (round to nearest even).
* ``amax`` = max |value| over the 128 BF16 values, NaN ignored.
  ``s = max(amax, 1e-4) * fp32(1 / 448)``, rounded UP to a power of two from
  its bits: ``e = ((bits(s) + 0x7FFFFF) >> 23) & 0xFF``, ``scale = 2^(e - 127)``.
* ``q_fp8 = e4m3_satfinite(value * 2^-(e - 127))`` (the multiply is exact).
* ``weights_out = (fp32(weights) * fp32(softmax_scale * head_scale)) * scale``,
  i.e. the FP8 scale is folded into the per-head weight that DeepGEMM's
  ``fp8_mqa_logits`` multiplies in.

Every value depends only on its own (token, head) row, so the operator is
batch-invariant.
"""

from __future__ import annotations

import torch
from torch import Tensor

FP8_MAX = 448.0
AMAX_FLOOR = 1e-4


def _fma_f32(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Correctly rounded FP32 ``fma(a, b, c)`` on FP32 tensors.

    ``a * b`` is exact in FP64; ``s = a*b + c`` is rounded once there and its
    rounding error ``e`` recovered exactly (TwoSum). Rounding ``s`` to FP32
    is then exact except when ``s`` lands on an FP32 rounding midpoint, where
    ``e`` decides the direction.
    """
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bb = s - p
    e = (p - (s - bb)) + (cd - bb)
    t = s.float()
    toward = torch.where(s >= t.double(), float("inf"), -float("inf")).to(t.dtype)
    n = torch.nextafter(t, toward)
    midpoint = (t.double() + n.double()) * 0.5 == s
    directed = torch.where(e > 0, torch.maximum(t, n), torch.minimum(t, n))
    return torch.where(midpoint & (e != 0), directed, t)


def _mul_f32(a: Tensor, b: Tensor) -> Tensor:
    return (a.double() * b.double()).float()


def _check(q, positions, cos_sin_cache, weights) -> None:
    if q.ndim != 3 or q.dtype != torch.bfloat16:
        raise TypeError(f"q must be BF16 [tokens, heads, head_dim], got {tuple(q.shape)} {q.dtype}")
    tokens, heads, head_dim = q.shape
    if positions.shape != (tokens,) or positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be int [{tokens}], got {tuple(positions.shape)}")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("cos_sin_cache must be FP32 or BF16 [max_position, rope_dim]")
    rope_dim = cos_sin_cache.shape[1]
    if rope_dim % 16 or not 0 < rope_dim <= head_dim or head_dim % 16:
        raise ValueError(f"rope_dim {rope_dim} / head_dim {head_dim} must be multiples of 16")
    if weights.shape != (tokens, heads) or weights.dtype != torch.bfloat16:
        raise TypeError(f"weights must be BF16 [{tokens}, {heads}], got {tuple(weights.shape)}")


def indexer_q_rope_fp8(
    q: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    weights: Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[Tensor, Tensor]:
    """``q`` BF16 ``[T, H, D]`` -> (FP8 E4M3 ``[T, H, D]``, FP32 weights ``[T, H]``)."""
    _check(q, positions, cos_sin_cache, weights)
    tokens, heads, head_dim = q.shape
    half = cos_sin_cache.shape[1] // 2
    nope = head_dim - 2 * half
    cs = cos_sin_cache[positions.long()].float()
    cos, sin = cs[:, None, :half], cs[:, None, half:]
    qf = q.float()
    q0, q1 = qf[..., nope::2], qf[..., nope + 1 :: 2]
    rot0 = _fma_f32(q0, cos, -_mul_f32(q1, sin))
    rot1 = _fma_f32(q0, sin, _mul_f32(q1, cos))
    rot = torch.stack([rot0, rot1], dim=-1).reshape(tokens, heads, 2 * half).to(torch.bfloat16)
    full = torch.cat([q[..., :nope], rot], dim=-1).float()

    magnitude = full.abs()
    amax = torch.where(magnitude.isnan(), 0.0, magnitude).amax(dim=-1)  # inf counts
    floor = torch.tensor(AMAX_FLOOR, dtype=torch.float32, device=q.device)
    inv_max = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=q.device)
    s = torch.maximum(amax, floor) * inv_max
    bits = s.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent = ((bits + 0x7FFFFF) >> 23) & 0xFF
    scale = (exponent << 23).to(torch.int32).view(torch.float32)
    inverse = ((254 - exponent) << 23).to(torch.int32).view(torch.float32)
    # torch's FP8 cast turns overflow into NaN; the kernel saturates.
    scaled = (full * inverse[..., None]).clamp(-FP8_MAX, FP8_MAX)
    q_fp8 = scaled.to(torch.float8_e4m3fn)

    combined = torch.tensor(float(softmax_scale * head_scale), dtype=torch.float32, device=q.device)
    weights_out = (weights.float() * combined) * scale
    return q_fp8, weights_out


class IndexerQRopeFp8Op:
    """PyTorch golden for ``indexer_q_rope_fp8``."""

    def __call__(self, q, positions, cos_sin_cache, weights, softmax_scale, head_scale):
        return indexer_q_rope_fp8(q, positions, cos_sin_cache, weights, softmax_scale, head_scale)
