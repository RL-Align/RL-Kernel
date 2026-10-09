# SPDX-License-Identifier: Apache-2.0
"""The rotary tables must not depend on how the owning module was cast."""

from types import SimpleNamespace

import torch

from rl_engine.integrations.qwen3_next_tp_blocks import TP4FullAttention


def _reference_rotate(value, positions, rope_theta=10000000.0):
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(value.dtype)[:, None], angles.sin().to(value.dtype)[:, None]
    rotary, tail = value[..., :64], value[..., 64:]
    rotated = torch.cat((-rotary[..., 32:], rotary[..., :32]), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, tail), dim=-1)


def test_inverse_frequencies_are_fp32_and_exact_at_dim_zero():
    inv_freq = TP4FullAttention.inverse_frequencies(10000000.0, torch.device("cpu"))
    assert inv_freq.dtype == torch.float32 and inv_freq.shape == (32,)
    assert inv_freq[0].item() == 1.0
    assert torch.equal(
        inv_freq, 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    )


def test_rotate_matches_fp32_reference_for_any_module_dtype():
    generator = torch.Generator().manual_seed(7)
    value = (torch.randn((23, 1, 256), generator=generator) * 1.5).to(torch.bfloat16)
    positions = torch.arange(23)
    module = SimpleNamespace(rope_theta=10000000.0)
    actual = TP4FullAttention._rotate(module, value, positions)
    assert torch.equal(actual, _reference_rotate(value, positions))


def test_bf16_rounded_frequencies_would_change_rotated_keys():
    """A BF16-cast rotary table flips rotated dims 1..~47 at positions >= 1; keep it FP32."""
    generator = torch.Generator().manual_seed(8)
    value = (torch.randn((23, 1, 256), generator=generator) * 1.5).to(torch.bfloat16)
    positions = torch.arange(23)
    exact = _reference_rotate(value, positions)
    rounded_theta = SimpleNamespace(rope_theta=10000000.0)
    inv_freq = TP4FullAttention.inverse_frequencies(rounded_theta.rope_theta, value.device)
    rounded = inv_freq.to(torch.bfloat16).float()
    angles = positions.float()[:, None] * rounded[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(value.dtype)[:, None], angles.sin().to(value.dtype)[:, None]
    rotary, tail = value[..., :64], value[..., 64:]
    rotated = torch.cat((-rotary[..., 32:], rotary[..., :32]), dim=-1)
    with_rounded = torch.cat((rotary * cos + rotated * sin, tail), dim=-1)
    differing = (with_rounded != exact)[:, 0]
    assert differing.any()
    assert not differing[0].any()  # position 0 has angle 0
    assert not differing[:, 64:].any()  # the unrotated tail is untouched
    assert not differing[:, 0].any() and not differing[:, 32].any()  # dim 0 is exact in BF16
