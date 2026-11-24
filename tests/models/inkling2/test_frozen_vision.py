# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from rl_engine.models.inkling2.frozen_vision import rotary_2d, vision_attention


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_rotary_and_image_isolation(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    device = "cpu" if platform == "cpu" else "cuda"
    attn = nn.Module()
    attn.num_heads, attn.head_dim = 2, 4
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        layer = nn.Linear(8, 8, bias=False, device=device)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(8, device=device))
        setattr(attn, name, layer)
    x = torch.arange(32, device=device).float().reshape(4, 8) / 32
    pos = torch.tensor([[0, 0], [1, 2], [0, 0], [2, 1]], device=device)
    assert torch.equal(rotary_2d(x.reshape(4, 2, 4), pos)[0], x[0].reshape(2, 4))
    a = vision_attention(attn, x, pos, [2, 2])
    b = vision_attention(attn, x[2:], pos[2:], [2])
    torch.testing.assert_close(a[2:], b)
