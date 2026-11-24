# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.models.inkling2.nvfp4_audit import dequant_nvfp4, native_capability


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_nibble_and_scale_order(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    device = "cpu" if platform == "cpu" else "cuda"
    packed = torch.tensor([[[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F]]],
                          device=device, dtype=torch.uint8)
    y = dequant_nvfp4(packed, torch.full((1, 1, 1), 2., device=device),
                      torch.tensor([3.], device=device), torch.tensor([2.], device=device))
    expected = torch.tensor([.5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5,
                             -2, -3, -4, -6, 0], device=device) * 3
    assert torch.equal(y.flatten(), expected.bfloat16())
    if platform != "cpu":
        assert not native_capability(y.device)["native_nvfp4_qualified"]


def test_invalid_outer_scale():
    with pytest.raises(ValueError, match="scales"):
        dequant_nvfp4(torch.zeros(1, 1, 8, dtype=torch.uint8), torch.ones(1, 1, 1),
                      torch.ones(1), torch.zeros(1))
