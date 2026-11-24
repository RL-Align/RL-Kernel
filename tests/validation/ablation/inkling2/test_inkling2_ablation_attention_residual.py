# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.models.inkling2.attn_res_mix import block_attention_residual
from rl_engine.validation.ablation.inkling2.attention_residual import attention_residual_probe
from rl_engine.validation.ablation.inkling2.overlay import compare_boundaries


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_residual_faults_detected(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    device = "cpu" if platform == "cpu" else "cuda"
    a = torch.tensor([[8.0, 0.0, 0.0, 0.0]], device=device)
    b = torch.tensor([[0.0, 2.0, 0.0, 0.0]], device=device)
    p = torch.tensor([[0.0, 0.0, 4.0, 0.0]], device=device)
    q = torch.ones(4, device=device)
    base = block_attention_residual([a, b], p, q)
    changed = attention_residual_probe([a, b], p, q, fault="skip_value_norm")
    result = compare_boundaries({"attn_res": base}, {"attn_res": changed})
    assert result["first_drift"] == "attn_res"
    b = -a
    p = torch.tensor([[1e-8, 1.0, 0.0, 0.0]], device=device)
    q = torch.zeros_like(q)
    base = block_attention_residual([a, b], p, q)
    changed = attention_residual_probe([a, b], p, q, fault="reverse_sources")
    result = compare_boundaries({"attn_res": base}, {"attn_res": changed})
    assert result["first_drift"] == "attn_res"
