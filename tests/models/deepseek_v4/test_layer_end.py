# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The DeepSeek-V4 layer end against DeepSeek-V4's layer end written in PyTorch.

mHC head -> final RMSNorm -> lm_head -> raw logprob (DSV4 RFC task 8, #508). The
reference runs mHC head and RMSNorm in fp32 with bf16 output, lm_head in fp32,
and log_softmax. Batch invariance of the layer end is checked in
test_layer_end_batch_invariance.py.
"""

import pytest
import torch
import torch.nn.functional as F

from rl_engine.reference.mhc import NativeMHCHeadOp
from rl_engine.reference.norm.rms_norm import NativeRMSNormOp
from tests.helpers.contract_tolerance import assert_matches_fp32_reference

_N, _C, _V = 4, 4096, 4096  # DeepSeek-V4-Flash-0731 hc_mult and hidden_size; test vocabulary


def _aiter_available():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return False
    try:
        from rl_engine.backends.rocm.mhc.aiter_mhc import AiterMHCHeadOp

        AiterMHCHeadOp()
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _aiter_available(), reason="ROCm GPU with aiter DeepSeek-V4 mHC ops required."
)


def _weights():
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    k = _N * _C
    return {
        "head_fn": randn(_N, k) / k**0.5,
        "head_scale": 0.5 + torch.rand(1, generator=gen, device="cuda"),
        "head_base": 0.1 * randn(_N),
        "norm_weight": 1.0 + 0.1 * randn(_C),
        "lm_head_weight": (0.02 * randn(_V, _C)).to(torch.bfloat16),  # BF16 as in the checkpoint
    }


def _rows(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(m, _N, _C, generator=gen, device="cuda").to(torch.bfloat16)
    return residual, torch.randint(0, _V, (m,), generator=gen, device="cuda")


def _layer_end(residual, targets, w):
    from rl_engine.models.deepseek_v4.layer_end import mhc_layer_end

    with torch.no_grad():
        return mhc_layer_end(
            residual,
            w["head_fn"],
            w["head_scale"],
            w["head_base"],
            w["norm_weight"],
            w["lm_head_weight"],
            targets,
        )


def test_layer_end_matches_reference():
    from rl_engine.models.deepseek_v4.layer_end import LAYER_END_STAGES

    w = _weights()
    residual, targets = _rows(64, 1)
    stages = _layer_end(residual, targets, w)

    head = NativeMHCHeadOp()(residual, w["head_fn"], w["head_scale"], w["head_base"])
    norm = NativeRMSNormOp()(head, w["norm_weight"])
    logits = F.linear(norm.float(), w["lm_head_weight"].float())
    logprob = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
    assert tuple(stages) == LAYER_END_STAGES
    assert [stages[s].dtype for s in LAYER_END_STAGES] == [
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
    ]
    for name, expected in (("head", head), ("logits", logits)):
        assert_matches_fp32_reference(
            stages[name], expected, judgment="forward_accuracy", op_class="reduction", name=name
        )
    assert_matches_fp32_reference(
        stages["logprob"], logprob, judgment="forward_accuracy", op_class="logprob", name="logprob"
    )
