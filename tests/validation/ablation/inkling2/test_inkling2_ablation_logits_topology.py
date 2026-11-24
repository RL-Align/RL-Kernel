# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.models.inkling2.lm_head_logprob import selected_logprob, softcap_valid_logits
from rl_engine.validation.ablation.inkling2.logits_topology import logits_probe, validate_ownership
from rl_engine.validation.ablation.inkling2.overlay import compare_boundaries


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_logits_faults(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    device = "cpu" if platform == "cpu" else "cuda"
    raw = torch.tensor([[90.0, 80.0, 100.0]], device=device)
    target = torch.tensor([0], device=device)
    base = selected_logprob(softcap_valid_logits(raw, valid_vocab_size=2), target)
    for fault in ("include_padding", "omit_softcap"):
        changed = logits_probe(raw, target, valid_vocab_size=2, fault=fault)
        assert compare_boundaries({"logp": base}, {"logp": changed})["first_drift"] == "logp"


def test_ownership_rejects_each_axis():
    expected = {"tp": (0, 1), "ep": (0, 1), "cp": (0, 1)}
    validate_ownership(expected, expected)
    for axis in expected:
        with pytest.raises(ValueError, match=axis):
            validate_ownership(expected, {**expected, axis: (1, 0)})
