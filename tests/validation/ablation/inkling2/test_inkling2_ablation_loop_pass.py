# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.models.inkling2.loop_schedule import run_inkling2_mini_schedule
from rl_engine.validation.ablation.inkling2.loop_pass import (
    run_pass_order_probe,
    validate_pass_order,
)
from rl_engine.validation.ablation.inkling2.overlay import compare_boundaries


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_reordered_pass_drift(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    x = torch.ones(2, device="cpu" if platform == "cpu" else "cuda")

    def layer(visit, state, message):
        return state + (visit.pass_index or 0) + 1 + (0 if message is None else message * 0.01)

    def capture(state, pass_id):
        return state + pass_id

    base = run_pass_order_probe(x, layer, capture)
    assert torch.equal(base[47], run_inkling2_mini_schedule(x, layer, capture))
    mutated = run_pass_order_probe(x, layer, capture, (0, 2, 1))
    assert compare_boundaries(base, mutated)["first_drift"] == 18
    with pytest.raises(ValueError, match="count"):
        run_pass_order_probe(x, layer, capture, (0, 1))
    with pytest.raises(ValueError, match="requires passes"):
        validate_pass_order((0, 2, 1))
