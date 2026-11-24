# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.validation.ablation.inkling2.overlay import compare_boundaries, overlay


def test_overlay_preserves_existing_rows():
    rows = [{"id": f"A{i}"} for i in range(8)] + [{"id": f"C{i}"} for i in range(3)]
    base = {"rows": rows, "cartesian_product": False}
    result = overlay(base)
    assert result["rows"] == rows and "inkling2_subprobes" not in base
    assert len(result["inkling2_subprobes"]) == 15
    assert result["inkling2_platforms"] == ["cuda", "rocm"]
    with pytest.raises(ValueError, match="matrix"):
        overlay({"rows": [], "cartesian_product": False})


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_byte_exact_first_drift(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    device = "cpu" if platform == "cpu" else "cuda"
    a = torch.tensor([0.0, 1.0], device=device)
    b = torch.tensor([-0.0, 1.0], device=device)
    result = compare_boundaries({"prelude": a, "pass1": a}, {"prelude": a, "pass1": b})
    assert result["first_drift"] == "pass1" and result["mismatches"]["pass1"] == 1
    assert not result["ws1_certified"]
