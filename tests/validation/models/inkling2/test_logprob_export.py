# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.validation.models.inkling2.logprob_export import compare_exports, export_selected_logprob


def export(engine, platform, device="cpu", fixture=True, reverse=False):
    keys = torch.tensor([7, 3], device=device)
    ids = torch.tensor([12, 23], device=device)
    lp = torch.tensor([-1.0, -2.0], device=device)
    routes = torch.arange(8, device=device).view(1, 1, 8).expand(2, 48, 8).clone()
    order = torch.tensor([1, 0] if reverse else [0, 1], device=device)
    trace = {"metadata": {"weights_sha256": "a" * 64, "input_sha256": "b" * 64},
             "records": [{"direction": "forward", "role": "partial", "logical": j,
                          "sha256": "c" * 64} for j in range(48)]}
    return export_selected_logprob(
        lp[order], keys[order], ids[order], routes[order], weight_version="v1",
        weights_sha256="a" * 64, input_sha256="b" * 64, platform=platform,
        engine=engine, runtime_revision="pinned", kernel_id="test", fixture=fixture,
        visit_trace=trace,
    )


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_reorder_and_first_drift(platform):
    a, b = export("miles", platform), export("sglang", platform, reverse=True)
    assert compare_exports(a, b)["equal"]
    b["logprob_bits"][0] += 1
    b["route_hashes"][18] = "d" * 64
    b["visit_hashes"][6] = "e" * 64
    r = compare_exports(a, b)
    assert r["bit_mismatch_count"] == 1 and r["first_logical_token"] == 3
    assert r["first_route_visit"] == 18 and r["first_boundary_visit"] == 6
    assert not r["equal"] and not r["ws1_certified"]
    b["weight_version"] = "v2"
    with pytest.raises(ValueError, match="incompatible"):
        compare_exports(a, b)


def test_cpu_cannot_claim_real_export():
    with pytest.raises(RuntimeError, match="accelerator"):
        export("miles", "cuda", fixture=False)


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_real_device_export(platform):
    if not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm"):
        pytest.skip(f"{platform} GPU required")
    a = export("miles", platform, device="cuda", fixture=False)
    b = export("sglang", platform, device="cuda", fixture=False, reverse=True)
    assert compare_exports(a, b)["equal"]
