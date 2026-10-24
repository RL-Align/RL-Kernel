# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Prepared L3 GPU acceptance through the default-closed delivery mechanism."""

import pytest
import torch

from tests.helpers.adaln_indexed import (
    check_absent_vs_zero_dy,
    check_domain_routes,
    check_leaf_participation,
    check_small_case,
    check_view_edges,
    resolve_qualified_indexed,
    small_case,
)

pytestmark = pytest.mark.cuda_only


@pytest.fixture(params=["cuda", "triton"])
def activated(request):
    return resolve_qualified_indexed(request.param)


def test_activated_indexed_small_forward_backward_and_trace(activated):
    op, trace = activated
    check_small_case(op, "cuda")
    assert trace["actual_backend"] in {"cuda", "triton"}
    assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
    assert trace["fallback"] is False and trace["fallback_reason"] is None
    assert trace["observation"] == "successful_forward"
    assert trace["hardware_qualified"] is None and trace["binary_sha256"] is None
    assert trace["profiler_observed"] is False


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("strict", [False, True])
def test_indexed_dispatch_adds_no_route_scan_or_scalar_read(backend, strict):
    from rl_engine.runtime.registry import KernelRegistry
    from tests.helpers.adaln_indexed import assert_bytes

    op, trace = resolve_qualified_indexed(backend)
    x, m, index, _, _ = small_case("cuda")
    if not strict:
        op, trace = KernelRegistry().get_adaln_modulation_op(
            x.device, hidden=2, modulate_index=index
        )
    # Warm compilation separately; count only the public direct/dispatch calls.
    check_small_case(op, "cuda")
    op.implementation(x, m, modulate_index=index, eps=7.0)
    observed, outputs = [], []
    for callable_op in (op.implementation, op):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True
        ) as profile:
            outputs.append(callable_op(x, m, modulate_index=index, eps=7.0))
        observed.append({event.key: event.count for event in profile.key_averages()})
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        assert_bytes(actual, expected)
    for event in ("aten::all", "aten::item"):
        # CUDA retains its native ABI check in addition to the Python check.
        assert observed[0][event] > 0
        assert observed[1][event] == observed[0][event]
    assert trace["actual_backend"] == backend and trace["fallback"] is False
    assert trace["profiler_observed"] is False


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_activated_indexed_absent_vs_zero(activated, explicit_zero):
    check_absent_vs_zero_dy(activated[0], "cuda", explicit_zero)


@pytest.mark.parametrize(
    "frozen,outputs", [("x", "both"), ("modulation", "both"), (None, "gate"), (None, "y")]
)
def test_activated_indexed_leaf_participation(activated, frozen, outputs):
    check_leaf_participation(activated[0], "cuda", frozen, outputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["strided", "expanded"])
def test_activated_indexed_source_leaf_gradients(activated, dtype, layout):
    check_view_edges(activated[0], "cuda", dtype, layout)


@pytest.mark.parametrize("broadcast", [False, True])
def test_activated_indexed_routes(activated, broadcast):
    check_domain_routes(activated[0], "cuda", torch.bfloat16, torch.int16, broadcast)


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("resolution", ["indexed", "shared"])
def test_activated_strict_derives_mode_on_every_call(backend, resolution):
    op, trace = resolve_qualified_indexed(backend, indexed=resolution == "indexed")
    x, m, index, _, _ = small_case("cuda")
    for indexed in (True, False, True):
        op(x, m if indexed else m[:2], eps=7.0, modulate_index=index if indexed else None)
        assert trace["arithmetic_profile"] == (
            "adaln-indexed-fp32-tree-v1" if indexed else "adaln-shared-fp32-tree-v1"
        )
        assert trace["actual_backend"] == backend
    with pytest.raises(ValueError, match="values must be 0 or 1"):
        op(x, m, modulate_index=index + 2)
    assert trace["actual_backend"] is None and trace["observation"] is None


@pytest.mark.parametrize(
    "unsupported", ["H", "changed_hidden", "B", "device", "eps_overflow", "eps_underflow"]
)
def test_activated_strict_rejects_changed_execution_domain(activated, unsupported):
    op, trace = activated
    check_small_case(op, "cuda")
    h = {"H": 4097, "changed_hidden": 3}.get(unsupported, 2)
    b = 65536 if unsupported == "B" else 2
    device = "cpu" if unsupported == "device" else "cuda"
    x = torch.zeros(b, 1, h, device=device)
    m = torch.zeros(2 * b, 3 * h, device=device)
    index = torch.zeros(1, 1, dtype=torch.int32, device=device)
    eps = {"eps_overflow": 1e300, "eps_underflow": 2**-150}.get(unsupported, 7.0)
    with pytest.raises((ValueError, RuntimeError), match="strict AdaLN"):
        op(x, m, modulate_index=index, eps=eps)
    assert trace["actual_backend"] is None and trace["observation"] is None


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq", [4096, 6889, 6032])
@pytest.mark.parametrize("route", ["segmented", "dispersed"])
def test_activated_indexed_target_h3072(backend, dtype, seq, route):
    import os

    from tests.helpers.adaln_indexed import check_geometry_bytes

    if os.environ.get("RLK_ADALN_REAL_SHAPES") != "1":
        pytest.skip("user-run workload: >=8 GiB free GPU / >=16 GiB available host")
    op, trace = resolve_qualified_indexed(backend, hidden=3072)
    check_geometry_bytes(op, "cuda", dtype, 3072, seq=seq, route=route)
    assert trace["actual_backend"] == backend and not trace["fallback"]
