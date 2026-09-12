# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA tests for P5-2 clamp_swiglu_weighted."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.moe import fixtures, oracle
from rl_engine.moe.cuda_provider import (
    ClampSwiGLUWeightedCudaProvider,
)

_HAS_P5_CUDA = bool(
    torch.cuda.is_available()
    and _EXT_AVAILABLE
    and _C is not None
    and hasattr(_C, "clamp_swiglu_weighted_forward")
    and hasattr(_C, "clamp_swiglu_weighted_backward")
)

requires_p5_cuda = pytest.mark.skipif(
    not _HAS_P5_CUDA,
    reason="P5-2 CUDA extension unavailable",
)


def _assert_exact(
    got: torch.Tensor,
    want: torch.Tensor,
) -> None:
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert torch.equal(got, want)


def _assert_saved_state(
    saved: dict[str, torch.Tensor],
    gate: torch.Tensor,
    up: torch.Tensor,
    p_s: torch.Tensor | None,
) -> None:
    expected_keys = {
        "gate32",
        "up32",
    }

    if p_s is not None:
        expected_keys.add("p_s")

    assert set(saved) == expected_keys

    _assert_exact(
        saved["gate32"],
        gate.float().contiguous(),
    )

    _assert_exact(
        saved["up32"],
        up.float().contiguous(),
    )

    if p_s is not None:
        _assert_exact(
            saved["p_s"],
            p_s.contiguous(),
        )


@requires_p5_cuda
def test_boundary_fixture_is_byte_exact() -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    gate, up, p_s = (tensor.cuda() for tensor in fixtures.make_swiglu_boundary_inputs())

    dh = fixtures.make_grad_output(
        "swiglu_boundary",
        tuple(gate.shape),
    ).cuda()

    h_ref, saved_ref = oracle.clamp_swiglu_weighted_fwd(
        gate,
        up,
        p_s,
    )

    grads_ref = oracle.clamp_swiglu_weighted_bwd(
        dh,
        saved_ref,
    )

    h_got, saved_got = provider.clamp_swiglu_weighted_fwd(
        gate,
        up,
        p_s,
    )

    grads_got = provider.clamp_swiglu_weighted_bwd(
        dh,
        saved_got,
    )

    _assert_exact(h_got, h_ref)

    _assert_saved_state(
        saved_got,
        gate,
        up,
        p_s,
    )

    for got, want in zip(
        grads_got,
        grads_ref,
        strict=True,
    ):
        assert got is not None
        assert want is not None
        _assert_exact(got, want)


@requires_p5_cuda
def test_shared_variant_has_no_weight_no_clamp_and_no_dp_s() -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    gate = torch.tensor(
        [[12.0, -12.0, 10.0, -10.0]],
        device="cuda",
    )

    up = torch.tensor(
        [[12.0, -12.0, 10.0, -10.0]],
        device="cuda",
    )

    dh = torch.tensor(
        [[1.0, -0.5, 0.25, -2.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )

    h_ref, saved_ref = oracle.clamp_swiglu_weighted_fwd(
        gate,
        up,
        None,
    )

    dgate_ref, dup_ref, dp_ref = oracle.clamp_swiglu_weighted_bwd(
        dh,
        saved_ref,
    )

    h_got, saved_got = provider.clamp_swiglu_weighted_fwd(
        gate,
        up,
        None,
    )

    dgate_got, dup_got, dp_got = provider.clamp_swiglu_weighted_bwd(
        dh,
        saved_got,
    )

    _assert_exact(h_got, h_ref)

    _assert_saved_state(
        saved_got,
        gate,
        up,
        None,
    )

    _assert_exact(dgate_got, dgate_ref)
    _assert_exact(dup_got, dup_ref)

    assert dp_got is None
    assert dp_ref is None


@pytest.mark.parametrize(
    "weighted",
    [
        False,
        True,
    ],
)
@requires_p5_cuda
def test_packed_layout_is_byte_exact_and_zero_copy(
    weighted: bool,
) -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    gate, up, p_s = (tensor.cuda() for tensor in fixtures.make_swiglu_boundary_inputs())

    gate_up = torch.cat(
        (
            gate,
            up,
        ),
        dim=1,
    ).contiguous()

    route_weights = p_s if weighted else None

    dh = fixtures.make_grad_output(
        "swiglu_boundary",
        tuple(gate.shape),
    ).cuda()

    h_ref, saved_ref = oracle.clamp_swiglu_weighted_fwd(
        gate,
        up,
        route_weights,
    )

    grads_ref = oracle.clamp_swiglu_weighted_bwd(
        dh,
        saved_ref,
    )

    h_got, saved_got = provider.clamp_swiglu_weighted_packed_fwd(
        gate_up,
        route_weights,
    )

    grads_got = provider.clamp_swiglu_weighted_packed_bwd(
        dh,
        saved_got,
    )

    _assert_exact(
        h_got,
        h_ref,
    )

    expected_saved_keys = {
        "gate_up32",
    }

    if route_weights is not None:
        expected_saved_keys.add("p_s")

    assert set(saved_got) == expected_saved_keys

    assert gate_up.dtype == torch.float32
    assert saved_got["gate_up32"].data_ptr() == gate_up.data_ptr()

    for got, want in zip(
        grads_got,
        grads_ref,
        strict=True,
    ):
        if want is None:
            assert got is None
        else:
            assert got is not None
            _assert_exact(
                got,
                want,
            )


@requires_p5_cuda
def test_repeated_runs_are_byte_exact() -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    generator = torch.Generator().manual_seed(63)

    gate = (
        torch.randn(
            7,
            257,
            generator=generator,
        )
        * 12.0
    ).cuda()

    up = (
        torch.randn(
            7,
            257,
            generator=generator,
        )
        * 12.0
    ).cuda()

    p_s = torch.rand(
        7,
        generator=generator,
    ).cuda()

    dh = (
        torch.randn(
            7,
            257,
            generator=generator,
        )
        .to(torch.bfloat16)
        .cuda()
    )

    h_first, saved_first = provider.clamp_swiglu_weighted_fwd(
        gate,
        up,
        p_s,
    )

    grads_first = provider.clamp_swiglu_weighted_bwd(
        dh,
        saved_first,
    )

    _assert_saved_state(
        saved_first,
        gate,
        up,
        p_s,
    )

    for _ in range(4):
        h_repeat, saved_repeat = provider.clamp_swiglu_weighted_fwd(
            gate,
            up,
            p_s,
        )

        grads_repeat = provider.clamp_swiglu_weighted_bwd(
            dh,
            saved_repeat,
        )

        _assert_exact(h_repeat, h_first)

        _assert_saved_state(
            saved_repeat,
            gate,
            up,
            p_s,
        )

        for got, want in zip(
            grads_repeat,
            grads_first,
            strict=True,
        ):
            assert got is not None
            assert want is not None
            _assert_exact(got, want)


@requires_p5_cuda
def test_empty_batch_and_fail_closed_validation() -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    empty = torch.empty(
        0,
        32,
        device="cuda",
    )

    p_s = torch.empty(
        0,
        device="cuda",
    )

    dh = torch.empty(
        0,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )

    h, saved = provider.clamp_swiglu_weighted_fwd(
        empty,
        empty,
        p_s,
    )

    dgate, dup, dp_s = provider.clamp_swiglu_weighted_bwd(
        dh,
        saved,
    )

    _assert_saved_state(
        saved,
        empty,
        empty,
        p_s,
    )

    assert h.shape == (0, 32)
    assert h.dtype == torch.bfloat16
    assert dgate.shape == (0, 32)
    assert dup.shape == (0, 32)
    assert dp_s is not None
    assert dp_s.shape == (0,)

    with pytest.raises(
        RuntimeError,
        match="CUDA tensor",
    ):
        provider.clamp_swiglu_weighted_fwd(
            empty.cpu(),
            empty.cpu(),
            p_s.cpu(),
        )

    with pytest.raises(
        ValueError,
        match="share shape",
    ):
        provider.clamp_swiglu_weighted_fwd(
            empty,
            torch.empty(
                0,
                16,
                device="cuda",
            ),
            p_s,
        )

    with pytest.raises(
        TypeError,
        match="p_s must have dtype fp32",
    ):
        provider.clamp_swiglu_weighted_fwd(
            empty,
            empty,
            p_s.to(torch.bfloat16),
        )
