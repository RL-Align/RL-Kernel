# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3-parallel-v2``: the gate/up, O and down projections run their rows' fixed FP32-tile GEMM.

The CPU tests pin the launch to ``FixedFP32Backend``'s (the #554 adapter that the rows'
operators use) and the kernel each tensor-parallel path picks. The GPU tests check that the
launch is bitwise ``FixedFP32Backend`` at the rows' TP=1 shapes, that a column shard is bitwise
those columns, and that the FP32 partial rounds to the BF16 output.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import fixed_fp32_gemm as fx
from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.backends.shared.triton.gemm import h3_projection as g

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Compiled:
    hash = "h"
    asm = {"cubin": b""}

    class metadata:
        num_warps = 4


class _Kernel:
    """Records the keyword options of a launch."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, kwargs))
            return _Compiled()

        return launch


def test_launch_is_fixed_fp32_backends(monkeypatch):
    pytest.importorskip("triton")
    from rl_engine.backends.cuda.gemm import h3_projection_fp32 as adapter
    from rl_engine.backends.shared.triton.gemm import det_gemm

    kernel = _Kernel()
    monkeypatch.setattr(det_gemm, "_det_gemm_fp32_kernel", kernel)
    backend = adapter.FixedFP32Backend()
    backend.gemm(torch.zeros(3, 4, dtype=torch.bfloat16), torch.zeros(4, 5, dtype=torch.bfloat16))
    ((_, options),) = kernel.launches
    assert options == fx.LAUNCH


def test_the_down_rows_launch_differs_only_in_index_width(monkeypatch):
    """#535's down backend leaves ``INDEX_64`` at the kernel's default (``False``); every other
    option is ``fx.LAUNCH``. The index width changes no value while offsets fit in 32 bits."""

    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.gemm import det_gemm, h3_down_fp32_backend

    kernel = _Kernel()
    monkeypatch.setattr(det_gemm, "_det_gemm_fp32_kernel", kernel)
    backend = h3_down_fp32_backend.FixedFP32Backend()
    backend.gemm(torch.zeros(3, 4, dtype=torch.bfloat16), torch.zeros(4, 5, dtype=torch.bfloat16))
    ((_, options),) = kernel.launches
    assert "INDEX_64" not in options  # the kernel's default, False
    defaults = {"COMPENSATED": False, "CHUNK_K": 0, "EVEN_K": False, "EVEN_N": False}
    assert {**defaults, **options, "INDEX_64": True} == fx.LAUNCH  # the rest is fx.LAUNCH
    assert 32768 * 14336 < 2**31  # the down row's largest operand: no 32-bit offset overflows


def test_rejects_non_bf16_operands():
    x = torch.zeros(2, 3)
    with pytest.raises(TypeError, match="BF16"):
        fx.gemm(x, x.t())


def _fixed_kernel_calls(fn) -> int:
    """How many calls in ``fn`` pass the keyword ``kernel="fixed"`` (from its syntax tree)."""

    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(
            kw.arg == "kernel" and isinstance(kw.value, ast.Constant) and kw.value.value == "fixed"
            for kw in node.keywords
        )
    )


def test_each_tensor_parallel_projection_uses_its_rows_kernel():
    assert par.CONTRACT == "h3-parallel-v2"
    assert par.KERNELS == {"projection": g.gemm, "fixed": fx.gemm}
    import inspect

    from rl_engine.validation.models import h3_sglang_rollout, h3_tp

    # the call itself, not its text: a comment or a reformat neither passes nor breaks this
    assert _fixed_kernel_calls(h3_tp.run_tp_block_rows) == 1  # gate/up; Q/K/V keep projection
    assert _fixed_kernel_calls(h3_sglang_rollout._canonical) >= 1
    assert inspect.signature(par._units_partial).parameters["kernel"].default is fx.gemm


@needs_gpu
@pytest.mark.parametrize(
    "rows, k, n", [(264, 7168, 5376), (925, 5376, 28672), (37, 14336, 5376), (1, 5376, 7168)]
)
def test_bitwise_the_rows_backend(rows, k, n):
    pytest.importorskip("triton")
    from rl_engine.backends.cuda.gemm.h3_projection_fp32 import FixedFP32Backend

    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, k, device="cuda", generator=gen).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).bfloat16()
    backend = FixedFP32Backend()
    assert torch.equal(fx.gemm(x, w, b_t=True), backend.det_gemm_fwd_rhs_transposed(x, w))
    grad = torch.randn(rows, n, device="cuda", generator=gen).bfloat16()
    assert torch.equal(fx.gemm(grad, w), backend.det_gemm_fwd(grad, w))  # dX
    assert torch.equal(fx.gemm(grad, x, a_t=True), backend.det_gemm_db_transposed(x, grad))  # dW
    partial = fx.gemm(x, w, b_t=True, out_dtype=torch.float32)
    assert torch.equal(partial.bfloat16(), fx.gemm(x, w, b_t=True))


@needs_gpu
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_column_shard_is_bitwise_those_columns(tp):
    pytest.importorskip("triton")
    gen = torch.Generator(device="cuda").manual_seed(tp)
    x = torch.randn(264, 5376, device="cuda", generator=gen).bfloat16()
    w = (torch.randn(28672, 5376, device="cuda", generator=gen) / 73).bfloat16()
    full = fx.gemm(x, w, b_t=True)
    shard = par.shard_gate_up(w, tp, 1)
    hidden, gate = fx.gemm(x, shard, b_t=True).chunk(2, dim=1)
    width = 14336 // tp
    assert torch.equal(hidden, full[:, width : 2 * width])
    assert torch.equal(gate, full[:, 14336 + width : 14336 + 2 * width])


@needs_gpu
def test_gate_up_lora_base_is_the_rows_kernel():
    """With B = 0 a gate/up LoRA projection (``kernel="fixed"``) is the row's GEMM, by value."""

    pytest.importorskip("triton")
    from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl

    gen = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(264, 5376, device="cuda", generator=gen).bfloat16()
    hidden = (torch.randn(1792, 5376, device="cuda", generator=gen) / 73).bfloat16()
    gate = (torch.randn(1792, 5376, device="cuda", generator=gen) / 73).bfloat16()
    a = (torch.randn(8, 5376, device="cuda", generator=gen) / 73).bfloat16()
    b_hidden = torch.zeros(1792, 8, device="cuda").bfloat16()
    b_gate = torch.zeros(1792, 8, device="cuda").bfloat16()
    outs = pl.column_parallel_lora_linear(
        x, [hidden, gate], [(a, [b_hidden, b_gate])], 2.0, kernel="fixed"
    )
    assert len(outs) == 2  # hidden and gate, both checked
    for out, w in zip(outs, (hidden, gate), strict=True):
        assert torch.equal(out.float(), fx.gemm(x, w, b_t=True).float())


needs_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the GEMM rows' operators run on SM90 only",
)


@needs_sm90
@pytest.mark.parametrize("row", ["h3_attention_o_gemm", "h3_ffn_down_gemm", "h3_ffn_gate_up_gemm"])
def test_tp_pieces_against_the_rows_operators(row):
    """Gate/up forward, O / down dX and every dW shard are bitwise the rows' TP=1 operators."""

    pytest.importorskip("triton")
    from tools.validation.models import h3_gemm_rows_crosscheck as cc

    result = cc.check(row, 264, seed=1)
    assert result["pass"], result


def test_crosscheck_diff_compares_bytes_and_flags_non_finite():
    """``-0.0`` vs ``+0.0`` is equal by value, not by bytes; a NaN gives a JSON-safe ``None``."""

    import json

    from tools.validation.models import h3_gemm_rows_crosscheck as cc

    zero = torch.zeros(2, 3, dtype=torch.bfloat16)
    d = cc._diff(zero, -zero)
    assert not d["bitwise"] and d["finite"] and d["max_abs"] == 0.0
    assert cc._diff(zero, zero.clone())["bitwise"]
    nan = zero.clone()
    nan[0, 0] = float("nan")
    d = cc._diff(nan, zero)
    assert not d["finite"] and d["max_abs"] is None
    json.dumps(d, allow_nan=False)
    assert not cc._diff(zero, zero.float())["bitwise"]  # another dtype is not bitwise


def test_an_unknown_kernel_name_raises():
    with pytest.raises(ValueError, match="unknown kernel 'fixd'"):
        par.kernel_of("fixd")
    assert par.kernel_of("fixed") is fx.gemm and par.kernel_of("projection") is g.gemm


@needs_gpu
@pytest.mark.parametrize(("m", "k", "n"), [(0, 64, 32), (16, 64, 0), (16, 0, 32)])
def test_empty_operands_give_the_exact_empty_or_zero_result(m, k, n):
    a = torch.randn(m, k, device="cuda").bfloat16()
    b = torch.randn(k, n, device="cuda").bfloat16()
    for out_dtype in (torch.bfloat16, torch.float32):
        c = fx.gemm(a, b, out_dtype=out_dtype)
        assert c.shape == (m, n) and c.dtype == out_dtype and not c.any()


@pytest.mark.parametrize("tp", [0, 3, 16])
def test_a_tp_size_that_does_not_divide_the_units_is_refused(tp):
    w = torch.zeros(16, 32)
    with pytest.raises(ValueError, match="tp must be one of"):
        par.check_column_widths([w], tp)
    with pytest.raises(ValueError, match="tp must be one of"):
        par._units_partial(torch.zeros(2, 32), w, tp)


def test_crosscheck_bounds_two_summation_orders_and_rejects_a_wrong_result():
    """Two FP32 orders of one sum are within the bound; a materially wrong reduction is not."""

    from tools.validation.models import h3_gemm_rows_crosscheck as cc

    gen = torch.Generator().manual_seed(5)
    k, n, rows = 7168, 16, 24
    x = torch.randn(rows, k, generator=gen)
    w = torch.randn(n, k, generator=gen) / k**0.5
    exact = x.double() @ w.t().double()
    order_a = (x[:, :, None] * w.t()[None]).sum(1).float().bfloat16()
    perm = torch.randperm(k, generator=gen)
    order_b = (x[:, perm, None] * w.t()[perm][None]).sum(1).float().bfloat16()
    for y in (order_a, order_b):
        bound = cc._error_bound(x, w.t(), k, y, order_a)
        d = cc._diff(y, order_a, bound)
        assert d["finite"] and d["within_bound"]
    bad = (exact * 1.05).float().bfloat16()  # 5% off is far beyond the summation error
    d = cc._diff(bad, order_a, cc._error_bound(x, w.t(), k, bad, order_a))
    assert d["within_bound"] is False
    assert cc._diff(order_a, order_a)["within_bound"] is None  # no bound given


def test_crosscheck_max_abs_is_none_on_a_shape_or_dtype_mismatch():
    from tools.validation.models import h3_gemm_rows_crosscheck as cc

    a = torch.zeros(2, 3, dtype=torch.bfloat16)
    assert cc._diff(a, torch.zeros(1, 3, dtype=torch.bfloat16))["max_abs"] is None
    assert cc._diff(a, torch.zeros(2, 3))["max_abs"] is None
