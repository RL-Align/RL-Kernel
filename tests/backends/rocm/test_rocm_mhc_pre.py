# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Checks for the ROCm mHC pre op, split into DSV4 RFC tasks 1, 2 and 3 (#508).

aiter computes all three tasks inside one op, ``mhc_pre_dsv4``, which only
returns post_mix, comb_mix and layer_input. The tests call that public op only
and separate the tasks by constructing inputs that make every step upstream of
the task under test exact and independent of accumulation order. A bits
difference across M can then only come from the task under test:

- task 1 (projection and RMS): sinkhorn_repeat=0, scale=1, base=0 make comb_mix
  equal (x @ fn_res^T) / r for the 16 res columns. One-hot fn rows make the
  projection exact (isolating RMS); residual entries in {+-1, +-2} make the sum
  of squares exact in any order (isolating the projection).
- task 2 (split/activation and Sinkhorn): one-hot fn rows and {+-1, +-2}
  residuals make the logits exact, so post_mix and comb_mix only depend on the
  activation and Sinkhorn. sinkhorn_repeat=t exposes each Sinkhorn round.
- task 3 (aggregation): zero pre rows of fn and pre biases of +-100 make the
  pre weights exactly fl(1 + 1e-6) or fl(1e-6), so layer_input only depends on
  the four-stream sum and its bf16 cast.

Each batch-invariance check runs with the integrated op (per-M config table)
and, as a diagnostic, with aiter's own config choice. M covers the WS1
manifest's token counts (8, 59), aiter's K-split switch points (1024 on) and
the upper bound of every bucket of the gfx942 mHC pre table, so every tuned
config runs. The task 1 cases are also checked under chunking, reordering and
padding, and in other storage strides, with the integrated op. The whole op is
checked in tests/backends/rocm/test_rocm_mhc_batch_invariance.py.

Accuracy against the fp32 reference uses the WS1 contract tolerance, for the
whole op and for the residual gradient under extreme Sinkhorn logits.
"""

import pytest
import torch

from rl_engine.backends.rocm import config_utils
from rl_engine.backends.rocm.mhc.aiter_mhc import FIXED_PRE_CONFIG, PRE_BIT_PARAMS, check_pre_inputs
from rl_engine.reference.mhc import NativeMHCPreOp, sinkhorn_dsv4
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import OperatorRequirements
from rl_engine.validation.common.batch_invariance import (
    RowwiseCase,
    bits_equal,
    differing_after_chunk_reorder_pad,
    differing_after_restride,
    differing_in_batch,
)
from tests.helpers.contract_tolerance import assert_matches_fp32_reference
from tests.helpers.rocm_launch_configs import bucket_bounds

_N, _C = 4, 4096  # DeepSeek-V4-Flash-0731: hc_mult, hidden_size
_K = _N * _C
_MIX = 2 * _N + _N * _N  # fn rows: [pre n | post n | res n*n]
_PRE, _POST, _RES = slice(0, _N), slice(_N, 2 * _N), slice(2 * _N, _MIX)
_M_VALUES = sorted(
    {1, 8, 16, 17, 59, 1024, 2048, 4096, 8192, 16384}
    | set(bucket_bounds("mhc_pre", f"MHC_PRE-n={_N}-C={_C}.json"))
)
_EPS = 1e-6


def _ops():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return None
    try:
        from aiter.ops.triton.fusions import mhc_pre_dsv4

        from rl_engine.backends.rocm.mhc.aiter_mhc import AiterMHCPreOp

        op = AiterMHCPreOp()
    except (ImportError, RuntimeError):
        return None

    def aiter_default(residual, fn, scale, base, **kwargs):
        post, comb, layer_input = mhc_pre_dsv4(residual, fn, scale, base, **kwargs)
        return post.squeeze(-1), comb, layer_input

    return {"rlkernel_config": op, "aiter_default_config": aiter_default}


_OPS = _ops()
requires_aiter = pytest.mark.skipif(
    _OPS is None, reason="ROCm GPU with aiter DeepSeek-V4 mHC ops required."
)


def _config_and_m(*, default_split_k_may_differ):
    """(config, M) pairs. aiter's own config cannot launch for M in [8192, 16384)."""
    params = []
    for config in ("rlkernel_config", "aiter_default_config"):
        for m in _M_VALUES:
            marks = [requires_aiter]
            if config == "aiter_default_config" and 8192 <= m < 16384:
                marks.append(pytest.mark.xfail(strict=True, reason="needs 128 KB LDS, has 64 KB"))
            elif config == "aiter_default_config" and m >= 1024 and default_split_k_may_differ:
                marks.append(pytest.mark.xfail(reason="aiter's own config splits K for M >= 1024"))
            params.append(pytest.param(config, m, marks=marks, id=f"{config}-M{m}"))
    return params


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal_residual(m, seed):
    return torch.randn(m, _N, _C, generator=_gen(seed), device="cuda").to(torch.bfloat16)


def _exact_residual(m, seed):
    """Entries in {-2, -1, 1, 2}: the sum of squares is an integer <= 65536, exact in fp32."""
    gen = _gen(seed)
    magnitude = torch.randint(1, 3, (m, _N, _C), generator=gen, device="cuda")
    sign = 2 * torch.randint(0, 2, (m, _N, _C), generator=gen, device="cuda") - 1
    return (magnitude * sign).to(torch.bfloat16)


def _random_fn(seed=0):
    return torch.randn(_MIX, _K, generator=_gen(seed), device="cuda") / _K**0.5


def _one_hot_fn(rows):
    """fn row r has a single 1 (exact in bf16), so its projection is one input element."""
    fn = torch.zeros(_MIX, _K, device="cuda")
    for r in range(_MIX)[rows]:
        fn[r, (r * 997) % _K] = 1.0
    return fn


def _ones_scale():
    return torch.ones(3, device="cuda")


def _zero_base():
    return torch.zeros(_MIX, device="cuda")


def _case(name, op, make_residual, fn, scale, base, output_index, **kwargs):
    def run(rows):
        return [op(rows[0], fn, scale, base, **kwargs)[output_index]]

    return RowwiseCase(name, (name,), run, lambda m, seed: [make_residual(m, seed)])


def _assert_batch_invariant(case, m):
    differing = differing_in_batch(case, m)
    assert not differing, f"M={m}: {differing} differs from M=1"


def _pre_forward_backward(op, residual, fn, scale, base, grads):
    """Outputs and dResidual (fp32) of op for one output gradient per output."""
    leaf = residual.detach().clone().requires_grad_(True)
    outputs = op(leaf, fn, scale, base)
    torch.autograd.backward(outputs, [g.to(o.dtype) for g, o in zip(grads, outputs)])
    return [o.detach().float() for o in outputs], leaf.grad.float()


# ---------------------------------------------------------------------------
# Task 1: projection and RMS (fp32_gemm_rms)
# With sinkhorn_repeat=0, scale=1 and base=0, comb_mix is (x @ fn_res^T) / r.
# ---------------------------------------------------------------------------


# The three task 1 cases: (residual, fn) builders. The output is comb_mix.
_TASK1_INPUTS = {
    "projection_and_rms": (_normal_residual, _random_fn),
    # One-hot fn rows: the projection is a single input element, exact in any
    # order, so a difference can only come from the RMS (sum of squares)
    # accumulation.
    "rms_alone": (_normal_residual, lambda: _one_hot_fn(_RES)),
    # Residual entries in {+-1, +-2}: r is exact in any order, so a difference
    # can only come from the projection accumulation.
    "projection_alone": (_exact_residual, _random_fn),
}


def _task1_case(config, inputs):
    make_residual, make_fn = _TASK1_INPUTS[inputs]
    return _case(
        "projection_over_r",
        _OPS[config],
        make_residual,
        make_fn(),
        _ones_scale(),
        _zero_base(),
        1,
        sinkhorn_repeat=0,
    )


@pytest.mark.parametrize("config,m", _config_and_m(default_split_k_may_differ=True))
def test_task1_projection_and_rms_batch_invariance(config, m):
    _assert_batch_invariant(_task1_case(config, "projection_and_rms"), m)


@pytest.mark.parametrize("config,m", _config_and_m(default_split_k_may_differ=True))
def test_task1_rms_alone_batch_invariance(config, m):
    _assert_batch_invariant(_task1_case(config, "rms_alone"), m)


@pytest.mark.parametrize("config,m", _config_and_m(default_split_k_may_differ=True))
def test_task1_projection_alone_batch_invariance(config, m):
    _assert_batch_invariant(_task1_case(config, "projection_alone"), m)


# The same three cases under chunking, reordering and padding, and in other
# storage strides, with the integrated op.
@requires_aiter
@pytest.mark.parametrize("inputs", list(_TASK1_INPUTS))
def test_task1_chunk_reorder_pad_invariance(inputs):
    differing = differing_after_chunk_reorder_pad(_task1_case("rlkernel_config", inputs), 4096)
    assert not differing, f"{inputs}: {differing}"


@requires_aiter
@pytest.mark.parametrize("inputs", list(_TASK1_INPUTS))
def test_task1_restride_invariance(inputs):
    differing = differing_after_restride(_task1_case("rlkernel_config", inputs), 1024)
    assert not differing, f"{inputs}: {differing}"


# RMS formula. With mean(x^2) = 2^-20, close to eps = 1e-6, the two candidate
# formulas differ by about 30%: DeepSeek-V4 and aiter use rsqrt(mean + eps); the
# issue's rule is 1 / (sqrt(mean) + eps). This records which one aiter computes;
# the choice is an open decision for the RFC author. On MI325X aiter differed
# from the rsqrt formula by 1.5e-5 relative at this small mean; the cause was
# not investigated, so the tolerance is 1e-4.
@requires_aiter
def test_task1_rms_formula_is_rsqrt_of_mean_plus_eps():
    m = 64
    sign = torch.sign(_normal_residual(m, 1).float())
    residual = (sign * 2.0**-10).to(torch.bfloat16)  # every |x| = 2^-10, mean(x^2) = 2^-20
    fn = _one_hot_fn(_RES)
    _, comb, _ = _OPS["rlkernel_config"](
        residual, fn, _ones_scale(), _zero_base(), sinkhorn_repeat=0
    )
    projection = (residual.float().flatten(-2) @ fn[_RES].t()).view(m, _N, _N)
    mean = torch.tensor(2.0**-20)
    dsv4 = projection * torch.rsqrt(mean + _EPS)
    issue_rule = projection / (torch.sqrt(mean) + _EPS)
    assert (comb - dsv4).abs().max() <= 1e-4 * dsv4.abs().max()
    assert (comb - issue_rule).abs().max() > 0.1 * issue_rule.abs().max()


# Projection precision. One nonzero input element per token makes the projection
# equal to one fn entry: aiter rounds fn to bf16 before multiplying, while the
# DeepSeek-V4 reference multiplies in fp32. This records aiter's behavior; the
# project's choice is an open decision for the RFC author.
@requires_aiter
def test_task1_projection_rounds_fn_to_bf16():
    m = 64
    residual = torch.zeros(m, _N, _C, device="cuda", dtype=torch.bfloat16)
    residual[:, 0, 0] = 1.0
    fn = _random_fn(seed=3)
    _, comb, _ = _OPS["rlkernel_config"](
        residual, fn, _ones_scale(), _zero_base(), sinkhorn_repeat=0
    )
    r_inv = torch.rsqrt(torch.tensor(1.0 / _K) + _EPS)
    fp32_weights = (fn[_RES, 0] * r_inv).view(_N, _N).expand(m, _N, _N)
    bf16_weights = (fn[_RES, 0].bfloat16().float() * r_inv).view(_N, _N).expand(m, _N, _N)
    assert (comb - bf16_weights).abs().max() <= 1e-6 * bf16_weights.abs().max()
    assert (comb - fp32_weights).abs().max() > 1e-4 * fp32_weights.abs().max()


# ---------------------------------------------------------------------------
# Task 2: split/activation and Sinkhorn (hc_split_sinkhorn)
# One-hot fn rows and {+-1, +-2} residuals make the logits exact, so post_mix
# and comb_mix depend only on the activation and the Sinkhorn kernel.
# ---------------------------------------------------------------------------


def _task2_params():
    gen = torch.Generator(device="cuda").manual_seed(5)
    scale = 0.5 + torch.rand(3, generator=gen, device="cuda")
    base = 0.5 * torch.randn(_MIX, generator=gen, device="cuda")
    return _one_hot_fn(slice(_N, _MIX)), scale, base


@pytest.mark.parametrize("config,m", _config_and_m(default_split_k_may_differ=False))
def test_task2_post_and_sinkhorn_batch_invariance(config, m):
    fn, scale, base = _task2_params()
    for output_index, name in ((0, "post_mix"), (1, "comb_mix")):
        case = _case(name, _OPS[config], _exact_residual, fn, scale, base, output_index)
        _assert_batch_invariant(case, m)


# Each Sinkhorn round against DeepSeek-V4's algorithm, starting from the op's own
# logits (sinkhorn_repeat=0) so that only the Sinkhorn arithmetic is compared.
@requires_aiter
def test_task2_each_sinkhorn_round_matches_reference():
    fn, scale, base = _task2_params()
    residual = _exact_residual(256, 6)
    op = _OPS["rlkernel_config"]
    logits = op(residual, fn, scale, base, sinkhorn_repeat=0)[1]
    worst = {}
    for rounds in range(1, 21):
        comb = op(residual, fn, scale, base, sinkhorn_repeat=rounds)[1]
        worst[rounds] = (comb - sinkhorn_dsv4(logits, rounds, _EPS)).abs().max().item()
    assert max(worst.values()) < 2e-6, f"max abs diff per round: {worst}"


# Split and POST multiplier: post_mix = mult * sigmoid(logit) for the post rows,
# with the logit computed on the host in the kernel's operation order.
@requires_aiter
@pytest.mark.parametrize("mult", [2.0, 3.0])
def test_task2_post_mix_split_and_multiplier(mult):
    fn, scale, base = _task2_params()
    residual = _exact_residual(256, 7)
    post = _OPS["rlkernel_config"](residual, fn, scale, base, hc_post_mult_value=mult)[0]
    x = residual.float().flatten(-2)
    r_inv = 1.0 / torch.sqrt(x.square().sum(-1, keepdim=True) / _K + _EPS)
    logit = r_inv * scale[1] * (x @ fn[_POST].t()) + base[_POST]
    expected = mult * torch.sigmoid(logit)
    assert (post - expected).abs().max() <= 1e-6 * expected.abs().max()


_EXTREME_RES_LOGITS = pytest.mark.parametrize(
    "res_logits",
    [
        torch.zeros(_N * _N),  # all equal
        torch.linspace(-60, 60, _N * _N),  # wide spread
        torch.tensor([80.0] + [0.0] * (_N * _N - 1)),  # one dominant entry
    ],
    ids=["equal", "wide_spread", "one_dominant"],
)


# Extreme logits: with a zero fn and scale 0 the logits equal base exactly.
@_EXTREME_RES_LOGITS
@requires_aiter
def test_task2_sinkhorn_extreme_logits(res_logits):
    base = torch.zeros(_MIX, device="cuda")
    base[_RES] = res_logits.cuda()
    fn = torch.zeros(_MIX, _K, device="cuda")
    residual = _normal_residual(16, 8)
    comb = _OPS["rlkernel_config"](residual, fn, torch.zeros(3, device="cuda"), base)[1]
    expected = sinkhorn_dsv4(base[_RES].view(1, _N, _N), 20, _EPS).expand(16, _N, _N)
    assert torch.isfinite(comb).all()
    assert (comb - expected).abs().max() < 2e-6


# Gradient under the same extreme logits, through aiter's PyTorch backward and
# the fp32 reference. Construction: one-hot fn on the res rows only and
# scale = (0, 0, 0.1), so each res logit is base + 0.1 * x_k / r for one input
# element x_k: within a few tenths of the extreme base values, but with a
# nonzero derivative. Only comb_mix gets an output gradient, so dResidual
# passes through Sinkhorn, the projection and RMS. That gradient is scaled by
# 100 so dResidual reaches O(1) where the absolute tolerance is not loose.
@_EXTREME_RES_LOGITS
@requires_aiter
def test_task2_sinkhorn_extreme_logits_grad(res_logits):
    m = 16
    base = torch.zeros(_MIX, device="cuda")
    base[_RES] = res_logits.cuda()
    fn = _one_hot_fn(_RES)
    scale = torch.tensor([0.0, 0.0, 0.1], device="cuda")
    residual = _normal_residual(m, 17)
    grads = [
        torch.zeros(m, _N, device="cuda"),
        100.0 * torch.randn(m, _N, _N, generator=_gen(18), device="cuda"),
        torch.zeros(m, _C, device="cuda", dtype=torch.bfloat16),
    ]
    _, grad = _pre_forward_backward(_OPS["rlkernel_config"], residual, fn, scale, base, grads)
    _, ref_grad = _pre_forward_backward(
        NativeMHCPreOp().forward_fp32, residual, fn, scale, base, grads
    )
    assert torch.isfinite(grad).all()
    assert ref_grad.any()
    assert_matches_fp32_reference(
        grad, ref_grad, judgment="gradient_accuracy", op_class="reduction", name="dResidual"
    )


# ---------------------------------------------------------------------------
# Task 3: four-stream aggregation (h_aggregate)
# Zero pre rows of fn and pre biases of +-100 make the pre weights exactly
# fl(1 + 1e-6) or fl(1e-6), so layer_input depends only on the four-stream sum
# and its bf16 cast.
# ---------------------------------------------------------------------------

_PRE_BIAS = torch.tensor([100.0, -100.0, 100.0, 100.0])


def _task3_params():
    fn = _random_fn(seed=9)
    fn[_PRE] = 0.0
    base = 0.1 * torch.randn(_MIX, generator=_gen(10), device="cuda")
    base[_PRE] = _PRE_BIAS.cuda()
    return fn, 0.5 + torch.rand(3, generator=_gen(11), device="cuda"), base


def _exact_pre_weights():
    one = torch.tensor(1.0, dtype=torch.float32)
    return torch.where(_PRE_BIAS > 0, one + _EPS, torch.tensor(_EPS)).cuda()


@pytest.mark.parametrize("config,m", _config_and_m(default_split_k_may_differ=False))
def test_task3_aggregation_batch_invariance(config, m):
    fn, scale, base = _task3_params()
    case = _case("layer_input", _OPS[config], _normal_residual, fn, scale, base, 2)
    _assert_batch_invariant(case, m)


# Precondition of the task-3 isolation: the pre weights really are the exact
# constants. With only stream i nonzero, layer_input must equal bf16(w_i * x_i).
@requires_aiter
def test_task3_pre_weights_are_exact():
    fn, scale, base = _task3_params()
    w = _exact_pre_weights()
    differing = []
    for i in range(_N):
        residual = torch.zeros(256, _N, _C, device="cuda", dtype=torch.bfloat16)
        residual[:, i] = _normal_residual(256, 20 + i)[:, i]
        layer_input = _OPS["rlkernel_config"](residual, fn, scale, base)[2]
        if not bits_equal(layer_input, (w[i] * residual[:, i].float()).to(torch.bfloat16)):
            differing.append(i)
    assert not differing, f"streams whose pre weight is not the expected constant: {differing}"


# Summation order and cast point. aiter sums the four streams as
#   fma(w2, x2, w0 * x0) + fma(w3, x3, w1 * x1)
# in fp32 and casts to bf16 once at the end: streams two apart are paired, each
# pair is one fused multiply-add. Found on MI325X by comparing against all 24
# sequential orders, the 3 pairwise groupings and their FMA variants; only this
# one matched every element. Fails if aiter changes it.
@requires_aiter
def test_task3_aggregation_order_and_bf16_cast():
    fn, scale, base = _task3_params()
    residual = _normal_residual(1024, 12)
    layer_input = _OPS["rlkernel_config"](residual, fn, scale, base)[2]
    w = _exact_pre_weights().double()
    x = residual.double()

    def fma(i, j):
        """w_i * x_i + fl(w_j * x_j) with one rounding (exact product in fp64)."""
        return (w[i] * x[:, i] + (w[j] * x[:, j]).float().double()).float()

    expected = (fma(2, 0) + fma(3, 1)).to(torch.bfloat16)
    assert bits_equal(layer_input, expected)


@requires_aiter
def test_task3_empty_batch():
    fn, scale, base = _task3_params()
    outputs = _OPS["rlkernel_config"](_normal_residual(0, 13), fn, scale, base)
    assert [tuple(o.shape) for o in outputs] == [(0, _N), (0, _N, _N), (0, _C)]


# ---------------------------------------------------------------------------
# Whole op: correctness against the fp32 reference, the registry selection,
# the shipped table and the semantic descriptor
# ---------------------------------------------------------------------------


@requires_aiter
def test_pre_matches_reference_forward_and_grad():
    fn = _random_fn()
    scale = 0.5 + torch.rand(3, generator=_gen(14), device="cuda")
    base = 0.1 * torch.randn(_MIX, generator=_gen(15), device="cuda")
    residual = _normal_residual(64, 16)
    grads = [
        torch.randn(64, _N, device="cuda"),
        torch.randn(64, _N, _N, device="cuda"),
        torch.randn(64, _C, device="cuda").to(torch.bfloat16),
    ]
    outputs, grad = _pre_forward_backward(_OPS["rlkernel_config"], residual, fn, scale, base, grads)
    ref_outputs, ref_grad = _pre_forward_backward(
        NativeMHCPreOp().forward_fp32, residual, fn, scale, base, grads
    )
    for name, out, ref in zip(("post_mix", "comb_mix", "layer_input"), outputs, ref_outputs):
        assert_matches_fp32_reference(
            out, ref, judgment="forward_accuracy", op_class="reduction", name=name
        )
    assert_matches_fp32_reference(
        grad, ref_grad, judgment="gradient_accuracy", op_class="reduction", name="dResidual"
    )


@requires_aiter
def test_registry_selects_aiter_on_rocm():
    assert type(kernel_registry.get_op("mhc_pre", device="cuda")).__name__ == "AiterMHCPreOp"


def test_shipped_table_loads_and_is_used(monkeypatch):
    monkeypatch.setattr(config_utils, "_device_arch", lambda index: "gfx942")
    config_utils.load_table.cache_clear()
    config_utils._bucket_table.cache_clear()
    config_utils._SELECTED.clear()
    try:
        file_name = f"MHC_PRE-n={_N}-C={_C}.json"
        path = config_utils.CONFIG_ROOT / "gfx942" / "mhc_pre" / file_name
        table = config_utils.load_table(path, PRE_BIT_PARAMS)
        assert table is not None and "any" in table
        chosen = config_utils.get_config(
            "mhc_pre", file_name, 1, FIXED_PRE_CONFIG, PRE_BIT_PARAMS, torch.device("cuda", 0)
        )
        assert chosen == table["M_LEQ_1"]
    finally:
        config_utils.load_table.cache_clear()
        config_utils._bucket_table.cache_clear()
        config_utils._SELECTED.clear()


def test_semantic_descriptor_declares_forward_batch_invariance_on_rocm():
    resolution = kernel_registry.semantic.session().resolve(
        semantic_op="mhc_pre",
        requested_backend="rlkernel.rocm.aiter_mhc_pre_dsv4.batch_invariant.v1",
        target="rollout",
        requirements=OperatorRequirements(
            device="rocm", dtype="bfloat16", alignment_properties={"batch_invariant": True}
        ),
    )
    properties = resolution.descriptor.determinism_or_alignment_properties
    assert properties["backward_batch_invariant"] is False


# Unsupported inputs raise instead of running unvalidated (checked on CPU).
@pytest.mark.parametrize(
    "change,error",
    [
        ({"residual": torch.zeros(2, _N, 64)}, TypeError),  # fp32 residual
        ({"residual": torch.zeros(2, 2, 64, dtype=torch.bfloat16)}, ValueError),  # n = 2
        ({"fn": torch.zeros(_MIX, _N * 64, dtype=torch.bfloat16)}, TypeError),
        ({"fn": torch.zeros(_MIX, _N * 32)}, ValueError),
        ({"scale": torch.zeros(2)}, ValueError),
        ({"base": torch.zeros(_MIX, dtype=torch.float64)}, TypeError),
    ],
    ids=["fp32_residual", "two_streams", "bf16_fn", "fn_shape", "scale_shape", "fp64_base"],
)
def test_unsupported_inputs_are_rejected(change, error):
    inputs = {
        "residual": torch.zeros(2, _N, 64, dtype=torch.bfloat16),
        "fn": torch.zeros(_MIX, _N * 64),
        "scale": torch.zeros(3),
        "base": torch.zeros(_MIX),
    }
    check_pre_inputs(**inputs)
    with pytest.raises(error):
        check_pre_inputs(**dict(inputs, **change))
