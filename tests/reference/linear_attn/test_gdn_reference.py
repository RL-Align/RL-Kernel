# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU oracles and regressions for the single-token references from PR #469."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.reference.linear_attn import CausalConv1dUpdateOp, GatedDeltaRuleRecurrentStepOp


def _inputs(dtype, k=33, v=5):
    generator = torch.Generator().manual_seed(469)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)

    # H=2, HV=4 exercises grouped head indexing; noncontiguous active blocks.
    return (
        rand(4, 4 * k + 4 * v),
        rand(4, 4),
        rand(4, 4),
        rand(4).float(),
        rand(4).float(),
        rand(6, 4, v, k) * 0.1,
        torch.tensor([4, 0, 1, -1], dtype=torch.int32),
    )


def _gdn_oracle(args, normalize):
    # Independent scalar-head, FP64 matrix formulation: no reference helpers,
    # packed batched indexing, or fixed-chunk contractions are reused here.
    packed, a, b, A_log, dt_bias, state, indices = args
    k, v = state.shape[-1], state.shape[-2]
    out = torch.zeros(4, 1, 4, v, dtype=torch.float64)
    updated = state.double().clone()
    for row, block in enumerate(indices.tolist()):
        if block <= 0:
            continue
        for head in range(4):
            key_head = head // 2
            q = packed[row, key_head * k : (key_head + 1) * k].double()
            key = packed[row, (2 + key_head) * k : (3 + key_head) * k].double()
            value = packed[row, 4 * k + head * v : 4 * k + (head + 1) * v].double()
            if normalize:
                q = q / (q.square().sum() + 1e-6).sqrt()
                key = key / (key.square().sum() + 1e-6).sqrt()
            gate = a[row, head].double() + dt_bias[head].double()
            softplus = gate if gate > 20 else torch.log1p(gate.exp())
            decay = torch.exp(-A_log[head].double().exp() * softplus)
            h = state[block, head].double() * decay
            delta = (value - h @ key) * b[row, head].double().sigmoid()
            h = h + torch.outer(delta, key)
            out[row, 0, head] = h @ (q * k**-0.5)
            updated[block, head] = h
    return out, updated


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
def test_gdn_recurrence_matches_independent_fp64_oracle(dtype, normalize):
    args = _inputs(dtype)
    before = args[-2].clone()
    op = GatedDeltaRuleRecurrentStepOp()
    kwargs = dict(scale=33**-0.5, num_k_heads=2, use_qk_l2norm=normalize)
    out32, state32 = op.forward_fp32(*args, **kwargs)
    expected_out, expected_state = _gdn_oracle(args, normalize)
    torch.testing.assert_close(out32.double(), expected_out, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(state32.double(), expected_state, atol=2e-6, rtol=2e-5)
    out, state = op(*args, **kwargs)
    assert out.shape == (4, 1, 4, 5)
    assert out.dtype == state.dtype == dtype
    assert out32.dtype == state32.dtype == torch.float32
    assert torch.equal(out, out32.to(dtype))
    assert torch.equal(state, state32.to(dtype))
    assert torch.equal(args[-2], before)
    assert torch.count_nonzero(out[[1, 3]]) == 0
    assert torch.equal(state[[0, 2, 3, 5]], before[[0, 2, 3, 5]])
    forwarded = op.forward(*args, **kwargs)
    assert torch.equal(out, forwarded[0])
    assert torch.equal(state, forwarded[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k,v", [(7, 3), (32, 32), (33, 5), (128, 128)])
def test_gdn_batch_invariance_and_repeated_state_cast(dtype, k, v):
    # Adapted from #469's golden-only batch and state-rounding checks, without
    # its CUDA device, model fingerprint, or measured GPU drift bounds.
    args = list(_inputs(dtype, k, v))
    op = GatedDeltaRuleRecurrentStepOp()
    kwargs = dict(scale=k**-0.5, num_k_heads=2)
    full, state = op(*args, **kwargs)
    rows = torch.tensor([2, 0])
    subset = [x[rows] if i in (0, 1, 2, 6) else x for i, x in enumerate(args)]
    part, part_state = op(*subset, **kwargs)
    assert torch.equal(part, full[rows])
    assert torch.equal(part_state, state)
    for row in (0, 2):
        alone = [x[row : row + 1] if i in (0, 1, 2, 6) else x for i, x in enumerate(args)]
        out, cache = op(*alone, **kwargs)
        assert torch.equal(out[0], full[row])
        block = args[-1][row]
        assert torch.equal(cache[block], state[block])
    for _ in range(3):
        _, widened = op.forward_fp32(*args, **kwargs)
        _, rounded = op(*args, **kwargs)
        assert torch.equal(rounded, widened.to(dtype))
        args[-2] = rounded


@pytest.mark.parametrize("heads", [1, 4, 8, 16, 33])
def test_gdn_batch_invariance_across_gate_vector_widths(heads):
    generator = torch.Generator().manual_seed(469)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    # Cross scalar/vector gate boundaries, including a non-vector-aligned head count.
    args = [
        rand(5, 2 * 7 + heads * 3),
        rand(5, heads),
        rand(5, heads),
        rand(heads),
        rand(heads),
        rand(6, heads, 3, 7) * 0.1,
        torch.arange(1, 6),
    ]
    op = GatedDeltaRuleRecurrentStepOp()
    full, state = op(*args, scale=7**-0.5, num_k_heads=1)
    for rows in (torch.tensor([4, 0]), torch.tensor([2]), torch.arange(4, -1, -1)):
        subset = [x[rows] if i in (0, 1, 2, 6) else x for i, x in enumerate(args)]
        part, part_state = op(*subset, scale=7**-0.5, num_k_heads=1)
        assert torch.equal(part, full[rows])
        blocks = args[-1][rows]
        assert torch.equal(part_state[blocks], state[blocks])
        untouched = torch.ones(state.shape[0], dtype=torch.bool)
        untouched[blocks] = False
        assert torch.equal(part_state[untouched], args[-2][untouched])


def test_gdn_backward_matches_independent_fp64_oracle():
    args = tuple(
        x.detach().requires_grad_() if x.is_floating_point() else x for x in _inputs(torch.float32)
    )
    op = GatedDeltaRuleRecurrentStepOp()
    out, state = op(*args, scale=33**-0.5, num_k_heads=2)
    expected_out, expected_state = _gdn_oracle(args, normalize=True)
    inputs = args[:-1]
    gradients = torch.autograd.grad(out.sum() + state.sum(), inputs)
    expected_gradients = torch.autograd.grad(expected_out.sum() + expected_state.sum(), inputs)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected.double(), rtol=2e-5, atol=2e-6)
    for gradient in gradients[:3]:
        assert torch.count_nonzero(gradient[[1, 3]]) == 0
    assert torch.equal(gradients[-1][[0, 2, 3, 5]], torch.ones_like(args[-2][[0, 2, 3, 5]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("activation", [None, "silu", "swish"])
@pytest.mark.parametrize("width", [1, 4])
def test_conv_matches_scalar_oracle_and_layout(dtype, dim_first, activation, width):
    torch.manual_seed(469)
    x = torch.randn(4, 3)  # Incoming fp32 token must round to the cache dtype.
    cache = torch.randn(6, 3, width - 1).to(dtype)
    weight = torch.randn(3, width).to(dtype)
    bias = torch.randn(3).to(dtype)
    indices = torch.tensor([4, 0, 1, 0], dtype=torch.int64)
    expected_cache = cache.clone()
    expected = torch.zeros_like(x)
    for row, block in enumerate(indices.tolist()):
        if block == 0:
            continue
        for channel in range(3):
            window = torch.cat((cache[block, channel], x[row, channel : channel + 1].to(dtype)))
            acc = bias[channel].float()
            for tap in range(width):
                # Multiply in FP64 then explicitly round to the operand dtype.
                product = (window[tap].double() * weight[channel, tap].double()).to(dtype)
                acc = acc + product.float()
            if activation is not None:
                acc = acc / (1 + torch.exp(-acc))
            expected[row, channel] = acc
            expected_cache[block, channel] = window[1:]
    state = cache if dim_first else cache.transpose(-1, -2).contiguous()
    before = state.clone()
    op = CausalConv1dUpdateOp()
    kwargs = dict(bias=bias, activation=activation, dim_first=dim_first)
    out, updated = op(x, state, weight, indices, **kwargs)
    assert torch.equal(out, expected)
    assert out.dtype == x.dtype and updated.dtype == dtype
    assert torch.equal(updated if dim_first else updated.transpose(-1, -2), expected_cache)
    assert torch.equal(state, before)
    out32, state32 = op.forward_fp32(x.to(torch.bfloat16), state, weight, indices, **kwargs)
    rounded, rounded_state = op.forward(x.to(torch.bfloat16), state, weight, indices, **kwargs)
    assert out32.dtype == torch.float32
    assert torch.equal(rounded, out32.bfloat16())
    assert torch.equal(rounded_state, state32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_conv_batch_invariance(dtype, activation, with_bias):
    # #469's golden-only regrouping check, using a small CPU channel fixture.
    torch.manual_seed(469)
    x = torch.randn(17, 8).bfloat16()
    state = torch.randn(19, 8, 3).to(dtype)
    weight = torch.randn(8, 4).bfloat16()
    bias = torch.randn(8).bfloat16() if with_bias else None
    indices = torch.randperm(17) + 1
    op = CausalConv1dUpdateOp()

    def run(rows):
        return op(x[rows], state, weight, indices[rows], bias=bias, activation=activation)

    full, cache = run(torch.arange(17))
    picks = [torch.tensor([r]) for r in range(17)]
    picks += [torch.arange(2), torch.arange(3, 10), torch.randperm(17)]
    for rows in picks:
        out, updated = run(rows)
        assert torch.equal(out, full[rows])
        assert torch.equal(updated[indices[rows]], cache[indices[rows]])


@pytest.mark.parametrize("kind", ["gdn", "conv"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "indices_shape"])
def test_invalid_shapes_and_dtypes(kind, invalid):
    if kind == "gdn":
        args = list(_inputs(torch.float32))
        op = GatedDeltaRuleRecurrentStepOp()
        kwargs = dict(scale=1.0, num_k_heads=2)
    else:
        args = [torch.ones(4, 3), torch.zeros(6, 3, 3), torch.ones(3, 4), torch.arange(1, 5)]
        op, kwargs = CausalConv1dUpdateOp(), {}
    if invalid == "shape":
        args[0] = args[0][:, :-1]
    elif invalid == "dtype":
        args[0] = args[0].int()
    else:
        args[-1] = args[-1].unsqueeze(0)
    with pytest.raises(ValueError):
        op(*args, **kwargs)


def test_softplus_threshold_and_conv_activation_validation():
    from rl_engine.reference.linear_attn.gated_delta_rule import _softplus

    x = torch.tensor([19.0, 20.0, 21.0, 100.0])
    expected = torch.tensor(
        [torch.log1p(torch.exp(value)) if value <= 20 else value for value in x]
    )
    assert torch.equal(_softplus(x), expected)
    with pytest.raises(ValueError, match="activation must be"):
        CausalConv1dUpdateOp()(
            torch.ones(1, 1),
            torch.zeros(2, 1, 1),
            torch.ones(1, 2),
            torch.tensor([1]),
            activation="relu",
        )


def test_import_and_execution_without_engine_or_extension(tmp_path):
    # Fail on attempted imports, even if an installed provider would otherwise
    # hide an accidental dependency. Execute both real references in that process.
    root = Path(__file__).resolve().parents[3]
    script = """
import importlib.abc
import sys
import torch

class BlockProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vllm', 'triton', 'transformers'} or fullname.startswith(
            ('rl_engine._C', 'rl_engine.models', 'rl_engine.backends', 'rl_engine.integrations')
        ):
            raise AssertionError('unexpected dependency: ' + fullname)

sys.meta_path.insert(0, BlockProviders())
sys.path.insert(0, sys.argv[1])
from rl_engine.reference.linear_attn import CausalConv1dUpdateOp, GatedDeltaRuleRecurrentStepOp
out, state = GatedDeltaRuleRecurrentStepOp()(
    torch.ones(1, 3), torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1),
    torch.zeros(1), torch.zeros(2, 1, 1, 1), torch.tensor([1]), scale=1., num_k_heads=1
)
assert out.count_nonzero() and state.count_nonzero()
out, state = CausalConv1dUpdateOp()(
    torch.ones(1, 1), torch.zeros(2, 1, 1), torch.ones(1, 2), torch.tensor([1])
)
assert out.count_nonzero() and state.count_nonzero()
from rl_engine.contracts.operators.gdn import GDNContract
from rl_engine.reference.linear_attn import gated_delta_rule_sequence
out, state = gated_delta_rule_sequence(
    torch.ones(2, 3), torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(1),
    torch.zeros(1), torch.zeros(1, 1, 1, 1), torch.tensor([0]), torch.tensor([0, 2]),
    contract=GDNContract(1, 1, 1, 1, scale=1.)
)
assert out.shape == (2, 1, 1) and out.count_nonzero() and state.count_nonzero()
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
