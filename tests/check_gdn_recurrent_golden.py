# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The GDN recurrent-step golden against the kernel vLLM actually runs (RFC #428 C6).

Named ``check_`` rather than ``test_``, following ``tests/distributed/check_*.py``:
this module imports real vLLM, and ``tests/test_framework_operator_integrations.py``
asserts ``vllm`` is absent from ``sys.modules`` -- an invariant any collected test
importing vLLM would break for the whole session. Run it explicitly:

    pytest tests/check_gdn_recurrent_golden.py -v

The provider here is ``fused_recurrent_gated_delta_rule_packed_decode``, not
``fused_sigmoid_gating_delta_rule_update``. A pure RL rollout decode -- N
sequences each emitting one token, no speculative decoding -- returns early into
the packed path because ``VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`` defaults to
true. Aligning against the sigmoid-gating kernel would be validating a path
production does not take.

Claim levels (RFC #428 section 2.1): L0 and L1 for the golden. L2 is not claimed;
the golden agrees with the provider at fp32-ULP scale but not bitwise, because
its contractions run in a fixed 32-wide chunk order rather than the kernel's
tree.

The fixed-seed 128-step rounding test below is a bounded synthetic regression,
not evidence of a universal drift plateau or model-logit agreement. Earlier
1024-step and prefill tables had no reproducible runner and are withdrawn until
those experiments are checked in. Operator outputs are not model logits.

"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

from rl_engine.kernels.ops.pytorch.linear_attn import GatedDeltaRuleRecurrentStepOp  # noqa: E402

# Qwen3-Next-80B-A3B-Instruct: linear_num_key_heads / linear_num_value_heads /
# linear_key_head_dim / linear_value_head_dim.
_H, _HV, _K, _V = 16, 32, 128, 128
_SCALE = _K**-0.5
_PACKED_DIM = _H * _K * 2 + _HV * _V


def _vllm_step():
    pytest.importorskip("vllm", reason="vLLM is required to compare against the provider")
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    return fused_recurrent_gated_delta_rule_packed_decode


def _inputs(batch, num_blocks, state_dtype, io_dtype, seed, indices=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rand = lambda *shape, dtype: torch.randn(  # noqa: E731
        *shape, device="cuda", dtype=dtype, generator=g
    )
    if indices is None:
        indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    return {
        "mixed_qkv": rand(batch, _PACKED_DIM, dtype=io_dtype),
        "a": rand(batch, _HV, dtype=io_dtype),
        "b": rand(batch, _HV, dtype=io_dtype),
        "A_log": rand(_HV, dtype=torch.float32),
        "dt_bias": rand(_HV, dtype=torch.float32),
        "state": rand(num_blocks, _HV, _V, _K, dtype=state_dtype) * 0.1,
        "indices": indices,
    }


def _run_provider(inp, io_dtype=torch.bfloat16):
    """Returns (out, mutated_state). The kernel updates the state in place."""
    step = _vllm_step()
    state = inp["state"].clone()
    out = torch.empty(inp["mixed_qkv"].shape[0], 1, _HV, _V, device="cuda", dtype=io_dtype)
    step(
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        _SCALE,
        state,
        out,
        inp["indices"],
        use_qk_l2norm_in_kernel=True,
    )
    return out, state


def _run_golden(inp):
    return GatedDeltaRuleRecurrentStepOp().forward(
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        inp["state"].clone(),
        inp["indices"],
        scale=_SCALE,
        num_k_heads=_H,
    )


# --------------------------------------------------------------------------- #
# 1. Against the provider
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 4, 17, 64])
@pytest.mark.parametrize(
    "state_dtype, out_atol, state_atol",
    [
        # fp32 state: agreement is at fp32-ULP scale.
        (torch.float32, 1e-3, 1e-5),
        # bf16 state: the store rounds every token, so the state carries a
        # bf16 ULP. This is the configuration knob behind the drift probe.
        (torch.bfloat16, 1e-3, 5e-3),
    ],
)
def test_golden_matches_packed_decode_provider(batch, state_dtype, out_atol, state_atol):
    inp = _inputs(batch, batch + 2, state_dtype, torch.bfloat16, seed=batch)
    out_ref, state_ref = _run_provider(inp)
    out_got, state_got = _run_golden(inp)

    assert (out_got.float() - out_ref.float()).abs().max().item() <= out_atol
    assert (state_got.float() - state_ref.float()).abs().max().item() <= state_atol


def test_null_block_id_is_skipped_by_both():
    """Index 0 means "no state": zeros out, and the block is left alone."""
    indices = torch.tensor([1, 0, 2, 0], device="cuda", dtype=torch.int32)
    inp = _inputs(4, 4, torch.float32, torch.bfloat16, seed=7, indices=indices)

    out_ref, state_ref = _run_provider(inp)
    out_got, state_got = _run_golden(inp)

    for row in (1, 3):
        assert bool((out_ref[row] == 0).all()), f"provider row {row}"
        assert bool((out_got[row] == 0).all()), f"golden row {row}"
    assert torch.equal(state_got[0], inp["state"][0]), "block 0 must be untouched"


# --------------------------------------------------------------------------- #
# 2. L1 -- a sequence is unaffected by the others sharing the batch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_golden_is_batch_invariant(state_dtype):
    """Bitwise: the same sequence, alone or in a batch of 64, at any position."""
    batch = 64
    inp = _inputs(batch, batch + 2, state_dtype, torch.bfloat16, seed=3)
    full_out, full_state = _run_golden(inp)

    op = GatedDeltaRuleRecurrentStepOp()
    for row in (0, 1, 31, 63):
        alone = {
            key: (value[row : row + 1] if key in ("mixed_qkv", "a", "b", "indices") else value)
            for key, value in inp.items()
        }
        out, state = op.forward(
            alone["mixed_qkv"],
            alone["a"],
            alone["b"],
            alone["A_log"],
            alone["dt_bias"],
            alone["state"].clone(),
            alone["indices"],
            scale=_SCALE,
            num_k_heads=_H,
        )
        assert torch.equal(out[0], full_out[row]), f"out row {row}"
        block = int(inp["indices"][row])
        assert torch.equal(state[block], full_state[block]), f"state block {block}"


# --------------------------------------------------------------------------- #
# 3. The state-dtype rounding is modelled, not skipped
# --------------------------------------------------------------------------- #
def test_bf16_state_rounding_stays_within_fixed_fixture_bound():
    """Bound this fixed 128-step fixture; no general contraction claim."""
    batch, steps = 8, 128
    inp = _inputs(batch, batch + 2, torch.float32, torch.bfloat16, seed=11)
    op = GatedDeltaRuleRecurrentStepOp()

    # Both runs start from the SAME bf16-representable state, so the initial
    # cast is a no-op and drift[0] can only come from a per-token store.
    seed_state = inp["state"].to(torch.bfloat16).float()
    state_fp32 = seed_state.clone()
    state_bf16 = seed_state.to(torch.bfloat16).clone()
    assert torch.equal(state_fp32, state_bf16.float()), "the two runs must start equal"
    drift = []
    for step in range(steps):
        g = torch.Generator(device="cuda").manual_seed(100 + step)
        token = torch.randn(batch, _PACKED_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
        common = (inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])
        _, state_fp32 = op.forward(
            token, *common, state_fp32, inp["indices"], scale=_SCALE, num_k_heads=_H
        )
        _, state_bf16 = op.forward(
            token, *common, state_bf16, inp["indices"], scale=_SCALE, num_k_heads=_H
        )
        scale = max(state_fp32.float().abs().max().item(), 1e-9)
        drift.append((state_fp32.float() - state_bf16.float()).abs().max().item() / scale)

    assert drift[0] > 0.0, "a bf16 state must round on the very first store"
    # Preserve the original regression bound for this fixed fixture.
    assert max(drift) < 0.05, f"relative state drift reached {max(drift):.3e}"
    # The back half must not be materially worse than the front half.
    assert max(drift[steps // 2 :]) < 2.0 * max(drift[: steps // 2]) + 1e-3


# --------------------------------------------------------------------------- #
# 4. The causal-conv1d state update, the other half of a decode step
# --------------------------------------------------------------------------- #
_CONV_DIM, _CONV_WIDTH = 8192, 4  # Qwen3-Next: key_dim*2 + value_dim, linear_conv_kernel_dim


def _conv_update():
    pytest.importorskip("vllm", reason="vLLM is required to compare against the provider")
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    return causal_conv1d_update


def _conv_inputs(batch, state_dtype, seed, with_bias=True, indices=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rand = lambda *shape, dtype: torch.randn(  # noqa: E731
        *shape, device="cuda", dtype=dtype, generator=g
    )
    if indices is None:
        indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    return {
        "x": rand(batch, _CONV_DIM, dtype=torch.bfloat16),
        # The cache must have room for every index used, plus the null block.
        "state": rand(batch + 2, _CONV_DIM, _CONV_WIDTH - 1, dtype=state_dtype),
        "weight": rand(_CONV_DIM, _CONV_WIDTH, dtype=torch.bfloat16),
        "bias": rand(_CONV_DIM, dtype=torch.bfloat16) if with_bias else None,
        "indices": indices,
    }


def _run_conv_pair(inp, activation="silu"):
    from rl_engine.kernels.ops.pytorch.linear_attn import CausalConv1dUpdateOp

    state_ref, out_ref = inp["state"].clone(), torch.empty_like(inp["x"])
    _conv_update()(
        inp["x"],
        state_ref,
        inp["weight"],
        inp["bias"],
        activation,
        conv_state_indices=inp["indices"],
        out=out_ref,
    )
    out_got, state_got = CausalConv1dUpdateOp().forward(
        inp["x"],
        inp["state"].clone(),
        inp["weight"],
        inp["indices"],
        bias=inp["bias"],
        activation=activation,
    )
    return (out_ref, state_ref), (out_got, state_got)


@pytest.mark.parametrize("batch", [1, 4, 17, 64])
def test_conv_state_update_is_bitwise_exact(batch):
    """The rolled window is what the next token consumes, so it must be exact.

    Holds for an fp32 and a bf16 cache alike -- the rolling is a copy, not a
    computation, which is also why bias and activation are not varied here: they
    cannot reach the state.
    """
    for state_dtype in (torch.float32, torch.bfloat16):
        inp = _conv_inputs(batch, state_dtype, seed=batch)
        (_, state_ref), (_, state_got) = _run_conv_pair(inp)
        assert torch.equal(state_got, state_ref), f"{state_dtype} batch={batch}"


@pytest.mark.parametrize("batch", [1, 4, 17, 64])
def test_conv_output_matches_provider_with_fp32_cache(batch):
    """An fp32 cache reproduces the provider up to fp32 ULP on a few elements.

    Measured: 1 element of 8192 at B=1, 15 of 524288 at B=64. The bound is on
    the magnitude and on a handful of elements rather than on a tight rate --
    at small batches a single straddling element is already 1.2e-4 of the
    tensor, which says nothing about accuracy.
    """
    inp = _conv_inputs(batch, torch.float32, seed=batch)
    (out_ref, _), (out_got, _) = _run_conv_pair(inp)
    mismatch = int((out_got.float().view(torch.int32) != out_ref.float().view(torch.int32)).sum())
    assert mismatch <= 32, f"{mismatch} of {out_got.numel()} elements differ"
    assert (out_got.float() - out_ref.float()).abs().max().item() <= 1e-2


@pytest.mark.parametrize("batch", [1, 17])
def test_conv_output_bf16_cache_matches_rounded_product_path(batch):
    """Product rounding is reproduced; activation ULP residuals remain allowed."""
    inp = _conv_inputs(batch, torch.bfloat16, seed=batch)
    (out_ref, _), (out_got, _) = _run_conv_pair(inp)
    assert (out_got.float() - out_ref.float()).abs().max().item() <= 7e-2
    assert int((out_got != out_ref).sum()) <= 32


def test_conv_null_block_id_is_skipped():
    from rl_engine.kernels.ops.pytorch.linear_attn import CausalConv1dUpdateOp

    indices = torch.tensor([1, 0, 2, 0], device="cuda", dtype=torch.int32)
    inp = _conv_inputs(4, torch.float32, seed=5, with_bias=False, indices=indices)
    out_got, state_got = CausalConv1dUpdateOp().forward(
        inp["x"], inp["state"].clone(), inp["weight"], indices, bias=None
    )
    for row in (1, 3):
        assert bool((out_got[row] == 0).all())
    assert torch.equal(state_got[0], inp["state"][0])


def test_conv_rejects_unknown_activation_and_bad_shapes():
    from rl_engine.kernels.ops.pytorch.linear_attn import CausalConv1dUpdateOp

    op = CausalConv1dUpdateOp()
    inp = _conv_inputs(2, torch.float32, seed=1)
    with pytest.raises(ValueError, match="activation must be"):
        op.forward(inp["x"], inp["state"], inp["weight"], inp["indices"], activation="relu")
    with pytest.raises(ValueError, match="conv_state tail must be"):
        op.forward(
            inp["x"],
            inp["state"][..., :1],
            inp["weight"],
            inp["indices"],
            activation=None,
        )


# --------------------------------------------------------------------------- #
# 5. forward_fp32, and the branches the provider comparison never reaches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_recurrent_forward_fp32_widens_the_state(state_dtype):
    """forward_fp32 must run the recurrence without the per-token rounding.

    With a bf16 cache that means widening it -- the whole point of having the
    method is to be able to compare against the rounded path.
    """
    op = GatedDeltaRuleRecurrentStepOp()
    inp = _inputs(4, 6, state_dtype, torch.bfloat16, seed=1)
    out, state = op.forward_fp32(
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        inp["state"].clone(),
        inp["indices"],
        scale=_SCALE,
        num_k_heads=_H,
    )
    assert out.dtype is torch.float32 and state.dtype is torch.float32

    rounded, _ = _run_golden(inp)
    if state_dtype is torch.bfloat16:
        # The rounded path went through bf16; the fp32 path did not.
        assert not torch.equal(rounded.float(), out)


def test_recurrent_call_matches_forward():
    op = GatedDeltaRuleRecurrentStepOp()
    inp = _inputs(4, 6, torch.float32, torch.bfloat16, seed=2)
    args = (
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        inp["state"].clone(),
        inp["indices"],
    )
    kwargs = dict(scale=_SCALE, num_k_heads=_H)
    a_out, a_state = op(*args, **kwargs)
    b_out, b_state = op.forward(*args, **kwargs)
    assert torch.equal(a_out, b_out) and torch.equal(a_state, b_state)


def test_recurrent_without_in_kernel_l2norm_differs():
    """`use_qk_l2norm=False` is the prefill convention and must be reachable."""
    op = GatedDeltaRuleRecurrentStepOp()
    inp = _inputs(4, 6, torch.float32, torch.bfloat16, seed=3)
    common = (
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
    )
    normed, _ = op.forward(
        *common, inp["state"].clone(), inp["indices"], scale=_SCALE, num_k_heads=_H
    )
    raw, _ = op.forward(
        *common,
        inp["state"].clone(),
        inp["indices"],
        scale=_SCALE,
        num_k_heads=_H,
        use_qk_l2norm=False,
    )
    assert not torch.equal(normed, raw)


def test_conv_dim_first_false_matches_the_transposed_layout():
    """`dim_first=False` is vLLM's "SD" cache layout, not a dead branch."""
    from rl_engine.kernels.ops.pytorch.linear_attn import CausalConv1dUpdateOp

    op = CausalConv1dUpdateOp()
    inp = _conv_inputs(4, torch.float32, seed=9)
    ds_out, ds_state = op.forward(
        inp["x"], inp["state"].clone(), inp["weight"], inp["indices"], bias=inp["bias"]
    )
    sd_out, sd_state = op.forward(
        inp["x"],
        inp["state"].transpose(-1, -2).contiguous(),
        inp["weight"],
        inp["indices"],
        bias=inp["bias"],
        dim_first=False,
    )
    assert torch.equal(ds_out, sd_out)
    assert torch.equal(ds_state, sd_state.transpose(-1, -2))


def test_conv_forward_fp32_and_call_entry_points():
    from rl_engine.kernels.ops.pytorch.linear_attn import CausalConv1dUpdateOp

    op = CausalConv1dUpdateOp()
    inp = _conv_inputs(4, torch.float32, seed=10)
    args = (inp["x"], inp["state"].clone(), inp["weight"], inp["indices"])
    out, _ = op.forward(*args, bias=inp["bias"])
    called, _ = op(*args, bias=inp["bias"])
    fp32, _ = op.forward_fp32(*args, bias=inp["bias"])
    assert torch.equal(out, called)
    assert fp32.dtype is torch.float32
    torch.testing.assert_close(fp32, out.float(), atol=1e-2, rtol=1e-2)


def test_conv_provider_preserves_bf16_product_cancellation():
    inp = _conv_inputs(1, torch.bfloat16, seed=1, with_bias=False)
    inp["state"].zero_()
    inp["state"][1, :, -1] = 1.0078125
    inp["weight"].zero_()
    inp["weight"][:, -2] = 1.0078125
    inp["weight"][:, -1] = 1.0
    inp["x"].fill_(-1.015625)
    (provider, _), (golden, _) = _run_conv_pair(inp, activation=None)
    assert torch.count_nonzero(provider) == 0
    assert torch.equal(provider, golden)
