# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Which GDN components are batch-invariant, per component, measured.

vLLM refuses strict mode for this model at engine init:

    VLLM batch_invariant mode is not supported for GDN_ATTN.

The guard is a capability declaration -- `selector.py:235` raises when
`VLLM_BATCH_INVARIANT` is set and the mamba backend's
`supports_batch_invariance()` is False, which `gdn_attn` does not override. So the
question that decides whether full-model L2 is reachable is not "can the guard be
bypassed" (it must not be) but "are the components on the strict path actually
batch-invariant, and if not, which one is not".

Measured on B200 (sm_100), vLLM 0.30.0, per component:

=========================================  ======================================
component                                  batch invariance
=========================================  ======================================
`causal_conv1d_update`                     bitwise (8 seeds x 4 positions,
                                           fp32 and bf16 cache)
`fused_recurrent_..._packed_decode`        bitwise (12 seeds x 4 positions,
                                           fp32 and bf16 state)
FLA Triton `chunk_gated_delta_rule`        bitwise w.r.t. batch composition
FlashInfer GDN prefill                     **not** bitwise
=========================================  ======================================

FlashInfer is the *default* prefill backend on SM100 (`_resolve_gdn_prefill_backend`
returns "flashinfer" for compute capability 100 with `head_k_dim == 128` and CUDA
>= 13), so the default configuration is genuinely not batch-invariant and the
guard is correct rather than conservative. The deviation is rounding-scale, not
structural: adding a companion sequence moved 12149 of 262144 output elements,
`max|diff| = 2.44e-04` against a tensor of scale 7.18e-02 (3.4e-03 relative), and
left the final state within 5.7e-03 relative while almost no bit patterns matched
-- the signature of a pervasive small change in reduction order, consistent with
chunk tiling that depends on the batch's token count.

**The actionable consequence**: pinning `gdn_prefill_backend="triton"` makes every
component measured here bitwise batch-invariant. That is the configuration a
full-model L2 attempt should use. It does not by itself justify flipping
`supports_batch_invariance()`, which is a statement about the whole backend --
metadata construction, chunked-prefill splitting, CUDA graphs and the speculative
paths are outside this file's scope and remain unmeasured.

These assertions therefore state what is true (the three invariant components) and
bound the deviation of the fourth. They deliberately do NOT assert that FlashInfer
*differs*: that would be asserting a defect persists, and would fail the day
upstream fixes it, which is the opposite of what we want to learn.

Named ``check_`` rather than ``test_`` because it imports real vLLM; see
``tests/distributed/check_*.py``. Run via ``ci/scripts/run_qwen3_next_operator_gates.sh``
or explicitly:

    CUDA_VISIBLE_DEVICES=1 pytest tests/models/qwen3_next/check_qwen3_next_batch_invariance.py -v
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from rl_engine.validation.common.tensor_identity import tensor_bitwise_equal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# Qwen3-Next-80B-A3B-Instruct GDN dimensions.
_H, _HV, _K, _V = 16, 32, 128, 128
_SCALE = _K**-0.5
_CONV_DIM, _CONV_WIDTH = 8192, 4
_PACKED_DIM = _H * _K * 2 + _HV * _V

#: Positions within the batch whose target sequence is re-run alone.
_POSITIONS = (0, 1, 31, 63)
_BATCH = 64


def _randn(generator, *shape, dt=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dt, generator=generator)


def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    """``t / sqrt(sum(t*t) + 1e-6)`` over the last dim.

    q/k arrive L2-normalized in the real path: prefill runs
    ``fused_post_conv_prep(apply_l2norm=True)`` and then passes
    ``use_qk_l2norm_in_kernel=False``. Feeding raw randn with that flag off leaves
    them unnormalized and the delta-rule update diverges to NaN -- a property of
    the inputs, not of the kernel. The division is by the norm cast to ``t``'s
    dtype, which is what the measurements recorded in the execution doc used.
    """
    norm = torch.sqrt((t.float() * t.float()).sum(-1, keepdim=True) + 1e-6)
    return t / norm.to(t.dtype)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _decay(hv: int, tokens: int, g: torch.Generator) -> torch.Tensor:
    """``g = -exp(A_log) * softplus(a)``, which is always <= 0.

    Feeding a sign-free random tensor instead lets ``exp(g) > 1`` and the
    recurrence diverges to NaN, which looks like a kernel fault and is not one.
    """
    a_log = torch.randn(hv, device="cuda", dtype=torch.float32, generator=g)
    a = torch.randn(1, tokens, hv, device="cuda", dtype=torch.float32, generator=g)
    return -torch.exp(a_log) * torch.nn.functional.softplus(a)


# --------------------------------------------------------------------------- #
# 1. Causal conv1d state update -- bitwise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_causal_conv1d_update_is_batch_invariant(state_dtype):
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    for seed in range(8):
        g = _gen(seed)
        x = torch.randn(_BATCH, _CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
        # One cache line per sequence plus the null block.
        state = torch.randn(
            _BATCH + 2, _CONV_DIM, _CONV_WIDTH - 1, device="cuda", dtype=state_dtype, generator=g
        )
        weight = torch.randn(
            _CONV_DIM, _CONV_WIDTH, device="cuda", dtype=torch.bfloat16, generator=g
        )
        bias = torch.randn(_CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
        indices = torch.arange(1, _BATCH + 1, device="cuda", dtype=torch.int32)

        full_state, full_out = state.clone(), torch.empty_like(x)
        causal_conv1d_update(
            x, full_state, weight, bias, "silu", conv_state_indices=indices, out=full_out
        )

        for row in _POSITIONS:
            lone_state = state.clone()
            lone_out = torch.empty_like(x[row : row + 1])
            causal_conv1d_update(
                x[row : row + 1],
                lone_state,
                weight,
                bias,
                "silu",
                conv_state_indices=indices[row : row + 1].clone(),
                out=lone_out,
            )
            block = int(indices[row])
            assert tensor_bitwise_equal(lone_out[0], full_out[row]), f"out seed={seed} row={row}"
            assert tensor_bitwise_equal(lone_state[block], full_state[block]), f"state row={row}"


# --------------------------------------------------------------------------- #
# 2. Packed recurrent decode -- bitwise. This is the rollout decode provider.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_packed_recurrent_decode_is_batch_invariant(state_dtype):
    pytest.importorskip("vllm")
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode as decode,
    )

    def run(mixed, a, b, a_log, dt_bias, state, indices):
        working = state.clone()
        out = torch.empty(mixed.shape[0], 1, _HV, _V, device="cuda", dtype=mixed.dtype)
        decode(
            mixed,
            a,
            b,
            a_log,
            dt_bias,
            _SCALE,
            working,
            out,
            indices,
            use_qk_l2norm_in_kernel=True,
        )
        return out, working

    for seed in range(12):
        g = _gen(seed)
        mk = partial(_randn, g)
        mixed, a, b = mk(_BATCH, _PACKED_DIM), mk(_BATCH, _HV), mk(_BATCH, _HV)
        a_log, dt_bias = mk(_HV, dt=torch.float32), mk(_HV, dt=torch.float32)
        state = mk(_BATCH + 2, _HV, _V, _K, dt=state_dtype) * 0.1
        indices = torch.arange(1, _BATCH + 1, device="cuda", dtype=torch.int32)

        full_out, full_state = run(mixed, a, b, a_log, dt_bias, state, indices)
        for row in _POSITIONS:
            lone_out, lone_state = run(
                mixed[row : row + 1],
                a[row : row + 1],
                b[row : row + 1],
                a_log,
                dt_bias,
                state,
                indices[row : row + 1].clone(),
            )
            block = int(indices[row])
            assert tensor_bitwise_equal(lone_out[0], full_out[row]), f"out seed={seed} row={row}"
            assert tensor_bitwise_equal(lone_state[block], full_state[block]), f"state row={row}"


# --------------------------------------------------------------------------- #
# 3 & 4. Chunked prefill, per backend
# --------------------------------------------------------------------------- #
def _prefill_inputs(seqs, seed):
    """Inputs for ``seqs`` packed into one varlen batch.

    Each sequence draws from its own generator (``seed + i``), so the first
    sequence's tensors are identical whatever follows it. One generator shared
    across the batch would make them depend on the total token count: the CUDA
    RNG offset advances in rounded blocks, which hides that for short batches
    and stops hiding it for longer ones, silently comparing different data.
    """
    a_log = torch.randn(_HV, device="cuda", dtype=torch.float32, generator=_gen(seed + 10_000))
    parts = {name: [] for name in ("q", "k", "v", "g", "beta")}
    for index, tokens in enumerate(seqs):
        g = _gen(seed + index)
        mk = partial(_randn, g)
        parts["q"].append(_l2_normalize(mk(1, tokens, _HV, _K)))
        parts["k"].append(_l2_normalize(mk(1, tokens, _HV, _K)))
        parts["v"].append(mk(1, tokens, _HV, _V))
        a = torch.randn(1, tokens, _HV, device="cuda", dtype=torch.float32, generator=g)
        parts["g"].append(-torch.exp(a_log) * torch.nn.functional.softplus(a))
        parts["beta"].append(
            torch.rand(1, tokens, _HV, device="cuda", dtype=torch.float32, generator=g)
        )
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seqs), 0).tolist()], device="cuda", dtype=torch.int32
    )
    return {
        **{name: torch.cat(tensors, dim=1) for name, tensors in parts.items()},
        "cu_seqlens": cu,
        "initial_state": torch.zeros(len(seqs), _HV, _V, _K, device="cuda", dtype=torch.float32),
    }


#: The target sequence, then the companions appended after it. The long ones are
#: what a shared generator used to get wrong; see `_prefill_inputs`.
_COMPANIONS = ([], [64], [17, 95], [3], [1], [64, 64, 64, 64], [2048])
_TARGET = 64


def test_target_inputs_do_not_depend_on_companions():
    """Guards the comparisons below: they must compare the same target data."""
    alone = _prefill_inputs([_TARGET], seed=0)
    for companions in _COMPANIONS:
        batch = _prefill_inputs([_TARGET, *companions], seed=0)
        for name in ("q", "k", "v", "g", "beta"):
            assert tensor_bitwise_equal(batch[name][:, :_TARGET], alone[name]), (name, companions)


def test_triton_chunked_prefill_is_batch_invariant():
    """Supplementary native batch probe; shared full/chunk/decode uses packed recurrence."""
    pytest.importorskip("vllm")
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

    reference = None
    for companions in _COMPANIONS:
        inp = _prefill_inputs([_TARGET, *companions], seed=0)
        out, final_state = chunk_gated_delta_rule(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["beta"],
            scale=_SCALE,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=inp["cu_seqlens"],
            use_qk_l2norm_in_kernel=False,
        )
        target_out, target_state = out[0, :_TARGET], final_state[0]
        assert not torch.isnan(target_out).any(), f"NaN with companions={companions}"
        if reference is None:
            reference = (target_out.clone(), target_state.clone())
            continue
        assert tensor_bitwise_equal(target_out, reference[0]), f"out, companions={companions}"
        assert tensor_bitwise_equal(target_state, reference[1]), f"state, companions={companions}"


def test_flashinfer_chunked_prefill_deviation_stays_rounding_scale():
    """The DEFAULT prefill backend on SM100, which is not batch-invariant.

    Only the magnitude is asserted. Asserting that it *differs* would be asserting
    a defect persists, and would start failing the day upstream makes it
    invariant -- which is the outcome we want. A structural regression (a
    deviation well past rounding scale) still fails here.
    """
    pytest.importorskip("vllm")
    pytest.importorskip("flashinfer")
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fi_chunk_gated_delta_rule

    reference = None
    for companions in _COMPANIONS:
        inp = _prefill_inputs([_TARGET, *companions], seed=0)
        out, final_state = fi_chunk_gated_delta_rule(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["beta"],
            inp["initial_state"],
            True,
            cu_seqlens=inp["cu_seqlens"],
            use_qk_l2norm_in_kernel=True,
        )
        target_out, target_state = out[0, :_TARGET], final_state[0]
        assert not torch.isnan(target_out).any(), f"NaN with companions={companions}"
        if reference is None:
            reference = (target_out.clone(), target_state.clone())
            continue
        for name, got, want in (
            ("out", target_out, reference[0]),
            ("state", target_state, reference[1]),
        ):
            scale = max(want.float().abs().max().item(), 1e-9)
            relative = (got.float() - want.float()).abs().max().item() / scale
            assert relative <= 5e-2, (
                f"{name} deviation {relative:.3e} relative with companions="
                f"{companions} is past rounding scale; measured baseline was "
                "3.4e-03 (out) and 5.7e-03 (state)"
            )


# --------------------------------------------------------------------------- #
# 5. Metadata construction
#
# `GDNMetadataBuilder` derives the chunk partition from `cu_seqlens` via
# `prepare_chunk_indices` / `prepare_chunk_offsets`. Per-sequence chunk counts
# are `cdiv(own_length, FLA_CHUNK_SIZE)`, so a sequence's partition cannot depend
# on its companions -- only its sequence id and its offset into the flat chunk
# list shift. This is pure index arithmetic, so it is checked exactly, on CPU.
# --------------------------------------------------------------------------- #
def _chunk_metadata(seqs):
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    cu = torch.tensor([0, *torch.cumsum(torch.tensor(seqs), 0).tolist()], dtype=torch.int32)
    return (
        prepare_chunk_indices(cu, FLA_CHUNK_SIZE),
        prepare_chunk_offsets(cu, FLA_CHUNK_SIZE),
        FLA_CHUNK_SIZE,
    )


@pytest.mark.parametrize("position", [0, 1, 2])
def test_chunk_metadata_partition_is_batch_invariant(position):
    """A sequence's own chunk partition must not move when the batch changes.

    The sequence id and the flat offset legitimately shift with position; what
    must not change is how many chunks the sequence is cut into and which of its
    own tokens each chunk covers.
    """
    pytest.importorskip("vllm")
    target = 150  # deliberately not a multiple of the chunk size
    reference = None
    for others in ([], [64], [17, 95], [3], [1, 1, 1]):
        seqs = list(others)
        seqs.insert(min(position, len(seqs)), target)
        seq_id = min(position, len(others))

        indices, offsets, chunk_size = _chunk_metadata(seqs)
        mine = indices[indices[:, 0] == seq_id]
        # Chunk ids within the sequence, and the flat span it occupies.
        partition = (
            mine[:, 1].tolist(),
            int(offsets[seq_id + 1] - offsets[seq_id]),
        )
        expected_chunks = -(-target // chunk_size)  # ceil div
        assert partition[0] == list(range(expected_chunks)), f"others={others}"
        assert partition[1] == expected_chunks, f"offset span, others={others}"
        if reference is None:
            reference = partition
        else:
            assert partition == reference, f"others={others}"


# --------------------------------------------------------------------------- #
# 6. Chunked-prefill splitting
#
# vLLM may process one prompt over several forward passes, handing the recurrent
# state between them. Whether that equals a single pass is the prefill analogue
# of the prefill-vs-decode question, and it is what decides L2 for prompts longer
# than `max_num_batched_tokens`.
# --------------------------------------------------------------------------- #
def _run_triton_prefill(inp, tokens, initial_state):
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

    lo, hi = tokens
    cu = torch.tensor([0, hi - lo], device="cuda", dtype=torch.int32)
    return chunk_gated_delta_rule(
        inp["q"][:, lo:hi],
        inp["k"][:, lo:hi],
        inp["v"][:, lo:hi],
        inp["g"][:, lo:hi],
        inp["beta"][:, lo:hi],
        scale=_SCALE,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=False,
    )


@pytest.mark.parametrize("split", [128, 100, 64, 192])
def test_triton_chunked_prefill_split_matches_one_pass(split):
    """Splitting a prompt and handing over the state, versus one pass.

    `split=128` and `split=64` are multiples of FLA_CHUNK_SIZE; `split=100` and
    `split=192` are not, which is the realistic case because an engine-step
    boundary is set by the token budget, not by the kernel's chunk size.
    """
    pytest.importorskip("vllm")
    total = 256
    inp = _prefill_inputs([total], seed=0)

    zero = torch.zeros(1, _HV, _V, _K, device="cuda", dtype=torch.float32)
    whole_out, whole_state = _run_triton_prefill(inp, (0, total), zero.clone())
    first_out, first_state = _run_triton_prefill(inp, (0, split), zero.clone())
    second_out, second_state = _run_triton_prefill(inp, (split, total), first_state)

    joined = torch.cat([first_out[0], second_out[0]], dim=0)
    aligned = split % 64 == 0
    assert not torch.isnan(whole_out).any(), "degenerate inputs, not a kernel fault"

    out_rel = (
        (joined.float() - whole_out[0].float()).abs().max()
        / max(whole_out[0].float().abs().max().item(), 1e-9)
    ).item()
    state_rel = (
        (second_state.float() - whole_state.float()).abs().max()
        / max(whole_state.float().abs().max().item(), 1e-9)
    ).item()

    if aligned:
        # A chunk-aligned split must be bitwise: the kernel's own chunk
        # boundaries already fall there, so no reduction order changes.
        assert tensor_bitwise_equal(joined, whole_out[0]), f"split={split} out, rel={out_rel:.3e}"
        assert tensor_bitwise_equal(second_state, whole_state), f"split={split} state"
    else:
        # An unaligned split re-cuts the sequence, so the reduction order moves.
        # Bound it; do not assert equality, and do not assert inequality either.
        assert out_rel <= 5e-2, f"split={split} out deviation {out_rel:.3e}"
        assert state_rel <= 5e-2, f"split={split} state deviation {state_rel:.3e}"
