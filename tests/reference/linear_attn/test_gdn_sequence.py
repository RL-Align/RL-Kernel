# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Task 20 CPU sequence/state behavior through the public reference boundary."""

from dataclasses import replace
from math import sqrt

import pytest
import torch

from rl_engine.contracts.operators.gdn import GDNContract
from rl_engine.reference.linear_attn import gated_delta_rule_sequence


def test_hand_worked_recurrence_updates_slot_zero_without_mutating_state():
    contract = GDNContract(1, 1, 1, 1, scale=0.25, use_qk_l2norm=False)
    state = torch.tensor([2.0, 7.0]).reshape(2, 1, 1, 1)
    before = state.clone()
    out, updated = gated_delta_rule_sequence(
        torch.tensor([[2.0, 0.5, 3.0]]),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1),
        torch.zeros(1),
        state,
        torch.tensor([0]),
        torch.tensor([0, 1]),
        contract=contract,
    )
    # decay=1/2, beta=1/2: H'=1+(3-1*0.5)*0.5*0.5=1.625.
    assert torch.equal(out, torch.tensor([[[0.8125]]]))
    assert torch.equal(updated.flatten(), torch.tensor([1.625, 7.0]))
    assert torch.equal(state, before)
    assert updated.data_ptr() != state.data_ptr()


def _case(h=2, hv=4, k=3, v=2, dtype=torch.float32, state_dtype=torch.float32):
    """Build seeded packed inputs with sequence lengths 3/2 and state slots 0/2."""
    generator = torch.Generator().manual_seed(642)
    contract = GDNContract(h, hv, k, v, scale=k**-0.5)

    def rand(*shape):
        return (torch.randn(*shape, generator=generator) * 0.2).to(dtype)

    return contract, [
        rand(5, 2 * h * k + hv * v),
        rand(5, hv),
        rand(5, hv),
        rand(hv).float(),
        rand(hv).float(),
        rand(4, hv, v, k).to(state_dtype),
        torch.tensor([0, 2]),
        torch.tensor([0, 3, 5], dtype=torch.int32),
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_k_heads", 0),
        ("num_k_heads", True),
        ("num_v_heads", 3),
        ("key_dim", -1),
        ("value_dim", 1.5),
        ("scale", float("nan")),
        ("scale", True),
        ("use_qk_l2norm", 1),
        ("profile", "miles"),
    ],
)
def test_contract_rejects_invalid_geometry_or_unknown_profile(field, value):
    kwargs = dict(num_k_heads=2, num_v_heads=4, key_dim=3, value_dim=2, scale=0.5)
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        GDNContract(**kwargs)


@pytest.mark.parametrize(
    "index,replacement,message",
    [
        (0, torch.zeros(5, 19), "mixed_qkv"),
        (0, torch.zeros(5, 20, dtype=torch.float64), "mixed_qkv"),
        (1, torch.zeros(5, 3), "a"),
        (2, torch.zeros(5, 4, dtype=torch.bfloat16), "b"),
        (3, torch.zeros(4, dtype=torch.bfloat16), "A_log"),
        (4, torch.zeros(3), "dt_bias"),
        (5, torch.zeros(4, 4, 3, 2), "state"),
        (5, torch.zeros(4, 4, 2, 3, dtype=torch.float16), "state"),
        (6, torch.tensor([0, 0]), "indices"),
        (6, torch.tensor([-1, 2]), "indices"),
        (6, torch.tensor([0, 4]), "indices"),
        (6, torch.tensor([0.0, 2.0]), "indices"),
        (6, torch.tensor([[0, 2]]), "indices"),
        (7, torch.tensor([0, 4, 3, 5]), "cu_seqlens"),
        (7, torch.tensor([1, 3, 5]), "cu_seqlens"),
        (7, torch.tensor([0, 3, 4]), "cu_seqlens"),
        (7, torch.tensor([0.0, 3.0, 5.0]), "cu_seqlens"),
        (7, torch.tensor([], dtype=torch.int64), "cu_seqlens"),
    ],
)
def test_sequence_validates_complete_invocation_before_execution(index, replacement, message):
    contract, args = _case()
    args[index] = replacement
    with pytest.raises(ValueError, match=message):
        gated_delta_rule_sequence(*args, contract=contract)


def _bits_equal(left, right):
    """Compare raw storage bits, including signed zeros and NaN payloads."""
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def test_logical_layout_matches_strided_qkv_and_v_major_view_of_k_major_storage():
    contract, args = _case(k=3, v=5)
    before = [tensor.clone() for tensor in args]
    qkv_storage = args[0].new_full((args[0].shape[0], 2 * args[0].shape[1]), 123)
    strided_qkv = qkv_storage[:, ::2]
    strided_qkv.copy_(args[0])
    k_major_state = args[5].transpose(-1, -2).contiguous()
    logical_state = k_major_state.transpose(-1, -2)
    assert not strided_qkv.is_contiguous() and not logical_state.is_contiguous()
    strided_args = [strided_qkv, *args[1:5], logical_state, *args[6:]]
    for contiguous, strided in zip(args, strided_args, strict=True):
        _bits_equal(contiguous, strided)

    expected = gated_delta_rule_sequence(*args, contract=contract)
    actual = gated_delta_rule_sequence(*strided_args, contract=contract)
    for contiguous, strided in zip(expected, actual, strict=True):
        _bits_equal(contiguous, strided)
    for contiguous, strided, original in zip(args, strided_args, before, strict=True):
        _bits_equal(contiguous, original)
        _bits_equal(strided, original)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_state_bank_capacity_and_slot_do_not_change_active_recurrence(state_dtype):
    contract, args = _case(dtype=torch.bfloat16, state_dtype=state_dtype)
    minimal = args[5][:1].clone()
    larger = torch.full((100, *minimal.shape[1:]), 0.25, dtype=state_dtype)
    active_slot = 37
    larger[active_slot] = minimal[0]
    before = [minimal.clone(), larger.clone()]
    offsets = torch.tensor([0, args[0].shape[0]], dtype=torch.int32)

    small_out, small_state = gated_delta_rule_sequence(
        *args[:5], minimal, torch.tensor([0]), offsets, contract=contract
    )
    large_out, large_state = gated_delta_rule_sequence(
        *args[:5], larger, torch.tensor([active_slot]), offsets, contract=contract
    )
    _bits_equal(small_out, large_out)
    _bits_equal(small_state[0], large_state[active_slot])
    inactive = torch.arange(100) != active_slot
    _bits_equal(large_state[inactive], before[1][inactive])
    for updated, initial, original in zip(
        (small_state, large_state), (minimal, larger), before, strict=True
    ):
        assert updated.shape == initial.shape
        _bits_equal(initial, original)
        assert updated.data_ptr() != initial.data_ptr()
        updated.fill_(123)
        _bits_equal(initial, original)


@pytest.mark.parametrize("cuts", [[0, 5], [0, 0, 2, 2, 5], [0, 1, 2, 3, 4, 5]])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_temporal_chunks_preserve_outputs_complete_bank_and_graph(cuts, state_dtype):
    contract, args = _case(state_dtype=state_dtype)
    args[6:] = [torch.tensor([0]), torch.tensor([0, 5])]
    args[5].requires_grad_()
    before = args[5].detach().clone()
    full, final = gated_delta_rule_sequence(*args, contract=contract)
    state, outputs = args[5], []
    for start, end in zip(cuts, cuts[1:], strict=False):
        chunk = [x[start:end] for x in args[:3]] + args[3:5]
        out, state = gated_delta_rule_sequence(
            *chunk, state, args[6], torch.tensor([0, end - start]), contract=contract
        )
        outputs.append(out)
    _bits_equal(torch.cat(outputs), full)
    _bits_equal(state, final)
    _bits_equal(args[5], before)
    assert state.data_ptr() != args[5].data_ptr()
    grad = torch.autograd.grad(outputs[-1].float().sum(), args[5])[0]
    assert torch.isfinite(grad).all() and grad[0].count_nonzero()
    assert grad[1:].count_nonzero() == 0


@pytest.mark.parametrize("offsets,slots", [([0], []), ([0, 0, 0], [0, 2])])
def test_empty_batch_or_sequences_return_independent_unchanged_state(offsets, slots):
    contract, args = _case()
    args[:3] = [x[:0] for x in args[:3]]
    args[6:] = [torch.tensor(slots, dtype=torch.int64), torch.tensor(offsets)]
    args[5].requires_grad_()
    output, state = gated_delta_rule_sequence(*args, contract=contract)
    assert output.shape == (0, 4, 2) and output.dtype == torch.float32
    _bits_equal(state, args[5])
    assert state.data_ptr() != args[5].data_ptr()
    assert torch.equal(torch.autograd.grad(state.sum(), args[5])[0], torch.ones_like(state))
    state.detach().fill_(123)
    assert not torch.equal(state, args[5])


def test_packed_unequal_empty_permutation_isolation_and_regrouping():
    contract, args = _case(dtype=torch.bfloat16, state_dtype=torch.bfloat16)
    args[6:] = [torch.tensor([0, 1, 3]), torch.tensor([0, 3, 3, 5])]
    full, final = gated_delta_rule_sequence(*args, contract=contract)
    # Includes an empty sequence and moves the shorter one before the longer one.
    rows = torch.tensor([3, 4, 0, 1, 2])
    reordered = [x[rows] for x in args[:3]] + args[3:6]
    out, state = gated_delta_rule_sequence(
        *reordered, torch.tensor([3, 0, 1]), torch.tensor([0, 2, 5, 5]), contract=contract
    )
    _bits_equal(out, full[rows])
    _bits_equal(state, final)
    _bits_equal(final[1:3], args[5][1:3])
    for seq, (start, end) in enumerate([(0, 3), (3, 3), (3, 5)]):
        alone = [x[start:end] for x in args[:3]] + args[3:6]
        result, bank = gated_delta_rule_sequence(
            *alone, args[6][seq : seq + 1], torch.tensor([0, end - start]), contract=contract
        )
        _bits_equal(result, full[start:end])
        _bits_equal(bank[args[6][seq]], final[args[6][seq]])
    changed = [x.clone() for x in args]
    changed[0][:3] += 10
    changed_out, changed_state = gated_delta_rule_sequence(*changed, contract=contract)
    _bits_equal(changed_out[3:], full[3:])
    _bits_equal(changed_state[1:], final[1:])
    with torch.no_grad():
        no_grad = gated_delta_rule_sequence(*args, contract=contract)
    _bits_equal(no_grad[0], full)
    _bits_equal(no_grad[1], final)


def _oracle(args, contract):
    """Evaluate an independent FP64 per-head recurrence without step helpers.

    Round state to its storage dtype after each token; read output before that
    store. This checks mathematical accuracy, not the FP32 reduction schedule.
    """
    qkv, a, b, A_log, bias, initial, indices, offsets = args
    bank = list(initial.double().unbind())
    outputs = []
    h, hv, k, v = (contract.num_k_heads, contract.num_v_heads, contract.key_dim, contract.value_dim)
    for seq, (start, end) in enumerate(zip(offsets.tolist(), offsets.tolist()[1:], strict=False)):
        slot = indices[seq].item()
        for token in range(start, end):
            head_outputs, head_states = [], []
            for head in range(hv):
                key_head = head // (hv // h)
                q = qkv[token, key_head * k : (key_head + 1) * k].double()
                key = qkv[token, (h + key_head) * k : (h + key_head + 1) * k].double()
                value = qkv[token, 2 * h * k + head * v : 2 * h * k + (head + 1) * v].double()
                if contract.use_qk_l2norm:
                    q = q / (torch.linalg.vector_norm(q).square() + 1e-6).sqrt()
                    key = key / (torch.linalg.vector_norm(key).square() + 1e-6).sqrt()
                x = a[token, head].double() + bias[head].double()
                softplus = x if x.item() > 20 else torch.logaddexp(x, x.new_zeros(()))
                decay = (-A_log[head].double().exp() * softplus).exp()
                old = bank[slot][head] * decay
                residual = (value - old @ key) / (1 + (-b[token, head].double()).exp())
                updated = old + torch.outer(residual, key)
                head_outputs.append(updated @ (q * contract.scale))
                head_states.append(updated.to(initial.dtype).double())
            outputs.append(torch.stack(head_outputs).to(qkv.dtype))
            bank[slot] = torch.stack(head_states)
    return torch.stack(outputs), torch.stack(bank).to(initial.dtype)


@pytest.mark.parametrize("geometry", [(1, 1, 1, 2), (2, 4, 7, 3), (2, 6, 33, 5)])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_sequence_matches_independent_mathematical_oracle(geometry, normalize, dtype, state_dtype):
    contract, args = _case(*geometry, dtype=dtype, state_dtype=state_dtype)
    contract = replace(contract, use_qk_l2norm=normalize)
    output, state = gated_delta_rule_sequence(*args, contract=contract)
    expected, expected_state = _oracle(args, contract)
    # FP32 matrix vs padded reduction: small fixtures permit tight error.
    # Low precision tolerates one storage ULP near rounding boundaries.
    atol, rtol = {
        torch.float32: (2e-7, 2e-5),
        torch.float16: (2e-5, 2e-3),
        torch.bfloat16: (2e-4, 1e-2),
    }[dtype]
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
    satol, srtol = (2e-3, 1e-2) if state_dtype == torch.bfloat16 else (3e-7, 2e-5)
    torch.testing.assert_close(state, expected_state, atol=satol, rtol=srtol)
    assert output.dtype == dtype and state.dtype == state_dtype


def test_vllm_profile_stores_state_each_token_and_reads_output_before_store():
    contract = GDNContract(1, 1, 1, 1, scale=1, use_qk_l2norm=False)
    output, state = gated_delta_rule_sequence(
        torch.tensor([[1, 0.125, 0.296875]]).repeat(2, 1),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.zeros(1),
        torch.zeros(1),
        torch.ones(1, 1, 1, 1, dtype=torch.bfloat16),
        torch.tensor([0]),
        torch.tensor([0, 2]),
        contract=contract,
    )
    # Dyadic inputs, decay=beta=1/2. The second step reads rounded 0.515625.
    _bits_equal(output.flatten(), torch.tensor([0.5146484375, 0.27435302734375]))
    _bits_equal(state.flatten(), torch.tensor([0.2734375], dtype=torch.bfloat16))


def test_vllm_profile_computes_beta_in_fp32_even_with_bf16_operands():
    contract = GDNContract(1, 1, 1, 1, scale=1, use_qk_l2norm=False)
    output, state = gated_delta_rule_sequence(
        torch.ones(1, 3, dtype=torch.bfloat16),
        torch.zeros(1, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, dtype=torch.bfloat16),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, 1, 1, 1),
        torch.tensor([0]),
        torch.tensor([0, 1]),
        contract=contract,
    )
    # sigmoid(1) rounded once to FP32, not first rounded to BF16 by miles' gate.
    _bits_equal(state.flatten(), torch.tensor([0.7310585975646973]))
    _bits_equal(output.flatten(), torch.tensor([0.73046875], dtype=torch.bfloat16))


def test_profile_gating_extremes_keep_gradients_finite_through_public_entry():
    contract = GDNContract(1, 2, 1, 1, scale=1, use_qk_l2norm=False)
    qkv = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    a = torch.tensor([[100.0, -100.0]], requires_grad=True)
    b = torch.tensor([[100.0, -100.0]], requires_grad=True)
    args = [
        qkv,
        a,
        b,
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(1, 2, 1, 1),
        torch.tensor([0]),
        torch.tensor([0, 1]),
    ]
    output, state = gated_delta_rule_sequence(*args, contract=contract)
    torch.testing.assert_close(output.flatten(), torch.tensor([2.0, 1.0]), atol=1e-6, rtol=0)
    grads = torch.autograd.grad(output.sum() + state.sum(), (a, b))
    assert all(torch.isfinite(grad).all() for grad in grads)


def _packed_two_chunks(args, contract):
    """Split lengths [3, 0, 2] into [2, 0, 1] + [1, 0, 1], restoring token order."""
    state, outputs = args[5], []
    for rows, offsets in [([0, 1, 3], [0, 2, 2, 3]), ([2, 4], [0, 1, 1, 2])]:
        chunk = [x[rows] for x in args[:3]] + args[3:5]
        output, state = gated_delta_rule_sequence(
            *chunk, state, args[6], torch.tensor(offsets), contract=contract
        )
        outputs.append(output)
    return torch.cat(outputs)[torch.tensor([0, 1, 3, 2, 4])], state


def test_packed_chunk_gradients_match_independent_oracle_and_reach_initial_state():
    contract, args = _case()
    contract = replace(contract, use_qk_l2norm=False)
    args[6:] = [torch.tensor([0, 1, 3]), torch.tensor([0, 3, 3, 5])]
    for tensor in args[:6]:
        tensor.requires_grad_()
    full = gated_delta_rule_sequence(*args, contract=contract)
    chunked = _packed_two_chunks(args, contract)
    expected = _oracle(args, contract)
    _bits_equal(full[0], chunked[0])
    _bits_equal(full[1], chunked[1])

    def gradients(result):
        output, bank = result
        return torch.autograd.grad(output.square().sum() + bank.square().sum(), args[:6])

    full_grads, chunk_grads, oracle_grads = map(gradients, (full, chunked, expected))
    for full_grad, chunk_grad, oracle_grad in zip(
        full_grads, chunk_grads, oracle_grads, strict=True
    ):
        assert torch.isfinite(full_grad).all() and full_grad.count_nonzero()
        torch.testing.assert_close(full_grad, oracle_grad, atol=2e-7, rtol=2e-5)
        # Shared parameter gradients can accumulate in different autograd orders.
        torch.testing.assert_close(full_grad, chunk_grad, atol=2e-7, rtol=2e-5)


def test_507_configuration_is_a_supported_parameter_subset_not_a_new_profile():
    contract, args = _case(2, 4, 128, 128, dtype=torch.bfloat16)
    args[6:] = [torch.tensor([1, 2, 3]), torch.tensor([0, 3, 3, 5])]
    full = gated_delta_rule_sequence(*args, contract=contract)
    chunked = _packed_two_chunks(args, contract)
    expected = _oracle(args, contract)
    _bits_equal(full[0], chunked[0])
    _bits_equal(full[1], chunked[1])
    torch.testing.assert_close(full[0], expected[0], atol=2e-4, rtol=1e-2)
    torch.testing.assert_close(full[1], expected[1], atol=3e-7, rtol=2e-5)
    assert contract.profile == "vllm_decode_v1"
    assert full[0].dtype == torch.bfloat16 and full[1].dtype == torch.float32


def test_507_width4_bf16_conv_state_composes_with_gdn_continuation():
    from rl_engine.reference.linear_attn import CausalConv1dUpdateOp

    contract, args = _case(1, 2, 128, 128, dtype=torch.bfloat16)
    args[6:] = [torch.tensor([1]), torch.tensor([0, 5])]
    args[0].requires_grad_()
    args[5].requires_grad_()
    dim = args[0].shape[1]
    cache = torch.full((4, dim, 3), 0.125, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.full((dim, 4), 0.25, dtype=torch.bfloat16, requires_grad=True)

    def run(start, end, conv, recurrent):
        pieces = []
        for token in range(start, end):
            mixed, conv = CausalConv1dUpdateOp()(args[0][token : token + 1], conv, weight, args[6])
            pieces.append(mixed)
        out, recurrent = gated_delta_rule_sequence(
            torch.cat(pieces),
            args[1][start:end],
            args[2][start:end],
            *args[3:5],
            recurrent,
            args[6],
            torch.tensor([0, end - start]),
            contract=contract,
        )
        return out, conv, recurrent

    before = cache.detach().clone()
    full = run(0, 5, cache, args[5])
    prefix = run(0, 2, cache, args[5])
    suffix = run(2, 5, prefix[1], prefix[2])
    _bits_equal(full[0], torch.cat((prefix[0], suffix[0])))
    _bits_equal(full[1], suffix[1])
    _bits_equal(full[2], suffix[2])
    _bits_equal(cache, before)
    assert suffix[1].data_ptr() != cache.data_ptr()
    grads = torch.autograd.grad(suffix[0].float().square().sum(), (args[0], args[5], cache))
    assert all(torch.isfinite(grad).all() and grad.count_nonzero() for grad in grads)
    assert grads[0][:2].count_nonzero()  # prompt tokens survive both state handoffs


def test_empty_invocation_still_validates_state_and_dtype():
    contract, args = _case()
    args[:3] = [x[:0] for x in args[:3]]
    args[6:] = [torch.tensor([], dtype=torch.int64), torch.tensor([0])]
    args[5] = args[5].half()
    with pytest.raises(ValueError, match="state"):
        gated_delta_rule_sequence(*args, contract=contract)


def test_cpu_entry_rejects_non_cpu_requests_without_attempting_execution():
    contract, args = _case()
    args[:7] = [tensor.to("meta") for tensor in args[:7]]
    with pytest.raises(ValueError, match="CPU reference"):
        gated_delta_rule_sequence(*args, contract=contract)


def test_profile_33_wide_cancellation_fixture_pins_padded_reduction():
    """Distinguish the retained padded schedule from a whole-axis FP32 sum.

    Qualified AVX2/AVX512 paths yield 22; whole-axis and exact sums yield 21.
    This pins implementation behavior, not a hardware-independent reduction
    tree. A new PyTorch/device path needs qualification before changing it.
    """
    values = [
        2**24,
        -4,
        1,
        3,
        2,
        -5,
        4,
        0,
        -2,
        -1,
        2,
        1,
        3,
        3,
        4,
        -2,
        4,
        2,
        4,
        4,
        -4,
        -2,
        -5,
        2,
        2,
        2,
        2,
        -5,
        -1,
        2,
        5,
        0,
        -(2**24),
    ]
    output, _ = gated_delta_rule_sequence(
        torch.cat((torch.ones(33), torch.zeros(34))).reshape(1, -1),
        torch.full((1, 1), -100.0),
        torch.zeros(1, 1),
        torch.zeros(1),
        torch.zeros(1),
        torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 33),
        torch.tensor([0]),
        torch.tensor([0, 1]),
        contract=GDNContract(1, 1, 33, 1, scale=1, use_qk_l2norm=False),
    )
    assert output.item() == 22.0


def test_bf16_profile_keeps_normalized_keys_in_fp32_until_state_store():
    _, state = gated_delta_rule_sequence(
        torch.tensor([[0.5, 0.25, 0.75, 0.5, 1.0]], dtype=torch.bfloat16),
        torch.zeros(1, 1, dtype=torch.bfloat16),
        torch.zeros(1, 1, dtype=torch.bfloat16),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, 1, 1, 2),
        torch.tensor([0]),
        torch.tensor([0, 1]),
        contract=GDNContract(1, 1, 2, 1, scale=1),
    )
    # Initial state zero and beta=1/2: state is exactly half the normalized key.
    expected = torch.tensor([0.375 / sqrt(0.8125 + 1e-6), 0.25 / sqrt(0.8125 + 1e-6)])
    torch.testing.assert_close(state.flatten(), expected, atol=5e-8, rtol=0)


def test_decay_adds_bias_in_fp32_before_softplus():
    _, state = gated_delta_rule_sequence(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.bfloat16),
        torch.full((1, 1), 0.5, dtype=torch.bfloat16),
        torch.zeros(1, 1, dtype=torch.bfloat16),
        torch.zeros(1),
        torch.tensor([-0.501]),
        torch.ones(1, 1, 1, 1),
        torch.tensor([0]),
        torch.tensor([0, 1]),
        contract=GDNContract(1, 1, 1, 1, scale=1, use_qk_l2norm=False),
    )
    # exp(-softplus(-.001)) = sigmoid(.001), distinguish BF16 bias rounding.
    torch.testing.assert_close(
        state.flatten(), torch.tensor([0.5002499999791666]), atol=6e-8, rtol=0
    )


@pytest.mark.parametrize("heads", [1, 4, 8, 16, 33])
def test_sequence_agrees_with_batched_step_across_gate_vector_widths(heads):
    from rl_engine.reference.linear_attn import GatedDeltaRuleRecurrentStepOp

    generator = torch.Generator().manual_seed(469)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    # Every sequence has one token. This connects the shared sequence entry to
    # grouped decode without assuming CPU vector/scalar tails round identically.
    args = [
        rand(5, 2 * 7 + heads * 3),
        rand(5, heads),
        rand(5, heads),
        rand(heads),
        rand(heads),
        rand(6, heads, 3, 7) * 0.1,
        torch.arange(1, 6),
    ]
    batched, bank = GatedDeltaRuleRecurrentStepOp()(*args, scale=7**-0.5, num_k_heads=1)
    sequence, final = gated_delta_rule_sequence(
        *args, torch.arange(6), contract=GDNContract(1, heads, 7, 3, scale=7**-0.5)
    )
    _bits_equal(sequence, batched[:, 0])
    _bits_equal(final, bank)
