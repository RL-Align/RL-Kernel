# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Packed decode recurrence boundary checks; no complete-model L2 claim."""

import pytest
import torch

from rl_engine.integrations.engines.train.vllm.qwen3_next_gdn import (
    decode_loop_sequence,
    packed_recurrent_sequence,
)
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact


def inputs(lengths, seed=7, blocks=None):
    assert torch.cuda.is_available(), "CUDA is required; this acceptance suite cannot skip"
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    tokens = sum(lengths)
    # Actual TP4 shard: 16 / 4 key heads, 32 / 4 value heads, K=V=128.
    tensors = [
        rand(tokens, 2048),
        rand(tokens, 8),
        rand(tokens, 8),
        rand(8, dtype=torch.float32),
        rand(8, dtype=torch.float32),
        rand(blocks or len(lengths) + 1, 8, 128, 128, dtype=torch.float32) * 0.01,
    ]
    indices = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    return tensors, indices, cu


def run(tensors, indices, cu):
    return packed_recurrent_sequence(*tensors, indices, cu, num_k_heads=4)


@torch.no_grad()
@pytest.mark.parametrize("lengths", [(8,), (64, 8, 0, 16), (1024, 256, 64, 8)])
def test_packed_matches_independent_sequences_and_reordering(lengths):
    tensors, indices, cu = inputs(lengths)
    original_state = tensors[-1].clone()
    output, state = run(tensors, indices, cu)
    for seq, length in enumerate(lengths):
        start, end = cu[seq : seq + 2].tolist()
        one = [*(value[start:end] for value in tensors[:3]), *tensors[3:]]
        expected, expected_state = run(one, indices[seq : seq + 1], torch.tensor([0, length]))
        exact(output[start:end], expected)
        exact(state[seq + 1], expected_state[seq + 1])
    permutation = list(reversed(range(len(lengths))))
    rows = torch.cat([torch.arange(cu[seq], cu[seq + 1], device="cuda") for seq in permutation])
    changed = [*(value.index_select(0, rows) for value in tensors[:3]), *tensors[3:]]
    reordered_cu = torch.tensor([0, *torch.tensor([lengths[i] for i in permutation]).cumsum(0)])
    reordered, reordered_state = run(changed, indices[permutation], reordered_cu)
    exact(reordered, output.index_select(0, rows))
    exact(reordered_state, state)
    exact(tensors[-1], original_state)


@torch.no_grad()
def test_1024_token_chunk_continuation_is_bitwise_exact():
    tensors, indices, cu = inputs((1024,))
    whole, whole_state = run(tensors, indices, cu)
    state, pieces = tensors[-1], []
    cuts = (0, 1, 8, 64, 256, 513, 1024)
    for start, end in zip(cuts, cuts[1:]):
        chunk = [*(value[start:end] for value in tensors[:3]), *tensors[3:5], state]
        output, state = run(chunk, indices, torch.tensor([0, end - start]))
        pieces.append(output)
    exact(torch.cat(pieces), whole)
    exact(state, whole_state)


def test_chunk_boundary_retains_prompt_and_initial_state_gradients():
    tensors, indices, cu = inputs((8,))
    tensors = [tensor.requires_grad_() for tensor in tensors]
    prompt = [*(value[:4] for value in tensors[:3]), *tensors[3:]]
    _, prompt_state = run(prompt, indices, torch.tensor([0, 4]))
    prompt_state.retain_grad()
    response = [*(value[4:] for value in tensors[:3]), *tensors[3:5], prompt_state]
    out, state = run(response, indices, torch.tensor([0, 4]))
    (out.float().square().sum() + state.square().sum() * 0.01).backward()
    assert prompt_state.grad is not None and prompt_state.grad.abs().max() > 0
    for value in tensors:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().max() > 0
    assert tensors[0].grad[:4].abs().max() > 0


@pytest.mark.parametrize("bad", [[1, 8], [0, 9], [0, 6, 5, 8]])
def test_bad_packing_is_rejected(bad):
    tensors, indices, _ = inputs((8,))
    with pytest.raises(ValueError, match="cu_seqlens"):
        run(tensors, indices, torch.tensor(bad))


def test_reserved_and_duplicate_slots_are_rejected():
    tensors, indices, cu = inputs((4, 4))
    for bad in ([0, 1], [1, 1]):
        with pytest.raises(ValueError):
            run(tensors, torch.tensor(bad, device="cuda", dtype=torch.int32), cu)


@torch.no_grad()
@pytest.mark.parametrize(
    "lengths, blocks",
    [
        ((1,), None),
        ((8,), None),
        ((64, 8, 0, 16), None),
        ((1024, 256, 64, 8), None),
        ((1024, 777, 1500, 64), None),
        # A rollout-sized cache bank. The op is functional: it returns a new bank and leaves
        # every slot of the caller's bank, the 16 it reads included, byte for byte unchanged.
        ((512,) * 16, 362),
    ],
)
def test_one_launch_forward_equals_one_decode_step_per_token(lengths, blocks):
    tensors, indices, cu = inputs(lengths, seed=11, blocks=blocks)
    original_state = tensors[-1].clone()
    output, state = run(tensors, indices, cu)
    expected, expected_state = decode_loop_sequence(*tensors, indices, cu, num_k_heads=4)
    exact(output, expected)
    exact(state, expected_state)
    exact(tensors[-1], original_state)


def test_backward_wiring_equals_the_decode_loop_gradient():
    """The one-launch backward differentiates the decode loop by design, so this checks its
    wiring (saved inputs, which inputs need gradients, an empty sequence, the state bank's
    routing), not the gradient math; that is checked against FP64 below."""
    lengths = (16, 0, 9, 32)
    grads = []
    for function in (packed_recurrent_sequence, decode_loop_sequence):
        tensors, indices, cu = inputs(lengths, seed=13)
        tensors = [tensor.requires_grad_() for tensor in tensors]
        out, state = function(*tensors, indices, cu, num_k_heads=4)
        (out.float().square().sum() + state.square().sum() * 0.01).backward()
        grads.append([tensor.grad for tensor in tensors])
    for fused, loop in zip(*grads):
        assert fused is not None and fused.abs().max() > 0
        exact(fused, loop)


def test_all_empty_sequences_return_a_new_bank_and_backpropagate():
    tensors, indices, cu = inputs((4,))
    tensors = [tensor[:0] if i < 3 else tensor for i, tensor in enumerate(tensors)]
    state = tensors[-1].clone().requires_grad_()
    out, new_state = packed_recurrent_sequence(
        *tensors[:-1], state, indices, torch.tensor([0, 0]), num_k_heads=4
    )
    assert out.shape[0] == 0
    assert new_state.data_ptr() != state.data_ptr()
    exact(new_state, state)
    new_state.square().sum().backward()
    exact(state.grad, 2 * state.detach())


def _reference_recurrence(qkv, a, b, A_log, dt_bias, state, indices, cu, *, num_k_heads):
    """Independent FP64 PyTorch gated delta rule, written from vLLM 0.30.0's kernel.

    Per token: q, k L2-normalised (eps 1e-6), q scaled by 128 ** -0.5; h *= exp(g);
    v -= h k; v *= beta; h += v k^T; o = h q. g = -exp(A_log) softplus(a + dt_bias) and
    beta = sigmoid(b). Key heads are shared by consecutive value heads, as in the kernel.
    """
    qkv, a, b = qkv.double(), a.double(), b.double()
    A_log, dt_bias, bank = A_log.double(), dt_bias.double(), state.double()
    tokens, heads = qkv.shape[0], bank.shape[1]
    width = num_k_heads * 128
    group = heads // num_k_heads
    q_all = qkv[:, :width].reshape(tokens, num_k_heads, 128).repeat_interleave(group, 1)
    k_all = qkv[:, width : 2 * width].reshape(tokens, num_k_heads, 128)
    k_all = k_all.repeat_interleave(group, 1)
    v_all = qkv[:, 2 * width :].reshape(tokens, heads, 128)
    g = -A_log.exp() * torch.nn.functional.softplus(a + dt_bias)
    beta = torch.sigmoid(b)
    outputs, slots = [], {}
    for n, (start, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        slot = int(indices[n])
        h = bank[slot]
        for t in range(start, end):
            q = q_all[t] / (q_all[t].square().sum(-1, keepdim=True) + 1e-6).sqrt() * 128**-0.5
            k = k_all[t] / (k_all[t].square().sum(-1, keepdim=True) + 1e-6).sqrt()
            h = h * g[t].exp()[:, None, None]
            v = (v_all[t] - (h * k[:, None, :]).sum(-1)) * beta[t][:, None]
            h = h + v[:, :, None] * k[:, None, :]
            outputs.append((h * q[:, None, :]).sum(-1))
        if end > start:
            slots[slot] = h
    final = torch.stack([slots.get(i, bank[i]) for i in range(bank.shape[0])])
    return torch.stack(outputs), final


@pytest.mark.parametrize("lengths", [(5, 0, 3), (16, 0, 9, 32)])
def test_gradients_match_an_independent_reference(lengths):
    """The decode-loop gradient (used by the one-launch backward) against FP64 autograd of
    a separate PyTorch recurrence, so a shared bug in the decode loop would be caught."""
    tensors, indices, cu = inputs(lengths, seed=21)
    weights = None
    results = []
    for function in ("fused", "reference"):
        leaves = [tensor.detach().clone().requires_grad_() for tensor in tensors]
        if function == "fused":
            out, state = packed_recurrent_sequence(*leaves, indices, cu, num_k_heads=4)
        else:
            out, state = _reference_recurrence(*leaves, indices, cu, num_k_heads=4)
        if weights is None:
            generator = torch.Generator(device="cuda").manual_seed(5)
            weights = torch.randn(
                out.shape, device="cuda", generator=generator, dtype=torch.float64
            )
        loss = (out.double() * weights).sum() + state.double().square().sum() * 0.01
        loss.backward()
        results.append((out.detach().double(), [leaf.grad.double() for leaf in leaves]))
    (fused_out, fused_grads), (reference_out, reference_grads) = results

    def relative(x, y):
        return float((x - y).norm() / y.norm().clamp_min(1e-30))

    assert relative(fused_out, reference_out) < 1e-2
    for fused, reference in zip(fused_grads, reference_grads):
        # A zero reference would make the relative error vacuous.
        assert reference.abs().max() > 0
        assert relative(fused, reference) < 1e-2, relative(fused, reference)
