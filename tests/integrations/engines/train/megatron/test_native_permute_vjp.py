# SPDX-License-Identifier: Apache-2.0
"""CPU contract of the fixed P6 permute VJP; the ROCm kernel is checked in tests/backends/rocm."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.operators.moe import MoEOutputCombineContractError
from rl_engine.integrations.engines.train.megatron import native_combine
from rl_engine.reference.moe.fixed_output_combine import fixed_output_combine_reference

moe_utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
permute = moe_utils.permute
unpermute = moe_utils.unpermute


class Dispatcher:
    """The two native all-to-all boundaries, calling module-global permute/unpermute."""

    def __init__(self, routing_map):
        self.routing_map = routing_map
        self.num_out_tokens = int(routing_map.sum())

    def dispatch_preprocess(self, hidden_states, probs):
        self.hidden_shape = hidden_states.shape
        tokens, permuted_probs, self.sorted_indices, _, _ = permute(
            hidden_states,
            self.routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            fused=False,
            drop_and_pad=False,
            return_batch_invariant_inverse_map=False,
        )
        return tokens, permuted_probs

    def combine_postprocess(self, hidden_states):
        return unpermute(
            hidden_states, self.sorted_indices, self.hidden_shape, routing_map=self.routing_map
        )


def routing(t, e=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    experts = torch.rand(t, e, generator=g).topk(6, dim=1).indices
    routing_map = torch.zeros(t, e, dtype=torch.bool).scatter_(1, experts, True)
    probs = torch.rand(t, e, generator=g) * routing_map
    return routing_map, probs


def fixed_reference(dy, routing_map):
    """Independent oracle: per token, FP32 sum of its rows by ascending expert, one cast."""
    experts, tokens = routing_map.T.nonzero(as_tuple=True)
    out = torch.empty((routing_map.shape[0], dy.shape[1]), dtype=dy.dtype)
    for token in range(routing_map.shape[0]):
        rows = (tokens == token).nonzero().flatten()
        rows = rows[experts[rows].argsort()]
        acc = torch.zeros(dy.shape[1], dtype=torch.float32)
        for row in rows.tolist():
            acc = acc + dy[row].float()
        out[token] = acc.to(dy.dtype)
    return out


def raw_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.fixture
def cpu_combine(monkeypatch):
    # The ROCm candidate has no CPU path; the CPU reference has the same contract.
    def reference(prepared, expert_output, shared_output=None, *, mapping_run_id):
        return fixed_output_combine_reference(
            prepared, expert_output, shared_output, mapping_run_id=mapping_run_id
        ).output

    monkeypatch.setattr(native_combine, "output_combine", reference)


def bound(routing_map):
    dispatcher = Dispatcher(routing_map)
    native_combine.bind_native_combine(SimpleNamespace(token_dispatcher=dispatcher))
    return dispatcher


@pytest.mark.parametrize("t,h", [(1, 8), (7, 33), (64, 128)])
def test_forward_is_native_index_select_and_backward_is_fixed_p6(cpu_combine, t, h):
    routing_map, probs = routing(t, seed=t)
    hidden = torch.randn(t, h, generator=torch.Generator().manual_seed(1)).bfloat16()
    dy = torch.randn(t * 6, h, generator=torch.Generator().manual_seed(2)).bfloat16()
    dp = torch.randn(t * 6, generator=torch.Generator().manual_seed(3))

    native = Dispatcher(routing_map)
    x0, p0 = hidden.clone().requires_grad_(), probs.clone().requires_grad_()
    n_tokens, n_probs = native.dispatch_preprocess(x0, p0)
    (n_probs * dp).sum().backward()

    fixed = bound(routing_map)
    x1, p1 = hidden.clone().requires_grad_(), probs.clone().requires_grad_()
    f_tokens, f_probs = fixed.dispatch_preprocess(x1, p1)
    raw_equal(f_tokens, n_tokens)
    raw_equal(f_tokens, hidden.index_select(0, native.sorted_indices))
    raw_equal(f_probs, n_probs)
    assert torch.equal(fixed.sorted_indices, native.sorted_indices)
    torch.autograd.backward((f_tokens, f_probs), (dy, dp))
    raw_equal(x1.grad, fixed_reference(dy, routing_map))
    raw_equal(p1.grad, p0.grad)


def test_backward_repeats_bytewise(cpu_combine):
    t, h = 32, 64
    routing_map, probs = routing(t, seed=5)
    hidden = torch.randn(t, h).bfloat16()
    dy = (torch.randn(t * 6, h) * torch.logspace(-3, 3, h)).bfloat16()
    grads = []
    for _ in range(5):
        x = hidden.clone().requires_grad_()
        tokens, _ = bound(routing_map).dispatch_preprocess(x, probs)
        tokens.backward(dy)
        grads.append(x.grad)
    for grad in grads[1:]:
        raw_equal(grad, grads[0])
    raw_equal(grads[0], fixed_reference(dy, routing_map))


def test_combine_postprocess_binding_is_retained(cpu_combine):
    t, h = 8, 16
    routing_map, probs = routing(t, seed=9)
    dispatcher = bound(routing_map)
    tokens, permuted_probs = dispatcher.dispatch_preprocess(torch.randn(t, h).bfloat16(), probs)
    rows = (tokens.float() * permuted_probs[:, None]).bfloat16()
    out = dispatcher.combine_postprocess(rows)
    raw_equal(out, fixed_reference(rows, routing_map))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fused=True),
        dict(drop_and_pad=True),
        dict(tokens_per_expert=torch.ones(256)),
        dict(align_size=16),
        dict(return_batch_invariant_inverse_map=True),
    ],
)
def test_unsupported_native_permute_modes_are_refused(cpu_combine, kwargs):
    routing_map, probs = routing(4, seed=1)
    dispatcher = bound(routing_map)
    fixed_permute = dispatcher.dispatch_preprocess.__func__.__globals__["permute"]
    with pytest.raises(ValueError, match="ordinary dropless"):
        fixed_permute(torch.randn(4, 8), routing_map, probs=probs, num_out_tokens=24, **kwargs)


def test_padded_routing_is_refused_by_the_p6_contract(cpu_combine):
    routing_map, probs = routing(4, seed=2)
    routing_map[0, (~routing_map[0]).nonzero()[0]] = True  # a seventh row, as padding would add
    x = torch.randn(4, 8).bfloat16().requires_grad_()
    tokens, _ = bound(routing_map).dispatch_preprocess(x, probs)
    with pytest.raises(MoEOutputCombineContractError):
        tokens.backward(torch.ones_like(tokens))


def test_dispatcher_without_native_permute_boundary_is_refused():
    class Other(Dispatcher):
        def dispatch_preprocess(self, hidden_states, probs):
            return hidden_states, probs

    with pytest.raises(ValueError, match="permute boundary"):
        native_combine.bind_native_combine(
            SimpleNamespace(token_dispatcher=Other(routing(2, seed=0)[0]))
        )
