# SPDX-License-Identifier: Apache-2.0
"""CPU-only staged math, parameter ownership and checkpoint integration tests."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rl_engine.contracts.operators.expert_training import ExpertTrainingConfig, PackedExpertWeight
from rl_engine.models.deepseek_v4.expert_training import ExpertTrainingBlock, RoutedExpertMLP
from rl_engine.ops.moe.expert_training import expert_backward, expert_forward
from rl_engine.reference.moe.expert_training import ReferenceExpertTrainingBackend as Reference


def case(mode="combined", *, rows=5, recompute=False, backend=None, hidden=32, ffn=32):
    rng = torch.Generator().manual_seed(48)
    config = ExpertTrainingConfig(hidden, ffn, 3, mode, alpha=0.375)
    weights = []
    for n, k in ((2 * ffn, hidden), (hidden, ffn)):
        if config.use_base:
            codes = torch.randint(0, 256, (3, n, k // 2), generator=rng, dtype=torch.uint8)
            scales = torch.full((3, n, k // 32), 122, dtype=torch.uint8)
            weights.append(PackedExpertWeight(codes, scales, (3, n, k)))
        else:
            weights.append(None)
    adapters = None
    if config.use_lora:
        adapters = tuple(
            (torch.randn(shape, generator=rng) * 0.15).bfloat16()
            for shape in ((3, hidden), (2 * ffn, 3), (3, ffn), (hidden, 3))
        )
    module = RoutedExpertMLP(
        config,
        backend=backend or Reference(),
        w1=weights[0],
        w2=weights[1],
        adapters=adapters,
        recompute=recompute,
    )
    x = (torch.randn(rows, hidden, generator=rng) * 0.25).bfloat16().requires_grad_()
    p = torch.linspace(0, 0.75, rows).requires_grad_()
    offsets = torch.tensor([0, rows // 2, rows // 2, rows], dtype=torch.int64)
    return module, x, p, offsets


def run(module, x, p, offsets):
    module.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_()
    p = p.detach().requires_grad_()
    y = module(x, p, offsets)
    dy = torch.linspace(-0.3, 0.5, y.numel()).reshape(y.shape).bfloat16()
    y.backward(dy)
    return (
        y.detach(),
        x.grad,
        p.grad,
        {name: t.grad.clone() for name, t in module.named_parameters() if t.requires_grad},
    )


def equal_results(a, b):
    for lhs, rhs in zip(a[:3], b[:3]):
        assert torch.equal(lhs, rhs)
    assert a[3].keys() == b[3].keys()
    for name in a[3]:
        assert torch.equal(a[3][name], b[3][name]), name


@pytest.mark.parametrize("mode", ["base_only", "lora_only", "combined"])
@pytest.mark.parametrize("rows", [0, 1, 5])
def test_save_recompute_and_rollout_share_forward(mode, rows):
    module, x, p, offsets = case(mode, rows=rows)
    saved = run(module, x, p, offsets)
    equal_results(saved, run(module, x, p, offsets))
    module.recompute = True
    equal_results(saved, run(module, x, p, offsets))
    with torch.no_grad():
        assert torch.equal(saved[0], module(x, p, offsets))
    if rows == 0:
        assert all(torch.count_nonzero(t) == 0 for t in saved[3].values())


@pytest.mark.parametrize("reentrant", [False, True])
def test_framework_checkpoint_matches_native_recompute(reentrant):
    module, x, p, offsets = case()
    expected = run(module, x, p, offsets)
    module.zero_grad(set_to_none=True)
    # Both input tensors are explicit checkpoint arguments; no hidden detach.
    y = checkpoint(lambda a, b: module(a, b, offsets), x, p, use_reentrant=reentrant)
    dy = torch.linspace(-0.3, 0.5, y.numel()).reshape(y.shape).bfloat16()
    y.backward(dy)
    actual = (y.detach(), x.grad, p.grad, {name: t.grad for name, t in module.named_parameters()})
    equal_results(expected, actual)


def test_lora_trains_with_frozen_inputs_and_two_optimizer_updates():
    module, x, p, offsets = case(recompute=True)
    x, p = x.detach(), p.detach()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
    original_base = {name: t.clone() for name, t in module.named_buffers()}
    original = {name: t.clone() for name, t in module.named_parameters()}
    outputs = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = module(x, p, offsets)
        outputs.append(out.detach().clone())
        # Make each update larger than a BF16 parameter ULP in this small fixture.
        (out.float().square().sum() * 128).backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in module.parameters())
        assert all(t.grad is None for t in module.buffers())
        optimizer.step()
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])
    assert all(not torch.equal(original[name], t) for name, t in module.named_parameters())
    assert all(torch.equal(original_base[name], t) for name, t in module.named_buffers())


class Spy(Reference):
    def __init__(self):
        self.calls = []

    def quantize(self, x):
        self.calls.append(("quant", x.clone()))
        return super().quantize(x)

    def lora_forward(self, x, a, b, alpha):
        self.calls.append(("lora", x.clone()))
        return super().lora_forward(x, a, b, alpha)

    def activation_forward(self, gate, up, p):
        out = super().activation_forward(gate, up, p)
        self.calls.append(("activation", out[0].clone()))
        return out


def test_original_bf16_goes_to_lora_and_weight_precedes_fc2_quantization():
    backend = Spy()
    module, x, p, offsets = case(backend=backend)
    module(x, p, offsets)
    assert [label for label, _ in backend.calls] == ["quant", "lora", "activation", "quant", "lora"]
    assert torch.equal(backend.calls[1][1], x)
    assert torch.equal(backend.calls[2][1], backend.calls[3][1])
    assert torch.equal(backend.calls[2][1], backend.calls[4][1])
    assert torch.count_nonzero(backend.calls[2][1][0]) == 0  # p_s[0] == 0
    assert torch.count_nonzero(x[0]) != 0


def test_fp32_branches_merge_before_bf16_rounding():
    class MergeProbe(Reference):
        dh_seen = None

        def base_backward(self, dy, w, offsets):
            return torch.full((dy.shape[0], w.shape[-1]), 1.00390625)

        def lora_backward(self, dy, x, a, b, alpha, u):
            return torch.full(x.shape, 0.00390625), torch.zeros(a.shape), torch.zeros(b.shape)

        def activation_backward(self, dh, gate, up, p, state, need_route_grad):
            self.dh_seen = dh.clone()
            return torch.ones_like(gate), torch.ones_like(up), torch.zeros_like(p)

    backend = MergeProbe()
    module, x, p, offsets = case(backend=backend, rows=1)
    _, saved = expert_forward(
        module.config, backend, x, p, offsets, *module.base_weights(), module.adapters()
    )
    grads = expert_backward(
        module.config,
        backend,
        torch.ones_like(x),
        x,
        p,
        offsets,
        *module.base_weights(),
        module.adapters(),
        saved,
        need_route_grad=True,
    )
    assert torch.all(backend.dh_seen == 1.0078125)
    assert torch.all(grads.dx == 1.0078125)
    early = torch.tensor(1.00390625).bfloat16() + torch.tensor(0.00390625).bfloat16()
    assert early.item() != 1.0078125


def test_lora_only_gradients_against_independent_fp64_and_finite_difference():
    module, x, p, offsets = case("lora_only", rows=2, hidden=4, ffn=3)
    with torch.no_grad():
        p.fill_(0.5)
    actual = run(module, x, p, offsets)
    leaves = [t.detach().double().requires_grad_() for t in (x, p, *module.adapters())]
    dy = torch.linspace(-0.3, 0.5, actual[0].numel()).reshape(actual[0].shape).bfloat16().double()

    def smooth(values):
        xx, pp, a1, b1, a2, b2 = values
        z = ((xx @ a1.t()) @ b1.t()) * module.config.alpha
        g, u = z.chunk(2, dim=1)
        h = (torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10)) * pp[:, None]
        return (((h @ a2.t()) @ b2.t()) * module.config.alpha * dy).sum()

    expected = torch.autograd.grad(smooth(leaves), leaves)
    actual_grads = [actual[1], actual[2], *actual[3].values()]
    # Compare to the smooth formula, separately from the rounded staged profile.
    for got, want in zip(actual_grads, expected):
        torch.testing.assert_close(got.double(), want, rtol=0.06, atol=2e-7)
    for index in (0, 1, 2, 3, 4, 5):
        plus, minus = [t.detach().clone() for t in leaves], [t.detach().clone() for t in leaves]
        plus[index].reshape(-1)[0] += 1e-5
        minus[index].reshape(-1)[0] -= 1e-5
        finite = (smooth(plus) - smooth(minus)) / 2e-5
        assert finite.item() == pytest.approx(
            expected[index].reshape(-1)[0].item(), rel=1e-6, abs=1e-10
        )


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("mutated", ["x", "p", "offsets", "a1", "w1_codes", "w2_scales"])
def test_mutation_between_forward_and_backward_is_rejected(recompute, mutated):
    module, x, p, offsets = case(recompute=recompute)
    out = module(x, p, offsets)
    target = {"x": x, "p": p, "offsets": offsets}.get(mutated)
    if target is None:
        target = getattr(module, mutated)
    with torch.no_grad():
        target.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        out.sum().backward()


def test_profile_change_and_missing_route_gradient_are_rejected():
    module, x, p, offsets = case()
    out = module(x, p, offsets)
    module.backend.profile = "different"
    with pytest.raises(RuntimeError, match="profile changed"):
        out.sum().backward()

    class DropsRoute(Reference):
        def activation_backward(self, *args):
            dg, du, _ = super().activation_backward(*args)
            return dg, du, None

    module, x, p, offsets = case(backend=DropsRoute())
    with pytest.raises(RuntimeError, match="dropped"):
        module(x, p, offsets).sum().backward()


@pytest.mark.parametrize("mode", ["base_only", "lora_only", "combined"])
def test_strides_and_one_row_expert_placement(mode):
    module, x, p, offsets = case(mode)
    x_strided = torch.stack((x.detach(), x.detach()), -1)[..., 0]
    p_strided = torch.stack((p.detach(), p.detach()), -1)[..., 0]
    assert not x_strided.is_contiguous() and not p_strided.is_contiguous()
    equal_results(run(module, x, p, offsets), run(module, x_strided, p_strided, offsets))
    with torch.no_grad():
        all_rows = module(x, p, offsets)
        for row in range(x.shape[0]):
            expert = 0 if row < offsets[1] else 2
            one_offsets = torch.tensor([0 if i <= expert else 1 for i in range(4)])
            single = module(x[row : row + 1], p[row : row + 1], one_offsets)
            assert torch.equal(single, all_rows[row : row + 1])


def test_state_dict_reload_and_alpha_mismatch():
    module, x, p, offsets = case()
    state = copy.deepcopy(module.state_dict())
    with torch.no_grad():
        expected = module(x, p, offsets)
        module.a1.add_(1)
        assert not torch.equal(expected, module(x, p, offsets))
    module.load_state_dict(state)
    with torch.no_grad():
        assert torch.equal(expected, module(x, p, offsets))
    state["_extra_state"]["config"]["alpha"] = 99
    state["w1_codes"].zero_()
    before = module.w1_codes.clone()
    with pytest.raises(ValueError, match="checkpoint schema/config mismatch"):
        module.load_state_dict(state)
    assert torch.equal(module.w1_codes, before)


def test_config_and_extra_trainable_parameters_are_rejected():
    module, x, p, offsets = case()
    module.register_parameter("unexpected_base", nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="unsupported trainable"):
        module(x, p, offsets)
    del module.unexpected_base
    module.config = replace(module.config, alpha=99)
    with pytest.raises(RuntimeError, match="config changed"):
        module(x, p, offsets)


@pytest.mark.parametrize(
    "invalid", ["x_dtype", "p_shape", "offset_end", "offset_order", "adapter_dtype"]
)
def test_invalid_inputs_fail_before_backend(invalid):
    backend = Spy()
    module, x, p, offsets = case(backend=backend)
    if invalid == "x_dtype":
        x = x.float()
    elif invalid == "p_shape":
        p = p[:, None]
    elif invalid == "offset_end":
        offsets[-1] += 1
    elif invalid == "offset_order":
        offsets[1], offsets[2] = 3, 1
    else:
        module.a1 = nn.Parameter(module.a1.float())
    with pytest.raises((ValueError, TypeError)):
        module(x, p, offsets)
    assert backend.calls == []


def test_bad_backend_dtype_and_quant_layout_are_rejected():
    class EarlyRound(Reference):
        def base_forward(self, *args):
            return super().base_forward(*args).bfloat16()

    module, x, p, offsets = case(backend=EarlyRound())
    with pytest.raises(TypeError, match="base output"):
        module(x, p, offsets)

    class WrongLayout(Reference):
        def quantize(self, x):
            return replace(super().quantize(x), layout="group128")

    module, x, p, offsets = case(backend=WrongLayout())
    with pytest.raises(ValueError, match="block-32"):
        module(x, p, offsets)


def test_shared_branch_runs_once_on_original_tokens_and_keeps_gradients():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora = nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x * self.lora

    module, x, p, offsets = case()
    shared = Shared()
    block = ExpertTrainingBlock(module, shared)
    original = torch.ones(2, 32, dtype=torch.bfloat16, requires_grad=True)
    slots = torch.tensor([8, 2, 11, 4, 7])
    outputs = block(x, p, offsets, slots, shared_x=original)
    assert shared.calls == 1 and outputs.shared.shape == (2, 32)
    assert torch.equal(outputs.output_slot, slots) and outputs.route_weight_applied
    slots.zero_()
    assert outputs.output_slot.tolist() == [8, 2, 11, 4, 7]
    (outputs.routed.float().sum() + outputs.shared.float().sum()).backward()
    assert x.grad is not None and p.grad is not None
    assert shared.lora.grad is not None and original.grad is not None


def test_shared_and_slot_misuse_is_rejected():
    module, x, p, offsets = case()
    slots = torch.arange(x.shape[0])
    block = ExpertTrainingBlock(module, nn.Identity())
    with pytest.raises(ValueError, match="original shared_x"):
        block(x, p, offsets, slots)
    with pytest.raises(ValueError, match="unique"):
        block(x, p, offsets, torch.zeros_like(slots), shared_x=x)


def test_frozen_lora_parameter_stays_frozen():
    module, x, p, offsets = case()
    module.a1.requires_grad_(False)
    module(x, p, offsets).float().sum().backward()
    assert module.a1.grad is None
    assert module.b1.grad is not None and x.grad is not None and p.grad is not None


def test_zero_route_weight_can_still_receive_a_gradient():
    module, x, p, offsets = case()
    result = run(module, x, p, offsets)
    assert torch.count_nonzero(result[0][0]) == 0
    assert torch.count_nonzero(result[1][0]) == 0
    assert result[2][0] != 0  # no division by p_s, and no dropped p_s == 0 branch


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 31},
        {"experts": 0},
        {"mode": "unknown"},
        {"alpha": float("nan")},
    ],
)
def test_invalid_module_config(kwargs):
    options = dict(hidden=32, ffn=32, experts=3)
    options.update(kwargs)
    with pytest.raises(ValueError):
        ExpertTrainingConfig(**options)


def test_partial_adapter_and_unexpected_layout_are_rejected():
    module, _, _, _ = case()
    w1, w2 = module.base_weights()
    with pytest.raises(ValueError, match="A1, B1, A2 and B2"):
        RoutedExpertMLP(
            module.config, backend=Reference(), w1=w1, w2=w2, adapters=module.adapters()[:2]
        )
    with pytest.raises(ValueError, match="shape/layout"):
        RoutedExpertMLP(
            module.config,
            backend=Reference(),
            w1=replace(w1, layout="preshuffled"),
            w2=w2,
            adapters=module.adapters(),
        )


def test_autocast_cannot_move_expert_boundaries():
    module, x, p, offsets = case()
    expected = run(module, x, p, offsets)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = run(module, x, p, offsets)
    equal_results(expected, actual)
