# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared provider assertions for the Nemotron experts operator.

Imported by the CUDA and ROCm suites so both columns check one contract; the
module name has no `test_` prefix, so pytest never collects it directly.

Two geometries: `CORE` stays small enough for a fast suite while still crossing
tile boundaries (17 tokens, 8 experts, 256/128 widths), and `REAL` is the
Nemotron Nano shape (2688/1856, 128 experts, shared 3712) used by the
operator-level cases.
"""

from __future__ import annotations

import torch

from rl_engine.reference.moe.nemotron_experts import routed_experts, shared_experts
from rl_engine.reference.moe.nemotron_router import dispatch_tokens, route_scores

SLOTS = 6
CORE = {"width": 256, "intermediate": 128, "experts": 8, "shared_width": 384}
REAL = {"width": 2688, "intermediate": 1856, "experts": 128, "shared_width": 3712}
# Empirical thresholds for this workload, not universal error bounds. The floor
# is set by the contract itself: leaves round to BF16 (eps 7.8e-3) and the
# combine adds six signed terms before the single exit cast. Measured relative L2
# against an unrounded FP64 pipeline: native 7.1e-3 / 8.8e-3 and Triton 5.4e-3 /
# 1.0e-2 (CORE / REAL geometry) for the forward, similar for the gradients.
# Bitwise batch and relocation invariance, not this number, is the operator's
# actual claim.
FORWARD_L2 = 2e-2
PAYLOAD_L2 = 2e-2
WEIGHT_L2 = 2e-2
ROUTE_L2 = 2e-2
# The two providers agree to ~1.6 BF16 ulp: they share the leaf width, the leaf
# rounding and the exit cast but not the association of the leaf sums, and the
# output is BF16, so a sub-ulp cross-provider bound is not attainable. Measured
# 8.3e-3 on CORE and 1.3e-2 on REAL geometry.
CROSS_PROVIDER_L2 = 2e-2
GRAD_TARGETS = ("packed", "x", "weights", "w_up", "w_down", "w_sh1", "w_sh2")


def bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape, f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ), "bitwise mismatch"


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.double() - expected.double()).norm()
    return difference.div(expected.double().norm().clamp_min(1e-30)).item()


def build_case(
    tokens,
    device,
    *,
    width,
    intermediate,
    experts,
    shared_width=None,
    seed=434,
    concentrated=False,
    dtype=torch.bfloat16,
):
    """Router output, payload and stacked weights for one workload."""
    generator = torch.Generator().manual_seed(seed)
    payload = torch.randn(tokens, width, generator=generator) * 0.5
    scores = torch.rand(tokens, experts, generator=generator)
    bias = torch.zeros(experts)
    if concentrated:
        # Six heavily corrected experts collect every route; the rest are empty.
        bias[: min(SLOTS, experts)] = 4.0
    ids, route_weights = route_scores(scores, bias, min(SLOTS, experts))
    permutation, offsets, packed = dispatch_tokens(payload, ids, experts)
    case = {
        "packed": packed.to(device=device, dtype=dtype).contiguous(),
        "x": payload.to(device=device, dtype=dtype).contiguous(),
        "weights": route_weights.to(device=device).contiguous(),
        "permutation": permutation.to(device=device),
        "offsets": offsets.to(device=device),
        "w_up": (torch.randn(experts, intermediate, width, generator=generator) * 0.05)
        .to(device=device, dtype=dtype)
        .contiguous(),
        "w_down": (torch.randn(experts, width, intermediate, generator=generator) * 0.05)
        .to(device=device, dtype=dtype)
        .contiguous(),
        "ids": ids,
    }
    if shared_width is not None:
        case["w_sh1"] = (
            (torch.randn(shared_width, width, generator=generator) * 0.05)
            .to(device=device, dtype=dtype)
            .contiguous()
        )
        case["w_sh2"] = (
            (torch.randn(width, shared_width, generator=generator) * 0.05)
            .to(device=device, dtype=dtype)
            .contiguous()
        )
    return case


def cotangent_for(case, seed=7):
    generator = torch.Generator().manual_seed(seed)
    width = case["x"].shape[1]
    values = torch.randn(case["x"].shape[0], width, generator=generator) * 0.01
    return values.to(device=case["x"].device, dtype=case["x"].dtype)


def call(provider, case):
    return provider(
        case["packed"],
        case["x"],
        case["weights"],
        case["permutation"],
        case["offsets"],
        case["w_up"],
        case["w_down"],
        case.get("w_sh1"),
        case.get("w_sh2"),
    )


def _leaves(case):
    leaves = {}
    for name, value in case.items():
        tensor = value.detach().clone()
        if name in GRAD_TARGETS:
            tensor.requires_grad_(True)
        leaves[name] = tensor
    return leaves


def gradient_names(case):
    shared = "w_sh1" in case
    return [name for name in GRAD_TARGETS if name in case and (name != "x" or shared)]


def run(provider, case, cotangent=None):
    """Forward plus, for a fixed cotangent, every input gradient."""
    leaves = _leaves(case)
    out = call(provider, leaves)
    if cotangent is None:
        return out, {}
    names = gradient_names(case)
    grads = torch.autograd.grad(out, [leaves[name] for name in names], cotangent)
    for name, gradient in zip(names, grads):
        # A gradient must be shaped like its input: autograd broadcasts a stray
        # leading dimension away instead of raising, which would hand callers a
        # `[1, ...]` `.grad` for a rank-two parameter.
        assert gradient.shape == leaves[name].shape, (
            f"{name} gradient shape {tuple(gradient.shape)} != input "
            f"{tuple(leaves[name].shape)}"
        )
    return out, dict(zip(names, grads))


def exact(case, cotangent):
    """FP64 wide output and gradients from the same inputs, no exit cast."""
    names = gradient_names(case)
    leaves = {name: case[name].detach().double().requires_grad_() for name in names}
    wide = routed_experts(
        leaves["packed"],
        leaves["weights"],
        case["permutation"],
        case["offsets"],
        leaves["w_up"],
        leaves["w_down"],
    )
    if "w_sh1" in case:
        wide = wide + shared_experts(leaves["x"], leaves["w_sh1"], leaves["w_sh2"])
    grads = torch.autograd.grad(wide, [leaves[name] for name in names], cotangent.double())
    return wide, dict(zip(names, grads))


def isolated_token(case, token):
    """A one-token workload whose packed rows are that token's six copies."""
    ids = case["ids"][token]
    slots = int(ids.numel())
    experts = case["w_up"].shape[0]
    device = case["x"].device
    counts = torch.bincount(ids, minlength=experts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(device)
    permutation = torch.argsort(ids, stable=True).to(device)
    row = case["x"][token : token + 1]
    isolated = {
        "packed": row.expand(slots, -1).contiguous(),
        "x": row.clone().contiguous(),
        "weights": case["weights"][token : token + 1].clone(),
        "permutation": permutation,
        "offsets": offsets,
        "w_up": case["w_up"],
        "w_down": case["w_down"],
        "ids": ids.reshape(1, slots),
    }
    if "w_sh1" in case:
        isolated["w_sh1"] = case["w_sh1"]
        isolated["w_sh2"] = case["w_sh2"]
    return isolated


def relocated(case, order):
    """The same workload with its token rows moved by `order`."""
    device = case["x"].device
    device_order = order.to(case["x"].device)
    payload = case["x"].index_select(0, device_order)
    ids = case["ids"].index_select(0, order)
    weights = case["weights"].index_select(0, device_order)
    experts = case["w_up"].shape[0]
    permutation, offsets, packed = dispatch_tokens(payload.detach().cpu().float(), ids, experts)
    moved = dict(case)
    moved.update(
        {
            "packed": packed.to(device=device, dtype=case["x"].dtype).contiguous(),
            "x": payload.contiguous(),
            "weights": weights.contiguous(),
            "permutation": permutation.to(device=device),
            "offsets": offsets.to(device=device),
            "ids": ids,
        }
    )
    return moved


def _assert_close(actual, expected, limit, label):
    error = relative_l2(actual, expected)
    assert error <= limit, f"{label}: relative L2 {error:.3e} > {limit:.3e}"


def run_core_assertions(provider, device, geometry=None, *, concentrated=False, seed=434):
    """FP64 qualification, batch invariance, relocation and repeatability."""
    geometry = dict(CORE if geometry is None else geometry)
    width = geometry["width"]
    case = build_case(17, device, seed=seed, concentrated=concentrated, **geometry)
    reference = build_case(17, device, seed=seed, concentrated=concentrated, **geometry)
    cotangent = cotangent_for(case, seed=seed + 1)

    out, grads = run(provider, case, cotangent)
    assert out.dtype == torch.bfloat16 and out.shape == (17, width)
    wide, expected = exact(case, cotangent)
    _assert_close(out, wide, FORWARD_L2, "forward vs FP64")
    _assert_close(grads["packed"], expected["packed"], PAYLOAD_L2, "dPacked vs FP64")
    _assert_close(grads["w_up"], expected["w_up"], WEIGHT_L2, "dW_up vs FP64")
    _assert_close(grads["w_down"], expected["w_down"], WEIGHT_L2, "dW_down vs FP64")
    _assert_close(grads["weights"], expected["weights"], ROUTE_L2, "dWeights vs FP64")
    if "w_sh1" in case:
        _assert_close(grads["x"], expected["x"], PAYLOAD_L2, "dX vs FP64")
        _assert_close(grads["w_sh1"], expected["w_sh1"], WEIGHT_L2, "dW_sh1 vs FP64")
        _assert_close(grads["w_sh2"], expected["w_sh2"], WEIGHT_L2, "dW_sh2 vs FP64")

    # A token's rows must not depend on the batch: same content, same bits.
    inverse = torch.argsort(case["permutation"])
    for token in sorted({0, 5, 16}):
        single = isolated_token(case, token)
        one_out, one_grads = run(provider, single, cotangent[token : token + 1])
        bits(out[token], one_out[0])
        _assert_close(
            one_grads["weights"],
            expected["weights"][token : token + 1],
            ROUTE_L2,
            "isolated dWeights",
        )
        bits(grads["weights"][token], one_grads["weights"][0])
        if "w_sh1" in case:
            bits(grads["x"][token], one_grads["x"][0])
        slot_order = torch.argsort(case["ids"][token]).to(case["packed"].device)
        for slot in range(single["permutation"].numel()):
            full_row = inverse[token * slot_order.numel() + slot]
            isolated_row = slot_order[slot]
            bits(grads["packed"][full_row], one_grads["packed"][isolated_row])

    # Row relocation: the same rows appear in different tiles, same bits.
    order = torch.tensor([16, 3, 11, 0, 7, 14, 2, 9, 5, 12, 1, 15, 8, 4, 13, 6, 10])
    moved_case = relocated(case, order)
    device_order = order.to(cotangent.device)
    moved_out, moved_grads = run(provider, moved_case, cotangent.index_select(0, device_order))
    bits(out.index_select(0, device_order), moved_out)
    bits(grads["weights"].index_select(0, device_order), moved_grads["weights"])
    if "w_sh1" in case:
        bits(grads["x"].index_select(0, device_order), moved_grads["x"])
    moved_inverse = torch.argsort(moved_case["permutation"])
    for moved_token in (0, 5, 11):
        original = int(order[moved_token])
        for slot in range(SLOTS):
            bits(
                grads["packed"][inverse[original * SLOTS + slot]],
                moved_grads["packed"][moved_inverse[moved_token * SLOTS + slot]],
            )

    # Repeatability: an identical call reproduces every bit.
    repeat_out, repeat_grads = run(provider, reference, cotangent)
    bits(out, repeat_out)
    for name in gradient_names(case):
        bits(grads[name], repeat_grads[name])

    return out, grads


def cross_provider_report(provider, other, device, geometry=None, limit=CROSS_PROVIDER_L2):
    """Relative L2 between two providers of the same contract (no bitwise claim)."""
    geometry = dict(CORE if geometry is None else geometry)
    case = build_case(23, device, **geometry)
    cotangent = cotangent_for(case, seed=11)
    out, grads = run(provider, case, cotangent)
    other_out, other_grads = run(other, case, cotangent)
    report = {"forward": relative_l2(other_out, out)}
    for name in gradient_names(case):
        report[name] = relative_l2(other_grads[name], grads[name])
    for label, error in report.items():
        assert error <= limit, f"cross-provider {label}: relative L2 {error:.3e} > {limit:.3e}"
    return report
