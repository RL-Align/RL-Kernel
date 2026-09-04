# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-5 (#64) shared_expert_mlp: bit-wise alignment against the FP32 oracle.

Every comparison is byte-equality (sha256 over raw little-endian bytes) with
the oracle executed on the same device, per the P5 start-kit acceptance rules.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe import fixtures, oracle
from rl_engine.moe.contract import SharedBatch, tensor_sha256

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PROVIDER_SPECS = {
    "cuda": "rl_engine.moe.backends.shared_expert:CudaSharedExpertProvider",
    "triton": "rl_engine.moe.backends.shared_expert:TritonSharedExpertProvider",
}


@pytest.fixture(params=sorted(PROVIDER_SPECS))
def provider(request):
    from rl_engine.moe.provider import resolve_provider

    try:
        return resolve_provider(PROVIDER_SPECS[request.param])
    except NotImplementedError as exc:
        pytest.skip(f"{request.param} backend unavailable: {exc}")


def _run_oracle(batch: SharedBatch, dy: torch.Tensor):
    y, saved = oracle.shared_expert_mlp_fwd(batch)
    dx = oracle.shared_expert_mlp_bwd(dy, batch, saved)
    return y, dx


def _run_provider(provider, batch: SharedBatch, dy: torch.Tensor):
    y, saved = provider.shared_expert_mlp_fwd(batch)
    dx = provider.shared_expert_mlp_bwd(dy, batch, saved)
    return y, dx


@requires_cuda
@pytest.mark.parametrize("case", sorted(fixtures.SHARED_CASES))
def test_shared_cases_byte_equal(provider, case):
    batch = fixtures.make_shared_batch(case).to("cuda")
    y_gold, saved_gold = oracle.shared_expert_mlp_fwd(batch)
    dy = fixtures.make_grad_output(case, tuple(y_gold.shape)).to("cuda")
    dx_gold = oracle.shared_expert_mlp_bwd(dy, batch, saved_gold)
    y, dx = _run_provider(provider, batch, dy)
    assert y.dtype == torch.bfloat16 and dx.dtype == torch.float32
    assert tensor_sha256(y) == tensor_sha256(y_gold)
    assert tensor_sha256(dx) == tensor_sha256(dx_gold)


@requires_cuda
def test_batch_padding_invariance(provider):
    """fwd(x)[t] must equal fwd(x[t:t+1]) byte-for-byte (Axis-A invariance)."""
    batch = fixtures.make_shared_batch("shared_t16").to("cuda")
    y_full, _ = provider.shared_expert_mlp_fwd(batch)
    for t in range(batch.x.shape[0]):
        row_batch = SharedBatch(
            x=batch.x[t : t + 1].contiguous(),
            w_fc1=batch.w_fc1,
            w_fc2=batch.w_fc2,
        )
        y_row, _ = provider.shared_expert_mlp_fwd(row_batch)
        assert tensor_sha256(y_row) == tensor_sha256(y_full[t : t + 1]), f"row {t} diverged"


@requires_cuda
def test_frozen_weights_no_dw(provider):
    """Backward returns only dX; the shared base weights stay frozen."""
    batch = fixtures.make_shared_batch("shared_t16").to("cuda")
    w1_before = tensor_sha256(batch.w_fc1)
    w2_before = tensor_sha256(batch.w_fc2)
    y, saved = provider.shared_expert_mlp_fwd(batch)
    dy = fixtures.make_grad_output("shared_t16", tuple(y.shape)).to("cuda")
    dx = provider.shared_expert_mlp_bwd(dy, batch, saved)
    assert dx.shape == batch.x.shape and dx.dtype == torch.float32
    assert batch.w_fc1.grad is None and batch.w_fc2.grad is None
    assert not batch.w_fc1.requires_grad and not batch.w_fc2.requires_grad
    assert tensor_sha256(batch.w_fc1) == w1_before
    assert tensor_sha256(batch.w_fc2) == w2_before


@requires_cuda
def test_shared_output_independent_of_routed(provider):
    """Shared output is not premixed with the routed path (boundary fixture).

    Running the full routed pipeline (any p_s, any expert batch) between two
    shared calls must not change a single byte of the shared output, and the
    shared output must equal the standalone oracle result (no p_s applied).
    """
    shared = fixtures.make_shared_batch("shared_t16").to("cuda")
    y_gold, _ = oracle.shared_expert_mlp_fwd(shared)
    y_before, _ = provider.shared_expert_mlp_fwd(shared)

    routed = fixtures.make_expert_batch("base_plus_lora").to("cuda")
    y_routed, saved_routed = oracle.routed_expert_forward(routed, ops=provider)
    dy_routed = fixtures.make_grad_output("base_plus_lora", tuple(y_routed.shape)).to("cuda")
    oracle.routed_expert_backward(routed, saved_routed, dy_routed, ops=provider)

    y_after, _ = provider.shared_expert_mlp_fwd(shared)
    assert tensor_sha256(y_before) == tensor_sha256(y_gold)
    assert tensor_sha256(y_after) == tensor_sha256(y_gold)
    assert y_after.data_ptr() != shared.x.data_ptr()


@requires_cuda
@pytest.mark.parametrize("shape", [(16, 128, 64), (256, 1024, 512)])
def test_cuda_triton_byte_equal(shape):
    """The two backends agree with each other bit-for-bit.

    The larger shape samples enough values to expose rare transcendental
    1-ulp divergences that survive the BF16 round (caught once at T=256).
    """
    from rl_engine.moe.provider import resolve_provider

    providers = []
    for spec in PROVIDER_SPECS.values():
        try:
            providers.append(resolve_provider(spec))
        except NotImplementedError as exc:
            pytest.skip(f"backend unavailable: {exc}")
    t, hidden, ffn = shape
    gen = torch.Generator(device="cpu").manual_seed(hash(shape) % (2**31))
    batch = SharedBatch(
        x=torch.randn(t, hidden, generator=gen).to(torch.bfloat16).cuda(),
        w_fc1=(torch.randn(2 * ffn, hidden, generator=gen) / hidden**0.5).to(torch.bfloat16).cuda(),
        w_fc2=(torch.randn(hidden, ffn, generator=gen) / ffn**0.5).to(torch.bfloat16).cuda(),
    )
    dy = torch.randn(t, hidden, generator=gen).to(torch.bfloat16).cuda()
    results = [_run_provider(p, batch, dy) for p in providers]
    (y_a, dx_a), (y_b, dx_b) = results
    assert tensor_sha256(y_a) == tensor_sha256(y_b)
    assert tensor_sha256(dx_a) == tensor_sha256(dx_b)


@requires_cuda
def test_fail_closed_on_cpu_input(provider):
    batch = fixtures.make_shared_batch("shared_t1")  # stays on CPU
    with pytest.raises(NotImplementedError):
        provider.shared_expert_mlp_fwd(batch)
