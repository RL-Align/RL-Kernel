# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Invariance + correctness tests for P5-3 shared_grouped_lora_delta (#62).

Runs against both deterministic backends -- the CUDA path (composed from
``_C.det_gemm_*``) and the Triton path (composed from ``TritonDetGemmOp``) --
each of which must independently satisfy the invariance contract.

The torch-native provider is intentionally NOT covered here: it is the
non-deterministic benchmark baseline and fails batch-invariance by design
(48/75 probes, see benchmarks/benchmark_lora_delta.py).

Byte-equality against the oracle is asserted only for the Triton path. It keeps
one FP32 accumulator across the whole K loop and rounds once at the store, which
reproduces the oracle's ``_serial_dot``. The CUDA path reuses det_gemm's BF16
mid-split tree -- that tree exists so the result also holds under tensor
parallelism, and its per-leaf rounding costs it byte-equality.
"""

import dataclasses
import inspect

import pytest
import torch

from rl_engine.moe import fixtures, oracle
from rl_engine.moe.contract import ORACLE_PROFILE
from rl_engine.moe.shared_grouped_lora_delta_provider import (
    LoRADeltaCudaProvider,
    LoRADeltaTritonProvider,
)

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="P5-3 deterministic backends require CUDA SM80+",
)

# Each deterministic backend is validated independently.
_BACKENDS = [
    ("cuda", LoRADeltaCudaProvider),
    ("triton", LoRADeltaTritonProvider),
]

# BF16 carries 8 mantissa bits, so eps is ~7.8e-3. A backend that is merely
# accumulating in a different order stays well inside this; a transposed operand
# or a swapped return value lands at O(1).
_BF16_TOL = 1e-2

_ALPHA = 0.5

# Invariance is checked at production geometry, not at the fixture's 128x64.
# cuBLAS only switches tile/split-k strategy once K is large, so a small-K probe
# lets a non-deterministic implementation pass -- verified by mutating the Triton
# path to torch.matmul, which a 128-wide K failed to catch.
_M, _K, _N, _R = 24, 4096, 2048, 8


def _rand(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)


def _operands(m: int = _M):
    torch.manual_seed(0)
    return _rand(m, _K), _rand(_R, _K), _rand(_N, _R)


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    scale = max(want.abs().max().item(), 1e-12)
    return (got.float() - want.float()).abs().max().item() / scale


# --- contract -------------------------------------------------------------


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_returns_y_then_u_with_contract_dtypes(name, cls):
    # (y_fp32, u_bf16) -- swapping the two silently corrupts the caller whenever
    # r == N, so pin both the order and the dtypes.
    x, a, b = _operands()
    y, u = cls().shared_grouped_lora_delta_fwd(x, a, b, _ALPHA)
    assert y.shape == (_M, _N), f"{name}: y must be [M, N]"
    assert u.shape == (_M, _R), f"{name}: u must be [M, r]"
    assert y.dtype is torch.float32, f"{name}: y must be FP32 (contract D7)"
    assert u.dtype is torch.bfloat16, f"{name}: u must be BF16"


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_backward_returns_no_dw_and_leaves_inputs_untouched(name, cls):
    # Contract rule 1: base weights are frozen, so backward yields exactly
    # (dX, dA, dB) and mutates nothing in place.
    x, a, b = _operands()
    provider = cls()
    y, u = provider.shared_grouped_lora_delta_fwd(x, a, b, _ALPHA)
    dy = _rand(_M, _N)
    before = [t.clone() for t in (x, a, b, dy, u)]

    grads = provider.shared_grouped_lora_delta_bwd(dy, x, a, b, _ALPHA, u)

    assert len(grads) == 3, f"{name}: backward must return exactly (dX, dA, dB)"
    for label, g, want in zip(("dX", "dA", "dB"), grads, (x, a, b), strict=True):
        assert g.shape == want.shape, f"{name}: {label} shape must mirror its operand"
        assert g.dtype is torch.float32, f"{name}: {label} must be FP32 (contract D7)"
    for label, now, orig in zip(
        ("x", "a", "b", "dy", "u"), (x, a, b, dy, u), before, strict=True
    ):
        assert torch.equal(now, orig), f"{name}: backward mutated {label} in place"


# --- invariance -----------------------------------------------------------


@pytest.mark.parametrize("name,cls", _BACKENDS)
@pytest.mark.parametrize("seed", [0, 1, 2026])
def test_forward_batch_invariance(name, cls, seed):
    # A row's output must not change when other rows join the batch. Sweeping
    # sub-batch sizes and seeds matters: a violation only becomes visible when a
    # value lands near a BF16 rounding boundary, so a single probe can pass on a
    # non-deterministic implementation.
    torch.manual_seed(seed)
    full = _rand(64, _K)
    a, b = _rand(_R, _K), _rand(_N, _R)

    provider = cls()
    y_full, u_full = provider.shared_grouped_lora_delta_fwd(full, a, b, _ALPHA)

    for m in (1, 2, 3, 8, 17):
        y_sub, u_sub = provider.shared_grouped_lora_delta_fwd(full[:m], a, b, _ALPHA)
        assert torch.equal(u_sub, u_full[:m]), f"{name}: u broke at M={m} (seed {seed})"
        assert torch.equal(y_sub, y_full[:m]), f"{name}: y broke at M={m} (seed {seed})"


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_forward_chunked_prefill(name, cls):
    # Splitting M then concatenating must match the full call bitwise.
    x, a, b = _operands(256)
    provider = cls()
    full, _ = provider.shared_grouped_lora_delta_fwd(x, a, b, _ALPHA)
    chunked = torch.cat(
        [
            provider.shared_grouped_lora_delta_fwd(x[:100], a, b, _ALPHA)[0],
            provider.shared_grouped_lora_delta_fwd(x[100:], a, b, _ALPHA)[0],
        ],
        dim=0,
    )
    assert torch.equal(full, chunked), f"{name}: chunked prefill broke invariance"


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_forward_padding_invariance(name, cls):
    # Padding rows must not affect the valid rows' output.
    x, a, b = _operands(100)
    provider = cls()
    base, _ = provider.shared_grouped_lora_delta_fwd(x, a, b, _ALPHA)
    padded, _ = provider.shared_grouped_lora_delta_fwd(
        torch.cat([x, _rand(28, _K)], dim=0), a, b, _ALPHA
    )
    assert torch.equal(base, padded[:100]), f"{name}: padding changed valid-row output"


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_backward_dx_batch_invariance(name, cls):
    # dX is per-row, so it must be invariant to the surrounding batch.
    # dA and dB reduce over every row and are deliberately not checked this way.
    a, b = _operands()[1:]
    row = _rand(1, _K)
    big = _rand(256, _K)
    big[0] = row[0]

    provider = cls()
    _, u1 = provider.shared_grouped_lora_delta_fwd(row, a, b, _ALPHA)
    _, uN = provider.shared_grouped_lora_delta_fwd(big, a, b, _ALPHA)
    dy_big = _rand(256, _N)

    dx1, _, _ = provider.shared_grouped_lora_delta_bwd(
        dy_big[:1], row, a, b, _ALPHA, u1
    )
    dxN, _, _ = provider.shared_grouped_lora_delta_bwd(dy_big, big, a, b, _ALPHA, uN)
    assert torch.equal(dx1[0], dxN[0]), f"{name}: dX batch-invariance broken"


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_repeated_calls_are_bitwise_stable(name, cls):
    # Run-to-run determinism: identical inputs must give identical bits.
    x, a, b = _operands()
    provider = cls()
    first = provider.shared_grouped_lora_delta_fwd(x, a, b, _ALPHA)[0]
    for _ in range(5):
        assert torch.equal(
            first, provider.shared_grouped_lora_delta_fwd(x, a, b, _ALPHA)[0]
        ), f"{name}: repeated calls diverged"


# --- acceptance criteria from the P5-3 brief ------------------------------


@pytest.mark.parametrize("name,cls", _BACKENDS)
@pytest.mark.parametrize("insertion", ["fc1", "fc2"])
def test_both_insertion_points(name, cls, insertion):
    # The same operator is reused at two points: once after the packed gate/up
    # projection (a1/b1) and once after the down projection (a2/b2). Their
    # shapes differ, so both need a fixture.
    batch = fixtures.make_expert_batch("base_plus_lora").to(DEV)
    lora = batch.lora
    if insertion == "fc1":
        x, a, b = batch.x, lora.a1, lora.b1  # [M, hidden] -> [M, 2*ffn]
    else:
        # fc2 consumes the post-SwiGLU activation, so synthesise [M, ffn].
        torch.manual_seed(7)
        x = _rand(batch.x.shape[0], batch.ffn)
        a, b = lora.a2, lora.b2  # [M, ffn] -> [M, hidden]

    y_ref, u_ref = oracle.shared_grouped_lora_delta_fwd(x, a, b, lora.alpha)
    provider = cls()
    y, u = provider.shared_grouped_lora_delta_fwd(x, a, b, lora.alpha)

    assert y.shape == y_ref.shape, f"{name}/{insertion}: y shape mismatch"
    assert u.shape == u_ref.shape, f"{name}/{insertion}: u shape mismatch"
    assert _rel(y, y_ref) < _BF16_TOL, f"{name}/{insertion}: y diverged from oracle"


@pytest.mark.parametrize("name,cls", _BACKENDS)
@pytest.mark.parametrize("case", ["lora_only", "base_plus_lora"])
def test_lora_grads_are_finite_and_nonzero(name, cls, case):
    # A silently-zero gradient still satisfies every shape and dtype check, so
    # assert the adapters actually receive signal.
    batch = fixtures.make_expert_batch(case).to(DEV)
    lora = batch.lora
    x, a, b, alpha = batch.x, lora.a1, lora.b1, lora.alpha

    provider = cls()
    y, u = provider.shared_grouped_lora_delta_fwd(x, a, b, alpha)
    dy = fixtures.make_grad_output(case, tuple(y.shape)).to(DEV)
    grads = provider.shared_grouped_lora_delta_bwd(dy, x, a, b, alpha, u)

    for label, g in zip(("dX", "dA", "dB"), grads, strict=True):
        assert torch.isfinite(g).all(), f"{name}/{case}: {label} has non-finite entries"
        assert g.abs().sum() > 0, f"{name}/{case}: {label} is entirely zero"


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_base_weights_get_no_gradient(name, cls):
    # LoRA-only fine-tuning: the frozen base weights must come back untouched
    # and must not acquire a gradient anywhere in the pipeline.
    batch = fixtures.make_expert_batch("base_plus_lora").to(DEV)
    w1_before = batch.w1.codes.clone()
    w2_before = batch.w2.codes.clone()

    provider = cls()
    y, saved = oracle.routed_expert_forward(batch, ops=provider)
    dy = fixtures.make_grad_output("base_plus_lora", tuple(y.shape)).to(DEV)
    grads = oracle.routed_expert_backward(batch, saved, dy, ops=provider)

    assert torch.equal(batch.w1.codes, w1_before), f"{name}: w1 packed bytes mutated"
    assert torch.equal(batch.w2.codes, w2_before), f"{name}: w2 packed bytes mutated"
    assert batch.w1.codes.grad is None, f"{name}: w1 acquired a gradient"
    assert batch.w2.codes.grad is None, f"{name}: w2 acquired a gradient"
    for key in grads:
        assert "w" not in key.lower(), f"{name}: backward produced a base-weight grad {key!r}"


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_tampering_packed_base_bytes_changes_the_output(name, cls):
    # The brief forbids unpacking the base weight on the LoRA path: if training
    # unpacked to BF16 while serving keeps Marlin-packed weights, the rounding
    # points diverge. Flipping one packed byte must therefore show up in the
    # output -- if it does not, something is reading a stale unpacked copy.
    #
    # ExpertBatch.validate() fingerprints the packed bytes, so the tampered
    # batch has to re-derive its fingerprint before it can be used at all.
    provider = cls()

    # This operator cannot unpack anything: its signature never receives w1/w2.
    params = inspect.signature(provider.shared_grouped_lora_delta_fwd).parameters
    assert set(params) == {"x", "a", "b", "alpha"}, (
        f"{name}: the LoRA path must not take base weights, got {list(params)}"
    )

    batch = fixtures.make_expert_batch("base_plus_lora").to(DEV)
    y_clean, _ = oracle.routed_expert_forward(batch, ops=provider)
    # Control: an untampered rerun is bitwise identical, so the assertion below
    # is not passing on incidental noise.
    assert torch.equal(
        y_clean, oracle.routed_expert_forward(batch, ops=provider)[0]
    ), f"{name}: pipeline is not reproducible, the tamper check would be vacuous"

    tampered = fixtures.make_expert_batch("base_plus_lora").to(DEV)
    tampered.w1.codes[0, 0, 0] ^= 0xFF
    tampered = dataclasses.replace(
        tampered, weight_fingerprint=tampered.compute_weight_fingerprint()
    )
    tampered.validate()
    y_tampered, _ = oracle.routed_expert_forward(tampered, ops=provider)

    assert not torch.equal(y_clean, y_tampered), (
        f"{name}: flipping a packed base byte left the output unchanged -- "
        "the base weight is not being read in packed form"
    )


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_fingerprint_rejects_tampered_batch(name, cls):
    # The other half of the same contract: an unannounced edit to the packed
    # bytes must fail closed rather than silently produce a different answer.
    del cls
    batch = fixtures.make_expert_batch("base_plus_lora").to(DEV)
    batch.w1.codes[0, 0, 0] ^= 0xFF
    with pytest.raises(ValueError, match="fingerprint"):
        batch.validate()


# --- correctness ----------------------------------------------------------


@pytest.mark.parametrize("name,cls", _BACKENDS)
@pytest.mark.parametrize("case", ["lora_only", "base_plus_lora", "uneven_experts"])
def test_matches_oracle_within_bf16_tolerance(name, cls, case):
    # Every backend must agree with the oracle to BF16 precision. A larger gap
    # means a wrong operand or transpose, not a different reduction order.
    batch = fixtures.make_expert_batch(case).to(DEV)
    lora = batch.lora
    x, a, b, alpha = batch.x, lora.a1, lora.b1, lora.alpha

    y_ref, u_ref = oracle.shared_grouped_lora_delta_fwd(x, a, b, alpha)
    dy = fixtures.make_grad_output(case, tuple(y_ref.shape)).to(DEV)
    dx_ref, da_ref, db_ref = oracle.shared_grouped_lora_delta_bwd(
        dy, x, a, b, alpha, u_ref
    )

    provider = cls()
    y, u = provider.shared_grouped_lora_delta_fwd(x, a, b, alpha)
    dx, da, db = provider.shared_grouped_lora_delta_bwd(dy, x, a, b, alpha, u)

    for label, got, want in (
        ("y", y, y_ref),
        ("u", u, u_ref),
        ("dX", dx, dx_ref),
        ("dA", da, da_ref),
        ("dB", db, db_ref),
    ):
        err = _rel(got, want)
        assert err < _BF16_TOL, f"{name}/{case}: {label} rel err {err:.2e} exceeds BF16 noise"


@pytest.mark.parametrize("case", ["lora_only", "base_plus_lora", "uneven_experts"])
def test_triton_is_byte_equal_at_fixture_geometry(case):
    # Byte equality is what check_p5.py measures (sha256, zero tolerance), and
    # it holds at the fixture's K=128/64. It is NOT a general property of this
    # backend -- see test_triton_diverges_from_oracle_at_production_width -- so
    # the provider deliberately does not declare ORACLE_PROFILE.
    assert LoRADeltaTritonProvider.numeric_profile != ORACLE_PROFILE

    batch = fixtures.make_expert_batch(case).to(DEV)
    lora = batch.lora
    x, a, b, alpha = batch.x, lora.a1, lora.b1, lora.alpha

    y_ref, u_ref = oracle.shared_grouped_lora_delta_fwd(x, a, b, alpha)
    dy = fixtures.make_grad_output(case, tuple(y_ref.shape)).to(DEV)
    dx_ref, da_ref, db_ref = oracle.shared_grouped_lora_delta_bwd(
        dy, x, a, b, alpha, u_ref
    )

    provider = LoRADeltaTritonProvider()
    y, u = provider.shared_grouped_lora_delta_fwd(x, a, b, alpha)
    dx, da, db = provider.shared_grouped_lora_delta_bwd(dy, x, a, b, alpha, u)

    for label, got, want in (
        ("y", y, y_ref),
        ("u", u, u_ref),
        ("dX", dx, dx_ref),
        ("dA", da, da_ref),
        ("dB", db, db_ref),
    ):
        assert torch.equal(got, want), f"triton/{case}: {label} is not byte-equal to the oracle"


def test_triton_diverges_from_oracle_at_production_width():
    """Pin down the limit of the byte-equality above.

    The oracle accumulates one product at a time; the Triton op accumulates one
    BLOCK_K=32 tile at a time. Both keep an FP32 accumulator, so the two
    parenthesisations agree until a value lands on a rounding boundary between
    them -- which the fixture's K=128 never reaches but production width does.

    The point of this test is that the divergence is bounded and harmless, not
    that it is absent: it asserts the gap stays at FP32 epsilon. Anyone who
    later makes this path genuinely byte-equal at all K should delete this test
    and declare ORACLE_PROFILE -- the failure is the signal to do so.
    """
    provider = LoRADeltaTritonProvider()
    byte_equal = []

    # Sweep seeds rather than pinning one: whether a given input diverges is a
    # matter of where its values fall relative to a rounding boundary, and it
    # also depends on which instruction tl.dot lowers to (MMA on Ampere, WGMMA
    # on Hopper). The claim being tested is "not byte-equal in general", so a
    # single seed would make this test hostage to luck and to the GPU.
    for seed in (0, 1, 2, 42, 2026):
        torch.manual_seed(seed)
        x = _rand(24, _K)
        a, b = _rand(_R, _K), _rand(_N, _R)

        y_ref, u_ref = oracle.shared_grouped_lora_delta_fwd(x, a, b, _ALPHA)
        y, u = provider.shared_grouped_lora_delta_fwd(x, a, b, _ALPHA)

        # Bounded: FP32 eps is ~1.2e-7, so an order of magnitude above it leaves
        # ample headroom for a reduction-order difference while staying far
        # below BF16 noise. This half is the one that must always hold.
        for label, got, want in (("y", y, y_ref), ("u", u, u_ref)):
            err = _rel(got, want)
            assert err < 1e-5, (
                f"triton seed={seed}: {label} rel err {err:.2e} exceeds reduction noise"
            )

        byte_equal.append(torch.equal(y, y_ref) and torch.equal(u, u_ref))

    # If every seed agrees bitwise, the divergence documented in the class
    # docstring, the numeric_profile, and provenance()['oracle_agreement'] have
    # all gone stale -- and declaring ORACLE_PROFILE becomes worth revisiting.
    assert not all(byte_equal), (
        "triton is byte-equal at K=%d for every probed seed; re-evaluate "
        "declaring ORACLE_PROFILE and update the docstring" % _K
    )


@pytest.mark.parametrize("name,cls", _BACKENDS)
def test_provenance_reports_actual_backend(name, cls):
    # P5-6 fail-closed contract: provenance must name the backend that really
    # ran, so a silent fallback cannot pass unnoticed.
    prov = cls().provenance()
    for key in ("requested_backend", "actual_backend", "numeric_profile"):
        assert key in prov, f"{name}: provenance is missing {key}"
    assert name in prov["actual_backend"], (
        f"{name}: actual_backend {prov['actual_backend']!r} does not name the backend"
    )
