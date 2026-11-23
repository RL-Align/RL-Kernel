# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_image_velocity_step`` (RFC #738 ``image_velocity_step``): the FP32 velocity Euler step.

Pins:

* the reference is ``xt + (sigma - sigma_next) * v`` in FP32 with one cast, and equals the
  pinned ``MiniMaxH3Scheduler.step(euler_update_form="velocity")`` bit for bit (Diffusers-gated);
* the Triton kernel equals the reference bit for bit (scalar and per-row sigma, three dtypes,
  shapes that are not a multiple of the block), so it carries no fused multiply-add, and its
  backward equals the reference's autograd;
* it is a different FP32 program from the H3 blend step (#420 ``h3_ode_step``): same value in
  real arithmetic, different bytes;
* bad inputs fail like ``h3_ode_step``; the registry resolves both backends.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.reference.flow.ode_step import NativeH3OdeStepOp
from rl_engine.reference.flow.velocity_step import NativeH3ImageVelocityStepOp

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724]
SIGMAS += [0.23200887441635132, 0.0]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _case(rows=1029, width=96, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    xt = (torch.randn(rows, width, generator=g) * 3).to(dtype).to(device)
    v = (torch.randn(rows, width, generator=g) * 2).to(dtype).to(device)
    return xt, v


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32 if t.element_size() == 4 else torch.int16)


def _sigma(i: int, device="cpu"):
    return torch.tensor(SIGMAS[i], device=device), torch.tensor(SIGMAS[i + 1], device=device)


# --- reference ---------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_reference_is_the_declared_fp32_program(dtype) -> None:
    xt, v = _case(dtype=dtype)
    s, sn = _sigma(2)
    out = NativeH3ImageVelocityStepOp()(xt, v, s, sn)
    expected = (xt.float() + (s - sn) * v.float()).to(dtype)
    assert out.dtype == dtype and torch.equal(_bits(out), _bits(expected))


def test_is_not_the_h3_blend_program() -> None:
    xt, v = _case()
    s, sn = _sigma(3)
    velocity = NativeH3ImageVelocityStepOp()(xt, v, s, sn)
    blend, _ = NativeH3OdeStepOp()(xt, v, s, sn)
    assert torch.allclose(velocity, blend, rtol=1e-5, atol=1e-5)  # same Euler step ...
    assert not torch.equal(_bits(velocity), _bits(blend))  # ... different FP32 program


@pytest.mark.parametrize(
    "sigma, sigma_next, err",
    [
        (0.5, 0.75, ValueError),  # increasing
        (0.0, 0.0, ValueError),  # sigma must be > 0
        (float("nan"), 0.0, ValueError),
        (torch.ones(1029), 0.0, ValueError),  # mixed layout
        (torch.ones(7), torch.zeros(7), ValueError),  # neither scalar nor per row
    ],
)
def test_bad_sigmas_fail_like_h3_ode_step(sigma, sigma_next, err) -> None:
    xt, v = _case()
    with pytest.raises(err):
        NativeH3ImageVelocityStepOp()(xt, v, sigma, sigma_next)
    with pytest.raises(err):
        NativeH3OdeStepOp()(xt, v, sigma, sigma_next)


def test_shape_and_dtype_errors() -> None:
    xt, v = _case()
    with pytest.raises(ValueError):
        NativeH3ImageVelocityStepOp()(xt, v[:-1], 1.0, 0.5)
    with pytest.raises(TypeError):
        NativeH3ImageVelocityStepOp()(xt.double(), v.double(), 1.0, 0.5)


def test_per_row_sigma_equals_scalar_rows() -> None:
    xt, v = _case(rows=6)
    s = torch.tensor(SIGMAS[:6])
    sn = torch.tensor(SIGMAS[1:7])
    out = NativeH3ImageVelocityStepOp()(xt, v, s, sn)
    for r in range(6):
        row = NativeH3ImageVelocityStepOp()(xt[r : r + 1], v[r : r + 1], s[r], sn[r])
        assert torch.equal(_bits(out[r : r + 1]), _bits(row))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
def test_reference_equals_the_pinned_scheduler_step(device, dtype) -> None:
    sched_mod = pytest.importorskip("diffusers.schedulers.scheduling_minimax_h3")
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    scheduler = sched_mod.MiniMaxH3Scheduler(
        shift_provided_sigmas=False, euler_update_form="velocity"
    )
    timesteps = torch.tensor([1.0 - s for s in SIGMAS[:-1]], dtype=torch.float32)
    scheduler.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]), device=device)
    sample, _ = _case(dtype=dtype, device=device)
    ours = sample.clone()
    for i, t in enumerate(scheduler.timesteps):
        _, v = _case(dtype=torch.float32, device=device, seed=10 + i)
        sample = scheduler.step(v, t, sample, return_dict=False)[0]
        s, sn = scheduler.sigmas[i], scheduler.sigmas[i + 1]
        ours = NativeH3ImageVelocityStepOp()(ours, v, s.to(device), sn.to(device))
        assert torch.equal(_bits(ours), _bits(sample)), f"step {i}"


# --- Triton ------------------------------------------------------------------------------------


def _triton():
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.flow.velocity_step import H3ImageVelocityStepTritonOp

    return H3ImageVelocityStepTritonOp()


@CUDA
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("rows", [1, 1029, 64 * 64 + 3])
def test_triton_equals_the_reference_bitwise(dtype, rows) -> None:
    op = _triton()
    xt, v = _case(rows=rows, dtype=dtype, device="cuda")
    for i in range(6):
        s, sn = _sigma(i, "cuda")
        got = op(xt, v, s, sn)
        want = NativeH3ImageVelocityStepOp()(xt, v, s, sn)
        assert torch.equal(_bits(got), _bits(want)), f"step {i}"


@CUDA
def test_triton_has_no_fused_multiply_add() -> None:
    # A fused x + d * v rounds once; the declared program rounds twice. Over a million
    # random elements the two differ somewhere, so bitwise equality proves no contraction.
    op = _triton()
    xt, v = _case(rows=12_000, device="cuda")
    s, sn = _sigma(4, "cuda")
    d = (s - sn).double()
    fused = (xt.double() + d * v.double()).float()
    got = op(xt, v, s, sn)
    assert torch.equal(_bits(got), _bits(NativeH3ImageVelocityStepOp()(xt, v, s, sn)))
    assert not torch.equal(_bits(got), _bits(fused))


@CUDA
def test_triton_per_row_sigma_and_backward() -> None:
    op = _triton()
    xt, v = _case(rows=6, device="cuda")
    s = torch.tensor(SIGMAS[:6], device="cuda")
    sn = torch.tensor(SIGMAS[1:7], device="cuda")
    assert torch.equal(_bits(op(xt, v, s, sn)), _bits(NativeH3ImageVelocityStepOp()(xt, v, s, sn)))
    a, b = xt.clone().requires_grad_(), v.clone().requires_grad_()
    c, e = xt.clone().requires_grad_(), v.clone().requires_grad_()
    g = torch.randn_like(xt)
    op(a, b, s, sn).backward(g)
    NativeH3ImageVelocityStepOp()(c, e, s, sn).backward(g)
    assert torch.equal(_bits(a.grad), _bits(c.grad))
    assert torch.equal(_bits(b.grad), _bits(e.grad))


@CUDA
def test_triton_refuses_cpu_tensors() -> None:
    op = _triton()
    xt, v = _case()
    with pytest.raises(RuntimeError, match="GPU tensor"):
        op(xt, v, 1.0, 0.5)


# --- registry ----------------------------------------------------------------------------------


def test_registry_resolves_the_reference_and_lists_triton() -> None:
    from rl_engine.runtime.registry import KernelRegistry, OpBackend
    from rl_engine.runtime.semantic_registry import OperatorRequirements

    registry = KernelRegistry()
    ids = {d.backend_id for d in registry.semantic.backend_descriptors("h3_image_velocity_step")}
    assert ids == {
        "rlkernel.triton.h3_image_velocity_step.euler_velocity_fp32.v1",
        "rlkernel.pytorch.h3_image_velocity_step.euler_velocity_fp32.v1",
    }
    assert registry._priority_map["cpu"]["h3_image_velocity_step"] == [
        OpBackend.PYTORCH_H3_IMAGE_VELOCITY_STEP
    ]
    for device in ("cuda", "rocm"):
        assert registry._priority_map[device]["h3_image_velocity_step"] == [
            OpBackend.TRITON_H3_IMAGE_VELOCITY_STEP
        ]
    session = registry.semantic.session()
    resolution = session.resolve(
        semantic_op="h3_image_velocity_step",
        requested_backend="rlkernel.pytorch.h3_image_velocity_step.euler_velocity_fp32.v1",
        target="rollout",
        requirements=OperatorRequirements(
            device="cpu",
            dtype="float32",
            topology={"world_size": 1, "tensor_parallel_size": 1, "context_parallel_size": 1},
        ),
    )
    assert resolution.trace.status == "resolved"
    instance = session.instantiate(resolution)
    assert type(instance).__name__ == "NativeH3ImageVelocityStepOp"
    assert instance(torch.zeros(4, 24), torch.ones(4, 24), 1.0, 0.5).shape == (4, 24)


def test_operator_spec_is_declared() -> None:
    from rl_engine.validation.operators.operator_specs import OP_SPECS

    spec = OP_SPECS["h3_image_velocity_step"]
    assert spec.gold_method == "forward_fp32"
    assert set(spec.candidate_paths) == {"pytorch", "triton"}
