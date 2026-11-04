# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""SM90 experts: arithmetic pinning, contract assertions, registry and graphs.

Three independent kinds of evidence:

* `test_matches_det_gemm_bitwise` and the two `*_matches_det_gemm*` tests pin the
  kernel to the repository-wide GEMM arithmetic by rebuilding the same output
  through `_C.det_gemm_*` and comparing bits;
* `test_core` runs the shared provider assertions (FP64 qualification, batch
  invariance, row relocation, repeatability) for both providers;
* the remaining cases cover provider selection, graph capture, autocast and
  geometry rejection.
"""

from __future__ import annotations

import pytest
import torch

from tests.backends import nemotron_experts_cases as cases

pytestmark = [
    pytest.mark.cuda_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_SM90 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="NVIDIA SM90 required",
)


def _native_compiled() -> bool:
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    return (
        _EXT_AVAILABLE
        and hasattr(_C, "nemotron_experts_forward")
        and bool(_C.nemotron_experts_compiled())
    )


def _provider(name):
    if name == "native":
        if not _native_compiled():
            pytest.skip("native SM90 experts extension is not built")
        from rl_engine.backends.cuda.moe.nemotron_experts import NemotronExpertsOp

        return NemotronExpertsOp()
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.moe.nemotron_experts import TritonNemotronExpertsOp

    return TritonNemotronExpertsOp()


def _segments(case):
    spans = case["offsets"].tolist()
    for expert in range(case["w_up"].shape[0]):
        lo, hi = spans[expert], spans[expert + 1]
        if lo != hi:
            yield expert, lo, hi


@_SM90
@pytest.mark.parametrize("tokens", [1024, 4096])
def test_matches_det_gemm_bitwise(tokens):
    """Grouped up/down projections are the det_gemm tile tree, bit for bit."""
    if not _native_compiled():
        pytest.skip("native SM90 experts extension is not built")
    from rl_engine.backends.extension import _C

    case = cases.build_case(tokens, "cuda", **cases.REAL)
    _, up, z, down, _, shared_up, _, _ = _C.nemotron_experts_forward(
        case["packed"],
        case["x"],
        case["weights"],
        case["permutation"],
        case["offsets"],
        case["w_up"],
        case["w_down"],
        case["w_sh1"],
        case["w_sh2"],
    )
    checked = 0
    for expert, lo, hi in _segments(case):
        packed = case["packed"][lo:hi].contiguous()
        expected = _C.det_gemm_fwd_rhs_transposed(packed, case["w_up"][expert].contiguous())
        cases.bits(up[lo:hi], expected)
        # The down projection stores the BF16 tree value widened to FP32, and the
        # up epilogue materializes its relu-square operand.
        gate = torch.relu(up[lo:hi].float()).square().to(torch.bfloat16)
        cases.bits(z[lo:hi], gate)
        expected_down = _C.det_gemm_fwd_rhs_transposed(gate, case["w_down"][expert].contiguous())
        assert torch.equal(down[lo:hi], expected_down.float())
        checked += 1
    assert checked >= 2
    # The shared expert is the same kernel with one segment over [0, tokens).
    cases.bits(shared_up, _C.det_gemm_fwd_rhs_transposed(case["x"], case["w_sh1"]))


def _torch_backward_chain(case, up, grad_out):
    """dY, dZ and dU rebuilt through det_gemm on the provider's own `up`."""
    from rl_engine.backends.extension import _C

    original = case["permutation"]
    weight = case["weights"].reshape(-1)[original]
    dy = (grad_out.float()[original // 6] * weight[:, None]).to(torch.bfloat16)
    chain = {}
    for expert, lo, hi in _segments(case):
        dz = _C.det_gemm_fwd(dy[lo:hi].contiguous(), case["w_down"][expert].contiguous())
        scale = 2.0 * torch.relu(up[lo:hi].float())
        chain[expert] = (lo, hi, dy, dz, (dz.float() * scale).to(torch.bfloat16))
    return dy, chain


@_SM90
def test_dpacked_matches_det_gemm_bitwise():
    """dPacked exercises the transposed-weights tile and the relu-square VJP."""
    if not _native_compiled():
        pytest.skip("native SM90 experts extension is not built")
    from rl_engine.backends.extension import _C

    provider = _provider("native")
    case = cases.build_case(1024, "cuda", **cases.REAL)
    grad_out = cases.cotangent_for(case, seed=3)
    packed = case["packed"].detach().clone().requires_grad_(True)
    out = provider(
        packed,
        case["x"],
        case["weights"],
        case["permutation"],
        case["offsets"],
        case["w_up"],
        case["w_down"],
        case["w_sh1"],
        case["w_sh2"],
    )
    d_packed = torch.autograd.grad(out, packed, grad_out)[0]
    _, up, _, _, _, _, _, _ = _C.nemotron_experts_forward(
        case["packed"],
        case["x"],
        case["weights"],
        case["permutation"],
        case["offsets"],
        case["w_up"],
        case["w_down"],
        case["w_sh1"],
        case["w_sh2"],
    )
    assert out.shape == (1024, cases.REAL["width"])
    _, chain = _torch_backward_chain(case, up, grad_out)
    for expert, (lo, hi, _, _, du) in chain.items():
        expected = _C.det_gemm_fwd(du, case["w_up"][expert].contiguous())
        cases.bits(d_packed[lo:hi], expected)


@_SM90
@pytest.mark.parametrize("tokens", [32, 64, 1024])
def test_weight_gradients_match_det_gemm(tokens):
    """dW: transposed operand pair and per-expert segments, leaf aligned.

    Six experts absorb every route, so each segment holds exactly `tokens` rows.
    `det_gemm` keeps its SM90 tile tree only while the reduction length is a
    multiple of the 32-wide leaf - which `tokens` is - so the comparison is
    bitwise. An unaligned segment falls back to `det_gemm`'s scalar kernel, whose
    in-leaf order differs from mma, and is covered by the FP64 qualification
    instead.
    """
    if not _native_compiled():
        pytest.skip("native SM90 experts extension is not built")
    from rl_engine.backends.extension import _C

    geometry = dict(cases.REAL)
    geometry["experts"] = 6
    case = cases.build_case(tokens, "cuda", **geometry)
    provider = _provider("native")
    grad_out = cases.cotangent_for(case, seed=5)
    leaves = {
        name: case[name].detach().clone().requires_grad_(True)
        for name in ("packed", "w_up", "w_down")
    }
    out = provider(
        leaves["packed"],
        case["x"],
        case["weights"],
        case["permutation"],
        case["offsets"],
        leaves["w_up"],
        leaves["w_down"],
        case["w_sh1"],
        case["w_sh2"],
    )
    d_w_up, d_w_down = torch.autograd.grad(out, (leaves["w_up"], leaves["w_down"]), grad_out)
    _, up, _, _, _, _, _, _ = _C.nemotron_experts_forward(
        case["packed"],
        case["x"],
        case["weights"],
        case["permutation"],
        case["offsets"],
        case["w_up"],
        case["w_down"],
        case["w_sh1"],
        case["w_sh2"],
    )
    _, chain = _torch_backward_chain(case, up, grad_out)
    assert len(chain) == 6
    for expert, (lo, hi, dy, _, du) in chain.items():
        assert hi - lo == tokens and tokens % 32 == 0
        cases.bits(
            d_w_up[expert],
            _C.det_gemm_db_transposed(du, case["packed"][lo:hi].contiguous()).t(),
        )
        # dW_down reduces against the gated down-projection operand.
        gate = torch.relu(up[lo:hi].float()).square().to(torch.bfloat16)
        cases.bits(
            d_w_down[expert],
            _C.det_gemm_db_transposed(dy[lo:hi].contiguous(), gate).t(),
        )


@_SM90
@pytest.mark.parametrize(
    "geometry, tokens",
    [
        pytest.param(dict(cases.CORE), 17, id="core-unaligned"),
        pytest.param(dict(cases.CORE), 129, id="core-tiles"),
        pytest.param(dict(cases.REAL), 64, id="nemotron-real"),
    ],
)
def test_instruction_paths_agree_bitwise(monkeypatch, geometry, tokens):
    """The wgmma path and the mma.sync path are two spellings of one contract.

    Same leaf width, same leaf rounding, same midpoint tree, so every output and
    every gradient must match bit for bit on both.  `tokens=17` and `tokens=129`
    give per-expert segment lengths that are not multiples of the 32-wide leaf,
    which is where the weight gradient pads its last reduction tile, and the
    Nemotron geometry exercises truncated 128-wide column tiles (`N=1856`).
    """
    if not _native_compiled():
        pytest.skip("native SM90 experts extension is not built")
    from rl_engine.backends.extension import _C

    case = cases.build_case(tokens, "cuda", **geometry)
    cotangent = cases.cotangent_for(case, seed=11)

    # Resolve a fresh provider inside each environment scope, and check that the
    # operator really reports the path the test thinks it is exercising: a cached
    # selection would turn this into a self-comparison that can never fail.
    monkeypatch.setenv("RL_KERNEL_NEMOTRON_EXPERTS_BACKEND", "sm90_v1")
    assert _C.nemotron_experts_preferred_path() == "sm90_v1"
    reference, reference_grads = cases.run(_provider("native"), case, cotangent)

    monkeypatch.setenv("RL_KERNEL_NEMOTRON_EXPERTS_BACKEND", "sm90_v2")
    assert _C.nemotron_experts_preferred_path() == "sm90_v2"
    actual, actual_grads = cases.run(_provider("native"), case, cotangent)

    cases.bits(actual, reference)
    for label in cases.gradient_names(case):
        cases.bits(actual_grads[label], reference_grads[label])


@_SM90
@pytest.mark.parametrize("name", ["native", "triton"])
@pytest.mark.parametrize("concentrated", [False, True])
def test_core(name, concentrated):
    cases.run_core_assertions(_provider(name), "cuda", concentrated=concentrated)


@_SM90
@pytest.mark.parametrize("name", ["native", "triton"])
def test_contract_geometry(name):
    """The documented Nemotron Nano shape, including the shared expert."""
    cases.run_core_assertions(_provider(name), "cuda", geometry=cases.REAL)


@_SM90
def test_provider_cross_check():
    cases.cross_provider_report(_provider("native"), _provider("triton"), "cuda")


_SMALL = dict(cases.CORE)


def _small_case(tokens=17, **overrides):
    geometry = dict(_SMALL)
    geometry.update(overrides)
    return cases.build_case(tokens, "cuda", **geometry)


@_SM90
def test_provider_selection_prefers_native(monkeypatch):
    if not _native_compiled():
        pytest.skip("native SM90 experts extension is not built")
    from rl_engine.backends.cuda.moe import nemotron_experts as native
    from rl_engine.backends.shared.triton.moe.nemotron_experts import TritonNemotronExpertsOp
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("moe_experts_combine", device="cuda")
    assert isinstance(op, native.NemotronExpertsOp)
    assert op.provider_id == "rlkernel.nemotron_experts.sm90.v1"

    class _Uncompiled:
        nemotron_experts_forward = staticmethod(lambda *args, **kwargs: None)

        @staticmethod
        def nemotron_experts_compiled():
            return False

    monkeypatch.setattr(native, "_C", _Uncompiled())
    fallback = KernelRegistry().get_op("moe_experts_combine", device="cuda")
    assert isinstance(fallback, TritonNemotronExpertsOp)
    assert fallback.provider_id == "rlkernel.nemotron_experts.triton.v1"


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_unsupported_devices_fail_closed(device):
    from rl_engine.runtime.registry import KernelRegistry

    with pytest.raises(RuntimeError, match="SM90 CUDA or ROCm"):
        KernelRegistry().get_op("moe_experts_combine", device=device)


@_SM90
def test_provider_id_reports_selected_provider():
    from rl_engine.ops.moe.nemotron_experts import nemotron_experts_provider_id
    from rl_engine.runtime.registry import kernel_registry

    selected = kernel_registry.get_op("moe_experts_combine", device="cuda")
    assert nemotron_experts_provider_id(selected) in (
        "rlkernel.nemotron_experts.sm90.v1",
        "rlkernel.nemotron_experts.triton.v1",
    )
    assert nemotron_experts_provider_id() == nemotron_experts_provider_id(selected)


@_SM90
@pytest.mark.parametrize("name", ["native", "triton"])
def test_graph_replay_observes_input_updates(name):
    provider = _provider(name)
    case = _small_case(tokens=64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            cases.call(provider, case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cases.call(provider, case)
    for _ in range(3):
        case["packed"].normal_(std=0.5)
        case["x"].normal_(std=0.5)
        case["weights"].uniform_()
        case["w_up"].normal_(std=0.05)
        case["w_down"].normal_(std=0.05)
        case["w_sh1"].normal_(std=0.05)
        case["w_sh2"].normal_(std=0.05)
        graph.replay()
        expected = cases.call(provider, case)
        cases.bits(captured, expected)


@_SM90
@pytest.mark.parametrize("name", ["native", "triton"])
def test_autocast_invariance(name):
    provider = _provider(name)
    case = _small_case()
    cotangent = cases.cotangent_for(case)
    baseline, baseline_grads = cases.run(provider, case, cotangent)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual, actual_grads = cases.run(provider, case, cotangent)
    cases.bits(actual, baseline)
    for label in cases.gradient_names(case):
        cases.bits(actual_grads[label], baseline_grads[label])


@_SM90
def test_reject_bad_geometry():
    """Input validation is hardware-independent, so it is exercised once."""
    from rl_engine.ops.moe.nemotron_experts import MAX_TOKENS, nemotron_experts

    case = _small_case()

    def run(**overrides):
        arguments = dict(case)
        arguments.update(overrides)
        return nemotron_experts(
            arguments["packed"],
            arguments["x"],
            arguments["weights"],
            arguments["permutation"],
            arguments["offsets"],
            arguments["w_up"],
            arguments["w_down"],
            arguments.get("w_sh1"),
            arguments.get("w_sh2"),
        )

    with pytest.raises(ValueError, match="six rows per token"):
        run(packed=case["packed"][:-1])
    with pytest.raises(ValueError, match="weights must be"):
        run(weights=case["weights"].double())
    with pytest.raises(ValueError, match="contiguous"):
        run(weights=case["weights"].repeat(1, 2)[:, ::2])
    with pytest.raises(ValueError, match="supplied as a pair"):
        run(w_sh2=None)
    with pytest.raises(ValueError, match="permutation must cover"):
        run(permutation=case["permutation"][:-1])
    with pytest.raises(ValueError, match="experts\\+1"):
        run(offsets=case["offsets"][:-1])
    with pytest.raises(ValueError, match="at most"):
        oversized = torch.zeros((MAX_TOKENS + 1) * 6, 64, device="cuda", dtype=torch.bfloat16)
        nemotron_experts(
            oversized,
            torch.zeros(MAX_TOKENS + 1, 64, device="cuda", dtype=torch.bfloat16),
            torch.zeros(MAX_TOKENS + 1, 6, device="cuda", dtype=torch.float32),
            torch.zeros((MAX_TOKENS + 1) * 6, device="cuda", dtype=torch.int64),
            torch.zeros(129, device="cuda", dtype=torch.int64),
            torch.zeros(128, 64, 64, device="cuda", dtype=torch.bfloat16),
            torch.zeros(128, 64, 64, device="cuda", dtype=torch.bfloat16),
        )


@_SM90
@pytest.mark.parametrize("name", ["native", "triton"])
def test_empty_batch(name):
    provider = _provider(name)
    case = _small_case()
    empty = dict(case)
    for label, rows in (("packed", 0), ("x", 0), ("weights", 0), ("permutation", 0)):
        empty[label] = case[label][:rows]
    # A zero-token batch also needs a routing table that describes zero rows,
    # otherwise `offsets` still claims the 17-token workload.
    empty["offsets"] = torch.zeros_like(case["offsets"])
    leaves = {
        label: empty[label].requires_grad_(label in ("packed", "x", "weights"))
        for label in ("packed", "x", "weights")
    }
    out = provider(
        leaves["packed"],
        leaves["x"],
        leaves["weights"],
        empty["permutation"],
        empty["offsets"],
        empty["w_up"],
        empty["w_down"],
        empty["w_sh1"],
        empty["w_sh2"],
    )
    assert out.shape == (0, cases.CORE["width"]) and out.dtype == torch.bfloat16
    # The batch is empty, so backward must produce empty (or absent) gradients.
    grad_out = torch.zeros((0, cases.CORE["width"]), device="cuda", dtype=torch.bfloat16)
    grads = torch.autograd.grad(out, (leaves["x"],), grad_out, allow_unused=True)
    assert grads[0] is None or grads[0].numel() == 0
