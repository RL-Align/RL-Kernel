# SPDX-License-Identifier: Apache-2.0
import math
import struct
from fractions import Fraction

import pytest
import torch

from rl_engine.ops.norm.reproducible import (
    clip_grad_norm_exact_,
    exact_norm,
    integer_square_bins,
    norm_from_bins,
)


def reference_bins(values):
    bins = [0] * 770
    for value in values:
        (bits,) = struct.unpack("I", struct.pack("f", value))
        exponent = (bits >> 23) & 255
        mantissa = bits & 0x7FFFFF
        if exponent == 255:
            bins[768 + (mantissa == 0)] = 1
            continue
        if exponent:
            mantissa |= 1 << 23
        square = mantissa * mantissa
        for limb in range(3):
            bins[3 * exponent + limb] += (square >> (16 * limb)) & 65535
    return bins


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.0, -0.0],
        [3.0, -4.0],
        [2.0**-149, -(2.0**-126)],
        [2.0**127, -(2.0**100), 2.0**-149],
        [1.0, 2.0**-12, -(2.0**-24)] * 101,
    ],
)
def test_norm_matches_exact_rational_square_sum(values):
    exact = sum((Fraction(value) ** 2 for value in values), Fraction())
    assert norm_from_bins(reference_bins(values)) == math.sqrt(float(exact))


def test_nonfinite():
    assert math.isnan(norm_from_bins(reference_bins([float("nan"), float("inf")])))
    assert norm_from_bins(reference_bins([-float("inf")])) == float("inf")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="accelerator required")
def test_accelerator_bins_preserve_exact_sum_under_repartition():
    generator = torch.Generator(device="cuda").manual_seed(71)
    values = torch.randn(32769, generator=generator, device="cuda")
    values[:4] = torch.tensor([2.0**-149, -(2.0**-126), 2.0**127, 0.0], device="cuda")
    full = integer_square_bins([values])
    partitioned = integer_square_bins(list(values.split(733)))
    assert torch.equal(full, partitioned)
    assert full.cpu().tolist() == reference_bins(values.cpu().tolist())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two accelerators required")
def test_accelerator_device_context_is_restored():
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    values = torch.tensor([3.0, -4.0], device=f"cuda:{other}")
    assert norm_from_bins(integer_square_bins([values]).cpu().tolist()) == 5.0
    assert torch.cuda.current_device() == current


@pytest.mark.skipif(not torch.cuda.is_available(), reason="accelerator required")
@pytest.mark.parametrize("length", [1, 31, 32, 33, 257, 32771])
def test_accelerator_warp_bins_with_mixed_exponents_and_nonfinite(length):
    generator = torch.Generator(device="cuda").manual_seed(length)
    values = torch.randn(length, generator=generator, device="cuda")
    exponents = torch.randint(-120, 120, (length,), generator=generator, device="cuda")
    values *= torch.pow(2.0, exponents)
    if length >= 33:
        values[:16] = 0.125
        values[16:21] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, 2.0**-149], device="cuda"
        )
    actual = integer_square_bins([values]).cpu().tolist()
    assert actual == reference_bins(values.cpu().tolist())


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


def test_nvrtc_candidates_find_cuda13_wheel_layout(tmp_path):
    from rl_engine.ops.norm.reproducible import _nvrtc_candidates

    wheel = _touch(tmp_path / "cu13" / "lib" / "libnvrtc.so.13")
    _touch(tmp_path / "cu13" / "lib" / "libnvrtc.alt.so.13")
    candidates = _nvrtc_candidates(tmp_path, "13.0")
    assert candidates[0] == wheel
    assert not any(".alt." in name for name in candidates)
    assert candidates.index("libnvrtc.so.13") < candidates.index("libnvrtc.so.12")


def test_nvrtc_candidates_find_cuda12_wheel_layout(tmp_path):
    from rl_engine.ops.norm.reproducible import _nvrtc_candidates

    wheel = _touch(tmp_path / "cuda_nvrtc" / "lib" / "libnvrtc.so.12")
    candidates = _nvrtc_candidates(tmp_path, "12.8")
    assert candidates[0] == wheel
    assert candidates.index("libnvrtc.so.12") < candidates.index("libnvrtc.so.13")
    assert len(candidates) == len(set(candidates))


def test_load_nvrtc_reports_every_attempt(monkeypatch):
    from rl_engine.ops.norm import reproducible

    monkeypatch.setattr(reproducible, "_nvrtc_candidates", lambda: ["a.so", "b.so"])
    with pytest.raises(OSError, match="(?s)a.so.*b.so"):
        reproducible._load_nvrtc()


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="accelerator required")


@CUDA
def test_exact_norm_without_a_process_group_is_the_exact_local_norm():
    generator = torch.Generator(device="cuda").manual_seed(5)
    values = torch.randn(4099, generator=generator, device="cuda")
    expected = norm_from_bins(reference_bins(values.cpu().tolist()))
    assert exact_norm(values) == expected
    assert exact_norm(list(values.split(97))) == expected
    assert exact_norm([]) == 0.0


def _params(seed, shapes=((64, 32), (32,), (7, 3))):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    params = []
    for shape in shapes:
        p = torch.nn.Parameter(torch.zeros(shape, device="cuda"))
        p.grad = torch.randn(shape, generator=generator, device="cuda") * 3
        params.append(p)
    return params


@CUDA
@pytest.mark.parametrize("max_norm", [1.0, 1e6])
def test_clip_scales_like_pytorch(max_norm):
    ours, theirs = _params(9), _params(9)
    total = clip_grad_norm_exact_(ours, max_norm)
    torch_total = torch.nn.utils.clip_grad_norm_(theirs, max_norm)
    assert total == pytest.approx(float(torch_total), rel=1e-6)
    expected_coef = torch.clamp(
        torch.tensor(max_norm) / (torch.tensor(total, dtype=torch.float32) + 1e-6), max=1.0
    )
    for p, q in zip(ours, _params(9)):
        assert torch.equal(p.grad, q.grad * expected_coef.cuda())


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_gradients_keep_a_small_coefficient(dtype):
    """The FP32 coefficient multiplies the gradients, as torch's clip does. Here it is about
    8.6e-9, below FP16's smallest subnormal: cast to FP16 first, it would zero every gradient."""

    n = 1 << 20
    params = []
    for _ in range(3):
        p = torch.nn.Parameter(torch.zeros(n, device="cuda", dtype=dtype))
        p.grad = torch.full((n,), 32768.0, device="cuda", dtype=dtype)
        params.append(p)
    total = clip_grad_norm_exact_(params, 0.5)
    assert total == pytest.approx(32768.0 * (3 * n) ** 0.5, rel=1e-12)
    coef = torch.clamp(
        torch.tensor(0.5) / (torch.tensor(total, dtype=torch.float32) + 1e-6), max=1.0
    )
    assert coef.to(torch.float16) == 0  # what the old cast did to it
    want = (torch.full((n,), 32768.0, dtype=dtype).float() * coef).to(dtype).cuda()
    for p in params:
        assert p.grad.abs().min() > 0
        assert torch.equal(p.grad, want)


@CUDA
@pytest.mark.parametrize("shape", [(64, 32), (5,), ()])
def test_one_tensor_is_one_parameter(shape):
    """A bare Tensor is one parameter (torch accepts it too), not an iterable of its rows."""

    [alone] = _params(12, shapes=(shape,))
    [listed] = _params(12, shapes=(shape,))
    total = clip_grad_norm_exact_(alone, 0.5)
    assert (
        total == clip_grad_norm_exact_([listed], 0.5) == exact_norm([_params(12, (shape,))[0].grad])
    )
    assert total > 0
    assert torch.equal(alone.grad, listed.grad)


@CUDA
def test_clip_leaves_a_non_finite_norm_to_the_caller():
    params = _params(10)
    params[1].grad[0] = float("nan")
    before = [p.grad.clone() for p in params]
    assert math.isnan(clip_grad_norm_exact_(params, 1.0))
    for p, b in zip(params, before):
        assert torch.equal(p.grad.nan_to_num(), b.nan_to_num())


@CUDA
def test_counted_selects_the_norm_but_every_gradient_is_scaled():
    params = _params(11)
    only = exact_norm([params[0].grad])
    coef = torch.clamp(torch.tensor(0.5) / (torch.tensor(only) + 1e-6), max=1.0).cuda()
    expected = [p.grad * coef for p in params]
    total = clip_grad_norm_exact_(params, 0.5, counted=lambda p: p is params[0])
    assert total == only
    for p, e in zip(params, expected):
        assert torch.equal(p.grad, e)


def _two_rank_norm(rank, init, values, out):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=2)
    try:
        full = values.cuda()
        replicated = full[:5]  # held by both ranks, counted on rank 0 only
        shard = full[5:].chunk(2)[rank]
        counted = [shard, replicated] if rank == 0 else [shard]
        out[rank] = exact_norm(counted)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two accelerators required")
def test_two_ranks_give_the_single_process_norm(tmp_path):
    import torch.multiprocessing as mp

    values = torch.randn(10007, generator=torch.Generator().manual_seed(12))
    manager = mp.get_context("spawn").Manager()
    out = manager.dict()
    mp.start_processes(
        _two_rank_norm,
        args=((tmp_path / "init").as_uri(), values, out),
        nprocs=2,
        start_method="spawn",
    )
    expected = norm_from_bins(reference_bins(values.tolist()))
    assert out[0] == out[1] == expected


def _stub_megatron(monkeypatch):
    """Install stand-in ``megatron.core.optimizer.{clip_grads,optimizer}`` modules for the test."""

    import sys
    import types

    calls = []

    def original(grads_for_norm, norm_type=2, grad_stats_parallel_group=None):
        calls.append(norm_type)
        return -1.0

    clip_grads = types.ModuleType("megatron.core.optimizer.clip_grads")
    clip_grads.get_grad_norm_fp32 = original
    optimizer = types.ModuleType("megatron.core.optimizer.optimizer")
    optimizer.get_grad_norm_fp32 = original
    package = types.ModuleType("megatron.core.optimizer")
    package.clip_grads, package.optimizer = clip_grads, optimizer
    for name, module in {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.optimizer": package,
        "megatron.core.optimizer.clip_grads": clip_grads,
        "megatron.core.optimizer.optimizer": optimizer,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return clip_grads, optimizer, calls


@CUDA
def test_megatron_hook_returns_the_exact_norm(monkeypatch):
    """``install`` patches both Megatron entry points with ``exact_norm`` for the L2 norm,
    leaves other norm types to Megatron, and installs once."""

    from rl_engine.ops.norm import reproducible

    clip_grads, optimizer, calls = _stub_megatron(monkeypatch)
    reproducible.install()
    patched = clip_grads.get_grad_norm_fp32
    assert optimizer.get_grad_norm_fp32 is patched
    reproducible.install()
    assert clip_grads.get_grad_norm_fp32 is patched

    values = torch.randn(4099, generator=torch.Generator().manual_seed(5))
    grads = [values[:1000].cuda(), values[1000:].cuda()]
    expected = norm_from_bins(reference_bins(values.tolist()))
    assert patched(grads, 2) == expected
    assert patched(grads[0], 2.0) == norm_from_bins(reference_bins(values[:1000].tolist()))
    assert calls == []
    assert patched(grads, 1) == -1.0
    assert calls == [1]
