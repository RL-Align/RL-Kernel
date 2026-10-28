# SPDX-License-Identifier: Apache-2.0
"""MI300X acceptance for the real aiter overlap C4 path and mLite VJP."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from rl_engine.backends.rocm.attention.kv_compressor_c4 import AiterC4Backend
from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract, C4Error
from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.c4_helpers import byte_equal, fixture, reference

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


@pytest.fixture(scope="module")
def backend():
    return AiterC4Backend()


def op(backend, **kwargs):
    return C4Compressor(C4Contract(**kwargs), backend=backend)


def numerical_close(actual, expected):
    exponent = torch.floor(torch.log2(expected.abs().clamp_min(2.0**-126)))
    ulp = torch.exp2(exponent - 7)
    assert bool(((actual.double() - expected).abs() <= (2 * ulp + 2e-5)).all())


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 7, 16, 33])
def test_native_forward_independent_fp64_overlap_oracle(backend, groups):
    operator = op(backend)
    values, ids, ape, norm, cos, sin = fixture(groups, device="cuda")
    bound = operator.bind(values, ids, version="v1")
    output = operator(bound, ape, norm, cos, sin, version="v1").values
    expected = reference(
        bound.values.double(), ape.double(), norm.double(), bound.group_ids[:, 1], cos, sin, 1e-6
    )
    numerical_close(output, expected)
    assert backend.provenance["arch"] == "gfx942"
    assert len(bound.incomplete_groups) == 0


def test_native_forward_fixed_overlap_plan_and_cache_isolation(backend, monkeypatch):
    operator = op(backend)
    values, ids, ape, norm, cos, sin = fixture(8, device="cuda")
    captured = []
    original = backend.aiter.flydsl_fused_compress_attn

    def capture(**kwargs):
        assert kwargs["ratio"] == 4 and kwargs["overlap"] is True
        assert kwargs.get("k_split_num_waves", 1) == 1
        assert kwargs["k_per_block"] == 64
        captured.append(kwargs["plan_gpu"].clone())
        return original(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backend.aiter, "flydsl_fused_compress_attn", capture)
        output = operator(
            operator.bind(values, ids, version="v1"), ape, norm, cos, sin, version="v1"
        ).values
    assert captured and captured[0].shape == (8, 4)
    assert captured[0][:, 3].cpu().tolist() == [4, 0, 0, 0, 0, 0, 0, 0]
    assert torch.equal(captured[0][:, 2].cpu(), torch.tensor([3, 7, 11, 15, 19, 23, 27, 31]))
    expected = reference(
        values.double(), ape.double(), norm.double(), torch.arange(8), cos, sin, 1e-6
    )
    numerical_close(output, expected)


@pytest.mark.parametrize("seed", range(8))
def test_ragged_permuted_requests_padding_and_cp_ownership(backend, seed):
    operator = op(backend)
    values, ids, ape, norm, cos, sin = fixture(5, seed=seed, device="cuda")
    other, other_ids, *_ = fixture(5, request=99, seed=seed + 100, device="cuda")
    merged = torch.cat(
        (values, other, torch.full((3, 2048), float("nan"), dtype=values.dtype, device="cuda"))
    )
    metadata = torch.cat((ids, other_ids, torch.full((3, 2), -1, dtype=torch.int64, device="cuda")))
    owners = torch.arange(len(metadata), device="cuda").remainder(3).long()
    permutation = torch.randperm(
        len(metadata), generator=torch.Generator().manual_seed(seed)
    ).cuda()
    merged, metadata, owners = merged[permutation], metadata[permutation], owners[permutation]
    observed = {}
    for rank in range(3):
        bound = operator.bind(merged, metadata, owners=owners, cp_size=3, rank=rank, version="v1")
        output = operator(bound, ape, norm, cos, sin, version="v1").values
        for key, value in zip(bound.logical_keys, output, strict=True):
            observed[key] = value
    baseline = reference(
        values.double(), ape.double(), norm.double(), torch.arange(5), cos, sin, 1e-6
    )
    assert len(observed) == 10
    for group in range(5):
        numerical_close(observed[(7, group)], baseline[group])


@pytest.mark.parametrize("content_scale,gate_scale", [(0, 1), (2.0**-25, 1), (2.0**20, 1), (1, 80)])
def test_extreme_content_gate_scales_are_finite(backend, content_scale, gate_scale):
    operator = op(backend)
    values, ids, ape, norm, cos, sin = fixture(4, device="cuda")
    values[:, :1024] *= content_scale
    values[:, 1024:] *= gate_scale
    bound = operator.bind(values, ids, version="v1")
    actual = operator(bound, ape, norm, cos, sin, version="v1").values
    expected = reference(
        bound.values.double(), ape.double(), norm.double(), torch.arange(4), cos, sin, 1e-6
    )
    numerical_close(actual, expected)
    assert torch.isfinite(actual).all()


def train_chunks(backend, chunks, *, order=None, seed=19):
    operator = op(backend)
    values, ids, ape, norm, cos, sin = fixture(8, seed=seed, device="cuda", requires_grad=True)
    dy = torch.randn((8, 512), generator=torch.Generator().manual_seed(44)).bfloat16().cuda()
    with canonical_backward_session() as session:
        calls = []
        for start, end in chunks:
            input_start = max(0, start - 1)
            bound = operator.bind(
                values[input_start * 4 : end * 4].contiguous(),
                ids[input_start * 4 : end * 4],
                version="v1",
                emit_group_ids=tuple((7, group) for group in range(start, end)),
            )
            output = operator(
                bound,
                ape,
                norm,
                cos,
                sin,
                version="v1",
                parameter_id="layer.c4",
            )
            calls.append((output, dy[start:end]))
        sequence = list(range(len(calls))) if order is None else order
        for index in sequence:
            calls[index][0].values.backward(calls[index][1])
        session.validate_complete()
    return values.grad, ape.grad, norm.grad


@pytest.mark.parametrize("seed", [0, 19, 73])
def test_native_backward_order_invariance_and_independent_autograd(backend, seed):
    baseline = train_chunks(backend, [(0, 4), (4, 8)], seed=seed)
    for chunks, order in [
        ([(0, 2), (2, 4), (4, 8)], [0, 1, 2]),
        ([(4, 8), (0, 4)], [1, 0]),
        ([(0, 1), (1, 3), (3, 8)], [2, 0, 1]),
    ]:
        actual = train_chunks(backend, chunks, order=order, seed=seed)
        for got, expected in zip(actual, baseline, strict=True):
            byte_equal(got, expected)
    values, ids, ape, norm, cos, sin = fixture(8, seed=seed, device="cuda", requires_grad=True)
    dy = torch.randn((8, 512), generator=torch.Generator().manual_seed(44)).bfloat16().cuda()
    vd, ad, nd = (
        values.detach().double().requires_grad_(),
        ape.detach().double().requires_grad_(),
        norm.detach().double().requires_grad_(),
    )
    expected = torch.autograd.grad(
        (reference(vd, ad, nd, torch.arange(8, device="cuda"), cos, sin, 1e-6) * dy.double()).sum(),
        (vd, ad, nd),
    )
    for got, want in zip(baseline, expected, strict=True):
        torch.testing.assert_close(got.double(), want, rtol=4e-3, atol=3e-5)


def test_empty_input_no_native_launch_and_stream_reuse(backend, monkeypatch):
    operator = op(backend)
    values, ids, ape, norm, cos, sin = fixture(0, device="cuda")
    with monkeypatch.context() as patch:
        patch.setattr(
            backend.aiter,
            "flydsl_fused_compress_attn",
            lambda **_: (_ for _ in ()).throw(AssertionError("launch")),
        )
        output = operator(
            operator.bind(values, ids, version="v1"), ape, norm, cos, sin, version="v1"
        ).values
    assert output.shape == (0, 512)
    values, ids, ape, norm, cos, sin = fixture(8, device="cuda")
    bound = operator.bind(values, ids, version="v1")
    expected = operator(bound, ape, norm, cos, sin, version="v1").values
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = operator(bound, ape, norm, cos, sin, version="v1").values
        torch.cuda.current_stream().wait_stream(stream)
        byte_equal(actual, expected)


def test_mutation_and_graph_rejection(backend):
    operator = op(backend)
    values, ids, ape, norm, cos, sin = fixture(2, device="cuda", requires_grad=True)
    with canonical_backward_session():
        output = operator(
            operator.bind(values, ids, version="v1"),
            ape,
            norm,
            cos,
            sin,
            version="v1",
            parameter_id="mutation",
        )
        with torch.no_grad():
            values.add_(0.1)
        with pytest.raises((RuntimeError, C4Error), match="modified|changed"):
            output.values.sum().backward()
    bound = operator.bind(values.detach(), ids, version="v1")
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(C4Error, match="Graph"):
        with torch.cuda.graph(graph):
            operator(bound, ape.detach(), norm.detach(), cos, sin, version="v1")


def test_fresh_process_provider_fingerprint(backend, tmp_path):
    del backend
    script = Path(__file__).with_name("c4_process_probe.py")
    records = []
    for index in range(2):
        output = tmp_path / f"probe{index}.json"
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
            "TRITON_CACHE_DIR": str(tmp_path / f"cache{index}"),
        }
        subprocess.run(
            [sys.executable, str(script), str(output)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        records.append(json.loads(output.read_text()))
    assert records[0] == records[1]


@pytest.mark.parametrize("bias", [-(2.0**30), -(2.0**25), 0, 2.0**25])
def test_first_window_native_mask_excludes_padding_even_for_extreme_scores(backend, bias):
    operator = op(backend)
    values, ids, ape, norm, cos, sin = fixture(1, device="cuda")
    values[:, 1024:] = bias
    ape.zero_()
    bound = operator.bind(values, ids, version="v1")
    actual = operator(bound, ape, norm, cos, sin, version="v1").values
    expected = reference(
        values.double(), ape.double(), norm.double(), torch.tensor([0]), cos, sin, 1e-6
    )
    numerical_close(actual, expected)


def test_nonfinite_predecessor_context_is_not_hidden_by_emit_filter(backend):
    operator = op(backend)
    values, ids, *_ = fixture(2, device="cuda")
    values[0, 0] = float("nan")
    with pytest.raises(C4Error, match="finite"):
        operator.bind(values, ids, version="v1", emit_group_ids=[(7, 1)])


def test_fp64_oracle_really_preserves_fp64():
    values, ids, ape, norm, cos, sin = fixture(2, device="cuda")
    x = values.double().requires_grad_()
    a = ape.double().requires_grad_()
    n = norm.double().requires_grad_()
    y = reference(x, a, n, torch.arange(2, device="cuda"), cos, sin, 1e-6)
    assert y.dtype == torch.float64
    eps = 1e-8
    with torch.no_grad():
        perturbed = x.clone()
        perturbed[0, 0] += eps
    assert not torch.equal(
        y, reference(perturbed, a, n, torch.arange(2, device="cuda"), cos, sin, 1e-6)
    )
    assert all(t.dtype == torch.float64 for t in torch.autograd.grad(y.sum(), (x, a, n)))
