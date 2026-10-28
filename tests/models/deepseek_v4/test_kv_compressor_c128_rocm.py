# SPDX-License-Identifier: Apache-2.0
"""Real MI300X aiter kernels and pinned Megatron VJP acceptance."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from rl_engine.backends.rocm.attention.kv_compressor_c128 import AiterC128Backend
from rl_engine.contracts.operators.kv_compressor_c128 import C128Contract, C128Error
from rl_engine.models.deepseek_v4.kv_compressor_c128 import C128Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.c128_helpers import byte_equal, fixture, reference

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


@pytest.fixture(scope="module")
def backend():
    # Missing required providers fail the GPU acceptance, not silently skip.
    return AiterC128Backend()


def op(backend, **kwargs):
    return C128Compressor(C128Contract(**kwargs), backend=backend)


def numerical_close(actual, expected):
    # Two BF16 ULPs for the independent FP64 formula vs upstream FP32 exp2/FMA.
    # BI assertions elsewhere remain exact byte comparisons without tolerance.
    exponent = torch.floor(torch.log2(expected.abs().clamp_min(2.0**-126)))
    ulp = torch.exp2(exponent - 7)
    assert bool(((actual.double() - expected).abs() <= (2 * ulp + 2e-5)).all())


@pytest.mark.parametrize("groups", [0, 1, 2, 3, 7, 31, 32, 33, 65])
def test_native_forward_against_independent_fp64(backend, groups):
    operator = op(backend)
    x, ids, a, n, co, si = fixture(groups * 128 + 17, device="cuda")
    b = operator.bind(x, ids, version="v1")
    y = operator(b, a, n, co, si, version="v1").values
    expected = reference(
        b.values.double(),
        a.double(),
        n.double(),
        b.group_ids[:, 1],
        co,
        si,
        operator.contract.rms_eps,
    )
    numerical_close(y, expected)
    assert len(b.incomplete_groups) == 1
    assert operator.implementation_fingerprint
    assert backend.provenance["arch"] == "gfx942"


def test_fixed_recipe_and_fp32_pool_are_bitwise_across_batch_sizes(backend, monkeypatch):
    operator = op(backend)
    x, ids, a, n, co, si = fixture(33 * 128, device="cuda")
    pool = backend.hca.flydsl_hca_compress_forward
    compile_norm = backend.hca.compile_hca_norm_rope_scatter
    captured = []

    def capture(**kwargs):
        assert kwargs["k_split_num_waves"] == 8 and kwargs["slice_size"] == 64
        result = pool(**kwargs)
        captured.append(result)
        return result

    def norm(**kwargs):
        assert kwargs["k_waves"] == 1
        return compile_norm(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backend.hca, "flydsl_hca_compress_forward", capture)
        patch.setattr(backend.hca, "compile_hca_norm_rope_scatter", norm)
        full = operator(operator.bind(x, ids, version="v1"), a, n, co, si, version="v1").values
        full_pool = captured[-1]
        for group in (0, 1, 16, 32):
            start = group * 128
            b = operator.bind(x[start : start + 128], ids[start : start + 128], version="v1")
            solo = operator(b, a, n, co, si, version="v1").values
            byte_equal(solo, full[group : group + 1])
            byte_equal(captured[-1], full_pool[group : group + 1])


@pytest.mark.parametrize("seed", range(8))
def test_ragged_permuted_multi_request_padding_and_cp_bitwise(backend, seed):
    operator = op(backend)
    x, ids, a, n, co, si = fixture(4 * 128 + 63, seed=seed, device="cuda")
    baseline = operator(operator.bind(x, ids, version="v1"), a, n, co, si, version="v1").values
    other, other_ids, *_ = fixture(4 * 128 + 63, request=99, seed=seed + 10, device="cuda")
    values = torch.cat(
        (x, other, torch.full((5, 1024), float("nan"), dtype=x.dtype, device="cuda"))
    )
    keys = torch.cat((ids, other_ids, torch.full((5, 2), -1, dtype=torch.int64, device="cuda")))
    owners = torch.arange(len(keys), device="cuda").remainder(3).long()
    permutation = torch.randperm(len(keys), generator=torch.Generator().manual_seed(seed)).cuda()
    values, keys, owners = values[permutation], keys[permutation], owners[permutation]
    observed = {}
    for rank in range(3):
        b = operator.bind(values, keys, version="v1", cp_size=3, rank=rank, owners=owners)
        out = operator(b, a, n, co, si, version="v1").values
        for index, key in enumerate(b.logical_keys):
            assert key not in observed
            observed[key] = out[index]
    assert len(observed) == 8
    for group in range(4):
        byte_equal(observed[(7, group)], baseline[group])


@pytest.mark.parametrize("content_scale,gate_scale", [(0, 1), (2.0**-25, 1), (2.0**20, 1), (1, 80)])
def test_numerical_spectra_and_softmax_extremes(backend, content_scale, gate_scale):
    operator = op(backend)
    x, ids, a, n, co, si = fixture(device="cuda")
    x[:, :512] *= content_scale
    x[:, 512:] *= gate_scale
    b = operator.bind(x, ids, version="v1")
    result = operator(b, a, n, co, si, version="v1").values
    expected = reference(b.values.double(), a.double(), n.double(), b.group_ids[:, 1], co, si, 1e-6)
    numerical_close(result, expected)
    assert torch.isfinite(result).all()


def train_chunks(backend, chunks, *, order=None, seed=18, input_only=False):
    operator = op(backend)
    x, ids, a, n, co, si = fixture(384, seed=seed, device="cuda", requires_grad=True)
    if input_only:
        a, n = a.detach(), n.detach()
    dy = torch.randn((3, 512), generator=torch.Generator().manual_seed(9)).bfloat16().cuda()
    with canonical_backward_session() as session:
        calls = []
        for start, end in chunks:
            bound = operator.bind(x[start:end].contiguous(), ids[start:end], version="v1")
            out = operator(bound, a, n, co, si, version="v1", parameter_id="layer0.c128")
            calls.append((out, dy[start // 128 : end // 128]))
        for index in range(len(calls)) if order is None else order:
            out, grad = calls[index]
            out.values.backward(grad)
        session.validate_complete()
    return (x.grad, a.grad, n.grad), (x, ids, a, n, co, si, dy)


@pytest.mark.parametrize("seed", [0, 18, 73])
def test_native_backward_fp64_accuracy_and_microbatch_bitwise(backend, seed):
    baseline, inputs = train_chunks(backend, [(0, 384)], seed=seed)
    for chunks, order in [
        ([(0, 128), (128, 384)], [0, 1]),
        ([(256, 384), (0, 256)], [1, 0]),
        ([(256, 384), (128, 256), (0, 128)], [0, 2, 1]),
    ]:
        actual, _ = train_chunks(backend, chunks, order=order, seed=seed)
        for value, expected in zip(actual, baseline, strict=True):
            byte_equal(value, expected)
    x, ids, a, n, co, si, dy = inputs
    xd, ad, nd = (t.detach().double().requires_grad_() for t in (x, a, n))
    y = reference(xd, ad, nd, torch.arange(3, device="cuda"), co, si, 1e-6)
    expected = torch.autograd.grad(y, (xd, ad, nd), dy.double())
    # Input gradient returns through the BF16 projection boundary; parameter
    # contributions and canonical accumulation remain FP32.
    torch.testing.assert_close(baseline[0].float(), expected[0].float(), rtol=0.004, atol=2e-5)
    for got, want in zip(baseline[1:], expected[1:], strict=True):
        torch.testing.assert_close(got.double(), want, rtol=2e-4, atol=2e-5)


def test_backward_input_only_and_finite_difference_parameter_entries(backend):
    grads, inputs = train_chunks(backend, [(0, 384)], input_only=True)
    assert grads[0] is not None and grads[1:] == (None, None)
    # Finite differences of the continuous FP64 mathematical operator, not of
    # discontinuous BF16 output rounding, independently validate the master VJP.
    actual, inputs = train_chunks(backend, [(0, 384)])
    x, ids, a, n, co, si, dy = inputs
    xd, ad, nd = x.detach().double(), a.detach().double(), n.detach().double()
    groups = torch.arange(3, device="cuda")
    eps = 1e-4
    for which, index in (("ape", (0, 0)), ("ape", (127, 511)), ("norm", (0,)), ("norm", (511,))):
        parameter = ad if which == "ape" else nd
        original = parameter[index].item()
        parameter[index] = original + eps
        positive = (reference(xd, ad, nd, groups, co, si, 1e-6) * dy.double()).sum()
        parameter[index] = original - eps
        negative = (reference(xd, ad, nd, groups, co, si, 1e-6) * dy.double()).sum()
        parameter[index] = original
        difference = (positive - negative) / (2 * eps)
        got = actual[1 if which == "ape" else 2][index]
        torch.testing.assert_close(got.double(), difference, rtol=3e-4, atol=2e-5)


def test_empty_native_never_launches_and_graph_is_rejected(backend, monkeypatch):
    operator = op(backend)
    x, ids, a, n, co, si = fixture(127, device="cuda")
    b = operator.bind(x, ids, version="v1")

    def forbidden(**kwargs):
        raise AssertionError("native pool must not run for incomplete groups")

    with monkeypatch.context() as patch:
        patch.setattr(backend.hca, "flydsl_hca_compress_forward", forbidden)
        assert operator(b, a, n, co, si, version="v1").values.shape == (0, 512)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(C128Error, match="Graph"):
        with torch.cuda.graph(graph):
            operator(b, a, n, co, si, version="v1")


def test_parameter_and_rope_mutation_before_backward_fails(backend):
    for which in ("ape", "norm", "cos", "source"):
        operator = op(backend)
        x, ids, a, n, co, si = fixture(128, device="cuda", requires_grad=True)
        with canonical_backward_session():
            y = operator(
                operator.bind(x, ids, version="v1"), a, n, co, si, version="v1", parameter_id="test"
            )
            with torch.no_grad():
                {"ape": a, "norm": n, "cos": co, "source": x}[which].add_(0.1)
            with pytest.raises((RuntimeError, C128Error), match="modified|changed"):
                y.values.sum().backward()


def test_two_streams_read_only_binding(backend):
    operator = op(backend)
    x, ids, a, n, co, si = fixture(384, device="cuda")
    b = operator.bind(x, ids, version="v1")
    expected = operator(b, a, n, co, si, version="v1").values
    producer = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream in streams:
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            outputs.append(operator(b, a, n, co, si, version="v1").values)
    for stream in streams:
        producer.wait_stream(stream)
    for output in outputs:
        byte_equal(output, expected)


def test_independent_processes_and_real_two_rank_gloo_cp(backend, tmp_path):
    del backend
    script = "tests/models/deepseek_v4/c128_process_probe.py"
    records = []
    for i in range(2):
        output = tmp_path / f"fresh{i}.json"
        env = {**os.environ, "TRITON_CACHE_DIR": str(tmp_path / f"triton{i}")}
        subprocess.run(
            [sys.executable, script, "single", str(output)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        records.append(json.loads(output.read_text()))
    assert records[0] == records[1]
    output = tmp_path / "cp"
    subprocess.run(
        [sys.executable, script, "cp", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    operator = C128Compressor(C128Contract())
    x, ids, a, n, co, si = fixture(384, device="cuda")
    expected = operator(operator.bind(x, ids, version="v1"), a, n, co, si, version="v1").values
    merged = {}
    for rank in range(2):
        record = torch.load(Path(str(output) + f".rank{rank}.pt"), weights_only=True)
        for key, value in zip(record["ids"].tolist(), record["values"], strict=True):
            assert tuple(key) not in merged
            merged[tuple(key)] = value
    byte_equal(torch.stack([merged[(7, g)] for g in range(3)]), expected.cpu())
    evidence = {
        "fresh_processes": records,
        "transport": "2 CPU Gloo ranks, uneven 150/234 rows",
        "gpu_compute": "one shared MI300X, disjoint owned groups; NOT RCCL/GPU CP acceptance",
    }
    (tmp_path / "process-evidence.json").write_text(json.dumps(evidence, indent=2))
