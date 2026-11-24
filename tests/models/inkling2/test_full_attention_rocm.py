# SPDX-License-Identifier: Apache-2.0
"""INK-15 ROCm full-attention kernel tests (wave64 / MI300X)."""

import pytest
import torch

from rl_engine.models.inkling2.full_attention import full_attention, full_attention_rocm

pytestmark = pytest.mark.skipif(
    getattr(torch.version, "hip", None) is None or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)


@pytest.fixture(autouse=True)
def _binding():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_full_attention_forward"):
        pytest.skip("Inkling 2 full attention extension is unavailable")


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOL = {
    torch.float32: dict(atol=2e-5, rtol=1e-4),
    torch.float16: dict(atol=2e-2, rtol=2e-2),
    torch.bfloat16: dict(atol=4e-2, rtol=4e-2),
}
GRAD_TOL = {
    torch.float32: dict(atol=2e-4, rtol=2e-3),
    torch.float16: dict(atol=5e-2, rtol=5e-2),
    torch.bfloat16: dict(atol=1e-1, rtol=1e-1),
}


def _inputs(batch, q_len, k_len, q_heads, kv_heads, dim, dtype, *, seed=0, grad=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, device="cuda", dtype=torch.float32, generator=gen).to(dt)

    tensors = dict(
        query=rnd(batch, q_len, q_heads, dim),
        key=rnd(batch, k_len, kv_heads, dim),
        value=rnd(batch, k_len, kv_heads, dim),
        relative_bias=rnd(q_heads, q_len, k_len, dt=torch.float32),
        sink=rnd(q_heads, dt=torch.float32),
        output_gate=rnd(q_heads, dt=torch.float32),
    )
    if grad:
        for tensor in tensors.values():
            tensor.requires_grad_(True)
    return tensors


def _documents(batch, q_len, k_len):
    # Documents of length 7 over the keys; queries aligned to the key tail.
    key_docs = (torch.arange(k_len, device="cuda") // 7).expand(batch, -1).contiguous()
    query_docs = key_docs[:, k_len - q_len :].contiguous()
    return query_docs, key_docs


CASES = [
    # (batch, q_len, k_len, q_heads, kv_heads, dim)
    (1, 3, 4, 4, 2, 5),  # tiny / odd head dim
    (2, 1, 1, 24, 8, 128),  # one row
    (2, 70, 130, 24, 8, 128),  # not multiples of 64, q_len < k_len
    (1, 192, 192, 24, 8, 128),  # real Inkling heads
    (1, 9, 77, 6, 2, 96),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("variant", ["plain", "full"])
def test_forward_matches_reference(case, dtype, variant):
    batch, q_len, k_len, q_heads, kv_heads, dim = case
    t = _inputs(batch, q_len, k_len, q_heads, kv_heads, dim, dtype, seed=q_len)
    kwargs = dict(causal=True)
    if variant == "full":
        q_docs, k_docs = _documents(batch, q_len, k_len)
        kwargs.update(
            relative_bias=t["relative_bias"],
            sink=t["sink"],
            output_gate=t["output_gate"],
            query_documents=q_docs,
            key_documents=k_docs,
        )
    native = full_attention_rocm(t["query"], t["key"], t["value"], **kwargs)
    reference = full_attention(t["query"], t["key"], t["value"], **kwargs)
    assert native.dtype == dtype and native.shape == reference.shape
    torch.testing.assert_close(native, reference, **TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [True, False])
def test_backward_matches_reference(dtype, causal):
    batch, q_len, k_len, q_heads, kv_heads, dim = (2, 40, 40, 24, 8, 128)
    q_docs, k_docs = _documents(batch, q_len, k_len)
    weights = torch.randn(batch, q_len, q_heads, dim, device="cuda")
    grads = {}
    for name, fn in (("native", full_attention_rocm), ("reference", full_attention)):
        t = _inputs(batch, q_len, k_len, q_heads, kv_heads, dim, dtype, seed=11, grad=True)
        out = fn(
            t["query"],
            t["key"],
            t["value"],
            relative_bias=t["relative_bias"],
            sink=t["sink"],
            output_gate=t["output_gate"],
            query_documents=q_docs,
            key_documents=k_docs,
            causal=causal,
        )
        (out.float() * weights).sum().backward()
        grads[name] = {k: v.grad for k, v in t.items()}
    for name in grads["native"]:
        got, want = grads["native"][name], grads["reference"][name]
        assert got is not None and torch.isfinite(got).all(), name
        try:
            torch.testing.assert_close(got, want, **GRAD_TOL[dtype])
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from error


def test_backward_without_optional_inputs():
    t = _inputs(1, 33, 33, 24, 8, 128, torch.float32, seed=5, grad=True)
    inputs = (t["query"], t["key"], t["value"])
    out = full_attention_rocm(*inputs)
    ref = full_attention(*inputs)
    w = torch.randn_like(out)
    (out * w).sum().backward()
    got = [x.grad.clone() for x in inputs]
    for x in inputs:
        x.grad = None
    (ref * w).sum().backward()
    for g, x in zip(got, inputs, strict=True):
        torch.testing.assert_close(g, x.grad, **GRAD_TOL[torch.float32])


@pytest.mark.parametrize("dtype", DTYPES)
def test_run_to_run_bitwise_determinism(dtype):
    t = _inputs(2, 70, 130, 24, 8, 128, dtype, seed=3, grad=True)
    q_docs, k_docs = _documents(2, 70, 130)
    weights = torch.randn(2, 70, 24, 128, device="cuda")

    def run():
        for x in t.values():
            x.grad = None
        out = full_attention_rocm(
            t["query"],
            t["key"],
            t["value"],
            relative_bias=t["relative_bias"],
            sink=t["sink"],
            output_gate=t["output_gate"],
            query_documents=q_docs,
            key_documents=k_docs,
        )
        (out.float() * weights).sum().backward()
        return [out.detach().clone()] + [x.grad.clone() for x in t.values()]

    first = run()
    for _ in range(3):
        for a, b in zip(first, run(), strict=True):
            assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_invariance(dtype):
    t = _inputs(3, 50, 50, 24, 8, 128, dtype, seed=7)
    bias = torch.randn(3, 24, 50, 50, device="cuda")
    q_docs, k_docs = _documents(3, 50, 50)
    kwargs = dict(sink=t["sink"], output_gate=t["output_gate"])
    full = full_attention_rocm(
        t["query"],
        t["key"],
        t["value"],
        relative_bias=bias,
        query_documents=q_docs,
        key_documents=k_docs,
        **kwargs,
    )
    # Perturb the other batch rows entirely; row 1 must not move by a single bit.
    perturbed_q = t["query"].clone()
    perturbed_k = t["key"].clone()
    perturbed_v = t["value"].clone()
    for x in (perturbed_q, perturbed_k, perturbed_v):
        x[0].mul_(-3.0)
        x[2].add_(1.5)
    other = full_attention_rocm(
        perturbed_q,
        perturbed_k,
        perturbed_v,
        relative_bias=bias,
        query_documents=q_docs,
        key_documents=k_docs,
        **kwargs,
    )
    assert torch.equal(full[1], other[1])
    assert not torch.equal(full[0], other[0])
    # A batch of one reproduces the same bits as the row inside a larger batch.
    solo = full_attention_rocm(
        t["query"][1:2],
        t["key"][1:2],
        t["value"][1:2],
        relative_bias=bias[1:2],
        query_documents=q_docs[1:2],
        key_documents=k_docs[1:2],
        **kwargs,
    )
    assert torch.equal(full[1:2], solo)


def test_row_independent_of_other_queries():
    t = _inputs(1, 64, 64, 24, 8, 128, torch.bfloat16, seed=9)
    full = full_attention_rocm(t["query"], t["key"], t["value"], sink=t["sink"])
    q2 = t["query"].clone()
    q2[:, 10:] *= 2.0
    changed = full_attention_rocm(q2, t["key"], t["value"], sink=t["sink"])
    assert torch.equal(full[:, :10], changed[:, :10])


def test_empty_shapes():
    for q_len, batch in ((0, 1), (4, 0)):
        t = _inputs(batch, q_len, 4, 4, 2, 8, torch.float32)
        out = full_attention_rocm(t["query"], t["key"], t["value"], causal=False)
        assert out.shape == (batch, q_len, 4, 8)


def test_rejects_cpu_tensors():
    q = torch.randn(1, 2, 2, 4)
    with pytest.raises(ValueError):
        full_attention_rocm(q, q[:, :, :1], q[:, :, :1])
