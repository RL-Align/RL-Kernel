"""Native primitive feasibility/acceptance; not the completed item-26 training chain."""

import pytest
import torch

from rl_engine.backends.rocm.attention.indexer_training import AiterIndexerTrainingMath
from rl_engine.contracts.operators.indexer_projection import IndexerProjectionError

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="MI300X IEEE primitive validation",
)


@pytest.fixture(scope="module")
def math():
    return AiterIndexerTrainingMath()


def same(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize(
    "shape", [(0, 128, 65), (1, 128, 65), (7, 128, 65), (33, 256, 129), (65, 65, 128)]
)
def test_native_fp32_gemm(shape, math):
    m, k, n = shape
    torch.manual_seed(26)
    x = torch.randn((m, k), device="cuda", dtype=torch.float32)
    w = torch.randn((n, k), device="cuda", dtype=torch.float32)
    out = math.linear_fp32(x, w)
    expected = x.double() @ w.double().t()
    torch.testing.assert_close(out.double(), expected, rtol=2e-5, atol=3e-5)
    for row in range(m):
        same(out[row : row + 1], math.linear_fp32(x[row : row + 1], w))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "batch,h,k,n", [(0, 13, 128, 65), (1, 13, 128, 65), (7, 64, 128, 512), (33, 13, 65, 129)]
)
def test_vllm_batched_native_output(dtype, batch, h, k, n, math):
    torch.manual_seed(260)
    a = torch.randn(batch, h, k, device="cuda").to(dtype)
    w = torch.randn(batch, n, k, device="cuda").to(dtype)
    b = w.transpose(1, 2)
    out = math.bmm_fp32_output(a, b)
    expected = torch.bmm(a.double(), b.double())
    torch.testing.assert_close(out.double(), expected, rtol=2e-5, atol=3e-5)
    for row in range(batch):
        same(out[row : row + 1], math.bmm_fp32_output(a[row : row + 1], b[row : row + 1]))
        if dtype == torch.bfloat16:
            # Preserve item 16's basic-GEMM arithmetic, not merely ideal math.
            original = math.linear_backend.linear(
                a[row], w[row], output_dtype=torch.float32, role="indexer_vjp"
            )
            same(out[row], original)


@pytest.mark.parametrize("m,h,d", [(0, 13, 128), (1, 64, 128), (7, 13, 65), (33, 64, 512)])
def test_mha_row_dot_and_broadcast_sum(m, h, d, math):
    torch.manual_seed(261)
    a = torch.randn(m, h, d, device="cuda")
    b = torch.randn(m, 1, d, device="cuda").expand(m, h, d)
    out = math.row_dot(a, b)
    expected = (a.double() * b.double()).sum(-1)
    torch.testing.assert_close(out.double(), expected, rtol=2e-5, atol=3e-5)
    sums = math.row_sum(a)
    torch.testing.assert_close(sums.double(), a.double().sum(-1), rtol=2e-5, atol=3e-5)
    for row in range(m):
        same(out[row : row + 1], math.row_dot(a[row : row + 1], b[row : row + 1]))
        same(sums[row : row + 1], math.row_sum(a[row : row + 1]))


def test_tf32_process_is_rejected():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rl_engine.backends.rocm.attention.indexer_training "
            "import AiterIndexerTrainingMath; AiterIndexerTrainingMath()",
        ],
        env={**os.environ, "TRITON_F32_DEFAULT": "tf32"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode != 0 and "TF32 Indexer VJPs" in result.stderr


@pytest.mark.parametrize("fault", ["device", "linear_dtype", "bmm_shape", "row_dot_shape"])
def test_native_math_guards(fault, math):
    x = torch.ones((2, 128), device="cuda")
    w = torch.ones((3, 128), device="cuda")
    with pytest.raises(IndexerProjectionError):
        if fault == "device":
            math.linear_fp32(x.cpu(), w)
        elif fault == "linear_dtype":
            math.linear_fp32(x.bfloat16(), w.bfloat16())
        elif fault == "bmm_shape":
            math.bmm_fp32_output(x[None], w[None])
        else:
            math.row_dot(x[None], w[None])
