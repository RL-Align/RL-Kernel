"""Grouping independent gradient coordinates must preserve every BF16 bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA required",
)


@pytest.mark.parametrize("tokens", [1, 513, 4096])
@pytest.mark.parametrize("chunks", [2, 4, 8])
@pytest.mark.parametrize("column", [False, True])
def test_weight_gradient_matches_canonical_leaves(tokens, chunks, column, monkeypatch):
    from rl_engine.backends.cuda.ffn.ffn import _canonical_tp_weight_gradient
    from rl_engine.backends.cuda.gemm.det_gemm import det_gemm_linear_weight_gradient
    from rl_engine.distributed.algorithms.canonical_cp import weight_gradient

    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "cublaslt_nosplitk")
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", str(4 * chunks))
    torch.manual_seed(tokens + chunks)
    x = torch.randn(tokens, 4096 if column else 3584, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(tokens, 3584 if column else 4096, device="cuda", dtype=torch.bfloat16)
    width = (dy if column else x).size(1) // chunks
    leaves = [
        det_gemm_linear_weight_gradient(
            x if column else x[:, i * width : (i + 1) * width].contiguous(),
            dy[:, i * width : (i + 1) * width].contiguous() if column else dy,
        )
        for i in range(chunks)
    ]
    expected = torch.cat(leaves, dim=0 if column else 1)
    for actual in (
        _canonical_tp_weight_gradient(x, dy, tp_world=4, column=column, disable_split_k=True),
        weight_gradient(x, dy, None, chunks=chunks, column=column),
    ):
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("tokens", [1, 513, 4096])
@pytest.mark.parametrize("chunks", [2, 4, 8])
def test_down_input_gradient_preserves_independent_columns(tokens, chunks, monkeypatch):
    from rl_engine.backends.cuda.ffn.ffn import _canonical_tp_input_gradient
    from rl_engine.backends.cuda.gemm.det_gemm import det_gemm_linear_input_gradient

    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "cublaslt_nosplitk")
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", str(4 * chunks))
    torch.manual_seed(tokens + chunks)
    dy = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 3584, device="cuda", dtype=torch.bfloat16)
    expected = torch.cat(
        [
            det_gemm_linear_input_gradient(dy, piece.contiguous())
            for piece in w.chunk(chunks, dim=1)
        ],
        dim=1,
    )
    actual = _canonical_tp_input_gradient(dy, w, tp_world=4, column=False, disable_split_k=True)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
