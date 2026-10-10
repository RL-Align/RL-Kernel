import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA required",
)


@pytest.mark.parametrize("batch", [1, 4, 64, 257])
@pytest.mark.parametrize("chunks", [2, 4, 8])
def test_fused_columns_preserve_canonical_ffn_bits(batch, chunks, monkeypatch):
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "cublaslt_nosplitk")
    from rl_engine.backends.cuda.ffn.ffn import (
        _qwen3_ffn_canonical_columns,
        _qwen3_ffn_packed_inference,
    )

    torch.manual_seed(batch + chunks)
    x = torch.randn(batch, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(6144, 4096, dtype=torch.bfloat16, device="cuda")
    down = torch.randn(4096, 3072, dtype=torch.bfloat16, device="cuda")
    width = 3072 // chunks
    parts = [
        _qwen3_ffn_packed_inference(
            x,
            torch.cat(
                (
                    weight[i * width : (i + 1) * width],
                    weight[3072 + i * width : 3072 + (i + 1) * width],
                )
            ),
            down[:, i * width : (i + 1) * width].contiguous(),
        )
        for i in range(chunks)
    ]
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    result = _qwen3_ffn_canonical_columns(x, weight, down, chunks)
    assert torch.equal(result.view(torch.int16), parts[0].view(torch.int16))
