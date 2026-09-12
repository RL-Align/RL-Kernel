import pytest
import torch

from rl_engine.kernels.ops.cuda.loss.logp import FusedLogpGenericOp
from rl_engine.kernels.registry import KernelRegistry


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
        pytest.param(torch.float16, 2e-3, 2e-3, id="fp16"),
        pytest.param(torch.bfloat16, 2e-2, 2e-2, id="bf16"),
    ],
)
def test_musa_fused_logp_matches_reference_and_supports_backward(dtype, atol, rtol):
    from rl_engine import _C

    assert hasattr(_C, "fused_logp")
    assert hasattr(_C, "fused_logp_backward")
    logits = torch.randn(4, 257, device="musa", dtype=dtype, requires_grad=True)
    token_ids = torch.tensor([0, 17, 128, 256], device="musa", dtype=torch.long)
    upstream = torch.tensor([0.25, -1.5, 2.0, 0.75], device="musa", dtype=dtype)

    output = FusedLogpGenericOp()(logits, token_ids)
    reference = torch.log_softmax(logits.float(), dim=-1).gather(1, token_ids[:, None]).squeeze(1)
    assert torch.allclose(output.float(), reference, atol=atol, rtol=rtol)

    output.backward(upstream)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    reference_logits = logits.detach().float().requires_grad_(True)
    reference_output = (
        torch.log_softmax(reference_logits, dim=-1).gather(1, token_ids[:, None]).squeeze(1)
    )
    reference_output.backward(upstream.float())
    assert reference_logits.grad is not None
    assert torch.allclose(
        logits.grad.float(),
        reference_logits.grad.to(dtype).float(),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_registry_selects_fused_logp_backend():
    backend = KernelRegistry().get_op("logp", device="musa")
    assert backend.__class__.__name__ == "FusedLogpGenericOp"
