# SPDX-License-Identifier: Apache-2.0
"""The worker raw-score boundary preserves columns, versions and FP32 gradients."""
import pytest
import torch

from rl_engine.backends.shared.triton.logprob.logp import TritonLogpOp

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_selected_scores_full_vocabulary_accuracy_and_gradient():
    torch.manual_seed(83)
    logits = torch.randn(3, 129280, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([92132, 109502, 64432], device="cuda")
    actual = TritonLogpOp()(logits, targets)
    expected = (
        logits.detach().cpu().double().log_softmax(-1).gather(1, targets.cpu()[:, None])[:, 0]
    )
    torch.testing.assert_close(actual.cpu().double(), expected, atol=3e-6, rtol=0)
    actual.sum().backward()
    reference = -logits.detach().softmax(-1)
    reference[torch.arange(3, device="cuda"), targets] += 1
    torch.testing.assert_close(logits.grad, reference, atol=2e-6, rtol=2e-5)
