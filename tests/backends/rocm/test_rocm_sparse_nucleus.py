# SPDX-License-Identifier: Apache-2.0
"""Numerical and geometry regressions for the compiled sparse ROCm scorer."""
import pytest
import torch

from rl_engine.integrations.common.linear_logp import LinearLogpWrapper

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="ROCm GPU required"
)


@pytest.mark.parametrize("width", [3, 65, 128, 129, 513])
@pytest.mark.parametrize("temperature", [0.7, 1.3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_complete_support_forward_backward_and_row_batching(width, temperature, dtype):
    torch.manual_seed(29)
    logits = torch.randn(3, 1024, device="cuda", dtype=dtype, requires_grad=True)
    ids = torch.arange(width, device="cuda").expand(3, -1).contiguous()
    targets = torch.tensor([0, 1, 2], device="cuda")
    wrapper = LinearLogpWrapper()

    def score(x, support, target):
        return wrapper.from_local_logits_sparse_nucleus(
            x,
            target,
            support,
            tp_group=None,
            vocab_start_index=0,
            global_vocab_size=1024,
            real_vocab_size=1024,
            temperature=temperature,
            target="training",
            return_entropy=True,
        )

    actual, entropy = score(logits, ids, targets)
    expected = torch.log_softmax(logits[:, :width].float() / temperature, dim=-1)
    torch.testing.assert_close(actual, expected[torch.arange(3), targets], atol=3e-6, rtol=2e-6)
    torch.testing.assert_close(entropy, -(expected.exp() * expected).sum(-1), atol=3e-6, rtol=2e-6)
    assert not entropy.requires_grad
    actual.sum().backward()
    expected_grad = -expected.detach().exp()
    expected_grad[torch.arange(3), targets] += 1
    expected_grad /= temperature
    torch.testing.assert_close(
        logits.grad[:, :width],
        expected_grad.to(dtype),
        atol=3e-6,
        rtol=max(3e-6, torch.finfo(dtype).eps),
    )
    assert torch.count_nonzero(logits.grad[:, width:]) == 0
    # Arbitrary support width and batching must not change a row's bit pattern.
    one, one_entropy = score(logits[:1].detach(), ids[:1], targets[:1])
    padded_ids = torch.nn.functional.pad(ids, (0, 31), value=-1)
    padded, padded_entropy = score(logits.detach(), padded_ids, targets)
    assert torch.equal(one.view(torch.int32), actual[:1].view(torch.int32))
    assert torch.equal(padded.view(torch.int32), actual.view(torch.int32))
    assert torch.equal(one_entropy.view(torch.int32), entropy[:1].view(torch.int32))
    assert torch.equal(padded_entropy.view(torch.int32), entropy.view(torch.int32))


def test_duplicate_support_ids_do_not_change_probability():
    logits = torch.tensor([[1.0, -1.0, 0.0, 2.0]], device="cuda")
    wrapper = LinearLogpWrapper()
    kwargs = dict(
        tp_group=None,
        vocab_start_index=0,
        global_vocab_size=4,
        real_vocab_size=4,
        temperature=0.7,
        target="rollout",
    )
    target = torch.tensor([3], device="cuda")
    a = wrapper.from_local_logits_sparse_nucleus(
        logits, target, torch.tensor([[0, 3]], device="cuda"), **kwargs
    )
    b = wrapper.from_local_logits_sparse_nucleus(
        logits, target, torch.tensor([[3, 0, 3, -1]], device="cuda"), **kwargs
    )
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
