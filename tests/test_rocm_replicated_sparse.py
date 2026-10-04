# SPDX-License-Identifier: Apache-2.0
"""Compare fused rollout preparation with the unchanged training arithmetic."""
import pytest
import torch

from rl_engine.integrations.linear_logp import LinearLogpWrapper

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="ROCm required"
)


@pytest.mark.parametrize("width", [3, 65, 128, 129, 513])
@pytest.mark.parametrize("temperature", [0.3, 0.7, 1.6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_replicated_sparse_matches_training_bits(width, temperature, dtype):
    torch.manual_seed(134 + width)
    logits = torch.randn(4, 1024, device="cuda", dtype=dtype)
    logits[:, :2] = torch.tensor([-0.0, 0.0], device="cuda", dtype=dtype)
    ids = torch.randperm(1024, device="cuda")[:width].expand(4, -1).clone()
    if width > 3:
        ids[:, -2:] = ids[:, :2]  # duplicate targets, as in vLLM's output
        ids[1, -1] = -1
        ids[2, -1] = 1024  # invalid/padded vocabulary ID
    targets = ids[:, 0].contiguous()
    eager = LinearLogpWrapper()
    with torch.no_grad():
        expected = eager.from_local_logits_sparse_nucleus(
            logits,
            targets,
            ids,
            tp_group=None,
            vocab_start_index=0,
            global_vocab_size=1024,
            real_vocab_size=1024,
            temperature=temperature,
            target="training",
            return_entropy=True,
        )
        fused = LinearLogpWrapper()
        actual = fused.from_replicated_logits_sparse_nucleus(
            logits,
            targets,
            ids,
            real_vocab_size=1024,
            temperature=temperature,
            return_entropy=True,
        )
        single = fused.from_replicated_logits_sparse_nucleus(
            logits[:1],
            targets[:1],
            ids[:1],
            real_vocab_size=1024,
            temperature=temperature,
            return_entropy=True,
        )
    for a, b, one in zip(actual, expected, single):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        assert torch.equal(a[:1].view(torch.int32), one.view(torch.int32))


@pytest.mark.parametrize("ids_list", [[0], [0, 1], [1, 0, 0, -1]])
def test_replicated_sparse_signed_zero(ids_list):
    logits = torch.tensor([[-0.0, 0.0, -2.0]], device="cuda")
    ids = torch.tensor([ids_list], device="cuda")
    targets = ids[:, 0].contiguous()
    wrapper = LinearLogpWrapper()
    with torch.no_grad():
        expected = wrapper.from_local_logits_sparse_nucleus(
            logits,
            targets,
            ids,
            tp_group=None,
            vocab_start_index=0,
            global_vocab_size=3,
            real_vocab_size=3,
            temperature=0.7,
            target="training",
            return_entropy=True,
        )
        actual = wrapper.from_replicated_logits_sparse_nucleus(
            logits,
            targets,
            ids,
            real_vocab_size=3,
            temperature=0.7,
            return_entropy=True,
        )
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
