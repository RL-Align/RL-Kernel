# SPDX-License-Identifier: Apache-2.0
"""INK-47 CUDA fault injection for per-visit KV and convolution history."""

import pytest
import torch
import torch.nn.functional as F

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2.visit_cache import VisitKey, VisitStateCache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_visit_state_clone"),
    reason="requires the native INK-22 visit-state extension",
)


def _decode(kv, conv, tail):
    key, value = kv.unbind()
    # A zero query gives uniform attention over the prefix and current token.
    query = torch.zeros_like(key[:, :, :1])
    attended, _, _ = _C.deterministic_attention_forward(query, key, value, True, 1.0, None)
    conv_input = torch.cat((conv, tail), dim=-1)
    convolved = F.conv1d(conv_input, torch.ones_like(conv_input)).flatten()
    return torch.stack((attended[0, 0, 0, 0].float(), convolved[0].float()))


def _kv_history(value):
    key = torch.zeros((1, 1, 3, 128), dtype=torch.bfloat16, device="cuda")
    return torch.stack((key, torch.full_like(key, value)))


def _append_token(kv):
    current = kv.new_zeros((*kv.shape[:-2], 1, kv.shape[-1]))
    return torch.cat((kv, current), dim=-2)


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("wrong_pass_kv", [6.0, 5.0]),
        ("wrong_pass_conv", [3.0, 11.0]),
        ("chunk_reset", [0.0, 2.0]),
        ("stale_prefix", [9.0, 8.0]),
    ],
)
def test_cuda_decode_detects_cross_visit_and_stale_state(fault, expected):
    visit, next_pass = VisitKey(6, 0, 6), VisitKey(6, 1, 18)
    cache, other_prefix = VisitStateCache(), VisitStateCache()
    tail = torch.full((1, 1, 1), 2.0, dtype=torch.bfloat16, device="cuda")
    kv = _kv_history(4)
    conv = torch.ones_like(tail).expand(-1, -1, 3).contiguous()
    cache.put(visit, kv=kv, conv=conv)
    cache.put(next_pass, kv=_kv_history(8), conv=conv * 3)
    other_prefix.put(visit, kv=_kv_history(12), conv=conv * 2)

    stored = cache.require(visit)
    correct = _decode(_append_token(stored.kv), stored.conv, tail)
    assert torch.equal(correct, _decode(_append_token(kv), conv, tail))
    assert torch.equal(correct, torch.tensor([3.0, 5.0], device="cuda"))

    wrong_pass = cache.require(next_pass)
    stale = other_prefix.require(visit)
    faulty_kv, faulty_conv = {
        "wrong_pass_kv": (wrong_pass.kv, stored.conv),
        "wrong_pass_conv": (stored.kv, wrong_pass.conv),
        "chunk_reset": (stored.kv[..., :0, :], torch.zeros_like(stored.conv)),
        "stale_prefix": (stale.kv, stale.conv),
    }[fault]
    faulty = _decode(_append_token(faulty_kv), faulty_conv, tail)
    assert torch.equal(faulty, torch.tensor(expected, device="cuda"))

    # Fault injection affects the copied state for this decode only.
    replay = cache.require(visit)
    assert torch.equal(correct, _decode(_append_token(replay.kv), replay.conv, tail))
