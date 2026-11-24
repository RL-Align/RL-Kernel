# SPDX-License-Identifier: Apache-2.0
"""INK-47 ROCm fault injection for per-visit KV and convolution history.

Runs the CUDA ablation's decode oracle against states stored and fetched through
the native HIP visit-state clone (INK-22 ROCm) and asserts every injected fault
is detected, while the clean control matches bitwise.
"""

import pytest
import torch
import torch.nn.functional as F

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2 import clone_visit_state_rocm
from rl_engine.models.inkling2.visit_cache import VisitKey, VisitStateCache

pytestmark = pytest.mark.skipif(
    torch.version.hip is None
    or not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_visit_state_clone")
    or not hasattr(_C, "deterministic_attention_forward"),
    reason="requires ROCm PyTorch, a GPU, and the native INK-22 visit-state extension",
)

FAULTS = {
    "wrong_pass_kv": [6.0, 5.0],
    "wrong_pass_conv": [3.0, 11.0],
    "chunk_reset": [0.0, 2.0],
    "stale_prefix": [9.0, 8.0],
}
CORRECT = [3.0, 5.0]


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


def _setup(monkeypatch):
    calls = []
    original = _C.inkling2_visit_state_clone

    def spy(tensor):
        calls.append(tensor.data_ptr())
        return original(tensor)

    monkeypatch.setattr(_C, "inkling2_visit_state_clone", spy)
    visit, next_pass = VisitKey(6, 0, 6), VisitKey(6, 1, 18)
    cache, other_prefix = VisitStateCache(), VisitStateCache()
    tail = torch.full((1, 1, 1), 2.0, dtype=torch.bfloat16, device="cuda")
    kv = _kv_history(4)
    conv = torch.ones_like(tail).expand(-1, -1, 3).contiguous()
    cache.put(visit, kv=kv, conv=conv)
    cache.put(next_pass, kv=_kv_history(8), conv=conv * 3)
    other_prefix.put(visit, kv=_kv_history(12), conv=conv * 2)
    return visit, next_pass, cache, other_prefix, tail, kv, conv, calls


def test_rocm_clean_control_is_bitwise_and_uses_native_clone(monkeypatch):
    visit, _, cache, _, tail, kv, conv, calls = _setup(monkeypatch)
    assert len(calls) == 6  # three puts x (kv, conv) go through the HIP kernel
    stored = cache.require(visit)
    assert len(calls) == 8
    correct = _decode(_append_token(stored.kv), stored.conv, tail)
    direct = _decode(_append_token(kv), conv, tail)
    assert torch.equal(correct, direct)
    assert torch.equal(correct, torch.tensor(CORRECT, device="cuda"))
    # Re-running the clean decode is run-to-run bitwise identical.
    for _ in range(3):
        replay = cache.require(visit)
        assert torch.equal(correct, _decode(_append_token(replay.kv), replay.conv, tail))
    # The stored state is a bit-exact HIP copy of the source.
    assert torch.equal(clone_visit_state_rocm(kv), kv)


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_rocm_decode_detects_cross_visit_and_stale_state(fault, monkeypatch):
    visit, next_pass, cache, other_prefix, tail, kv, conv, _ = _setup(monkeypatch)
    stored = cache.require(visit)
    correct = _decode(_append_token(stored.kv), stored.conv, tail)
    assert torch.equal(correct, torch.tensor(CORRECT, device="cuda"))

    wrong_pass = cache.require(next_pass)
    stale = other_prefix.require(visit)
    faulty_kv, faulty_conv = {
        "wrong_pass_kv": (wrong_pass.kv, stored.conv),
        "wrong_pass_conv": (stored.kv, wrong_pass.conv),
        "chunk_reset": (stored.kv[..., :0, :], torch.zeros_like(stored.conv)),
        "stale_prefix": (stale.kv, stale.conv),
    }[fault]
    faulty = _decode(_append_token(faulty_kv), faulty_conv, tail)
    assert torch.equal(faulty, torch.tensor(FAULTS[fault], device="cuda"))
    assert not torch.equal(faulty, correct)  # first-drift: mismatch against the clean oracle

    # Fault injection affects the copied state for this decode only.
    replay = cache.require(visit)
    assert torch.equal(correct, _decode(_append_token(replay.kv), replay.conv, tail))
    assert torch.equal(replay.kv, kv) and torch.equal(replay.conv, conv)
