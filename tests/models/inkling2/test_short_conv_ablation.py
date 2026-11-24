# SPDX-License-Identifier: Apache-2.0
"""INK-50 CUDA fault injection at short-convolution state boundaries."""

import pytest
import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2.short_conv import STREAMS, causal_short_conv_cuda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_short_conv_forward_with_state"),
    reason="requires the native INK-10 short-convolution extension",
)


@pytest.mark.parametrize("stream", STREAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_short_conv_detects_shifted_state_and_reversed_taps(stream, dtype):
    offset = STREAMS.index(stream) + 1
    tokens = (torch.arange(1, 8, device="cuda", dtype=dtype) * offset).view(1, 7, 1)
    weight = torch.tensor([[1, 2, 4, 8]], device="cuda", dtype=dtype)

    full, full_state = causal_short_conv_cuda(tokens, weight)
    prefix, state = causal_short_conv_cuda(tokens[:, :4], weight)
    chunk, chunk_state = causal_short_conv_cuda(tokens[:, 4:], weight, state)
    assert torch.equal(torch.cat((prefix, chunk), dim=1), full)
    assert torch.equal(chunk_state, full_state)

    # An off-by-one save includes the preceding token and drops the newest.
    shifted_history = tokens[:, :3].transpose(1, 2)
    shifted, _ = causal_short_conv_cuda(tokens[:, 4:], weight, shifted_history)
    reversed_taps, _ = causal_short_conv_cuda(tokens[:, 4:], weight.flip(-1), state)
    expected_shifted = torch.tensor([57, 76, 93], device="cuda", dtype=torch.float32)
    assert torch.equal(shifted.flatten(), (expected_shifted * offset).to(dtype))
    assert torch.all(chunk > shifted)
    assert reversed_taps[0, 0, 0].item() == 41 * offset
    assert chunk[0, 0, 0].item() == 64 * offset


@pytest.mark.parametrize("boundary", [1, 3, 5])
def test_short_conv_detects_document_history_leak(boundary):
    weight = torch.ones(1, 4, device="cuda")
    left = torch.full((1, boundary, 1), 3.0, device="cuda")
    right = torch.arange(1, 6, dtype=torch.float32, device="cuda").view(1, 5, 1)
    _, history = causal_short_conv_cuda(left, weight)
    isolated, isolated_state = causal_short_conv_cuda(right, weight)

    packed, packed_state = causal_short_conv_cuda(torch.cat((left, right), dim=1), weight)
    leaked = packed[:, boundary:]
    first_token, state = causal_short_conv_cuda(right[:, :1], weight)
    remainder, final_state = causal_short_conv_cuda(right[:, 1:], weight, state)
    assert torch.equal(torch.cat((first_token, remainder), dim=1), isolated)
    assert torch.equal(final_state, isolated_state)
    assert torch.equal(packed_state, isolated_state)

    expected_leak = torch.tensor(
        [3 * min(boundary, max(3 - t, 0)) for t in range(right.shape[1])],
        device="cuda",
        dtype=torch.float32,
    )
    assert torch.equal((leaked - isolated).flatten(), expected_leak)
    assert leaked[0, 0, 0].item() > isolated[0, 0, 0].item()
