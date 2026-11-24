# SPDX-License-Identifier: Apache-2.0
"""INK-50 ROCm fault injection at short-convolution state boundaries.

The faults (shifted recurrent state, reversed taps, document-history leakage)
are injected into inputs of the native ROCm kernel and must be detected against
both the exact analytic values and the Python reference; the clean control must
match the reference bitwise.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)

from rl_engine.backends.extension import _C, _EXT_AVAILABLE  # noqa: E402
from rl_engine.models.inkling2.short_conv import (  # noqa: E402
    STREAMS,
    causal_short_conv,
    causal_short_conv_rocm,
)

if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_short_conv_forward_with_state"):
    pytest.skip("Inkling 2 ROCm extension is unavailable", allow_module_level=True)


def _first_drift(a: torch.Tensor, b: torch.Tensor) -> int | None:
    """Index of the first time step where two [1, time, 1] outputs differ."""
    differs = (a != b).flatten().nonzero()
    return int(differs[0]) if differs.numel() else None


@pytest.mark.parametrize("stream", STREAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rocm_short_conv_detects_shifted_state_and_reversed_taps(stream, dtype):
    offset = STREAMS.index(stream) + 1
    tokens = (torch.arange(1, 8, device="cuda", dtype=dtype) * offset).view(1, 7, 1)
    weight = torch.tensor([[1, 2, 4, 8]], device="cuda", dtype=dtype)

    # Clean control: chunked ROCm execution is bitwise equal to the full run and
    # to the Python reference.
    full, full_state = causal_short_conv_rocm(tokens, weight)
    ref_full, ref_state = causal_short_conv(tokens, weight)
    assert torch.equal(full, ref_full)
    assert torch.equal(full_state, ref_state)
    prefix, state = causal_short_conv_rocm(tokens[:, :4], weight)
    chunk, chunk_state = causal_short_conv_rocm(tokens[:, 4:], weight, state)
    assert torch.equal(torch.cat((prefix, chunk), dim=1), full)
    assert torch.equal(chunk_state, full_state)

    # Fault 1: an off-by-one save includes the preceding token and drops the newest.
    shifted_history = tokens[:, :3].transpose(1, 2).contiguous()
    shifted, _ = causal_short_conv_rocm(tokens[:, 4:], weight, shifted_history)
    expected_shifted = torch.tensor([57, 76, 93], device="cuda", dtype=torch.float32)
    assert torch.equal(shifted.flatten(), (expected_shifted * offset).to(dtype))
    assert torch.all(chunk > shifted)
    # The mismatch is visible from the very first chunk token.
    assert _first_drift(shifted, chunk) == 0
    ref_shifted, _ = causal_short_conv(tokens[:, 4:], weight, shifted_history)
    assert torch.equal(shifted, ref_shifted)  # the kernel itself stays exact

    # Fault 2: reversed taps change the first chunk output.
    reversed_taps, _ = causal_short_conv_rocm(tokens[:, 4:], weight.flip(-1), state)
    assert reversed_taps[0, 0, 0].item() == 41 * offset
    assert chunk[0, 0, 0].item() == 64 * offset
    assert _first_drift(reversed_taps, chunk) == 0


@pytest.mark.parametrize("boundary", [1, 3, 5])
def test_rocm_short_conv_detects_document_history_leak(boundary):
    weight = torch.ones(1, 4, device="cuda")
    left = torch.full((1, boundary, 1), 3.0, device="cuda")
    right = torch.arange(1, 6, dtype=torch.float32, device="cuda").view(1, 5, 1)
    _, history = causal_short_conv_rocm(left, weight)
    isolated, isolated_state = causal_short_conv_rocm(right, weight)

    # Clean control: a fresh document run is bitwise equal to the reference and
    # chunked processing with a reset state reproduces it.
    ref_isolated, ref_isolated_state = causal_short_conv(right, weight)
    assert torch.equal(isolated, ref_isolated)
    assert torch.equal(isolated_state, ref_isolated_state)
    first_token, state = causal_short_conv_rocm(right[:, :1], weight)
    remainder, final_state = causal_short_conv_rocm(right[:, 1:], weight, state)
    assert torch.equal(torch.cat((first_token, remainder), dim=1), isolated)
    assert torch.equal(final_state, isolated_state)

    # Fault: packing two documents without resetting leaks the left history.
    packed, packed_state = causal_short_conv_rocm(torch.cat((left, right), dim=1), weight)
    leaked = packed[:, boundary:]
    ref_packed, _ = causal_short_conv(torch.cat((left, right), dim=1), weight)
    assert torch.equal(packed, ref_packed)  # the kernel is exact; the packing is the fault
    assert torch.equal(packed_state, isolated_state)  # state hides the leak, outputs do not

    expected_leak = torch.tensor(
        [3 * min(boundary, max(3 - t, 0)) for t in range(right.shape[1])],
        device="cuda",
        dtype=torch.float32,
    )
    assert torch.equal((leaked - isolated).flatten(), expected_leak)
    assert leaked[0, 0, 0].item() > isolated[0, 0, 0].item()
    # First drift is at the first token after the boundary.
    assert _first_drift(leaked, isolated) == 0
    # Feeding the saved left history into the next document is the same leak.
    leaked_chunk, _ = causal_short_conv_rocm(right, weight, history)
    assert not torch.equal(leaked_chunk, isolated)
    assert _first_drift(leaked_chunk, isolated) == 0
