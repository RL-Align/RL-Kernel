# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T02 M10 scoring wiring: strict linear_logp reuse, raw bytes, fail-closed.

The scoring path consumes the existing strict SM90 linear_logp, so the positive
cases need the compiled extension on a CUDA device; every fail-closed case is
CPU-safe because it never reaches the kernel.
"""

import pytest
import torch

from rl_engine.validation.m10 import M10Status, build_m10_stub, score_m10

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="strict M10 scoring reuses the SM90 linear_logp and needs a CUDA device",
)


def _sm90_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE

        if not (_EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90")):
            return False
    except Exception:  # pragma: no cover
        return False
    return torch.cuda.get_device_capability()[0] == 9


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(),
    reason="strict M10 scoring needs the SM90 extension built with KERNEL_ALIGN_FORCE_SM90=1",
)


class TestM10FailClosed:
    """Fail-closed paths never touch the kernel, so they run without a GPU."""

    def test_lm_head_lora_returns_unsupported_without_scoring(self):
        result = score_m10(build_m10_stub(lm_head_lora=True))
        assert result.status is M10Status.UNSUPPORTED_CAPABILITY
        assert not result.passed
        assert result.selected_logprob is None
        assert result.selected_logprob_bytes == b""

    def test_weight_lock_drift_returns_before_scoring(self):
        result = score_m10(build_m10_stub(), expected_pre_update_weight_lock="other-lock")
        assert result.status is M10Status.PRE_UPDATE_WEIGHT_DRIFT
        assert result.selected_logprob_bytes == b""


@requires_sm90
class TestM10Scoring:
    def _cuda_input(self, **kwargs):
        return build_m10_stub(**kwargs)

    def test_stub_scores_active_tokens_only(self):
        inp = self._cuda_input(tokens=8, hidden=64, local_vocab=128, active_ratio=0.5)
        device = torch.device("cuda")
        result = score_m10(inp, device=device)
        assert result.status is M10Status.PASS
        assert result.passed
        assert result.active_token_count == 4
        assert result.selected_logprob is not None
        assert result.selected_logprob.shape == (8,)
        assert result.selected_logprob.dtype == torch.float32
        # exactly one raw FP32 per active token
        assert len(result.selected_logprob_bytes) == 4 * result.active_token_count

    def test_repeat_runs_are_byte_identical(self):
        inp = self._cuda_input(tokens=8, hidden=64, local_vocab=128)
        device = torch.device("cuda")
        first = score_m10(inp, device=device)
        second = score_m10(inp, device=device)
        assert first.selected_logprob_bytes == second.selected_logprob_bytes

    def test_provenance_records_the_frozen_contract(self):
        inp = self._cuda_input(tokens=4, hidden=64, local_vocab=128)
        result = score_m10(inp, device=torch.device("cuda"))
        provenance = dict(result.provenance)
        assert provenance["vocab_reduction_tile"] == 64
        assert provenance["actual_backend"] == "rlkernel.linear_logp.bitwise.v1"
        assert provenance["vocab_shard"] == [0, 128]
        assert provenance["sampling_record_fingerprint"] is not None
        # the frozen arithmetic contract travels with the result
        assert provenance["arithmetic_contract"]["output_dtype"] == "fp32"
        assert provenance["arithmetic_contract"]["vocab_group_width"] == 64

    def test_result_serializes_with_hex_bytes(self):
        inp = self._cuda_input(tokens=2, hidden=32, local_vocab=64)
        result = score_m10(inp, device=torch.device("cuda"))
        payload = result.to_dict()
        assert payload["selected_logprob_bytes"] == result.selected_logprob_bytes_hex()
        assert len(payload["selected_logprob_bytes"]) == 2 * 4 * 2
