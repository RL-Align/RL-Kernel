# SPDX-License-Identifier: Apache-2.0
import pytest

from rl_engine.validation.models.inkling2.performance import compare_performance


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_measured_ratio(platform):
    a = dict(platform=platform, hardware="fixture", gpu_count=8, ep=8, rollout_tp=1,
             checkpoint_sha256="a", recipe_sha256="b", dataset_sha256="c",
             miles_revision="m", sglang_revision="s", steps=2, warmup_steps=1,
             precision="bf16-dequant", mode="native", capture_mode="real-weight",
             samples=[dict(generated_tokens=100, step_seconds=2, prefill_ms=4,
                           decode_ms=5, peak_memory_bytes=1000)])
    b = {**a, "mode": "strict", "samples": [{**a["samples"][0], "step_seconds": 2.5}]}
    result = compare_performance(a, b)
    assert result["strict_over_native"]["tokens_per_second"] == .8
    assert not result["throughput_target_met"] and not result["bitwise_validated"]
    b["gpu_count"] = 4
    with pytest.raises(ValueError, match="unmatched"):
        compare_performance(a, b)
