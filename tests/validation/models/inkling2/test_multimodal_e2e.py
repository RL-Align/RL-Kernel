# SPDX-License-Identifier: Apache-2.0
import pytest

from rl_engine.validation.models.inkling2.multimodal_e2e import compare_multimodal


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_image_evidence_and_negative_controls(platform):
    common = dict(platform=platform, hardware="fixture", gpu_count=8, steps=1,
                  checkpoint_sha256="a"*64, dataset_sha256="b"*64, processor_sha256="c"*64,
                  recipe_sha256="d"*64, miles_revision="m", sglang_revision="s",
                  precision="bf16-dequant", vision_policy="frozen", capture_mode="real-weight")
    row = dict(step=1, image_input_sha256="e"*64, images=1, image_tokens=4,
               selected_tokens=8, feature_row_mismatches=0, bit_mismatches=0,
               logical_visits=48, fallback_free=True)
    native = {**common, "mode": "native", "step_metrics": [{**row, "bit_mismatches": 1}]}
    strict = {**common, "mode": "strict", "step_metrics": [dict(row)]}
    assert compare_multimodal(native, strict)["strict_zero_mismatch"]
    strict["step_metrics"][0]["feature_row_mismatches"] = 1
    assert not compare_multimodal(native, strict)["strict_zero_mismatch"]
    strict["step_metrics"][0]["images"] = 0
    with pytest.raises(ValueError, match="text-only"):
        compare_multimodal(native, strict)
