# SPDX-License-Identifier: Apache-2.0
import pytest

from rl_engine.validation.models.inkling2.e2e_acceptance import validate_pair


def pair(platform, steps=1):
    common = dict(platform=platform, hardware="B200" if platform == "cuda" else "MI355X",
                  gpu_count=8, ep=8, rollout_tp=1, steps=steps, precision="bf16-dequant",
                  checkpoint_sha256="a"*64, dataset_sha256="b"*64, recipe_sha256="c"*64,
                  miles_revision="miles", sglang_revision="sglang", capture_mode="real-weight")
    def report(mode):
        return {**common, "mode": mode, "step_metrics": [
            dict(step=i+1, replay_sha256="d"*64, selected_tokens=16,
                 bit_mismatches=2 if mode == "native" else 0, loss=.2, reward=.5,
                 kl=.01, logical_visits=48, fallback_free=True) for i in range(steps)]}
    return report("native"), report("strict")


def test_rocm_stage_and_drift():
    native, strict = pair("rocm", 10)
    r = validate_pair(native, strict, platform="rocm", steps=10)
    assert r["strict_zero_mismatch"] and not r["independently_certified"]
    strict["step_metrics"][0]["bit_mismatches"] = 1
    assert not validate_pair(native, strict, platform="rocm", steps=10)["strict_zero_mismatch"]


def test_rocm_reject_incomplete_or_fallback():
    native, strict = pair("rocm")
    strict["step_metrics"][0]["fallback_free"] = False
    with pytest.raises(ValueError, match="fallback"):
        validate_pair(native, strict, platform="rocm", steps=1)
    strict["step_metrics"].clear()
    with pytest.raises(ValueError, match="incomplete"):
        validate_pair(native, strict, platform="rocm", steps=1)
