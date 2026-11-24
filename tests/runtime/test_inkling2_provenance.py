# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.runtime.provenance.inkling2 import VisitEvidence, read_topology


def test_visit_evidence():
    ev = VisitEvidence({"platform": "rocm"})
    for j in range(48):
        ev.record(j, backend="test", kernel="test", cache_slot=j, conv_slot=j,
                  splits=1, fallback=None)
    assert ev.report()["complete"] and ev.report()["fallback_free"]
    assert not ev.report()["qualified"]
    with pytest.raises(ValueError):
        ev.record(0, backend="test", kernel="test", cache_slot=0, conv_slot=0,
                  splits=1, fallback=None)
    with pytest.raises(ValueError):
        read_topology(torch.device("cpu"), {}, platform="cuda")


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_gpu_readback(platform):
    if not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm"):
        pytest.skip(f"{platform} required")
    report = read_topology(torch.device("cuda:0"), {"tp": None}, platform=platform)
    assert report["platform"] == platform and report["groups"]["tp"]["size"] == 1
    assert report["architecture"].startswith("gfx" if platform == "rocm" else "sm")
