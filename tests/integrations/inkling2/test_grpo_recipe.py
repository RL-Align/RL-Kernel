# SPDX-License-Identifier: Apache-2.0
import pytest

from rl_engine.integrations.orchestrators.miles.inkling2_recipe import plan


def test_pinned_native_plan(tmp_path):
    run = plan(tmp_path, tmp_path, tmp_path, tmp_path / "a",
               platform="cuda", mode="native", steps=100)
    assert run["manifest"]["ep"] == 8
    assert run["manifest"]["response_length"] == 4096
    assert "execute(ScriptArgs" in run["argv"][2]
    assert "prepare(" not in run["argv"][2]


@pytest.mark.parametrize("platform,mode", [("cuda", "strict"), ("rocm", "native"),
                                          ("rocm", "strict")])
def test_external_runtime_gate(tmp_path, platform, mode):
    args = (tmp_path, tmp_path, tmp_path, tmp_path / "b")
    with pytest.raises(RuntimeError, match="qualified"):
        plan(*args, platform=platform, mode=mode, steps=200)
    run = plan(*args, platform=platform, mode=mode, steps=200,
               qualified_entry=tmp_path / "worker.py")
    assert run["manifest"]["platform"] == platform
    assert run["argv"][-2] == "--manifest"
