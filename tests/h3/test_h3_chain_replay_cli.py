# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reject incomplete chain dependencies before checking CUDA or loading weights."""

from __future__ import annotations

import sys

import pytest

from scripts import h3_chain_replay

STAGE_NAMES = [stage.name for stage in h3_chain_replay.h3_chain.STAGES]


@pytest.fixture
def no_cuda_check(monkeypatch):
    """Fail if invalid CLI arguments reach a CUDA availability check."""

    def unexpected_check():
        """Report an environment check performed before argument rejection."""

        pytest.fail("invalid arguments must be rejected before checking CUDA")

    monkeypatch.setattr(h3_chain_replay.torch.cuda, "is_available", unexpected_check)


@pytest.mark.parametrize(
    "stages, message",
    [
        ("", "unknown stages"),
        ("unknown", "unknown stages"),
        (STAGE_NAMES[1], "ordered prefix"),
        (STAGE_NAMES[-1], "ordered prefix"),
        (",".join(STAGE_NAMES[:1] + STAGE_NAMES[2:]), "ordered prefix"),
        (",".join(reversed(STAGE_NAMES)), "ordered prefix"),
        (",".join([STAGE_NAMES[0], STAGE_NAMES[0]]), "ordered prefix"),
    ],
)
def test_reject_invalid_stages(monkeypatch, capsys, no_cuda_check, stages, message):
    """Reject unknown stages and selections that break chain dependencies."""

    monkeypatch.setattr(sys, "argv", ["h3_chain_replay.py", "--stages", stages])
    with pytest.raises(SystemExit) as exc:
        h3_chain_replay.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("count", range(1, len(STAGE_NAMES)))
def test_backward_requires_all_stages(monkeypatch, capsys, no_cuda_check, count):
    """Require the full chain for parameter-gradient replay."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["h3_chain_replay.py", "--stages", ",".join(STAGE_NAMES[:count]), "--backward"],
    )
    with pytest.raises(SystemExit) as exc:
        h3_chain_replay.main()
    assert exc.value.code == 2
    assert "--backward requires every stage" in capsys.readouterr().err


@pytest.mark.parametrize("count", range(1, len(STAGE_NAMES) + 1))
def test_accept_ordered_prefix(monkeypatch, count):
    """Allow every nonempty prefix to advance to the device requirement."""

    monkeypatch.setattr(h3_chain_replay.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys, "argv", ["h3_chain_replay.py", "--stages", ",".join(STAGE_NAMES[:count])]
    )
    with pytest.raises(SystemExit, match="the chain replay needs a CUDA device"):
        h3_chain_replay.main()


@pytest.mark.parametrize("options", [[], ["--backward"]])
def test_accept_all_stages_by_default(monkeypatch, options):
    """Keep the complete chain as the default for forward and backward replay."""

    monkeypatch.setattr(h3_chain_replay.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["h3_chain_replay.py", *options])
    with pytest.raises(SystemExit, match="the chain replay needs a CUDA device"):
        h3_chain_replay.main()
