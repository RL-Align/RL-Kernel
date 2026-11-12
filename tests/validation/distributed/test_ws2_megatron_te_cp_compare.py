# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from tools.validation.distributed.ws2_megatron_te_cp_compare import (
    _compare_runs,
    _parse_cp_sizes,
    _validate_args,
)


def _artifact(path: Path, *, offset: float) -> Path:
    token_ids = [42, 7, 8]
    path.write_text(
        json.dumps(
            {
                "token_ids": token_ids,
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("ascii")
                ).hexdigest(),
                "active_token_logprobs": [
                    {"position": 1, "token_id": 7, "logprob": -1.0 + offset},
                    {"position": 2, "token_id": 8, "logprob": -2.0 + offset},
                ],
            }
        )
    )
    return path


def test_native_te_comparison_is_ordered_and_records_worst_token(tmp_path):
    left = _artifact(tmp_path / "cp1.json", offset=0.0)
    right = _artifact(tmp_path / "cp2.json", offset=0.02)

    report = _compare_runs(
        [
            {"cp_size": 1, "output": str(left)},
            {"cp_size": 2, "output": str(right)},
        ],
        atol=0.05,
    )

    assert report["pass"] is True
    assert report["max_abs"] == pytest.approx(0.02)
    assert report["worst"]["position"] == 1
    assert report["token_ids_sha256"]


def test_native_te_comparison_rejects_token_identity_mismatch(tmp_path):
    left = _artifact(tmp_path / "cp1.json", offset=0.0)
    right = _artifact(tmp_path / "cp2.json", offset=0.02)
    payload = json.loads(right.read_text())
    payload["active_token_logprobs"][1]["token_id"] = 99
    right.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="token IDs"):
        _compare_runs(
            [
                {"cp_size": 1, "output": str(left)},
                {"cp_size": 2, "output": str(right)},
            ],
            atol=0.05,
        )


def test_native_te_comparison_rejects_position_mismatch(tmp_path):
    left = _artifact(tmp_path / "cp1.json", offset=0.0)
    right = _artifact(tmp_path / "cp2.json", offset=0.02)
    payload = json.loads(right.read_text())
    payload["active_token_logprobs"][1]["position"] = 3
    right.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="positions"):
        _compare_runs(
            [
                {"cp_size": 1, "output": str(left)},
                {"cp_size": 2, "output": str(right)},
            ],
            atol=0.05,
        )


def test_native_te_comparison_rejects_invalid_token_hash(tmp_path):
    left = _artifact(tmp_path / "cp1.json", offset=0.0)
    right = _artifact(tmp_path / "cp2.json", offset=0.02)
    payload = json.loads(right.read_text())
    payload["token_ids_sha256"] = "0" * 64
    right.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="token_ids_sha256"):
        _compare_runs(
            [
                {"cp_size": 1, "output": str(left)},
                {"cp_size": 2, "output": str(right)},
            ],
            atol=0.05,
        )


def test_cp_sizes_require_exactly_cp1_and_cp2():
    assert _parse_cp_sizes("1,2") == (1, 2)

    for value in ("2", "1,2,4", "0,2", "2,1", "a,2"):
        try:
            _parse_cp_sizes(value)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid cp-sizes to fail: {value}")


def test_native_te_rejects_nonfinite_tolerance(tmp_path):
    args = Namespace(
        teacher_script=tmp_path / "teacher.py",
        model=tmp_path / "model",
        token_artifact=tmp_path / "tokens.json",
        tensor_parallel_size=2,
        atol=float("nan"),
    )
    args.teacher_script.touch()
    args.model.mkdir()
    args.token_artifact.touch()

    with pytest.raises(ValueError, match="finite and non-negative"):
        _validate_args(args, (1, 2))
