# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Launch-config dispatch of the batch-invariant aiter ops (backends/rocm/config_utils.py).

Runs without a GPU: the arch is fixed to gfx942 and the tables are written to a
temporary directory. Checks aiter's M_LEQ_x rule, the fallback to the op's
default config, that a table whose entries differ in a parameter that changes
the bits or that has no "any" entry is rejected, and the record of the configs
that ran. The shipped
tables are checked with their ops.
"""

import json

import pytest
import torch

from rl_engine.backends.rocm import config_utils

_DEFAULT = {"NUM_KSPLIT": 1, "BLOCK_M": 32}
_BIT_PARAMS = ("NUM_KSPLIT",)
_FILE = "OP-N=8-K=64.json"
_DEVICE = torch.device("cuda", 0)


@pytest.fixture
def config_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config_utils, "_device_arch", lambda index: "gfx942")
    monkeypatch.setattr(config_utils, "CONFIG_ROOT", tmp_path)
    config_utils.load_table.cache_clear()
    config_utils._bucket_table.cache_clear()
    config_utils._SELECTED.clear()
    yield tmp_path
    config_utils.load_table.cache_clear()
    config_utils._bucket_table.cache_clear()
    config_utils._SELECTED.clear()


def _write(root, table, arch="gfx942"):
    op_dir = root / arch / "op"
    op_dir.mkdir(parents=True)
    (op_dir / _FILE).write_text(json.dumps(table))


def _get(m, file_name=_FILE):
    return config_utils.get_config("op", file_name, m, _DEFAULT, _BIT_PARAMS, _DEVICE)


def test_select_entry_follows_aiter_gemm_rule():
    table = {"M_LEQ_1": {"x": 1}, "M_LEQ_256": {"x": 256}, "any": {"x": "any"}}
    chosen = [config_utils.select_entry(table, m)["x"] for m in (0, 1, 2, 256, 257)]
    assert chosen == [1, 1, 256, 256, "any"]


def test_table_for_the_shape_is_used(config_root):
    _write(config_root, {"M_LEQ_16": dict(_DEFAULT, BLOCK_M=16), "any": _DEFAULT})
    assert _get(1) == dict(_DEFAULT, BLOCK_M=16)
    assert _get(17) == _DEFAULT


def test_other_shape_or_arch_uses_default(config_root, monkeypatch):
    _write(config_root, {"M_LEQ_16": dict(_DEFAULT, BLOCK_M=16), "any": _DEFAULT})
    assert _get(1, "OP-N=8-K=32.json") == _DEFAULT
    monkeypatch.setattr(config_utils, "_device_arch", lambda index: "gfx950")
    assert _get(1) == _DEFAULT


def test_returned_config_is_a_copy(config_root):
    _get(1)["BLOCK_M"] = 0
    assert _get(1) == _DEFAULT


def test_table_whose_entries_differ_in_a_bit_param_is_rejected(config_root):
    _write(config_root, {"M_LEQ_1": _DEFAULT, "any": dict(_DEFAULT, NUM_KSPLIT=2)})
    with pytest.raises(ValueError, match="NUM_KSPLIT"):
        _get(1)


def test_table_with_an_unknown_key_is_rejected(config_root):
    _write(config_root, {"M_LEQ_1": _DEFAULT, "M_GEQ_2": _DEFAULT, "any": _DEFAULT})
    with pytest.raises(ValueError, match="M_GEQ_2"):
        _get(1)


def test_table_without_any_entry_is_rejected(config_root):
    _write(config_root, {"M_LEQ_1": _DEFAULT, "M_LEQ_16": _DEFAULT})
    with pytest.raises(ValueError, match=f"{_FILE}.*'any'"):
        _get(1)


def test_selected_configs_record_what_ran(config_root):
    _write(config_root, {"M_LEQ_16": dict(_DEFAULT, BLOCK_M=16), "any": _DEFAULT})
    for m in (5, 1, 16, 17, 300):
        _get(m)
    for m in (3, 1000):
        _get(m, "OP-N=8-K=32.json")
    source = str(config_root / "gfx942" / "op" / _FILE)
    records = config_utils.selected_configs()
    assert records == [
        {
            "arch": "gfx942",
            "op": "op",
            "file": _FILE,
            "bucket": "M_LEQ_16",
            "source": source,
            "config": dict(_DEFAULT, BLOCK_M=16),
            "m_min": 1,
            "m_max": 16,
        },
        {
            "arch": "gfx942",
            "op": "op",
            "file": _FILE,
            "bucket": "any",
            "source": source,
            "config": _DEFAULT,
            "m_min": 17,
            "m_max": 300,
        },
        {
            "arch": "gfx942",
            "op": "op",
            "file": "OP-N=8-K=32.json",
            "bucket": "default",
            "source": "default",
            "config": _DEFAULT,
            "m_min": 3,
            "m_max": 1000,
        },
    ]
    records[0]["config"]["BLOCK_M"] = 0
    assert _get(1) == dict(_DEFAULT, BLOCK_M=16)
