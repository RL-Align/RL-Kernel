# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Launch-config dispatch for the batch-invariant aiter ops, modelled on aiter's.

aiter picks a kernel's launch config from JSON files (``get_gemm_config`` in
aiter/ops/triton/utils/gemm_config_utils.py): a file per arch and op, a
shape-specialized file when one exists, and inside it the ``M_LEQ_x`` bucket
for M. aiter does not promise that any of its configs are batch invariant, so
RL-Kernel keeps its own files, laid out the same way:

    configs/<arch>/<op>/<FILE>.json      e.g. configs/gfx942/mhc_pre/MHC_PRE-n=4-C=4096.json

- A file holds one tuned table: ``M_LEQ_x`` entries and an ``"any"`` entry. The
  entry for M is the smallest x with M <= x (aiter's GEMM rule), else ``"any"``.
- Without a file for the input's GPU arch and the exact shape, the op's
  default config is used for every M (aiter's DEFAULT.json / code fallback).

Batch invariance needs every M of a shape to use configs that give the same
output bits. Each op names the parameters that change the bits (measured on
gfx942, see the op's page under docs/operators/); every entry of a file must
have the same value for each of them, which is checked when the file is loaded.
"""

from __future__ import annotations

import bisect
import functools
import json
from pathlib import Path

import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import _device_arch

CONFIG_ROOT = Path(__file__).with_name("configs")
_BUCKET = "M_LEQ_"


def gpu_arch(device: torch.device) -> str:
    """The arch of the GPU ``device`` is on, e.g. "gfx942"."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _device_arch(index)


@functools.lru_cache(maxsize=None)
def load_table(path: Path, bit_params: tuple[str, ...]) -> dict | None:
    """The table at ``path``, or None if there is no such file.

    Raises ValueError for a key other than ``M_LEQ_x`` / ``"any"``, for a table
    without an ``"any"`` entry, or if two entries differ in one of ``bit_params``.
    """
    if not path.is_file():
        return None
    with open(path) as f:
        table = json.load(f)
    for key in table:
        if key != "any" and not (key.startswith(_BUCKET) and key[len(_BUCKET) :].isdigit()):
            raise ValueError(f"{path}: key {key!r} is not M_LEQ_<x> or 'any'")
    if "any" not in table:
        raise ValueError(f"{path}: no 'any' entry for M above the largest M_LEQ_<x>")
    for param in bit_params:
        values = {key: config.get(param) for key, config in table.items()}
        if len(set(values.values())) > 1:
            raise ValueError(
                f"{path}: {param} changes the output bits, so every entry must have the "
                f"same value; got {values}"
            )
    return table


def select_entry(table: dict, m: int) -> dict:
    """aiter's GEMM rule: the smallest ``M_LEQ_x`` with ``m <= x``, else ``"any"``."""
    for bound in sorted(int(key[len(_BUCKET) :]) for key in table if key != "any"):
        if m <= bound:
            return dict(table[f"{_BUCKET}{bound}"])
    return dict(table["any"])


@functools.lru_cache(maxsize=None)
def _bucket_table(
    arch: str, op: str, file_name: str, bit_params: tuple[str, ...]
) -> tuple[str, dict | None, tuple[int, ...]]:
    """Source path or "default", the table or None, and its sorted ``M_LEQ_x`` bounds."""
    path = CONFIG_ROOT / arch / op / file_name
    table = load_table(path, bit_params)
    if table is None:
        return "default", None, ()
    bounds = tuple(sorted(int(key[len(_BUCKET) :]) for key in table if key != "any"))
    return str(path), table, bounds


# (arch, op, file, bucket) -> record of the chosen config; see selected_configs().
# The config depends on M only through its bucket, so this stays as small as the tables.
_SELECTED: dict[tuple[str, str, str, str], dict] = {}


def get_config(
    op: str,
    file_name: str,
    m: int,
    default: dict,
    bit_params: tuple[str, ...],
    device: torch.device,
) -> dict:
    """Launch config for ``m`` rows on ``device``'s GPU: from
    configs/<arch>/<op>/<file_name> if it exists, else ``default``. Called on every
    launch, so the choice is cached per (arch, op, file, bucket)."""
    arch = gpu_arch(device)
    source, table, bounds = _bucket_table(arch, op, file_name, bit_params)
    if table is None:
        bucket = "default"
    else:
        # aiter's GEMM rule, as in select_entry: the smallest x with m <= x, else "any".
        i = bisect.bisect_left(bounds, m)
        bucket = f"{_BUCKET}{bounds[i]}" if i < len(bounds) else "any"
    key = (arch, op, file_name, bucket)
    record = _SELECTED.get(key)
    if record is None:
        record = {
            "arch": arch,
            "op": op,
            "file": file_name,
            "bucket": bucket,
            "source": source,
            "config": dict(default) if table is None else dict(table[bucket]),
            "m_min": m,
            "m_max": m,
        }
        _SELECTED[key] = record
    elif m < record["m_min"]:
        record["m_min"] = m
    elif m > record["m_max"]:
        record["m_max"] = m
    return dict(record["config"])


def selected_configs() -> list[dict]:
    """Every launch config chosen in this process, one record per bucket: GPU arch, op,
    table file, bucket (``M_LEQ_x``, "any", or "default" without a table), the table
    path or "default", the config, and the smallest and largest M that used it.
    Records which configs actually ran."""
    return [dict(record, config=dict(record["config"])) for record in _SELECTED.values()]
