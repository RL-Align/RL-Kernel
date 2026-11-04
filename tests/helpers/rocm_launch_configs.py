# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Row counts that cover every bucket of a shipped ROCm launch-config table."""

from rl_engine.backends.rocm import config_utils


def bucket_bounds(op: str, file_name: str, arch: str = "gfx942") -> list[int]:
    """M = x for every ``M_LEQ_x`` bucket of configs/<arch>/<op>/<file_name>.

    A batch-invariance test that runs these M values runs every bucket's config.
    The ``"any"`` entry must repeat the largest bucket, so it is covered too.
    """
    path = config_utils.CONFIG_ROOT / arch / op / file_name
    table = config_utils.load_table(path, ())
    if table is None:
        raise FileNotFoundError(path)
    bounds = sorted(int(key[len("M_LEQ_") :]) for key in table if key != "any")
    if table["any"] != table[f"M_LEQ_{bounds[-1]}"]:
        raise ValueError(f"{path}: 'any' differs from M_LEQ_{bounds[-1]}, so no M covers it")
    return bounds
