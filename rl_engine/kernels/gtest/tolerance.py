# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CONTRACT_PATH = Path(__file__).with_name("tolerance_contract.yaml")


def load_contract(path: str | Path = _CONTRACT_PATH) -> dict[str, Any]:
    """Load the dtype/operator-class tolerance contract."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["load_contract"]
