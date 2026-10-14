# SPDX-License-Identifier: Apache-2.0
"""Stable v1 WS1 operator identifiers, independent of physical module placement."""

import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def _module_names():
    return json.loads(files(__package__).joinpath("module_names.json").read_text())


def stable_operator_path(value):
    cls = type(value)
    module = _module_names().get(cls.__module__, cls.__module__)
    return f"{module}.{cls.__qualname__}"
