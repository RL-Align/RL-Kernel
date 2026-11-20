# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 text contracts, kept distinct from the V4 execution profile."""

from .config import DeepSeekV41Config, LayerPlan

__all__ = ["DeepSeekV41Config", "LayerPlan", "DeepSeekV41TextModel"]


def __getattr__(name):
    # Keep configuration/checkpoint tooling usable without importing GPU kernels.
    if name == "DeepSeekV41TextModel":
        from .model import DeepSeekV41TextModel

        return DeepSeekV41TextModel
    raise AttributeError(name)
