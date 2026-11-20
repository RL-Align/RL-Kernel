# SPDX-License-Identifier: Apache-2.0
"""Logical ownership shared by checkpoint loading and pipeline state synchronization."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DSV4CheckpointPartition:
    layer_ids: tuple[int, ...]
    pre_process: bool
    post_process: bool
    num_layers: int

    def owns(self, key):
        match = re.match(r"^layers\.(\d+)\.", key)
        if match:
            layer = int(match[1])
            if layer >= self.num_layers:
                raise ValueError(f"checkpoint layer outside model: {key}")
            return layer in self.layer_ids
        if key == "embed.weight":
            return self.pre_process
        if key in ("head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"):
            return self.post_process
        raise ValueError(f"unknown checkpoint ownership: {key}")
