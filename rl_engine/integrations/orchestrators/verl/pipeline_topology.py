# SPDX-License-Identifier: Apache-2.0
"""Validate local worker slices of the opt-in PP2 rollout replica."""


def validate_pp2_layout(*, nnodes, node_rank, gpus_per_node, visible_devices):
    """Allow 1x2 or 2x1 only; each server owns exactly its local GPU slice."""
    ids = visible_devices.split(",")
    if (
        type(nnodes) is not int
        or type(node_rank) is not int
        or type(gpus_per_node) is not int
        or (nnodes, gpus_per_node) not in ((1, 2), (2, 1))
        or not 0 <= node_rank < nnodes
        or len(ids) != gpus_per_node
        or len(set(ids)) != len(ids)
        or any(not item.isdecimal() for item in ids)
    ):
        raise ValueError("PP2 requires 1x2 or 2x1 GPUs with distinct local device IDs")
