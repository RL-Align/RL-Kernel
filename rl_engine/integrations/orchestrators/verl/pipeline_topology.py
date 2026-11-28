# SPDX-License-Identifier: Apache-2.0
"""Validate local worker slices of the opt-in pipeline rollout replica."""


def validate_pp2_layout(*, nnodes, node_rank, gpus_per_node, visible_devices, stages=2):
    """One GPU per pipeline stage; each server owns exactly its local GPU slice.

    `stages` is the rollout pipeline size (default 2: 1x2 or 2x1 GPUs).
    """
    ids = visible_devices.split(",")
    if (
        type(nnodes) is not int
        or type(node_rank) is not int
        or type(gpus_per_node) is not int
        or type(stages) is not int
        or nnodes < 1
        or gpus_per_node < 1
        or nnodes * gpus_per_node != stages
        or not 0 <= node_rank < nnodes
        or len(ids) != gpus_per_node
        or len(set(ids)) != len(ids)
        or any(not item.isdecimal() for item in ids)
    ):
        raise ValueError(
            f"PP{stages} requires {stages} GPUs in equal node slices with distinct local device IDs"
        )
