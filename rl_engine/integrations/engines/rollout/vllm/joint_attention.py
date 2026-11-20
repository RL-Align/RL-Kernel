# SPDX-License-Identifier: Apache-2.0
"""Explicit shared forward/packed-cache boundary; no global engine monkeypatch."""

from rl_engine.models.deepseek_v4.joint_attention import (
    JointAttentionPlan,
    JointMetadata,
    gather_packed_joint_kv,
)

__all__ = ["JointAttentionPlan", "JointMetadata", "gather_packed_joint_kv"]
