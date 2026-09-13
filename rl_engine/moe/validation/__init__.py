# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T09 comparison engine for P3 router validation (contract §2.5 / §4-T09).

Submodules:
- ``first_mismatch`` : locate the first divergence between two event streams
  by the six-tuple ``(absolute_layer, site, pass, event_index,
  global_token_id, rank)`` and attribute it to an owner task / issue.
- ``comparison``     : the four-stage ordered comparator
  (identity -> discrete -> score/weight -> gradient) with fail-closed gates.
"""
