# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""P3 router package (DSV4-Flash MoE router, contract `p3-task-selection.md`).

Layout sanctioned by the P3 contract §2.1:
- ``naive_topk6``  : T09-owned total-order Top-6 checker (cross-checks T01 golden)
- ``router_torch_reference.py`` : T01-owned raw Torch reference (to be published
  with the start kit; do not duplicate or silently replace it)
"""
