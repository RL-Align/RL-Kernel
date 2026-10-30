# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MoE router package (DSV4-Flash MoE router).

Layout:
- ``t01_verdicts`` : frozen verdict-code table + priority arbitration
  (the shared vocabulary every layer references)
- ``naive_topk``      : total-order Top-K checker (cross-checks the
  stable Top-K golden; K-agnostic, K=6 is only the DSV4-Flash default)
- ``validation/``     : the validation engine — canonical fingerprints,
  first-mismatch localization, the ordered comparator, validation runners
  (WS1 + WS2), paired diagnostics, the synthetic producer
- ``router_torch_reference.py`` : raw Torch reference (to be published
  with the start kit; do not duplicate or silently replace it)
"""

from .t01_verdicts import RouterVerdict

__all__ = ["RouterVerdict"]
