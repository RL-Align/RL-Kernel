# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Comparison engine for MoE router validation.

Submodules:
- ``first_mismatch``     : locate the first divergence between two event
  streams by the six-tuple ``(absolute_layer, site, pass, event_index,
  global_token_id, rank)`` and attribute it to the owning component.
- ``comparison``         : the four-stage ordered comparator
  (identity -> discrete -> score/weight -> gradient) with fail-closed gates.
- ``t01_fingerprint``        : canonical serialization + semantic/artifact hashes.
- ``runner``             : the shared ``ValidationReport`` type plus every
  validation-stage runner — L1/L2/L3a/L3b (WS1) and rank completeness +
  cross-config ownership checks (WS2).
- ``paired_check``       : Torch paired diagnostics (anchor-pending gated).
- ``t01_synthetic_producer`` : T01 seeded deterministic fixtures.
"""
