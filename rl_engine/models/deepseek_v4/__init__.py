# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 on the batch-invariant SM90 kernel set.

- ``attention``: one attention layer (T07), paged caches, :class:`Dsv4Batch`.
- ``router`` / ``moe``: the MoE block (P3 router, P5 experts, P6 combine), with dX.
- ``model``: decoder layers with mHC, and the model forward (``Dsv4Model``).
- ``loader``: build the model from a Hugging Face DeepSeek-V4 checkpoint.
"""
