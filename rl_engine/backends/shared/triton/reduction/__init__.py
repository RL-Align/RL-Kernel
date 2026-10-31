# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Ordered reductions to embed inside another Triton kernel.

These helpers are for enclosing @triton.jit kernels. The optional gluon submodule
exposes matching functions for @gluon.jit callers. Import it explicitly; importing
this package does not load Gluon. Neither frontend selects the enclosing kernel.

sum/product accept either loaded numeric vectors or scalar pointers. Input type
selects the path at compile time. Optional length masks a loaded prefix without
resizing its tree; pointer input requires length and uses fixed-width tiles.
COMPUTE_DTYPE=None retains input precision. Pointer-only BLOCK_SIZE and
PIPELINE_STAGES configure loading; MAX_GROUP_SIZE sets the initial sequential groups.
Group totals follow one fixed lower-half/upper-half tree to a scalar.

sum_partials/product_partials fold stored tile results in ascending order.
All helpers return scalars without allocating, storing or launching kernels.
Callers own scheduling, producer dependencies and compiler math settings.
Generic stages live in ordered_reduce for custom integration and diagnostics.

reduce/reduce_partials expose the same implementations with caller-owned binary
functions. sum/product and their partial helpers supply standard operations and
neutral values. Custom operations need no registry or implementation-module import.
"""

from .ordered_reduce import product, product_partials, reduce, reduce_partials, sum, sum_partials

__all__ = ["sum", "product", "sum_partials", "product_partials", "reduce", "reduce_partials"]
