"""Ordered sum and product helpers to embed inside a Triton kernel.

sum_tile and product_tile reduce already-loaded or computed power-of-two vectors.
COMPUTE_DTYPE=None uses values.dtype; an explicit dtype converts values first.
Mask invalid lanes with zero or one respectively.
These helpers add no whole-vector seed and do not load from pointers.

sum_vector and product_vector take a pointer and length, handling loads, tiles,
tail padding and accumulation. COMPUTE_DTYPE=None uses the pointer's element type.
All tiles run in the calling program; BLOCK_SIZE does not limit total length.

sum_partials and product_partials take a pointer and count of already-computed
tile results, then merge them in order. They supply the operation, initial value
and storage dtype by default; callers need not configure a generic reducer.

All helpers return a scalar for the caller to use or store. The enclosing
operator owns allocation, launches and dependencies between producers and the
merge. Generic stages and binary combiners stay in the implementation module.
"""

from .ordered_reduce import (
    product_partials,
    product_tile,
    product_vector,
    sum_partials,
    sum_tile,
    sum_vector,
)

__all__ = [
    "sum_tile",
    "product_tile",
    "sum_vector",
    "product_vector",
    "sum_partials",
    "product_partials",
]
