"""Eager references matching the Triton ordered-reduction package exports.

sum_tile and product_tile reduce a power-of-two last dimension; callers supply
neutral padding. compute_dtype=None uses values.dtype; an explicit dtype casts first.
sum_vector and product_vector handle arbitrary lengths, tiling and padding;
compute_dtype=None uses the input dtype. Leading dimensions batch independent
reductions. Run eagerly and use no_grad when inputs require gradients.

sum_partials and product_partials merge stored tile results with only the tensor
required. Operation, initial value and storage dtype are supplied by default.
Generic stages and combiners live in ordered_reduce for diagnostic tests.
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
