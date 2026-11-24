# SPDX-License-Identifier: Apache-2.0
"""INK-46: shared-parameter gradient order and ownership diagnostics."""

def validate_shared_owner(owners: tuple[int, int, int]) -> None:
    if len(owners) != 3 or len(set(owners)) != 1:
        raise ValueError("core visits must share one physical weight owner")


def gradient_order_probe(gradients, *, fold, order=(0, 2, 1), shard=None):
    if sorted(order) != [0, 1, 2]:
        raise ValueError("gradient visit coverage is incomparable")
    return fold(gradients, visit_order=order, shard=shard)
