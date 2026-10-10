# SPDX-License-Identifier: Apache-2.0
"""Bijective sequence reorders without a generic scatter-add backward."""

import torch


def _index(tensor, order, sequence_dim, batch_dim):
    shape = [1] * tensor.ndim
    shape[sequence_dim] = order.size(1)
    shape[batch_dim] = order.size(0)
    values = order if batch_dim < sequence_dim else order.transpose(0, 1)
    return values.reshape(shape).expand_as(tensor)


class _SequencePermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, order, sequence_dim, batch_dim):
        ctx.save_for_backward(order)
        ctx.sequence_dim, ctx.batch_dim = sequence_dim, batch_dim
        return torch.gather(tensor, sequence_dim, _index(tensor, order, sequence_dim, batch_dim))

    @staticmethod
    def backward(ctx, gradient):
        (order,) = ctx.saved_tensors
        inverse = torch.argsort(order, dim=1)
        result = torch.gather(
            gradient,
            ctx.sequence_dim,
            _index(gradient, inverse, ctx.sequence_dim, ctx.batch_dim),
        )
        # Generic gather backward adds into a zero-initialized tensor. Keep
        # that addition's signed-zero semantics without sorting tensor-wide
        # scatter indices or performing any multi-contributor reduction.
        return result + 0, None, None, None


def permute_sequence(tensor, order, *, sequence_dim, batch_dim):
    """Apply an internal argsort permutation, one per batch element.

    Callers supply a complete bijection obtained from argsort, never arbitrary
    gather indices. The backward is its inverse and has one source per output.
    """
    if torch.is_grad_enabled() and tensor.requires_grad:
        return _SequencePermutation.apply(tensor, order, sequence_dim, batch_dim)
    return torch.gather(tensor, sequence_dim, _index(tensor, order, sequence_dim, batch_dim))
