# SPDX-License-Identifier: Apache-2.0
"""Native DSV4 normalized Hadamard128 with a self-adjoint input VJP."""

import torch


def rotate128(values, native=None):
    """Same physical 8-row tail allocation as the existing Indexer C4 provider."""
    if values.dtype != torch.bfloat16 or values.shape[-1] != 128:
        raise ValueError("native DSV4 rotation requires BF16 vectors of width128")
    if not values.numel():
        return torch.empty_like(values)
    if native is None:
        from aiter.ops.quant import rotate_activation

        native = rotate_activation
    shape = values.shape
    flat = values.reshape(-1, 128)
    rows = flat.shape[0]
    extent = (rows + 7) // 8 * 8
    source = (
        flat.contiguous()
        if extent == rows
        else torch.zeros((extent, 128), dtype=flat.dtype, device=flat.device)
    )
    if extent != rows:
        source[:rows].copy_(flat)
    destination = torch.empty_like(source)
    native(destination, source)
    return destination[:rows].reshape(shape)


class _Hadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return rotate128(values)

    @staticmethod
    def backward(ctx, grad):
        # Normalized Walsh-Hadamard128 is symmetric and orthogonal.
        return rotate128(grad)


def native_hadamard128(values):
    result = _Hadamard.apply(values)
    return result
