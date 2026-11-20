# SPDX-License-Identifier: Apache-2.0
"""Public custom callbacks across loaded, pointer and parallel merge paths."""

from functools import reduce as scalar_reduce
from operator import add, mul, xor

import pytest
import torch

from tests.helpers.tensor_checks import assert_bitwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm is required")


# Keep CPU collection independent of optional GPU frontends.
if torch.cuda.is_available():
    from tests.backends.shared.triton.reduction import _kernels as kernels


@pytest.mark.parametrize(
    "operation",
    ("integer-xor", "boolean-xor", "modular-add", "sum-int8", "product-int16", "sum-int64"),
)
@pytest.mark.parametrize("width,length", ((16, 0), (16, 11), (64, 33), (1024, 513)))
def test_public_custom_callback_survives_every_stage(operation, width, length):

    callback, scalar, identity, dtype = {
        "integer-xor": (kernels.boolean_xor, xor, 0, torch.int64),
        "boolean-xor": (kernels.boolean_xor, xor, False, torch.bool),
        "modular-add": (kernels.modular_add, lambda a, b: (a + b) % 257, 0, torch.int64),
        "sum-int8": (kernels.caller_add, add, 0, torch.int8),
        "product-int16": (kernels.caller_multiply, mul, 1, torch.int16),
        "sum-int64": (kernels.caller_add, add, 0, torch.int64),
    }[operation]
    values = ((torch.arange(2 * width) * 13 + 5) % 257).reshape(2, width).to(dtype)
    if operation == "product-int16":
        values.fill_(1)
        values[:, :3] = torch.tensor([2, -3, 5], dtype=dtype)
    if operation in ("integer-xor", "sum-int64") and length:
        values[0, 0] = 2**53 + 1
    expected = torch.tensor(
        [scalar_reduce(scalar, row[:length], identity) for row in values.tolist()],
        dtype=torch.int64,
        device="cuda",
    ).to(dtype)
    source = values.cuda()
    before = source.clone()
    outputs = [torch.empty_like(expected) for _ in range(2)]
    options = dict(num_warps=4, enable_fp_fusion=False)
    kernels.custom_input_probe[(2,)](
        source,
        *outputs,
        WIDTH=width,
        LENGTH=length,
        COMBINE_FN=callback,
        IDENTITY=identity,
        **options,
    )
    for output in outputs:
        assert_bitwise(output, expected)

    n_tiles = (length + 31) // 32
    partials = torch.empty((2, n_tiles), device="cuda", dtype=dtype)
    if n_tiles:
        kernels.custom_partial_probe[(2, n_tiles)](
            source,
            partials,
            WIDTH=width,
            LENGTH=length,
            COMBINE_FN=callback,
            IDENTITY=identity,
            **options,
        )
    initial = torch.full((2,), identity, device="cuda", dtype=dtype)
    for unroll in (1, 3):
        kernels.custom_merge_probe[(2,)](
            partials,
            initial,
            outputs[0],
            N_TILES=n_tiles,
            COMBINE_FN=callback,
            MERGE_UNROLL=unroll,
            **options,
        )
        assert_bitwise(outputs[0], expected)
    assert_bitwise(source, before)


def test_public_custom_merge_applies_runtime_seed_once():

    values = [[250, 12, 255, 3], [200, 250, 9, 15]]
    seeds = [17, 29]

    def callback(left, right):
        return (left + right) % 257

    initial = torch.tensor(seeds, device="cuda", dtype=torch.int64)
    output = torch.empty_like(initial)
    for count in (0, 1, 4):
        partials = torch.tensor(values, device="cuda", dtype=torch.int64)[:, :count].contiguous()
        expected = torch.tensor(
            [
                scalar_reduce(callback, row[:count], seed)
                for row, seed in zip(values, seeds, strict=True)
            ],
            device="cuda",
            dtype=torch.int64,
        )
        for unroll in (1, 3):
            kernels.custom_merge_probe[(2,)](
                partials,
                initial,
                output,
                N_TILES=count,
                COMBINE_FN=kernels.modular_add,
                MERGE_UNROLL=unroll,
                num_warps=4,
            )
            assert_bitwise(output, expected)
