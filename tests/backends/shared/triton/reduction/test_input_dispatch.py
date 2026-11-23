# SPDX-License-Identifier: Apache-2.0
"""Public input forms, argument validation and batch/padding prefix invariance."""

import pytest
import torch

from rl_engine.reference import reduction as reference
from tests.helpers.reduction_inputs import prefix_layouts, prefix_storage, prefix_values
from tests.helpers.tensor_checks import assert_bitwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm is required")


# Keep CPU collection independent of optional GPU frontends.
if torch.cuda.is_available():
    import triton.language as tl

    from tests.backends.shared.triton.reduction import _kernels as kernels


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("width,length", ((16, 0), (16, 3), (64, 32), (1024, 128), (1024, 1024)))
def test_same_public_function_distinguishes_values_and_pointer(operation, dtype, width, length):
    """Identical length arguments must not select the wrong input representation."""
    from rl_engine.backends.shared.triton import reduction as ordered

    values = torch.ones((3, width), dtype=dtype)
    if operation == "sum":
        values = (torch.arange(3 * width).reshape(3, width) % 17 - 8).to(dtype)
        if width == 64:
            values[0].zero_()
            # Both signs must lose +1; at 2/eps, -large + 1 is still exact.
            large = 4 / torch.finfo(dtype).eps
            values[0, [0, 8, 16, 24]] = torch.tensor([large, 1, -large, 1], dtype=dtype)
    else:
        values[:, ::2] = 2
        values[:, 1::2] = 0.5
    values[:, length:] = float("nan")
    pointer_block = 16
    ref = getattr(reference, operation)
    # Keep arithmetic in input dtype to catch implicit widening; both paths are
    # also exercised with explicit FP32 by the public API tests.
    expected_loaded = ref(values, length=length, block_size=pointer_block)
    expected_memory = ref(values, length=length, block_size=pointer_block, from_memory=True)
    if operation == "sum" and width == 64:
        # The loaded tree cancels before adding ones; each 16-wide pointer
        # tile rounds its one away before the two partials are merged.
        assert expected_loaded[0].item() == 2
        assert expected_memory[0].item() == 0
    source = values.cuda()
    before = source.clone()
    outputs = [torch.full((5,), -123, device="cuda", dtype=dtype) for _ in range(2)]
    options = dict(num_warps=4, enable_fp_fusion=False)
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False
    assert ordered.__all__ == [
        "sum",
        "product",
        "sum_partials",
        "product_partials",
        "reduce",
        "reduce_partials",
    ]
    kernels.input_dispatch_probe[(3,)](
        source,
        outputs[0][1:-1],
        outputs[1][1:-1],
        WIDTH=width,
        LENGTH=length,
        REDUCE_FN=getattr(ordered, operation),
        COMPUTE_DTYPE=None,
        POINTER_BLOCK_SIZE=pointer_block,
        **options,
    )
    for output, expected in zip(outputs, (expected_loaded, expected_memory), strict=True):
        assert_bitwise(output[1:-1], expected.cuda())
        assert_bitwise(output[[0, -1]], torch.full((2,), -123, device="cuda", dtype=dtype))
    assert_bitwise(source, before, "including the NaN suffix")


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize(
    "mode,length,message",
    (
        ("pointer", None, "pointer input requires length"),
        ("pointer", -1, "vector length must be nonnegative"),
        ("loaded", -1, "loaded length must be between zero and the vector width"),
        ("loaded", 17, "loaded length must be between zero and the vector width"),
        ("pointer_vector", 16, "input must be a scalar pointer"),
    ),
)
def test_public_dispatch_rejects_invalid_arguments_before_launch(operation, mode, length, message):
    """Require the specific static assertion beneath nested JIT call-site wrappers."""
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

    from rl_engine.backends.shared.triton import reduction as ordered

    values = torch.ones(16, device="cuda")
    output = torch.empty((), device="cuda")
    with pytest.raises(CompilationError) as error_info:
        kernels.invalid_input_probe[(1,)](
            values, output, MODE=mode, LENGTH=length, REDUCE_FN=getattr(ordered, operation)
        )
    # Each nested JIT call adds a source-location wrapper. Inspect the original
    # assertion so unrelated compilation failures cannot satisfy this test.
    error = error_info.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, CompileTimeAssertionFailure), str(error)
    assert error.error_message == message


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("length", (0, 1000, 1001, 1024, 1025, 2049))
def test_same_prefix_across_batch_padding_and_schedules(operation, dtype, length):
    """Compare the same effective values, never two different sequence lengths.

    The fixed prefix is evaluated alone and in larger, differently ordered
    batches with NaN-poisoned suffixes. Serial, pipelined and reversed-assignment
    parallel producers all retain the same tree and FP32 partial storage. This
    is a path-conformance check, not an independent accuracy reference.
    """

    from rl_engine.backends.shared.triton import reduction
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )

    block_size = 1024
    n_tiles = (length + block_size - 1) // block_size
    values = prefix_values(length, dtype, operation)
    reference_reduce = getattr(reference, operation)
    reduce = reduction.sum if operation == "sum" else reduction.product
    combine = reduce_add if operation == "sum" else reduce_multiply
    identity = 0.0 if operation == "sum" else 1.0
    expected = reference_reduce(
        values, compute_dtype=torch.float32, block_size=block_size, from_memory=True
    ).cuda()
    options = dict(num_warps=4, enable_fp_fusion=False)
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False

    for batch_size, width, target_row in prefix_layouts(length, block_size):
        cpu_storage = prefix_storage(values, batch_size, width, target_row)
        all_expected = reference_reduce(
            cpu_storage[:, :length], compute_dtype=torch.float32, from_memory=True
        ).cuda()
        storage = cpu_storage.cuda()
        before = storage.clone()
        output_storage = torch.full((batch_size + 2,), -123.0, device="cuda")
        output = output_storage[1:-1]

        for stages in (None, 3):
            output.fill_(float("nan"))
            kernels.configured_vector_probe[(batch_size,)](
                storage,
                output,
                storage.stride(0),
                LENGTH=length,
                BLOCK_SIZE=block_size,
                MAX_GROUP_SIZE=8,
                COMPUTE_DTYPE=tl.float32,
                REDUCE_FN=reduce,
                PIPELINE_STAGES=stages,
                TILE_UNROLL=None,
                **options,
            )
            assert_bitwise(output, all_expected, f"serial prefix, stages={stages}")
            assert_bitwise(output[target_row], expected, "target independent of batch position")

        partials = torch.empty((batch_size, n_tiles), device="cuda", dtype=torch.float32)
        initials = torch.full((batch_size,), identity, device="cuda")
        for reverse in (False, True):
            partials.fill_(float("nan"))
            if n_tiles:
                kernels.ordered_reduce_partial_probe[(batch_size, n_tiles)](
                    storage,
                    partials,
                    storage.stride(0),
                    LENGTH=length,
                    BLOCK_SIZE=block_size,
                    MAX_GROUP_SIZE=8,
                    COMBINE_FN=combine,
                    IDENTITY=identity,
                    REVERSE_TILES=reverse,
                    **options,
                )
            output.fill_(float("nan"))
            kernels.ordered_reduce_merge_probe[(batch_size,)](
                partials,
                initials,
                output,
                N_TILES=n_tiles,
                MERGE_UNROLL=3 if reverse else 1,
                COMBINE_FN=combine,
                **options,
            )
            assert_bitwise(output, all_expected, f"parallel prefix, reversed={reverse}")
            assert_bitwise(output[target_row], expected, "same prefix after parallel merge")

        assert_bitwise(storage, before, "prefix and poisoned padding preserved")
        assert_bitwise(
            output_storage[[0, -1]],
            torch.full((2,), -123.0, device="cuda"),
            "output guards",
        )
