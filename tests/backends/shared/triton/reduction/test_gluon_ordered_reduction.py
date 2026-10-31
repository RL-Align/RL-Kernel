# SPDX-License-Identifier: Apache-2.0
"""Shared Gluon parity, independent accuracy and scheduling invariance on CUDA/ROCm."""

import pytest
import torch

from rl_engine.reference import reduction as reference
from tests.helpers.reduction_inputs import prefix_storage
from tests.helpers.tensor_checks import assert_bitwise

# Keep CPU collection independent of optional GPU frontends.
if torch.cuda.is_available():
    pytest.importorskip("triton.experimental.gluon")
    from triton import language as tl
    from triton.experimental.gluon import language as gl

    from tests.backends.shared.triton.reduction import _gluon_kernels as gluon_kernels
    from tests.backends.shared.triton.reduction import _kernels as kernels


@pytest.fixture
def gluon_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required")
    pytest.importorskip("triton.experimental.gluon")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def warp_size(gluon_device):
    import triton

    with torch.cuda.device(gluon_device):
        return triton.runtime.driver.active.get_current_target().warp_size


@pytest.fixture
def math_options():
    options = dict(enable_fp_fusion=False)
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False
    return options


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("length", (0, 1, 9, 1023, 1024, 1025, 4105, 16385, 32769))
def test_memory_cross_frontend_and_execution_bits(
    gluon_device, warp_size, math_options, dtype, operation, length
):

    from rl_engine.backends.shared.triton import reduction as default

    width = 1 << max(0, (max(length, 1) - 1).bit_length())
    generator = torch.Generator().manual_seed(841)
    values = torch.randn((3, width), generator=generator).to(dtype)
    if operation == "product":
        values = (1 + values.float() / 4096).to(dtype)
    values[:, length:] = float("nan")
    x = torch.empty((3, width + 7), device=gluon_device, dtype=dtype)[:, :width]
    x.copy_(values)
    before = x.clone()
    expected = getattr(reference, operation)(
        values, length=length, from_memory=True, compute_dtype=torch.float32
    )
    native = torch.empty(3, device=gluon_device)
    ignored = torch.empty_like(native)
    # This probe supports contiguous rows; use a separate copy for its stride.
    kernels.input_dispatch_probe[(3,)](
        x.contiguous(),
        ignored,
        native,
        WIDTH=width,
        LENGTH=length,
        REDUCE_FN=getattr(default, operation),
        COMPUTE_DTYPE=tl.float32,
        POINTER_BLOCK_SIZE=1024,
        num_warps=4,
        **math_options,
    )
    assert_bitwise(native.cpu(), expected)
    for batch, elements, warps in ((8, 4, 1), (32, 8, 4)):
        out = torch.empty_like(native)

        def call(out=out, batch=batch, elements=elements, warps=warps):
            gluon_kernels.reduction_probe[(3,)](
                x,
                out,
                x.stride(0),
                length,
                width,
                1024,
                8,
                batch,
                elements,
                operation,
                False,
                gl.float32,
                warp_size,
                num_warps=warps,
                **math_options,
            )

        call()
        assert_bitwise(out, native)
        with torch.inference_mode():
            call()
        assert_bitwise(out, native)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(3):
            graph.replay()
        assert_bitwise(out, native)
        # Independently evaluate the mathematical operation in FP64.
        gold = (
            values[:, :length].double().sum(-1)
            if operation == "sum"
            else values[:, :length].double().prod(-1)
        )
        torch.testing.assert_close(out.cpu().double(), gold, atol=2e-4, rtol=2e-4)
    assert_bitwise(x, before)


@pytest.mark.parametrize("operation", ("sum", "product", "xor"))
@pytest.mark.parametrize(
    "width,group,length", ((1, 8, 1), (16, 8, 9), (128, 1, 97), (256, 32, 201), (2048, 8, 1025))
)
def test_loaded_and_custom_operation_keep_tree_and_dtype(
    gluon_device, warp_size, math_options, operation, width, group, length
):

    dtype = torch.int64 if operation == "xor" else torch.float32
    values = torch.arange(3 * width).reshape(3, width).to(dtype)
    if operation == "xor":
        values += 2**60
        expected = reference.reduce(
            values, combine_fn=lambda a, b: a ^ b, identity=0, length=length, max_group_size=group
        )
    else:
        values = (values % 7 - 3) if operation == "sum" else torch.ones_like(values) * 1.00001
        expected = getattr(reference, operation)(values, length=length, max_group_size=group)
    x = values.to(gluon_device)
    out = torch.empty(3, device=gluon_device, dtype=dtype)
    gluon_kernels.reduction_probe[(3,)](
        x,
        out,
        width,
        length,
        width,
        1024,
        group,
        32,
        8,
        operation,
        True,
        None,
        warp_size,
        num_warps=4,
        **math_options,
    )
    assert_bitwise(out.cpu(), expected)


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("count", (0, 1, 9, 256, 257, 515))
def test_builtin_partial_merge_uses_identity_across_windows(
    gluon_device, warp_size, math_options, operation, count
):
    """Sum/product convenience APIs use zero/one, including for empty partials."""

    values = torch.arange(3 * max(count, 1), dtype=torch.float32).reshape(3, -1)
    values = values % 9 - 4 if operation == "sum" else torch.ones_like(values) * 1.00001
    expected = getattr(reference, operation + "_partials")(values[:, :count])
    x = values.to(gluon_device)
    out = torch.empty(3, device=gluon_device, dtype=values.dtype)
    gluon_kernels.merge_probe[(3,)](
        x, out, x.stride(0), count, operation, warp_size, num_warps=4, **math_options
    )
    assert_bitwise(out.cpu(), expected)


@pytest.mark.parametrize("operation", ("sum", "product", "xor"))
@pytest.mark.parametrize("count", (0, 1, 9, 256, 257, 515))
def test_custom_partial_merge_carries_runtime_seed_once(
    gluon_device, warp_size, math_options, operation, count
):
    """Nonidentity row seeds survive empty input, window boundaries and tails."""
    from rl_engine.backends.shared.triton.reduction import gluon as ordered

    dtype = torch.int64 if operation == "xor" else torch.float32
    values = torch.arange(3 * max(count, 1)).reshape(3, -1).to(dtype)
    if operation == "xor":
        values += 2**60
        combine, callback = torch.bitwise_xor, gluon_kernels.xor_pair
    elif operation == "sum":
        values = values % 9 - 4
        combine, callback = torch.add, ordered.reduce_add
    else:
        values = torch.ones_like(values) * 1.00001
        combine, callback = torch.mul, ordered.reduce_multiply
    seeds = torch.tensor([17, 29, -5], dtype=dtype)
    expected = reference.reduce_partials(values[:, :count], combine_fn=combine, initial=seeds)
    x = values.to(gluon_device)
    initial = seeds.to(gluon_device)
    before = x.clone()
    out = torch.empty(3, device=gluon_device, dtype=dtype)
    gluon_kernels.seeded_merge_probe[(3,)](
        x,
        out,
        initial,
        x.stride(0),
        count,
        callback,
        warp_size,
        num_warps=4,
        **math_options,
    )
    assert_bitwise(out.cpu(), expected)
    assert_bitwise(x, before)
    assert_bitwise(initial.cpu(), seeds)


@pytest.mark.parametrize("count", (0, 1, 257))
@pytest.mark.parametrize("runtime_seed", (False, True), ids=("constexpr", "runtime"))
def test_fp64_partial_merge_preserves_seed_precision(
    gluon_device, warp_size, math_options, count, runtime_seed
):
    """FP64 seeds retain low bits through empty, single-partial and cross-window folds."""
    from rl_engine.backends.shared.triton.reduction import gluon as ordered

    seed = 1 + 2**-35
    # These binary fractions and their sums are exactly representable in FP64.
    partials = torch.full((1, max(count, 1)), 2**-40, device=gluon_device, dtype=torch.float64)
    initial = torch.tensor([seed], device=gluon_device, dtype=torch.float64)
    out = torch.empty_like(initial)
    if runtime_seed:
        gluon_kernels.seeded_merge_probe[(1,)](
            partials,
            out,
            initial,
            partials.stride(0),
            count,
            ordered.reduce_add,
            warp_size,
            num_warps=4,
            **math_options,
        )
    else:
        gluon_kernels.constexpr_seeded_merge_probe[(1,)](
            partials,
            out,
            partials.stride(0),
            count,
            seed,
            warp_size,
            num_warps=4,
            **math_options,
        )
    expected = torch.tensor([seed + count * 2**-40], dtype=torch.float64)
    assert_bitwise(out.cpu(), expected)
    assert_bitwise(initial.cpu(), torch.tensor([seed], dtype=torch.float64))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("length", (9, 1025, 4105, 262144, 262145))
def test_parallel_composition_preserves_all_rows_and_replay(
    gluon_device, warp_size, math_options, dtype, length
):
    """Embedded public helpers retain the tree across producer/merge schedules."""

    generator = torch.Generator().manual_seed(938)
    values = torch.randn((3, length), generator=generator).to(dtype)
    x = torch.full((3, length + 7), float("nan"), device=gluon_device, dtype=dtype)
    x[:, :length].copy_(values)
    before = x.clone()
    expected = reference.sum(values, from_memory=True, compute_dtype=torch.float32)
    n_tiles = (length + 1023) // 1024
    partials = torch.empty((3, n_tiles), device=gluon_device, dtype=torch.float32)
    for producer, merge, elements in ((1, 4, 4), (4, 1, 8)):
        out = torch.empty(3, device=gluon_device)

        def call(out=out, producer=producer, merge=merge, elements=elements):
            gluon_kernels.partial_probe[(3, n_tiles)](
                x,
                partials,
                x.stride(0),
                LENGTH=length,
                BLOCK=1024,
                GROUP=8,
                ELEMENTS=elements,
                WARP_SIZE=warp_size,
                num_warps=producer,
                **math_options,
            )
            gluon_kernels.merge_probe[(3,)](
                partials,
                out,
                n_tiles,
                COUNT=n_tiles,
                OP="sum",
                WARP_SIZE=warp_size,
                num_warps=merge,
                **math_options,
            )

        call()
        assert_bitwise(out.cpu(), expected)
        with torch.inference_mode():
            call()
        assert_bitwise(out.cpu(), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(3):
            graph.replay()
        assert_bitwise(out.cpu(), expected)
    assert_bitwise(x, before)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("loaded", (False, True))
@pytest.mark.parametrize("operation", ("sum", "product"))
def test_inferred_precision_and_prefix_do_not_depend_on_batch(
    gluon_device, warp_size, math_options, dtype, loaded, operation
):
    """Keep the target prefix bitwise equal alone and at different batch positions.

    The loaded vector width and arithmetic parameters remain fixed. Batch size,
    target row and physical row stride vary; poisoned suffixes must be ignored.
    Other rows contain different data, and every output is checked separately.
    """

    values = torch.randn(129, generator=torch.Generator().manual_seed(17))
    if operation == "product":
        values = 1 + values / 128
    values = values.to(dtype)
    single_row = None
    for batch_size, target_row, stride in (
        (1, 0, 256),
        (3, 0, 263),
        (3, 2, 263),
        (4, 1, 293),
    ):
        storage = prefix_storage(values, batch_size, stride, target_row)
        expected = getattr(reference, operation)(
            storage[:, :256], length=129, from_memory=not loaded, block_size=64
        )
        x = storage.to(gluon_device)
        before = x.clone()
        out = torch.empty(batch_size, device=gluon_device, dtype=dtype)
        gluon_kernels.reduction_probe[(batch_size,)](
            x,
            out,
            x.stride(0),
            129,
            256,
            64,
            8,
            8,
            4,
            operation,
            loaded,
            None,
            warp_size,
            num_warps=4,
            **math_options,
        )
        actual = out.cpu()
        assert_bitwise(actual, expected)
        if single_row is None:
            single_row = actual[0].clone()
        assert_bitwise(actual[target_row], single_row, "prefix independent of batch and position")
        assert_bitwise(x, before)
