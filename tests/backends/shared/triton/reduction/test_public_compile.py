# SPDX-License-Identifier: Apache-2.0
"""CPU-side JIT compilation and interpreter checks for public reduction helpers."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cuda_compile():
    """Build CUDA kernels for a fixed target, without allocating or launching them.

    Python/interpreter execution accepts keyword-only arguments that Triton's
    JIT frontend may not bind. Exercise the real compiler even on a CPU host
    with Triton installed; a missing runtime is an explicit skip.
    """
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "compile") or not hasattr(triton, "constexpr_function"):
        pytest.skip("A complete Triton runtime with constexpr_function is required")
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if "nvidia" not in backends:
        pytest.skip("Triton's NVIDIA compilation backend is required")

    def compile_probe(probe, constants, pointer_type):
        signature = {
            name: "constexpr" if name in constants else pointer_type for name in probe.arg_names
        }
        source = ASTSource(probe, signature=signature, constexprs=constants)
        compiled = triton.compile(
            source,
            target=GPUTarget("cuda", 86, 32),
            options={"num_warps": 4, "enable_fp_fusion": False},
        )
        assert compiled.asm["ptx"]

    return compile_probe


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("input_form", ("loaded", "pointer", "partials"))
def test_public_defaults_and_dtype_overrides_compile(cuda_compile, operation, input_form):
    """Compile omitted/None/explicit dtype calls through all convenience wrappers."""
    from rl_engine.backends.shared.triton import reduction
    from tests.backends.shared.triton.reduction._kernels import (
        inferred_tile_dtype_probe,
        inferred_vector_dtype_probe,
        public_partials_probe,
    )

    if input_form == "loaded":
        probe = inferred_tile_dtype_probe
        constants = {"BLOCK_SIZE": 64, "REDUCE_FN": getattr(reduction, operation)}
    elif input_form == "pointer":
        probe = inferred_vector_dtype_probe
        constants = {"LENGTH": 33, "REDUCE_FN": getattr(reduction, operation)}
    else:
        probe = public_partials_probe
        constants = {
            "N_TILES": 5,
            "REDUCE_FN": getattr(reduction, f"{operation}_partials"),
            "MERGE_UNROLL": 3,
        }
    cuda_compile(probe, constants, "*fp32")


@pytest.mark.parametrize("input_form", ("input", "partials"))
def test_custom_callback_and_runtime_seed_compile(cuda_compile, input_form):
    """Compile caller-owned callbacks with default precision and a loaded merge seed."""
    from tests.backends.shared.triton.reduction._kernels import (
        boolean_xor,
        custom_input_probe,
        custom_merge_probe,
    )

    if input_form == "input":
        probe = custom_input_probe
        constants = {"WIDTH": 64, "LENGTH": 33, "COMBINE_FN": boolean_xor, "IDENTITY": 0}
    else:
        probe = custom_merge_probe
        constants = {"N_TILES": 5, "COMBINE_FN": boolean_xor, "MERGE_UNROLL": 3}
    cuda_compile(probe, constants, "*i64")


def test_ordered_tree_accepts_interpreter_constexpr_sizes():
    """Exercise the single halving loop with CPU tensors in a fresh interpreter process.

    TRITON_INTERPRET must be set before importing the JIT helpers. Keep this
    isolated from GPU tests and bound execution so interpreter regressions
    cannot stall the suite. The reference declares the pairs independently.
    """
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "jit"):
        pytest.skip("A complete Triton runtime is required")
    script = textwrap.dedent(
        """
        import torch
        from rl_engine.reference.reduction.ordered_reduce import ordered_tile_reduce
        from tests.backends.shared.triton.reduction._kernels import (
            caller_add, caller_multiply, tile_reduce_probe,
        )
        from tests.helpers.tensor_checks import assert_bitwise

        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(939)
        for width in (1, 8, 16, 64, 1024):
            for group in (1, 8, 32):
                for combine, reference in ((caller_add, torch.add), (caller_multiply, torch.mul)):
                    values = 1 + torch.randn((2, width), generator=generator) * 0.01
                    values[0].fill_(-0.0)
                    expected = ordered_tile_reduce(
                        values, max_group_size=group, combine_fn=reference,
                    )
                    actual = torch.empty((2,), dtype=torch.float32)
                    tile_reduce_probe[(2,)](
                        values, actual, BLOCK_SIZE=width, MAX_GROUP_SIZE=group,
                        COMBINE_FN=combine, num_warps=4, enable_fp_fusion=False,
                    )
                    assert_bitwise(actual, expected, f"interpreter tree: {width}/{group}")

        from rl_engine.backends.shared.triton import reduction as ordered
        from rl_engine.reference import reduction as reference
        from tests.backends.shared.triton.reduction._kernels import input_dispatch_probe

        # Check both specializations of the same public function in one kernel.
        # The timeout also bounds this small dispatch matrix.
        for width, length in ((16, 0), (16, 3), (64, 32)):
            values = torch.ones((2, width), dtype=torch.float32)
            values[:, length:] = float("nan")
            for operation in ("sum", "product"):
                outputs = [torch.empty(2) for _ in range(2)]
                input_dispatch_probe[(2,)](
                    values, *outputs, WIDTH=width, LENGTH=length,
                    REDUCE_FN=getattr(ordered, operation), COMPUTE_DTYPE=None,
                    POINTER_BLOCK_SIZE=32, num_warps=4, enable_fp_fusion=False,
                )
                ref = getattr(reference, operation)
                assert_bitwise(outputs[0], ref(values, length=length))
                assert_bitwise(
                    outputs[1], ref(values, length=length, from_memory=True, block_size=32),
                )

        from tests.backends.shared.triton.reduction._kernels import (
            boolean_xor, custom_input_probe, custom_partial_probe, custom_merge_probe,
        )
        values = torch.arange(128, dtype=torch.int64).reshape(2, 64)
        values[0, 0] = 2**53 + 1
        for length in (0, 33):
            expected = torch.zeros(2, dtype=torch.int64)
            for column in range(length):
                expected = expected ^ values[:, column]
            outputs = [torch.empty_like(expected) for _ in range(2)]
            custom_input_probe[(2,)](
                values, *outputs, WIDTH=64, LENGTH=length,
                COMBINE_FN=boolean_xor, IDENTITY=0,
            )
            for output in outputs:
                assert_bitwise(output, expected)
            n_tiles = (length + 31) // 32
            partials = torch.empty((2, n_tiles), dtype=torch.int64)
            if n_tiles:
                custom_partial_probe[(2, n_tiles)](
                    values, partials, WIDTH=64, LENGTH=length,
                    COMBINE_FN=boolean_xor, IDENTITY=0,
                )
            initial = torch.tensor([17, 29], dtype=torch.int64)
            custom_merge_probe[(2,)](
                partials, initial, outputs[0], N_TILES=n_tiles,
                COMBINE_FN=boolean_xor, MERGE_UNROLL=3,
            )
            assert_bitwise(outputs[0], expected ^ initial)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[5],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
