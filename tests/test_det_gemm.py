# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Invariance + correctness tests for det_gemm (WS1).

Runs the ROCm-native Triton path directly. CUDA continues to exercise both its
existing native kernel and Triton, each independently satisfying the contract.
The PyTorch path (torch.matmul) is intentionally NOT tested here: it is the
non-deterministic reference baseline and would fail batch-invariance by design.
"""

import pytest
import torch

from rl_engine.kernels.gtest.tolerance import load_contract
from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.cuda.matmul import deterministic_gemm

try:
    import triton

    from rl_engine.kernels.ops.triton.matmul import deterministic_gemm_triton
    from rl_engine.kernels.ops.triton.matmul.det_gemm import (
        _DEFAULT_TREE_LEAF_CONFIG,
        _GFX942_QWEN_FORWARD_LEAF_CONFIGS,
        _GFX942_QWEN_TP_SHARD_FORWARD_LEAF_CONFIGS,
        _GFX942_QWEN_TP_SHARD_WGRAD_LEAF_CONFIGS,
        _GFX942_QWEN_WGRAD_LEAF_CONFIGS,
        _copy_tree_root_kernel,
        _det_gemm_tree_leaf_kernel,
        _det_gemm_tree_reduce_kernel,
        _det_gemm_tree_reduce_to_output_kernel,
        _device_tree_plan,
        _gfx942_qwen_tree_leaf_config,
        _triton_tree_gemm,
    )

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
IS_ROCM = getattr(torch.version, "hip", None) is not None
HAS_SUPPORTED_GPU = torch.cuda.is_available() and (
    IS_ROCM or torch.cuda.get_device_capability()[0] >= 8
)
IS_GFX942 = (
    IS_ROCM
    and torch.cuda.is_available()
    and str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).startswith("gfx942")
)

pytestmark = pytest.mark.skipif(
    not HAS_SUPPORTED_GPU,
    reason="det_gemm requires a ROCm GPU or CUDA SM80+",
)

# ROCm acceptance intentionally depends only on the Triton implementation.
_BACKENDS = [] if IS_ROCM else [("cuda", deterministic_gemm)]
if _HAS_TRITON:
    _BACKENDS.append(("triton", deterministic_gemm_triton))


def _rand(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)


def _assert_same_raw_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert expected.is_contiguous()
    assert torch.equal(
        actual.reshape(-1).view(torch.uint8),
        expected.reshape(-1).view(torch.uint8),
    )


def _leaf_workspace(
    a: torch.Tensor,
    b: torch.Tensor,
    config,
) -> torch.Tensor:
    m_size, k_size = a.shape
    n_size = b.size(1)
    plan = _device_tree_plan(k_size, a.device)
    workspace = torch.empty(
        (plan.host.node_count, m_size, n_size),
        dtype=torch.bfloat16,
        device=a.device,
    )
    tiles_m = triton.cdiv(m_size, config.block_m)
    tiles_n = triton.cdiv(n_size, config.block_n)
    grid = (
        (tiles_n, tiles_m, len(plan.host.leaf_nodes))
        if config.n_fastest
        else (len(plan.host.leaf_nodes), tiles_m, tiles_n)
    )
    _det_gemm_tree_leaf_kernel[grid](
        a,
        b,
        workspace,
        plan.leaf_starts,
        plan.leaf_lengths,
        plan.leaf_nodes,
        M=m_size,
        N=n_size,
        K=k_size,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=_K_TREE_LEAF,
        N_FASTEST=config.n_fastest,
        num_warps=config.num_warps,
    )
    return workspace.index_select(0, plan.leaf_nodes.to(torch.int64))


def _special_bf16(shape: tuple[int, ...], *, offset: int = 0) -> torch.Tensor:
    bits = torch.tensor(
        (
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F80,
            0xBF80,
            0x7F7F,
            0xFF7F,
            0x7F80,
            0xFF80,
            0x7F81,
            0x7FC1,
            0xFF81,
            0xFFFF,
        ),
        dtype=torch.uint16,
    )
    elements = 1
    for size in shape:
        elements *= size
    indices = (torch.arange(elements, dtype=torch.int64) + offset) % bits.numel()
    return bits[indices].view(torch.bfloat16).reshape(shape).to(DEV)


_K_TREE_LEAF = 32


def _k_tree_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Canonical FP32-leaf/BF16-node midpoint tree used by Triton."""

    a = a.detach().contiguous()
    b = b.detach().contiguous()

    def reduce_range(lo: int, hi: int) -> torch.Tensor:
        if hi - lo <= _K_TREE_LEAF:
            return (a[:, lo:hi].float() @ b[lo:hi, :].float()).to(torch.bfloat16)
        midpoint = lo + (hi - lo) // 2
        return reduce_range(lo, midpoint) + reduce_range(midpoint, hi)

    return reduce_range(0, a.size(1))


def _balanced_tree_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    level = parts
    while len(level) > 1:
        level = [level[index] + level[index + 1] for index in range(0, len(level), 2)]
    return level[0]


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
@pytest.mark.parametrize(
    ("shape", "transpose_output", "preserve_a_strides", "expected"),
    (
        ((1, 4096, 12288), False, False, (1, 128, 1, True)),
        ((8, 4096, 12288), False, False, (8, 128, 2, True)),
        ((32, 12288, 4096), False, False, (32, 128, 4, True)),
        ((4096, 1, 12288), True, True, (128, 64, 2, True)),
        ((12288, 8, 4096), True, True, (128, 64, 2, True)),
        ((4096, 32, 12288), True, True, (64, 64, 2, True)),
        ((16, 4096, 12288), False, False, (64, 64, 4, False)),
        ((32, 4096, 4096), False, False, (64, 64, 4, False)),
        ((4096, 16, 12288), True, True, (64, 64, 4, False)),
        ((4096, 8, 12288), True, False, (64, 64, 4, False)),
        ((32, 4096, 6144), False, False, (32, 128, 4, True)),
        ((32, 6144, 4096), False, False, (32, 128, 4, True)),
        ((16, 4096, 6144), False, False, (8, 64, 1, True)),
        ((16, 6144, 4096), False, False, (8, 64, 1, True)),
        ((32, 4096, 3072), False, False, (64, 64, 4, False)),
        ((32, 3072, 4096), False, False, (64, 64, 4, False)),
        ((32, 4096, 1536), False, False, (64, 64, 4, False)),
        ((32, 1536, 4096), False, False, (64, 64, 4, False)),
        ((4096, 32, 6144), True, True, (128, 64, 2, True)),
        ((6144, 32, 4096), True, True, (128, 64, 2, True)),
        ((4096, 32, 3072), True, True, (64, 64, 4, False)),
        ((3072, 32, 4096), True, True, (64, 64, 4, False)),
        ((4096, 16, 6144), True, True, (64, 64, 4, False)),
    ),
)
def test_gfx942_qwen_leaf_config_table_is_exact(
    shape,
    transpose_output,
    preserve_a_strides,
    expected,
):
    config = _gfx942_qwen_tree_leaf_config(
        *shape,
        transpose_output=transpose_output,
        preserve_a_strides=preserve_a_strides,
    )
    assert (
        config.block_m,
        config.block_n,
        config.num_warps,
        config.n_fastest,
    ) == expected


@pytest.mark.skipif(
    not (_HAS_TRITON and IS_GFX942),
    reason="leaf specialization raw-byte tests require ROCm gfx942",
)
@pytest.mark.parametrize("k_size", (1, 8, 31, 32, 33, 65, 4096, 12288))
def test_gfx942_leaf_configs_preserve_special_value_workspace_raw_bytes(k_size):
    a = _special_bf16((3, k_size))
    b = _special_bf16((k_size, 17), offset=7)
    expected = _leaf_workspace(a, b, _DEFAULT_TREE_LEAF_CONFIG)
    configs = tuple(
        dict.fromkeys(
            (
                *_GFX942_QWEN_FORWARD_LEAF_CONFIGS.values(),
                *_GFX942_QWEN_TP_SHARD_FORWARD_LEAF_CONFIGS.values(),
                *_GFX942_QWEN_TP_SHARD_WGRAD_LEAF_CONFIGS.values(),
                *_GFX942_QWEN_WGRAD_LEAF_CONFIGS.values(),
            )
        )
    )

    for config in configs:
        _assert_same_raw_bytes(_leaf_workspace(a, b, config), expected)


@pytest.mark.skipif(
    not (_HAS_TRITON and IS_GFX942),
    reason="leaf specialization raw-byte tests require ROCm gfx942",
)
def test_gfx942_leaf_configs_preserve_transposed_a_workspace_raw_bytes():
    source = _special_bf16((65, 3))
    a = source.t()
    b = _special_bf16((65, 17), offset=11)
    assert not a.is_contiguous()
    assert all(stride > 0 for stride in a.stride())
    expected = _leaf_workspace(a, b, _DEFAULT_TREE_LEAF_CONFIG)
    configs = tuple(
        dict.fromkeys(
            (
                *_GFX942_QWEN_FORWARD_LEAF_CONFIGS.values(),
                *_GFX942_QWEN_TP_SHARD_FORWARD_LEAF_CONFIGS.values(),
                *_GFX942_QWEN_TP_SHARD_WGRAD_LEAF_CONFIGS.values(),
                *_GFX942_QWEN_WGRAD_LEAF_CONFIGS.values(),
            )
        )
    )

    for config in configs:
        _assert_same_raw_bytes(_leaf_workspace(a, b, config), expected)


@pytest.mark.parametrize("tp_size", (2, 4, 8))
@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
def test_forward_matches_balanced_contiguous_k_shards_bitwise(tp_size):
    """The GEMM K-tree and the TP collective rank tree must be the same graph."""

    torch.manual_seed(8)
    # Qwen3-8B's down projection uses K=12288. Its 512 24-wide leaves also
    # exercise the non-power-of-two midpoint tree used by the SM90 kernel.
    m, k, n = 4, 12288, 64
    a, b = _rand(m, k), _rand(k, n)
    width = k // tp_size
    parts = [
        deterministic_gemm_triton(
            a[:, rank * width : (rank + 1) * width].contiguous(),
            b[rank * width : (rank + 1) * width].contiguous(),
        )
        for rank in range(tp_size)
    ]

    full = deterministic_gemm_triton(a, b)
    sharded = _balanced_tree_sum(parts)
    assert torch.equal(full, sharded), (
        f"full GEMM differed from balanced TP={tp_size} shards at "
        f"{int((full != sharded).sum().item())} elements"
    )


@pytest.mark.parametrize(
    "shape",
    [
        (128, 128, 128),  # aligned SM90 path
        (31, 128, 128),  # SM90 M padding
        (31, 70, 65),  # scalar fallback
    ],
)
def test_rhs_transposed_layout_matches_legacy_forward_bitwise(shape):
    torch.manual_seed(20)
    M, K, N = shape
    a = _rand(M, K)
    bt = _rand(N, K)

    expected = _C.det_gemm_fwd(a, bt.t().contiguous())
    actual = _C.det_gemm_fwd_rhs_transposed(a, bt)

    assert actual.is_contiguous()
    assert tuple(actual.shape) == (M, N)
    assert torch.equal(actual, expected)


def test_rhs_transposed_materializes_unaligned_contiguous_view_for_tma():
    torch.manual_seed(23)
    M, K, N = 128, 128, 128
    a = _rand(M, K)
    storage = _rand(N * K + 1)
    bt = storage[1:].view(N, K)
    assert bt.is_contiguous()
    assert bt.data_ptr() % 16 != 0

    expected = _C.det_gemm_fwd(a, bt.t().contiguous())
    actual = _C.det_gemm_fwd_rhs_transposed(a, bt)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 128),  # short-K tiled backward path
        (8, 128, 128),  # short-K tiled backward path
        (128, 128, 128),  # aligned SM90 path
        (128, 96, 64),  # SM90 logical-M padding and dim-1 crop
        (31, 70, 65),  # scalar fallback
    ],
)
def test_transposed_db_is_bitwise_and_canonical_contiguous(shape):
    torch.manual_seed(21)
    tokens, in_features, out_features = shape
    a = _rand(tokens, in_features)
    dc = _rand(tokens, out_features)

    expected = _C.det_gemm_db(a, dc).t().contiguous()
    actual = _C.det_gemm_db_transposed(a, dc)

    assert tuple(actual.shape) == (out_features, in_features)
    assert tuple(actual.stride()) == (in_features, 1)
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("shape", [(128, 128, 128), (31, 70, 65)])
def test_da_physical_transpose_contract_matches_legacy_path_bitwise(shape):
    torch.manual_seed(22)
    M, K, N = shape
    dc = _rand(M, N)
    b = _rand(K, N)

    expected = _C.det_gemm_fwd(dc, b.t().contiguous())
    actual = _C.det_gemm_da(dc, b)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_batch_invariance(name, gemm):
    # A row's output must not change when other rows join the batch.
    torch.manual_seed(0)
    K, N = 4096, 4096
    b = _rand(K, N)
    row = _rand(1, K)
    out1 = gemm(row, b)
    big = _rand(512, K)
    big[0] = row[0]
    outN = gemm(big, b)
    assert torch.equal(out1[0], outN[0]), f"{name}: forward batch-invariance broken"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_chunked_prefill(name, gemm):
    # Splitting M then concatenating must match the full GEMM bitwise.
    torch.manual_seed(1)
    M, K, N = 256, 4096, 4096
    a, b = _rand(M, K), _rand(K, N)
    full = gemm(a, b)
    chunked = torch.cat([gemm(a[:100], b), gemm(a[100:], b)], dim=0)
    assert torch.equal(full, chunked), f"{name}: chunked-prefill broke invariance"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_padding_invariance(name, gemm):
    # Padding rows must not affect valid rows' output.
    torch.manual_seed(2)
    M, K, N = 100, 4096, 4096
    a, b = _rand(M, K), _rand(K, N)
    base = gemm(a, b)
    a_pad = torch.cat([a, _rand(28, K)], dim=0)
    padded = gemm(a_pad, b)
    assert torch.equal(base, padded[:M]), f"{name}: padding changed valid-row output"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_correctness(name, gemm):
    torch.manual_seed(3)
    M, K, N = 128, 2048, 2048
    a, b = _rand(M, K), _rand(K, N)
    out = gemm(a, b).float()
    ref = _k_tree_gemm(a, b).float()
    contract = load_contract()
    thresholds = contract["accuracy"]["default"]["reduction"]["bfloat16"]
    torch.testing.assert_close(out, ref, atol=thresholds["atol"], rtol=thresholds["rtol"])


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_backward_batch_invariance(name, gemm):
    # dA for a row must be invariant to the surrounding batch.
    torch.manual_seed(4)
    K, N = 2048, 2048
    b = _rand(K, N)
    row = _rand(1, K).requires_grad_(True)
    gemm(row, b).sum().backward()
    g1 = row.grad.clone()
    big = _rand(256, K)
    big[0] = row.detach()[0]
    big.requires_grad_(True)
    gemm(big, b).sum().backward()
    assert torch.equal(g1[0], big.grad[0]), f"{name}: backward dA batch-invariance broken"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_backward_correctness(name, gemm):
    torch.manual_seed(5)
    M, K, N = 64, 1024, 1024
    a = _rand(M, K).requires_grad_(True)
    b = _rand(K, N).requires_grad_(True)
    g = _rand(M, N)
    gemm(a, b).backward(g)
    expected_da = _k_tree_gemm(g, b.detach().t().contiguous())
    expected_db = _k_tree_gemm(a.detach().t().contiguous(), g)
    contract = load_contract()
    thresholds = contract["accuracy"]["default"]["reduction"]["bfloat16"]
    torch.testing.assert_close(
        a.grad.float(),
        expected_da.float(),
        atol=thresholds["atol"],
        rtol=thresholds["rtol"],
    )
    torch.testing.assert_close(
        b.grad.float(),
        expected_db.float(),
        atol=thresholds["atol"],
        rtol=thresholds["rtol"],
    )


@pytest.mark.parametrize("name,gemm", _BACKENDS)
@pytest.mark.parametrize(
    "shape",
    [
        (4096, 4096, 12288),  # qkv
        (4096, 4096, 4096),  # o_proj
        (4096, 4096, 14336),  # mlp_up
        (4096, 14336, 4096),  # mlp_dn
        (4096, 4096, 32000),  # lm_head
    ],
)
def test_target_shapes_invariance(name, gemm, shape):
    # Standard-Transformer projection shapes stay batch-invariant.
    torch.manual_seed(6)
    M, K, N = shape
    b = _rand(K, N)
    row = _rand(1, K)
    big = _rand(64, K)
    big[0] = row[0]
    assert torch.equal(
        gemm(row, b)[0], gemm(big, b)[0]
    ), f"{name}: batch-invariance broken at shape {shape}"


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
def test_triton_ragged_tiles_mask_all_axes():
    """Non-multiple M/N/K tiles must not read or write past tensor bounds."""
    torch.manual_seed(7)
    a = _rand(80, 130).requires_grad_(True)
    b = _rand(130, 129).requires_grad_(True)
    out = deterministic_gemm_triton(a, b)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (80, 129)
    assert torch.equal(out, _k_tree_gemm(a, b))
    out.backward(_rand(80, 129))
    torch.cuda.synchronize()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
@pytest.mark.parametrize("shape", ((4, 65, 129), (8, 4096, 128), (4, 12288, 64)))
def test_triton_tree_matches_python_reference_forward_bitwise(shape):
    """Triton evaluates the canonical FP32-leaf/BF16-node tree exactly."""

    torch.manual_seed(47)
    m_size, k_size, n_size = shape
    a = _rand(m_size, k_size)
    b = _rand(k_size, n_size)

    expected = _k_tree_gemm(a, b)
    triton_output = deterministic_gemm_triton(a, b)

    assert torch.equal(expected, triton_output), (
        f"Triton/reference mismatch at {shape}: "
        f"{int((expected != triton_output).sum().item())} elements"
    )


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
def test_triton_tree_matches_python_reference_backward_bitwise():
    torch.manual_seed(48)
    a = _rand(32, 512)
    b = _rand(512, 128)
    grad_output = _rand(32, 128)
    triton_inputs = [value.detach().clone().requires_grad_(True) for value in (a, b)]

    deterministic_gemm_triton(*triton_inputs).backward(grad_output)

    expected = (
        _k_tree_gemm(grad_output, b.t().contiguous()),
        _k_tree_gemm(a.t().contiguous(), grad_output),
    )
    for expected_grad, triton_input in zip(expected, triton_inputs, strict=True):
        assert torch.equal(expected_grad, triton_input.grad)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
def test_triton_final_tree_level_direct_output_matches_legacy_copy_raw_bytes():
    """Changing the root destination must not change its BF16 bit pattern."""

    m_size, n_size = 3, 97
    workspace = _special_bf16((3, m_size, n_size), offset=5)
    legacy_workspace = workspace.clone()
    lower = torch.tensor([0], device=DEV, dtype=torch.int64)
    upper = torch.tensor([1], device=DEV, dtype=torch.int64)
    root = torch.tensor([2], device=DEV, dtype=torch.int64)
    block = 256
    grid = (1, triton.cdiv(m_size * n_size, block))
    _det_gemm_tree_reduce_kernel[grid](
        legacy_workspace,
        lower,
        upper,
        root,
        M=m_size,
        N=n_size,
        BLOCK=block,
    )
    expected = torch.empty((m_size, n_size), device=DEV, dtype=torch.bfloat16)
    _copy_tree_root_kernel[(triton.cdiv(expected.numel(), block),)](
        legacy_workspace,
        expected,
        2,
        expected.numel(),
        BLOCK=block,
    )

    actual = torch.empty_like(expected)
    _det_gemm_tree_reduce_to_output_kernel[(triton.cdiv(actual.numel(), block),)](
        workspace,
        actual,
        lower,
        upper,
        M=m_size,
        N=n_size,
        BLOCK=block,
    )
    _assert_same_raw_bytes(actual, expected)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
@pytest.mark.parametrize("k_size", (31, 65))
def test_triton_forward_out_buffer_preserves_identity_version_and_raw_bytes(k_size):
    """Caller storage works for both the leaf-root and reduced-root paths."""

    torch.manual_seed(51 + k_size)
    a = _rand(3, k_size)
    b = _rand(k_size, 17)
    expected = _triton_tree_gemm(a, b)
    output_buffer = torch.empty_like(expected)
    version_before = output_buffer._version

    actual = _triton_tree_gemm(a, b, out=output_buffer)

    assert actual is output_buffer
    assert output_buffer._version == version_before + 1
    _assert_same_raw_bytes(actual, expected)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 3),
        (3, 31, 5),
        (17, 32, 33),
        (33, 33, 17),
        (65, 65, 31),
    ),
)
def test_triton_tree_transposed_out_matches_legacy_root_copy_raw_bytes(shape):
    """Root placement may transpose addresses, but never the BF16 values."""

    torch.manual_seed(49)
    m_size, k_size, n_size = shape
    a = _rand(m_size, k_size)
    b = _rand(k_size, n_size)

    legacy = _triton_tree_gemm(a, b).t().contiguous()
    output_buffer = torch.empty(
        (n_size, m_size),
        dtype=torch.bfloat16,
        device=a.device,
    )
    version_before = output_buffer._version
    transposed = _triton_tree_gemm(
        a,
        b,
        transpose_output=True,
        out=output_buffer,
    )

    assert transposed is output_buffer
    assert output_buffer._version == version_before + 1
    assert transposed.stride() == (m_size, 1)
    _assert_same_raw_bytes(transposed, legacy)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
def test_triton_wgrad_reads_positive_stride_transpose_view_raw_bytes():
    """The copy-free a.T wgrad path must preserve the legacy GEMM bit graph."""

    torch.manual_seed(50)
    token_count, input_size, output_size = 33, 65, 17
    activations = _rand(token_count, input_size)
    grad_output = _rand(token_count, output_size)
    activation_t = activations.t()
    assert not activation_t.is_contiguous()
    assert all(stride > 0 for stride in activation_t.stride())

    legacy = (
        _triton_tree_gemm(
            activation_t.contiguous(),
            grad_output,
        )
        .t()
        .contiguous()
    )
    output_buffer = torch.empty(
        (output_size, input_size),
        dtype=torch.bfloat16,
        device=activations.device,
    )
    direct = _triton_tree_gemm(
        activation_t,
        grad_output,
        transpose_output=True,
        out=output_buffer,
        preserve_a_strides=True,
    )

    assert direct is output_buffer
    _assert_same_raw_bytes(direct, legacy)
