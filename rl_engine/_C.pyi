# rl_engine/_C.pyi
# This file is a type stub for the compiled C++ extension module.
import torch

def deterministic_collective_ipc_meta(
    tensor: torch.Tensor,
) -> tuple[list[int], int]: ...
def deterministic_collective_create(
    staging: torch.Tensor,
    handles: list[list[int]],
    offsets: list[int],
    rank: int,
) -> int: ...
def deterministic_collective_destroy(handle: int) -> None: ...
def deterministic_collective_stage(handle: int, input: torch.Tensor) -> None: ...
def deterministic_collective_all_reduce(handle: int, output: torch.Tensor) -> None: ...
def deterministic_collective_prepare_staged(handle: int, input: torch.Tensor) -> None: ...
def deterministic_collective_all_reduce_staged(
    handle: int, input: torch.Tensor, output: torch.Tensor
) -> None: ...
def deterministic_collective_all_reduce_fused(
    handle: int, input: torch.Tensor, output: torch.Tensor
) -> None: ...
def deterministic_collective_reduce_scatter(handle: int, output: torch.Tensor) -> None: ...
def deterministic_collective_all_gather(handle: int, output: torch.Tensor) -> None: ...
def deterministic_collective_all_gather_fused(
    handle: int, input: torch.Tensor, output: torch.Tensor
) -> None: ...
def deterministic_collective_all_gather_many(
    handle: int, inputs: list[torch.Tensor], outputs: list[torch.Tensor]
) -> None: ...
def deterministic_collective_rocm_all_reduce(
    rank_inputs: torch.Tensor,
    output: torch.Tensor,
) -> None: ...
def deterministic_collective_rocm_reduce_scatter(
    rank_inputs: torch.Tensor,
    output: torch.Tensor,
) -> None: ...
def fused_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def fused_logp_sm90(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...
def batch_invariant_logp_sm90(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> list[torch.Tensor]: ...
def fused_linear_logp_sm90(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None,
) -> list[torch.Tensor]: ...
def fused_linear_logp_sm90_global_target(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def fused_linear_logp_sm90_backward(
    grad_logp: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    bias: torch.Tensor | None,
    vocab_start_index: int,
    compute_grad_hidden: bool,
    compute_grad_weight: bool,
    compute_grad_bias: bool,
    use_global_lse: bool,
) -> list[torch.Tensor]: ...
def linear_logp_probs_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_local_probs_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_local_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_probs_bf16_to_dlogits_(
    probs: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def linear_logp_local_probs_bf16_to_dlogits_(
    probs: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    local_lse: torch.Tensor,
    global_lse: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def linear_logp_logits_bf16_to_dlogits(
    logits: torch.Tensor,
    dlogits: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    lse: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def embedding_sm90_forward(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
def embedding_sm90_forward_fp32(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
def lm_head_sm90_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor: ...
def lm_head_sm90_forward_fp32(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor: ...
def fused_logp_forward_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_fp32(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def fused_logp_forward_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def deterministic_logp_forward_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_tile_stats(
    logits: torch.Tensor,
    vocab_start: int,
    real_vocab: int,
    num_tiles: int,
) -> list[torch.Tensor]: ...
def hip_deterministic_logp_tile_stats(
    logits: torch.Tensor,
    vocab_start: int,
    real_vocab: int,
    num_tiles: int,
) -> list[torch.Tensor]: ...
def hip_deterministic_logp_backward(
    logits: torch.Tensor,
    lse: torch.Tensor,
    coef_logp: torch.Tensor,
    coef_lse: torch.Tensor,
    target_local: torch.Tensor,
    vocab_start: int,
    real_vocab: int,
    has_lse_grad: bool,
) -> torch.Tensor: ...
def deterministic_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
    key_padding_mask: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Returns [out, lse, P]."""
    ...

def deterministic_attention_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    P: torch.Tensor,
    causal: bool,
    scale: float,
    key_padding_mask: torch.Tensor | None,
) -> list[torch.Tensor]: ...
def deterministic_rope_apply_rocm(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sin_sign: float,
) -> torch.Tensor: ...
def deterministic_rope_apply_token_major_rocm(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    sin_sign: float,
) -> torch.Tensor: ...
def det_gemm_sm90_compiled() -> bool: ...
def det_gemm_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
def det_gemm_fwd_rhs_transposed(
    a: torch.Tensor,
    bt: torch.Tensor,
) -> torch.Tensor: ...
def det_gemm_da(dc: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
def det_gemm_db(a: torch.Tensor, dc: torch.Tensor) -> torch.Tensor: ...
def det_gemm_db_transposed(a: torch.Tensor, dc: torch.Tensor) -> torch.Tensor: ...
def silu_forward(x: torch.Tensor) -> torch.Tensor: ...
def silu_backward(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...
def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...
def swiglu_backward(
    dy: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> list[torch.Tensor]: ...
def rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> list[torch.Tensor]: ...
def rmsnorm_backward_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> torch.Tensor: ...
def rmsnorm_backward_dw(
    dy: torch.Tensor,
    x: torch.Tensor,
    rstd: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_collective_rocm_ipc_allocate(size_bytes: int) -> torch.Tensor: ...
def deterministic_collective_rocm_ipc_meta(tensor: torch.Tensor) -> tuple[list[int], int]: ...
def deterministic_collective_rocm_ipc_create(
    staging: torch.Tensor,
    handles: list[list[int]],
    offsets: list[int],
    rank: int,
) -> int: ...
def deterministic_collective_rocm_ipc_synchronize(handle: int) -> None: ...
def deterministic_collective_rocm_ipc_destroy(handle: int) -> None: ...
def deterministic_collective_rocm_ipc_stage(handle: int, input: torch.Tensor) -> None: ...
def deterministic_collective_rocm_ipc_prepare_staged(
    handle: int,
    input: torch.Tensor,
) -> None: ...
def deterministic_collective_rocm_ipc_all_reduce_staged(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
) -> None: ...
def deterministic_collective_rocm_ipc_all_reduce(
    handle: int,
    output: torch.Tensor,
) -> None: ...
def deterministic_collective_rocm_ipc_all_reduce_input(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
) -> None: ...
def deterministic_collective_rocm_ipc_reduce_scatter(
    handle: int,
    output: torch.Tensor,
) -> None: ...
def deterministic_collective_rocm_ipc_reduce_scatter_input(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
) -> None: ...
def deterministic_collective_rocm_ipc_reduce_scatter_many(
    handle: int,
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
) -> None: ...
def deterministic_collective_rocm_ipc_all_gather(
    handle: int,
    output: torch.Tensor,
) -> None: ...
def deterministic_collective_rocm_ipc_all_gather_input(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
) -> None: ...

# MiniMax-H3 kernels (RFC #420). These exist only in a build that compiles the H3 sources: not on
# ROCm, and not with KERNEL_ALIGN_USE_FAST_MATH=1 or relaxed NVCC_*_FLAGS (build_tools/extensions.py
# leaves the sources and the KERNEL_ALIGN_WITH_H3_KERNELS define out together). Like the ROCm and
# SM90 symbols above, they are declared here for type checking; the Python backends test for them
# (the ``*_available()`` helpers, called by every constructor) so dispatch falls back to the PyTorch
# reference instead of raising AttributeError.
def h3_timestep_sinusoid_forward(
    timestep: torch.Tensor,
    num_channels: int = 256,
    max_period: float = 10000.0,
    check_range: bool = True,
) -> torch.Tensor:
    """Return FP32 CUDA cosine-then-sine features of shape ``(T, num_channels)``.

    Require a nonempty 1-D FP32 CUDA timestep tensor, positive even channels
    and positive ``max_period``. Check finite values in ``[0, 1]`` when enabled;
    the Python operator supplies the analytic backward for this native forward.
    """
    ...

def h3_det_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: int = 0,
    save_pre_activation: bool = False,
) -> list[torch.Tensor]:
    """Return a deterministic CUDA projection and optional FP32 pre-activation.

    Require same-device/dtype FP32 or BF16 contiguous, 16-byte-aligned matrices
    ``x (T, K)`` and ``weight (N, K)``, ``T > 0`` and ``K`` divisible by the
    16-byte vector width. Optional contiguous bias is ``(N,)`` in that dtype.
    Activation 0 is identity and 1 is SiLU; BF16 forward requires SM80 or newer.
    Return ``[out]`` in the input dtype or ``[out, pre]`` when saving activation.
    """
    ...

def h3_det_linear_backward_input(
    grad: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Return a deterministic ``(T, K)`` CUDA input gradient in ``out_dtype``.

    Require contiguous FP32 ``grad (T, N)`` with ``T > 0`` and same-device
    contiguous, 16-byte-aligned FP32 or BF16 ``weight (N, K)``. Output dtype
    must be FP32 or BF16; output-column chunks accumulate in fixed order.
    """
    ...

def h3_det_linear_backward_input_partials(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the ``(C, T, K)`` FP32 per-chunk partials of ``grad @ weight`` before the fold.

    Same operands as ``h3_det_linear_backward_input``. Chunk ``c`` of ``C = ceil(N / 64)`` sums the
    output columns ``64c .. 64c + 63`` in ascending order; the tensor-parallel path reduces its
    shard's chunks with ``h3_det_linear_fold_chunks`` after the shards are gathered.
    """
    ...

def h3_det_linear_fold_chunks(partial: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Left-fold ``partial (C, T, K)`` FP32 over ``C`` in ascending order, cast once to ``out_dtype``.

    Require a non-empty contiguous FP32 CUDA tensor and an FP32 or BF16 ``out_dtype``; return
    ``(T, K)``.
    """
    ...

def h3_det_linear_backward_weight(
    grad: torch.Tensor, x: torch.Tensor, w_dtype: torch.dtype, with_bias: bool = True
) -> list[torch.Tensor]:
    """Return CUDA weight and optional bias gradients using an ascending row fold.

    Require contiguous FP32 ``grad (T, N)`` with ``T > 0`` and same-device
    contiguous, 16-byte-aligned FP32 or BF16 ``x (T, K)``. Return ``[dweight]``
    with shape ``(N, K)``, or ``[dweight, dbias]`` with ``dbias (N,)`` when
    requested, in FP32 or BF16 as specified by ``w_dtype``.
    """
    ...

def h3_adaln_row_gather_forward(
    rows: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    chunks: int = 6,
    modality_num: int = 3,
) -> torch.Tensor:
    """Gather AdaLN table rows into packed per-position chunks; return ``(chunks, S, H)``.

    ``rows`` is a 2-D CUDA ``(R, chunks * H)`` table with unit column stride and ``R`` a multiple of
    ``modality_num``; ``timestep_indices`` and ``token_tags`` are ``(S,)`` contiguous integer
    tensors (int64 or int32, one dtype) on its device, ``S > 0``. Position ``s`` reads table row
    ``timestep_indices[s] * modality_num + token_tags[s]`` and receives its ``chunks`` slices of
    width ``H`` in ``rows``' dtype (a pure copy). Out-of-range indices stop the kernel with a
    device-side assert.
    """
    ...

def h3_adaln_row_gather_backward(
    grad: torch.Tensor,
    sorted_pos: torch.Tensor,
    tile_begin: torch.Tensor,
    tile_end: torch.Tensor,
    seg_first_tile: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Sum a ``(chunks, S, H)`` gradient into table rows; return ``(R, chunks * H)`` in ``out_dtype``.

    ``sorted_pos`` holds the ``S`` positions sorted stably by table row, ``[tile_begin, tile_end)``
    cut that order into fixed-size tiles, and ``seg_first_tile`` (``R + 1`` entries) gives the
    first tile of each row (see ``_segment_tiles``). Tiles are summed in FP32 and folded in order,
    with no atomics. The metadata is checked on the device first: positions inside ``[0, S)``, tiles
    inside ``sorted_pos``, segments that cut the tile list exactly and in order.
    """
    ...

def h3_rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    index: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """RMSNorm over the last dimension of a contiguous CUDA ``(M, N)`` tensor; return ``[y, rstd]``.

    ``N`` is a multiple of 4 (and at most ``INT_MAX``); ``weight`` is ``(N,)`` in ``x``'s dtype
    (FP32, FP16 or BF16). The reduction replays PyTorch's vectorized RMSNorm, so ``y`` is bitwise
    ``nn.RMSNorm``. With ``shift``/``scale`` (views of one ``(R, N)`` table with unit column stride
    and a row stride of at least ``N``) and ``index`` (``(S,)`` int64, ``M`` a multiple of ``S``),
    row ``r`` also gets ``cast(cast(n * cast(1 + scale[i])) + shift[i])`` with
    ``i = index[r % S]``; an index outside ``[0, R)`` stops with a device-side assert. ``rstd`` is
    the FP32 ``(M,)`` statistic the backward reuses.
    """
    ...

def h3_rmsnorm_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    index: torch.Tensor | None = None,
    sorted_pos: torch.Tensor | None = None,
    tile_begin: torch.Tensor | None = None,
    tile_end: torch.Tensor | None = None,
    seg_first_tile: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Deterministic backward of ``h3_rmsnorm_forward``; return ``[dx, dweight]`` in the input dtype.

    ``grad`` and ``x`` are ``(M, N)``, ``rstd`` comes from the forward, and ``shift``/``scale``/
    ``index`` are the forward's. ``dx`` is row-local; ``dweight`` sums fixed 256-row tiles in order.
    With modulation the result continues with the FP32 ``(R, N)`` table gradients
    ``[d_shift, d_scale]``, summed over stable-sorted tiles of rows per table row: pass
    ``sorted_pos``, ``tile_begin``, ``tile_end`` and ``seg_first_tile`` as built by
    ``adaln_row_gather._segment_tiles``; they are checked on the device before use. Without
    modulation they must be omitted.
    """
    ...

def h3_gate_residual_forward(
    residual: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """Return ``residual + gate[index[r % S]] * y`` for contiguous CUDA ``(M, N)`` tensors.

    ``residual`` and ``y`` share a dtype (FP32, FP16 or BF16); ``gate`` is an ``(R, N)`` view of an
    AdaLN table with unit column stride and a row stride of at least ``N``; ``index`` is a
    contiguous int64 ``(S,)`` tensor with ``M`` a multiple of ``S``. The gate row is gathered in the
    kernel and the two roundings (gate * y, then the add) are the eager expression's, so the result
    is bitwise the eager one. An index outside ``[0, R)`` stops with a device-side assert.
    """
    ...

def h3_gate_residual_backward(
    grad: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    index: torch.Tensor,
    sorted_pos: torch.Tensor,
    tile_begin: torch.Tensor,
    tile_end: torch.Tensor,
    seg_first_tile: torch.Tensor,
) -> list[torch.Tensor]:
    """Return ``[d_y, d_gate]`` of ``h3_gate_residual_forward`` (``d_residual`` is ``grad`` itself).

    ``d_y = grad * gate[index]`` is bitwise the eager VJP. ``d_gate`` is ``(R, N)`` in the gate
    dtype: a deterministic FP32 sum over stable-sorted tiles of rows per table row, from
    ``sorted_pos``, ``tile_begin``, ``tile_end`` and ``seg_first_tile`` as built by
    ``adaln_row_gather._segment_tiles`` (``R`` is ``seg_first_tile.numel() - 1``); the metadata is
    checked on the device before use.
    """
    ...

def h3_rmsnorm_backward_partials(
    grad: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None,
    index: torch.Tensor | None,
    dw_rows: torch.Tensor,
    dw_begin: torch.Tensor,
    dw_end: torch.Tensor,
    seg_rows: torch.Tensor | None = None,
    seg_begin: torch.Tensor | None = None,
    seg_end: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Sequence parallel: FP32 tile partials of the rmsnorm weight (and table) gradients.

    Same operands as ``h3_rmsnorm_backward`` plus explicit int64 row lists: ``dw_rows`` with tiles
    ``[dw_begin, dw_end)`` over it, and with modulation ``seg_rows`` with ``[seg_begin, seg_end)``
    tiles; ``index`` then has one entry per row of ``x``. Returns ``[dw_partial (tiles, N)]`` or
    ``[dw_partial, seg_partial (seg_tiles, 2N)]``. Lists and index values are checked on the device
    before use; ``h3_rmsnorm_fold_partials`` folds the gathered partials.
    """
    ...

def h3_rmsnorm_fold_partials(
    dw_partial: torch.Tensor,
    weight: torch.Tensor,
    seg_partial: torch.Tensor | None = None,
    seg_first_tile: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Fold gathered ``h3_rmsnorm_backward_partials`` in tile order.

    Return ``[dweight]`` in ``weight``'s dtype, or ``[dweight, d_shift, d_scale]`` (FP32 ``(R, N)``)
    when ``seg_partial`` and ``seg_first_tile`` (``R + 1`` entries) are given.
    """
    ...

def h3_gate_grad_partials(
    grad: torch.Tensor,
    y: torch.Tensor,
    rows: torch.Tensor,
    tile_begin: torch.Tensor,
    tile_end: torch.Tensor,
) -> torch.Tensor:
    """Sequence parallel: FP32 ``(tiles, N)`` partials of ``sum(grad * y)`` over tiles of ``rows``.

    ``rows`` is an int64 list of rows of ``grad``/``y`` and ``[tile_begin, tile_end)`` cut it into
    tiles; the lists are bounds-checked on the device before use. That the tiles cover ``rows`` in
    order, with no gap and no repeated row, is the caller's contract and is not verified (unlike
    ``h3_gate_residual_backward``): a gap silently drops rows from ``d_gate``.
    """
    ...

def h3_gate_grad_fold(
    partial: torch.Tensor, seg_first_tile: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Fold gathered ``h3_gate_grad_partials`` per table row; return ``(R, N)`` in ``dtype``.

    ``seg_first_tile`` has ``R + 1`` entries into the tile axis of ``partial``, checked on the device.
    """
    ...

def h3_rmsnorm_backward_dx(
    grad: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    index: torch.Tensor | None = None,
) -> torch.Tensor:
    """The row-local ``dx`` of ``h3_rmsnorm_backward`` alone, in ``x``'s dtype (sequence parallel)."""
    ...

def h3_gate_residual_backward_dy(
    grad: torch.Tensor, gate: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """``d_y = grad * gate[index[r % S]]``, the exact part of ``h3_gate_residual_backward``."""
    ...
