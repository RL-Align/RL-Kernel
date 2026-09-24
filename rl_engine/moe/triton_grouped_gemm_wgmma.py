# SPDX-License-Identifier: Apache-2.0
"""Experimental SM90 MXFP8 x MXFP4 grouped GEMM with GPU scheduling.

GPU prep builds per-expert element offsets and an exact M-tile prefix. A fixed
number of programs traverses that prefix without reading GPU contents on host.
Large E uses a recursive GPU scan, never a CPU fallback. Default checked entry
points synchronize to report invalid content; ``validate_contents=False`` is
ONLY for inputs whose offsets and E8M0 codes the caller guarantees are valid.

All runtime checks, MX decoders, metadata preparation and compute kernels live
in this module; only the existing MXTensor data contract is imported locally.
Not registered as a provider until SM90 numeric/PTX/sanitizer gates pass.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from rl_engine.moe.mx_format import MX_BLOCK, MXTensor, NIBBLE_PACKING

# Device descriptor entries are int64 element offsets, not byte addresses.
ROW_BEGIN, ROW_END, ROW_COUNT, WEIGHT, WEIGHT_SCALE, INPUT, INPUT_SCALE, OUTPUT = range(8)
DESC_FIELDS = 8
SCAN_BLOCK = 1024

BM = 64
BN_FWD = 64
BN_BWD = 64
BK_BWD = 64
NUM_WARPS = 4
NUM_STAGES = 2


def validate_geometry(m: int, e: int, n: int, k: int, block_m: int, q: int) -> int:
    """Host-only bounds. No allocation or access to dynamic device contents."""
    if min(m, e, n, q) < 0 or k <= 0 or k % 32 or block_m <= 0:
        raise ValueError("invalid grouped GEMM geometry")
    if max(m, e, n, k) > 2**31 - 1 or (e == 0 and m != 0):
        raise ValueError("M/E/N/K must fit int32 dimensions; E=0 requires M=0")
    upper = min(m, (m + block_m - 1) // block_m + e - 1) if m else 0
    if upper * q >= 2**31:
        raise ValueError("conservative output tile bound exceeds int32 range")
    if max(m * k, m * n, e * n * (k // 2), e * n * (k // 32)) >= 2**63:
        raise ValueError("tensor element address exceeds int64 range")
    # arange additions and binary-search bounds also stay in int64 on device.
    return upper * q


def _validate_offsets(offsets, m, e, device, validate_contents):
    if offsets.dtype != torch.int32 or tuple(offsets.shape) != (e + 1,):
        raise ValueError("expert_offsets must be int32 [E+1]")
    if offsets.device != device or not offsets.is_contiguous():
        raise ValueError("expert_offsets must be contiguous on the input device")
    if validate_contents:
        # One scalar synchronization, not an offsets copy or CPU-built prefix.
        valid = (offsets[0] == 0) & (offsets[-1] == m) & (offsets[1:] >= offsets[:-1]).all()
        if not bool(valid):
            raise ValueError("expert_offsets must start at 0, end at M and be nondecreasing")


def _program_count(device, programs):
    if programs is None:
        return torch.cuda.get_device_properties(device).multi_processor_count
    if isinstance(programs, bool) or not isinstance(programs, int) or not 0 < programs < 2**31:
        raise ValueError("programs must be a positive int32 count")
    return programs


def _validate_cuda_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    for name, value in tensors.items():
        if value.device != device or not value.is_cuda or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the input CUDA device")


def _validate_scale_codes(**scales: torch.Tensor) -> None:
    # Checked/debug entry only: each scalar read synchronizes. A valid MXTensor
    # shape/dtype alone is not proof that its scale contents exclude code 255.
    for name, codes in scales.items():
        if bool((codes == 255).any()):
            raise ValueError(f"{name} contains invalid E8M0 code 255")


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError("P5-4 Triton WGMMA requires Triton")


def _check_sm90(tensor: torch.Tensor) -> None:
    # Preserve the HPC entry's exact SM90 check rather than accepting all SM9x.
    if not tensor.is_cuda or torch.cuda.get_device_capability(tensor.device) != (9, 0):
        raise RuntimeError("P5-4 Triton WGMMA requires a CUDA SM90 device")


def _require_sm90(tensor: torch.Tensor) -> None:
    _require_triton()
    _check_sm90(tensor)


if triton is not None:
    # Explicit constexpr globals: current Triton rejects ordinary Python globals
    # in JIT kernels. Keep host allocation dimensions as plain Python ints.
    _D_FIELDS = tl.constexpr(DESC_FIELDS)
    _D_BEGIN = tl.constexpr(ROW_BEGIN)
    _D_END = tl.constexpr(ROW_END)
    _D_ROWS = tl.constexpr(ROW_COUNT)
    _D_W = tl.constexpr(WEIGHT)
    _D_WS = tl.constexpr(WEIGHT_SCALE)
    _D_INPUT = tl.constexpr(INPUT)
    _D_IS = tl.constexpr(INPUT_SCALE)
    _D_OUTPUT = tl.constexpr(OUTPUT)

    @triton.jit
    def _pow2_int(k):
        # k: int32 in [-127, 127]. Return fp32 == 2**k exactly (no libm rounding).
        # Normal range k >= -126: exponent field is (k + 127), mantissa 0.
        # k == -127: subnormal 2^-127 == 0x00400000. P5 never reaches k < -127.
        bits = tl.where(k >= -126, (k + 127) << 23, tl.where(k == -127, 1 << 22, 0))
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _decode_e8m0(code):
        # OCP E8M0 scale: 2^(code - 127), exact. Code 255 (NaN) is rejected upstream.
        return _pow2_int(code.to(tl.int32) - 127)

    @triton.jit
    def _decode_e2m1(nibble):
        # OCP E2M1 nibble: sign in bit 3, magnitude in {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
        # Arithmetic decode (no table): exp == 0 is subnormal mant/2; otherwise
        # (1 + mant/2) * 2^(exp-1). All eight magnitudes are exact in FP32.
        n = nibble.to(tl.int32)
        sign = (n >> 3) & 1
        exp = (n >> 1) & 0x3
        mant = n & 0x1
        half = mant.to(tl.float32) * 0.5  # exact: 0.0 or 0.5
        norm = (1.0 + half) * _pow2_int(exp - 1)  # exact
        mag = tl.where(exp == 0, half, norm)
        return tl.where(sign != 0, -mag, mag)

    @triton.jit
    def _prep_kernel(
        OFFSETS,
        DESC,
        PREFIX,
        SUMS,
        N: tl.constexpr,
        K: tl.constexpr,
        E: tl.constexpr,
        BLOCK_M: tl.constexpr,
        B: tl.constexpr,
        BACKWARD: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        e = chunk.to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
        begin = tl.load(OFFSETS + e, e < E, other=0).to(tl.int64)
        end = tl.load(OFFSETS + e + 1, e < E, other=0).to(tl.int64)
        rows = end - begin
        d = DESC + e * _D_FIELDS
        tl.store(d + _D_BEGIN, begin, e < E)
        tl.store(d + _D_END, end, e < E)
        tl.store(d + _D_ROWS, rows, e < E)
        tl.store(d + _D_W, e * (N * (K // 2)), e < E)
        tl.store(d + _D_WS, e * (N * (K // 32)), e < E)
        if BACKWARD:
            tl.store(d + _D_INPUT, begin * N, e < E)
            tl.store(d + _D_IS, tl.full((B,), 0, tl.int64), e < E)
            tl.store(d + _D_OUTPUT, begin * K, e < E)
        else:
            tl.store(d + _D_INPUT, begin * K, e < E)
            tl.store(d + _D_IS, begin * (K // 32), e < E)
            tl.store(d + _D_OUTPUT, begin * N, e < E)
        tiles = (rows + BLOCK_M - 1) // BLOCK_M
        inclusive = tl.cumsum(tiles, 0)
        tl.store(PREFIX + e + 1, inclusive, e < E)
        tl.store(SUMS + chunk, tl.sum(tiles, 0))
        if chunk == 0:
            tl.store(PREFIX, 0)

    @triton.jit
    def _scan_kernel(X, Y, SUMS, SIZE: tl.constexpr, B: tl.constexpr):
        chunk = tl.program_id(0)
        i = chunk.to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
        x = tl.load(X + i, i < SIZE, other=0)
        tl.store(Y + i, tl.cumsum(x, 0), i < SIZE)
        tl.store(SUMS + chunk, tl.sum(x, 0))

    @triton.jit
    def _add_scan_carries(
        Y, CARRIES, SIZE: tl.constexpr, B: tl.constexpr, PREFIX_SHIFT: tl.constexpr
    ):
        chunk = tl.program_id(0)
        i = chunk.to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
        carry = tl.load(CARRIES + chunk - 1, chunk > 0, other=0)
        ptr = Y + i + PREFIX_SHIFT
        value = tl.load(ptr, i < SIZE, other=0)
        tl.store(ptr, value + carry, i < SIZE)

    @triton.jit
    def _lookup_expert(prefix_ptr, global_m_tile, E: tl.constexpr):
        """Lower_bound(prefix[e+1] > global_m_tile), including duplicate entries."""
        lo = tl.full((), 0, tl.int32)
        hi = tl.full((), E, tl.int32)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            next_prefix = tl.load(prefix_ptr + mid + 1)
            if global_m_tile < next_prefix:
                hi = mid
            else:
                lo = mid + 1
        e = lo
        local_tile = global_m_tile - tl.load(prefix_ptr + e)
        return e, local_tile

    @triton.jit
    def _fwd_kernel(
        A,
        AS,
        W,
        WS,
        DESC,
        PREFIX,
        Y,
        N: tl.constexpr,
        K: tl.constexpr,
        E: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        OUT_TILES: tl.constexpr,
    ):
        # int64 also prevents overflow on the final pid += stride.
        pid = tl.program_id(0).to(tl.int64)
        total = tl.load(PREFIX + E) * OUT_TILES
        while pid < total:
            global_m_tile = pid // OUT_TILES
            out_tile = pid % OUT_TILES
            e, local_m_tile = _lookup_expert(PREFIX, global_m_tile, E)
            d = DESC + e.to(tl.int64) * _D_FIELDS

            row_count = tl.load(d + _D_ROWS)
            local_m = local_m_tile.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
            valid_m = local_m < row_count
            out_n = out_tile.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
            valid_n = out_n < N
            out_n64 = out_n.to(tl.int64)

            # Each iteration covers exactly one MX scale block of 32 K elements.
            inner_k = tl.arange(0, 32)
            byte_k = inner_k >> 1
            nibble_shift = (inner_k & 1) << 2
            blocks = K // 32
            a_ptrs = A + tl.load(d + _D_INPUT) + local_m[:, None] * K + inner_k[None, :]
            w_ptrs = W + tl.load(d + _D_W) + out_n64[None, :] * (K // 2) + byte_k[:, None]
            as_base = AS + tl.load(d + _D_IS) + local_m * blocks
            ws_base = WS + tl.load(d + _D_WS) + out_n64 * blocks

            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for j in range(blocks):
                a_bits = tl.load(a_ptrs, mask=valid_m[:, None], other=0)
                a = a_bits.to(tl.float8e4nv, bitcast=True)
                packed = tl.load(w_ptrs, mask=valid_n[None, :], other=0)
                nibble = (packed.to(tl.int32) >> nibble_shift[:, None]) & 15
                b = _decode_e2m1(nibble).to(tl.float8e4nv)
                # One FP8 instruction reduction per MX block; pin this compiler
                # option rather than inheriting a version-dependent default.
                partial = tl.dot(a, b, max_num_imprecise_acc=32)

                sa_code = tl.load(as_base + j, mask=valid_m, other=127)
                sw_code = tl.load(ws_base + j, mask=valid_n, other=127)
                sa = _decode_e8m0(sa_code)
                sw = _decode_e8m0(sw_code)
                acc += (partial * sa[:, None]) * sw[None, :]

                a_ptrs += 32
                w_ptrs += 16

            tl.store(
                Y + tl.load(d + _D_OUTPUT) + local_m[:, None] * N + out_n[None, :],
                acc,
                mask=valid_m[:, None] & valid_n[None, :],
            )
            pid += tl.num_programs(0)

    @triton.jit
    def _bwd_kernel(
        DY,
        W,
        WS,
        DESC,
        PREFIX,
        DX,
        N: tl.constexpr,
        K: tl.constexpr,
        E: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_TILES: tl.constexpr,
    ):
        # int64 also prevents overflow on the final pid += stride.
        pid = tl.program_id(0).to(tl.int64)
        total = tl.load(PREFIX + E) * OUT_TILES
        while pid < total:
            global_m_tile = pid // OUT_TILES
            out_tile = pid % OUT_TILES
            e, local_m_tile = _lookup_expert(PREFIX, global_m_tile, E)
            d = DESC + e.to(tl.int64) * _D_FIELDS

            row_count = tl.load(d + _D_ROWS)
            local_m = local_m_tile.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
            valid_m = local_m < row_count

            out_k = out_tile.to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K)
            valid_k = out_k < K
            byte_k = out_k >> 1
            nibble_shift = (out_k & 1) << 2
            scale_k = out_k >> 5
            blocks = K // 32
            inner_n = tl.arange(0, BLOCK_N)
            inner_n64 = inner_n.to(tl.int64)

            dy_ptrs = DY + tl.load(d + _D_INPUT) + local_m[:, None] * N + inner_n[None, :]
            w_ptrs = W + tl.load(d + _D_W) + inner_n64[:, None] * (K // 2) + byte_k[None, :]
            ws_ptrs = WS + tl.load(d + _D_WS) + inner_n64[:, None] * blocks + scale_k[None, :]

            acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
            for n_tile in range(tl.cdiv(N, BLOCK_N)):
                valid_n = n_tile.to(tl.int64) * BLOCK_N + inner_n64 < N
                dy = tl.load(
                    dy_ptrs,
                    mask=valid_m[:, None] & valid_n[None, :],
                    other=0,
                )
                packed = tl.load(
                    w_ptrs,
                    mask=valid_n[:, None] & valid_k[None, :],
                    other=0,
                )
                nibble = (packed.to(tl.int32) >> nibble_shift[None, :]) & 15
                sw_code = tl.load(
                    ws_ptrs,
                    mask=valid_n[:, None] & valid_k[None, :],
                    other=127,
                )
                sw = _decode_e8m0(sw_code)
                b = (_decode_e2m1(nibble) * sw).to(tl.bfloat16)
                acc += tl.dot(dy, b)  # Reduction dimension is N, not output K.

                dy_ptrs += BLOCK_N
                w_ptrs += BLOCK_N * (K // 2)
                ws_ptrs += BLOCK_N * blocks

            tl.store(
                DX + tl.load(d + _D_OUTPUT) + local_m[:, None] * K + out_k[None, :],
                acc,
                mask=valid_m[:, None] & valid_k[None, :],
            )
            pid += tl.num_programs(0)

    @triton.jit
    def _schedule_probe_kernel(
        DESC, PREFIX, IDS, E: tl.constexpr, Q: tl.constexpr, BLOCK_M: tl.constexpr
    ):
        pid = tl.program_id(0).to(tl.int64)
        total = tl.load(PREFIX + E) * Q
        while pid < total:
            e, local = _lookup_expert(PREFIX, pid // Q, E)
            d = DESC + e.to(tl.int64) * _D_FIELDS
            row = local * BLOCK_M + tl.arange(0, BLOCK_M)
            begin = tl.load(d + _D_BEGIN)
            rows = tl.load(d + _D_ROWS)
            tl.store(IDS + (begin + row) * Q + pid % Q, pid, row < rows)
            pid += tl.num_programs(0)


def _inclusive_scan(x):
    size = x.numel()
    b = min(SCAN_BLOCK, triton.next_power_of_2(size))
    chunks = triton.cdiv(size, b)
    y = torch.empty_like(x)
    sums = torch.empty((chunks,), dtype=torch.int64, device=x.device)
    _scan_kernel[(chunks,)](x, y, sums, SIZE=size, B=b, num_warps=4)
    if chunks > 1:
        carries = _inclusive_scan(sums)
        _add_scan_carries[(chunks,)](y, carries, SIZE=size, B=b, PREFIX_SHIFT=0)
    return y


def _prepare_metadata(offsets, n, k, *, backward=False):
    e = offsets.numel() - 1
    desc = torch.empty((e, DESC_FIELDS), dtype=torch.int64, device=offsets.device)
    prefix = torch.empty((e + 1,), dtype=torch.int64, device=offsets.device)
    b = min(SCAN_BLOCK, triton.next_power_of_2(max(e, 1)))
    chunks = max(1, triton.cdiv(e, b))
    sums = torch.empty((chunks,), dtype=torch.int64, device=offsets.device)
    _prep_kernel[(chunks,)](
        offsets,
        desc,
        prefix,
        sums,
        N=n,
        K=k,
        E=e,
        BLOCK_M=BM,
        B=b,
        BACKWARD=backward,
        num_warps=4,
    )
    if chunks > 1:
        carries = _inclusive_scan(sums)
        _add_scan_carries[(chunks,)](prefix, carries, SIZE=e, B=b, PREFIX_SHIFT=1)
    return desc, prefix


def _launch_fwd(a, w, desc, prefix, y, programs):
    e, n, k = w.shape
    return _fwd_kernel[(programs,)](
        a.codes,
        a.scales,
        w.codes,
        w.scales,
        desc,
        prefix,
        y,
        N=n,
        K=k,
        E=e,
        BLOCK_M=BM,
        BLOCK_N=BN_FWD,
        OUT_TILES=triton.cdiv(n, BN_FWD),
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        enable_fp_fusion=False,
    )


def _launch_bwd(dy, w, desc, prefix, dx, programs):
    e, n, k = w.shape
    return _bwd_kernel[(programs,)](
        dy,
        w.codes,
        w.scales,
        desc,
        prefix,
        dx,
        N=n,
        K=k,
        E=e,
        BLOCK_M=BM,
        BLOCK_N=BN_BWD,
        BLOCK_K=BK_BWD,
        OUT_TILES=triton.cdiv(k, BK_BWD),
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        enable_fp_fusion=False,
    )


def grouped_gemm_fwd(
    a: MXTensor,
    w: MXTensor,
    expert_offsets: torch.Tensor,
    *,
    validate_contents: bool = True,
    programs: int | None = None,
) -> torch.Tensor:
    """FP32 Y[M,N]. Checked by default; trusted mode requires valid contents.

    ``validate_contents=False`` skips offsets/scale value checks and their host
    synchronizations. Caller must guarantee valid contents until GPU work ends;
    MXTensor construction alone is insufficient. Shape/device checks remain.
    ``programs`` controls the fixed grid for tuning; None uses the SM count.
    """
    if not isinstance(validate_contents, bool):
        raise TypeError("validate_contents must be bool")
    if not isinstance(a, MXTensor) or not isinstance(w, MXTensor):
        raise TypeError("expected MXTensor activation and weight")
    if a.elem_format != "e4m3" or w.elem_format != "e2m1":
        raise ValueError("expected E4M3 activation and E2M1 weight")
    _require_sm90(a.codes)
    if a.codes.ndim != 2 or w.codes.ndim != 3:
        raise ValueError("expected A[M,K] and W[E,N,K/2]")
    m, k = a.codes.shape
    e, n, half_k = w.codes.shape
    if k <= 0 or k % MX_BLOCK or half_k * 2 != k:
        raise ValueError("K must be positive, divisible by 32, and match W")
    if tuple(a.shape) != (m, k) or tuple(w.shape) != (e, n, k):
        raise ValueError("MXTensor logical shapes do not match codes")
    if tuple(a.scales.shape) != (m, k // 32) or tuple(w.scales.shape) != (e, n, k // 32):
        raise ValueError("MX scale shapes do not match codes")
    _validate_cuda_tensors(
        a.codes.device,
        a_codes=a.codes,
        a_scales=a.scales,
        w_codes=w.codes,
        w_scales=w.scales,
    )
    if a.packing != NIBBLE_PACKING or w.packing != NIBBLE_PACKING:
        raise ValueError("expected nibble-lo-first packing")
    if validate_contents:
        _validate_scale_codes(a_scales=a.scales, w_scales=w.scales)
    q = triton.cdiv(n, BN_FWD) if n else 0
    validate_geometry(m, e, n, k, BM, q)
    _validate_offsets(expert_offsets, m, e, a.codes.device, validate_contents)
    c = _program_count(a.codes.device, programs)
    # Triton launches on the active device. Honor tensors on cuda:1 even if the
    # caller's active device is cuda:0, preserving that device's current stream.
    with torch.cuda.device(a.codes.device):
        y = torch.empty((m, n), device=a.codes.device, dtype=torch.float32)
        if m and n:
            desc, prefix = _prepare_metadata(expert_offsets, n, k)
            _launch_fwd(a, w, desc, prefix, y, c)
    return y


def grouped_gemm_bwd(
    dy: torch.Tensor,
    w: MXTensor,
    expert_offsets: torch.Tensor,
    *,
    validate_contents: bool = True,
    programs: int | None = None,
) -> torch.Tensor:
    """FP32 dX[M,K]; no activation scale or dW. Same trusted contract as fwd."""
    if not isinstance(validate_contents, bool):
        raise TypeError("validate_contents must be bool")
    if not isinstance(w, MXTensor) or w.elem_format != "e2m1":
        raise TypeError("expected E2M1 MXTensor weight")
    _require_sm90(dy)
    if dy.ndim != 2 or w.codes.ndim != 3 or dy.dtype != torch.bfloat16:
        raise ValueError("expected BF16 dy[M,N] and W[E,N,K/2]")
    m, n = dy.shape
    e, wn, half_k = w.codes.shape
    k = half_k * 2
    if n != wn or k <= 0 or k % MX_BLOCK:
        raise ValueError("dy N and W N must match; K must be positive and divisible by 32")
    if tuple(w.shape) != (e, n, k) or tuple(w.scales.shape) != (e, n, k // 32):
        raise ValueError("MX weight shapes do not match codes")
    _validate_cuda_tensors(
        dy.device,
        dy=dy,
        w_codes=w.codes,
        w_scales=w.scales,
    )
    if w.packing != NIBBLE_PACKING:
        raise ValueError("expected nibble-lo-first packing")
    if validate_contents:
        _validate_scale_codes(w_scales=w.scales)
    q = triton.cdiv(k, BK_BWD)
    validate_geometry(m, e, n, k, BM, q)
    _validate_offsets(expert_offsets, m, e, dy.device, validate_contents)
    c = _program_count(dy.device, programs)
    with torch.cuda.device(dy.device):
        dx = torch.empty((m, k), device=dy.device, dtype=torch.float32)
        if n == 0:
            dx.zero_()
        elif m:
            desc, prefix = _prepare_metadata(expert_offsets, n, k, backward=True)
            _launch_bwd(dy, w, desc, prefix, dx, c)
    return dx
