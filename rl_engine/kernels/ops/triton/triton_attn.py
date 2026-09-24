import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Initialize offsets
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr)

    # Determine loop bounds for K and V
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Calculate QK^T
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # Block-wise Causal Masking
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        #  softmax max sum
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(beta, 1)

        # update scale V
        p_scale = beta / l_i_new[:, None]
        acc_scale = l_i / l_i_new * alpha

        acc = acc * acc_scale[:, None]
        v = tl.load(V_block_ptr)
        p = p_scale.to(v.dtype)
        acc += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    acc = acc.to(Out.dtype.element_ty)
    tl.store(O_block_ptr, acc)

    off_zh = off_hz * N_CTX
    l_ptrs = L + off_zh + offs_m
    m_ptrs = M + off_zh + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)

    o_offset = off_hz * stride_oh
    do_offset = off_hz * stride_doh

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(tl.program_id(0) * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_dom, stride_don),
        offsets=(tl.program_id(0) * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )

    o = tl.load(O_block_ptr)
    do = tl.load(DO_block_ptr).to(o.dtype)

    delta = tl.sum(o * do, axis=1)

    off_zh = off_hz * N_CTX
    tl.store(Delta + off_zh + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    M,
    Delta,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    qvk_offset = off_hz * stride_qh

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    lo = start_n * BLOCK_N if IS_CAUSAL else 0
    hi = N_CTX

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = tl.load(Q_block_ptr)
        do = tl.load(DO_block_ptr)

        off_zh = off_hz * N_CTX
        m_i = tl.load(M + off_zh + offs_m)
        l_i = tl.load(L + off_zh + offs_m)
        delta = tl.load(Delta + off_zh + offs_m)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]

        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        ds = p * (dp - delta[:, None]) * sm_scale

        dq_val = tl.dot(ds.to(q.dtype), k)

        dq_ptrs = DQ + qvk_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

        tl.atomic_add(dq_ptrs, dq_val.to(q.dtype))

        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))

    DK_block_ptr = tl.make_block_ptr(
        base=DK + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(DK_block_ptr, dk.to(k.dtype))
    tl.store(DV_block_ptr, dv.to(v.dtype))


class _TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, return_lse):
        # [batch, num_heads, seq_len, head_dim]
        # Triton tutorial standard layout requires specific contiguity
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk == Lv
        assert Lq in {16, 32, 64, 128, 256}

        ctx.sm_scale = sm_scale
        ctx.causal = causal

        batch, heads, seq_len, head_dim = q.shape

        out = torch.empty_like(q)
        M = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        L = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64 if head_dim > 64 else 128

        grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads, 1)

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            M,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            batch,
            heads,
            seq_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=head_dim,
            IS_CAUSAL=causal,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, out, L, M)

        if not return_lse:
            return out, None

        # Attention-domain LSE: log-sum-exp of the (scaled, masked) QK^T logits per
        # query row, in the same fixed reduction order the fwd kernel already used to
        # accumulate M (running max) and L (running sum-exp) online. This is distinct
        # from the vocab-domain LSE produced by the logp/linear_logp kernels.
        lse = M + torch.log(L)
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, do, _dlse):
        q, k, v, out, L, M = ctx.saved_tensors

        do = do.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        batch, heads, seq_len, head_dim = q.shape
        delta = torch.empty_like(L)

        BLOCK_M = 64
        BLOCK_N = 64

        grid_prep = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
        _bwd_preprocess[grid_prep](
            out,
            do,
            delta,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            batch,
            heads,
            seq_len,
            BLOCK_M=BLOCK_M,
            D_HEAD=head_dim,
        )

        grid_bwd = (triton.cdiv(seq_len, BLOCK_N), batch * heads, 1)
        _bwd_kernel[grid_bwd](
            q,
            k,
            v,
            ctx.sm_scale,
            out,
            do,
            dq,
            dk,
            dv,
            L,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            batch,
            heads,
            seq_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=head_dim,
            IS_CAUSAL=ctx.causal,
            num_warps=4,
            num_stages=1,
        )

        return dq, dk, dv, None, None, None


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
    return_lse: bool = False,
):
    """
    Universal backup Triton FlashAttention (support Forward / Backward)

    Args:
        q, k, v: Tensors of shape [batch, num_heads, seq_len, head_dim].
            It is recommended to switch to contiguous memory (.contiguous()).
        causal: Whether to turn on causal masking.
        sm_scale: Softmax scaling factor, default to 1.0 / sqrt(head_dim).
        return_lse: If True, also return the per-query-row attention-domain LSE
            (log-sum-exp of the scaled, masked QK^T logits), shape
            [batch, num_heads, seq_len], float32. Not a vocab-domain logprob LSE.
            The returned LSE is non-differentiable (diagnostics / backward-recompute
            use only), matching the external contract of `flash_attn`'s `softmax_lse`.
    Returns:
        `out` of shape [batch, num_heads, seq_len, head_dim] if `return_lse` is False,
        else `(out, lse)`.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

    out, lse = _TritonAttention.apply(q, k, v, causal, sm_scale, return_lse)
    return (out, lse) if return_lse else out


# ---------------------------------------------------------------------------
# Variable-length (packed) attention.
#
# Q/K/V are packed along the token dimension: [total_tokens, H, D], with no
# per-sequence padding. `cu_seqlens_{q,k}` are int32 [batch + 1] cumulative
# sequence-length offsets (cu_seqlens[0] == 0), the same convention as
# `flash_attn_varlen_func` and this repo's `pack` op (#182). Each program
# handles one (query-block, batch, head) triple; `causal_offset = seqlen_k -
# seqlen_q` reproduces the `Skv - Sq` causal anchor from the WS1
# `NativeAttentionOp` reference (docs/operators/attention.md) so prefill
# (Sq == Skv) and decode (Sq < Skv) share one formula. Boundary handling is
# via explicit masks (not `boundary_check`) because a block that runs past a
# sequence's length would otherwise read into the *next* packed sequence.
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_kernel_varlen(
    Q,
    K,
    V,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    L,
    M,
    Out,
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,
    stride_oh,
    stride_ok,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_start = tl.load(cu_seqlens_q + b)
    q_end = tl.load(cu_seqlens_q + b + 1)
    seqlen_q = q_end - q_start
    if start_m * BLOCK_M >= seqlen_q:
        return

    k_start = tl.load(cu_seqlens_k + b)
    k_end = tl.load(cu_seqlens_k + b + 1)
    seqlen_k = k_end - k_start
    causal_offset = seqlen_k - seqlen_q

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    valid_m = offs_m < seqlen_q

    q_ptrs = (
        Q + (q_start + offs_m[:, None]) * stride_qm + h * stride_qh + offs_d[None, :] * stride_qk
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum(seqlen_k, (start_m + 1) * BLOCK_M + causal_offset)
    else:
        hi = seqlen_k

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offs_n < seqlen_k

        k_ptrs = (
            K
            + (k_start + offs_n[:, None]) * stride_kn
            + h * stride_kh
            + offs_d[None, :] * stride_kk
        )
        k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        mask = valid_n[None, :]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= (offs_n[None, :] - causal_offset))
        qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(beta, 1)

        p_scale = beta / l_i_new[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        v_ptrs = (
            V
            + (k_start + offs_n[:, None]) * stride_vn
            + h * stride_vh
            + offs_d[None, :] * stride_vk
        )
        v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)
        p = p_scale.to(v.dtype)
        acc += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

    acc = acc.to(Out.dtype.element_ty)
    o_ptrs = (
        Out + (q_start + offs_m[:, None]) * stride_om + h * stride_oh + offs_d[None, :] * stride_ok
    )
    tl.store(o_ptrs, acc, mask=valid_m[:, None])

    l_ptrs = L + (q_start + offs_m) * H + h
    m_ptrs = M + (q_start + offs_m) * H + h
    tl.store(l_ptrs, l_i, mask=valid_m)
    tl.store(m_ptrs, m_i, mask=valid_m)


@triton.jit
def _bwd_preprocess_varlen(
    Out,
    DO,
    Delta,
    stride_om,
    stride_oh,
    stride_ok,
    stride_dom,
    stride_doh,
    stride_dok,
    total_q,
    H,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    valid_m = offs_m < total_q

    o_ptrs = Out + offs_m[:, None] * stride_om + h * stride_oh + offs_d[None, :] * stride_ok
    do_ptrs = DO + offs_m[:, None] * stride_dom + h * stride_doh + offs_d[None, :] * stride_dok

    o = tl.load(o_ptrs, mask=valid_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=valid_m[:, None], other=0.0).to(o.dtype)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + offs_m * H + h, delta, mask=valid_m)


@triton.jit
def _bwd_kernel_varlen(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    L,
    M,
    Delta,
    cu_seqlens_q,
    cu_seqlens_k,
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_dom,
    stride_doh,
    stride_dok,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    k_start = tl.load(cu_seqlens_k + b)
    k_end = tl.load(cu_seqlens_k + b + 1)
    seqlen_k = k_end - k_start
    if start_n * BLOCK_N >= seqlen_k:
        return

    q_start = tl.load(cu_seqlens_q + b)
    q_end = tl.load(cu_seqlens_q + b + 1)
    seqlen_q = q_end - q_start
    causal_offset = seqlen_k - seqlen_q

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    valid_n = offs_n < seqlen_k

    k_ptrs = (
        K + (k_start + offs_n[:, None]) * stride_kn + h * stride_kh + offs_d[None, :] * stride_kk
    )
    v_ptrs = (
        V + (k_start + offs_n[:, None]) * stride_vn + h * stride_vh + offs_d[None, :] * stride_vk
    )
    k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # Causal: query rows before `offs_n - causal_offset` never attend to this K/V
    # block, so the M-loop can start there instead of at row 0.
    lo = tl.maximum(0, start_n * BLOCK_N - causal_offset) if IS_CAUSAL else 0
    lo = (lo // BLOCK_M) * BLOCK_M

    for start_m in range(lo, seqlen_q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        valid_m = offs_m < seqlen_q

        q_ptrs = (
            Q
            + (q_start + offs_m[:, None]) * stride_qm
            + h * stride_qh
            + offs_d[None, :] * stride_qk
        )
        do_ptrs = (
            DO
            + (q_start + offs_m[:, None]) * stride_dom
            + h * stride_doh
            + offs_d[None, :] * stride_dok
        )
        q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=valid_m[:, None], other=0.0)

        m_i = tl.load(M + (q_start + offs_m) * H + h, mask=valid_m, other=0.0)
        l_i = tl.load(L + (q_start + offs_m) * H + h, mask=valid_m, other=1.0)
        delta = tl.load(Delta + (q_start + offs_m) * H + h, mask=valid_m, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        mask = valid_n[None, :] & valid_m[:, None]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= (offs_n[None, :] - causal_offset))
        qk = tl.where(mask, qk, float("-inf"))

        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
        p = tl.where(mask, p, 0.0)

        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(mask, ds, 0.0)

        dq_val = tl.dot(ds.to(q.dtype), k)
        dq_ptrs = (
            DQ
            + (q_start + offs_m[:, None]) * stride_qm
            + h * stride_qh
            + offs_d[None, :] * stride_qk
        )
        tl.atomic_add(dq_ptrs, dq_val.to(q.dtype), mask=valid_m[:, None])

        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk_ptrs = (
        DK + (k_start + offs_n[:, None]) * stride_kn + h * stride_kh + offs_d[None, :] * stride_kk
    )
    dv_ptrs = (
        DV + (k_start + offs_n[:, None]) * stride_vn + h * stride_vh + offs_d[None, :] * stride_vk
    )
    tl.store(dk_ptrs, dk.to(k.dtype), mask=valid_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=valid_n[:, None])


class _TritonAttentionVarlen(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        sm_scale,
        return_lse,
    ):
        total_q, H, head_dim = q.shape
        assert k.shape[1] == H and v.shape[1] == H, "GQA is not supported by the varlen path yet"
        assert head_dim in {16, 32, 64, 128, 256}

        batch = cu_seqlens_q.numel() - 1
        assert cu_seqlens_k.numel() - 1 == batch

        cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32)
        cu_seqlens_k = cu_seqlens_k.to(device=q.device, dtype=torch.int32)

        out = torch.empty_like(q)
        # Packed [total_q, H] layout (not [batch, H, seq_len]): total_q varies per
        # batch, so there is no fixed per-sequence stride to lay these out densely.
        M = torch.empty((total_q, H), device=q.device, dtype=torch.float32)
        L = torch.empty((total_q, H), device=q.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64 if head_dim > 64 else 128

        grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch * H)
        _fwd_kernel_varlen[grid](
            q,
            k,
            v,
            sm_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            L,
            M,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            H,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=head_dim,
            IS_CAUSAL=causal,
            num_warps=4,
            num_stages=2,
        )

        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.total_q = total_q
        ctx.save_for_backward(q, k, v, out, L, M, cu_seqlens_q, cu_seqlens_k)

        if not return_lse:
            return out, None

        lse = M + torch.log(L)
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, do, _dlse):
        q, k, v, out, L, M, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        do = do.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        total_q, H, head_dim = q.shape
        delta = torch.empty_like(L)

        BLOCK_M = 64
        BLOCK_N = 64

        grid_prep = (triton.cdiv(total_q, BLOCK_M), H)
        _bwd_preprocess_varlen[grid_prep](
            out,
            do,
            delta,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            total_q,
            H,
            BLOCK_M=BLOCK_M,
            D_HEAD=head_dim,
        )

        batch = cu_seqlens_q.numel() - 1
        grid_bwd = (triton.cdiv(ctx.max_seqlen_k, BLOCK_N), batch * H)
        _bwd_kernel_varlen[grid_bwd](
            q,
            k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,
            L,
            M,
            delta,
            cu_seqlens_q,
            cu_seqlens_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            H,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=head_dim,
            IS_CAUSAL=ctx.causal,
            num_warps=4,
            num_stages=1,
        )

        return dq, dk, dv, None, None, None, None, None, None, None


def triton_flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = True,
    sm_scale: float | None = None,
    return_lse: bool = False,
):
    """
    Packed variable-length FlashAttention (Triton), the cross-platform semantic
    baseline for RL rollout/training batches where sequences are concatenated
    rather than padded.

    Args:
        q: [total_q, H, D] packed queries.
        k, v: [total_k, H, D] packed keys/values. GQA (Hk != Hq) is not supported
            by this path yet (matches the existing dense Triton kernel's limitation).
        cu_seqlens_q, cu_seqlens_k: int32 [batch + 1] cumulative sequence-length
            offsets, cu_seqlens[0] == 0 (the `flash_attn_varlen_func` convention;
            also what `pack`, #182, produces).
        max_seqlen_q, max_seqlen_k: max per-sequence length in the batch (host
            ints), used to size the launch grid.
        causal: causal masking, anchored per-sequence via `Skv - Sq` (same
            convention as `NativeAttentionOp` in docs/operators/attention.md, so
            Sq == Skv is prefill and Sq < Skv is decode-with-cache).
        sm_scale: defaults to `1/sqrt(D)`.
        return_lse: if True, also return the packed `[total_q, H]` float32
            attention-domain LSE (non-differentiable; see `triton_flash_attention`).
    Returns:
        `out` of shape [total_q, H, D] if `return_lse` is False, else `(out, lse)`.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

    out, lse = _TritonAttentionVarlen.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        sm_scale,
        return_lse,
    )
    return (out, lse) if return_lse else out
