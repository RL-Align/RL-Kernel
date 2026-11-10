# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton [F1] kernels: post-norm, residual add, pre-norm, and the final norm.

Mirrors ``rl_engine.reference.norm.fused_add_norm``. The frozen semantics and the
two narrowing points are documented there; this file implements them and should
not be read as the contract.

One program per row, one block per row, so the reduction order is fixed by
``BLOCK_H`` and never depends on the batch layout. That is what makes Axis-A batch
invariance hold here.

RFC #501 asks for CUDA == Triton bitwise, and this operator meets it on all six
outputs at all three dtypes, on the CUDA build. The CUDA kernel reaches it with one
line of NVIDIA PTX, which a ROCm build cannot assemble and falls back from, so the
claim is scoped to CUDA; on ROCm the registry dispatches this Triton kernel and the
PyTorch gold and never the hipified CUDA one. A row reduction does not get there by writing the
same source in both languages; three specific things had to be made to agree, and
each is load-bearing here rather than incidental:

* **the reduction order.** ``tl.sum`` picks its own tree, so the fold is written
  out instead: lane ``t`` accumulates ``x[t + k*LANES]`` over ``k`` ascending,
  then a stride-halving tree over the lanes. That is the order
  ``csrc/cuda/norm/fused_add_norm.cu`` already used, and ``_reduction_shape``
  below owns the lane counts jointly with ``choose_threads`` there. Change one
  without the other and the backends come apart.
* **the divide by the width.** Triton's ``/`` lowers to ``div.full.f32``, an
  approximate division; the CUDA kernel's is ``div.rn.f32``. With the orders
  already matched, that difference alone left the statistics one ULP apart on 22%
  of rows. Hence ``tl.div_rn``.
* **the reciprocal square root.** nvcc's ``rsqrtf`` is libdevice's refined
  non-FTZ sequence, ``tl.rsqrt`` is the bare ``rsqrt.approx.ftz.f32``, and they
  disagree on about 12% of inputs. The CUDA side emits the same instruction now.

``tests/ops/norm/test_fused_add_norm_cuda.py::test_cuda_and_triton_are_bitwise_equal``
holds all three. ``tests/ops/norm/test_rms_norm.py`` still compares its backends
with a tolerance, "NOT torch.equal, precisely because the reduction order is
allowed to differ", which is the older position; this row shows the order can be
pinned instead, at no measurable cost.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32
from rl_engine.reference.norm.fused_add_norm import POST_NORM_EPS, RMS_NORM_EPS

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only builds
    _TRITON_AVAILABLE = False

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# The ceiling is on the number of statically unrolled chunks, not on a tile size.
# The kernels took a BLOCK_H tile before the reduction order was pinned and the
# message still described one, while the computed value had stopped being used by
# either caller. 128 chunks at 512 lanes is the same effective width ceiling the
# BLOCK_H form had, so nothing that compiled before stops compiling.
#
# The CUDA kernel has no matching ceiling: it loops rather than unrolling, so a
# width between these bounds is accepted there and rejected here. That divergence
# is real and is not something this row should settle by adding a limit to the .cu;
# it is recorded in the operator doc's Known Limitations.
_MAX_CHUNKS = 128


def _reduction_shape(width: int) -> tuple[int, int, int]:
    """Lane count, chunk count and tree depth, matching the CUDA kernel exactly.

    RFC #501 asks for CUDA == Triton bitwise. For a row reduction that is only
    reachable if both sides execute the same sequence of adds, so the order is
    fixed here rather than left to ``tl.sum``, and it is fixed to the order
    ``csrc/cuda/norm/fused_add_norm.cu`` already uses: lane ``t`` folds
    ``x[t + k*LANES]`` over ``k`` ascending, then a stride-halving tree over the
    lanes. ``choose_threads`` in that file is the source of the lane counts, so
    the two must be changed together.

    Lanes beyond the width load as 0.0 and a chunk past the end contributes an
    exact ``+ 0.0``, which is why a width that is not a multiple of the lane count
    still matches: the CUDA loop simply stops earlier for those threads.
    """
    lanes = 64 if width <= 64 else 128 if width <= 128 else 256 if width <= 256 else 512
    chunks = -(-width // lanes)
    return lanes, chunks, lanes.bit_length() - 1


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_add_norm_fwd_kernel(
        HIDDEN,
        RESIDUAL,
        POST_W,
        PRE_W,
        RES_OUT,
        PRE_OUT,
        RSTD_POST,
        RSTD_PRE,
        H: tl.constexpr,
        POST_EPS: tl.constexpr,
        PRE_EPS: tl.constexpr,
        LANES: tl.constexpr,
        CHUNKS: tl.constexpr,
        LEVELS: tl.constexpr,
    ):
        row = tl.program_id(0)
        # int64 before the multiply, not after. tl.program_id is int32 and H is a
        # constexpr int, so ``row * H`` is a 32-bit multiply that wraps at 2**31 and
        # the generated PTX widens the already-wrapped product. At H=6656 that is
        # row 322,639, about 17 GB of bf16 activations, reachable on one 80 GB card
        # in a long-context or large-batch rollout, and it faults rather than
        # returning wrong numbers.
        base = row.to(tl.int64) * H
        lanes = tl.arange(0, LANES)
        narrow = RES_OUT.dtype.element_ty

        # Post-norm statistic, in the order _reduction_shape documents: every add
        # is written out so neither compiler picks a tree of its own.
        acc = tl.zeros((LANES,), dtype=tl.float32)
        for k in tl.static_range(CHUNKS):
            idx = k * LANES + lanes
            v = tl.load(HIDDEN + base + idx, mask=idx < H, other=0.0).to(tl.float32)
            acc = acc + v * v
        for _ in tl.static_range(LEVELS):
            acc = tl.sum(tl.reshape(acc, (2, acc.shape[0] // 2)), axis=0)
        # tl.div_rn, not ``/``: Triton's divide lowers to div.full.f32, an
        # approximate division about two ULPs wide, while the CUDA kernel's
        # ``/ (float)H`` is div.rn.f32. That alone put the two statistics one ULP
        # apart on 22% of rows after the reduction orders already matched.
        rstd_post = tl.rsqrt(tl.div_rn(tl.sum(acc, axis=0), H) + POST_EPS)

        # Post-norm, both narrowings, and the pre-norm statistic in one pass. The
        # statistic accumulates the narrowed value as it is produced, in the same
        # ascending-chunk order, so the second reduction needs no re-read and no
        # barrier against the store below.
        acc2 = tl.zeros((LANES,), dtype=tl.float32)
        for k in tl.static_range(CHUNKS):
            idx = k * LANES + lanes
            m = idx < H
            hv = tl.load(HIDDEN + base + idx, mask=m, other=0.0).to(tl.float32)
            rv = tl.load(RESIDUAL + base + idx, mask=m, other=0.0).to(tl.float32)
            pw = tl.load(POST_W + idx, mask=m, other=0.0).to(tl.float32)
            # First narrowing: the post-norm's cast is its last step.
            post = hv * rstd_post * (1.0 + pw)
            res = rv + post.to(narrow).to(tl.float32)
            # Second narrowing: the pre-norm reads the rounded sum, as eager does.
            res_narrow = res.to(narrow)
            rn = res_narrow.to(tl.float32)
            acc2 = acc2 + tl.where(m, rn * rn, 0.0)
            tl.store(RES_OUT + base + idx, res_narrow, mask=m)
        for _ in tl.static_range(LEVELS):
            acc2 = tl.sum(tl.reshape(acc2, (2, acc2.shape[0] // 2)), axis=0)
        rstd_pre = tl.rsqrt(tl.div_rn(tl.sum(acc2, axis=0), H) + PRE_EPS)

        # The pre-norm reads the narrowed residual back from memory, as the CUDA
        # kernel does. Lane ``t`` reads the address lane ``t`` wrote above, so the
        # dependency is within a thread; the barrier states it rather than relying
        # on the compiler not reordering a load across an aliasing store.
        tl.debug_barrier()
        for k in tl.static_range(CHUNKS):
            idx = k * LANES + lanes
            m = idx < H
            rv = tl.load(RES_OUT + base + idx, mask=m, other=0.0).to(tl.float32)
            vw = tl.load(PRE_W + idx, mask=m, other=0.0).to(tl.float32)
            pre = rv * rstd_pre * (1.0 + vw)
            tl.store(PRE_OUT + base + idx, pre.to(PRE_OUT.dtype.element_ty), mask=m)

        tl.store(RSTD_POST + row, rstd_post)
        tl.store(RSTD_PRE + row, rstd_pre)

    @triton.jit
    def _fused_add_norm_bwd_kernel(
        G_RES,
        G_PRE,
        HIDDEN,
        RES_OUT,
        POST_W,
        PRE_W,
        RSTD_POST,
        RSTD_PRE,
        D_HIDDEN,
        D_RESIDUAL,
        PARTIAL_POST_W,
        PARTIAL_PRE_W,
        H: tl.constexpr,
        LANES: tl.constexpr,
        CHUNKS: tl.constexpr,
        LEVELS: tl.constexpr,
    ):
        row = tl.program_id(0)
        # int64 before the multiply; see the forward kernel.
        base = row.to(tl.int64) * H
        lanes = tl.arange(0, LANES)
        rstd_post = tl.load(RSTD_POST + row).to(tl.float32)
        rstd_pre = tl.load(RSTD_PRE + row).to(tl.float32)
        # From hidden, not from the cotangent. The reference derives it from the
        # input (dtype = hidden.dtype) and the CUDA kernel from the dispatch on
        # hidden.scalar_type(), so taking it from G_RES made an FP32 cotangent with
        # BF16 hidden turn both narrowings into the identity and silently return
        # the gradient of forward_fp32, a different computation. Autograd cannot
        # reach that -- cotangents arrive at the output dtype -- but the public
        # fused_add_norm_triton_backward can.
        narrow = HIDDEN.dtype.element_ty

        # Pre-norm dot product, in the order _reduction_shape documents.
        acc = tl.zeros((LANES,), dtype=tl.float32)
        for k in tl.static_range(CHUNKS):
            idx = k * LANES + lanes
            m = idx < H
            gp = tl.load(G_PRE + base + idx, mask=m, other=0.0).to(tl.float32)
            gain = 1.0 + tl.load(PRE_W + idx, mask=m, other=0.0).to(tl.float32)
            ro = tl.load(RES_OUT + base + idx, mask=m, other=0.0).to(tl.float32)
            acc = acc + tl.where(m, (gp * gain) * ro, 0.0)
        for _ in tl.static_range(LEVELS):
            acc = tl.sum(tl.reshape(acc, (2, acc.shape[0] // 2)), axis=0)
        dot_pre = tl.div_rn(tl.sum(acc, axis=0), H)

        # The residual output's two cotangents, accumulated the way eager does: it
        # is a model-dtype tensor, so autograd rounds each arrival to that dtype
        # and adds in that dtype. Keeping this in FP32 is one ULP per element and
        # puts the post-norm weight gradient 2.6-2.9x outside the contract row at
        # 512 and 2048 rows; see the reference's docstring. The post-norm dot
        # product accumulates in the same pass, in the same ascending-chunk order.
        acc2 = tl.zeros((LANES,), dtype=tl.float32)
        for k in tl.static_range(CHUNKS):
            idx = k * LANES + lanes
            m = idx < H
            gp = tl.load(G_PRE + base + idx, mask=m, other=0.0).to(tl.float32)
            gain = 1.0 + tl.load(PRE_W + idx, mask=m, other=0.0).to(tl.float32)
            ro = tl.load(RES_OUT + base + idx, mask=m, other=0.0).to(tl.float32)
            gr = tl.load(G_RES + base + idx, mask=m, other=0.0).to(tl.float32)
            hv = tl.load(HIDDEN + base + idx, mask=m, other=0.0).to(tl.float32)
            pgain = 1.0 + tl.load(POST_W + idx, mask=m, other=0.0).to(tl.float32)

            scaled_pre = gp * gain
            d_from_pre = rstd_pre * scaled_pre - ro * rstd_pre * rstd_pre * rstd_pre * dot_pre
            d_ro = (gr + d_from_pre.to(narrow).to(tl.float32)).to(narrow).to(tl.float32)

            tl.store(D_RESIDUAL + base + idx, d_ro, mask=m)
            tl.store(PARTIAL_PRE_W + base + idx, gp * ro * rstd_pre, mask=m)
            acc2 = acc2 + tl.where(m, (d_ro * pgain) * hv, 0.0)
        for _ in tl.static_range(LEVELS):
            acc2 = tl.sum(tl.reshape(acc2, (2, acc2.shape[0] // 2)), axis=0)
        dot_post = tl.div_rn(tl.sum(acc2, axis=0), H)

        # The hidden gradient reads the accumulated residual gradient back, as the
        # CUDA kernel does. Lane ``t`` reads the address lane ``t`` wrote above.
        tl.debug_barrier()
        for k in tl.static_range(CHUNKS):
            idx = k * LANES + lanes
            m = idx < H
            d_ro = tl.load(D_RESIDUAL + base + idx, mask=m, other=0.0).to(tl.float32)
            hv = tl.load(HIDDEN + base + idx, mask=m, other=0.0).to(tl.float32)
            pgain = 1.0 + tl.load(POST_W + idx, mask=m, other=0.0).to(tl.float32)
            scaled_post = d_ro * pgain
            d_hidden = rstd_post * scaled_post - hv * rstd_post * rstd_post * rstd_post * dot_post
            tl.store(D_HIDDEN + base + idx, d_hidden, mask=m)
            tl.store(PARTIAL_POST_W + base + idx, d_ro * hv * rstd_post, mask=m)


def _require_triton() -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for the fused add-norm kernels")


def _validate(hidden: Tensor, residual: Tensor, post_w: Tensor, pre_w: Tensor) -> None:
    for name, t in (("hidden", hidden), ("residual", residual)):
        if t.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {t.dtype}.")
        if not t.is_cuda:
            raise RuntimeError(f"{name} must be a GPU tensor, got device '{t.device}'.")
    if hidden.shape != residual.shape:
        raise ValueError(
            f"hidden and residual must share shape, got {tuple(hidden.shape)} "
            f"and {tuple(residual.shape)}."
        )
    if hidden.dtype != residual.dtype:
        raise ValueError(
            f"hidden and residual must share dtype, got {hidden.dtype} and {residual.dtype}."
        )
    # Same device as hidden, the way the weights below are checked. The launch
    # takes its device from hidden and then dereferences residual's pointer there.
    # Measured on two H100s, and which of the two outcomes you get depends on the
    # process rather than on the call: Triton validates the pointer with
    # cuPointerGetAttribute, which fails only while peer access is off, and one
    # cross-device .to() copy anywhere earlier turns it on (allocating on both
    # devices does not; cudaDeviceEnablePeerAccess answers 0 after two allocations
    # and 704, already enabled, after a single copy). Off, Triton refuses the launch
    # with ValueError("Pointer argument (at 1) cannot be accessed from Triton (cpu
    # tensor?)"), which names the wrong cause and an argument index. On, the launch
    # runs, reads the other device through the peer mapping, and returns an answer
    # bitwise equal to the same-device one. A stray tensor usually arrives on the
    # wrong device by way of a copy, so the silent path is the likely one. The CUDA
    # binding refuses both in fused_add_norm_check_pair, and silently accepting a
    # cross-device input is the wrong failure mode for an operator whose contract is
    # multi-GPU byte-for-byte.
    if residual.device != hidden.device:
        raise RuntimeError(
            f"residual must be on the same device as hidden ({hidden.device}), "
            f"got {residual.device}."
        )
    # Rank before the width, the way the oracle checks it. A 0-D hidden cleared
    # the dtype, device and shape-equality checks above, since a 0-D residual
    # equals a 0-D hidden, and then died on hidden.shape[-1] with "IndexError:
    # tuple index out of range". The reference raises a ValueError naming the
    # argument, and a wrapper whose failure mode differs from its oracle's is the
    # one-sided fix this review has already seen six times.
    if hidden.dim() < 1:
        raise ValueError("hidden must have at least one dimension, got a 0-D tensor.")
    width = hidden.shape[-1]
    if width == 0:
        # Otherwise the first thing that fails is .view(-1, 0), whose message is
        # about an ambiguous unspecified dimension and says nothing about the real
        # problem. A zero-width row has no statistic to compute, so there is no
        # sensible answer to return either.
        raise ValueError("hidden must have a non-zero last dimension, got width 0.")
    for name, w in (("post_weight", post_w), ("pre_weight", pre_w)):
        if w.dim() != 1 or w.shape[0] != width:
            raise ValueError(
                f"{name} must be 1-D of size x.shape[-1]={width}, "
                f"got {name}.shape={tuple(w.shape)}"
            )
        # The weights were checked for rank and size only, while the CUDA binding
        # checks their dtype and requires the two to agree (csrc/bindings/ops.cpp).
        # So post_weight bf16 with pre_weight fp32 ran here and raised there: the
        # same call changing behaviour with the backend, which is worse than either
        # answer. Matching the stricter side.
        if w.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {w.dtype}.")
        if not w.is_cuda:
            raise RuntimeError(f"{name} must be a GPU tensor, got device '{w.device}'.")
        # Same device as hidden, not merely "some GPU". The launch takes its device
        # and stream from hidden, so a weight on another one is dereferenced as if
        # it lived there. The CUDA binding checks this; this side checked is_cuda
        # only, which is the fourth time in this review that I had fixed one of the
        # two backends and left the other.
        if w.device != hidden.device:
            raise RuntimeError(
                f"{name} must be on the same device as hidden ({hidden.device}), "
                f"got {w.device}."
            )
    if post_w.dtype != pre_w.dtype:
        raise ValueError(
            f"post_weight and pre_weight must share dtype, got {post_w.dtype} "
            f"and {pre_w.dtype}."
        )
    _, chunks, _ = _reduction_shape(width)
    if chunks > _MAX_CHUNKS:
        raise ValueError(
            f"hidden width {width} needs {chunks} statically unrolled chunks, above "
            f"the {_MAX_CHUNKS} limit; a row is folded by one program so the "
            "reduction order stays independent of the batch layout, and the unroll "
            "is what bounds the width."
        )


def _validate_cotangents(hidden: Tensor, g_res: Tensor, g_pre: Tensor) -> None:
    """The cotangent half of the backward's contract, as the CUDA binding states it.

    Split out from the body so the narrowing test can reach the kernel with a
    cotangent dtype the public entry point no longer accepts; that is the only
    caller that bypasses it.
    """
    for name, g in (("grad_residual_out", g_res), ("grad_pre_normed", g_pre)):
        # The CUDA binding makes both dtype checks and this side made neither, so
        # the kernel took whatever arrived and cast it to FP32 on load. Measured at
        # 4x64 against BF16 hidden: an FP32 cotangent and an FP64 one each raise in
        # the binding and each returned a full set of gradients here, which is the
        # same call changing its answer with the backend.
        if g.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {g.dtype}.")
        if g.dtype != hidden.dtype:
            raise TypeError(f"{name} must have hidden's dtype ({hidden.dtype}), got {g.dtype}.")
        if g.device != hidden.device:
            raise RuntimeError(
                f"{name} must be on the same device as hidden ({hidden.device}), "
                f"got {g.device}."
            )
        if g.shape != hidden.shape:
            raise ValueError(
                f"{name} must share hidden's shape, got {tuple(g.shape)} "
                f"and {tuple(hidden.shape)}."
            )


def fused_add_norm_triton_forward(
    hidden: Tensor,
    residual: Tensor,
    post_weight: Tensor,
    pre_weight: Tensor,
    *,
    post_eps: float,
    pre_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns ``(residual_out, pre_normed, rstd_post, rstd_pre)``.

    The two statistics come back so the backward does not recompute them and
    cannot drift from what the forward actually used.
    """
    _require_triton()
    _validate(hidden, residual, post_weight, pre_weight)
    width = hidden.shape[-1]
    lanes, chunks, levels = _reduction_shape(width)

    hidden_2d = hidden.contiguous().view(-1, width)
    residual_2d = residual.contiguous().view(-1, width)
    rows = hidden_2d.shape[0]

    res_out = torch.empty_like(hidden_2d)
    pre_out = torch.empty_like(hidden_2d)
    rstd_post = torch.empty(rows, device=hidden.device, dtype=torch.float32)
    rstd_pre = torch.empty(rows, device=hidden.device, dtype=torch.float32)

    if rows == 0:
        # Explicit, not because this version needs it. Measured on Triton 3.6.0:
        # kernel[(0,)] is a no-op and the context survives, and the zero-row test
        # passes. But that is an undocumented launcher detail, the H100 used for the
        # benchmark runs 3.3.1, and the CUDA launcher has to guard it because a
        # zero-block grid is a sticky cudaErrorInvalidConfiguration there. Two lines
        # make the behaviour the same on both backends and independent of either.
        return (
            res_out.view_as(hidden),
            pre_out.view_as(hidden),
            rstd_post,
            rstd_pre,
        )
    _fused_add_norm_fwd_kernel[(rows,)](
        hidden_2d,
        residual_2d,
        post_weight.contiguous(),
        pre_weight.contiguous(),
        res_out,
        pre_out,
        rstd_post,
        rstd_pre,
        H=width,
        POST_EPS=float(post_eps),
        PRE_EPS=float(pre_eps),
        LANES=lanes,
        CHUNKS=chunks,
        LEVELS=levels,
        enable_fp_fusion=False,
    )
    return (
        res_out.view_as(hidden),
        pre_out.view_as(hidden),
        rstd_post,
        rstd_pre,
    )


def fused_add_norm_triton_backward(
    grad_residual_out: Tensor,
    grad_pre_normed: Tensor,
    hidden: Tensor,
    residual_out: Tensor,
    post_weight: Tensor,
    pre_weight: Tensor,
    rstd_post: Tensor,
    rstd_pre: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns ``(d_hidden, d_residual, partial_post_w, partial_pre_w)`` in FP32.

    The two weight entries are per-row contributions, matching the convention in
    ``norm/rmsnorm.py``; the caller folds them with ``reduce_rows_fp32`` so the
    declared reduction order over rows is preserved.
    """
    _require_triton()
    # This recomputed ``block`` inline and skipped _validate entirely, so the
    # public backward had no dtype, device, shape or _MAX_BLOCK check while the
    # CUDA path validates in C++. Mismatched cotangent shapes read and wrote the
    # wrong rows instead of raising.
    _validate(hidden, residual_out, post_weight, pre_weight)
    _validate_cotangents(hidden, grad_residual_out, grad_pre_normed)
    width = hidden.shape[-1]
    lanes, chunks, levels = _reduction_shape(width)
    rows_expected = hidden.numel() // width
    for name, r in (("rstd_post", rstd_post), ("rstd_pre", rstd_pre)):
        # The same four checks the CUDA binding makes. Rank and length alone were
        # not enough: the kernels read these with tl.load(RSTD_POST + row), an
        # implicit stride of one, so a strided view such as rstd[::2] ran and
        # produced different gradients. The CUDA side rejects it and this side did
        # not, which is the backend divergence the weight checks above exist to
        # avoid, one argument over.
        if not r.is_cuda:
            raise RuntimeError(f"{name} must be a GPU tensor, got device '{r.device}'.")
        if r.device != hidden.device:
            raise RuntimeError(
                f"{name} must be on the same device as hidden ({hidden.device}), "
                f"got {r.device}."
            )
        if not r.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
        if r.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {r.dtype}.")
        if r.dim() != 1 or r.shape[0] != rows_expected:
            raise ValueError(
                f"{name} must be 1-D with one entry per row ({rows_expected}), "
                f"got {name}.shape={tuple(r.shape)}"
            )

    hidden_2d = hidden.contiguous().view(-1, width)
    res_out_2d = residual_out.contiguous().view(-1, width)
    g_res_2d = grad_residual_out.contiguous().view(-1, width)
    g_pre_2d = grad_pre_normed.contiguous().view(-1, width)
    rows = hidden_2d.shape[0]

    d_hidden = torch.empty(rows, width, device=hidden.device, dtype=torch.float32)
    d_residual = torch.empty_like(d_hidden)
    partial_post = torch.empty_like(d_hidden)
    partial_pre = torch.empty_like(d_hidden)

    if rows == 0:
        return d_hidden, d_residual, partial_post, partial_pre  # see the forward
    _fused_add_norm_bwd_kernel[(rows,)](
        g_res_2d,
        g_pre_2d,
        hidden_2d,
        res_out_2d,
        post_weight.contiguous(),
        pre_weight.contiguous(),
        rstd_post,
        rstd_pre,
        d_hidden,
        d_residual,
        partial_post,
        partial_pre,
        H=width,
        LANES=lanes,
        CHUNKS=chunks,
        LEVELS=levels,
        enable_fp_fusion=False,
    )
    return d_hidden, d_residual, partial_post, partial_pre


class _FusedAddNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, residual, post_weight, pre_weight, post_eps, pre_eps):
        res_out, pre_out, rstd_post, rstd_pre = fused_add_norm_triton_forward(
            hidden,
            residual,
            post_weight,
            pre_weight,
            post_eps=post_eps,
            pre_eps=pre_eps,
        )
        ctx.save_for_backward(hidden, res_out, post_weight, pre_weight, rstd_post, rstd_pre)
        return res_out, pre_out

    @staticmethod
    def backward(ctx, grad_residual_out, grad_pre_normed):
        # Both cotangents are always tensors, never None. This Function does not
        # call ctx.set_materialize_grads(False), so autograd materialises zeros for
        # an output the caller did not consume. Measured: pre_normed.sum().backward()
        # succeeds on both backends and fills every input gradient. The None guard
        # in logprob/linear_logp.py exists because that Function does opt out, at
        # its line 1362. If this one ever opts out for the allocation, the two
        # .contiguous() calls below become the crash site.
        hidden, res_out, post_w, pre_w, rstd_post, rstd_pre = ctx.saved_tensors
        d_hidden, d_residual, partial_post, partial_pre = fused_add_norm_triton_backward(
            grad_residual_out,
            grad_pre_normed,
            hidden,
            res_out,
            post_w,
            pre_w,
            rstd_post,
            rstd_pre,
        )
        record_backward(
            "fused_post_norm_residual_pre_norm",
            kernel_id=(
                "rl_engine.backends.shared.triton.norm.fused_add_norm"
                "._fused_add_norm_bwd_kernel"
                "+rl_engine.ops.autograd.vjp_fp32.reduce_rows_fp32"
            ),
            impl="triton_fused_add_norm_declared_fp32_rowfold_dw",
            family="triton",
        )
        width = hidden.shape[-1]
        d_post_w = reduce_rows_fp32(partial_post.view(-1, width)).to(post_w.dtype)
        d_pre_w = reduce_rows_fp32(partial_pre.view(-1, width)).to(pre_w.dtype)
        return (
            d_hidden.view_as(hidden).to(hidden.dtype),
            d_residual.view_as(hidden).to(hidden.dtype),
            d_post_w,
            d_pre_w,
            None,
            None,
        )


class TritonFusedAddNormOp:
    """Triton [F1] wrapper shaped like the other operator-harness ops."""

    backward_impl = "triton_fused_add_norm_declared_fp32_rowfold_dw"

    def __init__(self) -> None:
        # Same gate as the CUDA class, for the same reason and before the same
        # finding arrives one backend over. The registry treats an exception from
        # op_class() as "not available" and tries the next candidate, so without
        # this a host with no Triton selected this backend and failed later inside
        # forward rather than falling through to the PyTorch reference.
        _require_triton()

    def __call__(
        self,
        hidden,
        residual,
        post_weight,
        pre_weight,
        *,
        post_eps=POST_NORM_EPS,
        pre_eps=RMS_NORM_EPS,
    ):
        return self.forward(
            hidden, residual, post_weight, pre_weight, post_eps=post_eps, pre_eps=pre_eps
        )

    def forward(
        self,
        hidden,
        residual,
        post_weight,
        pre_weight,
        *,
        post_eps=POST_NORM_EPS,
        pre_eps=RMS_NORM_EPS,
    ):
        return _FusedAddNormTriton.apply(
            hidden, residual, post_weight, pre_weight, post_eps, pre_eps
        )
