# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse-Glimmer-30B [F1] norm references: post-norm, residual add, pre-norm.

RFC #501 work item ``fused_post_norm_residual_pre_norm``. Every rule below comes
from the ``frozen_semantics`` block of the #558 architecture fingerprint rather
than from the HF config, because the config does not state most of it.

A decoder layer runs the order the fingerprint pins as ``norm_residual_order``::

    input_layernorm -> self_attn -> post_attention_layernorm -> residual_add
    -> pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> residual_add

so the operator fused here is the tail of each half-layer: take a sublayer output,
post-norm it, add it to the residual stream, and pre-norm the result for whatever
comes next. Two outputs, because both are live: the updated residual continues the
chain and the pre-normed tensor feeds the next sublayer.

``decoder_norm`` (all four per-layer norms)
    form ``centered_rmsnorm_one_plus_weight``: ``x * rstd * (1 + w)``. The ``1 +``
    is a kernel parameter, not something a caller folds into the weight, because
    the weights are BF16 and folding rounds the offset away.
    compute dtype FP32, ``rsqrt_form`` ``rsqrt``, ``weight_applied_before_cast``
    true, so the multiply happens in FP32 and the cast is the last step.
    eps is per site: ``rms_norm_eps`` for ``input_layernorm`` and
    ``pre_feedforward_layernorm``, ``post_norm_eps`` for the two post-norms.

``final_norm``
    form ``rmsnorm_times_weight``: ``x * rstd * w``, no offset, and
    ``rsqrt_form`` ``pow_neg_half`` rather than ``rsqrt``. Kept as a separate entry
    point for exactly that reason. The two spellings are bitwise equal for FP32 on
    SM 8.9, measured over 4096 rows at hidden 6656 for both eps values, but the
    fingerprint distinguishes them so this code does too.

Both forms share ``shape_invariant_meansq`` for the reduction. That is what makes
Axis-A batch invariance hold: the statistic is summed in fixed 32-wide chunks so no
intermediate shape depends on the batch layout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rl_engine.reference.norm.rms_norm import shape_invariant_meansq, shape_invariant_rowsum

# Muse-Glimmer-30B text config; see the #558 manifest.
HIDDEN_SIZE = 6656
RMS_NORM_EPS = 1e-5
POST_NORM_EPS = 1e-8

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_row(x: Tensor, name: str) -> None:
    # Rank before dtype: a 0-D tensor passed both checks here and the shape
    # comparison, then died on width = hidden.shape[-1] with "IndexError: tuple
    # index out of range", which says nothing about the argument. One dimension is
    # enough, a single row being a shape the oracle does accept.
    if x.dim() < 1:
        raise ValueError(f"{name} must have at least one dimension, got a 0-D tensor.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {x.dtype}.")


def _validate_weight(weight: Tensor, hidden: int, name: str) -> None:
    # The rows are held to this dtype set and the weights were not, while both
    # kernels reject anything outside it in fused_add_norm_check_pair and _validate.
    # Measured: an int64 or bool weight of ones was widened by weight.float() and
    # the oracle returned a full set of plausible values, so the reference answered
    # where the implementations it defines would have raised.
    if weight.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {weight.dtype}.")
    if weight.dim() != 1 or weight.shape[0] != hidden:
        raise ValueError(
            f"{name} must be 1-D of size x.shape[-1]={hidden}, "
            f"got {name}.shape={tuple(weight.shape)}"
        )


def _validate_backward(
    rows: tuple[tuple[Tensor, str], ...],
    weights: tuple[tuple[Tensor, str], ...],
    cotangents: tuple[tuple[Tensor, str], ...] = (),
    cotangent_dtype: torch.dtype | None = None,
    cotangent_hint: str = "",
) -> None:
    """The same bar the forward applies, on the closed-form backward.

    The forward validated and the backward did not, and the backward is the oracle
    the kernels are measured against, so a silently wrong answer here is worse than
    an exception anywhere else. Measured: a ``post_weight`` of length 1 broadcast
    through ``1.0 + post_weight.float()`` and the call returned four tensors of the
    right shapes carrying gradients for the wrong per-column gain, with nothing
    raised, while the forward rejected the same input.

    ``rows`` are inputs and share one dtype, which is what the forward requires of
    them. ``cotangents`` are checked for shape against the same reference and for
    dtype only when ``cotangent_dtype`` is given, because which dtype is correct
    there depends on which forward is being differentiated; the callers say.
    """
    reference, reference_name = rows[0]
    for tensor, name in rows:
        _validate_row(tensor, name)
        if tensor.shape != reference.shape:
            raise ValueError(
                f"{name} must share {reference_name}'s shape, got "
                f"{tuple(tensor.shape)} and {tuple(reference.shape)}."
            )
        if tensor.dtype != reference.dtype:
            raise ValueError(
                f"{name} must share {reference_name}'s dtype, got {tensor.dtype} "
                f"and {reference.dtype}."
            )
    for tensor, name in cotangents:
        _validate_row(tensor, name)
        if tensor.shape != reference.shape:
            raise ValueError(
                f"{name} must share {reference_name}'s shape, got "
                f"{tuple(tensor.shape)} and {tuple(reference.shape)}."
            )
        if cotangent_dtype is not None and tensor.dtype != cotangent_dtype:
            raise ValueError(
                f"{name} must have the dtype of the output it is a cotangent of "
                f"({cotangent_dtype}), got {tensor.dtype}.{cotangent_hint}"
            )
    for tensor, name in weights:
        _validate_weight(tensor, reference.shape[-1], name)


def _centered_norm(x_f: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``x * rstd * (1 + w)`` in FP32. Decoder-norm form, ``rsqrt`` spelling."""
    rstd = torch.rsqrt(shape_invariant_meansq(x_f) + float(eps)).unsqueeze(-1)
    return (x_f * rstd) * (1.0 + weight.float())


def _scaled_norm(x_f: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``x * rstd * w`` in FP32. Final-norm form, ``pow_neg_half`` spelling."""
    rstd = ((shape_invariant_meansq(x_f) + float(eps)) ** -0.5).unsqueeze(-1)
    return (x_f * rstd) * weight.float()


class NativeFusedAddNormOp(nn.Module):
    """Post-norm, residual add, pre-norm, as one operator.

    Returns ``(residual_out, pre_normed)``. ``residual_out`` is the updated
    residual stream and ``pre_normed`` is what the next sublayer consumes.

    Two narrowing points, both matching eager HF rather than being incidental.
    ``weight_applied_before_cast`` puts the post-norm's cast at its last step, so
    the value entering the add is already narrow. The sum is then narrowed again
    before the pre-norm reads it, because eager adds in the model dtype and the
    next norm sees that rounded value. ``forward_fp32`` has neither narrowing,
    which is why it is a different computation rather than a wider one.
    """

    def forward(
        self,
        hidden: Tensor,
        residual: Tensor,
        post_weight: Tensor,
        pre_weight: Tensor,
        *,
        post_eps: float = POST_NORM_EPS,
        pre_eps: float = RMS_NORM_EPS,
    ) -> tuple[Tensor, Tensor]:
        return self._run(
            hidden,
            residual,
            post_weight,
            pre_weight,
            post_eps=post_eps,
            pre_eps=pre_eps,
            output_dtype=hidden.dtype,
        )

    def forward_fp32(
        self,
        hidden: Tensor,
        residual: Tensor,
        post_weight: Tensor,
        pre_weight: Tensor,
        *,
        post_eps: float = POST_NORM_EPS,
        pre_eps: float = RMS_NORM_EPS,
    ) -> tuple[Tensor, Tensor]:
        """Ground truth: same arithmetic, FP32 out, and the add stays in FP32.

        Not simply ``forward`` with a wider output. The narrow cast between the
        post-norm and the add is part of ``forward``'s contract, so removing it
        changes the result; this entry point exists to measure how much.
        """
        return self._run(
            hidden,
            residual,
            post_weight,
            pre_weight,
            post_eps=post_eps,
            pre_eps=pre_eps,
            output_dtype=torch.float32,
        )

    def _run(
        self,
        hidden: Tensor,
        residual: Tensor,
        post_weight: Tensor,
        pre_weight: Tensor,
        *,
        post_eps: float,
        pre_eps: float,
        output_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        _validate_row(hidden, "hidden")
        _validate_row(residual, "residual")
        if hidden.shape != residual.shape:
            raise ValueError(
                f"hidden and residual must share shape, got {tuple(hidden.shape)} "
                f"and {tuple(residual.shape)}."
            )
        if hidden.dtype != residual.dtype:
            raise ValueError(
                f"hidden and residual must share dtype, got {hidden.dtype} "
                f"and {residual.dtype}."
            )
        width = hidden.shape[-1]
        _validate_weight(post_weight, width, "post_weight")
        _validate_weight(pre_weight, width, "pre_weight")

        post = _centered_norm(hidden.float(), post_weight, post_eps)
        # Both casts are load-bearing, see the class docstring. The pre-norm reads
        # the narrowed residual, not the FP32 sum: eager adds in the model dtype
        # and the next norm sees that rounded value, so feeding the unrounded sum
        # forward would quietly disagree with HF on the BF16 path.
        residual_out = (residual.float() + post.to(output_dtype).float()).to(output_dtype)
        pre_normed = _centered_norm(residual_out.float(), pre_weight, pre_eps)
        return residual_out, pre_normed.to(output_dtype)


class NativeFinalNormOp(nn.Module):
    """The model's last norm: ``rmsnorm_times_weight`` with ``pow_neg_half``."""

    def forward(self, x: Tensor, weight: Tensor, *, eps: float = RMS_NORM_EPS) -> Tensor:
        return self._run(x, weight, eps=eps, output_dtype=x.dtype)

    def forward_fp32(self, x: Tensor, weight: Tensor, *, eps: float = RMS_NORM_EPS) -> Tensor:
        return self._run(x, weight, eps=eps, output_dtype=torch.float32)

    def _run(self, x: Tensor, weight: Tensor, *, eps: float, output_dtype: torch.dtype) -> Tensor:
        _validate_row(x, "x")
        _validate_weight(weight, x.shape[-1], "weight")
        return _scaled_norm(x.float(), weight, eps).to(output_dtype)


# --------------------------------------------------------------------------- #
# Closed-form backward
# --------------------------------------------------------------------------- #
# Autograd already differentiates the references above. These functions exist
# because the CUDA and Triton kernels cannot, and they have to agree with autograd
# before that arithmetic gets written twice in two languages. The tests compare
# them against autograd on the reference and against FP64.
#
# For y = x * rstd * g with rstd = (meansq(x) + eps) ** -0.5 and g fixed per column:
#
#     dx_j = rstd * dy_j * g_j - x_j * rstd**3 * c,   c = rowsum(dy * g * x) / n
#     dw_j = sum over rows of dy_j * x_j * rstd
#
# g is (1 + w) for the decoder norms and w for the final norm, and dw is the same
# either way because d(1 + w)/dw is 1. The row reduction in c uses
# shape_invariant_rowsum for the same reason the forward uses the chunked mean: a
# plain sum makes the gradient depend on the batch layout.


def _centered_backward(
    dy_f: Tensor, x_f: Tensor, gain: Tensor, rstd: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(dx, dweight_rows)`` for one norm, all FP32.

    ``gain`` is the full multiplier the forward applied, so ``1 + w`` for a decoder
    norm and ``w`` for the final norm. ``rstd`` is the forward's statistic, passed
    in rather than recomputed so the backward cannot drift from the forward.
    """
    hidden = x_f.shape[-1]
    scaled = dy_f * gain
    c = (shape_invariant_rowsum(scaled * x_f) / float(hidden)).unsqueeze(-1)
    dx = rstd * scaled - x_f * rstd.pow(3) * c
    dweight_rows = dy_f * x_f * rstd
    return dx, dweight_rows


def fused_add_norm_backward(
    grad_residual_out: Tensor,
    grad_pre_normed: Tensor,
    hidden: Tensor,
    residual: Tensor,
    post_weight: Tensor,
    pre_weight: Tensor,
    *,
    post_eps: float = POST_NORM_EPS,
    pre_eps: float = RMS_NORM_EPS,
    grad_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gradients of ``NativeFusedAddNormOp.forward``, closed form.

    Returns ``(d_hidden, d_residual, d_post_weight_rows, d_pre_weight_rows)``. All
    four are FP32 tensors. The two weight entries are per-row FP32 contributions,
    not sums, matching ``parameter_vjp_contributions_fp32`` elsewhere in the repo;
    the caller folds them with ``reduce_rows_fp32``.

    One asymmetry in the values, stated here because a caller has to know it and a
    comment further down is too late: ``d_residual`` carries a narrowing, having
    been rounded to ``grad_dtype`` and widened again, and ``d_hidden`` is a raw FP32
    accumulation. Compared against a kernel or against autograd without a cast,
    ``d_hidden`` is the one that shows ULP-level differences, which is easy to read
    as a kernel fault. Cast it to the input dtype first, the way the autograd
    wrappers and the parity tests do. The reason for the asymmetry rather than its
    consequence is at the return statement.

    ``residual_out`` is both an output and the pre-norm's input, so the gradient
    reaching it is the sum of what comes back from downstream and what comes back
    through the pre-norm. Getting that wrong is invisible when the downstream
    cotangent is zero, which is why the tests drive both outputs.

    That sum is accumulated in ``grad_dtype``, which defaults to the input dtype,
    and this is the backward half of the forward's narrowing contract rather than
    a convenience. In eager the residual output is a model-dtype tensor with two
    gradient consumers, and autograd accumulates both arrivals in that dtype:
    ``round(g_res + round(d_from_pre))``. Keeping the sum in FP32 instead looks
    harmless per element, one ULP at most, but the post-norm weight gradient sums
    it over rows and the discrepancy compounds. Measured at bf16 against the
    reference's own autograd: FP32 accumulation puts ``d_post_w`` 2.6 to 2.9 times
    outside the contract row at 512 and 2048 rows; a single rounding of the FP32
    sum does not help; the eager accumulation brings it to 0.125 and then 0.000.
    The forward narrowing is where the value is rounded; this is where its
    gradient is. With ``grad_dtype=torch.float32`` both roundings are the
    identity and this is the gradient of ``forward_fp32``, whose cotangents are
    FP32 for any input dtype; the two go together, and the cotangents are checked
    against ``grad_dtype`` rather than against ``hidden`` for that reason.
    """
    dtype = hidden.dtype if grad_dtype is None else grad_dtype
    _validate_backward(
        ((hidden, "hidden"), (residual, "residual")),
        ((post_weight, "post_weight"), (pre_weight, "pre_weight")),
        cotangents=(
            (grad_residual_out, "grad_residual_out"),
            (grad_pre_normed, "grad_pre_normed"),
        ),
        # Not hidden's dtype: the outputs these are cotangents of are whatever the
        # differentiated forward returns, and grad_dtype is how the caller says
        # which forward that is. Tying them to hidden instead made the last
        # sentence of this docstring unreachable, since forward_fp32 on a narrow
        # hidden returns FP32 and so hands FP32 cotangents to autograd.
        cotangent_dtype=dtype,
        # The default is the narrowed forward, so FP32 cotangents against a narrow
        # hidden are rejected, and the message has to say what to do about that
        # rather than leave the caller to find grad_dtype. Defaulting to the
        # cotangents' own dtype instead was the alternative and is not taken: the
        # narrowing dtype would then follow the cotangent, which is the kernel bug
        # test_triton_backward_narrowing_follows_hidden_not_the_cotangent exists to
        # catch, moved into the oracle.
        cotangent_hint=(
            " Pass grad_dtype=torch.float32 for the gradient of forward_fp32, whose"
            " outputs are FP32 for every input dtype."
        ),
    )
    hidden_f = hidden.float()
    residual_f = residual.float()
    post_gain = 1.0 + post_weight.float()
    pre_gain = 1.0 + pre_weight.float()

    # Recompute the forward, narrowing where it narrows, to recover both
    # statistics and the intermediate the pre-norm actually saw.
    rstd_post = torch.rsqrt(shape_invariant_meansq(hidden_f) + float(post_eps)).unsqueeze(-1)
    post = ((hidden_f * rstd_post) * post_gain).to(dtype).float()
    residual_out = (residual_f + post).to(dtype).float()
    rstd_pre = torch.rsqrt(shape_invariant_meansq(residual_out) + float(pre_eps)).unsqueeze(-1)

    d_from_pre, d_pre_weight_rows = _centered_backward(
        grad_pre_normed.float(), residual_out, pre_gain, rstd_pre
    )
    # The residual output is live on two paths; accumulate them the way eager
    # does, in the model dtype, before going further back.
    d_residual_out = (grad_residual_out.float() + d_from_pre.to(dtype).float()).to(dtype).float()

    d_hidden, d_post_weight_rows = _centered_backward(
        d_residual_out, hidden_f, post_gain, rstd_post
    )
    # d_hidden comes back in FP32 while d_residual_out carries a narrowing, and the
    # asymmetry is deliberate rather than an oversight. The narrowing on
    # d_residual_out is observable inside this function: it is the value the
    # post-norm backward above consumes, so rounding it where eager rounds it
    # changes d_post_weight_rows. d_hidden is a return value with no consumer here,
    # so it stays an FP32 accumulation and the caller rounds it where its own
    # contract says to -- the kernels' autograd wrappers cast to the input dtype,
    # and the parity test compares g.to(dtype) against autograd.
    return d_hidden, d_residual_out, d_post_weight_rows, d_pre_weight_rows


def final_norm_backward(
    grad_output: Tensor, x: Tensor, weight: Tensor, *, eps: float = RMS_NORM_EPS
) -> tuple[Tensor, Tensor]:
    """Gradients of ``NativeFinalNormOp.forward_fp32``: ``(dx, dweight_rows)``.

    Separate from the decoder-norm path because the gain is ``w`` rather than
    ``1 + w`` and the statistic is spelled ``pow_neg_half``.

    ``grad_output`` may arrive in any supported dtype, including one that differs
    from ``x``. There is no narrowing inside this form, so the cotangent is upcast
    on entry and its storage dtype does not reach the answer: FP32 is what
    differentiating ``forward_fp32`` on a narrow ``x`` actually hands over, and a
    narrow cotangent is the same values pre-rounded by the caller. The decoder-norm
    backward does pin this, because there the accumulation dtype is part of the
    contract.
    """
    _validate_backward(
        ((x, "x"),), ((weight, "weight"),), cotangents=((grad_output, "grad_output"),)
    )
    x_f = x.float()
    rstd = ((shape_invariant_meansq(x_f) + float(eps)) ** -0.5).unsqueeze(-1)
    return _centered_backward(grad_output.float(), x_f, weight.float(), rstd)
