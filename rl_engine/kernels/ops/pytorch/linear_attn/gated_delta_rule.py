# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Gated DeltaNet single-token recurrent step (WS1 ground truth for RFC #428 C6).

This is the trainer-side reference for what vLLM runs during rollout decode:
``fused_recurrent_gated_delta_rule_packed_decode``. That kernel -- not
``fused_sigmoid_gating_delta_rule_update`` -- is the path a pure RL rollout
takes, because ``VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`` defaults to true and
a decode-only, non-speculative batch returns early into it.

Three things are transcribed from the kernel rather than from the HuggingFace
model, because they differ:

* **The gating is fused.** ``beta = sigmoid(b)`` and
  ``g = -exp(A_log) * softplus(a + dt_bias)`` are computed here, in fp32, with
  the kernel's ``softplus`` threshold branch. HF computes them separately in
  PyTorch, which is a different rounding path.
* **No ``repeat_interleave``.** The kernel indexes ``i_h = i_hv // (HV // H)``,
  so q/k stay at ``H`` heads while v has ``HV``. HF materializes the repeat.
* **The QK norm is an L2 norm over a plain sum**, ``x / sqrt(sum(x*x) + 1e-6)``,
  not an RMSNorm and not ``F.normalize``. It divides by ``sqrt`` rather than
  multiplying by ``rsqrt``; the two differ in the last bit, so the division is
  kept.

``scale`` is applied to ``q`` *after* the L2 norm; ``k`` is never scaled.

State ABI (mirrored, not reinvented)
------------------------------------
The recurrent state is paged: ``[num_blocks, HV, V, K]``, addressed per
sequence by ``ssm_state_indices``. Index ``<= 0`` is ``NULL_BLOCK_ID`` and means
"skip": the kernel writes zeros to the output and leaves the block untouched.
The accumulator is fp32 throughout the step, and the store rounds to the state
tensor's dtype -- which upstream allows to be fp32 *or* bf16. That rounding is
part of the recurrence and compounds across tokens, so it is modelled here
rather than skipped.
"""

from __future__ import annotations

import torch

__all__ = ["GatedDeltaRuleRecurrentStepOp", "NULL_BLOCK_ID", "SOFTPLUS_THRESHOLD"]

#: Paged-state sentinel: a sequence pointing here is skipped.
NULL_BLOCK_ID = 0

#: Above this, softplus is the identity (matches the kernel's constexpr).
SOFTPLUS_THRESHOLD = 20.0

#: Width of the fixed-order reduction chunks. The contraction order must not
#: depend on the batch layout, so it is pinned here exactly as
#: :func:`~rl_engine.kernels.ops.pytorch.norm.rms_norm.shape_invariant_rstd`
#: pins the RMSNorm statistic.
_REDUCTION_CHUNK = 32


def _validate_state_indices(indices, batch, blocks, device):
    """Each active row owns one cache block; inactive sentinels may repeat."""
    if indices.ndim != 1 or indices.shape[0] != batch:
        raise ValueError("state indices must be 1-D with one entry per sequence")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state indices must have int32 or int64 dtype")
    if indices.device != device:
        raise ValueError("state indices must be on the input device")
    active = indices[indices > NULL_BLOCK_ID]
    if bool((active >= blocks).any()):
        raise ValueError("active state index is out of range")
    if active.unique().numel() != active.numel():
        raise ValueError("active state indices must be unique")


def _chunked_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum the last dim in a fixed 32-wide chunk order.

    The single reduction primitive for this module. Deliberately not
    ``sum``/``matmul``/``einsum`` on the whole axis: their reduction order is
    unspecified and may vary with shape, which is what a batch-invariant claim
    cannot tolerate. Mirrors
    :func:`~rl_engine.kernels.ops.pytorch.norm.rms_norm.shape_invariant_rstd`.
    """
    tail = x.shape[-1]
    if tail % _REDUCTION_CHUNK != 0:
        return x.sum(dim=-1)
    return (
        x.reshape(*x.shape[:-1], tail // _REDUCTION_CHUNK, _REDUCTION_CHUNK).sum(dim=-1).sum(dim=-1)
    )


def _fixed_order_contract(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``sum(mat * vec, dim=-1)`` in a fixed chunk order."""
    return _chunked_sum(mat * vec)


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x / sqrt(sum(x*x) + eps)`` over the last dim, in fp32.

    A plain sum, not a mean: this is an L2 norm, not an RMSNorm. The division is
    kept rather than folded into a reciprocal-sqrt multiply because the kernel
    divides.
    """
    return x / torch.sqrt(_chunked_sum(x * x) + eps).unsqueeze(-1)


def _softplus(x: torch.Tensor, threshold: float = SOFTPLUS_THRESHOLD) -> torch.Tensor:
    """The kernel's branched softplus; the branch matters for large ``a + dt_bias``."""
    return torch.where(x <= threshold, torch.log1p(torch.exp(x)), x)


class GatedDeltaRuleRecurrentStepOp:
    """One decode token of the Gated DeltaNet recurrence, over a paged state.

    Shapes follow the provider, not the model definition:

    ============= ====================================== ===================
    tensor        shape                                  notes
    ============= ====================================== ===================
    ``mixed_qkv`` ``[B, H*K + H*K + HV*V]``              q | k | v, packed
    ``a``, ``b``  ``[B, HV]``
    ``A_log``     ``[HV]``                               fp32
    ``dt_bias``   ``[HV]``                               fp32
    ``state``     ``[num_blocks, HV, V, K]``             paged, V-major
    ``indices``   ``[B]``                                ``<= 0`` skips
    ============= ====================================== ===================

    Returns ``(out, state)`` with ``out`` of shape ``[B, 1, HV, V]``. The state
    is updated out of place; pass the result back to continue the recurrence.
    """

    def __call__(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        *,
        scale: float,
        num_k_heads: int,
        use_qk_l2norm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            state,
            ssm_state_indices,
            scale=scale,
            num_k_heads=num_k_heads,
            use_qk_l2norm=use_qk_l2norm,
        )

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        *,
        scale: float,
        num_k_heads: int,
        use_qk_l2norm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step once, rounding the stored state to ``state.dtype``."""
        return self._step(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            state,
            ssm_state_indices,
            scale=scale,
            num_k_heads=num_k_heads,
            use_qk_l2norm=use_qk_l2norm,
            state_dtype=state.dtype,
            output_dtype=mixed_qkv.dtype,
        )

    def forward_fp32(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        *,
        scale: float,
        num_k_heads: int,
        use_qk_l2norm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ground truth: keep the state and the output in fp32.

        The difference from :meth:`forward` is the per-token state rounding, so
        running both and comparing isolates how much that rounding costs.
        """
        return self._step(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            state,
            ssm_state_indices,
            scale=scale,
            num_k_heads=num_k_heads,
            use_qk_l2norm=use_qk_l2norm,
            state_dtype=torch.float32,
            output_dtype=torch.float32,
        )

    @staticmethod
    def _step(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        ssm_state_indices,
        *,
        scale,
        num_k_heads,
        use_qk_l2norm,
        state_dtype,
        output_dtype,
    ):
        if mixed_qkv.dim() != 2:
            raise ValueError(f"mixed_qkv must be 2-D [B, D], got {tuple(mixed_qkv.shape)}")
        if state.dim() != 4:
            raise ValueError(f"state must be 4-D [num_blocks, HV, V, K], got {tuple(state.shape)}")
        if ssm_state_indices.dim() != 1:
            raise ValueError("ssm_state_indices must be 1-D [B] for packed decode")

        batch = mixed_qkv.shape[0]
        hv, v_dim, k_dim = state.shape[-3:]
        if isinstance(num_k_heads, bool) or not isinstance(num_k_heads, int) or num_k_heads <= 0:
            raise ValueError("num_k_heads must be a positive integer")
        if min(hv, v_dim, k_dim) <= 0:
            raise ValueError("state head dimensions must be positive")
        heads = num_k_heads
        _validate_state_indices(ssm_state_indices, batch, state.shape[0], mixed_qkv.device)
        for name, tensor in (
            ("a", a),
            ("b", b),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
            ("state", state),
        ):
            if tensor.device != mixed_qkv.device:
                raise ValueError(f"{name} must be on the input device")
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must be floating point")
        if not mixed_qkv.is_floating_point():
            raise ValueError("mixed_qkv must be floating point")
        if A_log.shape != (hv,) or dt_bias.shape != (hv,):
            raise ValueError("A_log and dt_bias must have shape [HV]")
        if hv % heads != 0:
            raise ValueError(f"HV={hv} must be a multiple of num_k_heads={heads}")
        if a.shape != (batch, hv) or b.shape != (batch, hv):
            raise ValueError(
                f"a/b must be [B, HV] = {(batch, hv)}, got {tuple(a.shape)} / {tuple(b.shape)}"
            )
        expected = heads * k_dim * 2 + hv * v_dim
        if mixed_qkv.shape[1] != expected:
            raise ValueError(
                f"mixed_qkv last dim must be {expected} (q|k|v packed), "
                f"got {mixed_qkv.shape[1]}"
            )
        if ssm_state_indices.shape[0] != batch:
            raise ValueError("ssm_state_indices must have one entry per sequence")

        group = hv // heads
        qkv32 = mixed_qkv.float()

        # Unpack q | k | v. q and k carry H heads, v carries HV.
        q = qkv32[:, : heads * k_dim].reshape(batch, heads, k_dim)
        k = qkv32[:, heads * k_dim : 2 * heads * k_dim].reshape(batch, heads, k_dim)
        v = qkv32[:, 2 * heads * k_dim :].reshape(batch, hv, v_dim)

        if use_qk_l2norm:
            q = _l2_normalize(q)
            k = _l2_normalize(k)
        q = q * scale

        # i_h = i_hv // (HV // H): index, do not materialize a repeat.
        head_of = torch.arange(hv, device=q.device) // group
        q = q[:, head_of, :]  # [B, HV, K]
        k = k[:, head_of, :]

        # Fused gating, fp32, with the kernel's softplus branch.
        decay = -torch.exp(A_log.float()) * _softplus(a.float() + dt_bias.float())
        beta = torch.sigmoid(b.float())

        active = ssm_state_indices > NULL_BLOCK_ID
        out = torch.zeros(batch, 1, hv, v_dim, dtype=torch.float32, device=q.device)
        # The returned state carries `state_dtype`, not the caller's: forward_fp32
        # exists precisely to run the recurrence without the per-token rounding,
        # so it must be able to widen a bf16 cache to fp32.
        new_state = state.to(state_dtype).clone()
        if not bool(active.any()):
            return out.to(output_dtype), new_state

        rows = torch.nonzero(active, as_tuple=False).flatten()
        blocks = ssm_state_indices[rows].long()

        h = state[blocks].float()  # [R, HV, V, K]
        h = h * torch.exp(decay[rows]).unsqueeze(-1).unsqueeze(-1)

        k_sel, q_sel, v_sel = k[rows], q[rows], v[rows]
        # v -= h @ k, then v *= beta, then h += outer(v, k), then o = h @ q.
        v_sel = v_sel - _fixed_order_contract(h, k_sel.unsqueeze(-2))
        v_sel = v_sel * beta[rows].unsqueeze(-1)
        h = h + v_sel.unsqueeze(-1) * k_sel.unsqueeze(-2)
        o = _fixed_order_contract(h, q_sel.unsqueeze(-2))

        out[rows, 0] = o
        # The store rounds; that rounding is part of the recurrence.
        new_state[blocks] = h.to(state_dtype)
        return out.to(output_dtype), new_state
