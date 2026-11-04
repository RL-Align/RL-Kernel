# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Pure PyTorch references for the DeepSeek-V4 mHC pre, post and head steps.

The formulas follow the DeepSeek-V4-Flash-0731 reference implementation
(``inference/model.py``: ``Block.hc_pre``, ``Block.hc_post``, ``Block.hc_head``;
``inference/kernel.py``: ``hc_split_sinkhorn``), computed in fp32. Tensors use
the stream layout ``[..., n, C]``: ``n`` residual streams of hidden size ``C``
per token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _check_post_shapes(y, residual, post_mix, comb_mix):
    n, c = residual.shape[-2:]
    lead = residual.shape[:-2]
    if y.shape != (*lead, c):
        raise ValueError(f"y must have shape {(*lead, c)}, got {tuple(y.shape)}")
    if post_mix.shape not in ((*lead, n), (*lead, n, 1)):
        raise ValueError(f"post_mix must have shape {(*lead, n)}, got {tuple(post_mix.shape)}")
    if comb_mix.shape != (*lead, n, n):
        raise ValueError(f"comb_mix must have shape {(*lead, n, n)}, got {tuple(comb_mix.shape)}")


def sinkhorn_dsv4(logits, iters, eps):
    """DeepSeek-V4's asymmetric Sinkhorn on [..., n, n] logits; iters=0 returns them.

    First round: softmax over each row, plus eps, then divide each column by
    (column sum + eps). Each further round: divide rows, then columns, by
    (sum + eps).
    """
    if iters == 0:
        return logits
    comb = torch.softmax(logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


class NativeMHCPreOp:
    """
    Pure PyTorch mHC pre reference: mixing weights and the sublayer input.
    x = flatten(residual) in fp32;  r_inv = rsqrt(mean(x^2) + rms_eps)
    mixes = (x @ fn^T) * r_inv, split as [pre n | post n | comb n*n]
    pre  = sigmoid(mixes_pre * scale[0] + base_pre) + hc_pre_eps
    post = hc_post_mult_value * sigmoid(mixes_post * scale[1] + base_post)
    comb = Sinkhorn(mixes_comb * scale[2] + base_comb), sinkhorn_repeat rounds
    layer_input = sum_i pre[..., i] * residual[..., i, :]
    Returns (post_mix [..., n], comb_mix [..., n, n], layer_input [..., C]).
    """

    def __call__(self, residual, fn, scale, base, **kwargs):
        return self.forward(residual, fn, scale, base, **kwargs)

    def forward(self, residual, fn, scale, base, **kwargs):
        """layer_input cast to ``residual.dtype``, as DeepSeek-V4 does."""
        post, comb, layer_input = self._pre(residual, fn, scale, base, **kwargs)
        return post, comb, layer_input.to(residual.dtype)

    def forward_fp32(self, residual, fn, scale, base, **kwargs):
        """Ground truth: every output in fp32."""
        return self._pre(residual, fn, scale, base, **kwargs)

    @staticmethod
    def _pre(
        residual,
        fn,
        scale,
        base,
        *,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    ):
        n, c = residual.shape[-2:]
        rows = 2 * n + n * n
        if fn.shape != (rows, n * c):
            raise ValueError(f"fn must have shape {(rows, n * c)}, got {tuple(fn.shape)}")
        if scale.shape != (3,) or base.shape != (rows,):
            raise ValueError(
                f"scale and base must have shapes (3,) and {(rows,)}, "
                f"got {tuple(scale.shape)} and {tuple(base.shape)}"
            )
        x = residual.float().flatten(-2)
        r_inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
        mixes = F.linear(x, fn.float()) * r_inv
        scale, base = scale.float(), base.float()
        pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + hc_pre_eps
        post_logits = mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n]
        post = hc_post_mult_value * torch.sigmoid(post_logits)
        logits = (mixes[..., 2 * n :] * scale[2] + base[2 * n :]).unflatten(-1, (n, n))
        comb = sinkhorn_dsv4(logits, sinkhorn_repeat, hc_sinkhorn_eps)
        layer_input = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=-2)
        return post, comb, layer_input


class NativeMHCPostOp:
    """
    Pure PyTorch mHC post (stream mix-back) reference.
    out[..., j, :] = post_mix[..., j] * y + sum_h comb_mix[..., h, j] * residual[..., h, :]

    ``comb_mix`` is indexed [source stream, destination stream], so it is the
    transpose of H_res in the mHC paper. The old residual enters only through
    ``comb_mix``; it is not added a second time.
    """

    def __call__(self, y, residual, post_mix, comb_mix):
        return self.forward(y, residual, post_mix, comb_mix)

    def forward(self, y, residual, post_mix, comb_mix):
        """Compute in fp32 and cast to ``residual.dtype``, as DeepSeek-V4 does."""
        return self._post(y, residual, post_mix, comb_mix).to(residual.dtype)

    def forward_fp32(self, y, residual, post_mix, comb_mix):
        """Ground truth: compute in fp32 and keep fp32 output."""
        return self._post(y, residual, post_mix, comb_mix)

    @staticmethod
    def _post(y, residual, post_mix, comb_mix):
        _check_post_shapes(y, residual, post_mix, comb_mix)
        if post_mix.dim() == residual.dim():
            post_mix = post_mix.squeeze(-1)
        post_term = post_mix.float().unsqueeze(-1) * y.float().unsqueeze(-2)
        mix_term = torch.sum(
            comb_mix.float().unsqueeze(-1) * residual.float().unsqueeze(-2), dim=-3
        )
        return post_term + mix_term


class NativeMHCHeadOp:
    """
    Pure PyTorch mHC head reference: contract n streams into one hidden state.
    x = flatten(residual) in fp32
    mix = sigmoid((x @ fn^T) * rsqrt(mean(x^2) + rms_eps) * scale + base) + hc_pre_eps
    out = sum_i mix[..., i] * residual[..., i, :]
    """

    def __call__(self, residual, fn, scale, base, *, rms_eps=1e-6, hc_pre_eps=1e-6):
        return self.forward(residual, fn, scale, base, rms_eps=rms_eps, hc_pre_eps=hc_pre_eps)

    def forward(self, residual, fn, scale, base, *, rms_eps=1e-6, hc_pre_eps=1e-6):
        """Compute in fp32 and cast to ``residual.dtype``, as DeepSeek-V4 does."""
        out = self._head(residual, fn, scale, base, rms_eps=rms_eps, hc_pre_eps=hc_pre_eps)
        return out.to(residual.dtype)

    def forward_fp32(self, residual, fn, scale, base, *, rms_eps=1e-6, hc_pre_eps=1e-6):
        """Ground truth: compute in fp32 and keep fp32 output."""
        return self._head(residual, fn, scale, base, rms_eps=rms_eps, hc_pre_eps=hc_pre_eps)

    @staticmethod
    def _head(residual, fn, scale, base, *, rms_eps, hc_pre_eps):
        n, c = residual.shape[-2:]
        if fn.shape != (n, n * c):
            raise ValueError(f"fn must have shape {(n, n * c)}, got {tuple(fn.shape)}")
        if scale.shape != (1,) or base.shape != (n,):
            raise ValueError(
                f"scale and base must have shapes (1,) and {(n,)}, "
                f"got {tuple(scale.shape)} and {tuple(base.shape)}"
            )
        x = residual.float().flatten(-2)
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
        mixes = F.linear(x, fn.float()) * rsqrt
        mix = torch.sigmoid(mixes * scale.float() + base.float()) + hc_pre_eps
        return torch.sum(mix.unsqueeze(-1) * residual.float(), dim=-2)
