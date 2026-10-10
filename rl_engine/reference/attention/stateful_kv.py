# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""In-repo stateful KV cache (WS1 C7 / #273).

This is the B1 writer/reader: allocate → write → read. Concat-only
``NativeKVCacheAttnOp`` is a Level-A reference and does **not** satisfy B1.
Decode itself is performed by a declared attention candidate on the tensors
returned by ``read()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StatefulKVCache:
    """Mutable per-layer K/V buffers with an explicit write cursor.

    Layout is ``[n_layers, batch, n_kv_heads, max_seq_len, head_dim]``. Lengths
    are stored per batch row so padding does not advance the cursor for pad
    tokens when a validity mask is supplied.
    """

    n_layers: int
    batch: int
    n_kv_heads: int
    max_seq_len: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    k: torch.Tensor
    v: torch.Tensor
    lengths: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        n_layers: int,
        batch: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> StatefulKVCache:
        if min(n_layers, batch, n_kv_heads, max_seq_len, head_dim) <= 0:
            raise ValueError("stateful KV allocate dims must be positive")
        dev = torch.device(device)
        zeros = torch.zeros(
            (n_layers, batch, n_kv_heads, max_seq_len, head_dim),
            device=dev,
            dtype=dtype,
        )
        return cls(
            n_layers=int(n_layers),
            batch=int(batch),
            n_kv_heads=int(n_kv_heads),
            max_seq_len=int(max_seq_len),
            head_dim=int(head_dim),
            dtype=dtype,
            device=dev,
            k=zeros,
            v=zeros.clone(),
            lengths=torch.zeros((n_layers, batch), device=dev, dtype=torch.int64),
            valid=torch.zeros((n_layers, batch, max_seq_len), device=dev, dtype=torch.bool),
        )

    def reset(self) -> None:
        self.k.zero_()
        self.v.zero_()
        self.lengths.zero_()
        self.valid.zero_()

    def write(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        *,
        layer: int = 0,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Append ``k_new`` / ``v_new`` (``[B, Hkv, S_new, D]``) at the cursor."""

        self._check_layer(layer)
        if k_new.shape != v_new.shape:
            raise ValueError(
                f"k_new/v_new shape mismatch: {tuple(k_new.shape)} vs {tuple(v_new.shape)}"
            )
        if k_new.dim() != 4:
            raise ValueError(f"k_new must be [B, Hkv, S_new, D], got {tuple(k_new.shape)}")
        batch, n_kv, s_new, head_dim = k_new.shape
        if (batch, n_kv, head_dim) != (self.batch, self.n_kv_heads, self.head_dim):
            raise ValueError(
                "k_new layout must be [B, Hkv, S_new, D] matching the cache: "
                f"got {(batch, n_kv, head_dim)}, "
                f"want {(self.batch, self.n_kv_heads, self.head_dim)}"
            )
        if k_new.dtype != self.k.dtype or k_new.device != self.k.device:
            raise ValueError(
                "k_new/v_new must match cache dtype and device: "
                f"got dtype={k_new.dtype}/{v_new.dtype} device={k_new.device}/{v_new.device}, "
                f"cache dtype={self.k.dtype} device={self.k.device}"
            )

        start = int(self.lengths[layer, 0].item())
        # All rows share a packed write cursor for the BI path (pad tokens are
        # written as zeros when valid_mask is provided, but the cursor still
        # advances by S_new so decode positions stay aligned).
        if not torch.equal(
            self.lengths[layer], self.lengths[layer, :1].expand_as(self.lengths[layer])
        ):
            raise RuntimeError("per-row cache lengths diverged; WS1 B1 requires a shared cursor")
        end = start + int(s_new)
        if end > self.max_seq_len:
            raise RuntimeError(
                f"stateful KV overflow: writing {s_new} tokens at {start} "
                f"exceeds max_seq_len={self.max_seq_len}"
            )
        k_store = k_new
        v_store = v_new
        if valid_mask is not None:
            if valid_mask.shape != (batch, s_new):
                raise ValueError(
                    f"valid_mask must be [B, S_new]={(batch, s_new)}, "
                    f"got {tuple(valid_mask.shape)}"
                )
            keep = valid_mask.to(device=self.device, dtype=torch.bool)[:, None, :, None]
            k_store = torch.where(keep, k_new, torch.zeros_like(k_new))
            v_store = torch.where(keep, v_new, torch.zeros_like(v_new))
            valid_store = valid_mask.to(device=self.device, dtype=torch.bool)
        else:
            valid_store = torch.ones((batch, s_new), device=self.device, dtype=torch.bool)
        self.k[layer, :, :, start:end, :].copy_(k_store)
        self.v[layer, :, :, start:end, :].copy_(v_store)
        self.valid[layer, :, start:end].copy_(valid_store)
        self.lengths[layer].fill_(end)

    def read(self, *, layer: int = 0) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return written ``(k, v, length)`` tensors (never caller-side concat).

        During training, a later append mutates the backing buffer. Returning a
        clone when the buffer participates in autograd prevents that append from
        invalidating tensors saved by an earlier chunk's backward function. The
        clone keeps its ``CopySlices`` gradient edge back to the written K/V.
        In inference/no-grad mode this remains a zero-copy view.
        """

        self._check_layer(layer)
        length = int(self.lengths[layer, 0].item())
        k = self.k[layer, :, :, :length, :]
        v = self.v[layer, :, :, :length, :]
        if torch.is_grad_enabled() and (k.requires_grad or v.requires_grad):
            k = k.clone()
            v = v.clone()
        return k, v, length

    def read_valid_mask(self, *, layer: int = 0) -> torch.Tensor:
        """Return the validity mask for the written prefix."""

        self._check_layer(layer)
        length = int(self.lengths[layer, 0].item())
        return self.valid[layer, :, :length]

    def identity(self) -> dict[str, str]:
        return {
            "kind": "stateful_kv_buffer",
            "layout": "[n_layers, batch, n_kv_heads, max_seq_len, head_dim]",
            "writer": "StatefulKVCache.write",
            "reader": "StatefulKVCache.read",
            "validity_reader": "StatefulKVCache.read_valid_mask",
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
        }

    def _check_layer(self, layer: int) -> None:
        if layer < 0 or layer >= self.n_layers:
            raise IndexError(f"layer {layer} out of range for n_layers={self.n_layers}")


__all__ = ["StatefulKVCache"]
