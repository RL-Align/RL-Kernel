# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 FP32 timestep MLP (RFC #420 row ``timestep_mlp_fp32``).

``TimestepEmbedding(in_channels=256, time_embed_dim=5376, out_dim=2688)``::

    temb = linear_2(silu(linear_1(features)))      256 -> 5376 -> 2688

``time_embedder`` is in ``_keep_in_fp32_modules``: weights, biases,
activations and the output are FP32, and ``temb`` stays FP32 because every
AdaLN projection applies its own SiLU to it before casting (RFC #420 §4).
A BF16 argument is a contract violation (RFC probe H7), not a lower-precision
variant, so it is rejected.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_h3_timestep_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> None:
    """Check a same-device FP32 ``(T, K) -> (T, H) -> (T, D)`` MLP contract.

    Require ``T > 0``, weights of shapes ``(H, K)``/``(D, H)`` and biases
    of shapes ``(H,)``/``(D,)``; reject lower-precision parameters or inputs.
    """

    named = {"x": x, "w1": w1, "b1": b1, "w2": w2, "b2": b2}
    for name, tensor in named.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(
                f"{name} must be float32: the H3 time_embedder is an FP32 module "
                f"(got {tensor.dtype})"
            )
        if tensor.device != x.device:
            raise ValueError(f"{name} is on {tensor.device}, x is on {x.device}")
    if x.dim() != 2 or x.shape[0] == 0:
        raise ValueError(f"x must be a non-empty (T, K) matrix, got {tuple(x.shape)}")
    hidden, k_in = w1.shape if w1.dim() == 2 else (None, None)
    if k_in != x.shape[1]:
        raise ValueError(f"w1 must be (H, {x.shape[1]}), got {tuple(w1.shape)}")
    if b1.shape != (hidden,):
        raise ValueError(f"b1 must be ({hidden},), got {tuple(b1.shape)}")
    if w2.dim() != 2 or w2.shape[1] != hidden:
        raise ValueError(f"w2 must be (D, {hidden}), got {tuple(w2.shape)}")
    if b2.shape != (w2.shape[0],):
        raise ValueError(f"b2 must be ({w2.shape[0]},), got {tuple(b2.shape)}")


class NativeH3TimestepMLPOp:
    """PyTorch reference for the H3 timestep MLP.

    ``forward`` is the provider path (``nn.Linear`` / ``F.silu`` in FP32; run
    it with TF32 disabled, as the RL-Kernel contract requires).
    ``forward_fp32`` is the independent golden: the same graph in FP64,
    rounded once to FP32, so its own rounding error (~1e-16 relative) is far
    below the FP32 contract and its reduction order does not matter.
    """

    op_class = "reduction"

    def __call__(self, x, w1, b1, w2, b2):
        """Return FP32 ``(T, D)`` embeddings using the PyTorch provider graph."""

        return self.forward(x, w1, b1, w2, b2)

    def forward(self, x, w1, b1, w2, b2) -> torch.Tensor:
        """Validate and evaluate linear-SiLU-linear in FP32 on the input device.

        Inputs follow the ``(T, K) -> (T, H) -> (T, D)`` contract and retain
        PyTorch autograd support; CUDA callers must disable TF32 for this path.
        """

        validate_h3_timestep_mlp(x, w1, b1, w2, b2)
        return F.linear(F.silu(F.linear(x, w1, b1)), w2, b2)

    def forward_fp32(self, x, w1, b1, w2, b2) -> torch.Tensor:
        """Evaluate the validated MLP in FP64 and round its ``(T, D)`` output to FP32."""

        validate_h3_timestep_mlp(x, w1, b1, w2, b2)
        z = F.linear(x.double(), w1.double(), b1.double())
        h = z * torch.sigmoid(z)
        return F.linear(h, w2.double(), b2.double()).float()
