# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 sinusoidal timestep features (RFC #420 row ``timestep_sinusoid_h3``).

H3 builds ``Timesteps(num_channels=256, flip_sin_to_cos=True,
downscale_freq_shift=0)`` and feeds it the *distinct* timestep values of the
packed sequence, unscaled in ``[0, 1]`` (``t = 1 - sigma``)::

    freq[k] = exp(-ln(10000) * k / 128)          k = 0..127, FP32
    arg[t, k] = t * freq[k]                      FP32
    out[t] = [cos(arg[t]) | sin(arg[t])]         (T, 256) FP32

The output is always FP32: ``time_embedder`` is a ``_keep_in_fp32_modules``
module in the checkpoint, so the features never pass through BF16.
"""

from __future__ import annotations

import math

import torch

from rl_engine.kernels.ops.pytorch.h3 import H3_FREQ_DIM, H3_MAX_PERIOD

_TIMESTEP_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def validate_h3_timesteps(
    timestep: torch.Tensor, num_channels: int, *, check_range: bool = True
) -> None:
    """Fail closed on inputs the H3 convention does not define.

    ``check_range`` rejects values outside ``[0, 1]``: H3 consumes
    ``t = 1 - sigma`` unscaled, so a ``t * 1000`` caller (RFC probe H10) is a
    convention error, not a different valid input. It reads the tensor back
    to the host, which is cheap for the handful of distinct timesteps.
    """

    if not isinstance(timestep, torch.Tensor):
        raise TypeError("timestep must be a torch.Tensor")
    if timestep.dim() != 1:
        raise ValueError(f"timestep must be 1-D (num_timesteps,), got {tuple(timestep.shape)}")
    if timestep.numel() == 0:
        raise ValueError("timestep must hold at least one timestep")
    if timestep.dtype not in _TIMESTEP_DTYPES:
        raise TypeError(f"timestep must be fp32, bf16 or fp16, got {timestep.dtype}")
    if num_channels <= 0 or num_channels % 2 != 0:
        raise ValueError(f"num_channels must be a positive even number, got {num_channels}")
    if check_range:
        t32 = timestep.detach().float()
        # One host sync: NaN fails both comparisons, so this also rejects it.
        if not bool(((t32 >= 0) & (t32 <= 1)).all()):
            raise ValueError(
                "timestep must be finite and lie in [0, 1]: H3 consumes t = 1 - sigma "
                f"unscaled (got {t32.cpu().tolist()[:8]})"
            )


class NativeH3TimestepSinusoidOp:
    """PyTorch reference for the H3 sinusoidal timestep features.

    ``forward`` replays diffusers' ``get_timestep_embedding`` op for op on the
    input's device (the provider path). ``forward_fp32`` is the independent
    golden: the same formula evaluated in FP64 and rounded once to FP32.
    """

    op_class = "elementwise"

    def __call__(self, timestep: torch.Tensor, *, num_channels: int = H3_FREQ_DIM):
        """Return FP32 sinusoidal features for nonempty timesteps in ``[0, 1]``."""

        return self.forward(timestep, num_channels=num_channels)

    def forward(
        self,
        timestep: torch.Tensor,
        *,
        num_channels: int = H3_FREQ_DIM,
        check_range: bool = True,
    ) -> torch.Tensor:
        """Return FP32 ``(T, num_channels)`` cosine-then-sine features on the input device.

        Require nonempty 1-D FP32, BF16 or FP16 timesteps and positive even
        channels. Validate finite values in ``[0, 1]`` when ``check_range`` is
        true and preserve PyTorch autograd through the provider formula.
        """

        validate_h3_timesteps(timestep, num_channels, check_range=check_range)
        half = num_channels // 2
        # Same op sequence and dtypes as diffusers get_timestep_embedding with
        # flip_sin_to_cos=True, downscale_freq_shift=0, scale=1.
        exponent = -math.log(H3_MAX_PERIOD) * torch.arange(
            start=0, end=half, dtype=torch.float32, device=timestep.device
        )
        exponent = exponent / half
        freq = torch.exp(exponent)
        arg = timestep[:, None].float() * freq[None, :]
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)

    def forward_fp32(
        self,
        timestep: torch.Tensor,
        *,
        num_channels: int = H3_FREQ_DIM,
        check_range: bool = True,
    ) -> torch.Tensor:
        """Evaluate the validated sinusoid formula in FP64 and round once to FP32.

        Return ``(T, num_channels)`` on the timestep device with cosine channels
        followed by sine channels and gradients through the FP64 golden graph.
        """

        validate_h3_timesteps(timestep, num_channels, check_range=check_range)
        half = num_channels // 2
        k = torch.arange(half, dtype=torch.float64, device=timestep.device)
        freq = torch.exp(-math.log(H3_MAX_PERIOD) * k / half)
        arg = timestep.double()[:, None] * freq[None, :]
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1).float()

    @staticmethod
    def frequencies_fp32(num_channels: int, device: torch.device | str) -> torch.Tensor:
        """The FP32 frequency table the provider path multiplies by."""

        half = num_channels // 2
        exponent = -math.log(H3_MAX_PERIOD) * torch.arange(
            start=0, end=half, dtype=torch.float32, device=device
        )
        return torch.exp(exponent / half)
