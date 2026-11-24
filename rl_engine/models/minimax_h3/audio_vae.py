# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 Phase B ``audio_vae_contract``: MiniMax-H3 stereo AudioVAE encode/decode, 40 Hz layout.

The released H3 audio VAE is **mono**: a DAC-lineage encoder (hop 800 at 32 kHz, so 40 latents/s)
with a causal-attention ``pre_block`` that narrows the 2048-wide trunk to 32 latent channels, and a
BigVGAN decoder. MiniMax-H3 carries stereo as two batch items of that mono model and packs the
result **channel-major** into ``2 * T`` rows of width 32, which feed the ``32 -> 5376`` audio input
head (``audio_proj_in``). Diffusers (``AutoencoderKLMiniMaxH3Audio`` + the ``minimax_h3`` modular
pipeline at f53d5520), SGLang 0.5.21 and vLLM-Omni all implement the same layout:

* stereo order ``(left, right)`` = batch items ``(0, 1)`` = row blocks ``[0, T)`` and ``[T, 2T)``;
* encode takes the posterior **mean** (``mean_proj``), never a sample; ``logs_proj`` is unused;
* rows are normalised per channel, ``(latent - latents_mean) / latents_std``, in FP32 on the host;
* decode denormalises (``latent * std + mean``), runs BigVGAN and clamps to ``[-1, 1]``.

What this module adds is the strict execution policy. Measured on B200 (see
``benchmarks/models/benchmark_h3_audio_vae.py``), the library paths repeat bit-exactly but are not
batch invariant: a channel's latent depends on how many other clips share its encode call, because
the skinny ``pre_block.proj`` (2048 -> 32) GEMM changes its split with ``M = batch * T`` and, with
TF32 enabled, cuDNN convolutions change algorithm with the batch. The strict policy:

* runs every encode and decode call on **one mono channel** (micro-batch 1), so a channel's bytes
  are a function of its own waveform/latent only (batch invariance by construction);
* disables TF32 for cuBLAS and cuDNN and asks cuDNN for deterministic, non-benchmarked algorithms;
* encodes with cuDNN disabled and the math SDPA backend (the SGLang / vLLM-Omni encode recipe:
  convolution algorithm choice can then not depend on free workspace), and decodes with
  deterministic cuDNN (the SGLang decode recipe; 4x faster than native convolutions for BigVGAN).

The module tree is an op-for-op port of Diffusers f53d5520 with the original checkpoint's
parameter names, so ``model.safetensors`` loads as a passthrough (``strict=True``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from rl_engine.models.minimax_h3.trajectory import tensor_digest

CONTRACT = "h3-audio-vae-v1"
SAMPLE_RATE = 32_000
HOP_LENGTH = 800  # prod(encoder_rates) == prod(decoder_rates)
LATENTS_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 40
LATENT_CHANNELS = 32
STEREO_CHANNELS = 2
STEREO_ORDER = ("left", "right")
VIDEO_FPS = 24
AUDIO_INPUT_HEAD = (LATENT_CHANNELS, 5376)  # transformer ``audio_proj_in``: 32 -> 5376
ENCODER_RATES = (2, 4, 4, 5, 5)
DECODER_RATES = (5, 5, 2, 2, 2, 2, 2)


# --- layout ---------------------------------------------------------------------------------


def audio_latent_num_frames(num_frames: int, fps: float = VIDEO_FPS) -> int:
    """Generated audio latents per stereo channel for a ``num_frames`` video (Diffusers formula)."""

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return int(round(num_frames / fps * LATENTS_PER_SECOND))


def latent_frames_for_samples(num_samples: int) -> int:
    """Latents the encoder emits for ``num_samples`` (right-padded to the 800-sample hop)."""

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    return math.ceil(num_samples / HOP_LENGTH)


def row_channel_and_time(num_latents: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row ``(stereo channel, latent index)`` of the channel-major ``2 * T`` audio rows."""

    t = torch.arange(num_latents, dtype=torch.int64)
    channel = torch.arange(STEREO_CHANNELS, dtype=torch.int64).repeat_interleave(num_latents)
    return channel, t.repeat(STEREO_CHANNELS)


def pack_audio_rows(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    """Raw posterior means ``[2, 32, T]`` -> normalised channel-major rows ``[2 * T, 32]``.

    FP32, on the host.

    Same expression and placement as Diffusers' reference-encode step: transpose to ``[2, T, 32]``,
    subtract the mean, divide by the std, flatten channel-major.
    """

    _check_latents(latents)
    channels = latents.shape[1]
    lat = latents.detach().float().cpu().transpose(1, 2)
    mean = latents_mean.float().cpu().view(1, 1, channels)
    std = latents_std.float().cpu().view(1, 1, channels)
    return ((lat - mean) / std).reshape(-1, channels)


def unpack_audio_rows(rows: torch.Tensor, num_latents: int) -> torch.Tensor:
    """Channel-major rows ``[2 * T, 32]`` -> normalised ``[2, 32, T]`` (Diffusers' unpack step)."""

    if (
        rows.ndim != 2
        or rows.shape[0] != STEREO_CHANNELS * num_latents
        or rows.shape[1] != LATENT_CHANNELS
    ):
        raise ValueError(
            f"rows must be [{STEREO_CHANNELS} * {num_latents}, {LATENT_CHANNELS}], "
            f"got {tuple(rows.shape)}"
        )
    return rows.reshape(STEREO_CHANNELS, num_latents, rows.shape[-1]).permute(0, 2, 1).contiguous()


def validate_latent_stats(
    latents_mean: Sequence[float], latents_std: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``latents_mean`` / ``latents_std``: 32 finite values each, std positive.

    Shared by :meth:`AudioVAECheckpoint.read` and :class:`H3AudioVAEContract`, so a checkpoint
    with bad statistics is rejected before its model is built.
    """

    mean = torch.tensor(list(latents_mean), dtype=torch.float32)
    std = torch.tensor(list(latents_std), dtype=torch.float32)
    for name, stats in (("latents_mean", mean), ("latents_std", std)):
        if stats.shape != (LATENT_CHANNELS,) or not bool(torch.isfinite(stats).all()):
            raise ValueError(
                f"{name} must be {LATENT_CHANNELS} finite FP32 values, got {tuple(stats.shape)}"
            )
    if not bool((std > 0).all()):
        raise ValueError("latents_std must be positive")
    return mean, std


def _check_latents(latents: torch.Tensor) -> None:
    if latents.ndim != 3 or latents.shape[0] != STEREO_CHANNELS:
        raise ValueError(f"latents must be [2, C, T] (left, right), got {tuple(latents.shape)}")
    if latents.dtype != torch.float32:
        raise TypeError(f"latents must be float32, got {latents.dtype}")


# --- pinned preprocessing --------------------------------------------------------------------


def prepare_stereo_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    max_duration: float | None = None,
    allow_resample: bool = False,
) -> torch.Tensor:
    """Pinned reference-audio preprocessing: ``(C, samples)`` -> FP32 ``[2, samples]`` at 32 kHz.

    Mirrors Diffusers' ``_normalize_audio_condition``: cast to FP32, truncate at the *native* rate
    to ``int(max_duration * sample_rate)`` samples, upmix mono by repeating the channel, then
    resample once. Resampling (``torchaudio.transforms.Resample``) is not part of the strict
    contract; it runs only with ``allow_resample=True`` and the caller must record that.
    """

    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, STEREO_CHANNELS):
        raise ValueError(f"waveform must be (1 or 2, samples), got {tuple(waveform.shape)}")
    if not waveform.is_floating_point():
        raise TypeError(f"waveform must be floating point, got {waveform.dtype}")
    waveform = waveform.to(torch.float32)
    if max_duration is not None:
        if not max_duration > 0:
            raise ValueError(f"max_duration must be positive, got {max_duration}")
        waveform = waveform[:, : int(max_duration * sample_rate)]
    if waveform.shape[-1] == 0:
        raise ValueError("waveform is empty after truncation")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform has non-finite samples")
    if waveform.shape[0] != STEREO_CHANNELS:
        waveform = waveform.expand(STEREO_CHANNELS, -1).contiguous()
    if int(sample_rate) == SAMPLE_RATE:
        return waveform
    if not allow_resample:
        raise ValueError(
            f"strict audio preprocessing needs {SAMPLE_RATE} Hz input, got {sample_rate} Hz; "
            "resample upstream or pass allow_resample=True and record it"
        )
    import torchaudio

    # On the CPU, as Diffusers does, whatever the input device: same bytes, and the kernel is a
    # CPU buffer.
    resampled = torchaudio.transforms.Resample(int(sample_rate), SAMPLE_RATE)(waveform.cpu())
    return resampled.to(waveform.device)


# --- architecture (op-for-op port of diffusers f53d5520) --------------------------------------


def _wn(module: nn.Module) -> nn.Module:
    return weight_norm(module)


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Kaiser-windowed sinc low-pass ``[1, 1, k]``; the checkpoint buffer overwrites it."""

    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class LowPassFilter1d(nn.Module):
    filter: torch.Tensor

    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int) -> None:
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )


class UpSample1d(nn.Module):
    filter: torch.Tensor

    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.lowpass = LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(ratio, kernel_size)
        self.downsample = DownSample1d(ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            _wn(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=(6 * dilation) // 2)),
            Snake1d(dim),
            _wn(nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        pad = (x.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, 1),
            ResidualUnit(dim // 2, 3),
            ResidualUnit(dim // 2, 9),
            Snake1d(dim // 2),
            _wn(
                nn.Conv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int, strides: Sequence[int], d_latent: int) -> None:
        super().__init__()
        block: list[nn.Module] = [_wn(nn.Conv1d(1, d_model, kernel_size=7, padding=3))]
        for stride in strides:
            d_model *= 2
            block.append(EncoderBlock(d_model, stride))
        block += [Snake1d(d_model), _wn(nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1))]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)) * self.w1(x))


class CausalAttention(nn.Module):
    """Causal attention; heads are mean-pooled away, then ``head_dim`` avg-pools to ``out_dim``."""

    zero_k_bias: torch.Tensor

    def __init__(self, in_dim: int, out_dim: int, num_heads: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = F.linear(x, self.qkv.weight, torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = (
            qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3, 4)
            .unbind(0)
        )
        # diffusers' native dispatch: [B, S, H, D] -> [B, H, S, D] views, SDPA, and back.
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        out = out.permute(0, 2, 1, 3)
        out = torch.mean(out, dim=2)
        out = F.adaptive_avg_pool1d(out, self.out_dim)
        return self.proj(out)


class AttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, out_dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AMPBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: Sequence[int]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            _wn(
                nn.Conv1d(
                    channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2
                )
            )
            for d in dilation
        )
        self.convs2 = nn.ModuleList(
            _wn(
                nn.Conv1d(
                    channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2
                )
            )
            for _ in dilation
        )
        self.activations = nn.ModuleList(
            Activation1d(SnakeBeta(channels)) for _ in range(2 * len(dilation))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            x = conv2(act2(conv1(act1(x)))) + x
        return x


class BigVGANDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        initial_channels: int,
        rates: Sequence[int],
        kernel_sizes: Sequence[int],
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        resblock_dilations: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(rates)
        self.conv_pre = _wn(nn.Conv1d(in_channels, initial_channels, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(rates, kernel_sizes, strict=True)):
            conv = nn.ConvTranspose1d(
                initial_channels // (2**i),
                initial_channels // (2 ** (i + 1)),
                kernel,
                rate,
                padding=(kernel - rate) // 2,
            )
            self.ups.append(nn.ModuleList([_wn(conv)]))  # ``ups.<i>.0``: checkpoint nesting
        self.resblocks = nn.ModuleList()
        channels = initial_channels
        for i in range(self.num_upsamples):
            channels = initial_channels // (2 ** (i + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilations, strict=True):
                self.resblocks.append(AMPBlock(channels, kernel, dilation))
        self.activation_post = Activation1d(SnakeBeta(channels))
        self.conv_post = _wn(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = cast(nn.ModuleList, self.ups[i])[0](x)
            residual: torch.Tensor | None = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j](x)
                residual = block if residual is None else residual + block
            assert residual is not None
            x = residual / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0)


class H3AudioVAE(nn.Module):
    """Mono MiniMax-H3 audio autoencoder with the original checkpoint's module names."""

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: Sequence[int] = ENCODER_RATES,
        latent_dim: int = 2048,
        latent_channels: int = LATENT_CHANNELS,
        num_heads: int = 8,
        decoder_dim: int = 1024,
        decoder_rates: Sequence[int] = DECODER_RATES,
        decoder_kernel_sizes: Sequence[int] = (9, 9, 4, 4, 4, 4, 4),
    ) -> None:
        super().__init__()
        if math.prod(encoder_rates) != math.prod(decoder_rates):
            raise ValueError("encoder and decoder rates must share one hop length")
        self.hop_length = math.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.pre_block = AttnProjection(latent_dim, latent_channels, num_heads)
        self.mean_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.logs_proj = nn.Conv1d(latent_channels, latent_channels, 1)  # loaded, never evaluated
        self.dec_in_proj = nn.Conv1d(latent_channels, latent_dim, 1)
        self.decoder = BigVGANDecoder(latent_dim, decoder_dim, decoder_rates, decoder_kernel_sizes)

    def encode_mean(self, sample: torch.Tensor) -> torch.Tensor:
        """``[B, 1, samples]`` -> posterior mean ``[B, latent_channels, ceil(samples / hop)]``."""

        return self.encode_mean_native(sample).float()

    def encode_mean_native(self, sample: torch.Tensor) -> torch.Tensor:
        """:meth:`encode_mean` in the module's own dtype (an FP64 reference stays FP64)."""

        if sample.ndim != 3 or sample.shape[1] != 1:
            raise ValueError(f"sample must be [B, 1, samples], got {tuple(sample.shape)}")
        right_pad = (
            math.ceil(sample.shape[-1] / self.hop_length) * self.hop_length - sample.shape[-1]
        )
        if right_pad > 0:
            sample = F.pad(sample, (0, right_pad))
        hidden = self.encoder(sample)
        hidden = self.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
        return self.mean_proj(hidden)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalised ``[B, C, T]`` -> waveform ``[B, 1, T * hop]`` clamped to ``[-1, 1]``."""

        return self.decode_native(latents).float()

    def decode_native(self, latents: torch.Tensor) -> torch.Tensor:
        """:meth:`decode` in the module's own dtype (an FP64 reference stays FP64)."""

        if latents.ndim != 3:
            raise ValueError(f"latents must be [B, C, T], got {tuple(latents.shape)}")
        return self.decoder(self.dec_in_proj(latents))


_WN_KEYS = (
    ("parametrizations.weight.original0", "weight_g"),
    ("parametrizations.weight.original1", "weight_v"),
)


def checkpoint_state_dict(model: H3AudioVAE) -> dict[str, torch.Tensor]:
    """``model.state_dict()`` under the released checkpoint's names (``weight_g`` / ``weight_v``).

    Loading accepts those names directly (torch's weight-norm compatibility hook); this is the
    inverse, so a state dict round-trips between this module, the checkpoint and Diffusers.
    """

    out = {}
    for key, value in model.state_dict().items():
        for new, old in _WN_KEYS:
            if key.endswith(new):
                key = key[: -len(new)] + old
        out[key] = value
    return out


# --- strict execution policy ------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPolicy:
    """Backend pins for one VAE direction. ``micro_batch=None`` runs the whole batch in one call."""

    name: str
    micro_batch: int | None
    cudnn_enabled: bool
    sdpa_backend: str | None  # "math", "efficient" or None (torch default selection)
    allow_tf32: bool = False
    deterministic: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "micro_batch": self.micro_batch,
            "cudnn_enabled": self.cudnn_enabled,
            "cudnn_deterministic": self.deterministic,
            "cudnn_benchmark": False,
            "allow_tf32": self.allow_tf32,
            "sdpa_backend": self.sdpa_backend,
        }


STRICT_ENCODE = ExecutionPolicy(
    "h3-audio-encode-strict-v1", 1, cudnn_enabled=False, sdpa_backend="math"
)
STRICT_DECODE = ExecutionPolicy(
    "h3-audio-decode-strict-v1", 1, cudnn_enabled=True, sdpa_backend=None
)


def _has_fp32_precision_api() -> bool:
    """True on PyTorch builds with the per-backend ``fp32_precision`` TF32 settings."""

    return hasattr(torch.backends.cuda.matmul, "fp32_precision") and hasattr(
        torch.backends.cudnn, "conv"
    )


def _cudnn_op(name: str) -> Any:
    """``torch.backends.cudnn.conv`` / ``.rnn`` (PyTorch >= 2.9; absent from older stubs)."""

    return getattr(torch.backends.cudnn, name)


def _legacy_readable(read: Any) -> bool:
    try:
        read()
    except RuntimeError:  # PyTorch >= 2.9: the caller set this backend through the new API
        return False
    return True


_FP32_PRECISION_NODES = (
    "",
    "cuda.matmul",
    "cudnn",
    "cudnn.conv",
    "cudnn.rnn",
    "mkldnn",
    "mkldnn.matmul",
    "mkldnn.conv",
    "mkldnn.rnn",
)


def _fp32_precision_node(path: str) -> Any:
    node: Any = torch.backends
    for part in filter(None, path.split(".")):
        node = getattr(node, part, None)
    return node if node is not None and hasattr(node, "fp32_precision") else None


def _raw_fp32_precision() -> dict[str, str]:
    """Every backend's own ``fp32_precision`` string, ``"none"`` (inherit) included."""

    if not _has_fp32_precision_api():
        return {}
    nodes = ((path, _fp32_precision_node(path)) for path in _FP32_PRECISION_NODES)
    return {path: node.fp32_precision for path, node in nodes if node is not None}


def _save_tf32_state() -> tuple[Any, Any, dict[str, str]]:
    """The caller's full FP32 precision state for cuBLAS and cuDNN, per backend.

    Each backend is saved, set and restored through the API its current state is readable
    with. The legacy one is used where it can be read (always on PyTorch without
    ``fp32_precision``), and the matmul *precision string* is saved, not ``allow_tf32``:
    ``allow_tf32`` reads True for both "high" and "medium", so restoring it turned "medium"
    into "high" (PyTorch 2.4) or into a mixed state in which
    ``get_float32_matmul_precision()`` raises (PyTorch 2.13). A backend the caller configured
    with the per-backend API (where the legacy getter raises) keeps that API: writing the
    other one would mix the two.

    On PyTorch with ``fp32_precision`` the raw per-backend strings are saved as well: restoring
    the legacy matmul precision writes an explicit value (``"highest"`` -> ``"ieee"``) into
    backends that inherited (``"none"``), which would stop them following their parent after
    the scope. ``_restore_tf32_state`` puts those strings back.
    """

    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if _legacy_readable(torch.get_float32_matmul_precision):
        matmul_state: Any = ("legacy", torch.get_float32_matmul_precision())
    else:
        matmul_state = ("fp32_precision", matmul.fp32_precision)
    if _legacy_readable(lambda: cudnn.allow_tf32):
        cudnn_state: Any = ("legacy", cudnn.allow_tf32)
    else:
        cudnn_state = (
            "fp32_precision",
            _cudnn_op("conv").fp32_precision,
            _cudnn_op("rnn").fp32_precision,
        )
    return matmul_state, cudnn_state, _raw_fp32_precision()


def _set_tf32(state: tuple[Any, Any, dict[str, str]], allow: bool) -> None:
    """Set TF32 on or off for cuBLAS and cuDNN, each through the API ``state`` recorded."""

    matmul_state, cudnn_state, _ = state
    value = "tf32" if allow else "ieee"
    if matmul_state[0] == "legacy":
        torch.backends.cuda.matmul.allow_tf32 = allow
    else:
        torch.backends.cuda.matmul.fp32_precision = value
    if cudnn_state[0] == "legacy":
        torch.backends.cudnn.allow_tf32 = allow
    else:
        _cudnn_op("conv").fp32_precision = value
        _cudnn_op("rnn").fp32_precision = value


def _restore_tf32_state(state: tuple[Any, Any, dict[str, str]]) -> None:
    matmul_state, cudnn_state, raw = state
    if matmul_state[0] == "legacy":
        torch.set_float32_matmul_precision(matmul_state[1])
    else:
        torch.backends.cuda.matmul.fp32_precision = matmul_state[1]
    if cudnn_state[0] == "legacy":
        torch.backends.cudnn.allow_tf32 = cudnn_state[1]
    else:
        _cudnn_op("conv").fp32_precision = cudnn_state[1]
        _cudnn_op("rnn").fp32_precision = cudnn_state[2]
    # a backend that inherited before the scope inherits again (parents first)
    for path, value in raw.items():
        node = _fp32_precision_node(path)
        if node is not None and node.fp32_precision != value:
            node.fp32_precision = value


@contextlib.contextmanager
def execution_scope(policy: ExecutionPolicy) -> Iterator[None]:
    """Apply ``policy``'s backend flags and restore the previous ones on exit (process-global).

    Autocast is disabled for CPU and CUDA inside the scope: the contract is FP32, and an outer
    autocast region would otherwise run the convolutions and linear layers in FP16/BF16 while
    the provenance still reports the strict policy. cuDNN's ``enabled`` / ``benchmark`` /
    ``deterministic`` and the TF32 settings are set and restored here directly, not through
    ``torch.backends.cudnn.flags``: that helper always reads and writes the legacy
    ``allow_tf32`` flag, which raises once a caller has set cuDNN precision with the
    per-backend API.
    """

    from torch.nn.attention import SDPBackend, sdpa_kernel

    cudnn = torch.backends.cudnn
    cudnn_flags = (cudnn.enabled, cudnn.benchmark, cudnn.deterministic)
    tf32_state = _save_tf32_state()
    try:
        _set_tf32(tf32_state, policy.allow_tf32)
        cudnn.enabled = policy.cudnn_enabled
        cudnn.benchmark = False
        cudnn.deterministic = policy.deterministic
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.autocast("cpu", enabled=False))
            if torch.cuda.is_available():
                stack.enter_context(torch.autocast("cuda", enabled=False))
            if policy.sdpa_backend is not None:
                backend = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION}
                stack.enter_context(sdpa_kernel([backend[policy.sdpa_backend]]))
            yield
    finally:
        cudnn.enabled, cudnn.benchmark, cudnn.deterministic = cudnn_flags
        _restore_tf32_state(tf32_state)


def _run_micro_batched(fn: Any, x: torch.Tensor, micro_batch: int | None) -> torch.Tensor:
    if micro_batch is None:
        return fn(x)
    return torch.cat([fn(x[i : i + micro_batch]) for i in range(0, x.shape[0], micro_batch)])


# --- checkpoint ------------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioVAECheckpoint:
    """Config facts of a released ``audio_vae/`` component (Hub layout, not diffusers layout)."""

    path: Path
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None

    @classmethod
    def read(cls, path: str | Path, *, verify_sha256: str | None = None) -> AudioVAECheckpoint:
        root = Path(path)
        config = json.loads((root / "config.json").read_text())
        meta = json.loads((root / config["source_metadata_path"]).read_text())["metadata"]["kwargs"]
        mean, std = tuple(config["latents_mean"]), tuple(config["latents_std"])
        problems = []
        if int(config["sample_rate"]) != SAMPLE_RATE or int(meta["sample_rate"]) != SAMPLE_RATE:
            problems.append(f"sample_rate {config['sample_rate']}/{meta['sample_rate']}")
        if int(config["latent_channels"]) != LATENT_CHANNELS:
            problems.append(f"latent_channels {config['latent_channels']}")
        try:
            validate_latent_stats(mean, std)
        except ValueError as exc:
            problems.append(str(exc))
        if (
            tuple(meta["encoder_rates"]) != ENCODER_RATES
            or tuple(meta["decoder_rates"]) != DECODER_RATES
        ):
            problems.append(f"rates {meta['encoder_rates']}/{meta['decoder_rates']}")
        if not meta.get("attn_proj") or meta.get("decoder_type") != "bigvgan":
            problems.append("attn_proj/decoder_type")
        if problems:
            raise ValueError(f"{root} is not the H3 audio VAE contract: {', '.join(problems)}")
        digest = None
        if verify_sha256 is not None:
            digest = _file_sha256(_weights_path(root))
            if digest != verify_sha256:
                raise ValueError(f"audio VAE weights sha256 {digest} != pinned {verify_sha256}")
        return cls(root, mean, std, dict(meta), digest)

    @property
    def weights_path(self) -> Path:
        """The weight file ``config.json`` names: the one :meth:`read` hashes and :meth:`build`
        loads. Every loader of this component should open this path."""

        return _weights_path(self.path)

    def build(self, device: torch.device | str = "cpu") -> H3AudioVAE:
        """Instantiate the module and load :attr:`weights_path` with ``strict=True`` (FP32)."""

        from safetensors.torch import load_file

        model = H3AudioVAE(
            encoder_dim=int(self.kwargs.get("encoder_dim", 64)),
            latent_dim=int(self.kwargs.get("latent_dim", 2048)),
            decoder_dim=int(self.kwargs.get("decoder_dim", 1024)),
        )
        model.load_state_dict(load_file(str(self.weights_path)), strict=True)
        return model.to(device=device, dtype=torch.float32).eval()


def pinned_audio_vae_sha256(task: str = "FL2VA") -> str:
    """sha256 of ``<task>/audio_vae/model.safetensors`` in ``pinned_checkpoint.json``."""

    pinned = json.loads((Path(__file__).with_name("pinned_checkpoint.json")).read_text())
    for entry in pinned["dirs"][task]:
        if entry["path"] == f"{task}/audio_vae/model.safetensors":
            return str(entry["sha256"])
    raise KeyError(f"no pinned audio VAE weights for {task}")


def _weights_path(root: Path) -> Path:
    config = json.loads((Path(root) / "config.json").read_text())
    return Path(root) / config["source_safetensors_path"]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# --- contract --------------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioEncoding:
    latents: torch.Tensor  # raw posterior means [2, 32, T], FP32, (left, right)
    rows: torch.Tensor  # normalised channel-major rows [2 * T, 32], FP32, host
    num_latents: int
    latents_sha256: str
    rows_sha256: str


@dataclass(frozen=True)
class AudioDecoding:
    waveform: torch.Tensor  # [2, T * 800] FP32 in [-1, 1], (left, right)
    waveform_sha256: str


def _inexact_upsamples(model: nn.Module) -> list[str]:
    """Decoder upsampling layers whose output is not exactly ``stride x`` their input length.

    A ``ConvTranspose1d`` gives ``(L - 1) * stride - 2 * padding + kernel + output_padding``
    samples: ``L * stride`` only when ``kernel - stride == 2 * padding`` and there is no output
    padding. With ``padding = (kernel - stride) // 2`` an odd ``kernel - stride`` (e.g. kernel 8,
    stride 5) adds one sample per layer, which the hop (the stride product) does not show.
    """

    bad = []
    for name, m in model.named_modules():
        if isinstance(m, nn.ConvTranspose1d):
            k, st, pad, out_pad = m.kernel_size[0], m.stride[0], m.padding[0], m.output_padding[0]
            if k - st != 2 * pad or out_pad != 0 or m.dilation[0] != 1:
                bad.append(f"{name} (kernel {k}, stride {st}, padding {pad})")
    return bad


class H3AudioVAEContract:
    """Strict stereo encode/decode around a loaded :class:`H3AudioVAE`.

    ``encode_policy`` / ``decode_policy`` default to the strict policies; the benchmark passes the
    library recipes to measure them on the same module.
    """

    def __init__(
        self,
        model: H3AudioVAE,
        latents_mean: Sequence[float],
        latents_std: Sequence[float],
        *,
        encode_policy: ExecutionPolicy = STRICT_ENCODE,
        decode_policy: ExecutionPolicy = STRICT_DECODE,
    ) -> None:
        dtypes = {p.dtype for p in model.parameters()}
        if dtypes != {torch.float32}:
            raise TypeError(
                f"the H3 audio VAE contract is FP32-only, got parameter dtypes {dtypes}"
            )
        if getattr(model, "hop_length", None) != HOP_LENGTH:
            raise ValueError(
                f"the H3 audio VAE contract needs a {HOP_LENGTH}-sample hop, "
                f"got {getattr(model, 'hop_length', None)}"
            )
        inexact = _inexact_upsamples(model)
        if inexact:
            raise ValueError(
                "the H3 audio VAE contract needs every decoder upsample to give exactly "
                f"rate x length samples (T latents -> T * {HOP_LENGTH}); these do not: {inexact}"
            )
        self.latents_mean, self.latents_std = validate_latent_stats(latents_mean, latents_std)
        self.model = model.eval()
        self.encode_policy = encode_policy
        self.decode_policy = decode_policy

    @classmethod
    def from_checkpoint(
        cls, checkpoint: AudioVAECheckpoint, device: torch.device | str = "cpu", **kwargs: Any
    ) -> H3AudioVAEContract:
        return cls(
            checkpoint.build(device), checkpoint.latents_mean, checkpoint.latents_std, **kwargs
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def encode_latents(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Mono batch ``[B, samples]`` -> raw posterior means ``[B, 32, T]`` (encode policy)."""

        if waveforms.ndim != 2 or waveforms.dtype != torch.float32:
            raise ValueError(
                f"waveforms must be FP32 [B, samples], got {waveforms.dtype} "
                f"{tuple(waveforms.shape)}"
            )
        x = waveforms.to(self.device)[:, None]
        with execution_scope(self.encode_policy):
            return _run_micro_batched(self.model.encode_mean, x, self.encode_policy.micro_batch)

    @torch.inference_mode()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalised ``[B, 32, T]`` -> waveforms ``[B, T * 800]`` under the decode policy."""

        if (
            latents.ndim != 3
            or latents.dtype != torch.float32
            or latents.shape[1] != LATENT_CHANNELS
        ):
            raise ValueError(
                f"latents must be FP32 [B, {LATENT_CHANNELS}, T], got {latents.dtype} "
                f"{tuple(latents.shape)}"
            )
        z = latents.to(self.device)
        with execution_scope(self.decode_policy):
            out = _run_micro_batched(self.model.decode, z, self.decode_policy.micro_batch)
        return out[:, 0]

    def encode_stereo(self, waveform: torch.Tensor) -> AudioEncoding:
        """Prepared 32 kHz ``[2, samples]`` waveform -> latents and channel-major rows."""

        if waveform.ndim != 2 or waveform.shape[0] != STEREO_CHANNELS:
            raise ValueError(
                f"waveform must be [2, samples] (left, right), got {tuple(waveform.shape)}"
            )
        latents = self.encode_latents(waveform).cpu()
        rows = pack_audio_rows(latents, self.latents_mean, self.latents_std)
        return AudioEncoding(
            latents, rows, latents.shape[-1], tensor_digest(latents), tensor_digest(rows)
        )

    def denormalize(self, rows: torch.Tensor, num_latents: int) -> torch.Tensor:
        """Normalised rows -> denormalised ``[2, 32, T]`` on the model device (mul, then add)."""

        z = unpack_audio_rows(rows.float(), num_latents).to(self.device)
        std = self.latents_std.to(self.device).view(1, -1, 1)
        mean = self.latents_mean.to(self.device).view(1, -1, 1)
        return z * std + mean

    def decode_rows(self, rows: torch.Tensor, num_latents: int) -> AudioDecoding:
        """Normalised channel-major rows ``[2 * T, 32]`` -> stereo waveform ``[2, T * 800]``."""

        waveform = self.decode_latents(self.denormalize(rows, num_latents)).cpu()
        return AudioDecoding(waveform, tensor_digest(waveform))

    def provenance(self) -> dict[str, Any]:
        device = self.device
        info: dict[str, Any] = {
            "contract": CONTRACT,
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "latents_per_second": LATENTS_PER_SECOND,
            "latent_channels": LATENT_CHANNELS,
            "stereo_order": list(STEREO_ORDER),
            "row_layout": "channel-major [left T rows | right T rows] x 32",
            "encode_policy": self.encode_policy.as_dict(),
            "decode_policy": self.decode_policy.as_dict(),
            "latents_mean_sha256": tensor_digest(self.latents_mean),
            "latents_std_sha256": tensor_digest(self.latents_std),
            "device": str(device),
            "torch": torch.__version__,
        }
        if device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(device)
            info["capability"] = list(torch.cuda.get_device_capability(device))
            info["cuda"] = torch.version.cuda
            info["cudnn"] = torch.backends.cudnn.version()
        return info
