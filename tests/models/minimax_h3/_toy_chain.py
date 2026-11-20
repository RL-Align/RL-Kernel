# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Small stand-ins for the ``phase_b_full_chain`` tests (pure PyTorch, CPU, seconds).

* :class:`ToyVisualVAE` has the release ``f16t4d24`` layout (so the ``VisualVAEContract``
  frame map, normalization and FP16 rounding run for real) but a 1x1 network.
* :class:`ToyH3Transformer` has the ``MiniMaxH3Transformer3DModel`` call signature and module
  names (input projections, refiner, time embedder, a block list with full attention over the
  packed rows, AdaLN-style row modulation, output heads), so the probe sees the same paths.
* :class:`SyntheticTextConditioner` derives features from the conditioner input digest, with
  a vision block per visual medium, like the released presentation.
* :func:`tiny_audio_contract` is the sibling row's small ``H3AudioVAE`` under the strict policy.

The requests use a 64x64 canvas and the shortest legal duration (124 frames, 37 latent frames).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from rl_engine.models.minimax_h3 import audio_vae as av
from rl_engine.models.minimax_h3 import full_chain as fc
from rl_engine.models.minimax_h3 import ref2va
from rl_engine.models.minimax_h3 import visual_vae as vv

TEXT_DIM = 16
TEXT_SOURCE = "synthetic (tests)"
CANVAS = {"canvas_short_edge": 64, "canvas_max_pixels": 64 * 128}


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --- visual VAE ---------------------------------------------------------------------------


class _Posterior:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        self.mean, self.logvar = mean, logvar

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, generator=generator, dtype=self.mean.dtype)
        return self.mean + torch.exp(0.5 * self.logvar) * noise.to(self.mean.device)

    def mode(self) -> torch.Tensor:
        return self.mean


class ToyVisualVAE(nn.Module):
    """Release layout, 1x1 channel maps, nearest temporal and box spatial resampling."""

    use_tiling = False
    use_slicing = False

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        mean, std = vv.release_latent_stats()
        self.config = {
            "latent_channels": vv.LATENT_CHANNELS,
            "spatial_downsample_factors": [2, 2, 2, 2],
            "temporal_downsample_factors": [2, 2],
            "clip_length": vv.CLIP_LENGTH,
            "token_drop": vv.TOKEN_DROP,
            "latents_mean": mean.tolist(),
            "latents_std": std.tolist(),
        }
        self.layout = vv.VisualVAELayout.from_config(self.config)
        g = _gen(seed)
        self.enc = nn.Parameter(torch.randn(2 * vv.LATENT_CHANNELS, 3, generator=g))
        self.dec = nn.Parameter(torch.randn(3, vv.LATENT_CHANNELS, generator=g) * 0.2)

    def encode(self, x: torch.Tensor, return_dict: bool = False) -> tuple[_Posterior]:
        frames = x.shape[2]
        latent_frames = self.layout.latent_frames(frames)
        index = [min(frames - 1, j * frames // latent_frames) for j in range(latent_frames)]
        x = x[:, :, index]
        b, c, t, h, w = x.shape
        pooled = nn.functional.avg_pool2d(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), 16)
        pooled = pooled.reshape(b, t, c, h // 16, w // 16).permute(0, 2, 1, 3, 4)
        moments = torch.einsum("oc,bcthw->bothw", self.enc, pooled)
        mean, logvar = moments.chunk(2, dim=1)
        return (_Posterior(mean, logvar.clamp(-6.0, -1.0)),)

    def decode(self, z: torch.Tensor, return_dict: bool = False) -> tuple[torch.Tensor]:
        latent_frames = z.shape[2]
        frames = self.layout.pixel_frames(latent_frames)
        index = [min(latent_frames - 1, f * latent_frames // frames) for f in range(frames)]
        rgb = torch.einsum("oc,bcthw->bothw", self.dec, z[:, :, index])
        rgb = rgb.repeat_interleave(16, dim=3).repeat_interleave(16, dim=4)
        return (torch.tanh(rgb),)


def toy_visual_contract(device: str = "cpu") -> vv.VisualVAEContract:
    return vv.VisualVAEContract(ToyVisualVAE().to(device), require_release=False)


# --- audio VAE ----------------------------------------------------------------------------


def tiny_audio_contract(device: str = "cpu") -> av.H3AudioVAEContract:
    torch.manual_seed(0)
    model = av.H3AudioVAE(encoder_dim=4, latent_dim=64, decoder_dim=128)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("alpha", "beta")):
                p.uniform_(0.2, 1.5) if p.ndim == 3 else p.uniform_(-0.5, 0.5)
    mean = [0.1 * i - 1.0 for i in range(av.LATENT_CHANNELS)]
    std = [0.5 + 0.05 * i for i in range(av.LATENT_CHANNELS)]
    return av.H3AudioVAEContract(model.to(device).eval(), mean, std)


# --- transformer --------------------------------------------------------------------------


class ToyBlock(nn.Module):
    def __init__(self, hidden: int, g: torch.Generator) -> None:
        super().__init__()
        scale = hidden**-0.5
        self.wq = nn.Parameter(torch.randn(hidden, hidden, generator=g) * scale)
        self.wk = nn.Parameter(torch.randn(hidden, hidden, generator=g) * scale)
        self.wv = nn.Parameter(torch.randn(hidden, hidden, generator=g) * scale)
        self.w1 = nn.Parameter(torch.randn(hidden, 2 * hidden, generator=g) * scale)
        self.w2 = nn.Parameter(torch.randn(2 * hidden, hidden, generator=g) * scale * 0.5)
        self.modulation = nn.Parameter(torch.randn(3, hidden, 2 * hidden, generator=g) * scale)

    def forward(self, hidden: torch.Tensor, temb: torch.Tensor, adaln: torch.Tensor):
        table = torch.einsum("td,mde->tme", temb, self.modulation).reshape(-1, 2 * hidden.shape[-1])
        shift, gate = table[adaln].chunk(2, dim=-1)  # one AdaLN row per (timestep, modality)
        x = hidden[0] + shift
        attn = torch.softmax((x @ self.wq) @ (x @ self.wk).T / math.sqrt(x.shape[-1]), dim=-1)
        x = x + torch.tanh(gate) * (attn @ (x @ self.wv))
        x = x + torch.tanh(x @ self.w1) @ self.w2
        return x[None]


class _NormOut(nn.Module):
    def __init__(self, hidden: int, g: torch.Generator) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.randn(hidden, hidden, generator=g) * hidden**-0.5)

    def forward(self, hidden: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor):
        x = hidden / hidden.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
        return x * (1 + (temb @ self.scale)[timestep_indices])


class _Config(dict):
    """A mapping with attribute access, like Diffusers' ``FrozenDict`` config."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ToyH3Transformer(nn.Module):
    """``MiniMaxH3Transformer3DModel``'s signature and module names at toy width, FP32."""

    def __init__(
        self, hidden: int = 32, layers: int = 3, seed: int = 0, text_dim: int = TEXT_DIM
    ) -> None:
        super().__init__()
        g = _gen(seed)
        torch.manual_seed(seed)
        self.config = _Config(toy=True, hidden_size=hidden, num_layers=layers)
        self.proj_in = nn.Linear(vv.LATENT_CHANNELS * 4, hidden)
        self.audio_proj_in = nn.Linear(av.LATENT_CHANNELS, hidden)
        self.context_embedder = nn.Linear(text_dim, hidden)
        self.token_refiner = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.time_embedder = nn.Linear(16, hidden)
        self.transformer_blocks = nn.ModuleList([ToyBlock(hidden, g) for _ in range(layers)])
        self.norm_out = _NormOut(hidden, g)
        self.proj_out = nn.Linear(hidden, vv.LATENT_CHANNELS * 4)
        self.audio_proj_out = nn.Linear(hidden, av.LATENT_CHANNELS)
        self.calls = 0

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
    ):
        self.calls += 1
        video = self.proj_in(hidden_states.float())
        audio = self.audio_proj_in(audio_hidden_states.float())
        text = self.token_refiner(self.context_embedder(encoder_hidden_states.float()))
        hidden = text.new_zeros(1, position_ids.shape[0], text.shape[-1])
        hidden = hidden.index_copy(1, text_indices, text)
        hidden = hidden.index_copy(1, video_indices, video)
        hidden = hidden.index_copy(1, audio_indices, audio)
        freqs = torch.arange(8, dtype=torch.float32, device=timestep.device)
        angles = timestep[:, None].float() * 1000.0 * torch.exp(-freqs)
        temb = self.time_embedder(torch.cat([angles.sin(), angles.cos()], dim=-1))
        hidden = hidden + 0.01 * torch.sin(position_ids.float().sum(-1))[None, :, None]
        adaln = timestep_indices * 3 + token_tags
        for block in self.transformer_blocks:
            hidden = block(hidden, temb, adaln)
        hidden = self.norm_out(hidden, temb, timestep_indices)
        return (
            self.proj_out(hidden).index_select(1, video_indices),
            self.audio_proj_out(hidden).index_select(1, audio_indices),
        )


# --- conditioner --------------------------------------------------------------------------


class SyntheticTextConditioner:
    """Features from the conditioner-input digest: 6 text rows plus a vision block per visual."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.dtype = dtype

    def encode(self, request: fc.TextRequest) -> fc.TextConditions:
        seed = int(request.input_sha256[:12], 16)
        tags: list[int] = []
        for kind in request.media_kinds:
            if "audio" in kind or "waveform" in kind:
                tags += [1, 1]
            else:
                tags += [1, 1] + [0] * 3
        tags += [1] * 6
        embeds = torch.randn(1, len(tags), TEXT_DIM, generator=_gen(seed)).to(self.dtype)
        return fc.TextConditions(
            embeds, torch.tensor(tags, dtype=torch.long), {"source": TEXT_SOURCE}
        )


# --- requests -----------------------------------------------------------------------------


def _image(seed: int, height: int, width: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def fl2va_request(steps: int = 4) -> fc.FL2VARequest:
    return fc.FL2VARequest(
        prompt="a toy fl2va request",
        image=_image(1, 80, 48),
        last_image=_image(2, 64, 96),
        height=64,
        width=64,
        num_frames=124,
        num_inference_steps=steps,
        seed=7,
        **CANVAS,
    )


def ref2va_request(steps: int = 4) -> fc.Ref2VARequest:
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, (30, 36, 64, 3), dtype=np.uint8)
    soundtrack = 0.2 * torch.randn(1, 40_000, generator=_gen(4))
    audio = 0.2 * torch.randn(2, 20_000, generator=_gen(5))
    return fc.Ref2VARequest(
        prompt="a toy ref2va request",
        references=(
            ref2va.ImageReference(np.asarray(_image(6, 40, 72))),
            ref2va.VideoReference(frames, fps=24.0, audio=soundtrack, sample_rate=32_000),
            ref2va.AudioReference(audio, sample_rate=32_000),
        ),
        height=64,
        width=64,
        num_frames=124,
        num_inference_steps=steps,
        seed=11,
        image_short_edge=64,
        **CANVAS,
    )


def components(variant: str, device: str = "cpu", **kwargs: Any) -> fc.ChainComponents:
    model = ToyH3Transformer(text_dim=kwargs.pop("text_dim", TEXT_DIM)).to(device)
    return fc.ChainComponents(
        text=kwargs.pop("text", SyntheticTextConditioner()),
        denoiser=fc.TransformerDenoiser(model, variant=kwargs.pop("denoiser_variant", variant)),
        visual_vae=toy_visual_contract(device),
        audio_vae=tiny_audio_contract(device),
        device=device,
    )
