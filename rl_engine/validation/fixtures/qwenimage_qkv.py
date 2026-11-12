# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Seeded QKV-entry tensors at pinned Qwen-Image image/text/CFG layouts."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rl_engine.reference.model_specific.qwenimage.tp_qkv_bias_gemm import HIDDEN


# Diffusers 40f63690: pipeline_qwenimage.py, pack_latents and encode_prompt.
# Default max_sequence_length=512; the supported upper bound is 1024.
@dataclass(frozen=True)
class QKVWorkload:
    height: int
    width: int
    positive_lengths: tuple[int, ...]
    negative_lengths: tuple[int, ...] = ()
    images_per_prompt: int = 1

    @property
    def image_tokens(self):
        return (self.height // 16) * (self.width // 16)

    @property
    def batch(self):
        return len(self.positive_lengths) * self.images_per_prompt


QKV_WORKLOADS = {
    "image-1024-text17": QKVWorkload(1024, 1024, (17,)),
    "image-1664x928-text129": QKVWorkload(928, 1664, (129,)),
    "image-1328-text512": QKVWorkload(1328, 1328, (512,)),
    "batch2-mixed-text": QKVWorkload(1024, 1024, (17, 512)),
    "two-images-per-prompt": QKVWorkload(1024, 1024, (129,), images_per_prompt=2),
    "batch2-cfg-text1024": QKVWorkload(1328, 1328, (512, 1024), (17, 129)),
}
QKV_LAYOUTS = (
    "batched",
    "sequential_cfg",
    "requests",
    "microbatch",
    "reverse",
    "padding",
    "repeat",
)


def make_qkv_workload(name, dtype, device, seed=386):
    """Construct projected and modulated QKV inputs; no VAE or text encoder runs."""
    spec = QKV_WORKLOADS[name]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).to(device=device)

    batch = spec.batch
    branches = 2 if spec.negative_lengths else 1
    latent_h, latent_w = spec.height // 8, spec.width // 8
    latents = rand((batch, 16, latent_h, latent_w))
    packed = (
        latents.view(batch, 16, latent_h // 2, 2, latent_w // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, spec.image_tokens, 64)
    )
    restored = (
        packed.view(batch, latent_h // 2, latent_w // 2, 16, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape_as(latents)
    )
    assert torch.equal(restored, latents)
    image = F.layer_norm(F.linear(packed, rand((HIDDEN, 64), 64**-0.5)), (HIDDEN,), eps=1e-6)
    length_sets = [spec.positive_lengths]
    if spec.negative_lengths:
        length_sets.append(spec.negative_lengths)
    lengths = torch.tensor(length_sets, device=device).repeat_interleave(spec.images_per_prompt, 1)
    text_width = int(lengths.max().item())
    mask = torch.arange(text_width, device=device)[None, None, :] < lengths[:, :, None]
    encoded = rand((branches, len(spec.positive_lengths), text_width, 3584))
    encoded = encoded.repeat_interleave(spec.images_per_prompt, dim=1)
    encoded = encoded * mask[..., None]
    normalized = encoded * torch.rsqrt(encoded.square().mean(-1, keepdim=True) + 1e-6)
    text = F.layer_norm(
        F.linear(normalized, rand((HIDDEN, 3584), 3584**-0.5), rand((HIDDEN,), 0.1)),
        (HIDDEN,),
        eps=1e-6,
    )
    image = image.unsqueeze(0).expand(branches, -1, -1, -1)
    streams = {}
    for stream, values, valid in (
        (
            "image",
            image,
            torch.ones((branches, batch, spec.image_tokens), device=device, dtype=torch.bool),
        ),
        ("text", text, mask),
    ):
        scale, shift = rand((branches, batch, 1, HIDDEN), 0.1), rand(
            (branches, batch, 1, HIDDEN), 0.1
        )
        x = ((1 + scale) * values + shift).to(dtype)
        grad = rand((*x.shape[:-1], 3 * HIDDEN), (3 * HIDDEN) ** -0.5).to(dtype)
        grad = grad * valid[..., None]
        sample = torch.arange(batch, device=device)[None, :, None].expand_as(valid)
        branch = torch.arange(branches, device=device)[:, None, None].expand_as(valid)
        position = torch.arange(x.shape[2], device=device)[None, None, :].expand_as(valid)
        keys = torch.stack((sample, branch, position), -1).to(torch.int64)
        keys[..., 0] = torch.where(valid, keys[..., 0], -1)
        streams[stream] = {
            "x": x,
            "grad": grad,
            "keys": keys,
            "weight": rand((3 * HIDDEN, HIDDEN), HIDDEN**-0.5).to(dtype),
            "bias": rand((3 * HIDDEN,), 0.1).to(dtype),
        }
    return {
        "streams": streams,
        "metadata": {
            "name": name,
            "height": spec.height,
            "width": spec.width,
            "prompt_batch": len(spec.positive_lengths),
            "images_per_prompt": spec.images_per_prompt,
            "effective_batch": batch,
            "cfg_branches": branches,
            "positive_lengths": list(spec.positive_lengths),
            "negative_lengths": list(spec.negative_lengths),
            "latent_shape": list(latents.shape),
            "packed_shape": list(packed.shape),
            "image_qkv_shape": list(streams["image"]["x"].shape),
            "text_qkv_shape": list(streams["text"]["x"].shape),
        },
    }


def qkv_workload_rows(stream, layout):
    """Return row tensors and forward-call boundaries for the same logical inputs."""
    x, grad, keys = stream["x"], stream["grad"], stream["keys"]
    branches, batch, width = x.shape[:3]
    groups = []
    if layout in ("sequential_cfg", "requests"):
        for branch in range(branches):
            samples = (
                [(0, batch)] if layout == "sequential_cfg" else [(i, i + 1) for i in range(batch)]
            )
            for start, end in samples:
                active = keys[branch, start:end, :, 0] >= 0
                call_width = int(active.sum(-1).max().item())
                groups.append(
                    tuple(
                        v[branch, start:end, :call_width].reshape(-1, v.shape[-1])
                        for v in (x, grad, keys)
                    )
                )
    else:
        groups = [tuple(v.reshape(-1, v.shape[-1]) for v in (x, grad, keys))]
    xs, gs, ks = (
        groups[0] if len(groups) == 1 else [torch.cat([g[i] for g in groups]) for i in range(3)]
    )
    sizes = [g[0].shape[0] for g in groups]
    if layout == "reverse":
        xs, gs, ks = xs.flip(0), gs.flip(0), ks.flip(0)
    elif layout == "padding":
        xs = torch.cat((xs, torch.ones_like(xs[:3])))
        gs = torch.cat((gs, torch.zeros_like(gs[:3])))
        ks = torch.cat((ks, torch.full_like(ks[:3], -1)))
        sizes = [xs.shape[0]]
    elif layout == "microbatch":
        sizes = [xs.shape[0] // 2, xs.shape[0] - xs.shape[0] // 2]
    cuts, start = [], 0
    for size in sizes:
        cuts.append((start, start + size))
        start += size
    return xs, gs, ks, cuts


def canonical_qkv_indices(keys):
    """Map active output/input-gradient rows to sample, CFG branch, token order."""
    order = (keys[:, 0] >= 0).nonzero().flatten()
    for column in range(keys.shape[1] - 1, -1, -1):
        order = order[torch.argsort(keys[order, column], stable=True)]
    return order
