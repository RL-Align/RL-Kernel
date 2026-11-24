# SPDX-License-Identifier: Apache-2.0
"""INK-62: frozen vision encoder adapter with a portable math attention path."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def rotary_2d(x, positions):
    """Two-axis rotary layout: x coordinate first, then y coordinate."""
    dim = x.shape[-1]
    if dim % 4 or positions.shape != (x.shape[0], 2):
        raise ValueError("invalid 2-D rotary geometry")
    inv = 10000.0 ** (-torch.arange(0, dim, 4, device=x.device).float() / dim)
    angles = torch.cat((positions[:, 1:2].float() * inv,
                        positions[:, 0:1].float() * inv), dim=-1)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    even, odd = x.float()[..., ::2], x.float()[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos),
                       dim=-1).flatten(-2).to(x.dtype)


def vision_attention(attn, hidden, positions, lengths):
    """Execute the actual loaded projection modules, isolating each image."""
    heads, dim = attn.num_heads, attn.head_dim
    q, k, v = (p(hidden).reshape(-1, heads, dim)
               for p in (attn.q_proj, attn.k_proj, attn.v_proj))
    q, k = rotary_2d(q, positions), rotary_2d(k, positions)
    if sum(lengths) != q.shape[0] or any(length <= 0 for length in lengths):
        raise ValueError("invalid image sequence lengths")
    outputs = []
    with sdpa_kernel(SDPBackend.MATH):
        for qi, ki, vi in zip(q.split(lengths), k.split(lengths),
                             v.split(lengths), strict=True):
            y = F.scaled_dot_product_attention(
                qi.transpose(0, 1)[None], ki.transpose(0, 1)[None],
                vi.transpose(0, 1)[None], dropout_p=0.0, is_causal=False,
                scale=dim**-0.5,
            )
            outputs.append(y[0].transpose(0, 1))
    return attn.out_proj(torch.cat(outputs).reshape_as(hidden))


def _position_grid(grid, positions, heights, widths):
    def weights(indices, target, size):
        center = (size / target.float()) * (indices.float() + .5)
        cells = torch.arange(size, device=grid.device).float() + .5
        w = (1 - (cells[None] - center[:, None]).abs()).clamp(min=0)
        return w / w.sum(-1, keepdim=True)

    wy = weights(positions[:, 0], heights, grid.shape[0])
    wx = weights(positions[:, 1], widths, grid.shape[1])
    return torch.einsum("ni,nj,ijd->nd", wy, wx, grid.float()).to(grid.dtype)


@torch.no_grad()
def frozen_features(encoder, patches, grid_thw, *, platform):
    """Use loaded SGLang-compatible ViT modules without its CUDA-only FA4 path.

    Caller owns checkpoint loading and exact processor inputs. This is a
    correctness candidate, not a trainable ViT or integrated Miles actor.
    """
    if platform not in ("cuda", "rocm") or not patches.is_cuda:
        raise RuntimeError("frozen feature capture requires CUDA or ROCm")
    if (torch.version.hip is not None) != (platform == "rocm"):
        raise RuntimeError("platform mismatch")
    if platform == "rocm":
        arch = torch.cuda.get_device_properties(patches.device).gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise RuntimeError("initial ROCm vision profile requires gfx950")
    if (len(encoder.backbone.layers) != 23 or encoder.patch_size != 20
            or encoder.merge_size != 2 or encoder.text_hidden_size != 2304):
        raise ValueError("expected released 23-layer Mini vision geometry")
    if any(p.requires_grad for p in encoder.parameters()) or encoder.training:
        raise ValueError("encoder must be frozen and in eval mode")
    side = encoder.patch_size * encoder.merge_size
    if patches.dtype != torch.bfloat16 or patches.ndim != 5 or tuple(patches.shape[1:]) != (
        encoder.temporal_patch_size, side, side, 3
    ):
        raise ValueError("invalid processor patch layout")
    if (grid_thw.device.type != "cpu" or grid_thw.ndim != 2 or grid_thw.shape[1] != 3
            or grid_thw.dtype not in (torch.int32, torch.int64)):
        raise ValueError("expected CPU integer image grids")
    grids = grid_thw.tolist()
    if not grids or any(t != 1 or h <= 0 or w <= 0 for t, h, w in grids):
        raise ValueError("expected nonempty image grids with temporal size one")
    if sum(h * w for _, h, w in grids) != patches.shape[0]:
        raise ValueError("image patch count mismatch")
    if any(p.device != patches.device for p in encoder.parameters()):
        raise ValueError("encoder weights must share patch device")
    m, p = encoder.merge_size, encoder.patch_size
    rgb = patches * encoder._clip_std.to(patches.dtype) + encoder._clip_mean.to(patches.dtype)
    pixels = (2.0 * rgb - 1.0).mean(1).permute(0, 3, 1, 2).contiguous()
    pixels = pixels.to(encoder.proj_out.weight.dtype)
    native, coords, heights, widths, lengths = [], [], [], [], []
    for image, (_, h, w) in zip(pixels.split([h*w for _, h, w in grids]), grids, strict=True):
        image = image.reshape(h, w, 3, m, p, m, p)
        image = image.permute(0, 3, 1, 5, 4, 6, 2).reshape(-1, 3*p*p)
        n = h*w*m*m
        rows = torch.arange(n, device=patches.device)
        xy = torch.stack((rows // (w*m), rows % (w*m)), -1)
        native.append(torch.cat((image.new_zeros(1, image.shape[1]), image)))
        coords.append(torch.cat((xy.new_zeros(1, 2), xy)))
        heights.append(rows.new_full((n+1,), h*m))
        widths.append(rows.new_full((n+1,), w*m))
        lengths.append(n+1)
    e = encoder.backbone.embeddings
    coords = torch.cat(coords)
    pos_size = encoder.config["posemb_grid_size"]
    hidden = e(torch.cat(native))
    hidden = hidden + _position_grid(e.position_embedding.reshape(pos_size, pos_size, -1),
                                     coords, torch.cat(heights), torch.cat(widths))
    cls = torch.tensor([0, *lengths[:-1]], device=patches.device).cumsum(0)
    hidden[cls] = e.class_embedding + e.cls_position_embedding
    positions = (coords + 1).to(torch.int32)
    positions[cls] = 0
    hidden = encoder.backbone.ln_pre(hidden)
    for layer in encoder.backbone.layers:
        hidden = hidden + layer.ls_1(vision_attention(
            layer.self_attn, layer.ln_1(hidden), positions, lengths))
        hidden = hidden + layer.ls_2(layer.mlp(layer.ln_2(hidden)))
    pooled = encoder._pool(hidden, grids, lengths)
    return encoder.proj_out(F.gelu(encoder.proj_in(pooled)))
