# MiniMax-H3-Image VAE Contract

## Summary

RFC #738 rows `image_vae_contract` and `image_vae_tiling_contract`. They pin
how the released image VAE (`vae_diffusers/` of `MiniMaxAI/MiniMax-H3-Image`,
Diffusers `AutoencoderKLMiniMaxH3Image`) turns an RGB image into the
`(24, H/16, W/16)` latent the transformer consumes, and back into bytes. The
VAE is the MiniMax-H3 causal 3D CNN encoder run over one frame, plus a 24-layer
2D ViT decoder.

The contract covers:

- the checkpoint identity (config, sizes, SHA-256, safetensors header table);
- ImageNet pixel normalization and the output conversion (denormalize, clamp,
  truncate to a canvas-size `uint8` image);
- 24-channel latent normalization;
- the edit-reference recipe (RFC #738 section 3, steps 2-8);
- spatial tiling, with tiled and untiled as distinct runtime identities;
- execution flags, autocast and batch handling.

Every latent and every output stage leaves with a SHA-256 digest. The network
itself is reused from Diffusers at commit `875373ac`, unmodified. The code is
in `rl_engine/models/minimax_h3_image/image_vae.py`. The static contract
(identity, tile plans, recipes) is in `image_vae_contract.json` next to it.

Contract ids: `h3-image-vae-f16d24-v1` (encode/decode) and
`h3-image-vae-tiles-v1` (tiling).

## Reuse from MiniMax-H3

| Part | Source | Why it is the same |
| --- | --- | --- |
| Tile placement, plan digest | `minimax_h3.visual_vae.split_tiles` / `plan_tiles` / `TilePlan` | `AutoencoderKLMiniMaxH3Image._split_tiles`, `_blend` and `_stitch_tiles` are marked "Copied from" the H3 VAE at `875373ac`; a test checks every length from 16 to 4096 px |
| Pixel table | `minimax_h3.visual_vae.pixel_lut` | The image pipeline normalizes with the same `uint8.to(float32).div(255.0)`, mean and std as H3 |
| Runtime flags | `minimax_h3.visual_vae.pinned_runtime` | Same flags and the same process-wide lock |
| Digests | `minimax_h3.visual_vae.output_digests` | Same `dtype|shape|bytes` SHA-256 as `trajectory.tensor_digest` |

The image-specific parts are written here: 4-D tensors, the output
conversion, the reference crops and generator handling, the image tiling
checks, and the FP32-only mode check.

## Geometry

| Item | Value |
| --- | --- |
| Latent | `(B, 24, H/16, W/16)`, FP32 |
| Generation canvas | `H`, `W` multiples of 32 (VAE 16 x transformer patch 2); `decode` refuses others |
| Reference | any size; cropped at the top left to whole 16 px cells, then the latent to whole 2x2 patches |

The reference crops are real in the release path, not no-ops. A reference
whose short edge is already within `[1024, 2048]` keeps its own size
(`bound_short_edge`), so 1100x1250 pixels are encoded as 1088x1248, and a
1040x1072 reference gives a 65x67 latent that is cropped to 64x66.

## Tiling

Encode and decode run in 256 px tiles with at least 64 px of overlap. Tiling is
on by default in the release, and the released images are the tiled ones.

1. The plan uses the fewest tiles whose overlaps all reach the minimum.
2. It spreads the slack round-robin over the overlaps, in 16 px steps.
3. Encode blends latents over `overlap / 16`; decode blends output pixels over
   `overlap`. The blend is `a * (1 - i/n) + b * (i/n)` in the tile dtype: first
   with the tile above, then with the tile to the left.
4. Every non-last tile drops its trailing overlap before concatenation.

The release canvas 2048x2048 is 11 x 11 = 121 tiles with row and column starts
`0, 176, ..., 1600, 1792`, overlaps `80 x 8, 64, 64`.

`plan_tiles` fails closed where Diffusers would not:

- a zero overlap: `_stitch_tiles` crops with `tile[..., :-overlap]`, which for
  0 empties the tile, and the next blend fails;
- tile sizes or overlaps off the 16 px grid, an overlap not smaller than the
  tile, an image off the 16 px grid, or a plan that does not end on the edge.

The tiling identity (`record["tiling"]`) holds the mode (`tiled-256x256-o64x64`
or `untiled`), the plan digest, starts, sizes, overlaps, and the split, blend
and boundary rules. `ImageVAEContract(vae, tiled=...)` declares the mode, and
every call checks that the VAE is still in it.

## Pixel normalization and output conversion

Encode input is `uint8` RGB, gathered from the `[3, 256]` FP32 table that
freezes the CUDA semantics of `(x.float().div(255.0) - mean) / std`. On CPU,
PyTorch divides truly, so the release formula differs between CPU and CUDA; the
table gives the CUDA (release) values on both.

Decode output goes through two stages, each with its own digest:

| Stage | Formula | Digest key |
| --- | --- | --- |
| `unit` | `(x.float() * std + mean).clamp(0, 1)` | `unit` |
| `pixels` | `(unit * 255).clamp(0, 255).to(uint8)`: truncation, not rounding | `uint8` |

The pipeline then passes `q / 255` to `VaeImageProcessor`, which rounds
`x * 255` back to `q`, so the returned bytes are the truncated ones. Neither
stage is part of latent equality.

The contract ends at the canvas-size `uint8` image. Resizing it to the
requested output size (the decode step's PIL branch, taken when the canvas
differs from the request) belongs to `image_canvas_contract`
(`rl_engine/models/minimax_h3_image/canvas.py`).

## Latent normalization

| Direction | Formula | Kernels |
| --- | --- | --- |
| Normalize | `(z - mean) / std` | FP32 subtract, then FP32 true division |
| Denormalize | `z * std + mean` | FP32 multiply, then FP32 add (no FMA) |

`mean` and `std` are the released 24-entry `latents_mean` / `latents_std`. The
fingerprint proves every entry is FP32-representable. Both run on the VAE
device, as in the pipeline.

## Edit-reference recipe

`encode` (and `encode_references`, one call per reference) runs:

1. crop the pixels at the top left to whole 16 px cells;
2. normalize by the table;
3. encode with the FP32 VAE, autocast disabled;
4. `posterior.sample(generator)` from the request generator, one draw per
   reference, in reference order;
5. round to FP16 and back to FP32;
6. crop the latent at the top left to whole 2x2 patches;
7. `(z - mean) / std`.

The record holds the generator device, its initial seed, and the SHA-256 of its
state before and after every draw, so the draw order is sealed. The RFC's
steps 9-11 (reference noise, the `t = 0.999` mix, patchify) belong to
`image_reference_conditioning`.

## Execution and batch handling

`ImageVAEContract` runs every call under `pinned_runtime()` (cuDNN
deterministic, no benchmark, cuDNN TF32 on, matmul TF32 off), with any ambient
autocast disabled: the image pipeline encodes and decodes without autocast. It
runs one batch row at a time, so a row never depends on its batch neighbours;
with a shared generator, row `i` equals the same image encoded alone after the
draws of rows `0..i-1`. Diffusers' `use_slicing` must stay off.

Before every call the contract also refuses a VAE with any non-FP32 parameter.
The README ships the VAE as FP32, but the Diffusers docs load components with
`dtype=torch.bfloat16`, which (through `_keep_in_fp32_modules`) puts the
decoder blocks in BF16. That is another mode, and this contract does not
define it.

## Use

```python
from diffusers import AutoencoderKLMiniMaxH3Image
from rl_engine.models.minimax_h3_image import image_vae as iv

vae = iv.load_release_vae(AutoencoderKLMiniMaxH3Image, "<snapshot>/vae_diffusers", "cuda")
contract = iv.ImageVAEContract(vae)                 # release config, stats and tiles
refs = contract.encode_references(images_uint8, generator)   # one VAEOutput per reference
out = contract.decode(latents)                      # unit / uint8 + digests
print(out.digests, out.record["tiling"]["mode"], out.record["tiling"]["plan_sha256"])
```

## Validation

```bash
# Self-contained tests; with Diffusers 875373ac also the small random VAE (CPU, + CUDA)
PYTHONPATH=. python -m pytest tests/models/minimax_h3_image -q
# With the released weights (CUDA)
RL_KERNEL_H3_IMAGE_VAE_DIR=<snapshot>/vae_diffusers PYTHONPATH=. python -m pytest tests/models/minimax_h3_image -q
# Golden contract file, evidence report and charts
python tools/validation/models/h3_image_vae_evidence.py --check-manifest
PYTHONPATH=. python tools/validation/models/h3_image_vae_evidence.py \
    --vae-dir <snapshot>/vae_diffusers --out evidence.json --verify-sha256
python tools/validation/models/plot_h3_image_vae.py evidence.json
```

## Known limitations

- Run-to-run and batch-row equality and bitwise Diffusers parity are claimed
  on one device type (CUDA); encoder and decoder outputs are not claimed equal
  across GPU generations or between CPU and CUDA. Only pixel and latent
  normalization are device-independent.
- Validated on B200 (SM100). H100 and ROCm are follow-ups.
- FP32 only; the BF16 decoder-block mode is not defined here.
- No backward pass: the VAE is outside the RL gradient path.
