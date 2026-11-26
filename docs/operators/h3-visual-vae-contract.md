# MiniMax-H3 VisualVAE Contract

## Summary

RFC #420 Phase B row `visual_vae_contract`. It pins how the released MiniMax-H3
VisualVAE (`vae/` of `MiniMaxAI/MiniMax-H3`, Diffusers `AutoencoderKLMiniMaxH3`)
turns RGB video into the `f16t4d24` latent the Omni-Transformer consumes, and
back. The contract covers these parts:

- the layout: 24 channels, spatial factor 16, temporal factor 4;
- the frame map and the decode chunking;
- spatial tiling;
- pixel preprocessing;
- latent normalization;
- execution flags and batch handling.

Every latent leaves with a SHA-256 digest. Two encodes of the same `uint8`
pixels therefore either have the same digest or prove a drift.

The network itself is reused from Diffusers at commit `f53d5520`, unmodified;
see [Reuse decision](#reuse-decision). The code is in
`rl_engine/models/minimax_h3/visual_vae.py`, and the pinned identity is in
`visual_vae_fingerprint.json` next to it.

Contract id: `h3-visual-vae-f16t4d24-v1`.

## Layout

| Item | Value |
| --- | --- |
| Latent | `(B, 24, T_lat, H/16, W/16)`, FP32 |
| Encode clip | 17 pixel frames, 3 trailing latent frames dropped |
| Frame map | `17n+5 -> 5n+2`; a single frame -> 1, through the spatial encoder only |
| Decode | 5-latent chunks with a 2-latent overlap, cross-faded over 5 pixel frames, inverse of the frame map |
| Spatial | `H` and `W` must be multiples of 16 |

`VisualVAELayout` derives every count with integer arithmetic. `pixel_frames()`
mirrors `_decode` and is checked against the Diffusers decoder for 3 to 17
latent frames. A count of 1 or 2 latent frames cannot be decoded, because it
gives no full decode chunk.

## Tiling

The release encodes and decodes in tiles of 256 px with at least 64 px of
overlap, and tiling is on by default. `plan_tiles(h, w)` reproduces
`_split_tiles`:

1. It uses the fewest tiles whose overlaps all reach the minimum.
2. It spreads the slack round-robin over the overlaps, in 16 px steps.

The result covers the frame exactly. Encode blends latents over `overlap/16`;
decode blends pixels over `overlap`. The blend is `a*(1-p/n) + b*(p/n)`, in
the dtype of the tile. The plan has a digest, and every contract record carries
it (`tile_plan_sha256`).

Tiling changes the output: tiled and untiled latents differ. Tiling is
therefore part of the release identity, not an optimisation. The contract
refuses a VAE whose tile geometry differs from the release.

## Pinned preprocessing

The released pipeline computes `pixels.to(float32).div(255.0)` on the VAE device.
PyTorch performs a tensor-by-Python-scalar division differently on each device:

- **CPU:** a true division.
- **CUDA:** a multiply by the FP32 reciprocal.

The two disagree for 126 of the 256 `uint8` values. After ImageNet
normalization, 322 of the 768 per-channel values disagree.

`pixel_lut()` freezes the CUDA (release) semantics in a `[3, 256]` FP32 table.
Each entry is `((v * fl32(1/255)) - mean[c]) / std[c]`, with every step rounded
to FP32. `preprocess_pixels()` gathers from the table, which is a copy with no
arithmetic, so the result is identical on every device. The table is checked
bit for bit against the Diffusers op sequence on CUDA.

## Latent normalization

| Direction | Formula | Kernels |
| --- | --- | --- |
| Normalize | `(z - mean) / std` | FP32 subtract, then FP32 true division |
| Denormalize | `z * std + mean` | FP32 multiply, then FP32 add (separate kernels, no FMA) |

`mean` and `std` are the released `latents_mean` / `latents_std`, rounded from
the JSON doubles to FP32 exactly as `torch.tensor(config.latents_mean)` does.
They must be `[24]` FP32 tensors with `requires_grad=False`.

H3 divides by `std`, whereas Qwen-Image's `latent_normalize` (#577) multiplies
by `std_inv`. The two are not interchangeable bit for bit, and a test asserts
this. Normalization and denormalization are bitwise equal on CPU and CUDA.

The encode recipe for a condition (keyframe or Ref2VA reference) is the released
one:

1. Sample the posterior under a fresh CPU `torch.Generator` seeded 42, separately
   for every row.
2. Round the sample to FP16.
3. Normalize on the CPU.

`posterior="mode"` is available for callers that want the mean.

The decode follows the released decode step:

1. Denormalize on the device.
2. Decode under FP16 autocast over the FP32 weights.
3. Compute `x * pixel_std + pixel_mean` and clamp to `[0, 1]`.

## Execution and batch invariance

`VisualVAEContract` runs every call under `pinned_runtime()` with these flags,
and records a readback of them:

- cuDNN deterministic on;
- cuDNN benchmark off;
- cuDNN TF32 on (the PyTorch default, as in the release);
- matmul TF32 off.

It runs each batch row on its own. The RFC defines batch as a pure replication
axis, and the batched Diffusers encode is not batch-invariant; see below.
Diffusers' `use_slicing` must stay off, because the contract slices rows itself.

## Reuse decision

Three implementations exist. Each was surveyed at the commit shown:

| Library | Implementation | Bitwise relation to the Diffusers reference | Verdict |
| --- | --- | --- | --- |
| Diffusers `f53d5520` | `AutoencoderKLMiniMaxH3`, modular pipeline `encoders.py` / `decoders.py` | Is the reference (RFC #420 "initial reference runtime") | Reused, per row, under pinned flags |
| SGLang 0.5.21 | `runtime/models/vaes/minimax_h3_video_vae/` | Decoder differs by construction (see below) | Not reusable as the reference; not measured |
| vLLM-Omni `aae3f386` | Runs the checkpoint's own `trust_remote_code` VAE, plus fused Triton Q/K-norm/RoPE and residual kernels on SM90/SM100 | Same decoder family as SGLang | Not reusable as the reference |

SGLang's decoder differs from Diffusers in four places:

- It builds the RoPE positions in FP16.
- It runs `proj_in` / `proj_out` in FP32, so its tile and temporal blends run
  in FP32 rather than FP16.
- It denormalizes pixels as `(x - (-m/s)) / (1/s)`.
- It uses a fused QKV GEMM.

Neither SGLang nor vLLM-Omni pins cuDNN flags for the video encode.

Measured on the released weights on B200, the batched Diffusers encode
(`use_slicing=False`) is not batch-invariant. At batch 2 and 3, every row
differs from the same row run alone, in nearly every element. The maximum
difference is about 0.02, and it is the same with cuDNN deterministic set. The
profiled encoder convolutions are cuDNN TF32 implicit GEMMs, whose kernel
choice depends on the batch size. The batched decode was batch-invariant in the
same run.

The contract therefore reuses the Diffusers network and closes the gaps around
it: per-row execution, pinned flags, the preprocessing table and the release
checks. It writes no new kernel.

## Use

```python
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from rl_engine.models.minimax_h3 import visual_vae as vv

vae = vv.load_release_vae(AutoencoderKLMiniMaxH3, "<snapshot>/vae", "cuda")
contract = vv.VisualVAEContract(vae)          # checks layout, stats and tiles
cond = contract.encode(pixels_uint8)          # (B, 24, T_lat, H/16, W/16) FP32, normalized
print(cond.digest, cond.row_digests, cond.record["tile_plan_sha256"])
video = contract.decode(latents_normalized)   # (B, 3, F, H, W) FP32 in [0, 1]
```

`load_release_vae` checks `config.json` and the shard sizes against the pinned
checkpoint, and with `verify_sha256=True` it also checks the SHA-256 of each
shard. It loads the weights strictly in FP32 without `from_pretrained`, which
would need `accelerate` for the modules pinned to FP32.

## Validation

```bash
# Self-contained layout/tiling/preprocessing/normalization tests (CPU)
python -m pytest tests/models/minimax_h3/test_h3_visual_vae.py -q
# With Diffusers f53d5520 (small random VAE, CPU + CUDA) and the released weights
PYTHONPATH=.:<diffusers>/src RL_KERNEL_H3_VAE_DIR=<snapshot>/vae \
    python -m pytest tests/models/minimax_h3/test_h3_visual_vae.py -q
# Evidence report and charts
PYTHONPATH=.:<diffusers>/src python tools/validation/models/h3_visual_vae_evidence.py \
    --vae-dir <snapshot>/vae --out evidence.json --verify-sha256
python tools/validation/models/plot_h3_visual_vae.py evidence.json
```

## Known limitations

- The contract guarantees run-to-run and batch-row equality on one device type,
  and bitwise parity with the Diffusers recipe on CUDA. Encoder and decoder
  outputs are not claimed equal across GPU generations (Hopper vs Blackwell
  pick different cuDNN and attention kernels) or between CPU and CUDA. Only
  preprocessing and normalization are device-independent.
- Validated on B200 (SM100) only. H100 (SM90) evidence and ROCm are follow-ups.
- No backward pass: the VAE sits outside the RL gradient path in RFC #420.
- Decoded pixels are secondary evidence (RFC #420 Phase B exit criteria); the
  `uint8` media conversion of the video processor is not part of this contract.
