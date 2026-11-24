# MiniMax-H3 AudioVAE Contract (`h3-audio-vae-v1`)

RFC #420 Phase B row `audio_vae_contract`: stereo AudioVAE encode/decode and the 40 Hz latent
layout that feeds the `32 -> 5376` audio input head. Code: `rl_engine/models/minimax_h3/audio_vae.py`.

## Layout

| Fact | Value | Source of truth |
| --- | --- | --- |
| Waveform | FP32, 32 kHz, `[2, samples]`, order `(left, right)` | Diffusers pipeline, SGLang, vLLM-Omni |
| VAE | mono; stereo = batch items `(0, 1)` | `AutoencoderKLMiniMaxH3Audio` docstring |
| Hop | `prod(encoder_rates) = prod(decoder_rates) = 800` samples | checkpoint `metadata.json` |
| Latent | 32 channels at 40 latents/s per stereo channel, `T = ceil(samples / 800)` | |
| Generated length | `T = round(num_frames / 24 * 40)` (121 frames -> 202) | `audio_latent_num_frames` |
| Rows | channel-major `[left T rows ; right T rows] x 32` | Diffusers encode/unpack steps |
| Normalisation | `(latent - latents_mean) / latents_std`, per channel, FP32 on the host | `config.json` |
| Denormalisation | `latent * std + mean` (multiply, then add; no FMA) on the device | Diffusers decode step |
| Input head | rows feed `audio_proj_in`, `32 -> 5376` | pinned transformer config |

Encode takes the posterior **mean** (`mean_proj`); `logs_proj` is loaded but never evaluated.
Decode clamps to `[-1, 1]` and returns `[2, T * 800]`.

## Pinned preprocessing

`prepare_stereo_waveform` mirrors Diffusers' `_normalize_audio_condition`: cast to FP32, truncate
at the native rate to `int(max_duration * sample_rate)` samples, upmix mono by repeating the
channel, resample once. Resampling (`torchaudio.transforms.Resample`) is outside the strict
contract: non-32 kHz input is rejected unless `allow_resample=True`, which the caller records.
Empty, non-finite, integer and more-than-stereo input fails.

## Module

`H3AudioVAE` is an op-for-op port of Diffusers f53d5520 (DAC encoder with Snake activations,
causal-attention `pre_block` whose heads are mean-pooled and then average-pooled to 32 channels,
BigVGAN decoder with anti-aliased SnakeBeta) under the released checkpoint's parameter names:
`model.safetensors` loads with `strict=True`, `checkpoint_state_dict()` exports `weight_g` /
`weight_v` names again. Under equal backend flags and batch shapes it is bitwise equal to
Diffusers (tested on the released weights and through Diffusers' own pipeline blocks).
The Snake activations are eager: the checkpoint's remote code scripts them with
`torch.jit.script`, whose profiling executor re-fuses them after the first calls and changes
the first call's bytes.

## Strict execution policy

The library paths repeat bit-exactly but are not batch invariant: one channel's bytes depend on
how many other clips share its call, because the `pre_block.proj` GEMM (2048 -> 32, `M = B * T`)
changes its split with `M`, and with TF32 enabled cuDNN changes its convolution algorithm with
the batch. The strict policies fix the arithmetic per logical mono channel:

| Policy | Micro-batch | TF32 | cuDNN | SDPA |
| --- | --- | --- | --- | --- |
| `h3-audio-encode-strict-v1` | 1 mono channel | off | **disabled** | math |
| `h3-audio-decode-strict-v1` | 1 mono channel | off | deterministic, no benchmark | (unused) |

Micro-batch 1 makes a channel's bytes a function of its own waveform or latent only. Encode
disables cuDNN (the SGLang / vLLM-Omni encode recipe; on B200 the encoder is as fast as with deterministic
FP32 cuDNN, though 2-11x slower than Diffusers' default TF32 cuDNN),
so the convolution algorithm cannot depend on free workspace. Decode keeps deterministic cuDNN
(the SGLang decode recipe): native convolutions make BigVGAN about 4x slower on B200.
`execution_scope` applies the flags and restores them; they are process-global, so do not run
other CUDA work concurrently in the same process while a scope is open. It also disables CPU
and CUDA autocast, so an outer FP16/BF16 autocast region cannot run the FP32 contract in mixed
precision (the hashes are the same with and without one). On exit, the caller's full FP32
precision state comes back: the matmul precision string (`"medium"` included), not just
`allow_tf32`. Each backend is written through the API it was readable with: the legacy flags,
or per-backend `fp32_precision` if the caller set that one.

Reproducibility is claimed per platform (GPU architecture, cuDNN/cuBLAS build), not across
architectures. `provenance()` records the contract, both policies, the normalisation-stat
hashes, device, CUDA and cuDNN versions; `AudioVAECheckpoint.read(verify_sha256=...)` checks the
weights against `pinned_checkpoint.json`.

## Evidence

`tests/models/minimax_h3/test_h3_audio_vae.py` (CPU on a reduced-width module; Diffusers parity
when `AutoencoderKLMiniMaxH3Audio` is importable; released weights from `$RL_KERNEL_H3_AUDIO_VAE`
on CUDA). `benchmarks/models/benchmark_h3_audio_vae.py` compares Diffusers, SGLang's own module,
the vLLM-Omni recipe on the checkpoint's remote code and the strict path: cold-start drift,
batch invariance, error against FP64, bitwise parity and latency.
