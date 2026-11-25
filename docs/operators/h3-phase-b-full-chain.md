# MiniMax-H3 Phase B Full Chain

## Summary

RFC #420 Phase B row `phase_b_full_chain`. It runs one MiniMax-H3 request from its encoded
inputs to decoded video and audio, through the other Phase B rows and the Phase A core, and
seals what every stage produced. Two sealed runs of the same request can then be compared, and
the comparison names the first stage that drifted. Inside the denoising loop it also names
the step, the transformer module and the packed row.

Module: `rl_engine/models/minimax_h3/full_chain.py`. Runner:
`tools/validation/models/h3_full_chain.py`. Tests: `tests/models/minimax_h3/test_h3_full_chain.py`.

## Stages

| Stage | What runs | Owning row |
| --- | --- | --- |
| `request` | the request spec, the prompt and the raw media, hashed | this row |
| `media_preprocess` | canvas, keyframes / normalized references, VAE input pixels and waveforms | `fl2va_pipeline_contract`, `ref2va_pipeline_contract` |
| `text_conditions` | `prompt_embeds` (layer-50 hidden state) and per-row modality tags | `qwen3vl_layer50_contract` (pinned features until it lands) |
| `visual_vae_encode` | keyframe / reference latents: posterior sample, seed 42, FP16 rounding, normalization | `visual_vae_contract` |
| `audio_vae_encode` | reference soundtrack rows: posterior mean, mono micro-batch, channel-major rows | `audio_vae_contract` |
| `packed_layout` | the sealed packed-layout artifact and its five index tensors | pipeline contracts |
| `initial_rows` | noised conditions and target noise, in the reference draw order (CPU generator) | pipeline contracts |
| `schedule` | video (shift 12) and audio (shift 3) sigma grids, timesteps, row-timestep plan | `h3_sigma_schedule` |
| `denoise` | one transformer call and two Euler steps per step; the trajectory artifact | Phase A |
| `unpack` | final video latents `(1, 24, T, H, W)` and audio latents `(2, 32, T)` | this row |
| `video_decode` | decoded RGB in `[0, 1]` (secondary) | `visual_vae_contract` |
| `audio_decode` | decoded stereo waveform (secondary) | `audio_vae_contract` |

A stage that does not apply to a request (FL2VA has no audio conditions) is still recorded,
with no components and a `skipped` reason, so every artifact has the same twelve records.

## Artifact

`run_chain(request, components, label=..., text_source=...)` returns a `ChainRun` whose `artifact` holds:

- one record per stage: the sha256 (dtype, shape, raw bytes) of each tensor the stage hands
  on, as `[name, digest]` pairs in production order, plus the stage's runtime provenance
  (contract ids, backend flags, device, the denoiser identity and execution report);
- a hash chain over the stage records and a seal over the whole artifact; `verify` refuses
  an edited, truncated or reordered artifact;
- the embedded packed-layout artifact (`packed_layout.layout_artifact`) and trajectory
  artifact (`trajectory`), cross-checked against their stage records;
- the transformer probe: per step, the digest of every probed module's output in execution
  order (`proj_in`, `audio_proj_in`, `context_embedder`, `token_refiner`, `time_embedder`,
  `transformer_blocks.i`, `norm_out`, `proj_out`, `audio_proj_out`);
- decoded-media metrics (per-channel mean, range, RMS, peak), the runtime profile (versions,
  device, compute capability, TF32 flags; no host names or paths) and the gate policy.

The run label and free-form metadata stay out of the compared digests, so two runs with
different labels of the same request compare equal.

`keep_tensors=True` also returns a CPU copy of every stage output, every step's velocity and
latent, and the layout index tensors; `capture(step, module)` selects which transformer
module outputs are kept (all of them by default with `keep_tensors`).

## First drift and the gate

`compare(lhs, rhs, tensors_lhs=..., tensors_rhs=...)` checks both seals and that the two runs
are of the same request. It then reports:

- `first_drift`: the first stage and component whose digest differs.
  - Inside `denoise` it adds `step` and `transformer`: the first probed module output that
    differs, at the earliest step where one does.
  - It also adds `trajectory`: the first differing velocity or latent, velocity before latent
    and video before audio. A latent drift with equal velocities points at the scheduler
    arithmetic.
  - A module can drift without the trajectory drifting, when the difference is rounded away
    before the output heads. The report says so.
  - With tensors, every entry carries the mismatch count, the max absolute difference and
    the first differing row.
  - The row is mapped to its packed identity: modality; condition, reference or target;
    `(t, h, w)` rotary position; latent frame and patch, or stereo channel and latent index.
- `gate`: `hard_gate` passes only when every stage up to `unpack` is byte-equal, with the
  transformer probe excluded (it localizes; it is not latent data).
  `transformer_intermediates_equal` reports the probe separately (`null` when it was
  not compared: a run observed no module output at some step). `decoded_media_equal` and the PSNR / SNR between decodes are secondary: they never
  turn a latent failure into a pass, nor a latent pass into a failure (RFC #420 Phase B exit
  criterion). An undefined PSNR / SNR (identical media, a silent reference) is `null`, with the
  reason in `<name>_note`.

`DriftInjection(stage, component=..., row=...)` or
`DriftInjection("denoise", step=..., module=..., row=...)` moves one value by one ULP (or one
count for integer pixels). The tests use it to prove that every stage, module and row is
localized. A run whose injection matched nothing is refused (`InjectionNotRun`), not sealed. Row
evidence is given only for retained tensors whose digests are the recorded ones.

## Interfaces

- **Text.** `TextConditioner.encode(TextRequest) -> TextConditions`. `TextRequest` carries
  the prompt, the preprocessed media and `input_sha256`, a digest of exactly what the
  conditioner reads. `PinnedTextFeatures` replays a safetensors file written by
  `save_pinned_text_features`. The file is bound to that digest and refuses any other prompt
  or media. Its provenance (`source: pinned-precomputed`, producer, file sha256) is recorded
  in the `text_conditions` stage. The runner's `pin-text` produces such files with Diffusers'
  own `MiniMaxH3FL2VATextEncoderStep` / `MiniMaxH3Ref2VATextEncoderStep` and the released
  Qwen3-VL. When `qwen3vl_layer50_contract` lands, its encoder implements the same protocol.
- **Transformer.** `TransformerDenoiser(model, variant=..., mode=...)` wraps a
  `MiniMaxH3Transformer3DModel` (or anything with its signature and module names).
  - `variant` is the checkpoint partition the weights came from. A layout of the other
    variant is refused before any forward (H16).
  - `mode=None` runs the provider's own arithmetic (`P/P`).
  - `"strict"` / `"audit"` bind the RL-Kernel graph through `runtime.H3DiffusersAdapter` for
    the whole sampling loop, and the adapter's report goes into the provenance. The strict
    graph needs the full 50-block model and Hopper (cc 9.0) today.
- **Sampler.** `euler_step` is `MiniMaxH3Scheduler.step` (`diffusers@f53d552`) op for op.
  `x0` takes sigma from the timestep the transformer saw, and the blend takes it from the
  grid. `dual_schedule` uses `NativeH3SigmaScheduleOp` and refuses video and audio grids of
  different lengths, which the reference would zip silently.

## Sealed fixtures

`tests/models/minimax_h3/fixtures/h3_full_chain_{fl2va,ref2va}_toy.json` are sealed toy runs.
They use a 64x64 canvas, 124 frames, a toy VAE with the release layout, the small audio VAE of
`audio_vae_contract` and a toy H3-shaped transformer, all on CPU. Their bytes are host
specific. A second host with the same torch, ATen CPU capability, machine and thread count gave
other bytes in two places:

- FL2VA at `media_preprocess`, the keyframe stretched with Pillow's LANCZOS resize (80x48 to
  64x64). The raw keyframes were equal. Every PyPI Pillow wheel from 10.4 to 12.3 gives the
  committed bytes on the sealing host, and the released-weight runs' 2x LANCZOS resize gave the
  same bytes on both hosts, so the Pillow on the other host (version, build or CPU path; not
  recorded) computes this non-integer resize differently;
- Ref2VA at `audio_vae_encode`, the CPU convolutions of the audio VAE. oneDNN picks its kernels
  from the CPU's ISA, which torch's `cpu_capability` does not show: on the AVX512 sealing host,
  `ONEDNN_MAX_CPU_ISA=AVX2` gives the same first drift.

The tests therefore check:

- on every host: the committed fixtures' seal (hash chain, architecture fingerprint) and
  structure (stages, components, provenance keys, steps, probed modules) against a fresh run;
- on every host: byte determinism, by sealing in the test process and replaying in a fresh
  process, all 12 stages equal;
- with `RL_KERNEL_H3_REPLAY_CHAIN_FIXTURES=1`, on the host that sealed them: the committed
  bytes, bit for bit.

Runs are at 4 intra-op threads, because CPU convolutions also change bytes with the thread
count. Set `RL_KERNEL_H3_REGEN_CHAIN_FIXTURES=1` to reseal. The released-weight fixtures are
produced by the runner and kept with the evidence, not in the repository.

## Reproduce

```bash
# text features with the released conditioner (CPU is enough; ~10 s FL2VA, ~90 s Ref2VA)
PYTHONPATH=.:<diffusers-f53d552>/src python tools/validation/models/h3_full_chain.py pin-text \
    --workflow fl2va --text-root <ckpt> --out fl2va-text.safetensors --report pin.json
# one sealed run (provider arithmetic, first 4 released blocks; drop --layers for all 50)
PYTHONPATH=.:<diffusers-f53d552>/src python tools/validation/models/h3_full_chain.py run \
    --workflow fl2va --transformer-dir <ckpt>/transformer --vae-dir <ckpt>/vae \
    --audio-vae-dir <ckpt>/FL2VA/audio_vae --text-source pinned --text fl2va-text.safetensors \
    --layers 4 --label A --out fl2va-A.json --report fl2va-A-report.json
# the same run again, in a fresh process: only the label and the output files change
PYTHONPATH=.:<diffusers-f53d552>/src python tools/validation/models/h3_full_chain.py run \
    --workflow fl2va --transformer-dir <ckpt>/transformer --vae-dir <ckpt>/vae \
    --audio-vae-dir <ckpt>/FL2VA/audio_vae --text-source pinned --text fl2va-text.safetensors \
    --layers 4 --label B --out fl2va-B.json --report fl2va-B-report.json
python tools/validation/models/h3_full_chain.py compare fl2va-A.json fl2va-B.json --report cmp.json
python tools/validation/models/h3_full_chain.py toy --report toy.json   # CPU injection matrix
```

`--mode strict` runs the RL-Kernel graph, which needs Hopper. `--offload` keeps idle models
on the CPU, so the 66 GB transformer and the 10 GB VAE fit one 80 GB card.
