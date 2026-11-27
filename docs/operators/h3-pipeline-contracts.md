# MiniMax-H3 FL2VA and Ref2VA Pipeline Contracts

## Summary

RFC #420 Phase B rows `fl2va_pipeline_contract` and `ref2va_pipeline_contract`. They cover
what runs between the request and the first transformer call:

- the request is resolved and its media normalised;
- the conditioner presentation is tagged;
- the packed `[text | conditions | target audio | target video]` document is built, with its
  fp64 rotary grid, modality tags and index tensors;
- the initial rows are drawn.

The reference is the pinned Diffusers modular pipeline (`huggingface/diffusers@f53d552`,
`modular_pipelines/minimax_h3`), which the MiniMax-H3 release itself pins.

FL2VA (`transformer/`) and Ref2VA (`transformer_ref/`) have the same transformer config and
tensor table. They are still separate pipeline identities (RFC #420 H16): their layouts,
preprocessing and weights differ, and mixing any two of them fails closed.

Out of this row: the VAE encodes (`visual_vae_contract`, `audio_vae_contract`), the Qwen3-VL
hidden state (`qwen3vl_layer50_contract`) and the transformer itself. These rows consume
their outputs as inputs.

## Shared Core

`rl_engine/models/minimax_h3/packed_layout.py`.

- **Geometry.** `resolve_canvas_size` (768 short edge, `768 * 1344` area cap applied before
  each side rounds to a multiple of 32, so the canvas can end slightly above the cap, e.g.
  576 x 1856 at 23:7; aspect 1:4 to 4:1), `align_num_frames` (`17n + 5`), `video_latent_num_frames`
  (`5n + 2`), `audio_latent_num_frames` (40 per second, round half to even), and
  `resolve_num_frames`, which checks the *aligned* duration is 5-15 s.
- **Rotary grid.**
  - Every coordinate is fp64.
  - Spatial axes come from numpy `linspace(endpoint=False)` times 32, normalised by the
    square root of the latent area.
  - Time is non-uniform: one latent frame spans `5/3 * (1, 4, 4, 4, 4)`, and text rows sit at
    their row index.
  - Audio rows are channel-major, carry no height, and are pinned to the two extremes of
    their own block's width grid.
- **Two summations of the same span, both written out.**
  - The FL2VA `last` anchor uses numpy's pairwise sum: `numpy_pairwise_sum` reproduces
    numpy's `add.reduce`, so a numpy build cannot change it.
  - The Ref2VA clock advance uses Python's `sum()`. That changed algorithm in CPython 3.12,
    from plain left-to-right to Neumaier-compensated, so the reference's bytes depend on the
    interpreter. The two first differ at 6 latent frames.
  - `SPAN_SUMMATIONS` holds both. The default is `"neumaier"`, which is what Diffusers and
    SGLang compute on the RL-Kernel interpreter (3.12). The choice is recorded in the layout
    geometry and sealed.
- **`PackedLayout`.**
  - Fields: `position_ids` `(S, 3)` fp64, `token_tags`, `text_indices`, `video_indices`
    (conditioning rows first), `audio_indices` (reference rows first), the two
    conditioning-row counts and the geometry.
  - `validate()` checks:
    - the three index tensors partition the sequence;
    - text comes first;
    - every tag matches its rows (text rows are text or, inside a vision block, video);
    - dtypes and the CPU device.
  - `denoiser_inputs(device)` moves the five transformer kwargs without a dtype change.
- **Row timesteps.** `row_timesteps` / `row_timestep_plan`, as `build_row_timesteps`:
  - visual conditions sit at `max(t, 0.999)`;
  - reference soundtracks sit at `1.0`;
  - text inherits the video timestep;
  - the result is the fp32 `torch.unique(sorted, return_inverse)` pair.
- **Initial rows.**
  - `patchify_video_latents` orders patches frame-major, then row-major.
  - `noise_condition` computes `t * x0 + (1 - t) * noise` with `t` an fp32 tensor, because
    that is what the pipeline hands the scheduler. A Python-float `t` would keep `1 - t` in
    fp64 and give other bytes (probe H15).
  - `pack_condition_rows` makes one draw per condition, in packed order.
  - `draw_generated_rows` draws the target video latent, then the audio rows. A CPU
    generator draws on the CPU, as `randn_tensor` does.

## FL2VA

`rl_engine/models/minimax_h3/fl2va.py`, workflows `t2va` (zero keyframes) and `fl2va` (first,
last, first-and-last), all on `transformer/`.
Keyframes need Pillow, which is not a base dependency of RL-Kernel (`pip install Pillow`);
the module imports it only on the keyframe path, so `t2va` runs without it.

- **`prepare_keyframes`.**
  - The canvas comes from the first keyframe's aspect ratio, unless `height` / `width` are
    given. Without keyframes it is 16:9.
  - The first keyframe is stretched with PIL LANCZOS.
  - A follower is cover-cropped with the reference's arithmetic: `round` sizing,
    `(src - w) // 2` centring. `VaeImageProcessor`'s `crop` mode differs by a pixel on about
    half of all aspect ratios.
- **`fl2va_presentation`.**
  - Each keyframe adds `"<Picture i>: "` (text tag) followed by `<|vision_start|>`,
    `<|image_pad|> * n` and `<|vision_end|>` (video tag).
  - The prompt follows verbatim, with no chat template and no special tokens.
- **`build_fl2va_layout`.**
  - Layout: `[text | keyframes | target audio | target video]`.
  - A `first` keyframe sits at the target's first frame time.
  - A `last` keyframe sits on the last *pixel* frame: the pairwise span minus `5/3`, which is
    the last latent frame's time plus three pixel frames.
  - Anchor tuples other than `()`, `("first",)`, `("last",)` and `("first", "last")` are
    refused.
- **`fl2va_initial_rows`.** Draws one noise per keyframe, then the target video, then the
  target audio. It checks the keyframe latents fill exactly the rows the layout reserved.

## Variant Identity (H16)

- **`PIPELINES`.** Each variant has its transformer directory, its wrapper directory, its
  workflows and a preprocessing recipe. The recipe hash differs between the variants.
- **`checkpoint_component_hashes(variant)`.**
  - Covers the pinned transformer, the schedulers and the shared components (text encoder,
    processor, tokenizer, both VAEs).
  - Only the transformer hash differs between the variants.
- **`identify_transformer_dir(path)`.**
  - Names the variant a local transformer directory holds. All 14 shards differ between the
    variants, while config, index and tensor table are identical.
  - Shard hashes come from the Hub `local_dir` download metadata or the HF cache blob name,
    so it does not read 66 GB.
  - With `full_hash=True` it reads and hashes every shard, even when the metadata is
    present, and raises `VariantMismatch` if a recorded hash disagrees with the contents.
  - Mixed or unknown shards raise `VariantMismatch`.
- **`structural_workflow(layout)`.** The workflow a layout's structure belongs to:
  - no conditioning rows: `t2va`;
  - the target clock starts at the text length: FL2VA keyframes, which live on the target
    clock;
  - the target clock starts later: Ref2VA, because every reference advances it.
- **`check_pipeline_identity(layout, transformer_variant=..., artifact=...)`** raises
  `VariantMismatch` when:
  - a declared workflow disagrees with the structure;
  - the transformer is the other variant;
  - a sealed artifact describes a different layout.

## Ref2VA

`rl_engine/models/minimax_h3/ref2va.py`, workflow `ref2va` on `transformer_ref/`.
Reference media need Pillow, which is not a base dependency of RL-Kernel (`pip install
Pillow`); the module imports it only to normalise images and resize videos.

- **`validate_references`.**
  - Limits: at most 9 images, 3 videos, 3 audios and 12 references in total.
  - Audio alone is refused.
  - Diffusers' reference objects are accepted as they are.
- **`resolve_ref2va_plan`.** The canvas is 16:9 unless given: references never bind it. The
  frame count is aligned like the target's.
- **`normalize_references`.**
  - An image goes to its own 2048-pixel short edge, rounded to 32, with no area cap, LANCZOS.
  - A video is resampled to 24 fps by holding whole frames (`floor(i * 24 / fps + 0.5)`),
    truncated, then LANCZOS-resized onto the canvas its own aspect ratio resolves to.
  - A soundtrack is truncated at its source rate, upmixed to stereo and resampled once to
    32 kHz.
- **`sample_video_condition_frames`, `ref2va_presentation`.** What the conditioner reads:
  - frames at 2 fps;
  - merged pairs labelled `"<x.x seconds>"` with round-half-even;
  - `"<Audio j>: "` before `"<Video k>: "` for a video with sound, labels numbered per
    modality.
- **`vae_video_frame_count`.** The frame bound a video reference is sliced to, as in the
  pinned pipeline: from 22 frames on it snaps *down* to `17n + 5`; a shorter reference keeps
  all its frames, and the VAE repeats the last one up to a multiple of 17.
- **`reference_blocks`, `build_ref2va_layout`.**
  - Layout: `[text | references | target audio | target video]`.
  - How each reference advances the clock:
    - an image by one slot;
    - an audio by its latent count;
    - a video (its soundtrack packed just before it) by `max(audio latents, video span)`.
- **`ref2va_initial_rows`.** Noised visual references, then target noise. Clean reference
  soundtracks (`t = 1`), then target audio noise.
- **`map_transformer_ref`.**
  - Maps the `transformer_ref/` state dict through the `h3_checkpoint_mapper` tables.
  - The result carries the Ref2VA identity (`VariantState`), so `check_layout_weights`
    refuses it with an FL2VA layout and refuses FL2VA weights with a Ref2VA layout.

## Repeated Preprocessing (H-C7)

`layout_artifact(layout, inputs=...)` seals:

- the schema;
- the checkpoint pin;
- the pipeline identity: variant, workflow, structural workflow, transformer directory,
  recipe hash and checkpoint component hashes;
- the geometry and row counts;
- a sha256 per layout tensor;
- optional digests of what produced the layout, such as normalised media, token ids and
  initial rows.

Tools that consume the artifact:

- `verify_artifact` / `load_artifact` refuse an edited artifact.
- `first_difference` names the first differing field of two artifacts.

`tools/validation/models/h3_pipeline_contract_report.py --row fl2va|ref2va` repeats every case:

- in-process;
- on one thread;
- in two fresh processes with other thread counts.

It then requires identical artifacts and also records the Diffusers and SGLang comparisons,
the identity table and the layout build time.

## Tests

- `tests/models/minimax_h3/test_h3_packed_layout.py`:
  - geometry;
  - both summations against numpy and the running interpreter;
  - layout validation and timestep plan;
  - conditioning draw order and the fp32 mix;
  - identity and artifact sealing.
- `test_h3_fl2va_pipeline.py` and `test_h3_ref2va_pipeline.py`. Without Diffusers or SGLang
  installed, the comparisons skip. Each file covers:
  - every keyframe signature or reference kind;
  - H-C7 in-process, on one thread, in a fresh process and on CUDA;
  - H16 refusals;
  - the Diffusers static builders over sweeps;
  - the tiny public Diffusers pipeline (`hf-internal-testing/tiny-minimax-h3-modular-pipe`),
    run end to end on CPU, its step outputs captured and compared with ours byte for byte:
    the resized keyframes or normalised references and the canvas, the conditioner's
    modality tags (not its token ids), the five layout tensors and the condition-row counts,
    the row-timestep plan, and the initial rows (FL2VA: keyframe and target video rows when
    there are keyframes; Ref2VA: video and audio rows). The encoder latents are the
    pipeline's own, used as inputs, not compared;
  - SGLang's packed-sequence builders (unpadded prefix).

## Out Of Scope

- **Encoders.** The VAE encodes and the Qwen3-VL forward belong to the encoder rows.
- **SGLang's hybrid layouts.** SGLang's padding to 64 rows and its hybrid Ref2VA-plus-keyframe
  layout (ComfyUI) are not Diffusers layouts and these builders do not produce them. The
  structural check classifies a hybrid layout as Ref2VA; it does not recognise the keyframes.
- **A sealed end-to-end fixture per variant.** This needs the full model on Hopper and the
  `transformer_ref/` weights, and is `phase_b_full_chain`.
