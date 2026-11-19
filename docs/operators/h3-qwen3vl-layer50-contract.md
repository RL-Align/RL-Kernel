# MiniMax-H3 Qwen3-VL Layer-50 Text Feature Contract

## Summary

RFC #420 Phase B row `qwen3vl_layer50_contract`. MiniMax-H3 conditions its
Omni-Transformer on the full Qwen3-VL-32B (`text_encoder/` of
`MiniMaxAI/MiniMax-H3`, class `Qwen3VLForConditionalGeneration`). The feature
is not the last hidden state. It is `hidden_states[50]`: the raw, pre-norm
output of decoder layer index 49, of width 5120. That feature feeds the
5120 -> 5376 text projection and the 2-layer token refiner.

The contract pins three things:

- the tokenizer and its special tokens;
- the multimodal processor and the placeholder layout;
- the exact extraction of that hidden state.

Every feature leaves with a SHA-256 digest of its BF16 bytes, so two encodes of
the same presentation either share a digest or prove a drift.

The network is reused from HF transformers, unmodified; see
[Reuse decision](#reuse-decision). The code is in
`rl_engine/models/minimax_h3/qwen3vl_text.py`. The pinned identity sits next to
it in `qwen3vl_text_fingerprint.json`: checkpoint revision `42ed227e`, SHA-256 of
all 34 tokenizer, processor and `text_encoder/` files, configs and the
special-token table.

Contract id: `h3-qwen3vl-layer50-v1`.

## Release presentation

| Item | Release value (Diffusers `f53d5520`, `encoders.py`) |
| --- | --- |
| Template | none: the prompt verbatim, no chat template |
| Special tokens | `add_special_tokens=False`; the tokenizer adds no BOS/EOS either way |
| Batch | one presentation per forward, batch 1 |
| Padding / truncation | none / none |
| Attention mask | all ones |
| `mm_token_type_ids` | 0 text, 1 on `<\|image_pad\|>`, 2 on `<\|video_pad\|>` |
| Vision block | `<\|vision_start\|>` + `prod(grid_thw) / 4` pads + `<\|vision_end\|>` |
| Feature | `text_encoder.model(...).hidden_states[50]`, encoder dtype (BF16) |
| Modality tags | text 1, video 0 (vision-block rows too), audio 2 |

FL2VA prepends `"<Picture i>: "` and an image block for each keyframe. Ref2VA
labels each reference in the same way. Building those presentations belongs to
the `fl2va_pipeline_contract` and `ref2va_pipeline_contract` rows. This row
checks that whatever they build agrees with the processor's grids.

## Tokenizer and special tokens

`verify_release_tokenizer` checks three things:

- the vocabulary length, 151676;
- the 21 `special=True` added tokens, by content and id. Examples:
  `<|vision_start|>` = 151652, `<|image_pad|>` = 151655,
  `<|video_pad|>` = 151656;
- a pinned mixed English/Chinese probe prompt, and that
  `add_special_tokens=True` gives the same ids as `False`.

`tokenizer/` and `processor/` ship byte-identical tokenizer files, and the
fingerprint pins both.

`tokenize_prompt` fails closed when a prompt spells out a special token. The
tokenizer maps a literal `"<|image_pad|>"` in user text to the vision
placeholder id. With no pixels behind it, Diffusers runs anyway and conditions
on a placeholder embedding the release never trained against. Inside an
FL2VA/Ref2VA presentation it shifts every later vision block. Added tokens with
`special=False`, such as `<think>`, still tokenize to one id each, as in the
release.

## Multimodal processor

`verify_release_processor` pins the image and video processor geometry:

- patch 16, temporal patch 2, merge 2;
- mean and std 0.5;
- the `shortest_edge`/`longest_edge` pixel bounds.

`check_presentation` brackets every vision block. It checks each block's pad
count against `image_grid_thw` and `video_grid_thw`; a video contributes one
block per merged frame pair. It also checks that pixel tensors are present
exactly when blocks are. `mm_token_type_ids` reproduces
`Qwen3VLProcessor.create_mm_token_type_ids`.

The CPU image processor is batch-invariant. One call on several images gives,
bit for bit, the same `pixel_values` rows as one call per image
(`test_image_processor_is_batch_invariant_and_matches_presentation`).

## Layer-50 extraction

`Qwen3VLLayer50Encoder.encode` makes the same `text_encoder.model(...)` call as
Diffusers' `get_qwen3vl_prompt_embeds`. It stops that call with a forward hook
on decoder layer index 49. transformers 5.x records `hidden_states[k]` with a
forward hook on decoder layer `k-1` (`hidden_states[0]` is the embedding). So
the captured tensor is the one Diffusers returns, and it is unaffected by:

- the final norm, which only touches `hidden_states[64]`;
- the deepstack visual injection, which modifies the residual after layers
  0-2, upstream of layer 49.

Layers 50-63, the final norm and the 65 retained hidden states are never
computed.

`load_release_text_encoder` loads a 50-layer stack through the same
`from_pretrained` (on CPU, then moved to the device; `accelerate` is not
needed). The rotary `inv_freq` buffers are bitwise equal to a full load. The
loader fails unless every parameter came from the checkpoint and every skipped
key belongs to a layer >= 50. The 50-layer stack reads 12 of the 14 shards
(`required_shards`): shards 1-11 hold the embeddings and layers 0-49, and shard
14 holds the vision tower, the final norm and `lm_head`, which the stack loads
but the feature never uses. Shards 12 and 13 hold only layers >= 50.
`from_pretrained` opens every shard the index lists, so loading still needs the
full `text_encoder/` download.

Diffusers refuses a stack of exactly 50 layers, because its last hidden state is
post-norm. The hook path has no such trap, and a test pins both behaviours.

## Batch handling and runtime policy

`encode_many` runs one forward per presentation, so a row never sees its
neighbours. The runtime policy pins PyTorch's default numeric flags, which are
the release values:

- TF32 matmul off;
- BF16/FP16 reduced-precision reductions on;
- cuDNN TF32 on;
- the default SDPA backend priority.

`pinned_runtime` sets and restores the flags around every encode, and the
readback goes into the output record.

## Reuse decision

Survey of library paths for this step:

| Library | Text encoder | Layer 50 | Several prompts per forward | BI mode |
| --- | --- | --- | --- | --- |
| Diffusers `f53d5520` | HF `Qwen3VLForConditionalGeneration` (SDPA) | `hidden_states[50]`, all 64 layers run | no (batch 1) | none |
| SGLang 0.5.21 `multimodal_gen` | own `Qwen3VLModel` under TP | builds 50 layers, `Identity` final norm | no (encoder data parallel, one request per copy) | none |
| vLLM-Omni (partial copy available) | own `MiniMaxH3Qwen3VLEncoder` under TP | encoder module not in the copy | one request per encode | none visible |
| HF transformers 5.17 | `Qwen3VLModel` | `output_hidden_states` | yes, padded | none |

All three H3 runtimes use the same presentation: verbatim prompt,
`add_special_tokens=False`, no padding. All three batch a request's keyframe and
reference images into one vision-tower call.

The only batched prior-art path is HF padded batching. It is measured against
the per-row feature on B200 in [Results](#results).

The decision: reuse the HF path, one presentation per forward, early-exit at
layer index 49. This keeps the release bits (the Diffusers reference) and makes
batch composition irrelevant by construction. A separate RL-Kernel Qwen3-VL
implementation would not reproduce the released feature bits, and nothing here
needs one.

## Results

From `tools/validation/models/h3_qwen3vl_evidence.py` with the released weights (all 34 file
hashes verified), torch 2.13.0+cu130, transformers 5.17.0, Diffusers `f53d5520`. Inputs: 6 text
prompts of 1-288 tokens and 2 FL2VA-style presentations with 1 and 2 synthetic 480x832
keyframes. The PR carries the evidence JSONs and charts.

| Property | B200 (sm100) | H100 (sm90) |
| --- | --- | --- |
| Contract (50-layer stack) vs Diffusers `get_qwen3vl_prompt_embeds`, 8 cases | bitwise equal | bitwise equal |
| Contract hook on the full 64-layer model vs Diffusers | bitwise equal | bitwise equal |
| `hidden_states[50]` is the Diffusers feature; `[49]`/`[50]`/`[51]` distinct | yes | yes |
| Run to run | 1 digest per case | 1 digest per case |
| `encode_many`, 7 rotations of the 8 cases vs each case alone | 7/7 bitwise | 7/7 bitwise |
| HF padded batch (6 prompts, right or left pad) vs per row | not bitwise | not bitwise |
| Prompts spelling one of the 21 special tokens | 21/21 rejected | 21/21 rejected |

Prior-art padded batching changes the layer-50 feature by up to 1.09e+04 (B200) and 5.44e+03
(H100) max abs on the 288-token prompt; the contract's per-row feature is bitwise equal to
Diffusers on every case. On both GPUs the default selection matches forced cuDNN attention
bitwise, while forcing FlashAttention or the math backend changes the feature, which is why the
policy keeps the default backend priority.

Median latency per presentation, contract vs Diffusers (64 layers): 22.5-79.6 ms vs
29.6-89.6 ms on B200, 47.3-126.2 ms vs 62.5-141.7 ms on H100. Weights: 49.42 GiB (50 layers)
vs 62.13 GiB (64 layers).

## Tests

`tests/models/minimax_h3/test_h3_qwen3vl_layer50.py`:

- **Tiny random Qwen3-VL on CPU (53 layers):**
  - layer indexing: `hidden_states[50]`, not `[49]` or `[51]`, with and without
    a vision block;
  - equality with Diffusers `get_qwen3vl_prompt_embeds`;
  - truncated-stack equality and the post-norm trap;
  - repeatability;
  - batch-composition independence;
  - presentation checks.
- **Released files (`RL_KERNEL_H3_QWEN3VL_DIR`):**
  - SHA-256 pins and the tokenizer/processor pins;
  - token ids and attention masks identical across tokenizer instances and
    fresh processes;
  - special-token rejection;
  - processor batch invariance.
- **Released weights (CUDA, about 60 GB free):**
  - bitwise repeatability;
  - batch-composition independence of the 50-layer contract.
