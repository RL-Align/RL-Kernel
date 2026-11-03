# Item 20 — separate 128-dimensional Indexer C4 compressor

## Scope and dependency

This task is stacked on task 19 / PR #618 (`8793052`). Its PR still targets
`dsv4-rocm`. Merge #618 first; the task-20 commit adds the Indexer-specific
contract and adapters while reusing the C4 implementation rather than copying
its grouping, ownership, cache-index mapping and autograd machinery.

The main-KV contract remains D=512. `IndexerC4Contract` independently admits
D=128 with its own backend identity, family fingerprint, parameters, bound
data, version and canonical-gradient namespace. Passing the main-KV contract,
bound data or parameter geometry to the Indexer operator fails before launch.

## Boundary

```text
projected BF16 input [tokens,512] = [content256,score256]
APE FP32 [4,256], RMS weight FP32 [128], BF16 cos/sin [positions,32]
logical token IDs int64 [tokens,2] = [request,global_position]
    -> complete four-token groups / eight-token overlap context
    -> existing aiter compression, RMSNorm, interleaved tail RoPE
    -> existing aiter normalized Walsh-Hadamard(128)
    -> independent BF16 values [rows,128]
    -> explicit MXFP4 or FNUZ FP8 payload and scales
```

Default `precision="mxfp4"` returns uint8 packed E2M1 `[rows,64]` and unshuffled
E8M0 bytes `[rows,4]`, group size 32, low nibble first. It reuses aiter
`per_1x32_f4_quant_triton`. The alternative explicit `precision="fp8"` reuses
vLLM `per_token_group_quant_fp8` with FNUZ encoding and power-of-two FP32
per-row multipliers `[rows,1]`. These are distinct scale ABIs, not interchangeable
bytes. Item 16's independent numerical-domain admission still applies when
consuming these layouts.

Both train and rollout adapters expose the same BF16 values, payload and scales.
Quantization is detached: this task does **not** define a straight-through
gradient through MXFP4/FP8. A loss on `output.values` receives the explicitly
defined BF16-output compressor/Hadamard VJP. The complete quantized Indexer
training chain belongs to item 26.

## Reused numerical providers

- aiter `flydsl_fused_compress_attn`: fixed one-wave C4 overlap forward.
- aiter `rotate_activation`: actual native Hadamard; **no generic matmul fallback**.
- aiter MXFP4 and vLLM FP8 quantizers above.
- Pinned mLite `compressed_compact_graph`: FP32 functional compressor VJP.
- RLK `CanonicalBackwardSession`: logical-group ordered parameter reductions.

The upstream native D=128 Hadamard in the tested image uses eight physical rows
per wave and does not safely bound every partial last-wave allocation. The
adapter therefore pads **both** input and output buffers to eight rows and
returns only the logical rows. No numerical kernel is modified. Tests check
native buffer dimensions, numerical values and retention of live earlier
outputs/canaries across tail launches.

Backward uses the same native transform for H-transpose (H is symmetric), with
an explicit BF16 dY/transform boundary, followed by mLite's FP32 compact VJP.
The numerical oracle separately checks this rounding boundary and the FP64
compressor mathematics. Disjoint main/Indexer parameter namespaces are enforced
even if both adapters use the same caller layer ID.

## State and lifecycle

Complete groups, predecessor admission, CP final-token ownership, global RoPE
positions, output page mapping and mutation/version diagnostics are inherited
from task 19. `emit_group_ids` permits context-only predecessor groups at chunk
boundaries. C4 source tensors and parameters must not be mutated through any
alias while in use; version stamps are diagnostics, not storage write protection.
State and gradient storage are not shared between main and Indexer branches.

This task writes a private temporary paged buffer and returns independent rows.
Request cache lifecycle, persistent partial state and exactly-once MXFP4 cache
commit belong to item 22. It does not claim Graph capture, full framework model
registration, RCCL/multi-GPU CP transport or full-model raw-logprob acceptance.

## Reproduction on the tested member image

Tested: MI300X `gfx942`, Torch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`,
Triton 3.6.0, aiter 0.1.16.post3, FlyDSL 0.2.0, vLLM 0.26.0+rocm723.
The existing image was reused; no Torch/ROCm/aiter replacement was necessary.

Set `RLK_MLITE_DSV4_BACKWARD_SOURCE` to the unmodified `backward.py` from mLite
commit `327a238239448de6771beea185dc6902a8de3eaa`, path
`experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/attention/backward.py`.
The required SHA256 is
`6a1e78ca89b83f56391b5e5ac409e765338f1e3a625251d0d4c877c1e80b4a68`.
If mLite is installed, the shared backend can also resolve that module directly.

```bash
python3 -m pytest tests/contracts/operators tests/models/deepseek_v4 -q \
  --basetemp=/outputs/item20-tests --junitxml=/outputs/item20-tests.xml
python3 benchmarks/benchmark_indexer_compressor_c4.py \
  --repeats 30 --output /outputs/item20-benchmark.json
```

Runtime fingerprints include the numerical providers, native rotation binary,
software versions, source hashes, profile and rotation padding policy. Configure
`expected_implementation` to require an exact train/rollout match. Fingerprints
are not a substitute for testing a different dependency stack.
