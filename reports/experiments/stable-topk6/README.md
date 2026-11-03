# Task 31: stable Top-6 on MI300X

This implements issue #81 item 31 on `dsv4-rocm` baseline
`5ba769f681b7fa765ea6b29f9939454a858390ba`. The contract is score descending,
equal scores preferring smaller **global** expert IDs, and preserving that
selection order as the six output slots. No final expert-ID sort is allowed.

## Contract and upstream adaptation

The implementation adapts AITER 326cffa's `streaming_topk` structure and reuses
its monotone FP32-key helper. The additive companion patch changes tie keys
and the final ordering, accepts complete global-ID maps and retains original
FP32 payload bytes. It does not modify the original AITER or task-29/30 APIs.

The two source profiles are pinned unmodified AITER commit
`326cffa38c9eda16bef19375ba55334e4d3acd59`, and that base with the exact task-29
score patch shared by task 30. Their source hashes are explicitly validated.
The installed AITER package is not replaced. The companion patch applies cleanly
and resulting source hashes match its manifest. This is not an upstream release.

## Verification

```bash
AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task31/aiter:$PWD" \
  python -m pytest tests/ops/moe/test_stable_topk6.py \
  tests/runtime/test_stable_topk6_dispatch.py tests/runtime/test_kernel_registry.py -q
```

The complete suite ran on both admitted source profiles:

| Suite | Per-profile result |
| --- | ---: |
| MI300X exact ordering/payload, BI, gather VJP and runtime guards | 65 passed |
| CPU-safe contract/reference and exact dispatch/source guards | 19 passed |
| Existing registry regression | 12 passed |
| **Total for each of two profiles** | **96 passed; no failures or skips** |

The expected empty-graph warning comes from explicit capture rejection.
Ruff check/format, strict MkDocs build, patch application/fingerprints and
whitespace checks passed.

Tests cover all-equal scores; sixth-place ties across streaming blocks;
one-ULP-separated FP32 values; signed zeros and positive/negative subnormals;
finite FP32 extrema; padded 8→6 selection; E=6/7/31/127/128/129/255/256/257/513/
1025/32767; ten random seeds; M=0/1/16/17/32/65/257; mixed neighboring rows,
row placement and noncontiguous storage. Global-ID permutations, I32/I64 maps,
missing/duplicate/invalid IDs and partial-shard rejection are tested explicitly.
The original AITER entry must fail the tie and slot-order negative controls.

The independent CPU oracle lexicographically sorts `(-score,global_ID)` using
Python comparison and gathers original FP32 payloads. FP32 inputs are represented
exactly in Python floats. Tests compare exact IDs/columns, score bits and local
gather-VJP bits; no numerical tolerance is used. The local VJP scatters six
distinct payload gradients without atomics. It is not full routing backward.

`task31-mi300x.json` has 28 replay cases: 24 target-E=256 cases across four score
patterns and six batch sizes, plus four streaming/tail geometries. All discrete,
payload, slot, VJP and gradient-mode mismatch counts are zero. Two independent
processes/JIT caches using the two admitted shared-source profiles produce
identical input/map/upstream/ID/column/score/VJP hashes for every case;
`fresh-process-profile-hashes.json` records those comparisons.

## Real two-GPU global numbering validation

```bash
AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task31/aiter:$PWD" \
  python -m tools.validation.distributed.check_stable_topk6_global_ids \
  --devices 0 1 --output artifacts/task31/two-gpu-global-ids.json
```

`two-mi300x-global-ids.json` records **two MI300X GPUs on one node**, using
PyTorch's NCCL backend on ROCm (**RCCL all_gather**). Each rank owns reversed
even/odd global expert columns. They gather full scores/IDs, materialize columns
in different orders, then select using the same complete global numbering.

Each rank passes 16 cases: M=1/16/17/257 crossed with all ties, random ties,
adjacent scores and signed zeros. Global IDs and score payload bits match each
other and the independent complete-input oracle. Local selected-value VJPs also
match after remapping columns into canonical global order. This is real GPU
transport validation, not simulated shards or CPU Gloo. It does not certify
multi-node communication, distributed autograd or framework EP/TP hooks.

## Performance disclosure and scope

With 10 warmups and 30 GPU-event samples per case, median full-adapter forward
latency is approximately **229–292 µs**, raw AITER forward **31–40 µs**, and raw
local VJP **30–39 µs** in the checked-in replay. Full-adapter timings include
finite/global-coverage validation, ID snapshot/hash and host synchronization,
provenance and autograd. Raw timings include Python allocation/launch costs.
No serving speedup, isolated instruction timing or whole-model throughput is
claimed.

Inputs are deterministic synthetic FP32 selection scores, not checkpoint
weights, raw gate-to-score preprocessing or tokenizer inputs. Task 31 consumes
complete global scores; the caller owns materialization/transport. The semantic
backend is TP1 per complete-input consumer; the two-GPU test does not advertise
a native TP2 sharded-selection kernel. Selection bias, normalization/scaling,
capacity/overflow and auxiliary loss remain tasks 32–34; complete routing
backward and P6 gradient merging remain separate.

MI355X/gfx950, NVIDIA, Graph, multi-node/framework integration, LoRA updates
and full-model logprob acceptance are unvalidated. See
[the operator contract](../../../docs/operators/stable-topk6.md) for the direct
mapping from item-31 requirements to implementation/tests and reproduction.
