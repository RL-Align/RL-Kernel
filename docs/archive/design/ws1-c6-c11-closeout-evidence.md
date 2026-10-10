# WS1 C6–C11 closeout evidence

**Parent:** #266 · **Branch:** `feat/ws1-c6-c11-closeout-266`

C9 green is assembly only. H20 C8 and both C10 backend profiles are green on
`fdf5bcc5165820abb506291a29370225306514ca`. Full WS1 EXIT still requires the
final-commit required GitHub GPU CI run plus parent A/B/Final comments.

## What landed

| ID | Code | CPU evidence |
| --- | --- | --- |
| C6 | `rl_engine/validation/operators/kv_consistency.py`, `tools/validation/operators/check_decode_prefill.py` | `tests/validation/operators/test_kv_consistency.py` |
| C7 | `StatefulKVCache` + same harness, `tools/validation/operators/check_stateful_kv.py` | B1 writer/reader + generate-rescore; B2=`absent` |
| C9 | `rl_engine/models/qwen3/reference.py`, `tools/validation/models/ws1_chain_fwd_bwd.py` | topology / official fingerprint / profile resolution |
| C10 | `rl_engine/validation/models/chain_gate.py`, `tools/validation/models/ws1_chain_gate.py` | report schema + bitwise `atol=0` rule |
| C11 | `ci/scripts/run_ws1_chain_gate.sh`, `.github/workflows/ws1-chain-gpu.yml` | CPU schema jobs in `ci.yml`; GPU job is required |

Private `_DECODE_ATOL` / `_PADDING_ATOL` were removed from
`tests/ops/attention/test_kv_cache_attention.py`. Those checks now resolve C1
`forward_accuracy` / attention.

## Local GPU evidence (not EXIT)

NVIDIA GeForce RTX 4070 Ti SUPER, CC 8.9, this branch:

| Gate | Profile | Result |
| --- | --- | --- |
| C6 `check_decode_prefill.py` | `cuda_bf16` | passed; 6/6 cells; attn max_abs = 0 |
| C6 `check_decode_prefill.py` | `triton_cuda_bf16` | passed; 6/6 cells; attn max_abs = 0 |
| C7 `check_stateful_kv.py` | `cuda_bf16` | B1 + generate-rescore passed; B2=`absent` |
| C7 `check_stateful_kv.py` | `triton_cuda_bf16` | B1 + generate-rescore passed; B2=`absent` |

C9/C10/C11 full 36-layer + pinned weights were **not** run here (16 GB card).
That execute is the remaining closeout step on H20.

Local pytest totals are command-scoped development evidence, not an EXIT
criterion. Always quote the exact test command and commit with a pass count;
do not cite a bare aggregate such as `220 passed` as closeout evidence.

## H20 technical execute

Environment: NVIDIA H20 (CC 9.0), driver 580.76.05, CUDA 12.8,
PyTorch 2.8.0+cu128, Triton 3.4.0, pinned Qwen3-8B weight hash
`fc664a19c52c82b6f5ddb33d4fe2723181daeb93a344b16fee6369963e5a13a5`.

| Gate | Profile | Result |
| --- | --- | --- |
| C8 four-judgment matrix | CUDA + Triton | green=176, N/A=16 (pack), red=0 |
| C10 full model | `cuda_bf16` | passed, `first_drift=null`, clean SHA |
| C10 full model | `triton_cuda_bf16` | passed, `first_drift=null`, clean SHA |

Each C10 profile passed 7/7 forward-invariance rows, 2394/2394 parameter-gradient
invariance rows, 2/2 FP32 forward-accuracy rows, 798/798 FP32 gradient-accuracy
rows, 8/8 accuracy aggregates, all three train/infer aggregates, BN
train/infer parity, and 6/6 decode/prefill cases. The four Batch/Chunk cells
execute separate real paths; chunked training runs per-layer chunked operations
and verifies its logits against stateful chunked prefill before backward.

These local H20 results establish C10 technical readiness. They do not replace
C11's required GitHub workflow success and workflow URL on the final commit.

## H20 reproduction commands

Pinned Qwen3-8B snapshot (C2 revision + shard hashes):

```bash
export QWEN3_8B=/path/to/Qwen3-8B   # safetensors at revision b968826d9c46dd6066d109eabc6255188de91218
export KERNEL_ALIGN_FORCE_SM90=1

python tools/weights/prepare_ws1_weights.py --output "$QWEN3_8B" --verify-only

python tools/validation/operators/check_decode_prefill.py --backend-profile cuda_bf16
python tools/validation/operators/check_decode_prefill.py --backend-profile triton_cuda_bf16
python tools/validation/operators/check_stateful_kv.py --backend-profile cuda_bf16
python tools/validation/operators/check_stateful_kv.py --backend-profile triton_cuda_bf16

python tools/validation/models/ws1_chain_fwd_bwd.py --backend-profile cuda_bf16 --weights hf --weights-path "$QWEN3_8B"
python tools/validation/models/ws1_chain_fwd_bwd.py --backend-profile triton_cuda_bf16 --weights hf --weights-path "$QWEN3_8B"

python tools/validation/models/ws1_chain_gate.py --backend-profile cuda_bf16 --model qwen3-8b-dense --dtype bfloat16 --weights required --weights-path "$QWEN3_8B" --json
python tools/validation/models/ws1_chain_gate.py --backend-profile triton_cuda_bf16 --model qwen3-8b-dense --dtype bfloat16 --weights required --weights-path "$QWEN3_8B" --json
```

Bind the two C10 JSON files + `git rev-parse HEAD` + GPU/CC on the parent
issue before closing #266.

Each accepted JSON must report `backward_executed=true`,
`train_infer_executed=true`, `accuracy_executed=true`,
`gradient_accuracy_executed=true`, a null `first_drift`, verified weight
content hash, complete runtime observations for all nine required node kinds,
backward runtime kernel identities for `lm_head` / `rms_norm` / `det_gemm` /
`embedding`, `gpu_name`, representative `case_id`s, C8 evidence path, and
`git_dirty=false`. Schema is `ws1-c10-c11-v5`. Generate the JSON only after
committing the gate code; the CI wrapper rejects dirty-worktree evidence.

C10 compares `tensor.grad` after a real `loss.backward()` for every official
Qwen3-8B Dense trainable leaf (`gradient_scope=all_required_trainable_parameters`,
`all_parameter_gradients=true`): embedding, final norm, LM head, and all 36
layers of Q/K/V/O, QK-norm, RMSNorm, and MLP weights. Logprob accuracy vs the
FP32 gold cell is judged only by `max_abs_dlogp` / `approx_kl0` /
`clipfrac0`. Full-model decode/prefill covers short, long, varlen, left/right
padding, and B=1/N. Backward runtime records the kernel that actually ran
(`det_gemm` / RMSNorm / embedding), not a class-attribute string.

Gradient snapshots are copied to CPU in their native dtype rather than retained as
FP32 CUDA tensors, and their payloads are released after comparisons while report
keys remain. The chain GPU job generates C8 outside the repository, then C11 loads `c8_evidence_path` and requires its commit to equal
C10 `git_sha` with both worktrees clean. Packed forward and gradient accuracy must
appear explicitly as `BN/packed` versus `fp32_reference`.

## Not claimed

- Multi-GPU / vime / real vLLM vs Megatron
- C9 skeleton alone as EXIT
- Production paged-KV (C7 B2 is explicitly absent)
- C11/public EXIT until the required final-commit GitHub GPU workflow is green
