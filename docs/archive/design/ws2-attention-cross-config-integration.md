# WS2 Attention Cross-Configuration Integration

Implements PR4 of [#235](https://github.com/RL-Align/RL-Kernel/issues/235): wiring the
CP attention path into the cross-configuration planner/runtime for the Qwen3-8B
TP=2 CP=2 BF16 target.

Builds on [#236](https://github.com/RL-Align/RL-Kernel/pull/236) (attention contract
and dispatch metadata), [#238](https://github.com/RL-Align/RL-Kernel/pull/238)
(deterministic CP reference) and [#230](https://github.com/RL-Align/RL-Kernel/pull/230)
(cross-configuration framework).

## What "bind to the same contract" means here

The PR4 acceptance criteria say rollout and training descriptors must "bind to the
same semantic attention contract". Under the frozen deployment the two sides can
never produce identical `AttentionContract` instances:

| | training (Megatron) | rollout (vLLM) |
| --- | --- | --- |
| mode | full-sequence prefill | chunked prefill, later decode |
| CP | `context_parallel_size`, whole forward | `prefill_context_parallel_size`, prefill only |
| KV | no paging | paged KV with a block table |
| backend vocabulary | `AttnBackend{flash,fused,unfused,local,auto}` | `AttentionBackendEnum` |

Read literally, the criterion is unsatisfiable. It is therefore implemented as three
tiers, in `rl_engine/integrations/common/attention_binding.py`:

| tier | fields | rule | failure |
| --- | --- | --- | --- |
| `IDENTICAL` | checkpoint/model/token identity plus complete TP/CP GQA head and sequence ownership | equal bit for bit | `comparable=False`; no drift number from the pair means anything |
| `SEMANTIC` | reduction contract, dtype, exported LSE, first-class Split-KV request, complete actual Split-KV plan sets, CUDA QK-Norm/RoPE identity, cross-side determinism | both sides equal **and** equal to the WS2 mandate | fail closed |
| `RECORDED` | `mode`, `backend_id`, `reduction.engine`, RoPE materialization state and fusion boundary, KV-cache paging | free to differ | recorded into provenance and measured |

Two placements are load-bearing:

* **`reduction.engine` is `RECORDED`, not `SEMANTIC`.** Training may run the in-op
  deterministic reference while rollout runs a Transformer Engine merge oracle.
  Forcing them equal would defeat the oracle comparison that #235 PR2/PR3/PR5/PR6
  depend on.
* **TP/CP topology is `IDENTICAL`, not `RECORDED`.** TP selects local Qwen3 GQA head
  ownership and CP selects local sequence ownership. Different topology is a
  different local attention problem, not a backend detail.
* **`reduction.order`, `reduction.acc_dtype`, and actual Split-KV schedules are
  `SEMANTIC`.** Both runtimes must export the complete batch x TP x CP x KV-owner
  plan set, including logical boundaries, merge order, FP32 accumulation, final
  downcast, and fallback state. Configured policy alone never passes strict binding.

`comparable` and `passed` are separate flags. A pair with mismatched identity is not
comparable. A pair that is comparable but violates the reduction mandate is still
rejected -- the drift would be real but attributable to the wrong thing.

## H100 Attention input and projection boundary

Megatron/TE and vLLM/FlashInfer are the first-choice implementations.
`H100AttentionPreprocessor` runs a same-input H100 bitwise probe against the
deterministic RL-Kernel path. An unavailable native callable, a native exception,
or a failed probe switches both sides to `RMSNormCudaOp` + `RoPESM90Op` and
records the fallback reason and probe ID. The launcher passes the returned
backend evidence into `AttentionRuntimeReadback`:

```python
from rl_engine.ops.attention.preprocess import H100AttentionPreprocessor

prepared = H100AttentionPreprocessor(device)(
    q, k, q_norm_weight, k_norm_weight, position_ids
)
readback = AttentionRuntimeReadback(
    # contract, knobs, Split-KV plan set, source, and scope fields omitted here
    **prepared.readback_fields(),
)
```

Strict binding rejects a missing or unknown backend and rejects mixed native /
fallback execution. If both sides fall back, they must report the same deterministic
backend IDs and policy ID. Printing a configured backend without executing the
probe is not evidence.

The Attention boundary includes QKV projection, Q/K RMSNorm, RoPE, core
attention, KV-cache access, CP `(Out, LSE)` communication/merge, and o_proj.
`AttentionProjectionOp` freezes QKV/o_proj to BF16 input and output, FP32
accumulation, ascending-K reduction, and Split-K disabled. Native projection
callables are accepted only after a bitwise probe against `DetGemmOp`; otherwise
both sides use the deterministic fallback. Its collective contract records QKV
column-parallel plus backward TP all-reduce, o_proj row-parallel partial output,
and the SP all-gather/reduce-scatter directions. The model input RMSNorm and
residual add remain outside this Attention experiment.

## Determinism is not one thing

`rl_engine/validation/cross_config/determinism.py` probes both sides and compares
them, because the two frameworks mean different things by "deterministic":

| | Megatron `deterministic_mode` | vLLM `VLLM_BATCH_INVARIANT` |
| --- | --- | --- |
| `NCCL_ALGO` | asserts membership in a five-value set | hard-sets `allreduce:tree` |
| `NCCL_PROTO`, channels, threads | not managed | hard-set (`Simple`, `1`, `1`) |
| TF32 | **not managed at all** | disabled (`fp32_precision="ieee"`) |
| BF16 reduced-precision reduction | not managed | disabled |
| cuBLAS workspace / BLAS library | not managed | `:4096:8`, cuBLASLt |
| GEMM | cuBLAS / TE | Triton `matmul_persistent` |
| FlashAttention | forbidden | permitted |

`NCCL_ALGO`, `NCCL_PROTO` and `CUBLAS_WORKSPACE_CONFIG` change arithmetic, so a
mismatch there is blocking. The remaining differences -- including the TF32 and
BF16-reduction asymmetry, which under a pure BF16 GEMM path does not fire -- are
recorded so the asymmetry appears in every artifact rather than being invisible.

## Runtime adapters

Before this PR the only `RuntimeMaterializer` in the repository was
`CpuSmokeMaterializer` over a synthetic CPU model, and every named scenario
(`S1`/`S2`/`S3`) was planning-only. This PR adds the first two framework-shaped
adapters:

* `adapters/megatron.py` -- `MegatronProvenanceAdapter` (construction and
  distributed-context fingerprints, determinism probe, frozen-scope assertions) and
  `MegatronAttentionMaterializer`.
* `adapters/vllm.py` -- `VllmProvenanceAdapter` (including diagnostic vLLM split
  limits) and `VllmRolloutMaterializer`.
* `AttentionRuntimeReadback` -- the explicit handoff from an executed engine. It
  carries the reconstructed actual contract, actual knob values, frozen-scope
  verification, executed CUDA QK-Norm/RoPE identities and fallback state, and the
  complete Split-KV runtime plan set.

Constructing a contract is not runtime verification. Without a readback, adapter
applications are `UNOBSERVABLE`; only matching values reconstructed from a real
Megatron or vLLM execution are `APPLIED`. `bind_attention_runtime_readbacks` is the
strict public entry point used after both framework launchers collect that evidence.

Neither module imports `megatron` or `vllm`; configs are duck-typed, so the binding
rules are exercised on CPU in CI rather than only on a 2-node cluster.

## Fail closed, never substitute

`unsupported_reduction_reason` rejects requests that #236 cannot express, instead of
collapsing them onto the supported value:

| request | status | why |
| --- | --- | --- |
| `attention.reduction_order=arrival` | `UNSUPPORTED` | the control group must stay distinguishable from the treatment |
| `attention.reduction_downcast_at=per_block` | `UNSUPPORTED` | `DowncastPoint` declares only `final_write` |
| `attention.reduction_engine=te_oracle` | `UNSUPPORTED` | the TE merge oracle lands in #235 PR2/PR3; PR4's TE plan is provenance only |
| `attention.reduction_acc_dtype=bf16` | `UNSUPPORTED` | the CP `(out, lse)` merge accumulates in FP32 |
| configured contract without runtime readback | `UNOBSERVABLE` | requested values do not prove what executed |
| `rollout.context_parallel_size>1` with effective decode CP=1 | `ERROR`/`FALLBACK` | strict TP=2/CP=2 acceptance rejects the topology change |
| missing/mismatched/fallback Split-KV plan set | binding failure | Split-KV provenance must cover every batch/TP/CP/owner coordinate |
| missing/unknown QK-Norm or RoPE backend, or mixed native/fallback sides | binding failure | both sides must execute the same verified native policy or the common RL-Kernel CUDA fallback |

## Knobs

`adapters/knobs.py` extends `V1_KNOBS` additively. Added: training-side
`tensor_parallel_size` / `context_parallel_size` / `deterministic_mode` /
`cp_comm_type`, `rollout.batch_invariant` / `rollout.kv_block_size`, and the
reduction axis (`acc_dtype`, `order`, `downcast_at`, `engine`) plus
`attention.fusion_boundary` and `attention.split_kv_policy`.

`training.attention_backend` keeps its path but its value domain is replaced with
Megatron's `AttnBackend`; the HuggingFace names have no Megatron counterpart, so this
is a replacement rather than a mapping.

Not done here, because both change `V1_KNOBS` itself and would break existing
cross-config tests: removing `training.sharding` (Megatron has no such concept, and
DP=1 makes it moot) and renaming `rollout.context_parallel_size` to reflect that it
binds to `prefill_context_parallel_size`.

## Scenario

`configs/experiments/cross_config/cross_config_qwen3_8b_megatron_tp2_cp2_vllm.json` supersedes
`cross_config_s1_distributed_smoke.json` and
`cross_config_s3_qwen3_8b_tp4_cp4_bf16.json`, whose training sides used `sdpa` /
`flash_attention_2` and `sharding: fsdp` -- none of which exist under Megatron -- and
whose TP=4/CP=4 topology does not match the target.
`cross_config_s2_vllm_tp_vs_fsdp.json` has no Megatron-only counterpart and should be
retired rather than rewritten.

## Out of scope

Deliberately not in this PR:

* launching `torchrun`, initializing process groups, or executing core attention;
* pre-attention model RMSNorm and residual add;
* decode-mode materialization, which needs the validated `KVCacheSpec` from #235 PR6
  and is refused with that reference rather than stubbed;
* distributed drift benchmarks and report artifacts (#235 PR5);
* fused production backend alignment (#235 PR7) and backward (#235 PR8).
