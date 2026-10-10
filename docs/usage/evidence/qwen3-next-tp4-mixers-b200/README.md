# Qwen3-Next TP4 mixers: real-weight gates and attention prior art (B200)

| Directory | Measured at | Produced by |
| --- | --- | --- |
| `tp4-gdn/` | `9e6c4f2` | `torchrun --nproc-per-node 4 scripts/qwen3_next_tp_gdn_check.py --checkpoint <ckpt> --output tp4-gdn` |
| `tp4-attention/` | `9e6c4f2` | `torchrun --nproc-per-node 4 scripts/qwen3_next_tp_attention_check.py --checkpoint <ckpt> --output tp4-attention` |
| `tp4-moe/` | `9e6c4f2` | `TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 scripts/qwen3_next_tp_moe_check.py --checkpoint <ckpt> --output tp4-moe` |
| `attention/report.json`, `attention/figure.png` | `703c6dc` | two processes, then `scripts/plot_qwen3_next_attention_prior_art.py attention/report.json` (see below) |

Every file records its commit and `tracked_tree_dirty: false`; each run used a
clean clone with its own `rl_engine._C` built in place, on B200 GPUs (driver
580.126.20), torch 2.13.0+cu130, vLLM 0.30.0, NCCL 2.29.7. `<ckpt>` is the
official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`: the GDN and MoE gates read layer 0,
the attention gate layer 3. `703c6dc` adds only the attention prior-art runner
and plot on top of `9e6c4f2` (and the MoE evidence merge).

In the job that ran the gates, the mixer GPU checks
(`tests/check_qwen3_next_{attention,gdn_bridge,gdn_sequence,conv_bridge,core_matrix,shared_core,forward}.py`)
and the CPU tests passed together (123 tests); `tests/test_framework_operator_integrations.py`,
collected into the same process after vLLM was imported, failed as it must and
passes in its own process. The existing attention suites
(`test_deterministic_attention_cuda`, `test_attention`, `test_attention_correctness`,
`test_kv_cache_attention`, `test_attention_contract`, `test_attention_dispatch`)
gave 766 passed and the two `test_attention_dispatch` failures that `main` also has.

## Attention prior-art report

One TP4 rank of Qwen3-Next full attention: 4 query heads, 1 KV head, D=256,
scale 1/16, causal. A 1000-token target sequence is computed alone and in a
batch with sequences of 777, 1500 and 64 tokens (first and last);
`prefill_decode_bitwise` compares the last 64 rows and the last row computed
against the full KV with the same rows of the full prefill;
`backward_batch_bitwise` compares the target's dq/dk/dv computed alone with the
same sequence packed first and last among the companions (one launch where the
engine has a packed backward). Accuracy is against FP64; latencies are CUDA-event medians with
the candidate order reversed every iteration.

The report was produced in two processes of the same job and merged:

```bash
python scripts/qwen3_next_attention_prior_art.py \
  --only rl_kernel_cuda,torch_sdpa,vllm_fa2_auto,vllm_fa2_split1,vllm_triton_2d,flashinfer,fa4_cute,te_training,megatron_local \
  --out report-main.json
python scripts/qwen3_next_attention_prior_art.py --only te_fused --merge report-main.json --out report.json
```

Both run in the training environment (vLLM 0.30.0, Megatron-core 0.16.0rc0,
Transformer Engine 2.16.1, cuDNN 9.20) with `CUDNN_HOME` pointing at the
`nvidia-cudnn-cu13` wheel. Notes on the engines added in this revision:

- **FlashAttention-4** is the CuTe-DSL build vendored by vLLM
  (`vllm.vllm_flash_attn.cute`), varlen forward and its default backward. Its
  `deterministic=True` backward asserts for head dim 256 on SM100.
- **Transformer Engine**: with cuDNN 9.20 on SM100, TE offers its fused cuDNN
  backend for head dim 256 only in inference mode; in training mode it selects
  the unfused PyTorch fallback (`NVTE_DEBUG=1` shows the selection). Both are
  measured. The unfused fallback raises a shape error when the query is shorter
  than the KV (`padding_causal_bottom_right`, THD), recorded as
  `prefill_decode_failed`. `NVTE_ALLOW_NONDETERMINISTIC_ALGO` only affects the
  fused backward, which does not exist here, so it is not a separate row.
- The fused TE row runs in its own process because cudnn-frontend's runtime
  loader refuses to start when both `libcudart.so.12` and `libcudart.so.13`
  can be `dlopen`ed, and the compute nodes' system image provides a CUDA 12
  runtime next to the CUDA 13 one this stack uses. That process had a
  non-loadable `libcudart.so.12` stub first on `LD_LIBRARY_PATH`; only
  `libcudart.so.13` was loaded in either process. Its latencies come from that
  process, not interleaved with the other candidates.
- **Megatron-core**'s own `DotProductAttention` (not the TE extension) has no
  packed-sequence path, so it runs one launch per sequence with a bottom-right
  causal mask; its batch invariance follows from that.
