# Qwen3-Next TP4 mixers: real-weight gates and attention prior art (B200)

| Directory | Measured at | Produced by |
| --- | --- | --- |
| `tp4-gdn/` | `9e6c4f2` | `torchrun --nproc-per-node 4 scripts/qwen3_next_tp_gdn_check.py --checkpoint <ckpt> --output tp4-gdn` |
| `tp4-attention/` | `9e6c4f2` | `torchrun --nproc-per-node 4 scripts/qwen3_next_tp_attention_check.py --checkpoint <ckpt> --output tp4-attention` |
| `tp4-moe/` | `9e6c4f2` | `TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 scripts/qwen3_next_tp_moe_check.py --checkpoint <ckpt> --output tp4-moe` |
| `attention/report.json`, `attention/figure.png` | `6517077` | `python scripts/qwen3_next_attention_prior_art.py --out attention/report.json`, then `scripts/plot_qwen3_next_attention_prior_art.py` |

Every file records its commit and `tracked_tree_dirty: false`; each run used a
clean clone with its own `rl_engine._C` built in place, on B200 GPUs (driver
580.126.20), torch 2.13.0+cu130, vLLM 0.30.0, NCCL 2.29.7. `<ckpt>` is the
official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`: the GDN and MoE gates read layer 0,
the attention gate layer 3. `6517077` adds only the attention prior-art runner
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
`backward_repeat_bitwise` repeats the target's backward after the companions'
backwards ran. Accuracy is against FP64; latencies are CUDA-event medians with
the candidate order reversed every iteration.
