# Qwen3-Next distributed logprob parity, TP4 (B200)

One node with four B200 GPUs, torch 2.13.0+cu130, vLLM 0.30.0, NCCL 2.29.7, the
official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`, 1024 tokens. Every run used a clean
clone with `rl_engine._C` built in place.

| Directory | Measured at | Produced by |
| --- | --- | --- |
| `parity/runs/<setting>/rank-{0..3}.json` | `04db726` | `<NCCL env> torchrun --nproc-per-node 4 tools/validation/models/qwen3_next_logprob_parity.py --checkpoint <ckpt> --output parity/runs/<setting>` |
| `parity/report.json` | `04db726` | `python tools/validation/models/qwen3_next_logprob_parity_compare.py --runs parity/runs/* --out parity/report.json` |
| `prior-art/runs/<launch>/rank-0.json` | `33495ad` | `<NCCL env> torchrun --nproc-per-node 4 tools/validation/models/qwen3_next_logprob_prior_art.py --checkpoint <ckpt> --output prior-art/runs/tp4-<setting>`, and `--nproc-per-node 1` for `tp1` (training environment: Megatron-core 0.16.0rc0, VIME 0.3.2) |
| `prior-art/report.json` | `33495ad` | `python tools/validation/models/qwen3_next_logprob_prior_art.py --merge prior-art/runs/tp4-default prior-art/runs/tp4-<other settings> prior-art/runs/tp1 --out prior-art/report.json` |
| `figure.png` | — | `python tools/validation/models/plot_qwen3_next_distributed_logprob.py docs/usage/evidence/qwen3-next-distributed-logprob-b200` |

The 13 NCCL settings are the defaults plus `NCCL_ALGO` Ring and all-reduce Tree
(`allreduce:tree`; NCCL 2.29 has no Tree all-gather) × `NCCL_PROTO` LL, LL128,
Simple × one channel (`NCCL_MAX_NCHANNELS=1`) or the default channel count.
`33495ad` and `04db726` differ only in the parity check and the prior-art
runner; the measured code is the same. The copy here was made with
`tools/validation/models/qwen3_next_publish_parity_evidence.py`, which replaces
the extension's absolute path with `<PATH>` (`redaction.json`) and changes no
other byte. `SHA256SUMS` covers every file except this README and `figure.png`.

## Results

- Every TP4 cross-rank point -- embedding forward and weight gradient; GDN
  layer 0, full-attention layer 3 and MoE layer 0 forward and input gradient;
  selected logprob, LSE, entropy and logits gradient; the gathered sampling
  logits -- has one digest per rank across all 13 settings
  (`parity/report.json`: `deterministic_points_invariant: true`).
- TP4 selected logprob and LSE equal the TP1 computation on the same logits
  (`tp4_equals_tp1: true`); entropy does not (`entropy_tp4_equals_tp1: false`),
  as documented.
- Negative control: the same real attention `o_proj` partials summed with NCCL
  `all_reduce` give 5 different results across the 13 settings
  (`nccl_all_reduce_varies: true`).
- Prior art (`prior-art/report.json`): RL-Kernel WS2 and vLLM's all-gather +
  `log_softmax` are TP-degree and NCCL-setting invariant; Megatron-core
  `vocab_parallel_cross_entropy` and VIME's default logprob are neither (2
  results over 13 settings) and differ from the other two by up to 9.5e-7. All
  four are batch invariant and within 8.1e-7 of FP64.
