# Qwen3-Next selected-token logprob comparison (B200)

## Reuse decision

Measured with `benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py` on the
same official-LM-head logits (TP4 slices of one full-vocabulary product), at TP1
and at TP4 under the same 13 NCCL settings as the parity run:

| Implementation | TP4 == TP1 | NCCL-setting invariant | Batch invariant | Matches the others |
| --- | --- | --- | --- | --- |
| RL-Kernel WS2 (`shared_logprob`) | yes | yes | yes | — |
| vLLM sampler (all-gather + `log_softmax`) | yes | yes | yes | no |
| Megatron-core `vocab_parallel_cross_entropy` | no | no (2 results over 13 settings) | yes | no |
| VIME default `calculate_log_probs_and_entropy` | no | no (2 results over 13 settings) | yes | Megatron only |

vLLM's sampler path is exact on the rollout side, but VIME's training side has
no counterpart that reproduces it: the Megatron/VIME paths reduce max and
sum-exp with NCCL all-reduce and differ from it. Using one provider for both
engines is the only measured way to equal bits, so both engines keep the WS2
provider. Its cost is latency (2.55 ms versus 0.57 ms for Megatron at 1024 tokens); a faster tile-stats kernel is
follow-up work and would keep the same merge order.

All four agree with an FP64 log-softmax to within 8.1e-7 and the largest
difference between any two of them is 9.5e-7: the defaults are close, but not
equal, and only equal bits remove the replay/rollout difference the RFC targets.

## Evidence and reproduction

Measured at `c5cea58` on 4 × B200 with the official
`Qwen/Qwen3-Next-80B-A3B-Instruct` checkpoint revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`, torch 2.13.0+cu130,
vLLM 0.30.0, NCCL 2.29.7, Megatron-core 0.16.0rc0 and VIME 0.3.2.
The plotted TP parity data was measured at `99c244a`.

[Download raw results and figure (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-pr549-evidence/qwen3-next-distributed-logprob-b200.zip).
ZIP SHA-256: `a48bf29d19e40d22178185de81dcfacdd9138ca03f5dc347d23eed4a4fbd2cd9`.
The archive retains the original `SHA256SUMS` and report provenance. Extract it
under `artifacts/` before plotting. These are historical measurements; moving
runners does not certify a new GPU run. At the tested commit the runner lived
directly under `tools/validation/models/`; current commands are:

```bash
# Repeat TP4 for defaults and Ring / allreduce:tree × LL / LL128 / Simple
# × NCCL_MAX_NCHANNELS=1 / unset. See the TP parity report for those settings.
torchrun --nproc-per-node 4 benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \
    --checkpoint <ckpt> --output <runs>/tp4-default
torchrun --nproc-per-node 1 benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \
    --checkpoint <ckpt> --output <runs>/tp1
python benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \
    --merge <runs>/tp4-default <runs>/tp4-<other-settings> <runs>/tp1 --out report.json
python benchmarks/models/qwen3_next/plot_qwen3_next_distributed_logprob.py \
    artifacts/qwen3-next-distributed-logprob-b200
```

The comparison runner requires the measured engine versions above. Plotting
requires matplotlib and NumPy. The package contains comparison candidates,
the measurement/merge runner and the figure renderer; TP correctness gates
and their CI tests remain in the validation PR (#549).

## Post-review rerun (2026-10-12)

The relocated benchmark ran from clean commit
`67a45169a64e85bea0333dba7c446d5f74564222`, at TP1 and TP4 under all 13 NCCL
settings, with the same pinned environment, checkpoint and 1024-token shape
listed above. The extension was freshly built from validation commit
`1106e461`; the benchmark snapshot's native source files were checked to be
identical before reusing that binary.

[Download new raw reports and execution logs (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-pr549-evidence/qwen3-next-pr549-gpu-1106e461.zip).
ZIP SHA-256: `5fb49bc151fc0224dc92128c0279fb58998435cfa033b62565d9a62957be0e57`.
The current benchmark and merge commands above reproduce this run. Extract
its `parity/` and `prior-art/` directories together to render the new figure.

| Candidate | TP4 median latency | TP4 == TP1 | NCCL-setting invariant |
| --- | --- | --- | --- |
| RL-Kernel WS2 | 2.571 ms | yes | yes |
| vLLM gather | 1.511 ms | yes | yes |
| Megatron CE | 0.570 ms | no | no |
| VIME default | 0.565 ms | no | no |

All four candidates remained batch invariant and within 8.14e-7 of FP64;
the maximum pairwise logprob difference was 9.54e-7. These fresh measurements
confirm the historical comparison without changing its numerical conclusions.
