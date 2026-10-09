# Qwen3-Next routed MoE: prior art and TP4 gate (B200)

Measured at commit `2e950f3` from a clean clone (`tracked_tree_dirty: false` in
every file), with that clone's own `rl_engine._C`, on one node with four B200
GPUs (driver 580.126.20), torch 2.13.0+cu130, vLLM 0.30.0, Triton 3.7.1,
transformers 5.17.0, FlashInfer 0.6.18.post1, NCCL 2.29.7.

| File | Produced by |
| --- | --- |
| `report.json` | `TRITON_F32_DEFAULT=ieee python scripts/qwen3_next_moe_prior_art.py --out report.json` (one GPU) |
| `figure.png` | `python scripts/plot_qwen3_next_moe_prior_art.py report.json` |
| `tp4-moe/rank-{0..3}.json` | `TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 scripts/qwen3_next_tp_moe_check.py --checkpoint <Qwen3-Next-80B-A3B-Instruct> --output tp4-moe` |

The checkpoint is the official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`; the TP4 gate reads layer 0's MoE.

In the same job, `TRITON_F32_DEFAULT=ieee python -m pytest -q
tests/check_qwen3_next_forward.py tests/test_tensor_identity.py
tests/test_qwen3_next_forward_contract.py tests/test_qwen3_next_tp_blocks.py`
passed (65 tests). `tests/test_framework_operator_integrations.py` was collected
into that same process and failed, as it must once a `check_` file has imported
vLLM; run in its own process, as CI does, it passes.

## Reading the report

* Batch invariance: 8 probe tokens computed alone, and placed first and last in
  batches of 16, 64, 256 and 1024 tokens; `true` means bitwise equal everywhere.
  `dweight_zero_rows_bitwise` appends rows whose output gradient is zero.
* Accuracy: 256 tokens against the HF formula evaluated in FP64 with FP64
  routing; `tokens_with_different_expert_set` counts tokens whose ten selected
  experts differ from the FP64 selection.
* Latency: CUDA-event medians, candidates interleaved with the order reversed
  every iteration. Weights are random (scale 0.02), shape H=2048, 512 experts,
  top-10, expert width 512 (TP1).
