# Ragged deterministic GEMM trees on gfx942

The two-level ROCm reduction fusion assumes a complete binary tree. Midpoint
splits such as K=131 produce leaves at different depths. The previous device
plan attempted to find a leaf in the preceding level's internal-node table
and raised `KeyError: 0`. A real four-layer verl actor triggered this in the
vocabulary head's weight-gradient GEMM (the reduction dimension is token count).

The fix applies two-level fusion only to complete trees. Ragged trees keep the
same midpoint arithmetic and use the existing level-by-level GPU reductions.
This also applies to eager inference; skipping fused-plan construction alone
would leave intermediate nodes unwritten on that schedule.

```bash
python -m pytest -q tests/backends/rocm/test_det_gemm_ragged.py
```

MI300X: 3 passed, 0 failed, 0 skipped. K=65, 131 and 1537 exercise ordinary,
inference and vocabulary-head weight-gradient paths against the independent
FP32-leaf/BF16-node midpoint reference using exact bytes. The real original
four-layer TrainingWorker then completed backward and an Adam update.
