# Isolated AMD PR 662 integration

The baseline assembly remains separate. This integration applies the 41-file
PR diff at `e2e44e5f32bc956c1903580d2bfaaeb2fa641b88` to that assembly and keeps
its original upstream framework SHAs. It does not modify the AMD branch.

The original native binding failed to select this backend: P1Worker accepted
only `reference`/`rocm`, always requested recomputation, and supplied neither
logical keys nor a canonical backward session. The new opt-in
`rlk_mhc_backend="rocm_triton"` binds pre/post/head to the PR's actual Triton ops;
only split is enabled. The ordinary `rocm` baseline remains available.

The actual actor service declares logical cases before each engine command.
The engine validates its actual [batch,time] input token rows against those
cases, hashes stable sample IDs and supplies time-major (sample, position) keys.
Each of the eight native HyperConnection boundaries and the final head has a
unique parameter ID. The real native forward/backward schedule runs inside one
canonical session, and incomplete parameter submissions raise before the step.
Single-card parameter contributions are ordered lexicographically by these keys.
Native Float16/DDP wrapping must retain all original trainable FP32 controller
Parameters and their `keep_in_fp32` flags; the actual optimizer must register
them. No controller parameter is frozen to bypass canonical backward.

First completed operator calls log the actual Triton class, source file,
weight version and fn dtype. These records require actual execution. Loading
or selecting the backend alone does not count as a GPU result.

The PR's linear changes apply on top of the baseline's ragged midpoint-tree
reduction fix. The merged source retains explicit FP32 accumulators for
incomplete leaves and adds int64 row/column/element addressing. The required
GPU regression includes the original ragged K=65/131/1537 forward and LM-head
weight-gradient cases. RMSNorm remains the separately applied #681 path.

CPU checks: the imported contract/reference/dispatch suite reports 97 passed
and 3 GPU-dependent skips; logical-row/session and worker checks report
7 passed. Skips are not GPU passes. At the time of assembly every MI300X was
occupied by another job. Required operator GPU checks, actual four-layer model
execution, updated score matrix and independent-container repeat remain pending.
No #662 numerical parity or GPU qualification is claimed by these CPU checks.
