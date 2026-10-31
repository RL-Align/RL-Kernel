# DSV4 fixed-order TP/SP/DP collectives (opt-in)

`Dsv4CollectiveContract` seals the ordered communication phases, global rank
partitions, fixed input layouts, model version, memory limit, and loss divisor.
`Dsv4ParallelCollectives` executes that plan using the existing
`RCCLDeterministicCollective` on ROCm. It uses the repository's balanced-rank
tree arithmetic and rank-ordered AllGather transport rather than implementing
another numerical collective kernel. CPU/Gloo is available only by explicit
`allow_cpu_reference=True` for tests.

The caller must create the TP, SP, and DP process groups in a common global
order, then pass the local group for each dimension. The supplied world group
contains exactly the plan's ranks; a PP stage can pass its stage-local group.
All ranks construct the same contract, call `begin_step(step_id,
model_version=...)`, execute each named phase in order, and call `end_step()`.
Control-plane signatures are exchanged before data collectives to reject
version/order/layout drift. A distributed failure poisons the runtime and
requires reconstruction. `close()` releases the existing collective resources.

For loss normalization, only a DP FP32 AllReduce phase may declare a positive
denominator. Division is applied **after** fixed-tree reduction. The caller
supplies the intended global token/sequence denominator; this adapter does not
infer it from local batches. A change to token accounting requires a new plan.

The current profile explicitly rejects CUDA Graph capture and unsupported
backend/group sizes/dtypes. Empty ReduceScatter and asynchronous overlap have
not been signed off. Empty AllGather and AllReduce use the synchronized phase
handshake and skip the data transport. Control-plane object gathers cost latency; eliminating
them requires a separately validated static schedule and belongs to item 42.
This adapter is not yet wired into a full DSV4 optimizer/training step.

## Validation (MI300X `rl-kernel-ljj` container)

- Four-rank Gloo reference: TP AllGather, SP AllReduce, DP FP32 mean,
  DP ReduceScatter, plus version/phase/shape mismatch rejection.
- One-rank ROCm NCCL process group: all three operations on MI300X.
- Related distributed collective tests and lint should be run before merge.
- Multi-GPU ROCm, 8+8 multi-node, fault recovery, and model-level throughput
  are **not** validated: this container currently exposes one GPU.

This PR is an incremental item-40 implementation, not full acceptance of KLR
#81. It targets `dsv4-rocm` after the branch absorbs its needed `main` commits.
