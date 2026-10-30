# H100 rollout context parallelism

Rollout TP4/CP2 now completes two actual optimizer updates with strict byte
equality against rollout TP4/CP1, using training TP4/CP2 in both runs.
The implementation reuses `collective_for_group` / `all_gather_many` from the
existing training CP transport and the existing strict FA4 paged runtime.
It adds only the vLLM interleaved-page and query/output layout adapter.

KV remains sharded by logical token across CP ranks. The adapter gathers the
needed local pages, restores logical KV order, divides queries among CP ranks,
and gathers outputs without a floating-point attention merge. CUDA Graph
capture uses preallocated IPC storage. Signature checks remain active outside
capture; vLLM broadcasts identical scheduled batches to the PCP workers.
The separate vLLM decode-CP setting remains 1; PCP-sharded KV is maintained
during generation as well as prefill.

## Subsequent topology validation

The [final H100 matrix](h100-matrix-validation.md) records the fixed-TP8,
two-step full-model checks across training and rollout topologies. It includes
the CP8 gradient fix and singleton-TP-group correction. The measurements below
are earlier 512-response-token transport evidence and should not be mixed with
the final matrix's 128-token timing numbers.

## Measured results

`pcp2-ipc-r3` and reference `cpgrad-tp4cp2-r5` both succeeded. All three weight
versions compare equal across all 399 exported tensors. Both generated-token
sequences and gradient norms match exactly; the formal validator reports
zero training/rollout logprob byte mismatches across 8,192 active tokens.

| Step | Both runs' gradient norm | CP1 entire step | CP2 entire step |
|---|---:|---:|---:|
| 0 | 0.022594607365076035 | 45.082 s | 75.120 s |
| 1 | 0.03439536852782456 | 41.616 s | 72.066 s |

Mean entire-step time is 43.349 s versus 73.593 s (+69.8%). These include
full weight export/reload/auditing and are two-step measurements, not native
steady-state throughput. The earlier NCCL implementation took about 226 s
for generation alone; it has been replaced with the existing IPC transport.
Full decode graphs currently materialize KV to the model context bound,
which leaves a significant performance gap for short live sequences.

An additional eight-GPU attention test passes 18 CP2/4/8, interleave1/8/16,
prefill/decode cases, each in eager and CUDA Graph execution. It compares
output bytes against CP1 FA4, including ragged sequences and fewer queries
than CP ranks. This is not full-model validation of every topology.

## Run

```bash
./rlk verify --tp 4 --rollout-tp 4 --rollout-cp 2 --temperature 0.7 --top-p 0.95
```

Temperature, top-p and top-k continue to use the existing configurable
sampling path. The observed full-model run used response cap 512 and KL .01;
all responses were truncated and the nonzero gradients came from KL.
ROCm PCP and exhaustive topology/sampling/long-training coverage remain
unvalidated. See [the evidence](evidence/pr432-20260920/h100-pcp.json).
