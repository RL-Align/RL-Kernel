# H100 canonical CP backward validation

The original 0.021213% gradient-norm discrepancy was between **training
TP2/CP4 and TP4/CP2**, both with rollout TP4/CP1. The new two-step runs have
identical gradient norms, sampled tokens, and all 399 exported model parameters
at initialization and after each Adam update.

| Training topology | Step 0 gradient norm | Step 1 gradient norm |
|---|---:|---:|
| TP2/CP4 | 0.022594607365076035 | 0.03439536852782456 |
| TP4/CP2 | 0.022594607365076035 | 0.03439536852782456 |

All three weight-version comparisons have **0 differing parameter tensors**;
the versions change after each update. Each run also compares 8,192 active
training/rollout logprobs with **0 byte mismatches**. The shared weight hashes
and complete validation records are in
[the evidence JSON](evidence/pr432-20260920/h100-cp-gradient.json).

## Change

Packed CP parameter gradients now reconstruct logical sample/token order,
remove padding, and cancel the CP loss multiplier before summation. This covers
attention projections, FFN, RMSNorm (including TP-sharded Q/K heads), and
embedding gradients. Activation checkpoint recomputation retains the same CP
layout as the original forward pass.

Megatron's L2 norm now sums FP32 gradient squares using integer exponent bins
before the distributed reduction. It controls real gradient clipping; it does
not round logged values or impose a comparison tolerance. NVRTC compiles the
CUDA kernel once per process/device. An HIPRTC route is included but has not
been GPU-validated on ROCm. The H100 runner enables these changes for the
all-RL-Kernel arms; the native baseline remains unchanged. Weight auditing
hashes every exported tensor when update verification is requested.

## Reproduce

With the existing machine profile, run each command after the other completes:

```bash
./rlk run --steps 2 --require-updates --tp 2 --rollout-tp 4 \
  --temperature 0.7 --top-p 0.95 --max-response-len 512 --kl-coef 0.01
./rlk run --steps 2 --require-updates --tp 4 --rollout-tp 4 \
  --temperature 0.7 --top-p 0.95 --max-response-len 512 --kl-coef 0.01
```

Recorded development runs: `cpgrad-tp2cp4-r5`, `cpgrad-tp4cp2-r5`.
These used the isolated working tree based on `6737448b`, with the accompanying
CP/norm changes and `--allow-dirty`; both Ray jobs and acceptance checks passed.
After the full runs, the norm helper gained a device-context guard; its
single-device and alternate-device GPU regression checks both passed.

## Timing and scope

Second-step measured times include complete weight export, reload, and auditing:

| Training topology | Actor training | Entire step |
|---|---:|---:|
| TP2/CP4 | 3.611 s | 42.435 s |
| TP4/CP2 | 3.233 s | 41.616 s |

TP2/CP4 takes about 2.0% longer for this entire step (11.7% for actor training).
Only two steps were measured. The audit adds substantial overhead; these
numbers are not a new controlled comparison against native or steady-state
throughput. A separate 10M-element BF16-derived FP32 norm microbenchmark took
about 0.36 ms on H100.

Responses were capped at 512 tokens and all truncated; nonzero gradients came
from KL=0.01, with zero policy-gradient advantages. This establishes the
observed pair, not every topology, reward scenario, microbatch packing,
temperature/top-p combination, or cross-platform identity. In particular,
those historical runs derived canonical TP from the finer training/rollout
TP and could change packing with CP. The subsequent
[topology matrix](h100-matrix-validation.md) uses fixed virtual TP8 and a
fixed default logical token budget, and fixes the CP8 ring-average rounding.
Explicit packing overrides remain a different numerical configuration.

## Historical rollout CP2 failure and resolution

`vime200-rollout-cp2-verify-r1` actually launched training TP4/CP2 and rollout
TP4/CP2 on eight H100s, with temperature 0.7 and top-p 0.95. vLLM 0.16.0 failed
before generation with:

```text
AssertionError: PCP requires attention impls' support, but the impl RlKernelAttentionImpl does not support PCP.
```

At that revision only the CLI forwarded prefill CP. The subsequent adapter
implements the required token-sharded KV layout, query partition and output
gather using the existing training IPC collectives. `pcp2-ipc-r3` now passes
two complete updates with identical tokens, norms and all 399 exported
parameters against CP1. See [the PCP results](h100-pcp-validation.md), including
the remaining performance cost and validation scope. ROCm was not rerun.
