# H100 canonical TP/CP validation

The implementation reuses the existing TP4/CP2 IPC collectives and strict
attention runtime. vLLM PCP adds token-page layout conversion and query/output
partitioning. It does not use a separate communication kernel or merge partial
attention results with floating-point reductions.

## Completed full-model matrix

Eight H100 GPUs, Qwen3-8B, two real optimizer updates per run, temperature 0.7,
top-p 0.95, top-k disabled, response cap 128 and KL coefficient 0.01.
Training CP is 8 / TP. Each successful row has zero active train/rollout
logprob byte mismatches. Generated tokens, both gradient norms, and SHA256
fingerprints of all 399 exported parameter tensors at the initial state and
after both updates match the numerical reference `alltp8-t4-r4c1-v3` exactly.
Each run compares 2,048 active logprobs (28,672 across the 14 runs).

| Training TP/CP | Rollout TP/CP | Source snapshot | Logprob byte mismatches | Tokens/norms/weights | Mean full step | vs reference |
|---|---|---|---:|---|---:|---:|
| 1/8 | 1/8 | v6 | 0 | equal | 124.428 s | +195.0% |
| 1/8 | 4/1 | v3 | 0 | equal | 44.843 s | +6.3% |
| 2/4 | 4/1 | v4 | 0 | equal | 43.316 s | +2.7% |
| 4/2 | 1/1 | v6 | 0 | equal | 48.626 s | +15.3% |
| 4/2 | 1/2 | v6 | 0 | equal | 54.885 s | +30.1% |
| 4/2 | 1/4 | v6 | 0 | equal | 68.065 s | +61.4% |
| 4/2 | 1/8 | v4 | 0 | equal | 108.551 s | +157.4% |
| 4/2 | 2/1 | v6 | 0 | equal | 44.935 s | +6.6% |
| 4/2 | 2/2 | v6 | 0 | equal | 53.150 s | +26.0% |
| 4/2 | 2/4 | v4 | 0 | equal | 64.669 s | +53.3% |
| 4/2 | 4/1 | v3 | 0 | equal | 42.173 s | +0.0% |
| 4/2 | 4/2 | v3 | 0 | equal | 50.032 s | +18.6% |
| 4/2 | 8/1 | v6 | 0 | equal | 42.142 s | -0.1% |
| 8/1 | 4/1 | v4 | 0 | equal | 42.310 s | +0.3% |

Completed configurations: 14 (14 completed runs). This is a basis matrix plus selected
combinations, not all 40 Cartesian training/rollout pairs.

The two reference gradient norms are `0.05702933925260603` and
`0.0711779018075793`. Log metric aggregation (such as mean loss/entropy) can
still have small floating-point differences; the equality claims above name
the actual tensors and norms compared.

## Fixes exercised

- Fixed virtual TP8 and vocabulary size 152576 across physical topologies.
- Fixed default logical microbatch token budget 1024; explicit CLI override remains available.
- Loss backward normalization occurs before low-precision model derivatives.
- One CP replica contributes each complete canonical parameter gradient to the framework average;
  shared Q/K norm gradients additionally have one TP contributor. This avoids CP8 ring rounding.
- PCP retains real token-sharded KV, disjoint query partitions and rank-ordered output gather.
- TP1 rollout retains its singleton TP process group. Passing None previously made the
  strict logprob kernel use WORLD and incorrectly count CP ranks as vocabulary shards.
- vLLM 0.16.0 accidentally wrote `with pool and config` and skipped its weight pool.
  A version-pattern-specific wrapper enters the weight pool correctly. Existing IPC
  arenas are created before entering that pool, so vLLM sleep can release model weights
  while IPC stays resident. This reduced TP1/CP8 rollout sleep residency from about
  19.8 GiB to 4.5 GiB and allowed the colocated TP1 trainer to initialize.

- The VIME companion defers rollout weight wake-up until the offloaded trainer
  finishes disk export and sleeps. This removes the second TP1 memory peak
  after optimizer-state allocation, without changing the numerical update.

## Diagnostic gradient evidence

The TP1/CP8 versus TP4/CP2 replay comparison matched exact integer square-bin
statistics for 291 parameter tensors, and matched every FP32 element in 145
small gradient tensors. Square-bin identity is not a full byte hash of every
large gradient tensor. Temporary replay instrumentation was removed before
the final matrix. See the [gradient diagnostic evidence](evidence/pr432-20260920/h100-cp8-gradient.json).

## Reproduction and limits

```bash
./rlk verify --tp 1 --rollout-tp 1 --rollout-cp 8 --temperature 0.7 --top-p 0.95 --max-response-len 128
```

The default mode disables rollout-logprob reuse. Top-k can be supplied with
`--top-k`; CP is inferred from training TP. Paths and runtime configuration
remain in the one-time machine profile.

All responses in these short runs were truncated. Policy-gradient advantages
were zero; nonzero optimizer updates came from KL. These results do not certify
arbitrary inputs, nonzero reward training, long trajectories, different explicit
microbatch budgets, ROCm, PP/EP or multiple nodes.

Full-step timings include weight export, SHA256 auditing and reload. They are
two-step observations, not steady-state native comparisons. Full decode graphs
currently materialize KV to the model context bound, leaving a performance
cost for short live sequences.

The v3 development runs used the frozen source fingerprint recorded in
[source v3](evidence/pr432-20260920/h100-source-v3.json); v4 adds the singleton TP-group fix and its CPU regression.
It leaves TP>1 execution unchanged. v5 adds the vLLM weight-pool compatibility
fix, recorded in [source v5](evidence/pr432-20260920/h100-source-v5.json); operator arithmetic remains unchanged.
v6 also bundles VIME commit `4a25438f5172aa75a076a51574a0564937757aa1` with
deferred rollout wake during disk export; its source snapshot is [source v6](evidence/pr432-20260920/h100-source-v6.json).
The v3/v4 timings predate these storage-lifecycle fixes, so cross-version
rows are descriptive observations, not controlled performance comparisons.
These RL-Kernel development versions are based on d8b17d19 and
were run from a recorded dirty worktree. Failed development jobs remain in
the run archive and are not counted as passing results.

See [the machine-readable matrix](evidence/pr432-20260920/h100-matrix.json)
and [the earlier PCP transport measurements](h100-pcp-validation.md).

## Additional completed checks and publication scope

- CPU regression suite: **143 passed, 1 skipped**, covering canonical CP
  gradients, PCP layout and execution coverage, launchers, vocabulary padding,
  singleton TP groups and the weight-pool compatibility wrapper. The skip is
  a ROCm-only path on the H100 environment.
- Distributed PCP attention: **18 cases** across CP2/4/8, KV interleave 1/8/16,
  and prefill/decode query shapes, in eager execution and CUDA Graph replay.
- Two-GPU weight-pool/IPC lifecycle probe: weight values and resident IPC
  remain valid through sleep/wake; both workers clean up and exit successfully.

The requested publication contains only the completed 14-configuration basis
matrix. Supplemental topology, alternate-sampling and native-timing work was
stopped at the user's request and is not included in this report. No new
steady-state native comparison is claimed.

Runtime and companion sources are unchanged from the v6 fingerprint at
publication. The final weight-pool probe has explicit allocator/IPC cleanup;
that test-only cleanup and documentation updates postdate the v6 snapshot.
The README's separate 200-step figures are original images from the source
document, not plots from these two-step topology checks.
