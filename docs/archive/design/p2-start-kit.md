# P2 / DSV4 T01 start kit

This is the **T01 development start kit**, not closure of T02–T07, WS1/WS2,
performance, or the final P1/Miles integration. It is additive to the existing
RL-Kernel kernel registry and changes no existing production dispatch.

The requested PR base is `dsv4-p6-dev`; the supplied task contract describes
**P2 compressed attention**, so the package is named `rl_engine.p2`. “T0” in the
request maps to the document's T01 start-kit deliverable.

## Frozen identities

| Field | Version |
| --- | --- |
| Source contract | `p2-task-contract.v1` |
| Foundation boundary ABI | `foundation-attention-boundary.v1` |
| Start-kit schema | `p2-start-kit.v1` |
| State schema | `p2-state.v1` |
| Errors | `p2-error.v1` |
| CPU synthetic/reference profile | `p2.synthetic-fp32.v1` |
| Artifact | `p2-sealed-artifact.v1` |
| MXFP4 reference layout | `p2.mxfp4-reference.v1` |

Source document SHA-256:
`85c3d679a961dadfaca4a48fe7af60d9a420775f5f1fddbde0ccf1db221c09c9`.
The source was used as the semantic specification, not as authority to operate
unrelated accounts, alter other checkouts, merge branches, or launch its later
work packages.

`tests/p2/data/contract.v1.json` is the reviewed golden manifest: all 15 operator
signatures, 13 M2 boundary keys, saved backward boundaries, constants, comparison
policy, primitive reuse paths, modes, state layout, and failure codes.
`tests/p2/data/catalog.v1.json` freezes the case IDs and recipe identities for all
17 required fixture families. A catalog entry is an obligation, **not evidence
that its production gate passed**. In particular the backend and performance
families reserve future live certification.

## Quick start

Run from the repository root with Python >=3.10, PyTorch >=2.4.1, NumPy, and pytest.
No checkpoint, model download, credentials, or native extension build is required.

```bash
export OMP_NUM_THREADS=1
python -m rl_engine.p2 manifest
python -m rl_engine.p2 catalog
python -m rl_engine.p2 recordings --operator kv_compressor_c4 > c4-recording.json
python -m rl_engine.p2 negative-fixtures > negative-fixtures.json
python -m pytest tests/p2 -q
python -m rl_engine.p2 conformance --tokens 513 --output /tmp/p2-run-001
python -m rl_engine.p2 verify /tmp/p2-run-001
```

The artifact output directory must not already exist; runs are never silently
overwritten. Exit 0 means the requested scoped operation succeeded; exit 1 emits
a machine-readable failure code on stderr. `verify` does not require CUDA,
upstream live providers, the generation process, or the producer's machine.

Longer multi-page fixture:

```bash
python -m rl_engine.p2 conformance --tokens 1025 --output /tmp/p2-long-001
python -m rl_engine.p2 verify /tmp/p2-long-001
```

## What an owner can consume independently

`boundary_recordings()` contains inputs, expected output tensor bytes, saved
intermediates, and checksums for each of the 15 operators. The dimensions are the
contract's 64 heads, Main D=512, Index D=128, normalized BF16 hidden size 4096,
and 8 output groups with rank 1024. The raw Q projection fixture explicitly uses
an 8-dimensional synthetic input rank (the contract does not pin `Q_r`'s rank).

Large synthetic projection matrices are lossless selector recipes, not omitted
weights: for row `r`, column `(r+salt)%in_features` is 0.5 and every other entry is
0. `materialize_weight(recipe)` expands the dense tensor when a kernel owner
needs it. The analytic selector reference avoids materializing ~67M zero entries
just to export a small output-projection recording. Production GEMM must still
use the existing deterministic fixed-K primitive.

Each tensor recording includes dtype, shape, little-endian raw bytes (base64),
and SHA-256. The integer-counter fixture recipe is independent of RNG state:
`((arange(numel)+salt)%31)/16`. No private weights or real user data are used.
Generated floating-point output hashes belong to the recorded PyTorch/runtime
provenance; they are **not** a promise that every libm/PyTorch version will
regenerate transcendental outputs byte-identically.

The numerical reference exposes:

- Scale with `((u * 64^-0.5) * 128^-0.5)` and reverse backward association.
- Caller-provided FP32 GPT-J interleaved RoPE tables, exact NoPE preservation,
  out-of-place inverse, and absolute completed-group positions.
- C4 `S+APE` **before** previous-first/current-second overlap; the initial
  previous K is zero and its logits are `-inf`.
- C128 complete 128-token channelwise pooling; no incomplete commit.
- Existing `NativeRMSNormOp.forward_fp32`, then partial RoPE; Index additionally
  uses normalized H128 and MXFP4 packing. No second public RMSNorm is introduced.
- Block-local E2M1/UE8M0 packing, ties-to-even, even element in the low nibble.
  The exponent is exactly the source contract's
  `clip(ceil(log2(max(amax,6*2^-126)/6)),-127,127)`.
  Zero blocks use scale byte 1; negative zero keeps its sign nibble.
  Nonfinite input and FP32 dequantization overflow fail closed.
- Fixed adjacent-pair FP32 trees (right-zero-padded to a power of two), four
  inner-32 ICV trees followed by their outer tree, ReLU-before-weight/head-sum,
  and Top-512 ordered by descending score / ascending global ID with `-1` padding.
- One-denominator MQA+sink attention and explicit FP32 dQ/dKV/dsink.
  Pooling/RoPE/Hadamard backward is checked against functional autograd.

The supplied contract does **not** pin a production Main FP8 format, every
Megatron round point, or a live Foundation Python binding. Consequently this
profile records Main completed rows in FP32 and explicitly rejects production
certification. It does not guess an FP8 layout or silently select a different
backend. The existing GPU RoPE kernel remains the owner extension point for T02;
the CPU oracle here does not register a competing production RoPE implementation.

## State and planner mock

`PlannerMock` is an opaque byte-transport mock, not another production paged
store or a compressor implementation. Callers supply projected FP32 partial
bytes and completed Main/Index row bytes. The named lifecycle fixtures use small
synthetic payloads, deliberately **not** claimed to be full model cache pages.
Actual-dimensional compressor records are separate.

- Exactly one recent slot per contiguous global token, with window
  `[max(0,t-127),t]`, logical slot, page identity, and wrap generation.
- C4 Main **and independent Index** commits only at `(t+1)%4==0`.
- C128 Main commits only at `(t+1)%128==0`; C0 never has a compressor row.
- Independent Main/Index weight/APE/norm/state/page/quant identity namespaces.
- Partial groups, prior C4 overlap group, completed rows, page size, identity,
  generation, and checksum are included in a deep-copied snapshot.
- Invalid input is checked before mutation. Restore validates the entire
  snapshot. Chunk changes and eager-to-graph handoff preserve per-token bytes.
- Four independent reference-mode replays produce four verdicts, not one
  ambiguous “attention passed” field.

CP=1/2/4 and TP=1/2/4/8 mocks expose absolute token ranges, global head ranges,
unique completion-token ownership, and global Index-K visibility. The P4 mock
gathers **per-head contributions in global order before the fixed tree**, rather
than reducing runtime-dependent local partial sums.

## Provider and comparison policy

`RecordedProvider` implements `describe()` and `run(case_id, mode)`.
`RecordedProvider.from_artifact("/tmp/p2-run-001")` validates the seal and exposes
all 12 layer/mode combinations; for example
`run("P2-F-LAYER-c4.v1", "graph_decode")`. Its boundary index identifies the
independent operator recordings, not a claimed full-chain attention execution.
`ProviderRegistry` permits explicit `recorded` and `live` registrations with
matching contract/profile. Missing capabilities raise `UNSUPPORTED_CAPABILITY`;
exceptions are not caught to fall back to recordings.

`TraceEnvelope` carries schema/ABI, immutable case and checkpoint/layer identity,
mode, profile, runtime policy, route, provenance, per-token states and M2
boundaries. A live adapter must return the requested case/mode, natural route,
and actual provenance. The synthetic profile is the only registered profile;
real kernel readback and production layouts require a separately reviewed
implementation profile, not a boolean claiming support.

Comparison order:

1. Schema, case, checkpoint/layer/profile identity and runtime policy.
2. Every recent/Main/Index per-token snapshot, page and generation.
3. Natural-route requirement, when requested.
4. M2 output bytes.

No output/performance attribution is issued after a state failure. Discrete
metadata never enters a tolerance comparator. Split-K, Stream-K, Split-KV,
dynamic partition, atomics, moved round points, configured-as-actual provenance,
and silent fallback have distinct fail-closed statuses. There is no tolerance
switch that can turn such violations into PASS.

## Seals, evidence and limitations

An artifact consists of exactly:

```text
manifest.json     version, payload size/hash, resume identity, completion
payload.json      contract, catalog, recordings, states, negative cases, evidence matrix
seal.sha256       SHA-256 of the canonical manifest
```

Publication uses a sibling temporary directory, file flush/fsync and rename.
The reader rejects incomplete inventories, symlinks, path substitutions,
duplicate JSON keys, invalid tensor sizes/checksums, missing modes/operators,
identity drift, corrupt state, and resealed semantically invalid evidence.
The checksum seal is **integrity protection, not a cryptographic signature or
proof that an untrusted producer really ran a GPU**.

The 60-row required-evidence matrix (15 operators × 4 modes) remains explicit
about unsupported live WS1/WS2/integration capabilities. Native tile, warp,
stage, actual unroll, fast-math, FMA, vectorization and spill fields are `null`
when unavailable, never copied from configured values. Performance has its own
`NOT_CERTIFIED` verdict.

Executable negative fixtures include corrupt state, wrong page/generation,
early/duplicate commit, Main/Index alias, missing global visibility/owner,
silent fallback, forbidden reductions, round-point drift, unsupported backend,
two softmax denominators and sink/candidate errors.

## GPU / distributed infrastructure gates

These opt-in gates use real hardware but do not claim that unimplemented
compressed-attention kernels passed:

```bash
P2_RUN_GPU=1 python -m pytest tests/p2 -q
for ranks in 1 2 4 8; do
  PYTHONPATH=. torchrun --standalone --nproc-per-node=$ranks \
    scripts/p2_distributed_smoke.py
done
```

The GPU tests run on **each visible GPU**, check CPU/CUDA reference bytes, then
check 257 token boundaries for functional training, chunked prefill, eager
writes, and **real CUDA Graph** opaque-cache transport. Graph capture uses an
explicit stream on the corresponding device; all input/slot/cache addresses are
asserted stable during replay. The distributed script executes real NCCL
all-gather with 1/2/4/8 ranks and applies the global fixed tree. CP topology
validation there is still a mock; it is not a full CP attention implementation.

On the validation machine, default NCCL NVLS initialization failed with CUDA 401
while binding multicast memory. No system configuration was changed. The
separate, explicitly configured transport run used:

```bash
NCCL_NVLS_ENABLE=0 PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
  scripts/p2_distributed_smoke.py
```

The script reports this environment setting as **configured**, not actual NCCL
algorithm readback. The default-path failure and the configured-path results
must remain separate in the test report.

CPU CI runs on Python 3.10/3.12, exports a 513-token sealed artifact and verifies
it independently. CPU CI's opt-in GPU skips are not GPU PASS results.

## Contract delta and handoff

```text
Interface/Schema: P2 start kit / state / recorded provider
Current version: absent on target branch
Proposed version: p2-start-kit.v1 (source contract p2-task-contract.v1)
Change: additive, no existing dispatch/core changes
Migration: none; owners consume the frozen manifest and commands
Affected tasks: T02–T07, future P1/P3/P7 integration
Fixture/checksum change: new versioned catalog, recordings and negative fixtures
Blocking level: production FP8/round-point/live profiles require T01 review
```

Owner handoff must retain task/owner/reviewer, implementation and dependency
commit, backend/device/build readback, supported shapes/dtypes/modes, actual
kernel arithmetic, fixture checksums, forward/backward/negative/state verdicts,
debug command, artifact seal, unsupported capabilities, and downstream consumers.
Reviewer approval and final integration remain separate from this start-kit PR.
