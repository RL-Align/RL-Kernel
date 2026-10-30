# P3 T01 Router Start Kit

The P3 Router development package provides operator references, recorded inputs
and outputs, a CUDA Top-K implementation, and local consumer interfaces.
Implementation lives in `rl_engine/p3`, tests and frozen data in `tests/p3`, and
consumer examples in `examples/dsv4_p3_startkit`.

The development contract is `p3-router-task-contract.v23`, with operator ABI
`p3-op-abi.v4`, recording/provenance v2 and sealed artifact v2. Old archive versions
are rejected rather than silently converted. Numerical constants, tie order,
rounding points and the fixed six-way reduction tree retain their prior definitions.
Local validation is recorded in
[`p3-start-kit-2026-10-08.json`](../validation/p3-start-kit-2026-10-08.json).

The package exposes a **T01-S0 development profile**. Its status remains
**anchor_pending**. An approved Foundation binding, real Miles anchor and
recorded Megatron/Miles L3b evidence are still required for formal certification.
Local reference or Top-K results do not certify other owners' production kernels,
WS2, P3-R0, or live Integration.

## Layout and ownership

| Module | Responsibility |
| --- | --- |
| `contract.py` | Versioned operator ABI, Core/Envelope/Saved/Seed schemas, verdicts and event manifest |
| `oracle.py` | Bit-defined FP32 score, Hash/Learned routing and backward oracle |
| `reference.py` | Seven independent CPU operator entry points for T02-T06 |
| `torch_reference.py` | Independent original Torch semantics for paired diagnostics |
| `native/bitmath.h`, `bitmath.py` | Shared host/device arithmetic, rounding and fixed six-way tree |
| `native/stable_topk6.cuh`, `stable_topk6.py` | T01's real synchronous SM90/H100 Top-K and device ABI |
| `provider.py` | Input-bound recorded provider, durable invocation journal and sealed saved validation |
| `recordings.py`, `fixtures.py` | Versioned source cases and independent operator recordings |
| `assembler.py` | The only `assembler_abi.v1` stub; T05 replaces it in place |
| `checker.py`, `negative.py` | Identity/schema/provenance gates, comparisons and executable faults |
| `mocks.py` | Local ownership/materialization hooks for T07/T08 |
| `boundary.py` | Versioned local P4/P6/P7 consumer checks |
| `provenance.py`, `diagnostics.py` | Actual metadata, ULP diagnostics and selection margins |
| `artifact.py`, `__main__.py` | Seals, replay, completed-attempt reuse and CLI |

`router_contract.py`, `router_oracle.py` and `router_torch_reference.py` retain the
contract's module names and re-export the same objects from this package. The
canonical CLI is `python -m rl_engine.p3`. Earlier `rl_engine.moe.router_*` imports
must migrate to `rl_engine.p3.router_*`.

P3 consumes FP32 gate logits. It does not implement gate GEMM, collective
transport, expert computation or combine. P1 owns gate GEMM/backward; P5 applies
route weights; P6 performs fixed-slot combine. Other P3 owners can develop using
this package's recordings and stubs without live P1/P4/P5/P6 implementations.

## Contract and execution

The fixed model has H=4096, E=256, K=6, epsilon=1e-20 and scale=1.5. Hash layers
0-2 preserve the original table slots. Learned layers select from an independent
`q=s+b` buffer and gather weights from pre-bias `s`. Top-K orders by descending
score and ascending logical expert ID, with signed zeros comparing equal.

The supported capacity policy is `dropless_v1`: all six active slots remain valid,
`capacity=-1`, and only padding creates invalid slots. Finite capacity, overflow
truncation and renormalization require a separate mathematical contract delta.
The local consumer API explicitly rejects those policies.

All public operators accept `P3OpCtxHost` and return `P3OpResult`. Non-PASS results
have no payload. `ReferenceProvider` computes CPU references; `RecordedProvider`
loads verified, input-bound recordings. Their backend is explicitly
`recorded-cpu`, which cannot certify a CUDA execution.

Backward accepts typed `SavedScoreSealedV1` or `SavedRouteSealedV1`. Validation
checks schema, identity, manifest source and checksums before zero-active handling.
Checksums cover all raw saved rows, including padding. Padding computation is
excluded from numerical comparisons; the assembler produces canonical padding
records. Zero-active calls allocate no invocation ID and produce no saved state.

The actual CUDA provider checks status and invocation echo through a D2H copy on
the same stream followed by synchronization. A persistent journal reserves every
attempt and counter before launch. The runner holds an execution lock for the
scope `(run_id, engine_id, rank)` and never resets its journal between attempts.

## Reproduce

Use Python >=3.10, NumPy, PyTorch, pytest and a C++ compiler. CUDA tests additionally
require H100, CUDA PyTorch and `nvcc`. The examples below use `python`; this local
workspace uses `.venv/bin/python`. A full RL-Kernel native extension build,
checkpoint, Megatron and Miles are not needed for the start-kit checks.

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=1
python -m rl_engine.p3 manifest
python -m rl_engine.p3 catalog
python -m rl_engine.p3 recordings --operator learned_route_bwd
python -m rl_engine.p3 negative-fixtures
python scripts/generate_p3_fixtures.py --check
python -m pytest tests/p3 -q

python -m rl_engine.p3 check_p3 --output /tmp/p3-cpu-001
python -m rl_engine.p3 verify /tmp/p3-cpu-001
python -m rl_engine.p3 check_p3 --output /tmp/p3-cpu-001 --resume
# Equivalent checker entry point:
python scripts/check_p3.py --output /tmp/p3-cpu-001 --resume
```

Always use a fresh output directory for new execution. `--resume` reuses only a
completed, verified artifact with matching request, inputs and source fingerprint.
Old archive versions are rejected. Seals verify integrity; they are not signatures.
The verifier derives the permitted evidence matrix and rejects unsupported PASS
claims even when the file checksum has been recomputed.

`check_p3` reports `CASE_PASS` only within `T01_START_KIT`. `--require-ws1` fails
with `MISSING_PROVENANCE` while the Miles anchor is pending. Completing the anchor
alone will not supply the other owners' required CUDA evidence.

### H100

Select one available H100 before running:

```bash
export CUDA_VISIBLE_DEVICES=0
P3_RUN_GPU=1 python -m pytest tests/p3 -q -ra
python -m rl_engine.p3 check_p3 --backend cuda \
  --output /tmp/p3-h100-001 --state-dir /tmp/p3-persistent-state
python -m rl_engine.p3 verify /tmp/p3-h100-001
python -m rl_engine.p3 check_p3 --backend cuda \
  --output /tmp/p3-h100-001 --resume
```

`P3_RUN_GPU=1` makes missing hardware a failure. The CUDA artifact includes 20
active cases across four block sizes: 80 actual Top-K records. Score, weights and
backward in that artifact remain recorded CPU outputs, not CUDA certification.

Keep the same persistent state directory for a given scope. Do not delete the
journal or change directories to reset IDs. The default is `P3_STATE_DIR`, or
`~/.local/state/rl-kernel/p3` when unset. State must remain outside the sealed
artifact directory. Resume verifies existing evidence without relaunching CUDA.

## Stable Top-K and arithmetic

T04 includes `native/stable_topk6.cuh` and calls
`p3::stable_topk6_row(q_row, ids_row, status_record)`. This implements
`stable_topk6_device_abi.v1`; the enclosing kernel owns the invocation echo.
Input rows contain all 256 logical experts, and the six selected slots are never
reordered after selection. Nonfinite active selection scores fail closed.

The shared header fixes exp/log1p/sigmoid/softplus, BF16 rounding, IEEE sqrt and the
six-way sum. CUDA builds disable FTZ and FMA contraction and enable precise
division/sqrt. The host checks round-to-nearest and gradual underflow. Vendor
exp/log or approximate reciprocal sqrt must not replace the strict implementation.

The two round policies, `fp32_direct` and `bf16_round_then_widen`, remain separate.
The formal Miles policy is pending. Torch paired traces are diagnostics; score and
weight ULP distances and rank 6/7 and 8/9 margins never relax the strict byte gate.

## Frozen data and consumer handoff

The catalog has 24 source cases: 20 active and four zero-active/empty cases. Active
cases each record score forward/backward, Top-K and branch-specific route
forward/backward, totaling 100 recordings across seven interfaces. There are 30
executable negative fixtures.

Dedicated cases cover a real 6/7 cutoff, 8/9 bias precision, pre-bias weights, and
hotspot/padding behavior under dropless routing. The original logit near-tie case
is retained as a rounding-collapse regression, not used as cutoff evidence.

`tests/p3/data` contains `contract.v23.json`, `catalog.v2.json`,
`negative.v2.json`, `boundary.v1.json`, `l3b.pending.v1.json` and compressed golden
JSON. Gzip stores the original JSON bytes without changing payloads; the fixture
checker compares decompressed contents. The legacy `golden.v1.json.gz` remains an
independent byte baseline for the original 16 cases. Both golden archives stay
below the repository's added-file size limit.

The two JSON files in `examples/dsv4_p3_startkit` are process-independent consumer
examples. A synthetic engine pair is not Megatron/Miles L3b evidence.

```python
import json
from rl_engine.p3.boundary import consume
from rl_engine.p3.serialization import decode

with open("examples/dsv4_p3_startkit/handoff.v1.json") as stream:
    bundle = decode(json.load(stream))
identity = bundle["RoutePlan"]["identity"]
route = consume(bundle, "P4", expected_identity=identity)
seed = consume(bundle, "P6", expected_identity=identity)
```

Consumer checks enforce local versions and identity. `require_foundation=True`
rejects a missing approved Foundation binding. This local mock does not claim to
implement an approved shared Foundation ABI.

T02 consumes score fixtures; T03 consumes Hash fixtures; T04 consumes Top-K and
pre-bias fixtures; T05 replaces the assembler; T06 consumes sealed intermediates;
T07/T08 extend distributed hooks; T09 extends validation; T10 owns named hardware
CI and certification. T09/T05 review T01. P3-R0 and roadmap live integration remain
separate stages, so downstream live code does not block independent development.
