# P6 / DSV4 T01 start kit

This PR supplies the development contract, reference inputs and verification tools
for T02-T09. It does not close the fused-kernel, hardware, production or live
integration tasks. The package follows the repository layout used by the P2 T01
start kit in PR #447: `rl_engine/p6`, `tests/p6`, documentation and scoped CPU CI.
It has no import or development dependency on P2.

## Correction to the first revision

The first revision placed a standalone reference program in
`examples/dsv4_p6_startkit`. It lacked a repository package, independent operator
entry points, a sealed-artifact provider loader, an explicit replacement interface,
a required-evidence matrix and CI. This revision replaces that directory with the
repository package. The nine frozen cases retain their original forward, backward
and gradient-dispatch numerical bytes; schema and provenance are deliberately
versioned again. Old artifacts are not silently converted.

## Contract and scope

| Item | Version / scope |
| --- | --- |
| Development task contract | `p6-task-contract.proposed.v1` |
| Start-kit schema | `p6-start-kit.v1` |
| Arithmetic profile | `p6.synthetic-bf16.v1` |
| Trace envelope | `p6-trace-envelope.v1` |
| Evidence artifact | `p6-sealed-artifact.v1` |
| Foundation compatibility | `UNVERIFIED`, requires owner-approved binding |
| Production / performance certification | Not provided by T01 reference tests |

`tests/p6/data/contract.v1.json` freezes the proposed API, five operator signatures,
five boundary keys, saved-forward fields, policy, statuses and handoff obligations.
`catalog.v1.json` freezes nine case identities. The `proposed` label is intentional:
this PR cannot declare upstream owner review complete. Synthetic RoutePlan and
ExchangePlan references are opaque version/fingerprint pairs, not a second
implementation of Foundation's public ABI.

Run from a repository checkout with Python >=3.10, PyTorch, NumPy and pytest.
The numerical scalar oracle uses the standard library; importing `rl_engine`
still follows the repository's existing PyTorch import. No model checkpoint,
native-extension compilation or P3/P4/P5 live provider is needed.

```bash
export OMP_NUM_THREADS=1
python -m rl_engine.p6 manifest
python -m rl_engine.p6 catalog
python -m rl_engine.p6 recordings --operator fixed_order_combine_fwd
python -m rl_engine.p6 negative-fixtures
python -m pytest tests/p6 -q
python -m rl_engine.p6 conformance --device torch-cpu --hidden-size 4096 --output /tmp/p6-run-001
python -m rl_engine.p6 verify /tmp/p6-run-001
```

Fixture paths are checkout-relative, resolved from the package location, not the
current working directory. This is a repository development kit, not a standalone
wheel containing test fixtures.

## Independent developer entry points

| Task | What can be used before other tasks merge |
| --- | --- |
| T02 | `reference.canonical_unpermute_fwd`; recorded packed rows and inverse map |
| T03 | `reference.fixed_order_combine_fwd`; recorded canonical rows and valid mask |
| T04 | `reference.shared_residual_merge_fwd`; recorded FP32 accumulator and branches |
| T05 | `reference.fused_moe_combine_fwd`; complete input and every intermediate byte |
| T06 | `reference.fused_dx_fanin_bwd`; saved-forward artifact and independent gradients |
| T07 | `provider.Provider`, `RecordedProvider`, `ProviderRegistry`; explicit replacement seam |
| T08 | frozen cases, executable negative fixtures, EP layout/arrival mocks |
| T09 | typed envelopes, boundary hashes, evidence matrix, seal/verifier and scoped CI |

The `fused_*` names on `reference.py` express the production interface to implement;
these functions remain scalar CPU oracles. No Triton kernel or production registry
entry is created here. T02-T04 references are separately callable, with independent
recorded inputs. T05/T06 do not have to wait for their implementations.

`boundary_recordings()` returns 45 records: nine cases for each of five operators.
Every record binds its case ID, input, expected bytes, boundary, phase, plan/order
fingerprints and checksum. The `--hidden-size 4096` conformance option adds one
seeded H=4096 case, producing 50 operator records in that attempt.

## Numerical and gradient boundaries

- P4 returns independent, already-weighted per-slot rows. P5 owns route weighting;
  P6 does not apply it again. EP pre-summing is not part of the profile.
- The canonical key is `(global_token_id, topk_slot)`. Token IDs are opaque IDs,
  not local tensor offsets. Missing/duplicate valid slots and bad padding fail.
- Active rows are BF16-exact, widened to FP32. Six slots are added in ascending
  slot order, rounding each addition to FP32. Invalid slots are skipped. The first
  valid row initializes the accumulator; an all-invalid accumulator is positive
  zero. These signed-zero/invalid rules are proposed reference-profile semantics.
- The synthetic forward profile adds shared, then its explicitly supplied
  residual fixture, followed by one BF16 round-to-nearest-even. This does not
  authorize adding the real model's mHC four-stream residual twice. T01/P1 must
  bind the real residual tensor identity before live integration.
- The backward profile consumes and emits FP32, sums returned expert dX in slot
  order, then adds shared dX at the same expert-input boundary. A raw residual-fork
  gradient belongs to P1. Optional Router input-gradient integration belongs to
  the agreed P1/P3/autograd boundary. No `dW/dA/dB/dp_s` is computed by P6.
- `SavedForward` captures a serialized plan and fingerprint. Backward restores it
  against caller-provided run/microbatch/forward/checkpoint identity and expected
  fingerprint. Stale, wrong-run or corrupt metadata is rejected.
- Non-finite active inputs/results, overflow and subnormal arithmetic are outside
  this reference profile. The latter is an explicit capability restriction, not
  a claim of full BF16/FP32 production-domain support.

The golden cases include tail widths, non-monotonic large token IDs, invalid and
padding rows, zero tokens/routes, cancellation, BF16 ties and signed zero. The
cancellation case detects rank-local re-association; hard-coded tie/zero checks
anchor the frozen oracle output independently of generated fixtures.

## P4 mocks and provider seam

`mocks.ep_return` simulates EP=1/2/4/8, two placement rotations where applicable,
uneven/zero-count peers and reversed arrival. It writes to the declared logical
receive index. This exercises P6 invariance; it does not execute NCCL, establish
P4 correctness or certify multi-GPU WS2. `gradient_dispatch` is a recorded mock
that gathers the same dy for each valid saved route without materializing a
production `[T,6,H]` broadcast. P6 currently needs no new collective kernel.

```python
from rl_engine.p6.artifact import read_artifact
from rl_engine.p6.provider import RecordedProvider, ProviderRegistry

payload, _ = read_artifact('/tmp/p6-run-001')
record = payload['operator_recordings'][0]
registry = ProviderRegistry()
registry.register('recorded', RecordedProvider.from_artifact('/tmp/p6-run-001'))
envelope = registry.run('recorded', record['operator'], record['case_id'], record['inputs'])
```

A candidate provider implements `describe()` and
`run(operator, case_id, inputs)`. Registration checks the full versioned contract
and profile. A requested `live` route needs an explicitly registered provider and
actual provenance; it never falls back to recorded results. The only current
profile is synthetic reference conformance, so even a successful candidate is
not production certification. Neither registry nor recordings mutate the global
`KernelRegistry` or import another task's live implementation.

Comparison gates schema/provenance and operator-event validity, then input/run/
plan/order identity, then boundary bytes. A hash is an integrity/index field,
not a substitute for the raw-byte comparison. P7 can inspect the proposed typed
envelope; a live Foundation/P7 adapter still requires compatibility review.

## Artifact, safe reuse and evidence

An attempt contains `payload.json`, `manifest.json` and `seal.sha256`. Publication
refuses to overwrite an existing directory. The payload includes the contract,
catalog, source cases, independent recordings, 15 negative outcomes, 20 required
evidence rows, source fingerprint and runtime provenance. Only the five synthetic
reference rows can pass; live WS1, live WS2 and Integration remain NOT_CERTIFIED.

The offline verifier checks integrity, contract compatibility, complete operator
coverage, case bindings and CPU replay. It does not rerun GPU execution or prove
that the claimed hardware provenance is authentic. Checksums are not signatures.

`conformance --resume` reuses a **complete** verified attempt only when request,
source code, contract and exact input recordings match. It rejects partial attempts
and identity changes. This deliberately bounded reuse is not a distributed resume
implementation for P7's ArtifactStore.

## H100 reference validation (pending)

```bash
P6_RUN_GPU=1 python -m pytest tests/p6 -q
python -m rl_engine.p6 conformance --device cuda:0 --require-h100 --graph \
  --hidden-size 4096 --output /tmp/p6-h100-001
python -m rl_engine.p6 verify /tmp/p6-h100-001
```

Default pytest skips two opt-in GPU slices. With `P6_RUN_GPU=1`, missing CUDA fails
instead of skipping. The graph slice captures and replays real CUDA graphs with
changed fixed-address inputs; it checks the PyTorch reference, not T05/T06 kernels.
The CLI's H100 gate rejects CPU and non-H100 devices. NVIDIA and AMD evidence must
be maintained separately; this PR does not claim either hardware certification.

CPU CI covers Python 3.10/3.12, Ruff lint/format, pytest, H=4096 PyTorch reference
conformance and independent artifact verification. The validation JSON records
only commands actually executed for this revision.

Before T01 approval: freeze Foundation binding, P1 residual/P3 gradient wiring,
backward payload/cast rules and invalid/signed-zero policy with their owners.
Before production closure: add and validate T05/T06 implementations and the
requested hardware evidence. Before Integration: bind real P1/P3/P4/P5 providers
without changing canonical identity or silently converting schema.
