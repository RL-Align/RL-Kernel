# Consistency debugger

## Product contract

The debugger observes native training and rollout execution under the same
logical weights and frozen tokens, then locates the first **observed** differing
boundary. Diagnosis does not require an RL Kernel implementation of the model.
`rlk debug BUNDLE` analyzes evidence offline; framework-specific replay and
`--matrix auto` replacement verification are optional layers above diagnosis.
It never edits a production model, enables rollout-logprob reuse, or treats an
available replacement as a proven fix.

There are two entry paths:

* A saved run / frozen sample uses an installed runtime adapter and the existing
  machine profile. The first adapter is Qwen3 Dense + Vime/Megatron/vLLM.
* A portable capture bundle is analyzed locally without a machine profile or
  GPU. Any application can export explicitly mapped semantic boundaries using
  the Python capture API, even without an RL Kernel implementation.

Architecture metadata selects an adapter; it does not establish semantic
equivalence. Adapters specify exact boundary meanings, execution order, logical
token/feature ownership, input dependencies and observed contracts. Unknown
models must not silently fall back to Qwen3. Fused internals that are not visible
remain uncaptured. MoE and arbitrary production schedulers require adapters.

## User journey

1. Resolve the original configuration and show the replay scope. Prefer the
   source run's module group; a bare sample defaults explicitly to native M000.
   `--baseline` can select the original Mxxx configuration.
2. Validate actual weights/input identities and evidence coverage. Different
   physical TP/CP layouts are allowed; their logical tensors must correspond.
3. Replay the original configuration. An equal result means **not reproduced**,
   not fixed. Stop unnecessary experiments.
4. Compare boundaries in semantic execution order. Display layer, stage, token,
   input evidence, observed contract differences and follow-up checks.
5. Query replacement capabilities. Model/backend/dtype/topology and sampling
   eligibility are separate from empirical bitwise validation. A missing
   replacement never prevents diagnosis of an existing capture bundle.
6. For a reproduced mismatch, try a registered replacement in a fresh diagnostic
   job, and use module ablations if necessary. Keep weights, tokens, sampling and
   topology fixed across arms. Mxxx is a Qwen3 adapter-specific encoding, not the
   generic debugger's data model.
7. Report reproduction, localization, attribution evidence and replacement
   verification separately. `verified_on_replay` means the same captured replay passed;
   it is not a production deployment, a universal bitwise guarantee, or a
   performance measurement.

Manual `--matrix full` / `--matrix M111,M011` remains available. `--matrix
attribute` runs M000, M111 and the three mixed module ablations, then reports
which combination passed on this replay; it does not claim that an individual
module is necessary or sufficient in isolation. `--dry-run` materializes
potential jobs without launching them. `--report-only` recomputes reports while
preserving launch failures and cross-arm identity gates. `--resume DIR` checks
that the frozen tokens, mask and sampling values still match, then skips only
completed arms. Failed arms retry in fresh `-retryN` directories, preserving
previous snapshots and logs without mixing them into the retry. `rlk doctor`
performs host/runtime preflight before GPU submission.

## Architecture

* `debug/core.py`: immutable CPU snapshots, raw-bit comparison, logical shard
  reconstruction; no model or execution-framework imports.
* `debug/evidence.py`: portable capture protocol, explicit PyTorch boundary
  hooks, coverage/contract comparison and conservative causal classification.
* `debug/events.py`: versioned execution-event bundles, completion gates, tensor
  trees, explicit sample/token/feature coordinates and observed-order comparison.
* `debug/auto.py`: native eager observer, repeated invocations, actual parameter
  inventories and lightweight `ModuleBoundary` cross-framework mappings.
* `debug/adapters.py`: runtime adapter discovery and replacement eligibility.
  External adapters use installed `rlkernel.debug_adapters` entry points;
  manifests cannot request arbitrary Python imports.
* `debug/qwen3_runtime.py`, `qwen3_report.py`: Qwen3-specific semantics,
  observational framework hooks, weight tiling, route and endpoint evidence.
* `debug/replay.py`, `session.py`: frozen source loading and isolated replay
  orchestration. All artifacts stay outside tracked source directories.
* `debug/entry.py`: CLI dispatch and offline reporting.

## Evidence and conclusions

Bitwise equality compares bytes, including signed zero. NaN, infinity, empty
tensors, missing shards, missing required contracts, missing endpoints and
incomplete runs cannot pass. Identical observed inputs narrow the search;
unobserved cache/mask/kernel internals still prevent a definitive arithmetic
root-cause claim. Semantic order is supplied by the adapter, never inferred
from alphabetical module names or wall-clock timestamps.

The terminal and JSON report distinguish:

* `planned`: preflight and launch plans were generated without executing a job;
* `not_reproduced`: the original replay did not exhibit the mismatch;
* `localized`: a first observed divergence, with coverage limits;
* `verified_on_replay`: original mismatch reproduced and a replacement passed with the
  same inputs and weights;
* `unresolved`: mismatch persists after available trials;
* `checked_without_baseline`: arms ran without an original baseline result;
* `inconclusive`: insufficient evidence or execution failure.

Per-report `diagnosis.level` can also be `contract_difference`, and
`replacement.status` is `unsupported` when no live adapter applies.

Detailed reports retain raw tensor evidence and requested/observed routes.
Console output links to them and to per-job logs. Unknown evidence is never
rendered as a successful check. A report-only invocation uses the same verdict
rules as the live invocation.

## Scope and acceptance

The initial live adapter performs one-sample, pre-update **eager** replay.
Batch-, cache-scheduling-, graph- or update-only failures may disappear. For
those workloads, export evidence from the original production execution or add
a faithful replay adapter. Performance checks use the normal workload, not
instrumented tensor capture. This release does not automatically diagnose
individual floating-point instructions or adapt an unknown model's kernels.

Acceptance covers: a non-Qwen model diagnosed without an RL Kernel; missing or
wrong semantic evidence rejected; first divergence and upstream input cases;
adapter rejection before job submission; native/source baseline preservation;
replacement selection without changing production defaults; consistent live
and offline conclusions; unchanged sampling/topology flexibility and no
rollout-logprob reuse. Framework integration tests and device smoke results
must be reported separately from CPU synthetic tests.

## Native runtime extensions

The portable API now also exposes native Megatron provider/checkpoint loading,
metadata-derived TP/CP tensor ownership, logical paged KV capture, deferred
CUDA/HIP graph probes, MoE route capture, a fused-kernel debug-output protocol,
and backward/optimizer observation. See
[Native runtime capture extensions](native-runtime-capture.md) for executable
integration patterns, required native contracts and validation limits.

These opt-in APIs do not make the `rlk discover` TP1 eager launcher universally
distributed, infer arbitrary unfamiliar architectures, or expose opaque fused
internals without kernel instrumentation. Historical acceptance results below
retain their original scope.

## Portable capture API

An existing live adapter needs only the CLI command. A new model/application
integrates this API once; ordinary users consume its saved bundles with
`./rlk debug /path/to/bundle`, without a profile or installed GPU framework.
Names below are adapter-declared semantics, not automatic matches by class name.

For a model with no RL-Kernel adapter, a PyTorch application can start with the
generic automatic capture layer. It records every selected module invocation,
all tensor/scalar leaves in positional inputs, keyword inputs and outputs, and
the model input/output endpoints. Tensors retain their shapes and dtypes; no
last-dimension flattening is assumed to represent token coordinates:

```python
from rl_engine.alignment.debug import AutoCapture, fingerprint, prepare_auto

positions = list(range(sequence_length))
prepare_auto(
    bundle_dir, model,
    positions=positions,
    scope="production eager replay; automatic leaf boundaries",
)
identity = {
    "weights": {
        name: fingerprint(param) for name, param in model.state_dict().items()
    },
    "tokens": fingerprint(tokens),
    "positions": positions,
    "mask": mask.tolist(),
    "sampling": sampling,
}
with AutoCapture(bundle_dir, "training", identity, model, positions=positions):
    training_forward()
```

Run the same capture on the rollout side with its actual runtime identity and
`side="rollout"`, then run `./rlk debug /path/to/bundle`. The result can locate
the first observed module boundary without an adapter or RL Kernel. This is a
fast generic starting point, not a semantic guarantee: fused internals remain
invisible. For two unfamiliar native runtimes, `rlk discover` can now infer a
decoder-block correspondence before producing a normal evidence bundle. The
observer independently fingerprints actual
`state_dict` tensors at entry and exit, in addition to checking the supplied
case identity. Changed weights, unfinished forwards, missing events or workers,
uncaptured declared boundaries and unsupported opaque state cannot pass.

There is no implicit 256-module limit. An explicit `include`, `exclude` or
`max_modules` restricts the claim to the selected scope; pass the same selection
to preparation and both captures. Root endpoints are enabled by default.
Selectors that discard fields narrow the evidence; input equality never proves
equality of unobserved cache, mask or kernel state.

### Automatic discovery for an unfamiliar model

When both sides can already execute a local, eager, eval-mode prefill, the
discovery path does not require an RL Kernel adapter, model name, layer-name
convention, per-layer `ModuleBoundary`, or parameter rename table:

```bash
./rlk discover /path/to/checkpoint \
  --tokens /path/to/frozen-token-batch.json \
  --output /path/to/new-discovery
./rlk debug /path/to/new-discovery
```

The live wrapper runs the original native HF model and the original native vLLM
model. It inspects actual module execution and actual parameter storage, then
validates a mapping using:

1. a uniquely repeated executed block stack on each side;
2. actual parameter fingerprints and explicit primitive storage contracts;
3. affine projection inputs/outputs, including declared fused projection slices;
4. observed block order and exact within-run state handoffs; and
5. frozen token, position, mask, eval-mode and no-cache evidence.

It does not match by activation error, class-name similarity, module name,
alphabetical order or “closest” shape. It does not transpose an arbitrary
matrix. A GPT-style transposed `Conv1D` and a vLLM fused QKV/merged projection
are accepted only through their framework-declared storage axes and fused
partition metadata, with the underlying parameter bytes still matching.

The output contains `mapping-plan.json`, immutable side probes and a normal
`diagnostic-report.json`. The mapping plan is regenerated from probes during
offline analysis; editing the plan cannot change the result. If there are
multiple independent stacks, unsupported quantization, ambiguous row layout,
opaque residual/cache state, changed buffers, missing workers or no unique
correspondence, the status is `not_comparable` with `needs_mapping` and an
actionable reason. Automatic discovery currently supports TP=CP=1 eager
unpadded prefill; use the explicit mapped capture API for distributed sharding,
decode cache, graph execution or scheduler-specific layouts.

The first reported boundary is the first **automatically validated observed
boundary**, not a claim about operations hidden inside a fused module. For a
native mismatch before the first discovered block, the correct result is the
first block input with input evidence `unknown`; add an explicit embedding or
root boundary if the pre-block interval must be split further.

`examples/debug/automatic_discovery.py` is a device acceptance harness. It
creates unfamiliar GPT-2 and Llama checkpoints locally, runs original HF/vLLM
prefill without RL Kernel replacements, generates a controlled fault and checks
that both models are automatically mapped. The same wrapper ran against the
complete official Qwen3-8B checkpoint: it validated all 36 decoder blocks and
localized the native HF/vLLM mismatch at the first discovered block input.

### Different frameworks: map observations, not kernels

`ModuleBoundary` maps a logical role to a native module path on each side. The
role must mean the same computation; matching class names is not evidence.
An input selector receives `(module, args, kwargs)` and an output selector
receives `(module, args, kwargs, output)`. Selectors may return nested tensor
trees or `LogicalTensor(value_2d, [(sample_id, token_position), ...])`.
Do not cast floating-point tensors or re-execute an operator to manufacture a
matching intermediate.

```python
from rl_engine.alignment.debug import ModuleBoundary, LogicalTensor

def logical_output(module, args, kwargs, output):
    return LogicalTensor(output.reshape(len(logical_rows), -1), logical_rows)

# Training: path="transformer.blocks.7"; rollout: path="model.layers.7".
# Supply selectors for BOTH inputs and outputs if native layouts differ.
bindings = {
    "layer.7.block": ModuleBoundary(
        path=native_path,
        inputs=logical_input_selector,
        outputs=logical_output,
        layer=7,
    ),
}
prepare_auto(root, model, positions=positions, scope="mapped block boundaries",
             bindings=bindings)
with AutoCapture(root, side, actual_identity, model, positions=positions,
                 bindings=bindings, weights=canonical_runtime_weights,
                 root_inputs=canonical_model_inputs,
                 root_outputs=canonical_model_outputs):
    model_forward()
```

`canonical_runtime_weights()` returns actual tensors under common logical
parameter names, using explicit fused-weight splits/transposes as necessary.
Never copy one side's hashes to the other. This inventory is a small observation
adapter, not an RL Kernel implementation. It may also yield `(name, tensor)`
pairs lazily: distributed adapters can reconstruct and fingerprint one parameter
at a time instead of retaining a second complete model. Duplicate names fail.

For a fused boundary with no directly observable equivalent input, use
`ModuleBoundary(..., observe_inputs=False)`. Its output is still compared, but
input evidence remains **unknown**. Do not invent a normalization input by
re-executing an add/norm or by recording an empty placeholder as equal.

`EventCapture` also accepts `LogicalTensor` directly. Declare expected ranks
with `ranks={"training": [0], "rollout": [0, 1]}` and instantiate each worker
with its `rank`. Logical shards specify `feature_offset` and `feature_width`;
sample/position rows may be reordered or partitioned. Missing shards, overlaps,
dtype changes and disagreeing replicas are rejected. Ordinary unannotated
tensors on multiple ranks must be bitwise-identical replicas. The case/weight
identity must already describe complete logical weights, not local shard names.

### Meaning of first divergence and explicit limits

Execution events are keyed by `(forward frame, logical boundary, invocation)`,
not module registration order. Corresponding workers must agree on the observed
event sequence before a first-event claim is made. Different call counts or
event orders produce `not_comparable`, not a guessed alphabetical match.
Different physical ranks are never ordered by wall-clock timestamps.

The report includes the tensor-tree field, invocation, last equal observed
boundary and, for explicitly mapped tensors, sample/token/feature coordinates.
It distinguishes equal declared inputs from already different inputs. This is
an **observed eager event order**, not a complete causal DAG. Different
prefill/decode schedules, pipeline-parallel stage graphs, concurrent root
forwards, graph-only bugs and branch frontiers still require an explicit replay
or event adapter. A coarse fused boundary gives an interval, not a kernel cause.

Every worker writes an open session, append-only event snapshots, then a
successful completion record. A crashed or interrupted run is inconclusive.
Offline analysis streams boundary payloads rather than retaining all model
tensors at once. `rlkernel.events.v1` bundles are separate from, and do not
reinterpret, existing `rlkernel.capture.v1` bundles.

### Native HF/vLLM acceptance without a model kernel

`examples/debug/hf_vllm_native.py` creates a tiny random Llama checkpoint locally,
uses its original HF forward and the original vLLM engine, and maps actual
embedding/projection boundaries plus final hidden-state endpoints. It checks
independently reconstructed runtime parameter inventories, actual token IDs and
actual positions. The fixed batch has two equal-length, unpadded samples with
unique token IDs so physical row ownership can be checked unambiguously.

```bash
VLLM_PLUGINS='' python -m examples.debug.hf_vllm_native --output /path/to/new-run
./rlk debug /path/to/new-run/native
./rlk debug /path/to/new-run/injected
```

The native arm may legitimately differ; this tool diagnoses, rather than
silently replacing, native kernels. The injected arm must identify
`embedding.output` with equal observed inputs. This acceptance is TP=1 eager
prefill, not training backward, optimizer updates, paged decode, MoE or arbitrary
production-scheduler certification. The example intentionally disables optional
vLLM plugins and does not download a checkpoint or alter production defaults.

Verified on **2026-10-24**, AMD Instinct MI300X VF, PyTorch
`2.12.0+rocm7.14.0a20260608`, HIP `7.14.60850`, Transformers `5.10.4`,
vLLM `0.26.1rc1.dev543+g11d76fc72`:

| Arm | Observed result |
| --- | --- |
| Native HF/vLLM | Runtime weights and input gates pass. First observed difference: layer 0 attention output-projection **input**, sample 0, token 1, feature 1. The prior equal boundary is embedding output. 326/1024 elements differ at that boundary; max absolute difference `0.001953125`. This localizes an interval, not an output-projection kernel cause. |
| Injected embedding fault | First difference is `embedding.output`, sample 0, token 0, feature 0, with equal observed inputs. Exactly one element differs at that boundary. |

Both arms contain 12 paired events. Live and CLI `--report-only` JSON reports
agree exactly; exit code 1 denotes the expected observed mismatch. The final
regression command covering `test_debug_events.py`, `test_consistency_debugger.py`
and `test_alignment_diagnostics.py` passes **104 tests** on the MI300X host.
The same suite on a CPU-only Windows environment passes 103 tests and skips the
one device-specific smoke test. These counts are scoped to those three files,
not a claim that the entire repository test suite ran.

### Qwen3 Dense native parallel matrix

`examples/debug/qwen3_parallel.py` extends device acceptance to the existing
official Qwen3 Dense configurations using **native Megatron/Transformer Engine**
on the training side and native vLLM on the rollout side. It does not copy a
model forward or launch a Ray server. The existing Qwen3 grouped-QKV, gate/up
and TP parameter layout was extracted into `debug/qwen3_layout.py`; both the
legacy live debugger and this portable capture reuse it.

For the Qwen3-8B run, use the complete 36-layer checkpoint, not a reduced model.
The frozen case contains two unpadded sequences of 16 tokens each. Megatron's
native `get_batch_on_this_cp_rank` owns CP token partitioning, and native TE
executes CP attention with `cp_comm_type="p2p"`. Bindings derive logical
sample/position coordinates from actual runtime token and position tensors.
TP-sharded projection inputs carry explicit feature offsets; replicas and CP
rows are verified during reconstruction. Each worker independently reconstructs
actual parameters across its TP group and hashes them before and after forward.
The CP helper receives the **explicit context-parallel group**; defaulting to
WORLD would also partition across TP ranks. Before capture, TP replicas must
agree on the actual token/position slice. The comparator independently rejects
incomplete feature coverage even if the union of incorrectly partitioned tokens
still covers the whole batch.

Example inside the configured ROCm environment, from the repository root
(choose GPUs that are actually available):

```bash
python -m examples.debug.qwen3_parallel matrix \
  --model /workspace/ljj/debug-models/Qwen3-8B \
  --output /workspace/ljj/qwen3-debug-matrix-new \
  --devices 0,1,2,5 \
  --training 1x1,2x1,1x2,2x2 \
  --rollout-tps 1,2,4
```

Training entries are `TPxCP`. Each of the four training configurations runs
once; each rollout TP runs a native arm and a known embedding-output fault arm
in one engine session. Completed lane artifacts are immutable and paired
offline using hard links (copies when hard links are unavailable). Thus seven
framework launches cover twelve topology pairs and twelve corresponding fault
controls, rather than relaunching both frameworks for every pair.

`matrix.json` records the status, event count and first difference for every pair.
An acceptance **pass** means comparable real evidence and, for the fault arm,
correct localization with equal observed inputs. It does **not** mean native
Megatron and vLLM are bitwise identical, or that a kernel cause has been proven.
Compare `embedding`, each layer's attention output-projection and FFN down
projection, final norm output and the actual final-hidden endpoint. Output-only
final norm avoids fabricating an input for vLLM's fused residual normalization.
Logits, sampling, backward, optimizer updates, rollout CP and paged decode are
outside this matrix's numerical acceptance scope.

The `compare` subcommand performs only offline pairing from completed `lanes/`
in a prepared output directory. It is useful when independently captured lanes
are brought together; it does not silently relaunch a failed runtime or overwrite
an existing paired bundle. Each paired bundle also supports the normal
`rlk debug BUNDLE --report-only` path.

#### MI300X parallel acceptance, 2026-10-24

The full Qwen3-8B checkpoint was pinned to revision
`b968826d9c46dd6066d109eabc6255188de91218`. The tested stack was Megatron Core
`0.18.0`, Transformer Engine `2.14.0.dev0+e6ede467`, PyTorch
`2.12.0+rocm7.14.0a20260608`, HIP `7.14.60850`, Transformers `5.10.4` and vLLM
`0.26.1rc1.dev543+g11d76fc72.rocm714`. PP/DP were 1; rollout CP was 1.

Each cell below passed both the native diagnostic and injected-fault acceptance:

| Megatron TP / CP | vLLM TP=1 | vLLM TP=2 | vLLM TP=4 |
| --- | --- | --- | --- |
| 1 / 1 | pass | pass | pass |
| 2 / 1 | pass | pass | pass |
| 1 / 2 | pass | pass | pass |
| 2 / 2 | pass | pass | pass |

All **24 cases** contain 149 paired execution events and agree on all **399
logical parameter fingerprints** reconstructed from actual runtime weights.
Native embedding outputs agree bitwise in all twelve topology pairs. Their
first observed difference is uniformly `layer.0.attention_projection.input`,
sample 0, token 0, feature 123. The maximum difference at that boundary is
`0.0009765625` for training CP=1 and `0.001953125` for CP=2. These native
differences are reported, not hidden by a tolerance or labelled a proven cause.

All twelve fault controls localize `embedding.output` with equal observed inputs,
sample 0, token 0, feature 0, and exactly one differing element. All 24 CLI
`--report-only` reports match their initial matrix-analysis JSON exactly. A missing-worker
negative control is correctly `not_comparable`.

The first mixed-TP attempt exposed a WORLD-versus-CP batch-partitioning mistake
in the example integration. Its incomplete feature coverage was rejected rather
than interpreted as numerical drift. The affected TP=2 training lanes were
recaptured with an explicit CP group; already complete unaffected lanes were
reused unchanged. Final evidence run: `pr442-qwen3-parallel-20261024-03`.

The four diagnostic test files (`test_debug_events.py`,
`test_debug_qwen3_parallel.py`, `test_consistency_debugger.py`,
`test_alignment_diagnostics.py`) pass **123 tests** on MI300X. The CPU-only
Windows run passes 122 and skips the device smoke test. This is not a
whole-repository regression claim, backward/optimizer acceptance, or a
measurement of production performance.

### Legacy explicit boundary API

Existing explicitly ordered, normalized boundary bundles remain supported:

```python
from pathlib import Path
import torch
from rl_engine.alignment.debug import Capture, fingerprint

root = Path("/tmp/my-mismatch")
boundaries = [
    dict(id="layer0.input", module="custom", layer=0, stage="input",
         positions=[0, 1], width=4),
    dict(id="layer0.output", module="custom", layer=0, stage="output",
         positions=[0, 1], width=4, inputs=["layer0.input"], contracts=["scale"]),
]
Capture.prepare(root, boundaries, scope="layer0 only; actual production forward")

# Execute this part on each side with its OWN actual values. Never copy the
# training identity onto the rollout side without checking its actual state.
def observe_side(side, layer, x, tokens, mask, positions, sampling):
    identity = dict(
        weights={k: fingerprint(v) for k, v in layer.state_dict().items()},
        tokens=fingerprint(tokens), mask=fingerprint(mask),
        positions=fingerprint(positions), sampling=sampling,
    )
    capture = Capture(root, side, identity)
    capture.record("layer0.input", x, positions=positions.tolist())
    with capture.observe({
        "layer0.output": (layer, lambda module, args, kwargs, output: dict(
            value=output, positions=positions.tolist(),
            contracts={"scale": float(module.scale)},
            route=type(module).__qualname__,
        )),
    }):
        return layer(x)  # Original forward is executed exactly once.
```

Use side names `training`/`rollout` for any reference/candidate pair. For
distributed runs prepare the manifest once, then give each worker its rank.
Normalize tensors to `[logical_rows, local_features]`, provide actual logical
positions and feature offsets, and declare global feature width. Every required
row/feature must be present; replicas must agree. Hash corresponding logical
weight tiles rather than physical shards when TP differs. Logical identities
describe the same full replay, while per-record positions describe local rows.
For reused/recurrent modules declare separate event IDs. Inputs in the manifest
are dependencies, including residual branches; they must precede their output.

The generic API preserves the application's execution mode. Whether hooks are
supported during a graph/compiled execution is the exporting adapter's contract;
the debugger does not claim graph evidence from an eager capture. Shape/layout
or contract mismatches and interrupted/missing captures exit with code 2. Raw
bits differ with valid evidence: code 1. Captured evidence agrees: code 0.

## Live adapter extension

An installed package can register an entry point under
`rlkernel.debug_adapters`. Its factory returns the `RuntimeAdapter` protocol in
`debug/adapters.py`: a unique name, model/backend matcher, replacement capability
descriptions, and a replay runner. Multiple matching adapters are an error.
Manifests do not contain executable import strings. Qwen3's adapter checks
explicit 8B and 0.6B BF16 dense contracts and derives Megatron's architecture
arguments from the checkpoint config. Recognizing the Qwen3 family alone does
not authorize a different shape. The eight-GPU launcher can capture the native
0.6B path, but tied embedding/output weights make strict replacement ineligible;
its real-device validation below uses the portable HF exporter.
Kernel candidates advertise eligibility and require replay verification; they
do not advertise unmeasured bitwise or performance guarantees.

The original execution configuration takes precedence over default profiles;
explicit CLI flags win. Replacements change only an isolated diagnostic arm.
They are not installed into production or committed into model source. To
adopt a verified change, apply it explicitly and validate the original workload.

## Validation recorded on 2026-09-22

The portable API was exercised with the official pretrained **Qwen3-0.6B**
checkpoint (596,049,920 parameters), BF16 Hugging Face SDPA on one AMD MI300X VF.
The exporter declared 339 semantic boundaries across 28 layers for 17 fixed
tokens. Each side independently hashed the actual model state and input.

| Workload | CLI result | Observed evidence |
| --- | --- | --- |
| Full forward vs identical full forward | `equal`, exit 0 | All declared boundaries agree; 2,582,912 final-logit elements have zero bitwise differences |
| Full forward vs 8-token prefill plus tokenwise KV-cache decode | `diverged`, exit 1 | First observed difference: layer 0, FFN output, logical token 0, feature 624; declared gate/up inputs agree |

The first differing FFN row had one unequal BF16 element, maximum absolute
difference `3.0517578125e-05`. The second run used real execution differences,
without injected errors. This locates an observed boundary; it does not identify
which internal arithmetic instruction caused the discrepancy. Full forward and
cached forward are both HF paths, not a Megatron/vLLM training comparison.
The two captures, loading and report generation took about 23 seconds; this is
diagnostic runtime, not a production step-time benchmark.

No RL Kernel replacement was used or certified for 0.6B. Its dimensions, GQA
ratio and tied embedding/output weights differ from 8B, so a recognized family
does not imply replacement compatibility. The small-model test validates generic diagnosis
without requiring a compatible repair implementation.

The live **Qwen3-8B Megatron/vLLM** loop also passed on eight AMD MI300X VFs,
with training TP4/CP2 and rollout TP4/CP1. The user entry point was:

```bash
./rlk debug /path/to/frozen-replay.json --matrix auto
```

The configured machine profile supplies paths and topology. The frozen sample
supplies 13 tokens (9 prompt, 4 response), temperature 0.7, top-p 1.0 and top-k
-1. These are acceptance inputs, not hardcoded defaults. Neither arm enables
`--use-rollout-logprobs`.

| Arm | Observed boundaries | Independently scored response logprobs |
| --- | --- | --- |
| M000 | First difference at layer 0, Q projection, logical token 1, feature 1791; one BF16 element differs by `1.52587890625e-05` | 2 of 4 differ; maximum absolute difference `0.21364784240722656` |
| M111 | All captured boundaries across 36 layers and the output head agree bitwise | 0 of 4 differ; maximum absolute difference 0 |

Each arm contains 864 layer snapshots and 12 worker-specific output-head weight
records, with no capture errors. Logical weights and input identities match
across arms; executed backend checks pass, with no fallback. Both launchers and
arm processes exit successfully, and source fingerprints remain unchanged.
The live conclusion is `verified_on_replay`; `--report-only` exits 0 and produces
an identical report. Run ID: `closure-8b-accept-10`. Linux regression coverage:
124 passing tests, including malformed/missing evidence rejection, CLI topology
and sampling overrides, unaligned vocabulary shards, packed-sequence backend
records and TP-replica deduplication.

LM-head identity uses complete logical vocabulary-row hashes, excluding padding;
native and strict paths may use different shard sizes. Other weights retain
logical tiles. Worker capture exceptions are persisted and invalidate comparison
even when the execution framework reports a successful driver exit.

The baseline's Q-projection **input evidence is unknown** because the training
normalization is fused and not independently captured. The report establishes
the first observed boundary, not whether normalization or GEMM rounding caused
it. It verifies the complete M111 replacement on this replay, not the necessity
of each individual replacement. This acceptance does not certify all models,
CUDA configurations, rollout CP > 1, graph execution, optimizer updates or
production performance. The 0.6B portable captures were also reanalyzed with
the final implementation and retained their original equal/diverged conclusions.

## Workflow regression recorded on 2026-11-12

Fixed failed-arm resume, fresh retry directories, declared-baseline gating in
live/offline matrices, direct `diagnostics/` analysis and the missing
`tie_word_embeddings` default. Eight regression cases cover failed/incomplete
baseline and replacement arms, preservation of previous attempts, completed-arm
skipping, baseline ordering, analysis-root equivalence and the config default.

The 13-file diagnostic regression (the command in `native-runtime-capture.md`
plus `tests/test_debug_qwen3_parallel.py`) passed **283 tests with no skips** on an
AMD Instinct MI300X VF using PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211` and
Triton `3.6.0`. All seven `test_debug_device_runtime.py` cases and the existing
graph/event device smoke tests ran. Ruff and `git diff --check` passed.
The same suite on Windows/PyTorch `2.10.0+cpu` passed 273 tests and skipped ten
device/compiler/symlink-permission cases. This validates the workflow fixes and
bounded single-device protocols; it does not rerun the historical Qwen3-8B
multi-GPU matrix or certify production performance.
