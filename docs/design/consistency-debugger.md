# Consistency debugger

## Product contract

`rlk debug SOURCE --matrix auto` reproduces a discrepancy, locates the first observed
semantic boundary, checks for a compatible replacement, and verifies it in an
isolated replay. It never edits a production model, enables rollout-logprob
reuse, or treats an available replacement as a proven fix.

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
   verification separately. `verified` means the same captured replay passed;
   it is not a production deployment, a universal bitwise guarantee, or a
   performance measurement.

Manual `--matrix full` / `--matrix M111,M011` remains available. `--matrix
attribute` runs M000, M111 and the three mixed module ablations, then reports
which combination passed on this replay; it does not claim that an individual
module is necessary or sufficient in isolation. `--dry-run` materializes
potential jobs without launching them. `--report-only` recomputes reports while
preserving launch failures and cross-arm identity gates. `--resume DIR` checks
that the frozen tokens, mask and sampling values still match, then skips only
completed arms. `rlk doctor` performs host/runtime preflight before GPU
submission.

## Architecture

* `debug/core.py`: immutable CPU snapshots, raw-bit comparison, logical shard
  reconstruction; no model or execution-framework imports.
* `debug/evidence.py`: portable capture protocol, explicit PyTorch boundary
  hooks, coverage/contract comparison and conservative causal classification.
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
* `contract_difference`: an observed contract differs; causality is unproven;
* `verified`: original mismatch reproduced and a replacement passed with the
  same inputs and weights;
* `unresolved`: mismatch persists after available trials;
* `unsupported`: no live adapter/replacement for this configuration;
* `inconclusive`: insufficient evidence or execution failure.

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

## Portable capture API

An existing live adapter needs only the CLI command. A new model/application
integrates this API once; ordinary users consume its saved bundles with
`./rlk debug /path/to/bundle`, without a profile or installed GPU framework.
Names below are adapter-declared semantics, not automatic matches by class name.

For a model with no RL-Kernel adapter, a PyTorch application can start with the
generic automatic capture layer. It discovers leaf modules, records the first
tensor input and output event for each selected module, and leaves feature width
dynamic until the real forward runs on both sides:

```python
from rl_engine.alignment.debug import AutoCapture, fingerprint, prepare_auto

positions = list(range(sequence_length))
prepare_auto(
    bundle_dir, model,
    positions=positions,
    scope="production eager replay; automatic leaf boundaries",
    exclude=("lm_head",),
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
fast generic starting point, not a semantic guarantee: reused modules keep the
first event only, fused internals remain invisible, and different module graphs
must be mapped explicitly before their bundles are comparable.

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
