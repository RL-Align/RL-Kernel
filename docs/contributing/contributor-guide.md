# Contributor Guide

Use this guide before adding or moving a file. Start with the kind of PR you are
making, identify the layer that owns the behavior, and put its tests and supporting
resources alongside the corresponding area. One PR can span several layers without
putting all of its code in one directory.

The [repository layout](../architecture/repository-layout.md) contains the complete
directory overview. This guide explains how to use that layout when contributing.
Paths below are relative to the repository root; &lt;model&gt;, &lt;platform&gt;, &lt;engine&gt;,
and &lt;operator&gt; represent the specific component you are adding.

## Find the right place for your PR

| Your PR changes... | Start with... | Include the corresponding checks or documentation |
| --- | --- | --- |
| A model's topology, operator inventory, or assembly | rl_engine/models/&lt;model&gt;/ | tests/models/&lt;model&gt;/, model validation, and an explicit support scope |
| A layer reused by several models | rl_engine/layers/ | tests/layers/; add layer benchmarks when performance changes |
| An operator's public semantics or interface | rl_engine/ops/&lt;operator&gt;/ and rl_engine/contracts/ | tests/ops/, tests/contracts/, and docs/operators/ |
| A mathematical reference implementation | rl_engine/reference/ | tests/reference/ and the affected operator's accuracy tests |
| A CUDA, ROCm, MUSA, Ascend, or CPU kernel | rl_engine/backends/&lt;platform&gt;/&lt;operator&gt;/ | Backend tests, the operator contract, and measurements on the claimed hardware |
| A Triton implementation shared across devices | rl_engine/backends/shared/triton/ | Explicit platform eligibility and validation on each claimed platform |
| Device discovery or hardware capability detection | rl_engine/platforms/ | tests/platforms/ and affected dispatch tests |
| Dispatch eligibility, execution policy, or provenance | rl_engine/runtime/ | tests/runtime/, including unsupported configurations and fallback behavior |
| Partitioning, collective arithmetic, or communication | rl_engine/distributed/ | tests/distributed/ and the affected TP/CP/EP/DP configurations |
| A rollout engine adapter | rl_engine/integrations/engines/rollout/&lt;engine&gt;/ | Matching tests/integrations/engines/rollout/ coverage |
| A training engine adapter | rl_engine/integrations/engines/train/&lt;engine&gt;/ | Matching tests/integrations/engines/train/ coverage |
| A post-training framework or experiment lifecycle | rl_engine/integrations/orchestrators/&lt;framework&gt;/ | Orchestrator tests, integration configuration, and an integration guide |
| WS1, WS2, model consistency, or an ablation harness | rl_engine/validation/ | Matching harness tests, suite configuration, and reproducible evidence |
| A performance measurement or tuning procedure | benchmarks/ | Benchmark correctness tests where needed; measurements under reports/ |
| A CLI, build, container, or CI workflow | rl_engine/entrypoints/, build_tools/, docker/, or ci/ | The relevant entrypoint/build checks and updated usage instructions |
| Documentation only | The relevant topic under docs/ | Navigation and links, then a strict documentation build |

For example, a ROCm attention optimization used by both Qwen3 and Gemma belongs in
the ROCm attention backend. The model packages declare that they need attention;
they do not each receive a copy of the optimized kernel. Gemma still needs its own
model and integration validation before that optimization establishes Gemma support.

## Understand the directory boundaries

### Models and reusable layers

| Directory | Responsibility | What to put elsewhere |
| --- | --- | --- |
| rl_engine/models/ | Model assembly and shared model wrappers | Device-specific arithmetic belongs in backends |
| rl_engine/models/&lt;model&gt;/ | Architecture identity, dimensions, operator inventory, and model-specific assembly/reference wiring | Engine hooks belong in engine adapters; generic operators belong in ops |
| rl_engine/layers/ | Reusable compositions of operators, with an interface useful to more than one model | A layer used only to express one model's topology can stay with that model |

The model directories currently have different implementation states:

| Model family | Package | State in this refactor | Starting point |
| --- | --- | --- | --- |
| Qwen3-8B Dense | rl_engine/models/qwen3/ | Existing specification and reference assembly migrated | spec.py, reference.py, and tests/models/qwen3/ |
| DeepSeek V4 | rl_engine/models/deepseek_v4/ | Reserved package; adaptation is not implemented | Add the model specification and required operator mapping |
| Gemma | rl_engine/models/gemma/ | Reserved package; adaptation is not implemented | Add the model specification and required operator mapping |
| MiniMax H3 | rl_engine/models/minimax_h3/ | Reserved package; adaptation is not implemented | Add the model specification and required operator mapping |

An existing implementation is not a blanket certification of every model size,
dtype, hardware platform, topology, or training framework. State those dimensions
in your PR and link the evidence for the combinations you actually ran. Empty
packages, configuration examples, and CPU smoke tests do not establish GPU support.

### Operators, numerical contracts, references, and hardware

| Directory | Responsibility | Example |
| --- | --- | --- |
| rl_engine/ops/ | Semantic operator entry points and shared operator interfaces | Attention, GEMM, norm, activation, RoPE, embedding, MoE, logprob, loss, sampling, packing, and autograd |
| rl_engine/contracts/operators/ | Observable operator requirements: inputs, outputs, supported modes, and consistency rules | Attention, logprob, and loss contracts |
| rl_engine/contracts/profiles/precision/ | Versioned precision requirements and thresholds | Accuracy requirements against a reference |
| rl_engine/contracts/profiles/invariance/ | Invariance profile ownership | Requirements across supported execution configurations |
| rl_engine/contracts/diagnostics/ | Common ablation axes and diagnostic/report schemas | Definitions shared by multiple validation harnesses |
| rl_engine/reference/ | Mathematical or diagnostic reference implementations | A PyTorch attention reference |
| rl_engine/platforms/ | Device discovery and capability information | Determining which hardware features are available |
| rl_engine/backends/&lt;platform&gt;/&lt;operator&gt;/ | Python implementation, launch logic, and backend-specific wrappers | backends/cuda/attention/ and backends/rocm/attention/ |
| rl_engine/backends/shared/triton/ | One Triton implementation used by multiple validated devices | Shared arithmetic with platform-specific registration constraints |
| csrc/&lt;platform&gt;/ | Native CUDA, HIP, or other platform sources, grouped by operator where applicable | csrc/cuda/attention/ |
| csrc/common/ and csrc/bindings/ | Shared native support and Python/native registration | Common headers and extension bindings |

Use these rules when implementations overlap:

1. Reuse semantic operators across models. Put model dimensions and operator needs
   in the model specification instead of embedding model identity into a generic kernel.
2. Keep platform-exclusive implementations with their platform. A Triton kernel
   using ROCm-specific instructions belongs under backends/rocm/.
3. Share an implementation in backends/shared/triton/ only when its source and
   contract apply across the registered platforms. Validate each supported platform.
4. If a fusion genuinely depends on one model's architecture, use
   backends/&lt;platform&gt;/model_specific/&lt;model&gt;/ when adding that implementation.
   This is an extension convention, not an already implemented backend family.
5. Put tensor-layout conversion required by an engine in its adapter. Put the
   mathematical meaning of the operator in its semantic interface and contract.

### Runtime and distributed execution

| Directory or module | Responsibility |
| --- | --- |
| rl_engine/runtime/registry.py | Existing dispatch registration and priorities |
| rl_engine/runtime/semantic_registry.py | Operator capabilities, support constraints, and implementation provenance |
| rl_engine/runtime/operators.py | Bind resolved operators to execution and integration adapters |
| rl_engine/runtime/policy.py | Strict and diagnostic execution policies |
| rl_engine/runtime/plan.py and executor.py | Execution selections and stateless scoring/execution |
| rl_engine/runtime/provenance/ | Stable operator identities and evidence connecting a result to its implementation |
| rl_engine/runtime/selector/ and performance/ | Reserved ownership for future measured performance selection |
| rl_engine/distributed/algorithms/ | Partition semantics, collective arithmetic, and canonical reduction ordering |
| rl_engine/distributed/transports/ | Communication and transport bindings, including RCCL |

A faster implementation is eligible only after its numerical contract, dtype,
topology, device capabilities, and provenance satisfy the request. Performance
ranking must operate within that eligible set. The reserved selector and performance
packages do not currently provide a new automatic fastest-path router.

An engine's TP/CP hooks belong to its adapter; an engine-independent partition or
reduction algorithm belongs in distributed. Keep mathematical ordering separate
from the transport that moves data between ranks.

### Engines and post-training frameworks

| Directory | Responsibility |
| --- | --- |
| rl_engine/integrations/engines/rollout/ | Rollout-side engine interfaces and adapters |
| rl_engine/integrations/engines/rollout/vllm/ | vLLM operator hooks, runtime binding, sampling, memory, and parallel integration |
| rl_engine/integrations/engines/train/ | Training-side interfaces, contracts, and adapters |
| rl_engine/integrations/engines/train/megatron/ | Megatron operator hooks, runtime binding, and cross-configuration integration |
| rl_engine/integrations/engines/train/deepspeed/ | DeepSpeed adapter ownership; validate the specific behavior your PR adds |
| rl_engine/integrations/common/ | Logic and state actually shared between integrations, including weights and shared adapters |
| rl_engine/integrations/orchestrators/&lt;framework&gt;/ | Coordination of rollout, training, weight publication, and experiment lifecycle |
| rl_engine/integrations/orchestrators/vime/providers/ | VIME-specific providers |
| rl_engine/integrations/orchestrators/vime/experiments/ | VIME experiment runners and reproducibility logic |
| rl_engine/integrations/orchestrators/vime/patches/ | Versioned companion patches and their application instructions |

The miles and areal orchestrator packages are reserved. Add real implementation
and validation before describing either as supported. An orchestrator uses engine
adapters; it should not become another home for copies of kernels or model definitions.

### Validation harnesses and tests

rl_engine/validation/ contains reusable validation behavior. tests/ checks that
behavior and the production code. A standalone measurement program belongs in
benchmarks/, while a convenient command wrapper belongs in tools/validation/.

| Scope | Harness or implementation owner | Test location |
| --- | --- | --- |
| Operator semantics, accuracy, and WS1 invariance | ops/, contracts/, validation/operators/ | tests/ops/, tests/contracts/, tests/validation/operators/ |
| Reference implementations | reference/ | tests/reference/ |
| Platform-specific implementation behavior | backends/ and platforms/ | tests/backends/, tests/platforms/ |
| Model assembly and Dense chain validation | models/, layers/, validation/models/ | tests/models/, tests/layers/, tests/validation/models/ |
| WS2 collectives and partitioned execution | distributed/, validation/distributed/ | tests/distributed/{collectives,transports,tp,cp,ep,dp}/ |
| Dispatch, policy, and provenance | runtime/ | tests/runtime/ |
| Train and rollout adapters | integrations/engines/ | tests/integrations/engines/{train,rollout}/, then the engine name |
| Shared integrations and orchestrators | integrations/common/, integrations/orchestrators/ | tests/integrations/common/, tests/integrations/orchestrators/ |
| Paired scoring across configurations | validation/cross_config/ | tests/validation/cross_config/ |
| Ablation matrices and negative controls | validation/ablation/ | tests/validation/ablation/ |
| Validation reporting and shared helpers | validation/reports/, validation/common/ | Relevant suites under tests/validation/, including common/ |
| Full workflows, commands, and packaging | Entrypoints, build tools, and assembled integrations | tests/e2e/, tests/entrypoints/, tests/build/ |
| Benchmark harness behavior | benchmarks/ | tests/benchmarks/ |

The harness and implementation paths in the middle column are relative to
rl_engine/, except for benchmarks/ and the final workflow/build row.
Some directories are reserved extension points rather than populated suites.

Use rl_engine/validation/fixtures/ for fixtures needed by an installed validation
command, tests/helpers/ for helpers used only by pytest, and tests/data/ for
small deterministic test inputs. Keep shared validation mechanics in
validation/common/; generic package utilities belong in rl_engine/utils/ only
when no domain-specific owner fits.

WS1 and WS2 describe validation scopes. They are not additional backend trees.
For example, a ROCm WS1 attention case uses the existing ROCm attention
implementation and an operator validation configuration rather than a second kernel
under a directory named ws1.

### Configuration, benchmarks, tools, and delivery

| Directory or file | Responsibility |
| --- | --- |
| rl_engine/config/ | Load, validate, and interpret runtime/workload configuration |
| rl_engine/config/workloads/ | Packaged and versioned workload definitions used by the library |
| configs/models/, hardware/, and policies/ | Repository-level model, machine, and execution-policy presets under configs/ |
| configs/workloads/, ablations/, benchmarks/, and tuning/ | Reproducible run inputs and matrices under configs/ |
| configs/suites/{ws1,ws2,e2e}/ | Suite composition for the corresponding validation scope |
| configs/experiments/{cross_config,vime}/ | Named experiment definitions and inputs |
| configs/integrations/engines/{rollout,train}/ | Engine-role configuration |
| configs/integrations/orchestrators/ | Post-training framework configuration |
| configs/local/ | Machine-local configuration; keep credentials and private machine state untracked |
| benchmarks/{operators,layers,backends,distributed,models,e2e}/ | Measurements at the named scope |
| benchmarks/{common,tuning,profiling}/ | Shared measurement helpers, tuning experiments, and profiling |
| examples/{operators,post_training}/ | Small examples that show how to use an operator or integration |
| rl_engine/entrypoints/ | Installed command implementations |
| bin/ | Thin checkout launchers; the root rlk remains a compatibility entry point |
| tools/validation/ | Operator, model, distributed, and related validation commands |
| tools/weights/, env/, checks/, and benchmarking/ | Weight preparation, environment inspection, repository checks, and benchmark utilities under tools/ |
| tools/migration/ | Migration helpers and the source/destination map in layout.json |
| build_tools/ | Extension selection, compiler options, and build environment handling |
| setup.py, pyproject.toml, and MANIFEST.in | Build entry point, package metadata/entry points, and packaged source/resource declarations |
| ci/run.py and ci/scripts/ | Reusable suite launchers and CI execution commands |
| ci/jobs/, runners/, and providers/ | Reserved organization for job definitions, runner setup, and provider integration under ci/ |
| .github/workflows/ and .github/CODEOWNERS | GitHub triggers/job orchestration and review ownership |
| docker/ | Existing CUDA/ROCm image definitions and their build dependencies |
| requirements/ | Runtime, test, development, documentation, benchmark, and build dependencies; integrations/ and constraints/ are reserved |
| reports/experiments/ and reports/releases/ | Results deliberately checked in for an experiment or release |
| reports/archive/ | Historical measurements and evidence |
| artifacts/ | Ignored local outputs, temporary run evidence, and build products |

Several configuration categories are currently scaffolding. Add a runnable
configuration and its consumer together; a directory name alone does not make a
new configuration format or a CI job executable. If an installed command needs a
resource, package it with the owning module and verify it can be loaded outside
the repository checkout.

Keep documentation with its audience and purpose:

| Documentation directory | Content |
| --- | --- |
| docs/getting_started/, usage/, and operators/ | Installation, workflows, and operator behavior |
| docs/contributing/ | Contributor workflow, placement rules, testing, and documentation instructions |
| docs/architecture/ | Layer responsibilities, dispatch, and architectural decisions |
| docs/contracts/ | Human-readable numerical and distributed contracts |
| docs/integrations/ | Engine/orchestrator integration and reproduction guides |
| docs/validation/ and docs/benchmarking/ | Acceptance procedures and measurement methodology |
| docs/api/ and docs/cli/ | Public interface and command references |
| docs/archive/ | Historical plans and closeout documents |
| docs/blog/, community/, assets/, and mkdocs/ | Announcements, community information, site assets, and theme support |

Subdirectory names abbreviated after the first path in this table remain under
docs/. Add current design material to docs/architecture/; docs/design/ is a
legacy location. A historical closeout report describes its recorded commit and
environment, not the validation status of a new PR.

## Follow a contribution path

### Add a model

1. Start with the closest existing model and reuse its applicable operators and
   layers. For Qwen3 Dense, inspect rl_engine/models/qwen3/spec.py and reference.py.
2. Put architecture identity, dimensions, weight/configuration assumptions, and
   the operator inventory in models/&lt;model&gt;/. Keep engine-specific loading or
   hook behavior in the corresponding adapter.
3. If the model needs new arithmetic, add the semantic contract and implementation
   in their own operator/backend layers. A new model folder alone is insufficient.
4. Add model tests under tests/models/&lt;model&gt;/, the relevant model validation,
   and reproducible configurations. Record model revision, weights, tokenizer,
   dtype, platform, and topology for end-to-end claims.
5. Document implemented, unsupported, and untested combinations separately.
   Preserve the existing pinned Qwen3 workload when introducing another model.

### Add or optimize an operator

1. Find the existing semantic interface and contract. Extend them only when the
   observable operator behavior requires it.
2. Put a mathematical reference in reference/, device implementation in
   backends/&lt;platform&gt;/&lt;operator&gt;/, and native source in csrc/&lt;platform&gt;/.
3. Register capabilities through the existing runtime mechanisms. Declare dtype,
   device, topology, determinism, and fallback constraints truthfully.
4. Extend focused semantic/backend tests. Check the applicable forward, backward,
   accuracy, and invariance requirements, including unsupported requests.
5. Put measurements in the appropriate benchmark directory and explain the
   operator in docs/operators/, using the
   [operator documentation template](operator-doc-template.md).

For example, a ROCm GEMM optimization normally touches backends/rocm/gemm/,
possibly csrc/rocm/, the matching backend/operator tests, and operator benchmarks.
It needs a model-specific change only when it changes that model's assembly or
declared operator requirements.

### Add hardware support

Add discovery/capabilities in platforms/, implementations in the platform backend,
and native build selection in build_tools/ where needed. Reuse a shared Triton
implementation only for validated configurations. Exercise both platform discovery
and numerical behavior on the target device. Add dependency, Docker, and CI support
when those paths are runnable. A reserved MUSA or Ascend directory does not certify
a new native implementation or every operator on that platform.

### Integrate an engine or post-training framework

Choose engines/rollout/&lt;engine&gt;/ for rollout hooks and
engines/train/&lt;engine&gt;/ for training hooks. For coordination across engines,
use orchestrators/&lt;framework&gt;/. Extract logic into integrations/common/ when
multiple integrations actually need it. Mirror these locations in tests, place
run inputs under configs/integrations/, and document setup and supported modes
under docs/integrations/.

A framework PR should identify weight synchronization/versioning, tensor layouts,
scoring entry points, gradient/update scope, and independent train/rollout score
recomputation. A wrapper that imports successfully does not establish numerical
consistency across the integrated engines.

### Extend WS1, WS2, or an ablation matrix

Put reusable checking logic in the relevant validation/ package, suite inputs
in configs/suites/, and ablation inputs in configs/ablations/. Keep the operator
or distributed implementation in its existing owner. Add harness tests in the
matching tests/validation/ or tests/distributed/ suite. Include negative controls
that demonstrate the check detects the mismatch or unsupported condition it targets.

For a performance experiment, place the executable benchmark in benchmarks/ and
save local measurements in artifacts/. Check in selected evidence and methodology
under reports/experiments/ when the results need to accompany the PR.

### Move files or change packaging

Keep new implementation code in canonical modules. The old rl_engine.kernels,
alignment, testing, and legacy script locations exist for compatibility;
do not add new implementations there. When changing a compatibility alias, preserve
the shared registry, cache, and module state and test the old entry point as well
as the new one.

Update tools/migration/layout.json when extending this migration. Update imports,
resource loading, documentation examples, workflow paths, and package manifests
together. Preserve versioned workload/contract identity and historical evidence;
document a deliberate version change instead of silently rewriting those identities.
Verify installed resources and commands outside the checkout when packaging changes.

## Validate and describe your PR

Choose existing focused suites for the behavior you changed. Add a regression case
when there is a behavior or failure mode to protect. Test semantic outcomes and
negative controls rather than only checking that a new file exists. Run GPU and
distributed validation for the configurations your PR claims to support.

After setting up the [development environment](../getting_started/installation.md),
these commands provide starting points from the repository root:

```bash
# Packaging and legacy import compatibility.
python ci/run.py layout -q

# Operator and attention-contract examples; select the suites your PR affects.
python -m pytest tests/contracts/operators/test_attention_contract.py -q
python -m pytest tests/ops/attention/test_attention.py -q -k "not large and not gpu"

# Documentation and formatting checks.
mkdocs build --strict -f mkdocs.yaml
pre-commit run --files docs/contributing/contributor-guide.md docs/.nav.yml

# Every commit needs the contributor's DCO sign-off.
git commit -s
```

See [Testing](testing.md), [gtest usage](gtest-usage.md), and
[Dense CUDA/ROCm acceptance](../validation/refactor-acceptance.md) for the applicable
validation procedures. CPU CI success and hardware jobs skipped on a fork do not
replace target-device evidence.

In the PR description, include:

- The behavior changed and why each affected layer owns its part of the change.
- The model, hardware, dtype, topology, and engine/framework versions covered.
- Test and benchmark commands, results, and configurations not exercised.
- Numerical-contract evidence. For this refactor's Dense acceptance, require zero
  raw-bit mismatches between independently recomputed training and rollout scores
  on aligned active tokens; preserve the separate accuracy requirements against
  mathematical references.
- Comparable performance measurements with workload, warmup, repeated samples,
  median latency/throughput, and memory. This refactor reports differences for
  reviewer acceptance without an automatic percentage regression threshold.
- Updated usage/architecture documentation, navigation, and compatibility notes
  when users or contributors need them.

CUDA and ROCm acceptance are separate. Label unsupported and untested cells
explicitly, and report the backend that actually executed. A faster result cannot
waive a failed numerical contract.

## References

This guide uses the task-oriented tables and explicit implementation status seen
in vLLM's [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/#text-generation)
page as an organizational reference. Its
[model contribution guide](https://docs.vllm.ai/en/latest/contributing/model/)
also separates implementation, registration, and testing. The directory ownership
and numerical acceptance rules above are specific to RL-Kernel.
