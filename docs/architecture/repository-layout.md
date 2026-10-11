# Repository layout and ownership

This refactor starts at upstream `main` (`bea224b`). It borrows vLLM's separation
of model assembly, layers, platform discovery, implementation backends, engine
integration, native sources, and test infrastructure. RL-Kernel additionally
owns numerical contracts and train–rollout validation.

For a PR-by-PR placement guide, directory responsibilities, and validation
requirements, start with the [Contributor Guide](../contributing/contributor-guide.md).

## Canonical tree

```text
RL-Kernel/
├── rl_engine/
│   ├── config/                     # Validated workload/configuration loaders
│   │   └── workloads/              # Packaged, versioned WS1 workload identity
│   ├── models/
│   │   ├── qwen3/                  # Pinned Dense spec, operator inventory, model assembly
│   │   ├── deepseek_v4/             # Reserved; no registered implementation
│   │   ├── gemma/                   # Reserved; no registered implementation
│   │   └── minimax_h3/              # Reserved; no registered implementation
│   ├── layers/                     # Extension point for reusable model layers
│   ├── ops/                        # Hardware-independent operator entry points
│   │   └── {attention,gemm,norm,activation,rope,embedding,moe,logprob,loss,sampling,packing,autograd}/
│   ├── contracts/
│   │   ├── operators/              # Attention, logprob, loss numerical contracts
│   │   ├── diagnostics/            # Shared ablation axes and reporting schema
│   │   └── profiles/{precision,invariance}/
│   ├── reference/                  # PyTorch mathematical references and diagnostic paths
│   ├── platforms/                  # Device discovery and platform capabilities
│   ├── backends/
│   │   ├── {cuda,rocm,musa,ascend,cpu}/
│   │   │   └── {attention,gemm,norm,activation,rope,embedding,moe,logprob,loss,sampling,packing}/
│   │   └── shared/triton/          # Implementations shared across supported devices
│   ├── runtime/
│   │   ├── registry.py             # Existing contract-aware dispatch and priorities
│   │   ├── semantic_registry.py    # Capabilities, support checks, provenance
│   │   ├── operators.py            # Operator bridge and execution bindings
│   │   ├── policy.py               # Strict versus diagnostic execution policy
│   │   ├── plan.py                 # Executable P/P, P/R, R/P, R/R selections
│   │   ├── executor.py
│   │   ├── provenance/            # Stable versioned operator identities
│   │   └── {selector,performance}/ # Reserved for future measured selection policies
│   ├── distributed/
│   │   ├── algorithms/             # Canonical arithmetic/collective ordering
│   │   └── transports/             # RCCL and transport bindings
│   ├── integrations/
│   │   ├── common/                 # Shared adapters, weights, linear-logp, state
│   │   ├── engines/
│   │   │   ├── rollout/{vllm,interface.py}
│   │   │   └── train/{megatron,deepspeed,contract.py}
│   │   └── orchestrators/
│   │       ├── vime/{providers,experiments,patches}/
│   │       └── {miles,areal}/       # Reserved integration points
│   ├── validation/
│   │   └── {operators,models,distributed,cross_config,ablation,reports,common,fixtures}/
│   ├── entrypoints/                # Installed CLIs
│   └── utils/
├── csrc/
│   ├── bindings/                   # Python/native registration
│   ├── common/
│   └── {cuda,rocm,musa,ascend}/      # Native sources, grouped by operator family
├── build_tools/                     # Extension selection/compiler flags/environment helpers
├── configs/
│   ├── {models,hardware,policies,workloads,ablations,benchmarks,tuning,local}/
│   ├── suites/{ws1,ws2,e2e}/
│   ├── experiments/{cross_config,vime}/
│   └── integrations/
│       ├── engines/{rollout,train}/
│       └── orchestrators/vime/
├── tests/
│   ├── {contracts,reference,ops,layers,backends,platforms,runtime,models}/
│   ├── distributed/{collectives,transports,tp,cp,ep,dp}/
│   ├── integrations/
│   │   ├── common/
│   │   ├── engines/{rollout,train}/
│   │   └── orchestrators/
│   ├── validation/{operators,models,cross_config,ablation,common}/
│   └── {benchmarks,e2e,entrypoints,build,helpers,data}/
├── benchmarks/
│   └── {common,operators,layers,backends,distributed,models,e2e,tuning,profiling}/
├── examples/{operators,post_training}/
├── bin/                            # Thin checkout launchers; ./rlk remains supported
├── tools/
│   └── {validation,weights,env,checks,benchmarking,migration}/
├── ci/{run.py,scripts,jobs,runners,providers}/
├── .github/workflows/              # Provider-specific CI triggers and security policy
├── docker/                         # Existing CUDA/ROCm CI and development images
├── requirements/                   # Runtime, test, development, docs, benchmark, build
├── reports/{archive,experiments,releases}/
├── docs/{architecture,contracts,integrations,archive,...}/
├── artifacts/                      # Ignored local validation/build output
├── pyproject.toml                  # Package metadata, entry points and test configuration
├── setup.py                        # Thin build entry point
└── MANIFEST.in                     # Native sources/build tools and packaged resources
```

An empty package is a reserved ownership boundary, not an advertised supported
model, engine, or platform. Existing MUSA dispatch through validated shared
implementations is retained; a MUSA directory alone does not certify a native
MUSA kernel. Docker images and CI jobs are added only when executable support exists.

## Where overlapping attention and GEMM implementations belong

- `models/qwen3/spec.py` owns the Qwen3 topology and operator inventory. A model
  consumes common operators; it does not own copies of generic GEMM/attention.
- `ops/attention` and `ops/gemm` own the semantic entry points. `contracts` owns
  numerical requirements. `runtime` resolves implementation capabilities.
- `backends/cuda/attention`, `backends/rocm/attention`, and equivalent GEMM
  directories own device-specific implementations. Platform-exclusive MFMA and
  ROCm attention kernels stay in `backends/rocm`, even when implemented in Triton.
- `backends/shared/triton` owns a single implementation when the same source and
  contract are valid across devices. Register it for each supported platform;
  do not copy it into each platform directory.
- A future truly model-specific fusion belongs under
  `backends/<platform>/model_specific/<model>/`, with its semantic requirements
  declared by that model. Create it when there is an implementation to own.
- Engine tensor layouts and hooks belong in the corresponding rollout/train
  adapter, never in the numerical kernel or the model identity.

Strict routing must first establish contract eligibility, supported topology,
precision, and validated provenance. Performance rankings can only choose among
eligible implementations. This PR preserves the existing dispatcher, priorities,
strict/fallback policies and kernel algorithms. The reserved selector/performance
packages do not introduce an unvalidated fastest-path router.

## Tests and acceptance ownership

WS1 is a validation scope, not a second copy of each backend. Operator tests
live in `tests/ops`, implementation-specific checks in `tests/backends`, WS1
harness tests in `tests/validation/operators`, and Qwen3 assembly tests in
`tests/models/qwen3`. WS2 partition/collective tests live in `tests/distributed`;
engine hook tests mirror `integrations/engines/{rollout,train}`. End-to-end runs
combine model, backend, topology and engines through configuration.

Keep test filenames and assertions during migration. Source-inspection tests
must inspect the canonical implementation, not a compatibility wrapper. Benchmark
code lives under `benchmarks`; captured measurements and historical reports live
under `reports`. Large experiment launchers live with their orchestrator, while
`examples` holds short usage examples.

## Compatibility and migration

The machine-readable `tools/migration/layout.json` records source/destination
paths and split modules. Old `rl_engine.kernels`, `alignment`, `testing`, executor
and integration import paths are thin aliases; leaf aliases return the same
module object so registries, caches and runtime state are not duplicated. Legacy
script paths delegate to the canonical tool, and legacy shell entry points
forward arguments and exit codes. New code should import canonical modules.

The WS1 workload JSON and precision contract retain their original bytes and
hashes. Historical operator IDs such as `rl_engine.kernels.ops.*` remain stable
v1 identifiers. `runtime/provenance/identity.py` translates the relocated class
module to that stable identifier when producing WS1 evidence. Executable source
fingerprints still reflect the new code; recorded identity does not substitute
for runtime provenance validation. Old reports, companion patches and pinned
patch hashes are preserved. Relative resource aliases retain old experiment
entry points; package builds include the resolved resource contents.

See [Dense CUDA/ROCm acceptance](../validation/refactor-acceptance.md) before
accepting the refactor on GPU hardware.
