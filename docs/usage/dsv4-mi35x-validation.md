# DSV4 native four-layer GRPO on MI35x

This change enables the MI35x (`gfx950`) paths needed by the native four-layer
recipe in [PR #711](https://github.com/RL-Align/RL-Kernel/pull/711). Hardware
validation below uses **MI350X**. It does not establish MI355X hardware acceptance,
long-context training, full-model training, or arbitrary parallel layouts.

The [MI300X measured report](../../examples/operators/dsv4_4l/results/native_grpo_pp2_dp1_20261117/README.md)
remains the authority for its original results and remaining failures. These
changes do not rewrite that report or its acceptance thresholds.

## Architecture-specific changes

AMD documents two different FP8 encodings: CDNA3/gfx942 uses FNUZ; CDNA4/gfx950
uses OCP. See the [ROCm inference workload guidance](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html)
and [HIP low-precision types](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html).

- Select the native FP8 dtype for activation quantization and QNorm output.
  Preserve the existing gfx942 FNUZ weight reinterpretation and doubled scales.
  On gfx950, use the original OCP checkpoint bytes and original scales directly.
- Decode ordinary SWA cache entries using the encoding written by vLLM on the
  current device. Compressed-cache OCP decoding is unchanged.
- Enable gfx950 for the pinned joint-attention kernel, mHC, router projection,
  native hash routing, and packed W4A16 projection after running their numerical
  and invariance checks. Their kernel arithmetic and launch geometry are unchanged.
- Retain rejection of other architectures. This is not a blanket removal of
  gfx942 guards throughout the repository; unrelated operator paths remain gated.
- In the native gfx950 rollout's loading scope, select vLLM's unfused W4A16
  implementation for automatic DeepSeek-V4 MoE dispatch. The default AITER route
  has a batch-size-dependent activation format on gfx950. The native actor's
  recipe requires BF16 activations. Explicit backend requests and other models
  retain upstream selection; the selector is restored when loading ends.

The last item requires the accompanying
[`verl-mi35x-w4a16.patch`](../../examples/operators/dsv4_4l/patches/verl-mi35x-w4a16.patch)
at verl's native worker loading entry point. It is a scoped integration change
against the pinned verl commit, not a change to the training configuration.
Its MI350X end-to-end validation is recorded below.

Packed FP4/FP8 weights and scales stay frozen. The recipe trains selected
nonquantized parameters; it is **not LoRA**. Training configuration, data, layer
ownership, optimizer settings, sampling settings, and strict parity checks are
unchanged.

## Rebuild the pinned vLLM extensions for gfx950

The delivered image `vensenmu/rl-kernel:pr711-grpo-stress`, manifest
`sha256:45307c791baa2db3c4fc6f70be960b301fcc390d3a103d07bdd991f479b2396c`,
contains gfx942-only code objects in the inspected core extensions. The fused QKV
test failed on MI350X with `hipErrorInvalidDeviceFunction` before rebuilding.

Rebuild from the **same** vLLM source commit
`c07d58dc82d75f7d1a947b3264b460384ec3e18d`, using the image's existing toolchain and
dependencies. No vLLM Python/C++ source patch or dependency upgrade was used.
`PYTORCH_ROCM_ARCH` is the target selector described by the
[vLLM ROCm source-build documentation](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.rocm.inc.md).

These are the commands used **inside a separate development container** of that
image. `/cache` is a writable build directory. Apply the RL-Kernel patch to that
container's source separately; retain the supplied image unchanged.

```bash
export TMPDIR=/tmp
export PYTORCH_ROCM_ARCH=gfx950
export ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel
export ROCM_HOME="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"

cmake -S /opt/dsv4/vllm -B /cache/vllm-mi35x-build -G Ninja \
  -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE=/opt/dsv4-venv/bin/python \
  -DROCM_PATH="$ROCM_PATH" \
  -DCMAKE_HIP_ARCHITECTURES=gfx950 \
  -DCMAKE_INSTALL_PREFIX=/opt/dsv4/vllm \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DFETCHCONTENT_BASE_DIR=/opt/dsv4/vllm/.deps \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DCMAKE_JOB_POOLS=compile=16 \
  -DCMAKE_JOB_POOL_COMPILE=compile

cmake --build /cache/vllm-mi35x-build \
  --target _C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C -j 16
for component in _C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C; do
  cmake --install /cache/vllm-mi35x-build --component "$component"
done
```

The four rebuilt `.so` files and changed RL-Kernel runtime files were copied into
a derived image. No ROCm architecture override was used to disguise the device.
Source provenance records the RL-Kernel and companion verl patches as dirty
worktrees and records the unchanged source commits of vLLM and Megatron.
Binary hashes are recorded separately
because the vLLM source commit alone cannot identify a rebuilt extension.

## Rebuild the supplied Transformer Engine common library

The delivered TE common library is also gfx942-only. Its supplied source version
is `2.21.0.dev0`, with commit `b5c00f43283086bf0751f3bf3a179f1e921bbe59` recorded
in `/opt/dsv4/artifacts/te-source-sha.txt`. The source archive has no `.git`
directory; retain a source-file hash manifest as additional provenance.
AMD's [TE installation instructions](https://github.com/ROCm/TransformerEngine/blob/dev/docs/installation.rst)
document `NVTE_ROCM_ARCH=gfx950` for MI350.

Using the same toolchain exports as above, the common-library build is:

```bash
export NVTE_ROCM_ARCH=gfx950 NVTE_USE_ROCM=1
export HIP_DEVICE_LIB_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
export PATH="/opt/dsv4-venv/bin:$PATH"
export MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16

cmake -S /opt/dsv4/TransformerEngine/transformer_engine/common \
  -B /cache/te-mi35x-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/dsv4/TransformerEngine \
  -DPython_EXECUTABLE=/opt/dsv4-venv/bin/python \
  -DUSE_ROCM=ON -DUSE_FUSED_ATTN_AOTRITON=OFF \
  -DUSE_FUSED_ATTN_CK=ON -DUSE_HIPKITTENS_GEMM=ON \
  -DNVTE_BUILD_LEGACY_STATIC_NORM=ON \
  -DNVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=ON \
  -DCK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT=3 \
  -DTE_ROCM_PURELIB=/opt/dsv4-venv/lib/python3.12/site-packages \
  -DCMAKE_JOB_POOLS=compile=16 -DCMAKE_JOB_POOL_COMPILE=compile
cmake --build /cache/te-mi35x-build -j 16
cmake --install /cache/te-mi35x-build
```

This retains the supplied library's CK/HipKittens backends and absent AOTriton
backend. The build's QoLA manifest pins its private AITER source to
`2cc8c750461c035f0333711dfb3c02571161aa95` and applies the patches shipped in
the supplied QoLA source. It does not upgrade the separately installed AITER
Python package. If the development container cannot reach GitHub, run that
same QoLA `checkout` command on the host into the shared build cache, then set
`NVTE_AITER_SOURCE_DIR` to its container path before configuring CMake.

The rebuild completed with `GPU_TARGETS: gfx950`; all 24,730 files in the
supplied-source hash manifest remained unchanged. The original PyTorch binding
is a C++ binding to the rebuilt common library and was retained. The upstream
`test_multi_tensor_l2norm` matrix passed **288 cases** on MI350X. A separate
seven-step, three-parameter-tensor `FusedAdam` AdamW-mode probe matched
`torch.optim.AdamW` using PyTorch's default comparison tolerances (maximum
parameter absolute difference `2.980232238769531e-07`); moments remained finite
and nonzero. A multi-tensor gradient-scaling probe matched exactly.

The original combined test invocation could not collect TE's
`test_fused_optimizer.py` because the delivered environment lacks the test-only
`expecttest` dependency. That invocation is not counted as a pass. No dependency
was installed to bypass it; the standalone numerical probe imports the actual
optimizer directly. These checks do not establish full TE or GRPO acceptance.

## MI350X validation

Date: 2026-11-18. The delivered stack uses Torch `2.13.0+rocm10.0.0`,
Triton `3.8.0+git669b31ac`, AITER `0.1.29`, verl
`8196af485ae50e09aa42c1fc6e399f266fb166d3`, and Megatron
`97b50bdd936022ef58031c35db6d10581c181be0`.

The following targeted suites passed in the MI350X development environment,
totaling **567 tests** (including CPU dispatch and loading-scope tests):

| Tests | Count | Coverage |
| --- | ---: | --- |
| `test_native_fp8_arch`, `test_checkpoint_fp8`, `test_qnorm_vjp_on_cpu` | 7 | Native codec, real fused QKV/cache, checkpoint FP8, input/normalization gradients |
| `test_joint_attention_rocm`, `test_joint_backward_rocm` | 88 | Pinned attention forward/backward, modes, row ordering, CPU oracle |
| `test_rocm_mhc_pre`, `post`, `head`, `backward`, `batch_invariance` | 404 | mHC forward, gradients, canonical parameter reductions, batch invariance |
| `test_dispatch`, `test_mhc_contract` | 57 | Supported/unsupported architecture dispatch and contract metadata |
| `test_native_moe_arch` | 5 | Router CPU FP64 oracle, hash routing/VJP, original checkpoint W4A16 forward/VJP and frozen bytes |
| `test_mi35x_moe_loading` | 6 | CPU-only selector scope, unchanged explicit requests/other architectures, failure cleanup without fallback |

The existing FNUZ cache test now verifies the pinned upstream encoder's explicit
rejection on gfx950; OCP cases still execute. No tolerance was increased. The
capture-dependent tests requiring historical MI300X tensors are not counted as
MI350X passes.

### End-to-end status

Attempt 005 completed the native three-update run and passed the unchanged
independent verifier. See the [MI350X run report](../../examples/operators/dsv4_4l/results/native_grpo_pp2_dp1_mi350x_20261118/README.md)
for the original acceptance JSON and image/source provenance. Versions 0–3 each
passed 1,188 audit token observations with zero FP32 bit mismatches and zero
mean/max absolute difference. The final 1,000 stress requests passed 14,869
observations with unchanged actor/rollout state and no additional updates.
The complete launched job took 594.495 seconds and exited zero. This is one
completed MI350X run, not a performance comparison or independent-run
reproducibility claim.

Preserved earlier attempts remain failures. Attempt 003 failed
the initial v0 audit on the first fixed sample (3/3 FP32 logprobs differed), with
zero optimizer updates. Its automatic failure capture localizes the first
divergent module boundary to layer 0's routed MoE: embedding, mHC, attention,
attention output projection, and shared-expert tensors match byte for byte.
Those traces do not identify the precise internal AITER kernel responsible for
that sample; they localize the discrepancy to the routed-MoE boundary.
Attempt 004 adds the scoped W4A16 loading adapter and passes the initial v0
audit: 1,188 expected and compared token observations, zero bit mismatches,
zero mean/max absolute difference, finite values, and batch invariance. It then
fails during the first optimizer step in Transformer Engine's
`multi_tensor_l2norm` with `invalid device function`; no optimizer update
completed. The delivered `libtransformer_engine.so` contains gfx942-only code
objects, and its `build_info.txt` records `GPU_TARGETS: gfx942`. Rebuilding the
same supplied TE source for gfx950 and the focused checks above succeeded;
attempt 005 passed with derived image ID
`sha256:5edaed7137953d36e0370c6fc3554682bd4b2417a3305ba2cb3b9b1f5cf8edaa`.
TE reports version
`2.21.0.dev0`; the image's `te-source-sha.txt` records
`b5c00f43283086bf0751f3bf3a179f1e921bbe59`. The supplied tree has no Git metadata,
so a separate source-file SHA256 manifest also records what was actually rebuilt.

The unchanged launcher uses physical GPUs 4–5 for actor and 6–7 for rollout,
PP2/DP1 and TP1/EP1/CP1 on each side, original layers 0–2 / 3, eager
`rocm_triton/split`. The original checkpoint and delivered data are used, including
the extended audit prompts, response length 16, and 1,000 final stress repeats.
Acceptance still requires three optimizer updates, versions 0–3 audits, exact
FP32 actor/rollout parity, frozen packed state, and the original independent
`verify_grpo.py` checks.

Two earlier adaptation attempts stopped during rollout initialization: first at
the mHC gfx942 guard and then at the router-projection gfx942 guard. Neither made
an optimizer update. Those failures are preserved separately from the subsequent
attempt after their targeted tests passed.

### Additional checks and limitations

Running the CPU workflow's test list inside the delivered ROCm image is not the
same environment as its CPU-only CI job. That extra run produced 250 passes and
four failures: the GPU isolation preload denies the allocation test's read of
an unassigned KFD node, and three standalone layer-end tests reach the unchanged
gfx942-only `TritonBatchInvariantLinearOp`. The native GRPO head uses a different
implementation. These failures are not counted as passes or bypassed.

The broader test attempt with all GPUs hidden aborted when the installed vLLM
import queried device properties. It does not establish a CPU CI result.

Ruff passes on the other changed Python files. `triton_mhc.py` retains the same
four pre-existing diagnostics as the base revision (three E501 and one B905);
unrelated formatting and arithmetic code were left unchanged.
