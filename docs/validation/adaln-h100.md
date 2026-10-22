# AdaLN H100 Validation

This runbook validates [AdaLN modulation](../operators/adaln-modulation.md) on
H100/SM90. Run commands from the repository root in an environment with matching
PyTorch/CUDA, a CUDA toolkit and NVIDIA Compute Sanitizer.

The commands below are validation instructions; the recorded run is summarized
at the end. Retain the source revision,
working-tree status, environment, build log, test output, sanitizer logs and benchmark
JSON together. A skipped backend is not native CUDA acceptance.

## Workloads

| Image resolution | Image sequence length S | Hidden size H |
| --- | --- | --- |
| 1024 x 1024 | 4096 | 3072 |
| 1328 x 1328 | 6889 | 3072 |
| 1664 x 928 | 6032 | 3072 |

Sequence lengths are derived using VAE scale 8 and 2x2 latent packing.
They are model workloads, not tensor shapes explicitly specified by issue #386.
Validate both FP32 and BF16 and both CUDA and Triton.

## Record Environment and Build

```bash
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'

RL_KERNEL_REQUIRE_EXT=1 KERNEL_ALIGN_USE_FAST_MATH=0 TORCH_CUDA_ARCH_LIST=9.0 \
  .venv/bin/python setup.py build_ext --inplace
```

Build the current sources and start new test processes so they load the new
extension. Save the build output and verify the loaded extension identity;
passing tests against an older loaded binary do not validate this build.

## Forward, Backward and Dispatch

```bash
RL_KERNEL_REQUIRE_EXT=1 RLK_ADALN_REAL_SHAPES=1 \
  .venv/bin/python -m pytest tests/ops/norm/test_adaln_modulation.py \
  tests/validation/operators/test_adaln_gtest.py \
  tests/backends/test_adaln_shared_bounds.py tests/runtime/test_adaln_dispatch.py -q -rs

for candidate in cuda triton; do
  for dtype in bf16 fp32; do
    .venv/bin/python tools/validation/operators/check_operator.py --op adaln_modulation \
      --candidate "$candidate" --device cuda --dtype "$dtype" --batch 2 --seq 3 \
      --normalized-dim 3072 --check-grad --json
  done
done
```

The opt-in full-image tests compare both GPU backends/dtypes against independent
CPU FP32 forward and backward results. Inspect pass/skip output for each backend.
The CLI separately checks accuracy; it does not establish a WS1 system gate.

To isolate the GPU dispatch cases:

```bash
.venv/bin/python -m pytest tests/runtime/test_adaln_dispatch.py -k gpu -q -rs
```

## Memory and Race Checks

```bash
RL_KERNEL_REQUIRE_EXT=1 compute-sanitizer --error-exitcode=1 --tool memcheck \
  .venv/bin/python -m pytest tests/ops/norm/test_adaln_modulation.py \
  -k test_cuda_adaln_bit_equality_harness -q

RL_KERNEL_REQUIRE_EXT=1 compute-sanitizer --error-exitcode=1 --tool racecheck \
  .venv/bin/python -m pytest tests/ops/norm/test_adaln_modulation.py \
  -k test_cuda_adaln_bit_equality_harness -q
```

## Performance and Identity Records

```bash
for dtype in bf16 fp32; do
  .venv/bin/python benchmarks/operators/norm/benchmark_adaln_modulation.py \
    --real --dtype "$dtype" --seed 386 --warmup 10 --repeat 50 --json
done
```

JSON records raw event samples, medians, absolute peak allocated memory,
allocation before sampling, GPU/driver/runtime versions, working-tree status,
source hashes and the loaded CUDA extension file hash when available.

The binary hash identifies the loaded file; it does not prove that file was built
from the recorded source tree. Missing observations are null, and unavailable
CUDA/Triton backends are explicitly skipped.

Backend observation precedes timing. CUDA events measure the bare operator;
forward+backward includes newly allocated upstream `ones_like` tensors.
The ordinary LayerNorm/affine baseline has no fixed-order replay guarantee.

## Repository Integration

After focused acceptance, run the repository suite in an appropriately provisioned
environment. It includes hardware and integration tests beyond AdaLN:

```bash
.venv/bin/python -m pytest tests -q -rs
```

Keep failures or collection errors in the validation record; a focused AdaLN pass
does not imply the repository suite passed.

## Recorded H100 Acceptance (2026-10-22)

The locally retained `adaln-h100-evidence-20261021-01.tar.gz` records source
`69bd47f72a24d1177580d9b11b6efdfeb6c6bab5` against comparison base
`16b81175874ecf11d5ff05c00de3b69e9457591b` on one H100 PCIe (SM90), with
Python 3.11.17, PyTorch 2.12.1+cu126, Triton 3.7.1, CUDA toolkit 12.6.85 and
driver 595.91.07. These results are focused operator acceptance, not full WS1,
model, trajectory, repository-suite or cross-platform qualification.

| Stage | Recorded result |
| --- | --- |
| Correctness | 413 passed; no skips or xfails |
| Racecheck subset | 211 passed; 0 hazards, errors or warnings |
| Memcheck subset | 225 passed; 0 errors |
| Benchmark | 54 points completed; 36 samples per point |

The run retained a forced extension build, source-file hashes, SM90 binary
targets and per-process loaded-binary hash checks. The archived extension SHA256
matches `identity.json` and `status.json`:

```text
deee4d78b05f8ef7fff5f288141ac831b02fe24407f97fccdde474953117743a
```

The runtime source profile equals the profile retained by image build
[39741435857](https://github.com/hongleng/RL-Kernel/actions/runs/39741435857).
All repository tracked file hashes match that source revision except
`requirements.txt`, deliberately replaced by the image's pinned dependency list;
its hash matches the build artifact. Image infrastructure remains separate from
this operator PR.

Evidence erratum: the outer `image-reference.txt` and `image-build-url.txt`
incorrectly name the older build 39565761133 (digest prefix `310f692`). The newer
build 39741435857 publishes digest prefix `682fae9`. Profile agreement supports
the source/environment association but does not independently prove which image
digest the host deployed. Preserve the original archive and this correction.
The inner archive checksum was verified; the outer checksum sidecar was not
retrieved and is not claimed as verified.

The 54-point measurement uses the separate image acceptance harness: FP32/BF16,
the three real sequence lengths, CUDA/Triton/PyTorch, and forward/backward-only/
forward+backward with preallocated random upstream gradients. It is distinct from
the repository benchmark command above. Sample coefficients of variation range
from about 0.8% to 50.9%; these timings do not establish stable speedup ratios.

Subsequent documentation-only edits do not change the accepted executable source.
Any arithmetic, dispatch, build or test change needs an explicit reassessment and
affected H100 checks rerun; these results must not be relabelled as a fresh pass.
