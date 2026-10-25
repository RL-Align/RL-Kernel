# F2 reciprocal square root experiment

Run date: 2026-10-24, America/Los_Angeles.

On the tested RTX 4090 environment, CUDA `rsqrtf` and Triton `libdevice.rsqrt`
matched `torch.pow(v, -0.5)` bit for bit for all 1,114,119 sampled FP32 inputs.
Direct CUDA `powf` and Triton `libdevice.pow` did not match each other or PyTorch.
This supports rsqrt as the next CUDA/Triton implementation candidate while preserving
`torch.pow` in the model reference. It is not exhaustive equivalence, upstream approval,
or validation of the full fused operator.

## Environment and reproduction

- NVIDIA GeForce RTX 4090, SM89; driver 595.71.05.
- Python 3.12.15; PyTorch 2.14.1+cu132; Triton 3.8.0.
- CUDA Toolkit 13.2, nvcc V13.2.86.
- Remote artifacts: `/workspace/rlk-f2-rsqrt-RgdAfaDx`.
- Interpreter: `/venv/main/bin/python`.

Run `python probe.py` from this directory on the same GPU/software environment.
The script builds the included CUDA source and saves results and PTX. It uses
ctypes to launch on PyTorch's current CUDA stream, with explicit synchronization
before comparisons. No package installation was needed on the remote instance.

CUDA flags: `-O3 -std=c++17 -arch=sm_89 --fmad=false --ftz=false
--prec-div=true --prec-sqrt=true`; no `--use_fast_math`.
Triton options: `enable_fp_fusion=False`, `enable_reflect_ftz=False`.
TF32 is disabled in the experiment process (there are no GEMMs in the probe).
Triton 3.8 does not recognize `enable_fast_math`; the initial run stopped at option
validation and the corrected run used the supported options above.

## Inputs

Seed 501; all primary inputs are finite positive FP32:

1. 1,048,576 log-distributed normal values, approximately `2^-120` through `2^120`.
   This is a diagnostic sweep, not all valid F2 variances.
2. Seven boundary values: FP32 epsilon and its neighboring values, plus 1, 2, 4, 16.
3. 65,536 BF16 vectors of width 128 with scales spanning `2^-20` through `2^20`,
   including zero, unit and tiny rows. Squaring, the proposed seven-stage FP32 tree,
   mean and epsilon are computed once and shared by every candidate. This isolates
   reciprocal-square-root differences from reduction differences.

FP32 comparison uses integer views, not approximate equality. FP64 relative errors
use the same FP32 variance promoted to FP64, isolating this primitive's error.
BF16 comparisons cover 8,388,608 elements after norm and after Q scaling. These
downstream operations use common PyTorch code, not fused candidate kernels.

## Results

FP32 mismatches against `torch.pow(v, -0.5)`:

| Candidate | Broad sweep / 1,048,576 | Boundaries / 7 | F2 variances / 65,536 |
| --- | ---: | ---: | ---: |
| torch.rsqrt | 0 | 0 | 0 |
| CUDA rsqrtf | 0 | 0 | 0 |
| Triton libdevice.rsqrt | 0 | 0 | 0 |
| Triton tl.rsqrt | 0 | 0 | 0 |
| CUDA powf | 248,749 | 2 | 16,981 |
| Triton libdevice.pow | 444,561 | 1 | 27,306 |
| CUDA precise sqrt/div diagnostic | 355,870 | 2 | 22,615 |
| Triton precise sqrt/div diagnostic | 355,870 | 2 | 22,615 |

On F2-derived variances:

| Candidate | BF16 norm mismatches / 8,388,608 | BF16 Q scale mismatches / 8,388,608 |
| --- | ---: | ---: |
| CUDA rsqrtf / Triton rsqrt variants | 0 | 0 |
| CUDA powf | 25 | 25 |
| Triton libdevice.pow | 53 | 47 |
| Precise sqrt/div diagnostic, either backend | 42 | 40 |

CUDA rsqrtf and both Triton rsqrt variants match each other on all samples.
CUDA powf and Triton pow disagree on 413,262 broad-sweep values and 23,337 F2
variances. Thus choosing similarly named pow functions does not establish parity.

Maximum relative FP64 error for the matching rsqrt group was approximately
`1.19e-7` across the sampled inputs. Some other candidates have smaller maximum
error on these samples but fail the chosen PyTorch parity target; greater accuracy
does not imply identical rounding. See `results.json` for ULP differences and
concrete counterexamples.

## Generated code and interpretation

`cuda.ptx` contains `rsqrt.approx.f32` for the CUDA rsqrt kernel; the two Triton
rsqrt PTX files use the same instruction. The `approx` instruction name does not
mean the experiment enabled the fast-math compiler flag. It describes the selected
primitive, whose rounding is measured against PyTorch here.

The pow-library PTX contains explicit internal FMA instructions (and Triton pow
contains an internal `ex2.approx.ftz` instruction) even with contraction/FTZ options
disabled. Compiler options do not remove every explicitly implemented library
instruction. These pow candidates are diagnostic and are not selected for strict
implementation. No claim of globally FMA-free library code is made.

The sqrt/div path is a diagnostic control, not an approved replacement for
foundation's `pow_neg_half` expression. No production reference/kernel was changed.

## Limits and next action

This does not validate backward, RoPE, a compiled CUDA/Triton fixed-tree reduction,
full F2 batch/chunk invariance, A100/H100/ROCm, all FP32 inputs, subnormal variance,
NaN/infinity policy or performance. No speedup claim is supported.

Next, use CUDA rsqrtf and Triton libdevice.rsqrt as explicit candidates in fused
forward, retain the declared FP32 reduction and BF16 cast boundaries, and rerun
full-operator correctness/parity tests. Retain the software/compiler provenance and
revalidate on every supported target before broadening the guarantee.
