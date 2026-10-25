# F2 SM80 ordered shared-memory/shuffle reduction

The fixed-tree hybrid reduction is now integrated for **SM80 only**. Other targets
retain the original shared-memory implementation. The mapping preserves all 127
ordered FP32 addition nodes and the seven stages 64/32/16/8/4/2/1. Only the last
five stages exchange register values through warp-0 shuffle instructions.

## Evidence

- Structural node-by-node proof: `../f2-reduction-mapping-2026-10-24/`.
- Independent candidate: **65,536 signed FP32 rows** match the explicit fixed-tree
  reference bitwise, including a cancellation witness whose exact tree result is 2.
  Inputs have independently sampled powers from 2^-60 to 2^60 per element.
- CUDA baseline versus candidate: local/global Q/K, saved FP32 rstd and gradients
  match bitwise at T=1/128/257/2048/8192, Q32/K2/D128. T=257 uses archived stress data.
- Candidate under the actual Python/autograd wrapper: **80 tests passed** in 3.97 s.
- Standard extension rebuilt after integration: **80 tests passed** in 5.83 s.
- Integrated fixed-data output file SHA-256 remains
  `262a18bde3fc01848225b22b0945074b07e7fe163243498fde8f5b4e405be88a`.
- Compute Sanitizer Racecheck: **0 hazards, 0 errors, 0 warnings**.
- Compute Sanitizer Synccheck: **0 errors**.
  Both tools ran before integration and again with the rebuilt native extension.
  The sanitizer workload covers 128 signed rows and T=129 local/global forward and
  backward for the native and isolated implementations; it is not exhaustive.

Environment: A100 80GB PCIe, CUDA 13.0, PyTorch 2.14.1+cu130, Triton 3.8.0.
Remote workspace `/workspace/rlk-f2-integration-e3gNYR1a`, Python interpreter
`/workspace/rlk-f2-sm80-DpNQapig/test-venv/bin/python`.
No fast math was enabled. Candidate compilation explicitly disables FMA contraction
and FTZ; the production path uses existing explicit FP32 rounding intrinsics.

## Machine-code audit

`shuffle-experiment/candidate-5.sass` contains the standalone reduction. Shared
loads use byte offsets 0x100 and 0x80 for strides 64 and 32. The main warp path
then uses SHFL.DOWN offsets 0x10,0x8,0x4,0x2,0x1, with FADD guarded by the matching
lane predicates. The current lane value is the left operand, the shuffled value
the right operand. There are seven static FADD instructions in this probe and no
FP32 FFMA. Compiler-generated helper code contains an additional dynamic-offset
SHFL; six static SHFL instructions do not imply an extra addition stage. The
HFMA2 instructions in this helper materialize constants, not reduction operands.

The full operator's FADD counts are unchanged. Its BAR.SYNC counts are:

| Specialization | Original | Candidate and rebuilt native |
| --- | ---: | ---: |
| Global forward | 9 | 5 |
| Local forward | 10 | 6 |
| Global backward | 8 | 4 |
| Local backward | 9 | 5 |

These include non-reduction barriers. The reduction alone goes from eight to four.
The root publication barrier remains, as do forward rstd and local RoPE barriers.
Full baseline/candidate/native SASS and instruction-count JSON files are archived.
Source-level proof, instruction inspection, bitwise tests and sanitizers are
complementary evidence; this is not a formal verification of all machine behavior.

## Performance

Same-process baseline/candidate native-entry call timings, median microseconds:

| Tokens | Layer | Forward original → candidate | Backward original → candidate |
| ---: | --- | ---: | ---: |
| 1 | Global | 12.4 → 12.6 | 9.9 → 10.4 |
| 1 | Local | 52.2 → 52.8 | 50.8 → 51.7 |
| 2048 | Global | 83.2 → 73.7 | 95.5 → 88.3 |
| 2048 | Local | 158.5 → 150.3 | 172.3 → 164.8 |
| 8192 | Global | 313.3 → 276.1 | 362.3 → 332.3 |
| 8192 | Local | 495.8 → 467.2 | 553.8 → 521.0 |

Large-input calls improve roughly 4–12%; tiny-input calls show a small regression
(up to about 0.9 us here), so this is not a universal speedup. These timings include
native validation, allocation and launch overhead, without the outer Python wrapper.
They use five warmups and seven samples of ten synchronized calls. Clocks are not
pinned, baseline is measured first, and this is not a pure-kernel or model benchmark.
`shuffle-results.json` holds raw samples; `shuffle-native/results.json` separately
records the final public-wrapper timings and fixed-data comparisons.

## Reproduction and artifacts

`baseline.cu` preserves the pre-change source; `candidate.cu` adds the hybrid
reduction plus a signed-reduction probe, without replacing the production extension.
`experiment.py` builds it under a distinct module name and temporarily patches the
wrapper only inside its own test process. It imports `probe.py` from the earlier
integration experiment and uses its archived tensor inputs.

From the remote workspace, before production replacement, run:

```bash
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 RL_KERNEL_F2_TEST_JIT=0 \
  python shuffle-experiment/experiment.py
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=8.0 compute-sanitizer \
  --tool racecheck --error-exitcode 99 python shuffle-experiment/experiment.py --sanitize
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=8.0 compute-sanitizer \
  --tool synccheck --error-exitcode 99 python shuffle-experiment/experiment.py --sanitize
```

After production integration, the same experiment's native baseline is the new
implementation. To reproduce the original A/B comparison, build the saved
`baseline.cu` in a separate checkout; do not mistake a new-native/candidate run for
the original comparison. `shuffle-experiment/` preserves all executed logs and SASS.

Cross-GPU parity, nonfinite/NaN payload behavior, other architectures and throughput
under real training workloads remain unqualified. Existing final-error-threshold
and table-provenance work is unaffected.
