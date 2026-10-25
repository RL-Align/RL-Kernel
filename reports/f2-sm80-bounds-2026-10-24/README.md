# F2 CUDA forward bounds-check deduplication

Result: **80 targeted tests passed, no skips**, using the standard native extension
on A100 SM80. All 24 fixed-data result tensors remain bitwise identical to the
previous baseline, and serialized input/output SHA-256 values also match.

## Change

The Python forward wrapper still validates shapes, dtypes, devices and layout.
For CUDA only it delegates position-range checking to the checked native forward
entry point, instead of doing the same GPU-to-host check in both layers. Triton
keeps Python bounds validation. Native forward/backward validation is unchanged,
and no kernel arithmetic or compilation flags changed.

The wrapper translates only the specific native out-of-range RuntimeError to its
existing ValueError; other errors propagate. No validation results are cached.
Mutated positions are checked afresh on every call. CUDA graph capture remains
unsupported because native bounds validation still reads min/max on the host.

## Validation

`bounds-pytest.log`: 80 passed in 5.76 seconds, comprising the previous 73 targeted
checks and 7 new cases covering negative/upper-bound positions after a successful
call, empty tables with nonempty input, and unrelated native-error propagation.
Local CPU reference tests also passed (14); formatting and diff checks passed.

The existing `probe.py` from `../f2-sm80-integration-2026-10-24/` ran immediately
before and after the Python change on the same instance, using identical serialized
inputs/tables and comparing against the archived outputs. All comparisons passed.
`bounds-before/` and `bounds-after/` preserve full timing samples and hashes.

Environment: A100 80GB PCIe, CUDA 13.0, PyTorch 2.14.1+cu130, Triton 3.8.0.
Remote workspace: `/workspace/rlk-f2-integration-e3gNYR1a`.
Interpreter: `/workspace/rlk-f2-sm80-DpNQapig/test-venv/bin/python`.

```bash
RL_KERNEL_F2_TEST_JIT=0 python -m pytest \
  tests/ops/rope/test_qknorm_scaled_rope.py \
  tests/ops/rope/test_qknorm_scaled_rope_forward.py \
  tests/build/test_build_platform_collectives.py -q
python probe.py --inputs artifacts-jit/inputs.pt \
  --compare artifacts-jit/outputs.pt --out bounds-after
```

## Same-instance call timings

CUDA local forward median synchronized wall time, microseconds per call:

| Tokens | Before | After | Reduction |
| ---: | ---: | ---: | ---: |
| 1 | 150.5 | 80.0 | 46.8% |
| 128 | 147.0 | 80.5 | 45.2% |
| 2048 | 203.8 | 162.8 | 20.1% |

Unchanged controls remained close: global CUDA forward at T=1 was 24.9 -> 24.4 us;
local Triton at T=1 was 161.8 -> 160.3 us. Each measurement has five warmup calls
then seven samples of ten calls, synchronizing at sample boundaries. Clocks are
not pinned. These are call-level results including host validation/dispatch and
allocation, not pure-kernel speedups or model-training throughput claims.

## Remaining work

Further optimize kernel geometry for larger inputs, retain the explicit reduction
and rounding rules, and recheck fixed-data parity. Native min/max still require
host synchronization; removing that needs a separately designed safe validation
interface. Cross-GPU parity, H100/ROCm support, table provenance and final numerical
acceptance remain open.
