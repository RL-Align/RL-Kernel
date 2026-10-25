# F2 table identity validation on A100

The public Python F2 entry now requires shared RoPE tensors registered by the
canonical CUDA builder or the trusted-identity loader. Computation kernels and
VJP v2 arithmetic are unchanged. Existing cos/sin arguments remain supported,
but arbitrary raw tensors are intentionally rejected for local layers.

## Evidence

Standard extension built with `RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=8.0
MAX_JOBS=2 python setup.py build_ext --inplace`. No fast math enabled.

Complete targeted suite: **120 passed, no skips, 68.96 seconds**:

```bash
RL_KERNEL_F2_TEST_JIT=0 python -m pytest \
  tests/ops/rope/test_rope_table_identity.py \
  tests/ops/rope/test_qknorm_scaled_rope.py \
  tests/ops/rope/test_qknorm_scaled_rope_forward.py \
  tests/build/test_build_platform_collectives.py -q
```

Includes original forward/backward parity, fixed-tree/reference and independent
FP64 checks, batch/chunk invariance and bounds tests. New coverage checks trusted
load/copy isolation, invalid metadata/content, pair mixing, ordinary mutation,
inference-mode construction, equality to the prior reference table builder and
unchanged outputs after CPU artifact round trip. A patched digest that raises
confirms public dispatch does not rehash contents.

Environment: A100 80GB PCIe SM80, Python 3.12.15, PyTorch 2.14.1+cu130,
Triton 3.8.0, nvcc 13.0.88, driver 590.48.01. Remote workspace
`/workspace/rlk-f2-p2`, isolated interpreter `/workspace/rlk-f2-p2-venv/bin/python`.

Logs: `rlk-f2-p2-build.log`, `rlk-f2-p2-tests.log`; identity and timing:
`rlk-f2-p2-identity.json`. Source digests: `source-sha256.json`.

Full P=131072 table SHA-256 (metadata + cos/sin bits):
`c0df1dfb156f81993c6fcfebdd50eca72ce31099436a565dc9407f8d562ac2c3`.
This is the tested producer's identity, not a universal cross-platform hash.

Single initialization measurement, after CUDA context warmup: 0.874 seconds
including table construction and hashing. Mean host-only identity validation over
10,000 calls: 2.05 microseconds. These are diagnostic samples, not a complete
kernel benchmark or proof of no performance regression. Hash serialization uses
bounded int16 chunks, avoiding slow byte-by-byte Storage iteration; no NumPy
requirement is introduced. Full-suite time includes repeated full-table generation.

## Scope and limitations

Trusted expected identities must be supplied separately by runtime integration;
an arbitrary digest accompanying an untrusted payload is not mathematical proof.
The producer algorithm is explicit, but independent GPU/software builds are not
guaranteed bitwise identical: distribute one canonical table artifact and identity.

Ordinary PyTorch version counters detect mutations without a per-call GPU sync.
Concurrent mutation, `.data` or external-pointer writes bypassing those counters
are unsupported. Native/private backend functions remain internal computation
interfaces without standalone provenance checks. This is not a Python security
boundary. Cross-GPU lifecycle tests remain outstanding; no H100/SM89 results are
claimed for this revision. No sanitizer rerun was needed for this Python-only
change and none is claimed. No PR has been submitted.
