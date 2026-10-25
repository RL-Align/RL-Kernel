# Muse Glimmer F2 QK Norm Scale and RoPE Contract

Status: Draft with eager references and CUDA/Triton first-order autograd candidates.
Full platform qualification and final acceptance tolerances remain pending.

This contract scopes `qknorm_scaled_rope` (F2) and separates upstream model semantics
from proposed strict execution rules. A proposal below is not an upstream-approved guarantee.

## Sources and precedence

- [RFC #501](https://github.com/RL-Align/RL-Kernel/issues/501): F2 scope, strict numerical
  requirements, platforms and acceptance criteria.
- [Foundation PR #558](https://github.com/RL-Align/RL-Kernel/pull/558), inspected at
  `e1f86ecf340eeb58cb934f08468fc561590f84b0` (open at inspection):
  `OFFICIAL_FINGERPRINT` and `OFFICIAL_FROZEN_SEMANTICS` in
  `rl_engine/models/muse_glimmer/workload.py`, mirrored in the manifest.
- [Pinned Transformers implementation](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py):
  `MuseGlimmerRMSNorm`, `MuseGlimmerTextRotaryEmbedding`, `apply_rotary_pos_emb`
  and the attention forward method. Source file SHA-256:
  `e69eb21d6efd815a48499d4d0e4180afda2f443beb441a734d50efc2cf83f975`.
- [RoPE PR #228](https://github.com/RL-Align/RL-Kernel/pull/228): implementation reuse,
  not authority for Muse Glimmer precision or cast points.

Foundation fields govern the model semantics. The source-derived eager behavior below
fills in cast boundaries absent from those fields. Proposed strict rules must preserve
the model semantics or explicitly declare deviations. Reconcile this draft when the
foundation revision changes; record PyTorch, CUDA and Triton versions in validation reports.

## Scope

F2 consumes projected Q and K and returns normalized, scaled and optionally rotated Q/K.
It covers operator-trace kinds `qk_norm` and `rope_local`, including Q scaling.
Projection GEMMs, attention, cache writes/eviction and distributed gradient reductions
belong to other work items. F2 must provide forward and backward.

## Foundation frozen semantics

| Property | Requirement |
| --- | --- |
| Model dtype | BF16 |
| Full-model heads | 32 Q heads, 2 K heads; head dimension 128 |
| Norm | Weightless RMSNorm independently over each head, for both Q and K |
| Norm computation | FP32; epsilon `1e-5`; source expression `pow_neg_half` |
| Q scale | Multiply Q only by `3.87`, after the norm output cast |
| Order | Projection, Q/K norm, Q scale, local RoPE, cache write |
| Layer schedule | Three local layers then one global, repeated 13 times |
| RoPE | Local layers only; default type; theta `500000`; all 128 channels |
| Rotation layout | Rotate-half: channel `i` paired with `i + 64` |
| Tables | Compute cos/sin in FP32, cast to BF16 |
| Global layers | No positional encoding; norm and Q scaling remain active |
| Attention scale | Separate `head_dim**-0.5`; do not move Q scaling into attention |

## Source-derived forward cast boundaries

The following describes the pinned model's ordinary BF16 eager path, without a
replacement hub kernel or compiler fusion. `B` means BF16 round-to-nearest-even;
`F` means conversion to FP32. Individual FP32 operations also round.

For each head vector `x`:

```text
x32 = F(x)
m32 = mean(x32 ** 2) + FP32(1e-5)
r32 = pow(m32, -0.5)
n = B(x32 * r32)
q_scaled = B(F(n_q) * FP32(3.87))
k_normalized = n_k
```

The Python scalar multiplication is modeled using FP32 opmath followed by a BF16
output cast; do not pre-round the scalar `3.87` to BF16. Confirm this behavior in
the pinned validation environment.

For a local layer, let `z` be the scaled Q or normalized K. Select BF16 cos/sin by
the token's absolute position and repeat the half-width table across both halves:

```text
rotate_half(z) = concatenate(-z[64:128], z[0:64])
a = B(F(z) * F(cos))
b = B(F(rotate_half(z)) * F(sin))
output = B(F(a) + F(b))
```

Preserve both product casts inside a fused kernel. FP32 products followed by only
one final BF16 cast do not reproduce this eager behavior. Global layers return
`q_scaled` and `k_normalized` directly.

## Proposed interface and table ownership

- Contiguous BF16 Q `[T, Hq, 128]` and K `[T, Hk, 128]` on the same device.
- Local positions are int64 `[T]`, indexing a shared table by absolute position.
- Outputs preserve input shapes and dtype; inputs are not mutated.
- Full-model validation uses `(Hq, Hk) = (32, 2)`. Explicitly enumerate supported
  shard geometries before claiming TP integration; per-head arithmetic must not
  depend on head count.
- A single CUDA-built table supplies both CUDA and Triton, as required by RFC #501.
  Proposed storage is BF16 `[P, 64]` per cosine/sine table. Record theta, dimension,
  position coverage, construction algorithm and content hash. Specify the exact
  frequency/power/trigonometric algorithm before treating table generation as frozen.
- Local positions must satisfy `0 <= position < P` and the pinned context limit.
  Chunking must not reset positions. Global execution does not read the table.
- Empty input returns empty outputs without a zero-block launch.
- Reject unsupported dtype, shape, layout, device, layer mode or table identity;
  do not silently dispatch to a non-strict fallback. Document which identity checks
  are performed by the wrapper versus the runtime integration.

## Proposed strict arithmetic

Use FP32 accumulation independent of batch size, row placement and chunk size.
For the 128-element norm sum, propose this explicit tree:

```text
a[i] = FP32(x32[i] * x32[i])
for stride in [64, 32, 16, 8, 4, 2, 1]:
    a[i] = FP32(a[i] + a[i + stride])  for i < stride
mean = FP32(a[0] * FP32(1/128))
```

CUDA and Triton must implement the same logical tree; do not assume an unconstrained
`tl.sum` reproduces it. Disable fast math and prevent contraction/reassociation across
declared operations and cast boundaries. Avoid cross-block atomics for F2 reductions.

The executable candidate is `fixed_sum_128` (FP32 half-split tree v1), used by
`qknorm_scaled_rope_fixed_tree_reference`. `fixed_dot_128` separately rounds FP32
products before applying this tree and is used in the declared backward reference.
The original model reference remains unchanged. The fixed-tree forward reference
still uses `torch.pow` and ordinary autograd; its automatically generated gradients
are distinct from the explicitly grouped VJP below.

For 128 channels, the tree has seven addition stages and 127 additions per row.
For finite nonnegative summands with no underflow/overflow, its summation error is
bounded by `gamma_7 * sum(values)`, where `gamma_7 = 7u/(1-7u)` and `u = 2^-24`.
This bound excludes square-product error and later operations. For signed backward
dot products use an absolute bound based on `sum(abs(products))`; cancellation can
make relative error large. A balanced tree improves the worst-case depth bound over
a sequential fold but is not guaranteed to be more accurate on every input.

Tests include an order-sensitive cancellation witness, a positive-sum error bound,
and BF16 bit-pattern equality under token partitioning and permutation. Backend
parity results are limited to the reported environments; compiler-IR qualification
and performance work remain outstanding.

The forward candidates select CUDA `rsqrtf` and Triton `libdevice.rsqrt`, while the
model reference keeps `torch.pow(m32, -0.5)`. The [SM89 experiment](../../reports/f2-rsqrt-2026-10-24/README.md)
found zero FP32 mismatches against PyTorch on 1,114,119 sampled inputs with
PyTorch 2.14.1+cu132 and Triton 3.8.0. Direct CUDA/Triton pow candidates disagreed.
This is measured compatibility on that environment, not equivalence for all inputs
or architectures. Validate again on other supported targets. The earlier CUDA
sketch's `powf` placeholder is not used by the forward implementation.

## Backward boundary

Tables and positions are constants; only Q and K receive gradients. The mathematical
chain is local RoPE transpose, Q-only scale, then RMSNorm backward. For the gradient
`g` reaching the norm output, its real-arithmetic input derivative is:

```text
dx = r * g - x * r**3 * sum(g * x) / 128
```

This identity does not freeze its floating-point evaluation or make it bitwise equal
to eager autograd. BF16 table quantization also means transpose must not be justified
by assuming an exactly orthogonal stored rotation.

The implemented VJP v2 candidate uses the following explicit grouping. `B` denotes
BF16 round-to-nearest-even and `F` conversion to FP32. The upstream gradient `dy`
is BF16; each displayed FP32 arithmetic operation rounds separately.

```text
local only:
    a[i] = B(F(dy[i]) * F(cos[i % 64]))
    b[i] = B(F(dy[(i + 64) % 128]) * F(sin[i % 64]))
    g[i] = F(B(F(a[i]) + (F(b[i]) if i < 64 else -F(b[i]))))
global:
    g = F(dy)
Q only:
    g = F(B(g * FP32(3.87)))

r = saved_forward_rstd              # FP32; never recomputed in backward
u = FP32(F(x) * r)                  # retain FP32; do not cast u to BF16
dot = fixed_sum_128(FP32(g * u))
mean_dot = FP32(dot * FP32(1/128))
correction = FP32(u * mean_dot)
dx = B(FP32(r * FP32(g - correction)))
```

V2 avoids explicitly computing `r^3`, which underflowed for finite large inputs
in v1 and erased the correction term. For example, K=2^55 and a gradient of one
on the first 64 channels and zero on the rest produced normalized L2 error 1.0
against FP64. The corrected K gradient is +2^-56 / -2^-56 across the two halves.
The seven-stage ordered sum is unchanged, but its leaves and surrounding FP32
grouping change: v2 intentionally does not preserve v1 gradient bits. Historical
v1 artifacts remain evidence for that version only; forward/rstd bits remain valid.
This does not establish correctness for arbitrary overflow/nonfinite inputs.

The per-head dot uses the same seven-stage tree as forward, without cross-head or
cross-token atomics. This grouping is an F2 proposal, not a foundation-frozen
backward program or a bitwise match to eager autograd scheduling. Independent FP64
autograd checks the mathematical derivative; the explicit VJP reference checks
strict candidate arithmetic. No norm-weight gradient is needed.

## Validation and remaining decisions

Keep three distinct references/checks:

1. Model reference: pinned operation order and BF16 cast boundaries, using ordinary
   PyTorch/autograd. This does not prescribe PyTorch's internal reduction tree.
2. Strict reference: executable declared arithmetic, shared as the CUDA/Triton target.
3. Independent FP64 reference: mathematical correctness from the same quantized inputs,
   promoted to FP64, without BF16 intermediate casts. Declare whether a diagnostic uses
   shared quantized tables or independently generated FP64 tables; do not conflate them.

Require CUDA/Triton strict parity and batch invariance for outputs and gradients.
Check forward chunk invariance with identical absolute positions, local/global behavior,
long positions, cast-sensitive cases, zero/near-zero inputs, and unsupported-input errors.
Use `torch.equal` as specified by the RFC; if literal byte equality is claimed, add a
bit-pattern check because numerical equality alone does not distinguish signed zero.
Correctness uses FP64 error bounds and normalized gradient error, not strict parity
tolerances. Define the error metric, zero-reference handling and justified thresholds
before acceptance; do not loosen tolerances to conceal CUDA/Triton mismatches.

Each kernel PR supplies CUDA and Triton; Triton must support A100. ROCm uses the Triton
path, with cross-vendor equality measured in the RFC's later validation rounds.

Before declaring the numerical contract complete:

- [x] Select and test reciprocal-square-root candidates on SM89; other targets remain pending.
- [ ] Confirm the proposed reduction tree and record compiler arithmetic controls.
- [x] Specify candidate table construction algorithm and hash serialization (below).
- [x] Specify candidate backward cast points, accumulation order and saved intermediates.
- [ ] Pin supported geometries, software versions and correctness error thresholds.
- [ ] Record any deviation from foundation/source semantics for review.

Reported GPU correctness/parity results apply only to the tested cases and environments.
This draft does not claim universal equality, performance results or upstream approval.

## Reference implementation

`rl_engine/reference/rope/qknorm_scaled_rope.py` provides the BF16 model reference,
an independent FP64 formula and a reference table builder. The builder supports CPU
for testing; it is not the production table provenance implementation. `fp64_math`
accepts FP64 leaves for gradient comparison without BF16 leaf-gradient accumulation.

Run the reference tests with PyTorch and pytest installed:

```bash
python -m pytest tests/ops/rope/test_qknorm_scaled_rope.py -q
```

The relative-L2 bounds in these tests (0.02 forward, 0.03 backward) are provisional
smoke-test bounds on deterministic fixtures, not accepted kernel error thresholds.
They include independently generated FP64 versus BF16 table differences.

## Forward kernel candidates

`rl_engine.ops.rope.qknorm_scaled_rope.qknorm_scaled_rope_forward` requires an
explicit `backend="cuda"` or `backend="triton"`. It returns a named tuple with
BF16 `query`, `key` and FP32 `query_rstd`, `key_rstd` of shape `[T, H]`.
Both backends fuse normalization, Q scaling and optional rotation, consuming the
same externally supplied table. CUDA uses one block of 128 threads per head;
Triton uses one program per head with explicit gather stages for the same tree.

The raw forward entry point rejects gradient-enabled Q/K requiring gradients;
training uses the differentiable entry point below. Neither entry substitutes
the reference or another backend. The CUDA extension checks direct-call geometry and position bounds too.
Bounds validation synchronizes with the host, so this entry point does not yet
support CUDA graph capture. Public Python dispatch now requires a registered table pair; see the table identity contract below.

The CUDA source is part of normal NVIDIA extension builds, without an SM90 gate.
Builds using the project's fast-math option mark F2 unavailable rather than silently
changing its arithmetic. FP32 additions and multiplications use explicit rounding
intrinsics. Triton disables FP fusion and NVIDIA libdevice FTZ. ROCm selects the
shared Triton implementation, but requires separate numerical validation.

For isolated GPU validation without building the whole repository extension:

```bash
RL_KERNEL_F2_TEST_JIT=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pytest tests/ops/rope/test_qknorm_scaled_rope_forward.py -q
```

Use the target GPU's architecture instead of `8.9` when appropriate. This fixture
compiles only F2, uses PyTorch's default C++ standard, and adds explicit no-fast-math
CUDA flags. The CUDA/ROCm build now delegates the C++ standard to PyTorch rather
than forcing C++17 (PyTorch 2.14 requires C++20). The standard SM80 extension build
and 73 targeted tests passed; fixed-data outputs matched the isolated build bitwise.
See the [SM80 integration report](../../reports/f2-sm80-integration-2026-10-24/README.md)
for artifacts and preliminary call timings. Other platforms and cross-GPU bitwise
identity remain unvalidated.

The [SM89 forward validation](../../reports/f2-forward-2026-10-24/README.md) passed
37 tests (reference and forward suites, no skips), including direct CUDA/Triton
bit-pattern comparisons of BF16 outputs and FP32 rstd. This does not qualify backward,
other GPU targets, table provenance or the repository-wide extension build.

CUDA forward now delegates position-range checking to its checked native entry
point, avoiding a duplicate Python GPU-to-host bounds check. Python still validates
metadata and preserves its ValueError for out-of-range positions; Triton still
checks bounds in Python. No validation is cached. See the
[SM80 bounds-check regression](../../reports/f2-sm80-bounds-2026-10-24/README.md):
80 targeted tests passed and fixed-data outputs remained bitwise identical.

Triton now uses one warp per head on SM80, following a 1/2/4-warp sweep and
80-test regression. Other architectures retain four warps. The logical reduction
tree and rounding rules are unchanged. See the
[SM80 warp-tuning report](../../reports/f2-sm80-warps-2026-10-24/README.md)
for bitwise checks and device-side timing methodology.

Position bounds validation is now fused separately from the floating-point
operator: CUDA and Triton each scan int64 positions in one predicate kernel and
read one int32 flag on the host. This integer reduction does not change the FP32
arithmetic tree. Empty inputs skip validation, no validity is cached, and graph
capture remains unsupported. The [validation-fusion report](../../reports/f2-fused-bounds-2026-10-24/README.md)
records 94 passing tests, unchanged fixed-data outputs, sanitizer results and
Nsight confirmation of one validation kernel and one synchronization per check.

## Autograd entry point

`rl_engine.ops.rope.qknorm_scaled_rope.qknorm_scaled_rope` returns `(query, key)`
with a custom first-order autograd function. It saves Q/K, FP32 rstd and local
position/table tensors; autograd version counters detect mutation before backward.
Noncontiguous upstream gradients are made contiguous before kernel dispatch.
Unused output branches return no input gradient. Positions and tables are constants,
even if a table has `requires_grad=True`. Higher-order derivatives are explicitly
unsupported via `once_differentiable`.

```python
from rl_engine.ops.rope.qknorm_scaled_rope import qknorm_scaled_rope

q_out, k_out = qknorm_scaled_rope(
    q, k, positions, cos, sin, is_local=True, backend="triton"
)
loss = downstream_loss(q_out, k_out)
loss.backward()
```

CUDA/Triton backward use the same per-head mapping as forward. The raw CUDA
backward binding validates gradient and saved-rstd geometry, device and dtype.
Bounds validation remains synchronous; graph capture and runtime registry binding
are not provided by this entry point.

The [SM89 first-order validation](../../reports/f2-backward-2026-10-24/README.md)
passed 64 reference/forward/backward tests with no skips. CUDA/Triton gradients
matched bitwise on the tested cases; independent FP64 autograd comparisons used
the unchanged provisional 0.03 relative-L2 bound, checked per head in GPU tests.
This evidence is limited to the stated software/hardware and input fixtures.


## Table identity contract (A100 validated)

`rl_engine.ops.rope.rope_table.build_shared_rope_table(P, device="cuda")`
returns registered `(cos, sin)` tensors for the existing public F2 signature.
The NVIDIA CUDA producer executes eager FP32 `arange(0,128,2)/128`,
`pow(500000, exponent)`, reciprocal, absolute position multiplication, cos/sin,
then BF16 casts. Autocast is disabled. Algorithm ID is
`f2-torch-fp32-pow-recip-mul-cos-sin-bf16-v1`. This is a candidate implementation
contract, not a claim that upstream has approved a universal table artifact.

`table_identity(cos, sin)` returns immutable identity metadata. SHA-256 covers
sorted compact ASCII JSON metadata plus newline, then all cosine and sine BF16
bits in little-endian row-major order. Metadata includes algorithm, theta,
dimension, position start, length and layout; the digest excludes itself.
`load_shared_rope_table(cos, sin, expected=identity, device=target)` validates
metadata and content against a **trusted producer identity**, then registers
isolated copies. An identity supplied alongside an untrusted payload is not proof
of mathematical correctness. Runtime integrations must pin/distribute the trusted
identity separately. Independent builds across GPUs/PyTorch versions may differ;
share the same canonical artifact for cross-engine/cross-GPU consistency.

Public Python local forward/autograd accepts only the exact registered tensor
pair. Shape/device/layout and version counters are checked at dispatch; full
content hashing and CPU transfer occur only during construction/loading. Global
layers ignore tables. Tables remain constants for autograd even if requires_grad
is set. Ordinary in-place changes invalidate the pair; aliases/clones/transfers
must go through the loader. Concurrent mutation, `.data` writes and external
pointer writes bypassing PyTorch version counters are unsupported. Tables must
remain immutable. Weak registry references do not retain unused tensor storage.

Native extension and private Triton entry points retain their structural/bounds
checks and are internal interfaces, not standalone provenance enforcement.
CPU artifact loading supports tests; it does not introduce a CPU kernel fallback.
No canonical cross-platform hash is hardcoded or inferred from an arbitrary
external table. The standard SM80 extension and complete targeted suite passed
120 tests with no skips, including table round trips and public-entry rejection.
See [table identity validation](../../reports/f2-table-identity-2026-10-24/README.md)
for identity, timing samples and limitations.


## Native eager performance baseline

The [A100 native baseline report](../../reports/f2-native-benchmark-2026-10-24/README.md)
compares the pinned native eager functions with both public fused backends, covering
forward, backward and combined training calls at T=1/128/2048. It includes lookup
and dispatch overhead, excludes table creation, and makes no full-model speedup claim.


## A100–H100 validation

The [H100 report](../../reports/f2-h100-2026-10-24/README.md) records 121 passing
tests and zero bit differences across 144 tensor comparisons against A100 VJP v2.
The cross-device probe reuses identical inputs and the verified A100 table.
This validates the tested SM80/cu130 and SM90/cu132 environments, not all GPUs
or inputs. H100 used an isolated F2 build, not the full extension build.
