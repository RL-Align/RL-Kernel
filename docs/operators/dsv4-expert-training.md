# DSV4 ROCm expert training module

[Task 48 in #520](https://github.com/RL-Align/RL-Kernel/issues/520) connects the
P5 expert operators into a trainable local module. Training, recomputation and
rollout use the same forward function. Packed base weights stay frozen; LoRA
parameters and selected route weights have explicit backward paths.

The implementation provides `RoutedExpertMLP`, the staged first-order VJP and
an optional `ExpertTrainingBlock` wrapper. It does not install hooks into verl
or vime. Their model replacement, optimizer ownership and distributed gradient
handling belong to tasks 72/73 and P4/P6.

## Dependencies and current status

This PR contains task 48 only. The `dsv4-rocm` base does not yet contain a complete
production P5 provider. A full ROCm training run requires:

- Tasks 43/44: an explicit group-32 MXFP8 quantization and MXFP4 expert GEMM
  adapter, including BF16 backward with FP32 dX. See
  [#519](https://github.com/RL-Align/RL-Kernel/issues/519).
- Task 45: `RocmLoRADelta` from [#601](https://github.com/RL-Align/RL-Kernel/pull/601).
- Task 46: `RocmClampSwiGLUWeighted` from
  [#607](https://github.com/RL-Align/RL-Kernel/pull/607).
- Task 49: an explicit ordered route-weight VJP when `p_s` is trainable.
- Task 47: a trainable shared-expert module, if the optional shared branch is used.

Task 48 binds the real task 45/46 entry points through lazy imports. It requires
the task 43/44 and 49 adapters as constructor arguments; it does not create a
stand-in GEMM, reuse an unqualified AITER path or silently fall back to PyTorch.
Missing required dependencies raise before the routed forward starts. The local
CPU tests use a separately selected diagnostic provider and do not establish
ROCm support. The production dependencies still need integration and hardware
validation, in addition to the module tests provided here.

## Inputs and ownership

`RoutedExpertMLP` accepts BF16 `x[M,H]`, FP32 `p_s[M]` and integer
`expert_offsets[E+1]`. P4 has already dispatched and grouped the rows by local
expert. Offsets start at zero, end at M, and may repeat for empty experts. The
module preserves row order and returns BF16 `[M,H]`; it performs no scatter,
token reduction or communication. Empty local batches are supported.

`PackedExpertWeight` describes the canonical group-32 OCP layout: low-nibble-first
E2M1 bytes and separate E8M0 scales. FC1 has logical shape `[E,2F,H]`, with the
gate rows before the up rows; FC2 has `[E,H,F]`. Repacked layouts must be adapted
explicitly by the task 43/44/50 owners. Hidden and FFN widths must be divisible
by 32 when the base is enabled. Non-finite E8M0 scale code 255 is rejected.

The four shared LoRA matrices are BF16 parameters: A1 `[R,H]`, B1 `[2F,R]`,
A2 `[R,F]`, B2 `[H,R]`. Rank is positive and shared across both projections.
`alpha` is the effective multiplier, with no implicit division by rank. Partial
adapter tuples, unsupported dtypes and unexpected trainable parameters are
rejected. `base_only`, `lora_only` and `combined` are separate explicit modes.
The LoRA-only mode is useful for diagnostics; it is not evidence for the
quantized base path.

The module registers packed base tensors as buffers and LoRA tensors as
parameters. It makes no base-weight gradient request. LoRA can train even when
`x` and `p_s` do not require gradients; individual adapter parameters may also
be frozen. Standard autograd returns BF16 gradients at BF16 parameter/input
edges. FP32 master gradients and optimizer states are framework responsibilities.

## Forward and backward boundaries

Both projections compute the base branch from quantized activations while
passing the original BF16 activations to LoRA. The results are added in FP32.
FC1 remains FP32 at the activation input. Task 46 clamps, computes SiLU × up,
multiplies the route weight once, and writes BF16. That weighted BF16 tensor
feeds both FC2 activation quantization and FC2 LoRA. FC2 base and LoRA results
are added in FP32 before the final BF16 output cast.

Backward follows the P5-S0 staged contract:

1. Cast incoming dY to BF16 and evaluate FC2 base/LoRA VJPs.
2. Add their dH contributions in FP32, then cast once to BF16.
3. Apply task 46's gate/up VJP and, when required, task 49's route-weight VJP.
4. Concatenate dGate and dUp, then cast to BF16 before the FC1 VJPs.
5. Add base and LoRA dX in FP32. The low-level `expert_backward` returns this
   FP32 accumulator; the public autograd edge rounds to the input's BF16 dtype.

Activation quantization uses identity STE. It is not the true derivative of
the discrete quantizer. Parameter gradients retain task 45's specified BF16
inter-GEMM boundaries and FP32 accumulation. Task 49 receives dH, the original
gate/up values and the separate FP32 sigmoid, SiLU and unweighted activation.
No route gradient is reconstructed by dividing by `p_s`.

The diagnostic reference follows the staged formulas in
[the P5 start kit at 029d95c](https://github.com/RL-Align/RL-Kernel/blob/029d95ce30326de6cfae0ff0c05f33ffb9286370/rl_engine/moe/oracle.py).
It uses serial FP32 reductions. Its numerical profile differs from the ROCm
providers; accuracy comparisons and same-profile byte comparisons are separate.

## Recompute, updates and shared experts

With `recompute=False`, autograd saves gate/up, weighted activation, activation
intermediates and both LoRA U tensors. With `recompute=True`, it retains only
inputs, offsets and weights and calls the same `expert_forward` during backward.
All saved tensors participate in PyTorch mutation checks, including frozen
base codes/scales. A provider profile change between forward and backward is
also rejected. Higher-order gradients are not supported.

The module keeps no quantized-activation, adapter-value or transpose cache
across calls. Its packed views are rebuilt from registered buffers after device
moves or state-dict loads. Checkpoints carry the module schema and configuration,
including alpha; a mismatch is rejected before overwriting the module's tensors.
This is local checkpoint protection, not task 50's deployment/version protocol.

`ExpertTrainingBlock` accepts an already implemented task 47 module. One block
call executes it once on the original `shared_x[T,H]`, independently of the
dispatched row count. The return value contains separate routed/shared outputs,
copied output-slot IDs and `route_weight_applied=True`. It rejects a detached
shared output when a gradient is required. The task 47 module still owns its
parameter policy and arithmetic. P6 owns the final output and input-gradient
merge; this wrapper does not certify a distributed shared-once schedule.

## Module entry point

The objects below come from the checkpoint loader and the task 43/44/49 owners:

```python
from rl_engine.backends.rocm.moe.expert_training import RocmExpertTrainingBackend
from rl_engine.contracts.operators.expert_training import ExpertTrainingConfig
from rl_engine.models.deepseek_v4.expert_training import RoutedExpertMLP

backend = RocmExpertTrainingBackend(
    base=quantized_expert_backend,
    route_gradient=route_gradient_backend,
)
experts = RoutedExpertMLP(
    ExpertTrainingConfig(hidden, ffn, local_experts, mode="combined", alpha=alpha),
    backend=backend,
    w1=packed_fc1,
    w2=packed_fc2,
    adapters=(a1, b1, a2, b2),
    recompute=True,
)
y = experts(dispatched_bf16_x, selected_route_weights, expert_offsets)
```

Use the module's own recompute option when inputs are frozen but LoRA is
trainable. An external reentrant checkpoint needs a differentiable explicit
input; otherwise it can detach the output before the module's autograd executes.
The tests also exercise PyTorch reentrant/non-reentrant checkpoint calls with
differentiable inputs. These local checks are not verl/vime end-to-end tests.

## Validation

Run the module and binding tests on CPU:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/models/deepseek_v4/test_expert_training.py \
  tests/backends/rocm/test_expert_training_binding.py
```

Coverage includes base/LoRA/combined modes, empty and uneven experts, strided
inputs, zero route weights, staged gradient merging, checkpoint/recompute,
frozen-input LoRA training, consecutive optimizer updates, checkpoint reload,
unsupported configurations, and dropped-gradient detection. Finite differences
check the smooth FP64 LoRA-only formula, separately from the quantized STE and
BF16 rounding behavior.

After the production dependencies are available, run separately on each target:

```bash
RLK_REQUIRE_ROCM=1 \
RLK_EXPERT_BACKEND_FACTORY=your_provider_module:make_backend \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -s tests/backends/rocm/test_dsv4_expert_training.py
```

`make_backend()` must return a `RocmExpertTrainingBackend` wired to the actual
task 43/44/49 implementations. The test suite uses a small H=F=32, E=3 fixture;
it checks outputs and all requested gradients against CPU staged math, and
checks raw bytes across repeated runs, save/recompute, rollout, batch sizes and
row positions. Unsupported shapes must fail rather than select a substitute.
The hardware flag turns a missing GPU into failure. On a target GPU, a missing
provider factory or dependency is always a failure, not a skip.

The eager validation reads offsets and scale validity back to the host. Graph
capture, distributed execution, production model dimensions, performance, and
actual verl/vime integration remain separate acceptance work. An offline compile
or CPU test result is not evidence that this full ROCm module has run.
