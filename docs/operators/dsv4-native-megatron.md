# NVIDIA Megatron DSV4 P1 adapter

`bind_mhc_worker(model, "training", backend="rocm", mode="split", version=0)`
now recognizes NVIDIA MCore `HybridModel`, in addition to the existing
Megatron-Lite ABI. Bind after loading weights and before any execution/capture.

The native adapter binds each `HyperConnectionHybridLayer.hyper_connection`
pre/post boundary, its separate RMSNorm, the local HybridStack's learned final
contraction, final norm and vocabulary projection. It preserves Parameter
objects. The native attention/MoE branch and scheduler still execute. The
HybridStack forward is cloned with an instance-local contraction callable;
MCore's module globals and other model instances are not mutated.

Use `selected_raw_logprobs(model, logits, aligned_target_ids, weight_version=0)`
inside `worker.request(0)`. This calls the existing RLK deterministic Triton
selected-logprob operator directly; the vocabulary head calls
`TritonLMHeadOp.forward_fp32`. The observer reports execution stages `head`,
`normalized`, `lm_head`, and `raw_logprob`, including the actual weight version.
No temperature/top-k/top-p transformation is accepted at this boundary.

The initial scope is eager TP=CP=1, zero dropout, four BF16 residual streams,
no recomputation/MTP, separate RMSNorm and a BF16 vocabulary head (the original
DSV4 head is BF16). Unsupported configurations fail before binding. This does
not add an original packed FP4/FP8 checkpoint importer to Megatron-Bridge.
Bridge's default DSV4 import dequantizes quantized weights into BF16; it must
not be reported as an original packed-weight parity run.

MI300X validation at MCore `97b50bdd936022ef58031c35db6d10581c181be0`:

```bash
python -m pytest -q tests/integrations/common/test_native_megatron_mhc.py \
  tests/integrations/common/test_mhc_worker.py
```

Result: 6 passed, 0 failed, 0 skipped. The new integration unit constructs a
real native HybridModel with reduced random weights, runs forward, selected
raw logprob backward, changes a parameter and advances its version. This is
unit coverage, not four-layer checkpoint or verl actor/rollout parity evidence.

## Pristine upstream verl engine

Import `rl_engine.integrations.orchestrators.verl.native_engine` before constructing
`TrainingWorker`, then select `RLKNativeMcoreEngineConfig` / `strategy="rlk_megatron"`.
The opt-in engine calls the real native provider, DDP, optimizer and scheduler.
It loads the companion original-format checkpoint importer before wrapping;
`rlk_weight_version` gates every actor request and successful optimizer step.
`rlk_raw_logprobs` preserves the unprocessed FP32 score tensor while the ordinary
verl output path receives a separate logits tensor. Updated parameters can be
exported in the original checkpoint naming scheme for the existing rollout.

The import adapters cover the pinned Bridge 0.7 checkpoint symbol move, verl's
legacy version guard and its obsolete FSDP wrapper tuple. They leave dependency
files unchanged and restore the temporary Bridge import alias afterward.
`native_group_projection.py` validates and adapts exactly one grouped `wo_a`
call site on the loaded native attention instance. `native_rotation.py` binds
the existing ROCm Hadamard primitive to that instance's CSA indexer/compressor.
No native model class or module-global numerical function is replaced.

Tested on MI300X with pristine verl `8c94a12`, MCore `97b50bd`, vLLM `c07d58d`
and all 878 Bridge Python files matching their installed-wheel RECORD hashes.
The actual four-layer actor loads all 6,281 original tensors and returns two
finite raw FP32 scores. The independent real rollout remains numerically
different (2 compared / 2 mismatched); this is integration evidence, not parity.
CPU ABI/unwrap and analytic norm VJP checks: 3 passed, 0 skipped. Unsupported
parallel, recomputation, padding/document masks, PEFT and FSDP paths fail closed.
