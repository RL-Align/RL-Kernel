# DSV4 mHC worker integration (task 7)

Install the binding **inside each worker**, after the engine has created and loaded
its local model and before warmup, compilation or graph capture. Native baseline
workers should not install it. Pass the inner DeepseekV4Model, not an engine or
CausalLM wrapper. Keep attention, routed experts, pipeline transport, loss and
weight synchronization with their existing owners.

```python
from rl_engine.integrations.orchestrators.verl.mhc import bind_mhc_worker
# In a vime worker, instead import ...orchestrators.vime.mhc.bind_mhc_worker.
p1 = bind_mhc_worker(model, role="rollout", backend="rocm", mode="split", version=0)
with p1.request(version=0):
    output = model(**batch)
```

The training binder targets the Megatron-Lite DeepSeek V4 **impl=vllm** layer
ABI at [327a2382](https://github.com/ISEEKYAN/Megatron-LM/tree/327a238239448de6771beea185dc6902a8de3eaa).
Apply the packaged instance-head patch to that checkout before constructing
the training model. The upstream model calls its head and final norm as
module-level functions; the patch routes only bound instances through P1.
Use role `training` and hold the request through forward and backward.
The rollout binder targets the [AMD vLLM model](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/vllm/models/deepseek_v4/amd/model.py).
It returns completed residual streams at the layer boundary, so the outer
vLLM loop must not add the final post a second time. Both binders look up
model weights at call time and preserve the optimizer's parameter objects.

```sh
git -C "$MLITE_ROOT" apply --check "$RLK_ROOT/rl_engine/integrations/engines/train/megatron/patches/dsv4_p1_head.patch"
git -C "$MLITE_ROOT" apply "$RLK_ROOT/rl_engine/integrations/engines/train/megatron/patches/dsv4_p1_head.patch"
```

A worker must expose these engine ABIs. A vime or verl installation using an
unpatched Megatron-Core model is not covered by the mLite binder. These entry
points do not start distributed jobs or add DSV4 model support to an orchestrator.

**Arithmetic.** Split executes post, then pre from task 6. Fused combines post,
the BF16 residual rounding, the next controller's projection and sum of squares
in one Triton launch. The gates and Sinkhorn use the same code as split. The
fused VJP replays the split graph; there is no second backward formula. The
owned backend supports fusion for power-of-two hidden sizes up to 4096. No
backend substitutes itself if a
requested profile is unsupported. Enable fused in a deployment only after
byte comparison on the target hardware; there is no performance claim.

**Synchronization.** Serialize whole local requests with the version used for
rollout or training. After draining outstanding work, perform the real reload
inside the update block. Only acknowledge the new version to the orchestrator
after the block exits successfully. A failed reload blocks requests until a
complete reload succeeds. P1 itself does not own derived weights or repacking.

```python
with p1.updating(next_version, lambda: torch.cuda.synchronize(local_device)):
    engine_reload_or_copy_weights()
```

For static buffers use `replay = p1.capture(step, version=v, device=local_device)`.
Call `replay()` in place of `step()`; output buffers are overwritten. A weight
update invalidates the callable. Capture again for the new version. The
training closure must reset gradients, must not step the optimizer, and should
return the gradients to check. Do not use unguarded engine-managed graph
replay across updates. For eager traces pass an observer accepting framework,
role, version, backend, profile, stage name and tensor; clone tensors if
retaining them. Tracing is disabled during worker-managed graph capture.

**Verification.** CPU fixtures exercise the model ABIs for both orchestrators,
including M=1/16/17 and two updates. They do not start actual worker processes.
The Triton interpreter tests the real fused kernel, including hidden size 4096.
Device tests compare split and fused forwards and gradients, and graph replay
across weight updates. Run on each target and report GPU, versions, shape,
backend, schedule and bytes; an offline compile is not a device run.

```sh
python -m pytest -q tests/integrations/common/test_mhc_worker.py
TRITON_INTERPRET=1 python -m pytest -q tests/backends/rocm/test_dsv4_mhc_post_pre.py -k interpreter
RLK_REQUIRE_ROCM=1 python -m pytest -s -q tests/backends/rocm/test_dsv4_mhc_post_pre.py -k 'not interpreter'
```
