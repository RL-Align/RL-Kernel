# Vime Rollout LogP Probe

This page documents the minimal WS5 proof-of-concept for issue #120. It wires
one existing RL-Kernel operator into a single Vime rollout path using Vime's
public `--custom-generate-function-path` hook.

## Issue Checklist Mapping

- Smallest adapter/shim: `custom_generate` wraps one Vime generate call and one
  RL-Kernel `logp` probe.
- Explicit opt-in: `RL_KERNEL_VIME_LOGP_PROBE=1` is required before RL-Kernel is
  invoked.
- Instrumentation: `Sample.metadata["rl_kernel"]["vime_logp_probe"]` records
  structured evidence, including whether the operator was invoked and the
  process-local `call_count`.
- Run the minimal vime example: start from the #117 fully-async Qwen2.5-0.5B
  baseline and add the exact command/config below.
- Smoke test with mocks: `tests/test_vime_rollout_logp_probe.py` installs a fake
  Vime module when full Vime dependencies are unavailable.
- Fallback behavior: when RL-Kernel or its CUDA extension is unavailable,
  non-strict mode keeps the native generated sample unchanged; native Vime/vLLM
  generation failures still surface normally.

## What It Proves

The probe proves that a Vime rollout can invoke RL-Kernel from an opt-in custom
generate shim. It does not replace vLLM sampling or rollout-side logprob
computation. Vime's HTTP rollout path returns selected logprobs, not logits, so
the shim runs a small deterministic synthetic tensor through RL-Kernel's `logp`
operator and records structured evidence in `Sample.metadata`.

## Entry Point

Use this Vime custom generate path:

```text
rl_engine.integrations.vime.rollout_logp_probe.custom_generate
```

Enable the probe with:

```bash
export RL_KERNEL_VIME_LOGP_PROBE=1
```

Optional strict mode:

```bash
export RL_KERNEL_VIME_LOGP_STRICT=1
```

Strict mode raises if RL-Kernel import or backend dispatch fails. Without strict
mode, the shim records fallback metadata and returns Vime's native generated
sample unchanged.

## Minimal Vime Command

Starting from the #117 baseline
`vime/examples/fully_async/run-qwen2.5-0.5B-fully_async.sh`, add the custom
generate function to `ROLLOUT_ARGS`:

Add this line inside the script's `ROLLOUT_ARGS` array:

```bash
--custom-generate-function-path rl_engine.integrations.vime.rollout_logp_probe.custom_generate
```

Make RL-Kernel importable inside the Ray job runtime environment. Either install
RL-Kernel in the image, or include the checkout path and opt-in variable in the
script's `RUNTIME_ENV_JSON`:

```json
{
  "env_vars": {
    "PYTHONPATH": "/path/to/RL-Kernel:/root/Megatron-LM/:${SCRIPT_DIR}",
    "RL_KERNEL_VIME_LOGP_PROBE": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
  }
}
```

Then run the baseline script normally:

```bash
bash examples/fully_async/run-qwen2.5-0.5B-fully_async.sh
```

For a direct `train_async.py` or `ray job submit` command, the exact Vime
argument to add is:

```bash
--custom-generate-function-path rl_engine.integrations.vime.rollout_logp_probe.custom_generate
```

The run should produce samples whose metadata contains:

```python
sample.metadata["rl_kernel"]["vime_logp_probe"]
```

Expected fields include:

```text
enabled
invoked
call_count
op
backend
fallback
fallback_reason
output_shape
output_sum
```

The `invoked` field proves the shim reached `kernel_registry.get_op("logp")`
and executed the returned operator for that sample. `call_count` is a
process-local successful invocation counter.

## Fallback Behavior

The shim always calls Vime's native `vime.rollout.vllm_rollout.generate` first.
If native Vime/vLLM generation is unavailable, the run fails the same way the
native Vime path would fail; the shim does not hide that failure. If the probe is
disabled, it records `enabled=False` and returns the sample. If RL-Kernel is
unavailable, a backend is unavailable, or the CUDA extension is not built,
non-strict mode records `fallback=True` and returns the native generated sample
unchanged.

This keeps pure Vime inference and native RL paths unaffected when the probe is
disabled or when RL-Kernel cannot run.

## Local Smoke Test

The mock smoke test does not require a full Vime installation:

```bash
python -m pytest tests/test_vime_rollout_logp_probe.py
```

The test installs a fake `vime.rollout.vllm_rollout.generate`, exercises the
custom generate shim, verifies that RL-Kernel `logp` dispatch was invoked, and
checks non-strict fallback behavior.
