# Runtime Dispatch

RL-Kernel routes operators through `KernelRegistry`. Callers request an operator by
logical type, and the registry selects the first available backend for the current device.

## Dispatch Flow

1. Detect platform from `device_ctx`.
2. Load the priority list for the requested operator type.
3. Try each backend in priority order.
4. Cache successfully constructed operator instances.
5. Skip backends that already failed in the current process.

## LogP Priority

| Platform | Priority |
| --- | --- |
| CUDA | CUDA generic LogP by default; experimental SM90 fused LogP only when explicitly enabled, FlashInfer, Triton generic, PyTorch native |
| ROCm | AITER, Triton generic, PyTorch native |
| CPU | PyTorch native |

For CUDA devices with compute capability 9.0 or newer, the registry only inserts
the legacy SM90 LogP backend when `RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP=1` is
set. The fused linear logp SM90 backend is gated separately and remains the
default linear logp backend when the extension is built on Hopper.

## Relevant Files

- `rl_engine/kernels/registry.py`
- `rl_engine/platforms/device.py`
- `rl_engine/kernels/ops/`
