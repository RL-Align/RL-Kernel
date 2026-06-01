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
| CUDA | SM90 fused LogP when available, CUDA generic, FlashInfer, Triton generic, PyTorch native |
| ROCm | AITER, Triton generic, PyTorch native |
| CPU | PyTorch native |

For CUDA devices with compute capability 9.0 or newer, the registry inserts the SM90
LogP backend at the front of the CUDA priority list.

## Relevant Files

- `rl_engine/kernels/registry.py`
- `rl_engine/platforms/device.py`
- `rl_engine/kernels/ops/`
