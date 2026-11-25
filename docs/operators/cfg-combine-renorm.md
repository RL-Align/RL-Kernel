# BAGEL CFG combine and renormalization

## Summary

BAGEL combines a main latent velocity with up to two guidance branches during a diffusion step. The operator applies the timestep window separately to each request and keeps requests independent when their tokens share a packed tensor.

## Entry Point

```python
from rl_engine.ops.guidance import cfg_combine_renorm

velocity = cfg_combine_renorm(v_main, v_text, v_img, cu_seqlens=offsets,
                              timestep=t, renorm_type="global")
```

With text scale `st > 1`, the text mixture is `U = T + st * (V - T)`. When an image branch is present and `si > 1`, the image mixture is `I + si * (U - I)`. Text and image scales default to 4.0 and 1.5. When the effective text scale is at most 1, the output is V and both guidance branches receive zero gradient.

## Backends

| Backend | Wrapper | Native symbol | Validation |
| --- | --- | --- | --- |
| PyTorch CPU | `NativeCfgCombineRenormOp` | PyTorch FP32 tensor operations | CPU reference and gradient tests |
| PyTorch on NVIDIA CUDA | `NativeCfgCombineRenormOp` | Same-device FP32 gold | GPU run pending |
| Triton on NVIDIA CUDA | `TritonCfgCombineRenormOp` | Shared Triton kernels | GPU run pending on RTX 4090 / L40 / L40S |
| ROCm, Ascend, MUSA | — | — | Dispatch raises an error. ROCm validation is follow-up work. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `v_main` | Dense `[B,S,D]` or packed `[N,D]` | BF16, FP16, FP32 | Defines shape, device and output dtype |
| `v_text`, `v_img` | Same as main | Same as main | Same device. Text is required when any request uses guidance. Image is optional. |
| `cu_seqlens` | `[B+1]` | int32 or int64 | Packed only, same device, strictly increasing from 0 to N |
| `timestep` | Scalar or `[B]` | Compared in FP32 | Tensor on input device or Python scalar. Omit to skip the time-window check. |

The default interval is `(0.4, 1.0]`. A request outside the window takes the identity path. The renormalization factor for mixture X is `clamp(norm(V) / (norm(X) + eps), cfg_renorm_min, 1)`, with default minimum 0 and epsilon 1e-8.

- `global` computes one factor per request after both mixtures.
- `channel` computes one factor per token over its last dimension after both mixtures.
- `text_channel` rescales U per token before the optional image mixture.

Strict packed global mode requires offsets even when all requests are outside the interval. With `strict=False` and missing offsets the packed tensor forms one request. Packed channel modes also require request boundaries for per-request gating across multiple requests.

## Dispatch Behavior

The public function defaults to `backend="auto"` and resolves the operator through `kernel_registry.get_op("cfg_combine_renorm", device=v_main.device)`. CPU selects PyTorch. NVIDIA CUDA selects Triton. Set `backend="pytorch"` for the same-device reference or `backend="triton"` to require Triton. A missing GPU dependency or unsupported accelerator fails at selection or execution.

## Accuracy

The reference computes branch arithmetic, squared norms and the backward dot product in FP32. Adjacent leaves are paired in a fixed order. The Triton implementation uses fixed tiles and reduces partial results in request-local order. These operations use no TF32 matrix arithmetic. The public output is cast once to the input dtype; `forward_fp32` on the reference retains FP32. Clamp endpoints have zero scale derivative, and the norm at zero uses a zero subgradient.

The WS1 precision profile supplies separate reduction tolerances for forward and gradient accuracy. The invariance check compares the output and each input gradient bit for bit when neighbors and request order change.

## Performance Notes

The benchmark checks output and gradients against the native path before timing. It covers typical BAGEL latent widths (`D=64`) and token counts (1024 and 4096). NVIDIA latency, speedup and peak memory measurements are pending the rented-GPU run.

```bash
python benchmarks/operators/guidance/benchmark_cfg_combine_renorm.py --output-dir /tmp/cfg_combine_renorm_bench
```

## Tests

```bash
python -m pytest -q tests/reference/guidance tests/ops/guidance tests/backends/shared/triton/guidance
python tools/validation/operators/check_operator.py --op cfg_combine_renorm --candidate pytorch --device cpu --dtype fp32 --check-grad
python tools/validation/operators/check_operator.py --op cfg_combine_renorm --candidate triton --device cuda --dtype bf16 --check-grad
```

## Known Limitations

Only the velocity inputs participate in differentiation. Packed request boundaries and per-request gates are read on the host, which can synchronize CUDA execution. The CPU results cover the mathematical reference and dispatch behavior. CUDA numerical and performance results require a real NVIDIA device.
