# Sampling

The sampling backend provides a unified interface for rollout token sampling. It routes
to optimized vendor libraries when available and falls back to PyTorch sampling logic.

## Entry Point

```python
from rl_engine.ops.sampling.api import SamplerBackend

sampler = SamplerBackend()
tokens = sampler.sample(logits, top_k=50, top_p=0.95, temperature=1.0)
```

## Backends

| Platform | Backend | Status |
| --- | --- | --- |
| NVIDIA CUDA | FlashInfer | Active |
| AMD ROCm | AITER | Planned integration point |
| CPU / fallback | PyTorch | Active fallback |

## Supported Modes

- Unfiltered sampling from logits.
- Top-k sampling.
- Top-p sampling.
- Combined top-k and top-p sampling when the FlashInfer backend is available.
- Deterministic sampling when supported by the backend.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `logits` | `[B, V]` or compatible backend shape | Floating point | Converted to contiguous layout. |
| Output | `[B]` | Integer token IDs | One sampled token per row. |

## Tests and Benchmarks

```bash
python benchmarks/operators/sampling/benchmark_sampling.py
```

Add focused tests when changing backend routing, supported sampling modes, or numerical
behavior.

## Implementation Files

- `rl_engine/ops/sampling/api.py`
- `rl_engine/platforms/constants.py`
