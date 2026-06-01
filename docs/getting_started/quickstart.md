# Quick Start

RL-Kernel exposes operators through a runtime registry. The registry selects a backend
based on the current device and available compiled extensions.

```python
import torch
from rl_engine.kernels.registry import kernel_registry

logits = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16).contiguous()
token_ids = torch.randint(0, 4096, (16,), device="cuda", dtype=torch.int32)

logp = kernel_registry.get_op("logp")
selected_log_probs = logp(logits, token_ids)
```

For environments without a compiled CUDA/ROCm extension, the registry falls back to the
available PyTorch implementation when supported by the operator type.

## Validate Dispatch

```bash
python -m pytest rl_engine/tests/test_dispatch.py -v
```

## Validate Operator Accuracy

```bash
python tests/test_op_accuracy.py
```
