# Task 28: MI300X gate projection

The implementation follows the gate FP32-output and batch-invariance contract
from the DSV4 ROCm plan, including task 9's fixed K accumulation dependency.
The local model adapter and AITER wrapper are implemented; whole-model and
framework integration are outside this report's scope.

`task28-mi300x.json` records one MI300X VF (gfx942), the exact software/source
fingerprints, configuration, checkpoint tensor checksum, synthetic activation
seed, accuracy against FP64, and GPU-event latency. M=1/16/17/32/65 all have
zero raw-bit mismatches for the fixed row and between autograd-enabled and
inference-mode forwards. The actual checkpoint gate Parameter requires gradients.

The checkpoint is DeepSeek-V4-Flash-0731 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, tensor
`layers.0.ffn.gate.weight`, BF16 `[256,4096]`. Activations are synthetic;
this is not full-model logprob or tokenizer validation.

Focused verification:

```bash
python -m pytest tests/ops/moe/test_router_gate.py \
  tests/runtime/test_router_gate_dispatch.py tests/runtime/test_kernel_registry.py \
  tests/validation/operators/test_router_gate_fixture.py -q
```

Result after fixture-validation hardening: **78 passed, 1 skipped** (the initial
projection suite had 45 passed). The skipped test is the NVIDIA-only mm/CuTe
comparison. The expected empty-graph warning comes from explicitly rejecting
Graph capture. Tests cover near scores that BF16 would collapse, batch/stride/
row-placement variation, FP32 mathematical accuracy, gradients, version updates,
and unsupported inputs/backend failures. Ruff and the strict MkDocs build pass.

The 33 added CPU regressions verify bounded HTTP ranges, tensor byte spans,
checkpoint identity, and independent comparison with the frozen weight SHA256.
Invalid downloads fail before saving, and invalid replay files fail before GPU
selection. A fresh HTTP-range download passes the frozen SHA256 check.

The installed AITER uses its older Triton-only API; the RFC-pinned multi-backend
API is source-reviewed but not runtime-tested. MI355X, NVIDIA CUDA, framework
hooks, distributed execution, LoRA updates, and full-model acceptance remain
unvalidated. The measured gate path does not certify those configurations.

See [the operator documentation](../../../docs/operators/router-gate.md) for
commands to regenerate the approximately 2 MB checkpoint fixture and JSON report.
