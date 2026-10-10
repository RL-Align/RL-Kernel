# Single-GPU GRPO Example

`examples/post_training/grpo_single_gpu.py` is a minimal, reproducible GRPO training script for one
device. It uses a synthetic toy batch, so it does not require a dataset download,
rollout service, distributed launcher, vLLM, Ray, or DeepSpeed.

The example demonstrates the end-to-end path used by RL-Kernel tests:

- deterministic RL-shaped batch creation
- grouped rewards and GRPO-style advantage normalization
- selected-token log probability extraction through RL-Kernel dispatch
- clipped GRPO policy loss with reference KL
- a real optimizer step on a tiny trainable policy

## Run on One GPU

Build the CUDA extension before requiring the fused logp backend:

```bash
MAX_JOBS=2 python setup.py build_ext --inplace
```

Then run the strict A100/single-GPU path:

```bash
python examples/post_training/grpo_single_gpu.py \
  --device cuda \
  --require-fused-logp \
  --steps 2 \
  --num-prompts 1 \
  --samples-per-prompt 2 \
  --prompt-len 2 \
  --completion-len 3 \
  --vocab-size 16 \
  --hidden-dim 8
```

`--require-fused-logp` makes the script fail if CUDA dispatch falls back to a non-fused
backend. This keeps the example honest when validating the RL-Kernel fused logp path.

## CPU Smoke Test

The same script can run without CUDA for quick checks:

```bash
python examples/post_training/grpo_single_gpu.py \
  --device cpu \
  --steps 2 \
  --num-prompts 1 \
  --samples-per-prompt 2 \
  --prompt-len 2 \
  --completion-len 3 \
  --vocab-size 16 \
  --hidden-dim 8
```

The CPU path is intended for CI and local development smoke tests. It should not be used
to validate CUDA fused kernel dispatch.

## Validation Result

Local validation on an NVIDIA A100 with CUDA 12.4 completed successfully:

```text
backend=FusedLogpGenericOp
completed grpo_single_gpu steps=2 device=cuda backend=FusedLogpGenericOp
```

The example also reports `kernel_max_abs_error` by comparing the selected fused logp
output with the PyTorch reference path. During training, the fused kernel is used for
forward-path validation; the optimizer step uses the autograd reference logprobs when the
selected backend does not expose a differentiable autograd path.

The PR validation set used for this example:

- CUDA extension build passed with `MAX_JOBS=2 python setup.py build_ext --inplace`.
- Strict fused single-GPU run passed with `backend=FusedLogpGenericOp`.
- CPU smoke tests passed with `python -m pytest tests/e2e/test_grpo_single_gpu_example.py`.
- CI dispatch tests passed with `python -m pytest tests/runtime/test_dispatch.py -v`.
- Type checking passed with `python -m mypy --ignore-missing-imports rl_engine/`.
- Docs build passed with `mkdocs build --strict -f mkdocs.yaml`.
- Pre-commit equivalent checks passed with Black, isort, and flake8 using repository hook args.

## Test Coverage

Run the example smoke tests with:

```bash
python -m pytest tests/e2e/test_grpo_single_gpu_example.py
```

The tests cover direct script execution on CPU and verify that strict fused mode rejects a
fallback backend.
