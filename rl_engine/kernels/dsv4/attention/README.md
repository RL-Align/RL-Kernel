# DSv4 MQA attention and grouped output projection

Supports DSv4 C0, C4 and C128 layers.

- `mqa_joint_attention_sink.py`: one softmax over compressed rows, recent rows and sink.
  The sink has no V. `oracle.py` supplies the sequential FP32 reference.
- `o_proj/`: inverse GPT-J RoPE on `[448:512]`, eight grouped `wo_a` projections, then `wo_b`.
- `block.py`: connects attention and output projection using recorded inputs.

Inputs are Q `[T,64,512]`, K/V `[N,512]`, sink logits, a candidate plan, RoPE tables
and projection weights. The block returns `[T,4096]`. The caller manages the cache,
compressor and indexer. These tests cover recorded operators and a synthetic model.

CUDA attention requires `T > 0` and allows zero candidates. For DetGemm,
`block.py` converts the FP32 attention result to BF16. Supply BF16 projection weights.

DetGemm follows `RL_KERNEL_DET_GEMM_BACKEND`. On Hopper, `sm90` requires an
SM90-enabled extension; `cublaslt_nosplitk` requires `CUBLAS_WORKSPACE_CONFIG=:16:8`
and `CUBLASLT_WORKSPACE_SIZE=1` before Python starts. Missing strict backend support
raises an error. The test runner reports capability and skips unavailable strict
DetGemm integration tests; attention CUDA tests still run.

Tests are in `tests/dsv4/attention/`. Run these from the repository root:

```bash
python scripts/check_dsv4_attention.py --skip-cuda
python scripts/check_dsv4_attention.py
flock /tmp/rl-kernel-t06-gpu.lock python examples/dsv4_attention_min_model.py --tokens 128 --steps 3
```

The test runner takes the GPU lock itself. Set `CUDA_HOME`, `CC`, `CXX` or
`TORCH_CUDA_ARCH_LIST` to select the toolkit, compiler or GPU architecture.
