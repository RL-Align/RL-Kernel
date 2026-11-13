# Original DSV4 quantized checkpoint import into native MCore

`load_native_checkpoint` explicitly imports a single-file four-layer DSV4
checkpoint at TP=PP=EP=CP=1, before DDP and optimizer registration. It requires
every source tensor to be consumed. The MI300X run loaded all 6,281 tensors.

Original FP4 expert bytes and block32 E8M0 scales remain packed. The existing
staged expert training composition receives the explicit packed projection and
input-VJP provider. Original FP8 projection bytes/block128 scales also remain
resident. Their forward reuses AITER's production group128 FNUZ quantizer and
a fixed CK no-split-K GEMM; input VJP is explicit Triton. FNUZ weight preparation
maps negative zero and doubles E8M0 scales, preserving the original values.
No BF16 base-weight cache or TE FP4 capability override is used.

The native dense grouped output `wo_a` requires the optional MCore module
boundary in NVIDIA/Megatron-LM#9402. The model remains the real native
`HybridModel` with its attention, compressor, indexer, router, dispatch and
expert structure. The existing native P1 adapter supplies mHC, norm, head and
raw logprob at `backend="rocm", mode="split"`.

Quantized base weights are frozen. Original BF16/FP32 parameters remain
trainable, including embedding/head, norms, mHC, router and compressors. The
local MI300X `verl.workers.engine_workers.TrainingWorker` completed inference,
forward/backward and one real Adam update. Its 71 original-format nonquantized
weight/buffer tensors were exported and loaded by the actual verl-owned vLLM
worker. Version 1 and final-norm bytes were acknowledged after GPU sync.

The initial four-layer strict comparison is **not passing**: both before and
after the update, 2 finite original-FP32 token scores were compared and both
mismatched. Embedding and initial mHC/norm tensors match exactly; native
attention arithmetic still differs. Actor batch/position replay also differs.
These results must not be promoted to bitwise-parity claims.

This importer depends on the independent native P1, packed-MX provider and
ragged GEMM fixes. The real worker calls, version propagation and raw-logprob
export live in a separate verl patch, not in the numerical kernels.
