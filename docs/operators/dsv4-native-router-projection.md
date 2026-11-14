# Fixed original router projection across batch sizes

The first expanded real-worker matrix matched actor versus rollout within each
configuration, yet both engines changed a logical sample's scores between batch
one and batch two. Layer-0 residual, mHC, norm, original FP8 Q/KV projection,
attention and the FFN norm input all matched. The first confirmed difference was
router projection: 25,560/33,536 FP32 logits changed for the same original input
and checkpoint weight. First words: `0xc0258420` versus `0xc0258421`.

The native TE router GEMM and vLLM GateLinear hipBLASLt path both varied their
FP32 accumulation with the M dimension. The instance adapters bind only the gate
projection to RLK's existing fixed-order, no-split FP32 GEMM. They preserve the
original BF16/FP32 weight Parameter, FP32 output, unchanged hash/learned router
selection and real dispatcher. Forward does not quantize or replace the weight.
The explicit VJP retains FP32 upstream values and returns each original leaf's
dtype, using the same fixed GEMM family for input and weight derivatives.
The scope is original hidden size 4096 / 256 experts, bias-free, gfx942 eager.

After this change, the real four-layer A/B and B/A matrix compares all two A and
three B response scores exactly within and across engines, batch size and position.
Full updated matrix and independent container acceptance are recorded separately
by the harness. The capture regression uses the original checkpoint router weight
and real single/batch inputs, preserves the earlier failing logits as evidence,
then checks repeat/selected-row/batch-byte equality and finite nonzero VJPs against
an independent FP64 oracle. Oracle tolerances are derivative/accuracy checks;
response-score acceptance always compares original FP32 words exactly.
