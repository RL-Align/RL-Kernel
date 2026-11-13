# Preserve compressor FP32 projection outputs

The pinned ROCm rollout compressor evaluates BF16 operand GEMMs with FP32
outputs. Native TE previously returned BF16: the projected operands therefore
lost precision before pooling. `bind_compressor_projection(compressor)` retains
the original BF16 Parameter objects and emits FP32 results, including the real
TE bias-free ABI (an empty bias Tensor).

Torch at the locked runtime implements the forward `mm(out_dtype=FP32)` but
raises `derivative for aten::mm is not implemented`. RLK supplies explicit
FP32 input/weight products and converts each leaf gradient to its original
dtype. There is no framework-source patch or replacement model.

MI300X/gfx942 original four-layer checkpoint: both layer-2 compressor projection
regressions match all 132,096 original FP32 rollout elements each; input and
weight gradients are finite and nonzero (2 passed). A real TrainingWorker rerun
confirms both projected tensors remain byte-identical. C4, C128 and C4-indexer
projection bindings execute in that worker; original weights remain in their
checkpoint format. Full raw response scores still mismatch 2/2: fused pooling,
normalization and compressed-cache encoding are separate boundaries.

Initial scope is eager TP=PP=EP=CP=1 with original bias-free BF16 compressor
weights. This does not establish complete four-layer, real-worker batch-position
or post-update parity. Original fixture tests require checkpoint/trace paths.
