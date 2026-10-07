# Operators

Each upstreamed operator must be documented in this section. Treat the documentation page
as part of the operator contract: inputs, outputs, supported backends, dispatch behavior,
accuracy expectations, and known limitations should be clear before merge.

## Required Page Content

Every operator page should include:

- Purpose and target workload.
- Public Python entry point.
- Backend implementations and fallback behavior.
- Input and output tensor shapes, dtypes, devices, and contiguity requirements.
- Accuracy or numerical tolerance expectations.
- Minimal usage example.
- Related tests and benchmarks.

## Current Pages

- [SiLU / SwiGLU Activation](activation.md)
- [Standard Attention](attention.md)
- [Fused LogP](fused-logp.md)
- [Fused Linear LogP](linear-logp.md)
- [Batch-Invariant LogP](batch-invariant-logp.md)
- [Fused Linear LogP TP Test Runbook](linear-logp-tp-test.md)
- [GRPO Loss](grpo-loss.md)
- [RoPE](rope.md)
- [LM Head](lm_head.md)
- [Policy Ratio + KL Penalty](ratio-kl.md)
- [Pack and Pad](pack-and-pad.md)
- [Matmul](matmul.md)
- [Sampling](sampling.md)
- [Token Embedding](embedding.md)
- [MiniMax-H3 Timestep Sinusoid](h3-timestep-sinusoid.md)
- [MiniMax-H3 FP32 Timestep MLP](h3-timestep-mlp.md)
- [MiniMax-H3 AdaLN Projection](h3-adaln-projection.md)
- [MiniMax-H3 AdaLN Row Gather](h3-adaln-row-gather.md)
- [MiniMax-H3 RMSNorm and AdaLN Modulation](h3-rmsnorm.md)
- [Operator Doc Template](../contributing/operator-doc-template.md)
