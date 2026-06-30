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
- [Fused LogP](fused-logp.md)
- [Fused Linear LogP](linear-logp.md)
- [GRPO Loss](grpo-loss.md)
- [LM Head](lm_head.md)
- [Policy Ratio + KL Penalty](ratio-kl.md)
- [Sampling](sampling.md)
- [Token Embedding](embedding.md)
- [Operator Doc Template](../contributing/operator-doc-template.md)
