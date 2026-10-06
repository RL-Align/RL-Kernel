# timestep_embed_mlp 官方交付报告

当前提交用于草稿 PR 审查，尚未达到合并验收状态。
A100 历史测试见[阶段 5](05-final-review.md)，当前提交边界见[草稿 PR 状态](07-draft-pr-status.md)。

1. [合同与数学规格](01-contract-and-math.md)
2. [最小实现与 gtest](02-minimal-implementation.md)
3. [FP32/BF16 准确性与配置不变性](03-accuracy-and-invariance.md)
4. [A100 优化与性能](04-a100-optimization.md)
5. [A100 历史交付审查与复现](05-final-review.md)
6. [sm90 兼容入口（未实机验收）](06-sm90-port.md)
7. [草稿 PR 状态与待办](07-draft-pr-status.md)

[算子使用文档](../timestep-embed-mlp.md) · [源码 SHA256 清单](evidence/final-source-manifest.json)

生产 A100 矩阵24/24、额外矩阵8/8、官方gtest四条、pytest60项通过。
阶段1/2与阶段3前半部分保留当时的方案和失败，最终算法及结论以阶段3后半和阶段5为准。
所有原始失败、性能采样和真实Profiler trace留在evidence/，没有沿用研究线通过结论。
Triton多数场景尚无性能优势，H100/全模型/全仓原生扩展CI未验证。
