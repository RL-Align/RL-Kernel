# 草稿 PR 状态（2026-10-06）

> 本文保留迁移/草稿时的历史状态。2026-10-07 已完成 H100 实机验收和参考独立审查，见[最新报告](08-h100-reference-audit.md)。

提交用于审查 timestep_embed_mlp 独立 PyTorch 参考、CUDA/Triton 前后向实现、
官方 gtest 接入，以及 sm90 兼容验证入口；不声明可合并或完成 sm90 优化。

## sm80 / A100 历史测试

环境：A100-SXM4-40GB（sm80），CUDA 12.4.131，PyTorch 2.6.0+cu124，Triton 3.2.0。

- 生产 H3072、B1/3/16、seed386/9386、CUDA/Triton、FP32/BF16：24/24。
- H17、B33/64 的长样本归约：8/8。
- 官方 CLI 两后端 × 两精度：4/4；相关 pytest：60 passed。
- 配置不变性及四份真实 profiler trace 已记录，未使用 fallback。
- CUDA 部分小批次有加速；Triton 多数形状没有性能优势。

原始 JSON、失败迭代与日志在 evidence/a100/。这些结果对应历史源码清单
final-source-manifest.json；当前 CUDA/Triton 内核、参考实现及数学计算与该清单一致。
之后变更了 CUDA 加载器（显式 JIT-only 开关）、验证 shell 入口，并新增 sm90 入口。
因此不可把历史 A100 结果称为当前提交的完整 GPU 重测。
当前源文件哈希单独记录在 draft-source-manifest.json，不覆盖历史清单。

## sm90 版本

这是同一算法的兼容入口，不是 WGMMA/TMA 专用优化版本。
run_timestep_sm90.sh 检查计算能力 9.0、设置 TORCH_CUDA_ARCH_LIST=9.0、
强制从当前源码 JIT 编译，并使用独立 CUDA/Triton 缓存。
复现命令见 06-sm90-port.md。没有 sm90 实机编译、正确性或性能结果。

## 当前本地检查

本地缺少 pytest，改用标准库 unittest 执行 test_timestep_official.py 的 CPU 测试；
原计划用 CUDA_VISIBLE_DEVICES='' 跳过 GPU 用例，但 unittest 进程在输出测试结果前
发生 `free(): double free detected in tcache 2` 并 abort，未取得当前提交的新测试通过结果。
这是待排查的本地执行失败，不能据此断言是环境或算子根因。
两个 shell 入口通过 bash -n，提交通过 git diff --check。

## 合并前仍需完成

- 对参考修改补充独立精度证据：相同输入、权重、上游梯度下比较原参考、最终参考、
  独立高精度实现和两个 GPU 候选，覆盖历史失败形状及新种子、输出及全部五类梯度。
  现有 FP64 诊断仍保留 FP32 相位，不能当作端到端 FP64 真值证明。
  24/24 仅证明候选符合修改后的参考，尚不能单独证明参考修改更准确。
- 对手写线性反向补充独立检查，保留现有官方误差阈值。
- 在目标 sm90 机器运行完整验证，核对实际内核、无 fallback、数值与性能。
- 完成目标仓库 CI 和审查；目前未验证全仓原生扩展构建、全模型或多 GPU。

BF16 使用 FP32 中间计算，不承诺逐层 BF16 舍入重放；不支持 autocast/二阶梯度。
参数梯度不变性针对同一完整逻辑样本集，不承诺独立 BF16 microbatch .grad 加和不变。
