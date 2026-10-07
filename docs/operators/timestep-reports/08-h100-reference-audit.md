# H100 验收与参考独立审查（2026-10-07）

当前算子通过 H100 实机验收，以及针对 FP32 相位语义的独立高精度审查。
本次未改动参考实现、CUDA/Triton 算法或官方容差；新增审查工具、回归测试和证据。
技术证据支持进入维护者合并审查，不代表已经获得维护者批准。

## 固定源码与环境

完整验收使用提交 `ea55fa686706892ce94ebe7020829c19c108496c` 的源码归档。
后续新增审查脚本/测试/报告；原算子文件内容未变。DCO 签署修正会改变提交身份，
因此以 [source-manifest.json](evidence/h100/source-manifest.json) 的逐文件哈希为准。
独立审查 JSON 另记录审查脚本、参考和合同的哈希。

单卡 NVIDIA H100 SXM 80GB HBM3，计算能力 9.0、MIG disabled；Ubuntu 22.04，
Python 3.10.12、CUDA Toolkit 12.4.131、驱动 580.126.09、GCC 11.4，
PyTorch 2.6.0+cu124、Triton 3.2.0、pytest 8.3.5。TF32 禁用。
原始日志与依赖清单在 [evidence/h100](evidence/h100)，SHA256SUMS 覆盖该目录证据。

## 实机结果

- 强制 JIT 编译 sm90：通过；独立 CUDA/Triton 缓存。
- 原有相关测试：60/60；新增线性 VJP 数值差分后：63/63，无跳过。
- 官方 CLI：CUDA/Triton × FP32/BF16，4/4。
- H3072、B1/3/16、seed386/9386，两后端两精度：24/24。
- H17、B33/64 长样本归约：8/8。
- 四份 profiler trace 捕获实际候选内核，actual_backend 正确且 fallback_reason 为空。
- 完整原有验收含编译耗时 113.06 秒，不含环境安装。

[summary.json](evidence/h100/summary.json) 记录每阶段退出码；
[production.json](evidence/h100/production.json) 保留全部误差、不变性、计时采样。

## 独立参考审查方法

`scripts/audit_timestep_reference.py` 对完全相同的输入、权重和上游梯度比较：
旧版逐行 torch.mv + autograd、新 FP32 补偿参考、CPU FP64 普通 F.linear/F.silu
+ 原生 autograd，以及 CUDA/Triton。FP64 实现不复用候选归约或手写反向。
每个输入和上游张量均记录字节哈希。

生产 H3072；每精度包括 B16 seed9386 的 CUDA/CPU 两种上游 RNG、
B3 seed1701 与 B16 seed20261007 的 CPU 上游 RNG；共 8 组。
全部输出和五类梯度逐元素比较，仍使用官方 reduction atol/rtol，未放宽阈值。

高精度对照分两种，不能混为一谈：

1. **FP32 相位边界 oracle**：频率和相位先按现有合同构造为 FP32；在该相位上
   使用 FP64 三角函数、两层线性、SiLU 和自动微分。相位值固定在 FP32 边界，
   保留解析相位 Jacobian；后续不进行 FP32 舍入。这检查现有语义下的累计误差。
2. **全 FP64 数学诊断**：频率、相位及后续计算均用 FP64。它测量 FP32 频率/相位
   舍入造成的偏差，不作为替代官方 FP32 参考的门槛。

结果：[独立审查 JSON](evidence/h100/h100-reference-audit-final.json) 8/8 通过。
新参考及两候选均通过相位边界 oracle；两候选同时通过新参考。
历史敏感样例 FP32 B16 seed9386 / CUDA RNG 的 dt：旧参考误差最高达到
逐元素容许值的 **5.311 倍**，新参考为 **0.642 倍**。这为参考修正提供了
独立支持，不能推广为所有可能输入上的误差保证。

全 FP64 数学诊断在 3/4 个 FP32 样例至少有一项超出官方 FP32 容差；历史样例 dt
最高约 **80.892 倍**。这是明确保留的语义差异：本算子承诺 FP32 相位计算，
不能声称与全 FP64 连续数学表达在该容差内等价。BF16 的四组该诊断通过。

手写线性 VJP 另经 torch.autograd.gradcheck 的有限差分独立检查：B1/3/33、
非二次幂 K5、H7，覆盖输入/权重/偏置全部偏导。仅该隔离差分测试使用 FP64；
生产参考保持 FP32。[63 项测试日志](evidence/h100/h100-reference-pytest.log)。

## 性能和范围

20 次 CUDA Event 采样的中位数，相对脚本内 native_batched 基线：CUDA 前向+反向
约 1.51–1.85 倍；B16 的纯前向仍慢于基线。Triton 多数形状更慢。
这是一次微基准，不能当作全模型性能保证；没有 WGMMA/TMA 专用优化。

H100 验收和这次独立审查补齐了之前标出的算子级证据缺口。全模型训练、多卡、
全仓原生扩展构建不在此独立算子 PR 的实测范围；本地旧环境 double-free 未定因，
本次隔离 H100 环境的 CPU 参考及 GPU 测试全部通过。
不支持 autocast/二阶梯度；BF16 是存储/输出精度，不是逐层 BF16 舍入语义；
不承诺外部独立 microbatch 的 BF16 .grad 加和逐位一致。

## 复现新增检查

在已安装依赖并可运行 sm90 入口的源码根目录：

```bash
OMP_NUM_THREADS=4 TIMESTEP_CUDA_JIT_ONLY=1 TORCH_CUDA_ARCH_LIST=9.0 \
  python scripts/audit_timestep_reference.py --output /absolute/new-audit.json
python -m pytest tests/test_timestep_reference_audit.py -q
```

审查 JSON 若存在会拒绝覆盖。输出 passed 仅由相位边界对照、候选与参考一致性
及真实后端检查决定；旧参考失败和全 FP64 诊断差异均保留原始数值。

## 框架与自动检查

新增补跑与算子注册相关的 forward/gradient invariance、four-judgment、op_checks、
profiler、tolerance、operator inputs 七个测试文件：**155 passed**，无跳过。
精简验收包最初未包含 benchmarks 和 docs/design/ws1-c8-execute.json，导致一次
收集错误和一次样例缺失；补齐固定提交中的原文件后通过，未修改测试或实现。
三轮原始日志分别保留为 h100-framework-initial、missing-fixture、complete。

目标 test-qwenimage 不在 CI-Pipeline（main/test）或 WS1-gtest-GPU（main）的
自动触发分支中，因此 GitHub 检查绿灯不代表全仓 CI 通过。
原提交缺失 DCO 签署行已修正；提交树中的原有运行时代码保持不变。
