# 阶段 3：FP32/BF16 正确性及配置不变性（2026-10-02 完成）

## 初始完整矩阵

A100-SXM4-40GB；PyTorch 2.6.0+cu124，Triton 3.2.0，CUDA toolkit 12.4；
TF32 禁用。H=3072，B=1/3/16，seed=386/9386，CUDA/Triton，FP32/BF16，24 组合。
证据 evidence/a100/a100-baseline-matrix.json 包含源文件 SHA256、合同 SHA256、
全部误差、每行耗时、原语调度记录和实际 profiler kernel 名。
初始头文件缺失导致的 24 项运行错误保留于 a100-initial-matrix.json；不算数值失败。
实际编译依赖补充只读 CPATH：现有 python-headers/usr/include/python3.10、
x86_64-linux-gnu/python3.10 和 usr/include；系统和研究目录均未修改。

22/24 组合通过。全部 24 组不变性通过：重复、chunk_size=1/2、逻辑 ID 重排、
padding、strided、singleton 输出和 dt。所有比较按 uint8 视图逐字节，不将
+0/-0 的 torch.equal 等价误称逐位相同。参数梯度只比较完整同一逻辑样本集。
全部 4 份 profiler trace 验证 embedding、mm、silu、dt 的真实 CUDA kernel；
不是用 last_trace 标签冒充实际执行。全仓 `_C unavailable` 警告来自既有导入，
本算子的独立扩展实际执行，并未 fallback；证据同时验证 actual_backend 与 kernel。

## 两项数值失败与下一修正

两项均为 FP32，B=16，seed=9386 的 dt（其余五输出/参数梯度通过）：
CUDA max_abs=.017333984375，max_rel=.0006460209843；
Triton max_abs=.017578125，max_rel=.0006471021334。
官方 atol=rtol=1e-4 未修改。归约长链误差经内部 1000 倍时间尺度传播，在接近
抵消的 dt 上超过容差；目前这是待验证的归因，不是已经证明的唯一误差来源。
基线固定顺序长点积改成 CUDA 每 lane 补偿累加 + warp 树，Triton 特征维固定树。
K<=32 的参数/偏置样本归约改为直接逐输出固定顺序 FMA，避免 tile padding 浪费。
分派只看归约 K，不依赖 row batch 数；样本布局先恢复 canonical ID。
下一命令：validate_timestep_official.py --batches 16 --seeds 9386 --dtypes fp32。

候选修正还未通过，不能声称阶段完成。基线源码快照在远端 evidence/baseline-sources.tar.gz。


## 最终参考修正与通过证据

前述结果为保留的历史失败。诊断日志 local-dt-diagnosis.json、
local-dt-sum-diagnosis.json、a100/a100-dt-sum-diagnosis.json 将纯 PyTorch
FP32、仅诊断用 FP64、CUDA/Triton 分开比较。原 CPU row-mv 参考的线性 VJP
本身有明显误差；在本机某个 dt 上，row-mv 为 11.557373，FP64 诊断为
11.564941，单纯提高 GPU 精度反而远离原参考。因此不能把 BLAS 私有归约顺序
当作数学真值，也不能通过放宽官方容差消除问题。

最终 gold 是独立纯 PyTorch FP32 补偿树和显式线性 VJP，sigmoid 与三角函数
仍由 torch autograd 求导；CUDA/Triton 不调用该参考。普通 F.linear + F.silu
自动微分另作语义回归。所有生产验收仍为 FP32 参考和 FP32 累加，FP64 只用于
诊断，没有进入 gtest 或候选实现。补偿方法将两个 FP32 部分和及其舍入残差
合并，保留 high/low，最后回到一个 FP32；CUDA lane 顺序和 Triton 树形不同。
CUDA/Triton 长点积另用 FMA 得到乘积舍入残差。最终 dt 也使用补偿归约。

最终 `a100-delivery-matrix.json`：24/24 通过（H3072、B1/3/16、两种子、两精度、
两后端）。每组合包含前向及全部五梯度的独立 CPU 参考准确性比较和 byte-wise
配置检查。四份 `a100-delivery-matrix-trace-*.json` 均检查到真实 CUDA 内核。
最终四条 check_operator CLI 均通过，`a100-delivery-pytest.log` 为 60 passed，6.61s。
额外 `a100-long-sample-reduction.json` 的 H17、B33/64、seed386、两后端两精度
8/8 通过，覆盖 K>32 参数梯度分支；不能把小 H 证据当作大批次生产性能证据。

生产矩阵各类最大绝对误差（判定仍是逐元素 atol+rtol，非仅看最大值）：

| dtype | output | dt | dW1 | db1 | dW2 | db2 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1.490116e-7 | .001708984375 | 1.907349e-6 | 1.907349e-6 | 3.337860e-6 | 0 |
| BF16 | .003905415535 | .001953125 | .03125 | .00390625 | .03125 | 0 |

本地预编译模块定向复测：B16 seed9386 的 CUDA/Triton × FP32/BF16 共4/4通过。
最终源码与矩阵中算子、gold、CUDA/Triton、gtest 文件 hash 完全匹配。
验证脚本在生产矩阵后只修正了 `Path(__file__).resolve()` 的 runpy 路径处理；
新版本已被本地定向4项和 A100 额外8项覆盖，其完整源码 hash 存于额外矩阵及
final-source-manifest.json。原 runpy 失败保留在 local-packaged-runtime.log。

当前阶段完成。没有原生压缩工具；报告和检查点只作恢复依据，不宣称已执行压缩。
