# 阶段 4：A100 优化与性能（2026-10-02）

## 环境与方法

A100-SXM4-40GB；driver 550.107.02；CUDA toolkit 12.4.131；PyTorch 2.6.0+cu124；
Triton 3.2.0；Python 3.10.12。见 evidence/a100/environment-final.txt。
测试前后均无其他 compute process；没有终止未知作业，没有增购或关闭服务器。

H=3072，B=1/3/16，性能输入 seed386，参数和输出 FP32/BF16，内部均 FP32。
20 次 CUDA event 测量，中位数；3 次 warmup，编译不计入；包含 Python 调度、
分配和拷贝导致的 GPU 排队空隙，不代表纯 kernel 延迟。前后向使用显式 ones
上游梯度和 autograd.grad，不计 loss.sum。CPU/GPU 准确性验证另用随机上游梯度。

公平性能对照是普通批量 `F.linear(F.silu(F.linear(...)))`，内部 FP32，输出按
参数 dtype 回写，TF32 关闭。独立补偿 gold 只负责准确性，不用其低速制造加速比。
原始各次 samples_ms 与源码 SHA 在 a100-delivery-matrix.json。

## 改动与取舍

- 初版 16×16 tile 的长顺序 FMA 改成固定归约维补偿树；先保证抵消条件下的数值。
- CUDA warp 沿 K 合作、转置必要权重以合并访存；Triton 固定每程序4输出的归约树。
- K<=32 样本归约采用直接输出线程与顺序 FMA，减少空 tile 运算。
- 固定频率仅按 operator/device 缓存；没有缓存权重/激活或 CPU 库指纹。
  record_stream 保护多 stream 下常量缓存的分配器生命周期。
- 默认不分块时直接持有 z/h/y，去掉重复预分配、slice copy 和无用 layout arange。
  chunk 路径保留同一数学顺序；修改后完整矩阵与跨 stream 回归再次通过。

CUDA FP32 B1 前向从初版 .74096ms 降到 .2670ms，Triton 从 .986208ms 降到
.5152ms。两个数是各轮同环境测量，存在运行波动；不能把单个改善外推所有形状。

## 最终实测

下表单位 ms，F 为前向，F+B 为前后向。PyTorch 同行是独立测量的批量对照。

| backend | dtype | B | F | PyTorch F | F+B | PyTorch F+B |
|---|---|---:|---:|---:|---:|---:|
| cuda | fp32 | 1 | 0.2670 | 0.4043 | 1.1886 | 1.7935 |
| cuda | fp32 | 3 | 0.2867 | 0.3648 | 1.2416 | 1.7235 |
| cuda | fp32 | 16 | 0.5403 | 0.4439 | 1.3511 | 1.2795 |
| cuda | bf16 | 1 | 0.4027 | 0.5743 | 1.3431 | 2.2423 |
| cuda | bf16 | 3 | 0.4316 | 0.5184 | 1.5128 | 2.2003 |
| cuda | bf16 | 16 | 0.6154 | 0.5065 | 1.5019 | 1.8235 |
| triton | fp32 | 1 | 0.5152 | 0.3779 | 2.8263 | 1.6541 |
| triton | fp32 | 3 | 0.5502 | 0.3771 | 2.5529 | 1.7602 |
| triton | fp32 | 16 | 1.2416 | 0.7554 | 3.2512 | 2.2209 |
| triton | bf16 | 1 | 0.5319 | 0.4192 | 2.5174 | 1.8460 |
| triton | bf16 | 3 | 0.8155 | 0.5672 | 3.1407 | 2.2444 |
| triton | bf16 | 16 | 1.1311 | 0.5799 | 2.5337 | 2.6922 |

CUDA 小批次获得部分加速，但 B16 前向较慢；Triton 多数场景仍慢于普通批量
PyTorch。这个交付提供确定性和数值合同内的实现，不宣称全场景性能优势。
补偿归约和 Python 调度仍有成本，进一步融合 bias/activation、减少 backward
transpose 是后续可选优化，不能在没有新证据时声称收益。

复现命令与运行环境见阶段5，或使用 scripts/run_timestep_official.sh。
未做 H100/SM90 实测、跨机器CPU逐位对齐、全模型训练或多GPU测试。
