# P3 / T06 Router Backward：第一版 CUDA 核心

本实验实现合同 `p3-router-task-contract.v22` 的 `dweights → ds` 数学主体，供
Hash / Learned 两条反向路径共用。原始张量测试之外，另有测试用的 PyTorch 前向
生成保存数据，再直接调用 CUDA 反向。尚未接入团队共享的 `SavedRouteSealedV1`、
provider 或算子注册表；这里的结果不代表 P3 WS1 或训练框架集成已经完成。

## 从哪里读

1. `csrc/cuda/moe/router_backward_core.cu`：先读 `route_backward_kernel`，这是要学习和修改的核心。
2. `examples/p3_router_backward/prototype.py`：构建扩展、检查实验输入，以及 CPU 诊断参考。
3. `tests/test_p3_router_backward_core.py`：对照、重复 expert、padding、线程配置与异常输入测试。
4. `examples/p3_router_backward/bindings.cpp`：把 CUDA 启动函数暴露给独立 Python 扩展。
5. `forward_reference.py` 与 `tests/test_p3_router_backward_handoff.py`：测试前向、保存数据及直接反向的交接。

扩展按需编译；本实验不改动主库的 `setup.py` 和运行时分发。

## 从 Softmax / RMSNorm 经验迁移过来

你熟悉的「一个 block 处理一行，shared memory 交换数据，再同步」可以直接使用。
这次一行对应一个 token；输出一行有 256 个 expert 梯度。

| 工作 | 谁负责 | 为什么这样安排 |
| --- | --- | --- |
| 读六个 `g`、`p`、expert ID | thread 0 | 数量很小，第一版先把运算顺序写清楚 |
| 算六路 `c` 和六个 `da` | thread 0 | 精确遵循合同指定的加法树 |
| 保存六个 `da` 和 ID | shared memory | 同一 block 的其他 warp 都要读 |
| 等待计算完成 | 全部线程执行 `__syncthreads()` | 读 shared memory 前建立同步 |
| 写 `ds[token, expert]` | thread `expert` | 每个输出只有一个写入者，连续线程写连续地址 |

默认每 block 256 个线程。测试也运行 128 个线程，此时每个线程负责两个 expert；
每个输出的计算顺序不变，结果应该逐位相同。

### 六路求和：显式写树

```text
gp[i] = g[i] * p[i]
c = ((gp[0] + gp[1]) + (gp[2] + gp[3])) + (gp[4] + gp[5])
da[i] = (1.5 / Z) * (g[i] - c)
```

这里 `p` 是前向保存的未乘 1.5 的比例，`Z` 是前向保存的归一化分母。
Python 实验参数 `z` 表示这个 **Z**，不是前面 gate GEMM 的 logits。
反向直接使用保存的 `p` 和 `Z`，不重新归一化，也不重新加 epsilon。

普通 warp reduction 的结合顺序未必是上面这棵树，所以这里不调用通用
`warp_reduce_sum`。`__fmul_rn` / `__fadd_rn` / `__fsub_rn` / `__fdiv_rn`
分别指定一次 FP32 舍入；乘法和加法分开，不让编译器合成 FMA。
独立扩展也显式关闭 FMA contraction 和 flush-to-zero。
这些构成当前实验的算术选择，仍需与团队共享的 P3 参考实现和验证入口对接。

### 重复 expert：让输出线程按顺序收集

例如六个 ID 是 `[7, 7, 3, 255, 7, 3]`：

- thread 7 依次累加 `da[0]`、`da[1]`、`da[4]`。
- thread 3 依次累加 `da[2]`、`da[5]`。
- thread 255 写入 `da[3]`；其他线程写 `+0.0f`。

每个线程先把自己的累加器置零，然后从 slot 0 扫到 slot 5，最终写回一次。
这样无需浮点 `atomicAdd`，也不要求调用者预清零输出。
这段顺序累加与前面的六路加法树是两个不同步骤。

padding 分支在整个 block 内一致，直接写零并返回；它不会读取无效 ID、`p` 或 `Z`，
也不会出现一部分线程等待 barrier、另一部分线程提前返回的问题。

## 实验输入与错误检查

`route_backward_core(dweights, ids, p, z, row_active)` 返回 FP32 `[T,256]`。
输入必须连续，且位于同一 NVIDIA CUDA 设备：

| 输入 | dtype / shape |
| --- | --- |
| `dweights`、`p` | FP32 `[T,6]` |
| `ids` | INT32 `[T,6]` |
| `z` | FP32 `[T]` |
| `row_active` | BOOL `[T]` |

实验 wrapper 检查 active 行的有限值、ID 范围、正分母和非负比例；输出溢出时报错。
全 padding 或 `T=0` 时返回零张量，不编译或启动本扩展。
这些同步检查会增加开销，所以 wrapper 耗时不能当作 kernel 耗时。
底层 `_out` binding 仅供测试使用，调用它需要预先保证输入数值合法。

## 在 CUDA 环境中验证

需要现有隔离环境中的 PyTorch CUDA、Ninja 和匹配的 CUDA toolkit。
不用安装整个 RL-Kernel，也不用安装 pytest。先激活该隔离环境，让其 `bin` 目录
（包括 Ninja）进入 `PATH`，再在仓库根目录执行：

```bash
# 共享机器上先确认选定 GPU 空闲，再设置对应编号。
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=2
export OMP_NUM_THREADS=1
python -c 'import torch; assert torch.version.cuda and torch.cuda.is_available()'
python -m unittest discover -s tests -p test_p3_router_backward_core.py -v
# 同时运行算术核心和前后向交接测试：
python -m unittest discover -s tests -p 'test_p3_router_backward_*.py' -v
```

必要时设置 `CUDA_HOME` 指向自己的 toolkit，并把 `TORCH_EXTENSIONS_DIR`、`TMPDIR`
指向自己的构建与临时目录。首次测试会编译独立扩展。
没有 CUDA 时 CUDA 测试会跳过；这种结果不代表 GPU 验证通过。

测试把输出转成 INT32 比较完整位模式，因此也能区分正负零。
CPU FP32 参考逐步执行运算，用于诊断；FP64 autograd 从原始分数独立求导，检查数学。
另有能区分不同加法顺序的消去样例，防止误用通用 reduction。

有限差分另外从原始 256 个 expert 分数构造 `L = sum(g * w)`，保持选择结果固定，
逐个扰动原始分数，比较 `(L(s+h)-L(s-h))/(2h)` 与反向结果。重复 expert 的多个 slot
会随同一个源分数一起变化。样例覆盖无重复、部分重复、六路同一 expert，以及
`1e-20 / 1e-3 / 1 / 1e3` 四种分数尺度；`1e-20` 能观察 epsilon 的影响。
CPU 数学检查使用两个相对步长；CUDA 输出也直接对照 FP64 前向有限差分。
这类近似导数检查使用容差，不代替 FP32 逐位检查，也不跨 Top-K 选择边界求导。

### 前向保存数据到反向的直接交接

测试入口采用与独立前后向 provider 类似的调用方式，不新增生产 autograd 封装：

```python
from examples.p3_router_backward.forward_reference import learned_forward_reference
from examples.p3_router_backward.prototype import route_backward_core

weights, saved = learned_forward_reference(scores, bias, row_active)
ds = route_backward_core(dweights, *saved)
```

`hash_forward_reference` 根据 token ID 查表，保持六个 slot 的原序及重复 expert；
`learned_forward_reference` 按 `scores + bias` 选择，分数相同时按 expert ID 升序，
归一化只读取原始 `scores`。两者按固定六项树计算 `Z`、`p`、`weights`。
测试前向可在 CPU/CUDA 上运行，保存的 `ids/p/Z/row_active` 都是 detached clone，
直接传给现有反向入口，不在反向重选专家或重新计算归一化分母。

| 检查 | 证据 |
| --- | --- |
| 前后向交接 | Hash/Learned 测试前向实际生成 saved，再执行 CUDA backward；与 FP32 参考逐位比较，并与独立 FP64 求导作容差比较 |
| 选择语义 | bias 能改变入选专家，但不进入权重公式；用测试前向的 Torch 图检查 bias 没有梯度，并用故意错误的 post-bias 权重公式验证样例能检出差异 |
| tie / near-tie | 同分按 ID 排序；第六名附近相差一个 FP32 ULP 时，入选专家和对应梯度位置正确变化；不跨不连续选择边界做有限差分 |
| 保存数据 | 修改前向源分数、bias、Hash 表、token ID 和 mask 后，saved 与反向结果不变 |
| 不变性与空输入 | 单行/批量、padding 的 NaN/非法 token ID、128/256 线程、空 batch/全 padding |
| 故障注入 | 从实际测试前向生成可观察的消去样例，检出错误的六项求和树和重复 expert 累加顺序；CUDA 扩展不可用时必须报错 |

这些 producer 仅用于 T06 测试，不代替 T03/T04 实现。`SyntheticSavedRoute` 只是
四个张量的测试容器，没有 sealed identity、checksum、weight fingerprint 或运行时
错误协议；其张量仍可被调用者修改。源数据快照测试不等于身份错配拒绝测试。
选择无梯度检查验证的是测试前向语义，生产前向与自动求导连接仍需独立验证。

### 已执行的验证：2026-10-02

在 H100 80GB、PyTorch 2.9.1+cu128、nvcc 12.8.93 上编译运行：

- 14 项测试全部通过，无跳过；包含不同大小的逐位对照、128/256 线程配置和非默认 stream。
- Compute Sanitizer 的 memcheck、racecheck、initcheck 均通过；racecheck 无 warning。
- Sanitizer 运行时关闭 PyTorch CUDA 内存缓存，让未初始化读取检查覆盖新分配的输出。

完整输出、工具版本和源文件 SHA-256 保存在
[`validation/h100-20261002.json`](validation/h100-20261002.json)。
源码变更后应重新验证；这份 10 月 2 日记录没有性能数据或官方 P3 Gate 结论。

### 已执行的验证：2026-10-03

同一 H100 / PyTorch 2.9.1+cu128 / nvcc 12.8.93 环境下，16 项测试全部通过、无跳过：
13 项 CUDA 测试和 3 项 CPU 数学检查。新增两项分别检查公式与 CUDA 输出对前向有限差分
的符合程度，覆盖上述 12 组尺度/ID 样例。日志与源码指纹见
[`validation/h100-20261003.json`](validation/h100-20261003.json)。

CUDA 源码、binding 和 Python 算术入口均未修改，仍与 10 月 2 日 Sanitizer 验证的
SHA-256 相同；本次未重新运行 Sanitizer。选择路径无梯度的集成负向测试、正式 saved
identity、T01 oracle 和训练框架验收仍待对接。

### 已执行的验证：2026-10-07

在单张 H100 GPU 0、PyTorch 2.9.1+cu128、nvcc 12.8.93 上，28 项测试全部通过，
无跳过：21 项 CUDA 测试、7 项 CPU 数学/语义检查。新增 12 项覆盖上面的直接交接、
选择语义、tie/near-tie、快照、不变性和故障注入。运行前 GPU 空闲；完整日志、环境、
源文件 SHA-256 和验证范围见 [`validation/h100-20261007.json`](validation/h100-20261007.json)。

CUDA 核心和 binding 未修改，未重新跑性能测试或 Sanitizer。当前验证覆盖测试前向
到 CUDA 反向的连接；共享 saved identity/checksum、生产 autograd、T02/模型连接、
多 GPU 和多机验证仍未完成。

## 独立性能基线

在同一 CUDA 环境、仓库根目录运行：

```bash
python -m benchmarks.benchmark_p3_router_backward_core --json /tmp/t06-baseline.json
```

默认覆盖 `T=1,16,128,512,4096`，每个形状分别测试无重复和重复 expert，比较
128/256 线程的同一 CUDA 核心及一个固定运算顺序的 PyTorch GPU 实现。
每条路径在计时之外逐位对照 CPU 诊断参考，失败即停止。JSON 保存源码 SHA-256、
运行环境、样例参数、全部计时样本、中位数和 p95。

| 字段 | 测量内容 |
| --- | --- |
| `graph_device` | 将 32 次算术调用捕获成 CUDA Graph，用 CUDA events 测 replay 后除以 32；估计摊销后的 GPU 执行时间 |
| `eager_wall` | 一次普通调用加完成同步的实际耗时，包含 CPU 调用开销；CUDA `_out` 复用输出，Torch 对照分配输出 |
| `harness_wall` | 当前 Python 实验入口的实际耗时，包含输出分配、多次输入检查与同步 |

Graph 反复使用同一组驻留输入，属于缓存已预热的微基准；CPU 分配和调用开销不在
`graph_device` 内，但 GPU 清零、算术和写回都在。它不是单次 eager 请求延迟，也不表示
正式同步 ABI 可以直接被 capture。p95 是样本的 nearest-rank 分位数；Graph 样本本身
已经是每次 replay 的平均值，不代表单个 kernel 的尾延迟。

Torch 对照用逐 slot 的 gather/add/scatter 保持重复 ID 的累加顺序，不使用浮点 atomic。
它是局部算术的 eager 分解，不是 Megatron、Miles 或 Vime 原生 Router 的性能基线。
不根据两者的比值宣称真实训练加速。

可单独检查混合 padding 路径（无效行写入 NaN/非法 ID，检验屏蔽行为）：

```bash
python -m benchmarks.benchmark_p3_router_backward_core \
  --tokens 1 33 --padding-fraction 0.5 --warmup 3 --samples 5 --graph-batch 8 \
  --json /tmp/t06-padding-smoke.json
```

### H100 实测：2026-10-03

两轮使用相同设置，第二轮加 `--threads 256 128` 反转配置顺序。以下为无重复 ID
样例的中位数区间（单位均为微秒），区间表示两轮结果，不是置信区间：

| T | 128 线程 Graph GPU | 256 线程 Graph GPU | 256 线程完整实验入口 |
| --- | ---: | ---: | ---: |
| 1 | 1.649–1.656 | 1.620–1.623 | 473.454–504.567 |
| 16 | 1.711–1.712 | 1.677–1.678 | 474.598–492.019 |
| 128 | 1.853–1.858 | 1.833–1.836 | 485.118–505.147 |
| 512 | 1.933–1.941 | 1.930–1.954 | 490.697–510.537 |
| 4096 | 3.967–3.975 | 4.678–4.694 | 522.030–536.444 |

重复 ID 的结果相近。T=4096 时，128 线程的 Graph GPU 时间约低 15%；小 T 没有同样
优势，因此保留默认 256 线程。CUDA `_out` 普通调用加同步约 10–14 微秒；完整入口
约 0.47–0.54 毫秒。正式接入时应进一步分析值检查、分配和同步成本，不能将两个不同
计时口径直接当作加速比，也不能直接删除合同要求的检查和同步。

两轮各 30 个配置及另一次混合 padding smoke 均通过逐位对照；完整样本保存在
[第一轮](validation/benchmark-h100-20261003-run1.json)、
[反转顺序复测](validation/benchmark-h100-20261003-run2.json)、
[padding smoke](validation/benchmark-h100-20261003-padding.json)。
这些是合成数据的局部基线，不是模型级收益或 P3 集成验收结论。

## 下一步接入边界

按团队共享接口继续对接 `hash_route_bwd` / `learned_route_bwd`：

- 消费正式 sealed saved，验证身份、checksum、版本和权重来源。
- 接入指定的 device status、invocation echo、provider readback 协议。
- 使用团队共享参考实现、recorded fixtures 和 `check_p3` 做验证。

当前 raw-tensor 入口不承担这些职责，也没有伪造对应 schema 或 PASS 状态。
T02 的 `ds → dz`、gate GEMM 反向、专家计算及多卡通信不在这个核心内。
