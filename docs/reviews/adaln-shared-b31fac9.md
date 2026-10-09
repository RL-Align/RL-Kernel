# AdaLN shared：工作区审查与 TDD 证据（2026-10-09）

## 剩余缺口与本次执行范围

用户已授权继续按 `/tdd` 改实现、补测试；沿用公开 op 调用、autograd、registry、gtest
接口，不要求新的接口确认。不提交/推送，不启动 H100 长测试、巨大输入或无法估计耗时的构建。
下面是实施前待办；其最新完成状态和本次运行证据记录在下一节。
本文件下方第一轮审查及 followup 文档都属于历史源码快照，旧运行表不能算当前版本证据。

| 缺口 | 修补动作 | 实施前结论 |
| --- | --- | --- |
| CUDA B65536 超出 backward grid.y | Python/C++ 声明并检查 B≤65535，补65535/65536边界测试，launch前明确拒绝 | 源码PARTIAL，当前native运行UNPROVEN |
| CUDA int32寻址范围无检查 | 在复制/launch前检查索引乘积，Python/C++双层保护；巨大形状使用meta输入测试，不分配几GiB张量 | 源码PARTIAL，运行UNPROVEN |
| shared eps和nonfinite支持域未明确 | 拒绝FP32转换后Inf或正eps下溢到0，保留合法subnormal与eps=0；补公开反例并记录nonfinite LN范围 | PARTIAL |
| geometry×batch/padding交叉不完整 | 补全6种Triton配置及3种CUDA threads，比较默认B1同样本y/gate/dX/dMod bytes | PARTIAL |
| GPU两stream及第二triple覆盖不足 | 扩展已有小形状公开测试到各后端，独立FP32梯度、unused半区0、另一stream不受影响 | PARTIAL |
| H4095/4096/4097 native边界 | 补direct op与registry正负边界，两dtype | UNPROVEN |
| 当前源码/native binary闭环缺失 | 保存源/产物hash；仅在能够可靠估计快速完成时构建，否则提供构建与验收命令 | UNPROVEN |
| H100真实图像、安全性、性能 | 保留三真实S、racecheck/memcheck、benchmark和pinned环境命令与清单，不自行启动 | UNPROVEN |

## 本次 TDD 完成状态与当前源码证据

以下均通过公开 op、autograd、registry 或 gtest 观察，没有新增生产 helper mock。
已复用旧检查器、固定字节参考、独立 FP32 CPU LayerNorm/autograd 和模型六块切片用例。
比较点仍为 `b31fac9`，HEAD仍为 `120f4de7274798c6310d87799ebb6538f2f8f531`；
范围包含当前 tracked diff 与未跟踪的两个测试文件和审查文档。未提交或推送。

实现采用明确的有界拒绝：native 支持 B≤65535、H≤4096、每个输入 numel≤2^31-1。
Python 在 `.contiguous()` 前校验，C++ 在 narrowing/launch 前重新校验；没有扩大支持域，
因此不需要实际分配约4GiB的溢出反例。registry只按H选择后端，B/numel超限在公开调用
入口清楚拒绝，不算隐含fallback。meta测试验证的是公开Python形状preflight，不是native内核。

shared eps 检查 round-to-FP32 后仍有限，且正值不变成0；合法subnormal和eps=0保留。
nonfinite LN按FP32公式传播，eps=0且方差0可以产生NaN；不承诺LN运算的NaN payload
或非有限运算域的跨后端字节相等。gate/dgate无算术，仍按原dtype位透传。
这不是给整个nonfinite输入域宣称PROVEN，也没有改变独立indexed/select01合同。

| 缺口 / 可能错误的实现 | 最小反例与关键断言 | 已完成的修改 / 测试 | 当前结论 |
| --- | --- | --- | --- |
| B65536直接进入backward grid.y | B65536,S1,H1；公开调用必须在launch前明确拒绝；B65535的y/gate/dX=0，dMod gate=1 | Python/C++新增B≤65535校验；`shared_bounds::rejects_batch_above_grid_limit`及`maximum_batch_forward_backward` | Python拒绝 **PROVEN**（meta及SM86 CUDA tensor）；C++产物与合法最大B数值 **UNPROVEN**，测试已准备 |
| x地址或seq narrowing以int溢出，先连续化会尝试几GiB分配 | meta x形状[1,524289,4096]、[1,2^31,1]；必须报int32 indexing错误，无实数据分配 | Python/C++检查x和modulation numel≤2^31-1；`rejects_int32_address_overflow_without_allocating` | Python metadata拒绝 **PROVEN**；C++重新构建后的运行 **UNPROVEN** |
| double eps有限但FP32为Inf/0 | 1e39、1e-46必须明确ValueError；1e-40、2^-149、FP32最大有限值必须保持y0、dX=±rstd、dMod=[1,-1,0,0,0,0] | 共享Python校验用stdlib struct检查FP32舍入，C++也检查；公开`eps_overflow`及扩展`subnormal_eps` literal测试，两dtype | 三Python公开后端拒绝 **PROVEN**；合法边界CPU/SM86 Triton **PROVEN**；native合法eps数值 **UNPROVEN** |
| 把nonfinite LN误当finite验收；gate/VJP因NaN LN串值 | x=[NaN,1]或[Inf,1]，y/dX/dscale应NaN；dshift=[1,-1]、gate=[2,-0]、dgate=[2,-3]独立保持 | 声明nonfinite传播和qualification范围，新增`nonfinite_activation_propagates_nan_and_preserves_gate_vjp` | CPU/SM86 Triton指定反例 **PROVEN**；一般NaN payload与strict非有限域不在资格声明中 |
| 只单独比较geometry或relocation，错过组合路径 | H3072：默认B1,S3基准→B3每个position、S3/5、strided x和6H调制；y/gate/dX/dMod bytes一致、unused半区0 | `h3072_relocation_with_all_launch_configurations`覆盖全部6种Triton warps/tile及3种CUDA threads，两dtype | 全部Triton组合 **PROVEN**（12参数例）；native组合测试已写、运行 **UNPROVEN** |
| GPU第二triple/两stream不独立；只验梯度会漏错shift | image S5/text S3，各取前/后3H；另一个stream梯度不变、unused半区0；错误候选y+1仍有正确VJP | 原`gate_only_and_separate_stream_chunks`扩展到三后端，独立CPU gold的forward+backward、gate bytes及另一stream隔离；错误shift候选旧断言实际放过，新断言拒绝 | CPU/SM86 Triton **PROVEN**（8参数例，均遍历两stream）；native测试已写、运行 **UNPROVEN** |
| H上限或零填充off-by-one | H4095/4096：两dtype y/gate/dX/dMod对独立CPU数学及固定参考bytes；H4097直接拒绝、registry明确fallback | 新`hidden_boundary_matches_independent_math_and_fixed_bytes`及`shared_bounds` direct/registry三个H边界测试 | CPU/SM86 Triton合法H数学 **PROVEN**；Python direct拒绝及registry **PROVEN**；native合法H数值 **UNPROVEN** |
| 用旧_C或语义fingerprint证明当前C++ | 当前CUDA源码已有校验变更，现存_C未重建，不能因symbols/constructor可用就算通过 | 记录新source hash与旧binary hash，准备fresh build/验收命令，未调用未绑定当前源码的native数值路径 | 当前native内核资格 **UNPROVEN** |
| 三真实图像S、shared race、OOB、claimed平台性能缺当前证据 | S4096/6889/6032,H3072、两dtype、两个GPU实现；racecheck/memcheck必须0 errors；benchmark完整数据 | 更新04验收脚本、06racecheck脚本及WS1/CPU CI入口接入新测试；准备下面H100命令，未启动CI或长测试 | H100/真实图像/sanitizer/benchmark **UNPROVEN** |

当前合法eps的手算rstd literals由独立stdlib math+IEEE FP32舍入得到，分别为
`1.000002658868784e20`、`2.671373844909537e22`、`5.421011508662376e-20`。
隐藏边界的数学参考使用独立CPU FP32 leaves；固定CPU参考只作为字节/归约合同参考，
不将其当独立数学gold。三种比较不能互相替代。

### 当前版本运行表

下面最终suite和错误候选重跑对应本节末尾hash；red/局部green是本次TDD的中间源码证据。
旧 `.ninja_log` 只用于评估构建耗时：full build历史上超过17分钟，不作为正确性运行证据。
没有自行启动full build，也没有为了构建安装/下载依赖或索取新权限。
本机GPU检查沿用已有Python执行权限，NVIDIA RTX3050Ti/SM86、PyTorch2.12.1+cu126、
Python3.12.12、Triton3.7.1；未声明SM90或当前native binary资格。

| 本次检查 | 结果 / evidence |
| --- | --- |
| B限制red→green | meta公开入口1 failed→1 passed；[red](/tmp/rlk-adaln-close-batch-red.log)、[green](/tmp/rlk-adaln-close-batch-green.log)；[SM86公开拒绝](/tmp/rlk-adaln-close-batch-gpu-green.log)1 passed。没有故意启动非法native launch |
| int32范围red→green | 2 failed→bounds 3 passed；[red](/tmp/rlk-adaln-close-int32-red.log)、[green](/tmp/rlk-adaln-close-int32-green.log)，全是小输入/meta，无巨大分配 |
| eps范围red→green | CPU4 failed→4 passed；[red](/tmp/rlk-adaln-close-eps-red.log)、[green](/tmp/rlk-adaln-close-eps-green.log)；[合法/非法eps小形状CPU+Triton](/tmp/rlk-adaln-close-eps-boundaries.log)20 passed |
| 第二triple错shift断言red→green | [错误候选旧断言SURVIVED，exit1](/tmp/rlk-adaln-close-shift-mutant-red.log)→[新断言REJECTED，exit0](/tmp/rlk-adaln-close-shift-mutant-green.log)，不是生产算子数值错误的声明 |
| 全Triton geometry×topology | [log](/tmp/rlk-adaln-close-all-triton-launches.log)12 passed/9.68s；全部6配置×2dtype，每例含B3的每个position与S3/5 |
| 两stream/两triple | [log](/tmp/rlk-adaln-close-streams-green.log)8 passed/5.43s；含独立forward及backward |
| H4095/4096合法边界数学与bytes | [log](/tmp/rlk-adaln-close-hidden-math.log)8 passed/25.65s |
| B/H/寻址公开preflight及registry | [log](/tmp/rlk-adaln-close-hidden-preflight.log)8 passed/8.73s；未运行maximum_batch数值例 |
| nonfinite传播/独立gate VJP | [log](/tmp/rlk-adaln-close-nonfinite.log)8 passed/5.15s；只断言NaN传播，未将NaN算术payload计为byte资格 |
| 最终CPU/checker/preflight | [log](/tmp/rlk-adaln-close-final-cpu.log)：**113 passed，139 deselected，10.84s，0 skip/xfail** |
| 最终CPU+SM86 Triton+CUDA公开拒绝 | [log](/tmp/rlk-adaln-close-final-gpu.log)：**180 passed，91 deselected，11.00s，0 skip/xfail**；排除native数值、真实图像、旧完整geometry/sanitizer |
| 最终错误候选重跑 | [6个原候选](/tmp/rlk-adaln-close-mutants-six-final.log)、[8个后续候选](/tmp/rlk-adaln-close-mutants-eight-final.log)、[新shift候选](/tmp/rlk-adaln-close-shift-mutant-final.log)：**15/15拒绝**；[新候选脚本](/tmp/rlk_adaln_close_shift_mutant.py) |

静态检查通过：7个相关Python文件ruff，新增/本次修改文件isort及Black检查，
三个修改shell的`bash -n`、CPU CI YAML解析、`git diff --check`。

```bash
# 当前CPU命令：bounds是metadata/preflight，不运行native内核
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_adaln_gtest.py tests/test_adaln_shared_bounds.py \
  tests/test_op_checks.py tests/test_operator_inputs.py -q -rs \
  -k '(not triton and not cuda) or (shared_bounds and not maximum_batch)'

# 当前SM86短检查。eps_overflow和shared_bounds仅公开拒绝/registry，不调用native数值路径
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RL_KERNEL_REQUIRE_EXT=1 \
TRITON_CACHE_DIR=/tmp/rlk-adaln-review-triton .venv/bin/python -m pytest \
  tests/test_adaln_gtest.py tests/test_adaln_shared_bounds.py tests/test_adaln_modulation.py \
  -q -rs -k '(not cuda and not real_image and not gpu_padding and not launch_geometry) or eps_overflow or (shared_bounds and not maximum_batch)'
```

### 当前源码 SHA256

| 文件 | SHA256 |
| --- | --- |
| 未跟踪 `tests/test_adaln_gtest.py` | `db9617770d5637a4aa35e4d0688bb3e3e3abdb7c127f5d04288ab12ce6237119` |
| 未跟踪 `tests/test_adaln_shared_bounds.py` | `97c5a99ae6f73545304114786e935d30f21af8a4bbe73984c296972ae997b0ba` |
| CUDA wrapper | `e5a52b8e29c6be131176e0fdad64aa259cc780b5a4c7529bad98901b7ec53797` |
| PyTorch AdaLN | `96707e412a15790962a15a939ecc835eaf363829c515304a2467db2a294fb5c6` |
| Triton AdaLN | `ef2b13c8e13da8d324b160489efad4b2080d3a1058cf3b58d566c2bc92a3b4a2` |
| 当前 CUDA source | `aae87fb84f9b41954a727aa09e926ddfc4deb7c82130aca508607248d0670c05` |
| gtest checker | `70f9c648b5da5ace001d295569c4b05a15d2e630e84e1870b5734a5377d76b39` |
| gtest specs | `b78d15322199b4986f19697c7987dea879238b7d8b14fe2f1b05fc3fe0268e79` |
| 现存旧_C（没有本次构建来源证明） | `d7b91b28b81c7f53db47e757eab9305f617dbc5649666daa3237e9946eb350b6` |

### 待用户执行的native/H100检查，未启动

下方历史报告的H100构建命令仍是待执行方案；必须先从当前dirty/untracked源码重新构建，
不能复用上表旧_C。源码快照要包含两个未跟踪测试及本文件。建议保存build log、完整
compiler flags、源码/产物hash、Python/PyTorch/Triton/CUDA/driver/GPU信息，再执行：

```bash
# 仅在当前源码已fresh build的准备好环境中运行；这里未执行
RL_KERNEL_REQUIRE_EXT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests/test_adaln_gtest.py tests/test_adaln_shared_bounds.py \
  -q -rs -k cuda

# 完整真实形状与数值入口；04已包含新bounds，06包含完整native launch×topology
bash scripts/adaln_cuda_acceptance/04_test_adaln_cuda.sh

# 支持域已限定，巨大输入应在入口拒绝；memcheck检查真实支持边界，不分配4GiB反例
RL_KERNEL_REQUIRE_EXT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  "${CUDA_HOME:?set CUDA_HOME}/bin/compute-sanitizer" --tool memcheck --error-exitcode 7 \
  .venv/bin/python -m pytest tests/test_adaln_gtest.py tests/test_adaln_shared_bounds.py \
  -q -rs -p no:cacheprovider \
  -k 'cuda and (hidden_boundary or maximum_batch or all_launch_configurations)'

bash scripts/adaln_cuda_acceptance/05_benchmark_adaln_cuda.sh
```

验收必须覆盖：native全部canaries、最大B65535与超限65536、H4095/4096/4097、三threads、
两dtype、两stream/两triple；Triton全六配置；S4096/6889/6032真实图像；racecheck/memcheck
0 hazards/errors；benchmark有每个dtype/S/backend的forward及forward+backward数据。
要求零skip/xfail、native无fallback，source/binary在验收期间不变；配置和CLI报告不是运行证据。
一般NaN算术payload、ROCm或其他未固定Triton环境不在本机资格声明中。

## 第一轮审查及运行记录（历史源码快照）

> 下方内容及[第二轮报告](adaln-shared-b31fac9-followup.md)保留中间源码证据。
> 它们的运行表和hash不代表当前工作区；最新状态以上面“本次TDD”章节为准。
> H100/native构建与长验收没有在本次执行。

规格：[RL-Align/RL-Kernel #386](https://github.com/RL-Align/RL-Kernel/issues/386)。
比较点 `b31fac988faae9de71ab1f683e4a5c9cc8578b84`，HEAD
`120f4de7274798c6310d87799ebb6538f2f8f531`。范围是 **比较点到实际工作区**，
使用 `git diff b31fac9` 加 `git ls-files --others --exclude-standard`，包括索引、
未提交修改和未跟踪 `tests/test_adaln_gtest.py`。三点 HEAD diff 不代表本次范围。

开始时工作区有 11 个已修改文件和上述未跟踪文件；其既有修改均保留。
已读比较点之后五个提交、CUDA/PyTorch/Triton 实现、bindings、registry、
gtest reference/inputs/checker、两个测试文件、benchmark、acceptance scripts、CI 和文档。
indexed select01、projection、gate residual、最终 AdaLN、完整 MMDiT/WS2 不属于本次 shared 实现验收。

这里 **PROVEN** 只表示明确列出的当前版本、后端与有限反例已经运行通过；
**PARTIAL** 表示有实现或部分证据，仍有未运行的平台/几何/验收项；
**UNPROVEN** 表示没有当前工作区的有效运行证据。没有用旧日志填补当前证据。

## Standards

1. **P1，已修复：BF16 gtest 参考梯度被再次舍入成 BF16。**
   [gtest 规范](../contributing/gtest-usage.md)要求独立 FP32 reference。
   `GtestAdaLNReference.forward_fp32` 虽在内部转 FP32，但旧 checker 给 gold 克隆的
   求导叶子仍为 BF16；autograd 返回 BF16 dX/dMod，报告却标 `fp32_reference`。
   新增公开 `run_operator_suite` 断言所有 output/gradient checks 的 `gold_dtype`
   都为 `torch.float32`；BF16 H=7 的运行实际失败（1 failed、1 passed，exit 1）。
   修复为 `OperatorSpec/OperatorCase.gold_input_dtype` 显式启用 FP32 gold leaves；
   只转换浮点张量，candidate 和其他算子的默认输入策略保留。随后相关 47 项检查通过。
   这修复的是验证参考，未假称发现 CUDA 算术错误。
2. **判断性异味：registry 的两条 trace 分支重复若干精度/算法字段。**
   可能导致未来元数据漂移；目前无实际错误证据，没有为此增加抽象或重构。

另修复 focused shared CPU 测试无条件导入 Triton 的问题，并将两个 GPU 测试的
Triton import 移到 CUDA 可用检查之后。CPU 原本不需要这个可选依赖。

Standards：1 个已修复规范问题、1 个判断性异味；最严重项是参考梯度精度声明错误。

## Spec

原有 dtype/finite/容差断言不足以证明 BF16 输出边界；H=17/32 的 relocation/padding
也不能覆盖 H=3072 的批位置。现有 H=3072、B=2 geometry 测试比较的是各自 CPU
reference，没有把同一个样本从 B=1 移到 B=3 的每个位置。两项均已补公开入口反例。
另有明确的独立 FP32 gold 梯度错误，已按上述 red→green 修复。
shared 非法输入的反例也已补齐；当前 native binary、真实图像尺寸、racecheck、H100
和当前 benchmark 仍没有本轮运行证据。

Spec：3 个核心精度/拓扑缺口、1 组非法输入覆盖缺口、1 组平台验收证据缺口；
最严重项是原 gtest 的 FP32 gold 梯度并非 FP32。没有合并或重排 Standards 结论。

## 要求、独立参考、反例与关键断言

位置简称：`native` = `ops/pytorch/norm/adaln_modulation.py`；`cuda` =
`csrc/cuda/adaln_modulation.cu`；`triton` = `ops/triton/norm/adaln_modulation.py`。
新测试均在 [test_adaln_gtest.py](../../tests/test_adaln_gtest.py)，旧断言保留在
[test_adaln_modulation.py](../../tests/test_adaln_modulation.py)。

| 要求 / 当前实现与参考 | 具体可能错误的实现 | 最小反例 | 应失败的断言 / 测试 | 当前结论 |
| --- | --- | --- | --- | --- |
| 无 affine LayerNorm，shift/scale/gate 顺序，默认 eps=1e-6；三后端均按此实现；独立 CPU `F.layer_norm` + autograd | 用 RMSNorm，交换 shift/scale，gate 混入 y | x=[1,3]、shift=[.5,-.5]、scale=[1,0]、gate=[2,3]、eps=0；y=[-1.5,.5] | 旧定值 forward/backward；独立 shared H=7/3072、非均匀 dy/dg 的 accuracy 断言 | **PROVEN** CPU/SM86 Triton 的已跑小形状；CUDA **UNPROVEN** |
| y、gate 及 dX/dMod 独立 FP32 CPU reference；现已启用 FP32 gold leaves | 内部转 FP32但向 BF16 叶子求导，或求导后再 `.float()` 掩盖已舍入的参考 | BF16 B=2,S=3,H=7 gtest case | 所有四项 check 的 `gold_dtype == torch.float32`；独立 FP32 leaves 的 accuracy；旧错误实际 red、新 checker green | **PROVEN** 当前 CPU checker；native 平台验收另计 |
| BF16 normalization 中间量不得提前舍入 | 把 norm 转 BF16 再做 scale/shift | BF16 x=[-1,1]、shift=-1、scale=0、默认 eps；正通道应为 -2^-21，早舍入为 0 | `bf16_forward_casts_only_at_output` 对手算 literal 做 uint8 bytes 比较 | **PROVEN** CPU/SM86 Triton 该反例；完整内部精度政策 **PARTIAL** |
| `1+scale` 和 affine 保持 FP32 到输出边界 | BF16 的 `1+1/256` 先舍入成 1 | 同上，scale=1/256；正通道 BF16 应为 1/256，错误结果约为 -2^-21 | 同一 forward canary 的第二个参数例；不能用容差替代 | **PROVEN** CPU/SM86 Triton 该反例；CUDA **UNPROVEN** |
| backward dnorm、partial、token accumulators FP32 | BF16 dnorm、BF16 partial，或逐 token BF16 fold | B=1,S=3,H=2，x 每行[-3,3]，eps=7，scale=1/256，dy 首通道[256,1,-256] | `backward_keeps_partials_and_accumulators_fp32`：dshift=[1,0]，dscale=[-.75,0]；dX 按独立手算系数1799/32768；gate VJP=[-2,.5] | **PROVEN** CPU/SM86 Triton 该反例；CUDA **UNPROVEN** |
| BF16 backward 保存的 norm/rstd/partials 不能成为隐含额外 cast | 只在 BF16 backward 保存量转 BF16，forward 完全正确 | 相同 BF16 量化输入、dy/dg，H=7/3072，分别以 BF16 和提升后的 FP32 调用公开 op | `bf16_vjp_matches_fp32_with_one_final_cast`：y/gate/dX/dMod 必须等于 FP32 调用结果最终 cast 的 bytes；独立 math 另测，避免把 differential 当独立数学 gold | **PROVEN** CPU/SM86 Triton 的固定 seed；更广输入域 **PARTIAL** |
| H 固定 lower+upper pairwise tree，非二次幂零填充 | 改成 ascending left fold、按线程分组求和 | FP32 H=4 x=[2^24,1,-2^24,1]，pairwise mean=.5，left-fold mean=.25 | `hidden_reduction_uses_declared_pairwise_order` 的手算 FP32 literal bytes；旧 H=7/3072 accuracy/byte tests | **PROVEN** CPU/SM86 Triton 该反例；其他重排与 CUDA **PARTIAL** |
| dMod 按逻辑 token 递增 FP32 fold；无 atomic/Split-K/Stream-K | 用 pairwise token tree、调换 token/分块重关联 | S=4,H=2 x=[-1,1]，dy 首通道[2^24,1,-2^24,1] | `modulation_gradient_uses_ascending_token_order`：dshift=1、dscale=-1；改树可能得到2/-2 | **PROVEN** CPU/SM86 Triton 该反例；禁止算法同时有源码证据，native **UNPROVEN** |
| batch size、batch position、padding 下有效样本 bytes 不变 | H=3072 路径始终读 mod[0]、错误 batch stride、跨样本共享 dMod | B=1,S=3,H=3072 基线→B=3 全部位置；S=3/5；其他样本不同 x/m/dy/dg，padding dy=0 | `h3072_sample_bytes_survive_batch_positions_and_padding`：y/gate/dX/dMod uint8 相等；包含非连续 x 和六块 view；未使用调制块梯度为0 | **PROVEN** CPU/SM86 Triton BF16/FP32 本次矩阵；更大 B/S 与 CUDA **PARTIAL** |
| launch geometry / tiling 不改 bytes | H tree 随 CUDA threads、Triton warps/tile 改变；跨 warp 共享内存未同步 | H=3072,B=2,S=5，同一输入换 threads128/256/512，warps4/8、tile64/128/256 | 旧 `launch_geometry_preserves_bytes` 比较 y/gate/dX/dMod 对固定 CPU reference | 测试已存在，本轮未运行 geometry sweep；**UNPROVEN** 当前完整 sweep |
| 两 stream 独立调制、attention/MLP 两 triples、gate downstream gradient、strided 模型 view | 串用 image/text modulation；用错第二 triple；gate detached；漏掉 contiguous 复制 | image S=5、text S=3，各自 B=2,H=7、6H 调制；分别取前/后3H，非均匀 dy/dg 和 gate-only | `gate_only_and_separate_stream_chunks` 已扩展到两 triples，独立 CPU accuracy、未使用半区0、另一 stream 无梯度；H3072 topology 的 strided view | **PROVEN** CPU 两 stream/两 triples；SM86 Triton H3072 strided 第一 triple **PROVEN**；GPU 第二 triple **PARTIAL** |
| unsupported shared 输入清楚拒绝；CUDA H<=4096，几何参数有界 | empty/mixed shape 被 launch；FP16 悄悄进入；NaN eps 未拒绝；H>4096 native误选 | 任一空 B/S/H、rank2、错3H、mixed dtype/device、FP16、eps<0/NaN/Inf | 新 `shared_rejects_unsupported_inputs` 和 `shared_rejects_mixed_devices`；旧 large-H registry test；constructors 源码验证 launch 参数 | **PROVEN** CPU/SM86 Triton 输入矩阵；CUDA H/geometry 的当前 runtime **PARTIAL** |
| 实际 backend/fallback、算术 policy 与 kernel identity 可追踪 | native缺失但谎称CUDA；将语义version当编译binary hash；fallback算作strict成功 | native symbol missing、Triton missing、所有后端missing | 旧/新 registry resolution tests；当前 preflight selected native但仅说明构造/符号可用；actual execution须结合direct candidate+source/binary manifest | **PARTIAL**；语义 fingerprint 不绑定二进制，不能凭它证明当前 native |
| CUDA shared reduction race-free | 读shared[0]后其他warp立即重用buffer | H=64 或3072、多warp fwd/bwd | `compute-sanitizer --tool racecheck --error-exitcode 7`；数值 bytes 相等不是 race assertion | **UNPROVEN** 当前 binary；旧 zero-hazard 日志不计 |
| 三参考图像尺寸与短 synthetic，forward/backward、两 dtype、两个GPU实现 | S=6889 tail越界、dMod 漏 token、仅注册优先backend正确 | S={4096,6889,6032},H=3072，随机 dy/dg | 旧 real-image tests 同时独立 FP32 CPU accuracy、fixed-reference bytes、CUDA↔Triton bytes | synthetic 已有当前证据；三 full-image 当前 **UNPROVEN**，H100命令如下 |
| 当前 claimed CUDA platform benchmark、pinned环境 | 只用旧binary/旧timings，CUDA skip却当benchmark完成 | 当前源码构建→两dtype三S，fwd及fwd+bwd | benchmark backend标签、无native skip、完整timings/peak memory、build/runtime/source manifest | 实现存在，当前运行 **UNPROVEN** |
| 无TF32、fast math、compiler-dependent重关联；unsupported严格配置拒绝 | 编译开启fast math仍注册native，或Triton融合乘加 | fast-math-on build / 不同编译配置 | CUDA RN intrinsics、宏#error、setup条件排除symbols；Triton enable_fp_fusion=False；必须验独立build/provenance，不用普通数值测试冒充compiler proof | **PARTIAL** 静态证据；替代build/H100 profile **UNPROVEN** |

没有把原本正确的算子改成迎合测试的实现。补测时两个期望问题已纠正：大值
hidden-tree 反例的标准化值应为 ±sqrt(2)，不是 ±1/sqrt(2)；padding 零 VJP 可为
负零，规格要求有效行 bytes 不变，并未要求 padding 的零必须为正零。
有效行、gate 和有效 dMod 的原始字节比较没有放宽。

## 本轮运行证据

所有结果来自本次工作区。源文件与测试的 SHA256 见下，未提交或推送。

| 运行 | 命令 / artifact | 结果 |
| --- | --- | --- |
| 初始 red | `pytest tests/test_adaln_gtest.py -q -x -k outputs_and_gradients` | BF16 gold_dtype 断言失败，exit 1；1 failed、1 passed、6 deselected |
| 修复后 checker 回归 | `pytest tests/test_adaln_gtest.py tests/test_op_checks.py tests/test_operator_inputs.py -q -rs`，当时尚未追加其余 canaries | 47 passed / 19.19s |
| 最终 CPU/checker | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_adaln_gtest.py tests/test_op_checks.py tests/test_operator_inputs.py -q -rs -k 'not triton and not cuda'`；[/tmp CPU log](/tmp/rlk-adaln-review-cpu-20261009.log) | 68 passed、41 deselected / 16.94s；无 skip |
| 最终 CPU+SM86 Triton | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TRITON_CACHE_DIR=/tmp/rlk-adaln-review-triton .venv/bin/python -m pytest tests/test_adaln_gtest.py tests/test_adaln_modulation.py -q -rs -k 'not cuda and not real_image and not gpu_padding and not launch_geometry'`；[/tmp GPU log](/tmp/rlk-adaln-review-gpu-20261009.log) | 80 passed、48 deselected / 32.06s；无 skip。此选择不包含 native CUDA 或 geometry sweep |
| 断言敏感性 | [/tmp mutation probe](/tmp/rlk_adaln_review_mutations.py)，[/tmp log](/tmp/rlk-adaln-review-mutations-20261009.log) | 6/6 公共接口模拟错误被拒绝：early norm、early affine、BF16 token fold、token reorder、hidden left fold、错误 batch 调制 |
| CPU 可选依赖隔离 | [/tmp probe](/tmp/rlk_adaln_review_cpu_only.py)，[/tmp log](/tmp/rlk-adaln-review-cpu-portability-20261009.log)；拒绝所有 Triton imports，再运行 focused shared PyTorch independent cases | 4 passed、52 deselected / 3.44s |
| 静态检查 | ruff、isort、Black check：新增测试与 checker/spec；`git diff --check`；两个修改 shell 的 `bash -n` | 全部通过；Black 按 py310、line-length100 逐文件检查 |

Mutation probe 用错误候选替换测试的公共 callable，不修改生产 helper，也不是硬件证据。
独立 LayerNorm/autograd 检验数学正确性；同值 FP32/BF16 differential 检验 cast 边界；
literal 与 uint8 比较检验特定舍入/归约错误。三者分工，不能互相替代。

本机环境：Python 3.12.12、PyTorch 2.12.1+cu126、CUDA runtime 12.6、Triton 3.7.1，
RTX 3050 Ti Laptop / SM86。没有本轮 driver/toolkit build qualification 声明。
native `_C` 已存在但没有与当前源码绑定的 build artifact，本轮没有执行其数值测试。

| 文件 | SHA256 |
| --- | --- |
| `tests/test_adaln_gtest.py`（未跟踪文件，已纳入） | `9fea7d7ae184b74641ef8368734a446169c05d7063259956eb42d7a5d28e77f2` |
| `gtest/op_checks.py` | `8b1ff607983d40d60c3a21e03302884228c75ec1beb56b20dccf25215f9c18c1` |
| `gtest/operator_specs.py` | `93c44ff298f2896de40c0b43d9fc5314ef377f0d7c2294cb2420826116ac3917` |
| PyTorch AdaLN | `682a32ffad1ecbbdb28b7bdbb25413acabdde51c71c784656e2d272e16b19003` |
| Triton AdaLN | `7b08d9b4bc047a342875040372bd128f2cbc66eb02779e1c9f9520160b28f781` |
| CUDA source | `27074706332b1db32236b6cbb225ee99514938f38d1996ef4812e221d857e80d` |
| 现存 `_C.cpython-312-x86_64-linux-gnu.so` | `d7b91b28b81c7f53db47e757eab9305f617dbc5649666daa3237e9946eb350b6`；仅记录身份，非当前 native 验收 |

tracked diff 快照：[patch](/tmp/rlk-adaln-review-tracked-b31fac9.patch)；
未跟踪测试快照：[test source](/tmp/rlk-adaln-review-test_adaln_gtest.py)；
规格快照：[issue JSON](/tmp/rlk-adaln-review-issue386-20261009.json)。
这些 `/tmp` 文件应随验收 artifact 保存，不能假设长期存在。

## H100：只准备命令与验证清单，未执行

在用户准备好的、依赖已经安装且版本已固定的 H100 checkout 中执行。以下包含可能长时间
运行的构建、真实大图、racecheck 与 benchmark；本轮均没有启动。
H100 native 必须重新构建，不能沿用本机 SM86 `.so` 或仅按符号可用判定成功。

```bash
set -euo pipefail
export ADALN_EVIDENCE=/tmp/adaln-h100-current-worktree
mkdir -p "$ADALN_EVIDENCE"
git rev-parse HEAD > "$ADALN_EVIDENCE/head.txt"
git status --short --untracked-files=all > "$ADALN_EVIDENCE/status.txt"
git diff b31fac9 --binary > "$ADALN_EVIDENCE/tracked.patch"
git ls-files --others --exclude-standard > "$ADALN_EVIDENCE/untracked-files.txt"
cp tests/test_adaln_gtest.py tests/test_adaln_shared_bounds.py "$ADALN_EVIDENCE/"
cp docs/reviews/adaln-shared-b31fac9*.md "$ADALN_EVIDENCE/"
sha256sum csrc/cuda/adaln_modulation.cu \
  rl_engine/kernels/ops/{pytorch,triton,cuda}/norm/adaln_modulation.py \
  rl_engine/kernels/gtest/{op_checks,operator_specs,operator_inputs}.py \
  rl_engine/kernels/registry.py tests/test_adaln*.py setup.py csrc/ops.cpp \
  > "$ADALN_EVIDENCE/sources.sha256"
nvidia-smi > "$ADALN_EVIDENCE/nvidia-smi.txt"
nvcc --version > "$ADALN_EVIDENCE/nvcc.txt"
.venv/bin/python -m pip freeze > "$ADALN_EVIDENCE/packages.txt"

# 建议使用与项目兼容且已固定的 CUDA toolkit；不在这里安装/下载依赖。
FORCE_CUDA=1 KERNEL_ALIGN_USE_FAST_MATH=0 KERNEL_ALIGN_FORCE_SM90=1 \
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  .venv/bin/python setup.py build_ext --inplace --force \
  2>&1 | tee "$ADALN_EVIDENCE/build.log"
sha256sum rl_engine/_C*.so > "$ADALN_EVIDENCE/binary.sha256"

export RL_KERNEL_REQUIRE_EXT=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export RLK_ADALN_REAL_SHAPES=1
.venv/bin/python - <<'PY' 2>&1 | tee "$ADALN_EVIDENCE/preflight.log"
import sys, torch, triton
from rl_engine.kernels.registry import kernel_registry
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0] == 9
assert "H100" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
print(sys.version, torch.__version__, torch.version.cuda, triton.__version__)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
_, trace = kernel_registry.get_adaln_modulation_op("cuda", hidden=3072)
assert trace["selected_backend"] == "CUDA_ADALN_MODULATION" and not trace["fallback"], trace
print(trace)
PY
.venv/bin/python -m pytest tests/test_extension_smoke.py \
  tests/test_adaln_modulation.py tests/test_adaln_gtest.py tests/test_adaln_shared_bounds.py \
  -q -rs -p no:cacheprovider 2>&1 | tee "$ADALN_EVIDENCE/tests.log"

# 使用与 CUDA_HOME 相同且兼容的 sanitizer；检查退出码及完整 hazard/error summary。
"${CUDA_HOME:?set CUDA_HOME}/bin/compute-sanitizer" --tool racecheck \
  --racecheck-report analysis --error-exitcode 7 \
  .venv/bin/python -m pytest tests/test_adaln_modulation.py \
  tests/test_adaln_gtest.py -q -rs -p no:cacheprovider \
  -k 'cuda and (bit_equality or launch_geometry or sample_bytes or backward_keeps)' \
  2>&1 | tee "$ADALN_EVIDENCE/racecheck.log"

for backend in cuda triton; do
  for dtype in bf16 fp32; do
    .venv/bin/python scripts/check_operator.py --op adaln_modulation \
      --candidate "$backend" --device cuda --dtype "$dtype" \
      --batch 3 --seq 5 --normalized-dim 3072 --check-grad --json \
      > "$ADALN_EVIDENCE/gtest-$backend-$dtype.json"
  done
done
for dtype in bf16 fp32; do
  .venv/bin/python benchmarks/benchmark_adaln_modulation.py \
    --real --dtype "$dtype" --warmup 3 --repeat 10 \
    2>&1 | tee "$ADALN_EVIDENCE/benchmark-$dtype.log"
done
sha256sum -c "$ADALN_EVIDENCE/sources.sha256"
sha256sum -c "$ADALN_EVIDENCE/binary.sha256"
```

验收清单：

- 保存 worktree patch、所有相关 untracked 源文件、源码 hash、binary hash、完整 build/test logs。
- 源码/二进制在整个验收期间不变；确认 fast math disabled、SM90 构建、H100 实际执行。
- focused + gtest + extension smoke 全部通过，0 skip/xfail；三 S、两 dtype、两个 GPU backend 不能漏。
- BF16 forward 两个 literal、backward cancellation、FP32 single-cast VJP differential 全部通过。
- H3072 B=1→B=3 的每个位置、S3/5、非连续视图、y/gate/dX/dMod bytes 全部通过。
- CUDA 三 threads 与 Triton 六 warps/tile 配置的 geometry bytes 全部通过。
- real-image independent CPU accuracy 用共享 forward/gradient tolerance；fixed-reference 与 backend bytes 另验。
- gold 的两项 gradient dtype 为 FP32；不能只见 `report.passed` 或 dtype/finite 就宣称精度通过。
- sanitizer exit 0，0 hazards/errors；有 race 报告即失败，即使数值断言通过。
- native 没有 fallback/skip；benchmark 每个 dtype/S 都有 CUDA/Triton fwd/fwd+bwd 与 peak-memory 数据。
- CLI JSON 是 operator debugging evidence；缺少受验证 provenance 时不能当整个 WS1 系统 EXIT。

GPU CI 与 script 04 现在也会运行新 canaries。CI 未启动，workflow 配置不是运行证据。
