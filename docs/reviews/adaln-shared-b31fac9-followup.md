# AdaLN shared：新增缺口与当前版本证据（2026-10-09）

> 本文是第二轮中间源码快照。用户随后授权继续改实现、补测试；最新状态与证据已统一记录在
> [指定主报告](adaln-shared-b31fac9.md)。本文的 hash、运行表和未关闭项状态不代表最新工作区。

仍以 [issue #386](https://github.com/RL-Align/RL-Kernel/issues/386) 为规格，固定比较点
`b31fac988faae9de71ab1f683e4a5c9cc8578b84`，HEAD
`120f4de7274798c6310d87799ebb6538f2f8f531`。审查范围包括 `git diff b31fac9`
及所有未跟踪文件；原有用户修改保留，没有提交或推送。
[首轮报告](adaln-shared-b31fac9.md) 的源码 hash 与运行表属于此前快照，不能用作当前版本运行证据。

本次沿用 `/code-review` 的独立 Standards/Spec 审查及 `/tdd` 公开接口测试流程。
测试只通过公开 op 调用、`torch.autograd` 和 `run_operator_suite` 观察结果。
确实失败的生产行为逐项先 red 再修复；原本正确但缺断言的行为补 canary，另用公共
callable 错误候选验证断言敏感性，没有为了制造 red 修改生产实现。

PROVEN 仅指表中指定有限反例在当前 CPU/SM86 Triton 版本运行通过。
PARTIAL 表示有局部证据而未覆盖完整域。UNPROVEN 表示尚无当前版本运行证据。
这些状态不表示 CUDA/H100 全平台验收完成。

## Standards：验证器确实放过错误 dtype

**P1，已修复。** AdaLN BF16 输入必须输出 BF16，但无 provenance 的默认 gtest candidate
会将输出转 FP32 后比较，不检查声明的输出边界。公开错误候选
`tuple(output.float() for output in good.fn(**inputs))` 在数值与梯度比较中原本可以通过。
新增 forward-only 和 check-grad 两个反例均先失败。

最小修复：`OperatorCase.candidate_output_dtype` 和 spec 的 `output_matches_input_dtype`
显式启用 AdaLN 输出 dtype 检查；失败报告包含 `dtype mismatch`。
其他算子默认不启用，避免将本就应输出 FP32 的算子误判。gold leaves 的 FP32 策略继续保留。
修复后检查器相关 39 项通过，最终 CPU suite 89 项通过。

## Spec：三项实际实现错误与其他证明缺口

**P1，已修复：Triton 默认 FTZ 改变合法 subnormal eps。** 本机 Triton 3.7.1 默认
`enable_reflect_ftz=True`；本轮 probe 的 PTX 中出现 `sqrt.rn.ftz.f32`。
公开 `eps=1e-40`、常量零输入返回 NaN，而 CPU 按 FP32 eps 应输出零。
forward 与 backward rows 现均显式关闭 reflect FTZ；两 dtype 的 y/dX/dMod literal 验证通过。
不能仅凭 `sqrt_rn` 名称或 `enable_fp_fusion=False` 断言没有 FTZ。

**P2，已修复：Triton tree root 提取额外加正零。** `tl.sum(where(col==0,root,+0))`
把 tree 的负零变成正零，改变 mean 和最后输出符号。现直接 gather 单元素 root 再归约，
移除额外正零。不是放宽 byte assertion。

**P2，已修复：CPU BF16 gate FP32 往返转换损坏 NaN payload。** gate 没有算术，
原实现仍先转 FP32 再转回 BF16；现从原调制张量直接取 gate view。继续检查发现 CPU
backward 的 dgate 同样损坏上游 payload，第二轮 red→green 后改为原 dtype 直接拼入 dMod，
只在 dshift/dscale 各自输出边界 cast。finite 与 nonfinite identity VJP 都验证。
issue 未显式定义 LayerNorm 的 nonfinite 域，本项只证明 gate 位透传，不声称整个 nonfinite
输入域已符合规格。

以下新增测试均在 [test_adaln_gtest.py](../../tests/test_adaln_gtest.py)。
独立 literal 来自标出的标量手算；数学 accuracy 使用独立 CPU FP32 LayerNorm/autograd；
拓扑 differential 只用于 bytes 不变，不充当独立数学参考。

| 要求 / 当前实现与独立参考 | 具体错误实现 | 最小反例 | 应失败的断言 | 当前结论 |
| --- | --- | --- | --- | --- |
| BF16 kernel 输出边界；checker 现显式检查 dtype | 正确 BF16 y/gate 再 `.float()` 返回；旧 checker 数值全通过 | B2,S3,H7 BF16，公开错误 candidate | `gtest_rejects_fp32_outputs_for_bf16_inputs`：forward/check-grad report 都失败，两输出有 dtype mismatch | **PROVEN** CPU checker，实际 red→green |
| 固定 hidden tree 的零符号；CPU 独立 IEEE literal | Triton tree root 提取时再加 +0，擦掉负零 mean | H2，x=[-0,-0]，shift=-0，scale=0 | `signed_zero_follows_hidden_tree`：y 必须是 +0 bytes，错误实现是 -0 | **PROVEN** CPU/SM86 Triton，两 dtype；CUDA **UNPROVEN** |
| FP32 scalar/平方根不能无声明 FTZ；独立标量舍入 | `sqrt.rn.ftz` 把 FP32(1e-40) 当作0 | x=[0,0]，m=0，eps=1e-40；dy=[1,-1] | `subnormal_eps_is_not_flushed_to_zero`：y=0；rstd=1.000002658868784e20，dX=±rstd 最终 cast；dMod=[1,-1,0,0,0,0] | **PROVEN** CPU/SM86 Triton，两 dtype，实际 red→green；CUDA **UNPROVEN** |
| gate/dgate 按原 dtype 位透传，独立 raw literal | BF16 gate或dgate→FP32→BF16，NaN canonicalize | gate raw16=[0x7f81,0xffc1]；dg分别取[2,-3]或raw16=[0x7f82,0x7fff]；x、shift、scale有限 | `gate_preserves_bf16_nan_payload_and_identity_vjp`：gate原uint8 bytes不变，dg原位进入dMod | **PROVEN** CPU/SM86 Triton；CPU fwd/bwd各实际red→green；nonfinite LN合同 **PARTIAL** |
| variance 的 hidden tree 必须独立于 mean 的 canary 检验 | mean 正确，只把 squared-centered 的树换成左折叠 | H4 x=[4096,1.25,-4096,-1.25]，eps0；mean 两算法都0；variance 应8388609，错误为8388608 | `variance_reduction_uses_declared_pairwise_order`：y literal ±1.4142134189605713、±0.0004315836704336107 的 bytes | **PROVEN** CPU/SM86 Triton；对应 mutant 被拒；CUDA **UNPROVEN** |
| backward sum(dnorm) 的固定树 | 只把第一个 backward H sum 改左折叠 | x=[-1,1,-1,1]，eps0，dy=[2^24,1,-2^24,1] | `backward_hidden_sums_use_declared_pairwise_order(sign=1)`：dx=[2^24,0,-2^24,0]；错误中间通道为.25 | **PROVEN** CPU/SM86 Triton；单独 mutant 被拒；CUDA **UNPROVEN** |
| backward sum(dnorm*norm) 的固定树 | 只把第二个 backward H sum 改左折叠 | 同上，dy=[-2^24,1,2^24,1] | 同一测试 sign=-1；dx=[-2^24,0,2^24,0]；错误中间通道为.25 | **PROVEN** CPU/SM86 Triton；单独 mutant 被拒；CUDA **UNPROVEN** |
| BF16 最终边界 RN ties-to-even，forward/backward 都须检查 | 中点向上舍入或截断；随机用例很难撞中点 | x=[-1,1]，eps0，正通道 shift=f；S2 正通道 dy=[1,f]；f=1/256、3/256 | `boundary_rounds_halfway_values_to_even`：1+f 应分别 round 到1和1+4/256；y、dshift、dscale bytes | **PROVEN** CPU/SM86 Triton；四个 forward/backward 舍入 mutant 被拒；CUDA **UNPROVEN** |
| upstream stride 必须按公开 autograd 语义处理 | 按紧密连续地址读 dy，或忽略 dg 的 stride0 | B2,S3,H7；dy 从[B,H,S] transpose，dg 首样本 expand 至B2 | `backward_accepts_transposed_and_broadcast_upstreams`：dX/dMod 等于 contiguous upstream 的 bytes，且对独立 CPU FP32 gold 在声明容差内 | **PROVEN** CPU/SM86 Triton，两 dtype；raw-read mutant 被拒；CUDA **UNPROVEN** |
| 输入可分别冻结，输出可真正不参与 loss | 强制两个输入都求导；backward 假设两个输出一定都有传入 VJP | 仅m训练，y.sum且gate不用；仅x训练、dy=[1,0]、x=[-3,3]eps7；仅gate loss | `single_trainable_input_and_unused_output`：冻结输入.grad is None；dm=[1,1,-1,1,0,0]；dx=±7/128；gate-only dx=0/dm gate identity | **PROVEN** CPU/SM86 Triton，两 dtype；不扩展为二阶梯度或 LoRA 模型验收 |
| H3072 的 batch/padding 必须与 launch 同时改变 | warps8/tile256 路径只在 position>0 读取错 mod；单独 topology、单独 geometry 可漏 | default4/128的B1,S3基准；改4/64或8/256，同时B3的每个position与S3/5，strided输入与6H调制切片 | `h3072_relocation_with_alternate_triton_launch`：y/gate/dX/dMod bytes 等于默认 launch 的同一样本；unused半区0，padding dx数值0 | **PROVEN** SM86 Triton 两额外配置×两 dtype；全部六配置/native三threads交叉 **PARTIAL** |
| unsupported batch 几何必须在入口清楚处理 | C++ backward `grid.y=batch`，既未限B也未展平，B65536进入非法grid | B65536,S1,H1，gate-only backward；不到几MiB | 允许的输入应成功且dx0/dm gate1；若明确不支持，应在公开入口清楚拒绝；不能以 invalid configuration launch失败 | **PARTIAL** 源码定位在 CUDA L194；当前 native runtime **UNPROVEN**，未冒充已修复 |
| 地址乘积不能静默 int32 overflow | `row*hidden`、`(batch*seq+token)*hidden`、`batch*3*hidden` 全以int计算，无 numel 范围检查 | x寻址 B1,S524289,H4096 BF16（约4GiB）；mod寻址 forward B174763,S1,H4096 | 有界支持则公开入口拒绝；支持则 int64寻址并memcheck零错误；不是普通数值 allclose | **PARTIAL** 静态类型/边界证据；实际巨大输入与当前 native **UNPROVEN**；未分配/运行 |
| eps 公共值域与FP32 scalar转换策略须明确 | finite double eps 通过校验，但cast后为Inf或0；数值行为未定义 | eps1e39→FP32 Inf，eps1e-46→0；常量x的后者产生NaN | 按选定政策验证round-to-FP32结果，或清楚拒绝超范围值；不能只断言finite(eps) | **PARTIAL** CPU/SM86 probe 已观察转换结果；是否允许此域规格未明确，未擅自更改API |
| native H上限与非二次幂边界必须交叉验收 | off-by-one允许H4097或错拒H4096；尾部padding树错误 | H4095/4096/4097，两dtype，direct native及registry | 4095/4096正确，4097明确拒绝或有声明fallback；独立数学与bytes都检查 | **UNPROVEN** 当前 native边界；旧大H/irregularH测试不替代这组边界 |

CUDA grid.y 上限65535见
[NVIDIA compute-capability limits](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)。
第12/13项是未覆盖的实现范围错误候选，不是模型常用 B 的性能问题。
若将B限制到65535且H≤4096，modulation乘积自然小于int32上限，但x的B*S*H仍需单独检查。

## 当前版本运行证据

环境仍为 Python3.12.12、PyTorch2.12.1+cu126、Triton3.7.1、RTX3050Ti/SM86。
没有构建 native；现存 `.so` 未绑定当前源码，因此没有执行其数值测试。
下面最终 suite 与 mutation 运行对应文末 hash；red/局部green是本轮 TDD过程的中间源码证据。

| 运行 | 结果 / artifact |
| --- | --- |
| signed-zero red→green | 2 failed/2 passed→4 passed；[/tmp red](/tmp/rlk-adaln-more-signed-zero-red.log)、[green](/tmp/rlk-adaln-more-signed-zero-green.log) |
| subnormal eps red→green | 2 failed/2 passed→4 passed；[red](/tmp/rlk-adaln-more-subnormal-red.log)、[green](/tmp/rlk-adaln-more-subnormal-green.log) |
| CPU gate payload red→green | 1 failed→1 passed；[red](/tmp/rlk-adaln-more-gate-red.log)、[green](/tmp/rlk-adaln-more-gate-green.log) |
| CPU gate VJP payload red→green | 1 failed/1 passed→2 passed；[red](/tmp/rlk-adaln-more-gate-vjp-red.log)、[green](/tmp/rlk-adaln-more-gate-vjp-green.log)；[Triton局部检查](/tmp/rlk-adaln-more-gate-vjp-triton.log)2 passed |
| gtest dtype red→green | 2 failed→相关39 passed；[red](/tmp/rlk-adaln-more-dtype-red.log)、[green](/tmp/rlk-adaln-more-dtype-green.log) |
| 最终CPU/checker | **89 passed、83 deselected，7.29s，无skip**；[log](/tmp/rlk-adaln-more-final-cpu.log) |
| 最终CPU/SM86 Triton，包括额外launch交叉 | **124 passed、67 deselected，14.07s，无skip**；[log](/tmp/rlk-adaln-more-final-gpu.log)；未运行native、大图、旧完整geometry sweep |
| 最终新增错误候选 | **8/8拒绝**；[script](/tmp/rlk_adaln_more_mutations.py)、[log](/tmp/rlk-adaln-more-mutations-final.log) |
| 最终首轮错误候选重跑 | **6/6拒绝**；[script](/tmp/rlk_adaln_review_mutations.py)、[log](/tmp/rlk-adaln-more-old-mutants-final.log) |

```bash
# 最终 CPU 命令
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_adaln_gtest.py tests/test_op_checks.py tests/test_operator_inputs.py \
  -q -rs -k 'not triton and not cuda'

# 最终 SM86 短测试命令；明确排除未经当前源码构建的native及长检查
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TRITON_CACHE_DIR=/tmp/rlk-adaln-review-triton \
  .venv/bin/python -m pytest tests/test_adaln_gtest.py tests/test_adaln_modulation.py \
  -q -rs -k 'not cuda and not real_image and not gpu_padding and not launch_geometry'
```

## H100：新增验证清单与命令，未启动

先执行[首轮报告的源码快照、pinned构建和完整测试命令](adaln-shared-b31fac9.md)。
需要将本 followup 与新的 untracked test 一起带过去，不能用旧 hash/旧 binary。

在已经完成当前源码构建的 H100 环境中，新增 canaries 可单独检查：

```bash
RL_KERNEL_REQUIRE_EXT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests/test_adaln_gtest.py -q -rs \
  -k 'signed_zero or subnormal_eps or gate_preserves or rejects_fp32_outputs or variance_reduction or backward_hidden_sums or halfway_values or broadcast_upstreams or single_trainable or all_launch_configurations'
```

小内存 native batch 边界公开接口复现命令（这里未运行）：

```bash
RL_KERNEL_REQUIRE_EXT=1 .venv/bin/python - <<'PY'
import torch
from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
op = CudaAdaLNModulationOp()
for batch in (65535, 65536):
    x = torch.zeros(batch, 1, 1, device="cuda", requires_grad=True)
    m = torch.zeros(batch, 3, device="cuda", requires_grad=True)
    y, gate = op(x, m)
    dx, dm = torch.autograd.grad(gate, (x, m), torch.ones_like(gate))
    torch.cuda.synchronize()
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.equal(dx, torch.zeros_like(dx))
    expected = torch.zeros_like(m)
    expected[:, 2] = 1
    assert torch.equal(dm, expected)
    print("PASS batch", batch)
PY
```

按当前未限定B的API，第二个batch是预期暴露错误的点。若修复选择限定支持域，
需同步声明限制并将边界断言改为入口明确拒绝，不能将 invalid launch当正常拒绝。
int32 巨大输入反例只准备参数与memcheck清单；没有分配约4GiB输入，更没有启动该测试。

- 全部新增 canary 在当前 native/SM90 和Triton两dtype运行，记录零skip/xfail与完整输出。
- 把全部3种native threads、全部6种Triton warps/tile与H3072 relocation/padding交叉，
  直接比较同一默认B1基准的 y/gate/dX/dMod bytes；本机只新增两组Triton配置。
- native B65535/65536、H4095/4096/4097正负边界不能遗漏。
- 寻址选择64位支持或显式有界拒绝；若支持巨大输入，再用compute-sanitizer memcheck检查
  B1,S524289,H4096及modulation地址范围，保留0 errors证据。
- 对新FTZ修复保存当前Triton版本/compile options/PTX，确认fwd和bwd_rows均无sqrt FTZ；
  其他Triton版本/ROCm配置须独立qualification。
- 保留首轮要求的三真实image S、benchmark、racecheck、源码/二进制hash及pinnedruntime记录。

## 对应最终测试的源码 SHA256

| 文件 | SHA256 |
| --- | --- |
| 未跟踪 `tests/test_adaln_gtest.py` | `85de4e0760d6ead947b0af1e8bd4eb896cfdea6bccf4434ca46d5a3adf67e165` |
| `gtest/op_checks.py` | `70f9c648b5da5ace001d295569c4b05a15d2e630e84e1870b5734a5377d76b39` |
| `gtest/operator_specs.py` | `b78d15322199b4986f19697c7987dea879238b7d8b14fe2f1b05fc3fe0268e79` |
| PyTorch AdaLN | `1b573827279e89c97e4f293bf867fe5a3a77102c89cf9427ff0e1f429a351c57` |
| Triton AdaLN | `ef2b13c8e13da8d324b160489efad4b2080d3a1058cf3b58d566c2bc92a3b4a2` |
| CUDA source（未修改） | `27074706332b1db32236b6cbb225ee99514938f38d1996ef4812e221d857e80d` |

静态检查：修改的五个Python文件ruff/isort均通过；新增测试与checker的Black check通过，
`git diff --check`通过，验收shell的`bash -n`通过。

`/tmp`日志与mutation脚本是本次artifact，应另行归档。新增文档不会改变上述测试源码hash。
