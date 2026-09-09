# P5 Expert 启动套件（`P5-S0`）

启动套件为所有 P5 子议题（P5-1…P5-9）解除阻塞：冻结数据契约，为五个 WS1 算子提供位精确的 FP32 oracle，生成带种子的 golden fixture，并提供一条任何后端 PR 都可以独立运行的验收命令。

## 子议题命名（开发顺序发布于 #8）

`P5-N` 是 #8 中排序评论所定义的开发顺序标签；GitHub issue 编号仍然是链接引用的权威来源。

| 标签 | 范围 |
| --- | --- |
| P5-S0 | 本启动套件（契约、oracle、fixture、验收命令） |
| P5-1 | `mxfp8_act_quant`（前向 + STE 反向） |
| P5-2 | `clamp_swiglu_weighted`（前向 + dgate/dup/dp_s） |
| P5-3 | `shared_grouped_lora_delta`（前向 + dX/dA/dB） |
| P5-4 | `mxfp8_mxfp4_grouped_gemm`（前向 + 仅 dX） |
| P5-5 | `shared_expert_mlp`（前向 + 仅 dX） |
| P5-6 | `moe_provider_adapter`（Megatron + vLLM 注入） |
| P5-7 | WS2：EP placement、`expert_tensor_parallel_size = 1` gate |
| P5-8 | WS2：shared expert TP/SP + shared-once gate |
| P5-9 | WS2：EP>1 / 各种 placement 下 adapter fail-closed |

## 套件内容

| 模块 | 内容 |
| --- | --- |
| `rl_engine/moe/mx_format.py` | OCP MX 编解码器：E8M0 / E4M3 / E2M1、block-32 量化/反量化、半字节打包。定义 P5-1/P5-4 的 golden 字节。 |
| `rl_engine/moe/contract.py` | `ExpertBatch`、`SharedBatch`、`LoRAParams`、clamp 常量、张量 fingerprint。它是 Foundation `ExpertBatch` ABI（`p5-expertbatch-v1`）在 P5 中的本地子集。 |
| `rl_engine/moe/oracle.py` | 五个算子的 FP32 参考实现，以及完整的 routed/shared 前向—反向组合实现。 |
| `rl_engine/moe/provider.py` | `ExpertProvider` 协议、由 oracle 支持的 `ReferenceProvider`、fail-closed 的 `StubProvider`。 |
| `rl_engine/moe/fixtures.py` | 带种子的 fixture 用例和 golden-hash manifest（`tests/fixtures/p5/golden_hashes.json`，CI 锚点）。 |
| `rl_engine/moe/trace.py` | 边界哈希和 `first_divergence`（P5 本地的 `TraceEnvelope` 替代实现）。 |
| `scripts/check_p5.py` | 验收命令。 |

## 冻结的数值契约（#8 内容及本次决定的总结）

来自各 issue 的约定：

1. 仅进行 LoRA 微调；基础权重保持冻结——任何位置都**不包含 `dW`**。
2. Routed base 使用 MXFP8 activation × MXFP4 frozen weight；block = 32，scale = E8M0，元素格式为 E4M3 / E2M1（OCP Microscaling v1.0）。
3. 反向使用 BF16（不重新量化为 MXFP8）；每次归约都使用 FP32 累加器。
4. 路由权重 `p_s` 在 `clamp_swiglu_weighted` 中应用（`h = SiLU(min(gate,10)) · clamp(up,−10,10) · p_s`），在全局范围内恰好应用一次。
5. `mxfp8_act_quant` 的 amax 是行内 32 个元素的归约；反向使用 STE。
6. 单轮 SwiGLU：使用 FP32 数学运算，并仅在输出上进行一次 BF16 舍入。

本套件必须冻结的决定（已在 #8 标记为待评审；修改其中任何一项都需要重新生成 manifest，并递增 schema/profile id）：

| # | 决定 | 理由 |
| --- | --- | --- |
| D1 | **E4M3 编码：先在 FP32 中截断到 ±448，再执行 RNE 转换**（torch `float8_e4m3fn`）。直接使用 torch cast 会将溢出映射为 NaN；clamp+cast 等价于 PTX `cvt.satfinite`。 | 与硬件 satfinite 行为一致；由 golden 测试固定。 |
| D2 | **E8M0 scale 计算规则**：`shared_exp = floor(log2(amax)) − emax_elem`（E4M3 为 8，E2M1 为 2）；全零 block → code 127（scale 为 1）。`floor(log2)` 通过 `frexp` 精确计算。 | OCP 推荐规则；使用精确的整数运算。 |
| D3 | **Oracle 数值 profile `oracle-fp32-serial-v1`**：按索引升序串行累加，采用先乘后加的舍入方式（**不进行 FMA 融合**）。严格的 CUDA kernel 必须使用 `__fmul_rn`/`__fadd_rn` 以匹配该行为，或注册自己的 profile。 | 为实现字节级相等，必须固定归约顺序；升序串行归约便于审计。 |
| D4 | **LoRA 的 GEMM 间舍入**：`U = X·Aᵀ` 在执行 `Y = U·Bᵀ·α` 前舍入为 BF16；反向过程中，`dY·α` 和 `dU` 也在 GEMM 之间舍入为 BF16。 | 与双 GEMM BF16 pipeline 一致；两个引擎都必须遵守。 |
| D5 | **clamp 子梯度恰好位于边界时为零**（只有严格不等式范围内才传递梯度）。 | 边界取值规则必须确定且可复现；由测试固定。 |
| D6 | **Shared expert 不使用 clamp**（`h = SiLU(gate)·up`），按照 P5-5（#64）中固定的数学定义，复用单轮 SwiGLU，并令 `p_s = None`。 | P5-5（#64）的文字描述说“复用 `clamp_swiglu_weighted`（不带 `p_s`）”，但其数学公式显示不使用 clamp——**该问题已在 issue 上提出，尚待解决**。 |
| D7 | 反向返回的梯度为 FP32（累加器数据类型）；在下一个算子边界处舍入为 BF16。 | 与“BF16 backward，FP32 reductions”保持一致。 |

## 字节相等的适用范围

严格的字节级相等要求只适用于：**训练端与推理端使用相同数值 profile 且运行在相同设备上**。已提交的 manifest 固定了 CPU x86 oracle；`scripts/check_p5.py` 会在 provider 所使用的设备上重新计算 oracle，因此超越函数（如 sigmoid）不会在严格比较过程中跨设备传递。

不具备等效能力的硬件（例如没有 FP8 MMA、fnuz 格式或原生 MX 指令）必须注册自己的 profile，并明确给出容差——绝不能静默放宽要求（P5-4/P5-6 契约）。

## 子议题 PR 如何使用该套件

1. 继承 `ReferenceProvider`，只重写 PR 实际交付的算子（其余部分继续使用 oracle），并设置 `name`/`numeric_profile`。
2. 运行 `python scripts/check_p5.py --provider your.module:YourProvider [--device cuda]`。每个边界都必须字节级相等；否则退出码为 1。
3. 在 PR 描述中附上检查输出以及 `provenance()` 的结果。

Fixture 用例包括：`base_only_one_row`、`base_only_packed`、`lora_only`、`base_plus_lora`、`uneven_experts`（零行 expert）、`shared_t1`、`shared_t16`；此外还有算子边界用例 `act_quant_edges`（2 的幂、RNE ties、零行）和 `swiglu_boundary`（位于 clamp 边界、边界内及边界外的值）。

在有意修改契约后，重新生成 manifest：

```bash
python -m rl_engine.moe.fixtures --write-manifest
```

## 本套件不包含的内容

不包含 CUDA/Triton kernel，不包含 Megatron/vLLM 注入（P5-6），不包含 EP transport 或 combine（P4/P6），也不包含多 rank gate（P5-7…P5-9）。`output_slot` 会原样传递，为 P6 保留。
