# 补齐 P5-4 grouped-GEMM 的 device 侧 prep 环节

## Context（为什么改）

文件 `RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu` 里 P5-4 的 grouped GEMM 目前是**半吊子**：

- strict 内核 `moe_grouped_gemm_fwd_strict` / `moe_grouped_gemm_bwd_strict` 的数值逻辑已经正确（K 升序串行、`__fmul_rn`/`__fadd_rn`、FP32 输出、bwd 只回 dX），且它们本身**能跑在 GPU 上**。
- 但 `moe_prepare_experts`（[225-264](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L225-L264)）是** host 侧**函数：它调用 `moe_copy_offsets` 把 `expert_offsets` 拷回 CPU，再生成一个 host 内存里的 `std::vector<MoeGroupedGemmExpert>`。这个 vector **没有任何下游消费**（发射函数收的是原始指针），而且这条"拷回 CPU 再算"的路径被文件自己的注释（[201-203](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L201-L203)）明确禁止作为数值实现。

目标：把 prep 从 host 挪到 **device**——用一个 prep kernel 在 GPU 上把每个 expert 的分组指针算好、写进 device 内存，并接入 fwd/bwd 发射链，让 strict grouped GEMM 全程 device 化。**P5-1 激活量化不在本次范围**（输入是已量化的 codes/scales）。

## 契约（保持不变，必须 bit 级不变）

| 张量 | dtype | shape | 备注 |
|---|---|---|---|
| `a.codes` | uint8 E4M3 | `[M, K]` | row-major，K 连续；来自 P5-1 |
| `a.scales` | uint8 E8M0 | `[M, K/32]` | 与 codes 同行对齐 |
| `w.codes` | uint8 E2M1×2 | `[E, N, K/2]` | 低 nibble = 偶数 k |
| `w.scales` | uint8 E8M0 | `[E, N, K/32]` | |
| `expert_offsets` | int32 | `[E+1]` | 单调不减、首 0 尾 M；允许空 group |
| `Y`（fwd 出） | **FP32** | `[M, N]` | 不提前 round BF16 |
| `dy`（bwd 入） | BF16 | `[M, N]` | |
| `dX`（bwd 出） | **FP32** | `[M, K]` | W dequant→BF16，串行升序 n，FP32 累加 |

### 公开接口（MXTensor，保持不变）

```python
# provider.py 的 CudaP5GemmProvider 方法（与 oracle.py 签名一致）：
mxfp8_mxfp4_grouped_gemm_fwd(a: MXTensor, w: MXTensor, expert_offsets) -> Y   # FP32 [M, N]
mxfp8_mxfp4_grouped_gemm_bwd(dy, w: MXTensor, expert_offsets) -> dX          # FP32 [M, K]；无 saved、无 dW
```

- MXTensor 拆包（`a.codes/a.scales`、`w.codes/w.scales`）发生在 **Python 层 provider.py 的 `CudaP5GemmProvider`**（现状不变，[provider.py:275-287](RL_KERNEL/rl_engine/moe/provider.py#L275-L287)）。
- C++ 函数 `moe_mxfp8_mxfp4_grouped_gemm_forward(codes, scales, w_codes, w_scales, offsets)` 与 `_backward(dy, w_codes, w_scales, offsets)` 的**外部签名保持不变**（仍收 codes/scales 张量）。
- 本次改动主体在 C++ 内部（prep kernel + 内核消费描述符 + 发射链），不触碰 MXTensor 接口；唯一 Python 侧改动是 `CudaP5GemmProvider.provenance()`（见下方「P5_4 §4/§6 落地检查」）。

**数值路径一字不改**：只改"指针/路由怎么准备与传递"，不改逐元素累加、舍入、scale 应用顺序。

## 改动设计

### 1. 新增 device prep kernel（fwd + bwd 各一个，或一个模板）

仿 vLLM 的 `__get_group_gemm_starts`（`references/vllm/.../nvfp4_blockwise_moe_kernel.cu:54-118`），但适配 P5 契约（无 alpha、无 cute layout、无 128B TMA 断言）。

**fwd prep**（`__global__ void moe_prepare_experts_fwd_device`）：
- 入参：`MoeGroupedGemmExpert* experts`（device 输出数组 `[E]`）、`a_codes/a_scales/w_codes/w_scales/out` 基址、`expert_offsets`（device，直接读，不拷 CPU）、`M, N, K, E`。
- 每线程 `e = blockIdx.x*blockDim.x + threadIdx.x`，`e < E` 时：
  - `row_begin = expert_offsets[e]`，`row_count = expert_offsets[e+1] - row_begin`；
  - 写一个 `MoeGroupedGemmExpert` 到 `experts[e]`，指针字段（行/专家索引先 `(int64_t)` 提升，避免 32-bit 溢出）：
    - `experts[e].activation_codes_ptr  = a_codes  + (int64_t)row_begin * K;`
    - `experts[e].activation_scales_ptr = a_scales + (int64_t)row_begin * (K >> 5);`
    - `experts[e].packed_weight_codes_ptr = w_codes  + (int64_t)e * N * (K >> 1);`
    - `experts[e].weight_scales_ptr       = w_scales + (int64_t)e * N * (K >> 5);`
    - `experts[e].output_ptr = out + (int64_t)row_begin * N;`
  - stride 字段（int64_t）填常量，同样 `(int64_t)` 提升：`activation_row_stride=(int64_t)K`、`activation_scale_row_stride=(int64_t)(K>>5)`、`weight_expert_stride=(int64_t)N*(K>>1)`、`weight_scale_expert_stride=(int64_t)N*(K>>5)`、`output_row_stride=(int64_t)N`。
- 网格：`<<<ceil(E/128), 128>>>`。

**bwd prep**（`__global__ void moe_prepare_experts_bwd_device`）：结构同 fwd，但产出新 struct `MoeGroupedGemmBwdExpert`，字段为 `dy_ptr = dy + (int64_t)row_begin * N`、`dx_ptr = dx + (int64_t)row_begin * K`、`packed_weight_codes_ptr = w_codes + (int64_t)e * N * (K >> 1)`、`weight_scales_ptr = w_scales + (int64_t)e * N * (K >> 5)`、row/stride 元数据（同 fwd 的 `(int64_t)` 提升）。

> `MoeGroupedGemmExpert` 已是 POD（纯字段、无方法），可直接在 device 上聚合赋值 `experts[e] = {...}`，无需改 struct。新增的 bwd 描述符同样定义为 POD。

### 2. 新增 bwd 描述符 struct

紧邻现有 `MoeGroupedGemmExpert`（[143-160](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L143-L160)）新增：

```cpp
struct MoeGroupedGemmBwdExpert {
  int expert_id, row_begin, row_count, output_n, input_k, block_count;
  const __nv_bfloat16* dy_ptr;
  const std::uint8_t* packed_weight_codes_ptr;
  const std::uint8_t* weight_scales_ptr;
  float* dx_ptr;
  int64_t dy_row_stride, dx_row_stride,
          weight_expert_stride, weight_scale_expert_stride;
};
```

### 3. 新增 host launcher（类似 vLLM 的 `run_get_group_gemm_starts`）

仿 vLLM `references/vllm/.../nvfp4_blockwise_moe_kernel.cu:150-200` 的 `run_get_group_gemm_starts`，新增两个 host 函数（fwd/bwd 各一）作为 prep 的 host 入口，集中负责**确定设备型号/索引 + device_guard + 取 stream + 发射 prep kernel**：

```cpp
void moe_run_prepare_experts_fwd(
    MoeGroupedGemmExpert* experts,            // device 描述符缓冲（已分配）
    const torch::Tensor& activation_codes,    // 仅用于取 device 与基址
    const torch::Tensor& activation_scales,
    const torch::Tensor& packed_weight_codes,
    const torch::Tensor& weight_scales,
    torch::Tensor& output,
    const torch::Tensor& expert_offsets,
    int M, int N, int K, int E) {
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(activation_codes)); // 设备确定 + guard
  auto stream = at::cuda::getCurrentCUDAStream();                                // 当前 stream
  moe_prepare_experts_fwd_device<<<ceil(E/128), 128, 0, stream>>>(
      experts, activation_codes.data_ptr<uint8_t>(), activation_scales.data_ptr<uint8_t>(),
      packed_weight_codes.data_ptr<uint8_t>(), weight_scales.data_ptr<uint8_t>(),
      output.data_ptr<float>(), expert_offsets.data_ptr<int32_t>(), M, N, K, E);
}
```

- fwd/bwd wrapper 各自：分配输出 `output`/`dx` → 分配 `experts_buf`（`torch::empty({(int64_t)E * sizeof(MoeGroupedGemmExpert)}, options.dtype(kByte))`，`reinterpret_cast` 为描述符指针；bwd 用 `sizeof(MoeGroupedGemmBwdExpert)`）→ 调用对应 `moe_run_prepare_experts_*` → 再把 `experts`（+ `expert_offsets`）传给 strict 内核。
- device_guard / stream 逻辑集中在 launcher 内，与 vLLM `run_get_group_gemm_starts` 的结构对齐，wrapper 不再各自散落。
- 保留现有 wrapper 顶部的 `moe_copy_offsets` 校验（front=0/back=M/非递减）——仅 wrapper validation，非数值路径；数值 prep 已不依赖它。

### 4. strict 内核改为消费 device 描述符

**fwd** `moe_grouped_gemm_fwd_strict`（[321-360](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L321-L360)）签名改为：

```cpp
(const MoeGroupedGemmExpert* __restrict__ experts,
 const int32_t* __restrict__ expert_offsets, int M, int N, int K, int E)
```

线程内：
- `m = idx/N; n = idx%N; e = moe_find_expert(m, expert_offsets, E)`（路由仍用紧凑 offsets 二分，不变）；
- 取 `const auto& ex = experts[e];`，行内偏移 `dr = m - ex.row_begin`；
- `a_row  = ex.activation_codes_ptr  + dr*ex.activation_row_stride;`
- `a_srow = ex.activation_scales_ptr + dr*ex.activation_scale_row_stride;`
- `w_row  = ex.packed_weight_codes_ptr + n*(K>>1);`
- `w_srow = ex.weight_scales_ptr       + n*ex.block_count;`
- 写出 `out = ex.output_ptr + dr*ex.output_row_stride + n;`
- 块内 32 点积、`(partial*sa)*sw`、`__fmul_rn/__fadd_rn` **逐行保持不变**。

**bwd** `moe_grouped_gemm_bwd_strict`（[364-400](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L364-L400)）同理改为消费 `MoeGroupedGemmBwdExpert*`，`dy_row = ex.dy_ptr + dr*ex.dy_row_stride`，`dx = ex.dx_ptr + dr*ex.dx_row_stride + k`，w 指针用 `ex.packed_weight_codes_ptr`/`ex.weight_scales_ptr`，dequant→BF16 的串行 n 累加逻辑不变。

### 5. 更新发射函数 + 删除死代码

- `moe_launch_grouped_gemm_fwd_strict` / `_bwd_strict`（[404-425](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L404-L425)）参数改为收 `experts`（+ offsets），不再收 4 个平铺基址指针。
- 删除 host `moe_prepare_experts`（[225-264](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L225-L264)）；偏移计算逻辑已搬进 device prep kernel。
- **删除未使用的 `MoeGroupedGemmArgs`（[162-174](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L162-L174)）与 `MoeGroupedGemmBackwardArgs`（[176-187](RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu#L176-L187)）结构体**。
- `MoeQuantizeArgs`（P5-1 用）与 `moe_check_h100_sm90` 本次不动（P5-1 不在范围）。

## P5_4 §4/§6 落地检查（本次改动必须满足）

### strict one-row unpadded geometry（§4）

本次 strict 路径的 launch geometry 保持并**显式明确**为 one-row unpadded：

- **fwd** `moe_grouped_gemm_fwd_strict`：flat 1D grid `ceil(M*N / kMoeStrictBlock)` 块 × `kMoeStrictBlock` 线程，每线程算一个输出元素 `(m,n)`，K 方向 `j` 升序、块内 `k` 升序**串行**归约；`moe_find_expert` 按 `expert_offsets` 二分路由。
- **bwd** `moe_grouped_gemm_bwd_strict`：同构，每线程算 `(m,k)`，`n` 升序串行归约。

无 split-K、无 atomic merge、无 workspace、无 M/N/K tile、无 padding；scale 括号顺序 `(partial*sa)*sw` 固定；`__fmul_rn`/`__fadd_rn` 禁 FMA。这与 `CudaP5GemmProvider.capabilities()["geometry"]=["one-row"]`（[provider.py:256-265](RL_KERNEL/rl_engine/moe/provider.py#L256-L265)）及文件顶部 `strict: one-row/unpadded` 注释一致，即 "one-row" 参考几何；解码语义与 oracle 逐位一致（`与 Miles decode 一致`）。

> 术语澄清：此处 "one-row" = 每个输出元素完全在单线程内串行归约、不跨线程/跨行做归约（区别于 "packed" 的 M-tile / split-K / batch 几何），而非字面 "一个线程算一整行"。本次不引入 packed（WGMMA）路径，strict 即 one-row 参考。

### geometry byte-equal gate（§6）

门禁 `fwd(packed)[r] == fwd(one-row)` 针对 "packed"（WGMMA/tiled/batched）路径与 "one-row" 参考的比较。**本次 PR 不交付 packed 路径**，故门禁**平凡满足**（strict 本身即 one-row 参考，无 packed 可比较）。若后续交付 WGMMA 路径，必须：独立 numeric profile + 与 oracle 的 max-diff 报告，且不得静默顶替 strict path——该约束写入下方「备注」。

### provenance（§6）

`CudaP5GemmProvider.provenance()`（[provider.py:267-273](RL_KERNEL/rl_engine/moe/provider.py#L267-L273)）目前缺 tile/split-K 配置与 kernel/build fingerprint，需扩展为：

```python
def provenance(self) -> dict[str, Any]:
    return {
        "requested_backend": self.name,
        "actual_backend": "cuda-strict",
        "numeric_profile": self.numeric_profile,       # "p5-strict-ffma-v1"
        "geometry": "one-row-unpadded",
        "tile": None,          # 无 tiling
        "split_k": 1,          # 无 split-K
        "kernel_fingerprint": "mxfp8-mxfp4-grouped-gemm-strict-v1",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
```

`capabilities()`（[provider.py:256-265](RL_KERNEL/rl_engine/moe/provider.py#L256-L265)）已声明 `geometry ["one-row"]`、`implemented [fwd,bwd]`，无需改。`kernel_fingerprint` 字符串与 `.cu` 内新增的 `constexpr const char* kMoeGroupedGemmKernelFingerprint` 保持一字一致（单一事实源，人工同步）。

### 无 dW / base weight packed bytes 不变（§6）

- bwd 只回 `dX`，签名无 `dW`、无 `saved`（wrapper 不分配任何 weight grad）。
- base weight 全程 packed：`w.codes`/`w.scales` 只读，dequant 仅在 kernel register 内进行，**不写回 global**。Step 4 对拍时断言 `w.codes`/`w.scales` 的 sha256 前后不变。

## 函数上下游关系（text 图）

约定：★ = host 函数（需自动化正确性测试）；▽ = device 内核（不单独测，手动 debug）。

```text
依赖方向：上游（顶层）→ 下游（底层）。箭头 = 调用 / 产出。

FWD 链路
─────────────────────────────────────────────────────────────────────────────
[Python] provider.py mxfp8_mxfp4_grouped_gemm_fwd(a:MXTensor, w:MXTensor, offsets)
      │  拆包 a.codes/a.scales、w.codes/w.scales（不改）
      ▼
★ [Host] moe_mxfp8_mxfp4_grouped_gemm_forward(codes, scales, w_codes, w_scales, offsets)
      │  ①校验 ②分配 Y ③分配 experts_buf ④调 prep launcher ⑤调 GEMM launch
      ├──────────────────────────────┬─────────────────────────────────┐
      ▼                              ▼                                 ▼
★ [Host] moe_run_prepare_experts_fwd      ★ [Host] moe_launch_grouped_gemm_fwd_strict
      │  device_guard + stream           │  grid/block
      ▼                                  ▼
▽ [Dev] moe_prepare_experts_fwd_device    ▽ [Dev] moe_grouped_gemm_fwd_strict(experts,...)
      │  写 experts[e]={row,ptr,stride}   │  moe_find_expert + 解码(e4m3/e2m1/e8m0)
      ▼                                  │  读 experts[e] → 累加 → 写 Y[m,n]
   experts[]（device 描述符数组 [E]）─────┘
                                 ▼
                            Y（FP32 [M,N]）

BWD 链路（同构，dy + w → dX，无 dW）
★ [Host] moe_mxfp8_mxfp4_grouped_gemm_backward(dy, w_codes, w_scales, offsets)
      ├─► ★ moe_run_prepare_experts_bwd → ▽ moe_prepare_experts_bwd_device → experts_bwd[]
      └─► ★ moe_launch_grouped_gemm_bwd_strict → ▽ moe_grouped_gemm_bwd_strict → dX（FP32 [M,K]）
```

## 关键文件

- `RL_KERNEL/csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu` —— 主要改动文件（新增 2 个 prep kernel + 1 个 bwd 描述符 + 1 个 `constexpr` 指纹，改 2 个 strict 内核签名 + 2 个 wrapper + 2 个 launch，删 1 个 host 函数）。
- `RL_KERNEL/rl_engine/moe/provider.py` —— 仅改 `CudaP5GemmProvider.provenance()`（补 geometry/tile/split_k/kernel_fingerprint/cuda_version 字段），其余不动。
- 参考（只读）：`references/vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu:54-118`（prep kernel 范式）；`scripts/check_p5.py`（端到端验收入口）。

## 实现顺序与测试（自底向上，每个 host 函数完成即测）

> 约定：★ host 函数需自动化正确性测试；▽ device 内核不单独测（手动 debug）。
> 每步都先 `python setup.py build_ext --inplace` 编译通过，再跑对应测试，通过后才进入下一步。

**Step 0 —— 结构清理（无测试，编译通过即可）**
- 删 `MoeGroupedGemmArgs`、`MoeGroupedGemmBackwardArgs`；新增 `MoeGroupedGemmBwdExpert`。

**Step 1 —— prep（▽ prep kernel + ★ launcher）**
- 实现 `moe_prepare_experts_fwd_device` / `_bwd_device`；实现 ★ `moe_run_prepare_experts_fwd` / `_bwd`（device_guard + stream + 发射 prep）。
- 【测 ★ launcher】跑 launcher 后把 `experts_buf` 拷回 CPU，逐项断言：
  - `experts[e].row_begin == offsets[e]`
  - `experts[e].row_count == offsets[e+1] - offsets[e]`
  - `(experts[e].activation_codes_ptr - a_codes_base) == row_begin * K`（及 w/scale/out 指针偏移、各 stride 一致）
  - 覆盖：多 expert + 空 group（`row_count==0`）。

**Step 2 —— GEMM 消费描述符（▽ strict 内核 + ★ GEMM launch）**
- 改 `moe_grouped_gemm_fwd_strict` / `_bwd_strict` 签名收 `experts`；改 ★ `moe_launch_grouped_gemm_fwd_strict` / `_bwd_strict`。
- 【测】通过 wrapper 半成品（或临时入口）跑 fwd/bwd，对拍 oracle（`rtol=0,atol=0`）。
- 覆盖：`M=1` 单行、空 group、`K%32==0`、多 expert。

**Step 3 —— wire ★ 两个顶层 wrapper**
- wrapper 分配 `experts_buf` + 调 launcher + 调 GEMM launch。
- 【测】端到端：`_C.moe_mxfp8_mxfp4_grouped_gemm_forward/backward` 对拍 oracle，bit 级一致。
- 顺带断言：数值路径无 `moe_copy_offsets` / `.to(kCPU)` host sync（仅 wrapper 校验保留）。

**Step 4 —— provenance + 指纹 + 无 dW 断言（★ Python）**
- `.cu` 内加 `constexpr const char* kMoeGroupedGemmKernelFingerprint = "mxfp8-mxfp4-grouped-gemm-strict-v1"`。
- 改 `CudaP5GemmProvider.provenance()`：补 `geometry/tile/split_k/kernel_fingerprint/cuda_version`（`kernel_fingerprint` 与 `.cu` 常量一致）。
- 【测】`resolve_provider("cuda").provenance()` 返回上述字段；`capabilities()["geometry"]==["one-row"]` 且 `implemented` 含 fwd/bwd。

**Step 5 —— 端到端验收（★ 全链路）**
- 跑 `python scripts/check_p5.py --provider cuda --device cuda`：所有 e2e case（含 `uneven_experts` 空 group）PASS，`y/dx` 与 oracle 逐字节一致（sha256 相等）。
- 顺带断言：base weight 无 dW（`w.codes`/`w.scales` sha256 前后不变）；bwd 只回 `dX`。

对拍输入构造（复用 oracle）：`a = mx_quantize(x, "e4m3")` 造 a.codes/a.scales；造 `w`（E2M1 codes/scales）与 `expert_offsets`；`y_native` 与 `oracle.mxfp8_mxfp4_grouped_gemm_fwd(a, w, offsets)` 逐字节比较；bwd 同理对 `oracle.mxfp8_mxfp4_grouped_gemm_bwd`。

## 备注（非本次范围）

- P5-1 激活量化（`moe_mxfp8_act_quant_forward` 仍 fail-closed）**不改**。
- WGMMA / CUTLASS ptr-array 性能路径仍保持独立、不引入；本次只是把 strict 路径的 prep 设备化。
- 若后续交付 WGMMA（packed）路径：必须通过 `fwd(packed)[r] == fwd(one-row)` geometry 门禁；做不到则 strict path 明确降级 one-row、packed 单独 benchmark，并登记独立 numeric profile + 与 oracle 的 max-diff 报告，**禁止静默顶替 strict path**。`expert_tensor_parallel_size = 1`（不切 TP，单卡），本内核不涉及。
