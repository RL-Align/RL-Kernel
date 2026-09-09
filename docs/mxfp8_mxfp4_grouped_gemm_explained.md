# `mxfp8_mxfp4_grouped_gemm.cu` 自顶向下详解

> 学习笔记 · 目标文件：`csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu`（共 591 行）
> 面向读者：已经熟悉 CUDA / PyTorch C++ extension 基础，想逐层读懂这个 strict MX 分组 GEMM 实现。

---

## 0. 一句话定位

本文件实现 **P5-4 算子 `mxfp8_mxfp4_grouped_gemm` 的 strict CUDA 后端**：

- **MXFP8 激活**（E4M3 code + E8M0 scale，block=32）× **MXFP4 冻结权重**（E2M1 nibble + E8M0 scale，block=32）
- 输出为 **分组 GEMM** 的前向结果，以及 **仅 dX** 的反向
- 数值契约：与 CPU FP32 oracle **逐字节相等（bit-exact）**

关键词：分组 GEMM（grouped GEMM）、MoE 专家路由、MX（Microscaling）编码、串行升序累加、fail-closed。

---

## 1. 顶层架构：三个层次

```
┌─────────────────────────────────────────────────────────────────────┐
│  L1  对外 API 层（PyTorch binding，文件末尾 437–590 行）             │
│      moe_mxfp8_act_quant_forward          (fail-closed 占位)        │
│      moe_mxfp8_mxfp4_grouped_gemm_forward (完整实现)                │
│      moe_mxfp8_mxfp4_grouped_gemm_backward(完整实现)                │
│      ── 职责：收 torch::Tensor → 校验形状/dtype → 调 L2 → 返回 Tensor│
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 调用
┌──────────────────────────────▼──────────────────────────────────────┐
│  L2  调度/校验层（anonymous namespace 内）                           │
│      moe_launch_grouped_gemm_fwd_strict / _bwd_strict  (启动 kernel) │
│      moe_check_cuda_contiguous / moe_check_same_device /            │
│      moe_copy_offsets / moe_check_h100_sm90 / moe_prepare_experts   │
│      ── 职责：算 grid/block、启动 <<<>>>、host 端元数据准备           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ <<<grid, block, 0, stream>>>
┌──────────────────────────────▼──────────────────────────────────────┐
│  L3  设备计算层（__global__ / __device__）                           │
│      moe_grouped_gemm_fwd_strict   (前向 kernel)                    │
│      moe_grouped_gemm_bwd_strict   (反向 kernel)                    │
│      小部件：moe_find_expert（路由）                                 │
│      小部件：moe_e4m3_to_f32 / moe_e2m1_nibble_to_f32 /             │
│               moe_e8m0_to_f32（三种 MX 解码器）                      │
│      ── 职责：真正的数值计算，bit-exact 串行累加                     │
└─────────────────────────────────────────────────────────────────────┘
```

**支撑层（数据契约）**：结构体 `MoeGroupedGemmExpert / MoeGroupedGemmArgs / MoeGroupedGemmBackwardArgs`（138–193 行）定义了 A/B/C/scale 的指针布局；常量块（131–136 行）固定了 block=32 等合同值。

---

## 2. 数据契约：三种 MX 编码

这是理解所有内核的前提，先把它吃透。

### 2.1 激活：E4M3（1 字节 / 元素）

```
bit:   7    6 5 4 3    2 1 0
      [sign][ exp ][ mant ]
```

- 1 符号位、4 指数位（bias 7）、3 尾数位
- 逻辑形状 `[M, K]`，`activation_codes[m*K + k]` 存一个 E4M3 code byte

### 2.2 权重：E2M1（半个字节 / 元素，2 个打包进 1 字节）

```
bit:   3     2 1     0
      [sign][exp][mant]
```

- 1 符号位、2 指数位、1 尾数位
- 物理形状 `[E, N, K/2]`，**低 nibble = 偶 k，高 nibble = 奇 k**
- 地址公式：`packed_weight_codes[e*N*(K/2) + n*(K/2) + (k/2)]`

### 2.3 scale：E8M0（1 字节 / 块）

```
bit:   7 6 5 4 3 2 1 0
      [      exp(8)     ]
```

- 8 位纯指数，无符号位无尾数，表示 `2^(code - 127)`
- 每 32 个连续元素共享一个 scale（block = 32）
- code=127 → scale=1.0（全零 block 用）；code=255 → NaN（被 wrapper 拒绝）
- 激活 scale 形状 `[M, K/32]`；权重 scale 形状 `[E, N, K/32]`

### 2.4 专家路由：`expert_offsets`

- 长度 `E+1` 的 int32 数组，非递减，`first==0`，`last==M`
- 专家 e 负责连续行区间 `[offsets[e], offsets[e+1])`
- 相邻相等 = 空专家，应安全跳过

---

## 3. L1 对外 API 层（三个对外函数）

### ① `moe_mxfp8_act_quant_forward`（437–445 行）

**功能**：P5-1 激活量化。**当前是 fail-closed 占位**——只做输入校验，最后调用
`moe_cuda_skeleton_unimplemented("mxfp8_act_quant_fwd")` 直接 `TORCH_CHECK(false, ...)` 抛错。

**为什么这样做**：注释明确「P5-1 只保留前向 CUDA ABI，尚未实现实际量化 kernel」，
目的是让任何 provider **不能静默声称支持 P5**（fail-closed 契约）。

### ② `moe_mxfp8_mxfp4_grouped_gemm_forward`（447–523 行）

**功能**：前向入口。内部小部件构成：

1. **一连串 `moe_check_cuda_contiguous` / `moe_check_same_device`**（453–461 行）
   —— 校验每个 tensor 是 CUDA、连续、同设备。
2. **dtype 校验**（462–471 行）：四个 code/scale 必须是 `uint8`，`expert_offsets` 必须是 `int32`。
3. **shape 校验**（472–498 行）：
   - `activation_codes [M,K]`
   - `activation_scales [M,K/32]`
   - `packed_weight_codes [E,N,K/2]`
   - `weight_scales [E,N,K/32]`
   - `expert_offsets [E+1]`
4. **offsets 语义校验**（501–509 行）：通过 `moe_copy_offsets` 拷回 CPU，
   检查 `front()==0`、`back()==M`、非递减。
5. **分配输出 + 启动**（511–521 行）：`torch::empty({M,N}, float32)`，取 stream，
   调 `moe_launch_grouped_gemm_fwd_strict`。

### ③ `moe_mxfp8_mxfp4_grouped_gemm_backward`（525–590 行）

**功能**：反向入口（只算 dX）。结构与 forward 几乎对称，区别：

- 输入是 `dy`（BF16 `[M,N]`），不需要 activation code/scale。
- `K` 由 `packed_k * 2` 反推（557 行），因为权重只有打包后的 `K/2`。
- 输出 `dx` 是 FP32 `[M,K]`。
- 调 `moe_launch_grouped_gemm_bwd_strict`。

---

## 4. L2 调度/校验层

| 函数 | 位置 | 功能 |
|---|---|---|
| `moe_check_cuda_contiguous` | 195–198 | 校验 tensor 是 CUDA 且连续 |
| `moe_check_same_device` | 200–205 | 校验与第一个 tensor 同设备 |
| `moe_copy_offsets` | 210–214 | 把 int32 offsets 拷到 CPU vector（仅用于校验） |
| `moe_check_h100_sm90` | 216–229 | 若启用了 SM90 宏则校验 `major==9`（**当前未被调用，死代码**） |
| `moe_prepare_experts` | 231–270 | **host 端**预计算每个 expert 的起始指针 + stride（vLLM `__get_group_gemm_starts` 的镜像，当前也未被 hot path 使用） |
| `moe_cuda_skeleton_unimplemented` | 272–275 | `[[noreturn]]` 统一抛「未实现」错 |
| `moe_launch_grouped_gemm_fwd_strict` | 410–420 | 算 grid 并 `<<<>>>` 启动前向 kernel |
| `moe_launch_grouped_gemm_bwd_strict` | 422–431 | 同上，反向 |

### 关键实现：两个 launch 函数

```cpp
constexpr int kMoeStrictBlock = 128;   // 408 行，每个 block 128 线程
const std::int64_t total = M * N;       // 总输出元素数
const int grid = (total + 127) / 128;   // 向上取整分 block
moe_grouped_gemm_fwd_strict<<<grid, 128, 0, stream>>>(...);
```

**设计**：每个线程算一个输出元素，grid 用「总数 / 128 向上取整」，kernel 内部再做
`if (idx >= total) return;` 越界保护。

### 附：`moe_prepare_experts`（231–270 行）详解

这是 vLLM `__get_group_gemm_starts`（设备端 prep kernel）的 **host 端镜像**，逐字段对应：

| vLLM `__get_group_gemm_starts` | 本文件对应字段 | 含义 |
|---|---|---|
| `a_offsets[i] = a_base + expert_offset * half_k` | `activation_codes_ptr + row_begin*K` | A 起始指针 |
| `a_scales_offsets[i] = a_scales_base + sf_offset*gk` | `activation_scales_ptr + row_begin*blocks` | A scale |
| `b_offsets[i] = b_base + expert_id*n*half_k` | `packed_weight_codes_ptr + e*N*(K/2)` | B（按 expert 分块） |
| `b_scales_offsets[i] = b_scales_base + expert_id*n*gk` | `weight_scales_ptr + e*N*blocks` | B scale |
| `out_offsets[i] = out_base + expert_offset*n` | `output_ptr + row_begin*N` | C 起始指针 |
| `a/b/c_strides` | 各 stride 字段 | stride |

**关键区别**：vLLM 在 **device** 上算（喂给 CUTLASS grouped GEMM）；本文件在 **host** 上算，
且当前 strict kernel 并不消费这个 vector（而是用 `moe_find_expert` 现场反推）。

---

## 5. L3 设备计算层

### 5.1 三种 MX 解码器（codec，最底层原子操作）

#### ① `moe_e4m3_to_f32`（285–295 行）—— 激活解码

OCP **E4M3**（1 符号 / 4 指数 / 3 尾数，bias 7）：

```cpp
sign = code >> 7;                      // 取符号位
exp  = (code >> 3) & 0xF;              // 取 4 位指数
mant = code & 0x7;                     // 取 3 位尾数
frac = mant * (1.0f/8.0f);             // 尾数小数部分
if (exp == 0) return ± ldexpf(frac, -6);   // 次正规数：2^-6 * mant/8
value = ldexpf(1.0f + frac, exp - 7);  // 正规数：2^(exp-7) * (1 + mant/8)
return sign ? -value : value;
```

`ldexpf(x, e)` = `x * 2^e`，用精确的指数缩放避免浮点误差，保证与 oracle 逐字节一致。

#### ② `moe_e2m1_nibble_to_f32`（299–303 行）—— 权重解码

OCP **E2M1**（1 符号 / 2 指数 / 1 尾数），4 bit 打包在一个 nibble 里。用查表法：

```cpp
constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
mag = kMag[nibble & 0x7];          // 低 3 位 → 幅值
return (nibble & 0x8) ? -mag : mag; // bit3 是符号
```

这些值在 FP32 里**都是精确的**（0.5/1/1.5/2/3/4/6），所以查表无损。

#### ③ `moe_e8m0_to_f32`（306–308 行）—— scale 解码

OCP **E8M0** 是纯指数格式，表示 `2^(code-127)`：

```cpp
return ldexpf(1.0f, code - 127);   // 1.0 * 2^(code-127)
```

---

### 5.2 `moe_find_expert`（313–322 行）—— 专家路由

**功能**：给定行号 `m`，返回它属于哪个 expert。行已按 expert 排序，`offsets[e] <= m < offsets[e+1]`。

```cpp
int lo = 0, hi = E - 1;
while (lo < hi) {
  int mid = (lo + hi + 1) >> 1;         // 向上取整的二分
  if (offsets[mid] <= m) lo = mid;       // m 在右半
  else hi = mid - 1;                     // m 在左半
}
return lo;
```

这是「找最后一个满足 `offsets[mid] <= m` 的位置」的二分。空 expert
（`offsets[e]==offsets[e+1]`）自动跳过，因为没有 m 落进它。

---

### 5.3 前向 kernel `moe_grouped_gemm_fwd_strict`（327–366 行）

**线程映射**：一个线程负责一个 `(m, n)` 输出元素。

```cpp
idx = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程号
if (idx >= M*N) return;
m = idx / N;  n = idx % N;                     // 反解出 (m, n)
e = moe_find_expert(m, expert_offsets, E);     // 路由到专家
```

**指针定位**（4 个基址，全部按打包布局偏移）：

```cpp
blocks = K >> 5;                                  // K/32 个 scale 块
a_row   = a_codes + m * K;                        // 第 m 行 E4M3 code
a_srow  = a_scales + m * blocks;                  // 第 m 行 E8M0 scale
w_row   = w_codes + (e*N + n) * (K >> 1);         // 专家 e、第 n 列的 E2M1 打包权重
w_srow  = w_scales + (e*N + n) * blocks;          // 专家 e、第 n 列的权重 scale
```

**双层累加**（外层按 32 元素块，内层块内按 k 升序）：

```cpp
float acc = 0.0f;
for (int j = 0; j < blocks; ++j) {                 // 外层：块 j
  float partial = 0.0f;
  #pragma unroll
  for (int kk = 0; kk < 32; ++kk) {                // 内层：块内 kk
    int k = (j << 5) + kk;
    float av = moe_e4m3_to_f32(a_row[k]);          // 解激活
    uint8_t nib = (w_row[k >> 1] >> ((k & 1) << 2)) & 0xF;  // 取权重 nibble
    partial = __fadd_rn(partial, __fmul_rn(av, moe_e2m1_nibble_to_f32(nib)));
  }
  float sc = __fmul_rn(__fmul_rn(partial, moe_e8m0_to_f32(a_srow[j])),
                       moe_e8m0_to_f32(w_srow[j]));   // 先乘 a_scale 再乘 w_scale
  acc = __fadd_rn(acc, sc);
}
out[m*N + n] = acc;
```

**三个关键点**：

1. **nibble 提取**：`w_row[k>>1]` 是包含第 k 个权重的那一字节；`(k&1)<<2` 决定位移量
   ——偶 k 取低 4 位（位移 0），奇 k 取高 4 位（位移 4）。`& 0xF` 隔离出 nibble。
2. **`__fmul_rn` / `__fadd_rn`**：强制「乘 / 加各自独立舍入到最近偶数」
   （round-to-nearest-even），**禁止 FMA 融合**。这是与 oracle 逐字节相等的关键。
3. **scale 应用顺序**：`(partial * scale_a) * scale_w`，括号顺序固定，也是 bit-exact 契约的一部分。

---

### 5.4 反向 kernel `moe_grouped_gemm_bwd_strict`（370–406 行）

**线程映射**：一个线程负责一个 `dX[m,k]` 元素。

```cpp
idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= M*K) return;
m = idx / K;  k = idx % K;
e = moe_find_expert(m, expert_offsets, E);
```

**预计算**：

```cpp
b        = k >> 5;              // k 属于哪个 32 块
half_k   = k >> 1;              // k 对应的字节下标（2 个 fp4 打包）
nib_shift = (k & 1) << 2;       // 该 k 是低 nibble(0) 还是高 nibble(4)
w_codes_e = w_codes + e * N * (K>>1);   // 专家 e 的权重 code 基址
w_scales_e = w_scales + e * N * blocks; // 专家 e 的权重 scale 基址
dy_row = dy + m * N;                    // 第 m 行 dy
```

**沿 N 累加**（dX = dy × W^T，转置权重）：

```cpp
float acc = 0.0f;
for (int n = 0; n < N; ++n) {
  uint8_t nib = (w_codes_e[n*(K>>1) + half_k] >> nib_shift) & 0xF;   // W[e,n,k]
  float wfull = __fmul_rn(moe_e2m1_nibble_to_f32(nib),
                          moe_e8m0_to_f32(w_scales_e[n*blocks + b])); // 反量化权重
  float wbf = __bfloat162float(__float2bfloat16(wfull));              // 截断到 BF16
  acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(dy_row[n]), wbf));
}
dx[m*K + k] = acc;
```

**关键点**：`__float2bfloat16` 把反量化后的权重**舍入成 BF16**（对应 oracle 的
`w_full.to(bf16)`），体现「BF16 backward」契约——反向在 BF16 精度下做，但累加器仍是 FP32。

---

## 6. 数据流 Pipeline 图

### 6.1 前向数据流

```
        ┌───────────────┐   ┌────────────────────┐   ┌──────────────────────┐
        │ activation    │   │ packed weight      │   │ expert_offsets       │
        │ E4M3 [M,K]    │   │ E2M1 nibbles       │   │ int32 [E+1]          │
        └───────┬───────┘   │ [E,N,K/2]          │   └──────────┬───────────┘
                │           └─────────┬──────────┘              │
   ┌────────────┴───────────┐         │                         │
   │ activation scales     │         │   ┌─────────────────────┴───┐
   │ E8M0 [M,K/32]         │         │   │ weight scales           │
   └────────────┬───────────┘         │   │ E8M0 [E,N,K/32]         │
                │                     │   └────────────┬────────────┘
                ▼                     ▼                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  一个线程 = 一个输出元素 (m,n)                                  │
   │  1) moe_find_expert(m)  →  e                                  │
   │  2) 定位 a_row / a_srow / w_row / w_srow                       │
   │  3) 外层循环 block j（K/32 个）:                               │
   │       内层 kk∈[0,32):                                         │
   │         av  = E4M3_decode(a_row[k])                           │
   │         nib = (w_row[k>>1] >> ((k&1)<<2)) & 0xF               │
   │         wv  = E2M1_decode(nib)                                │
   │         partial += av * wv        (__fmul_rn + __fadd_rn)     │
   │       sc  = (partial * E8M0_decode(a_srow[j]))                │
   │              * E8M0_decode(w_srow[j])                         │
   │       acc += sc                                                │
   └──────────────────────────────────────────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │ output FP32   │
        │ [M,N]         │
        └───────────────┘
```

### 6.2 反向数据流（dX only）

```
   dy BF16 [M,N]        weight codes E2M1x2 [E,N,K/2]   weight scales E8M0 [E,N,K/32]
        │                          │                             │
        ▼                          ▼                             ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  一个线程 = 一个 dX 元素 (m,k)                                       │
   │  1) moe_find_expert(m) → e                                          │
   │  2) 预计算 b=k>>5, half_k=k>>1, nib_shift=(k&1)<<2                  │
   │  3) 沿 n 累加:                                                       │
   │       nib   = 取出 W[e,n,k] 的 nibble                                │
   │       wfull = E2M1_decode(nib) * E8M0_decode(w_scales_e[n,b])       │
   │       wbf   = float2bfloat16(wfull)   ← 反量化后截断到 BF16         │
   │       acc  += dy[m,n] * wbf            (__fmul_rn + __fadd_rn)      │
   └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
   dx FP32 [M,K]
```

### 6.3 三种编码的 bit 级解码图

```
E4M3 (activation code, 1 byte):
  ┌───┬───────────┬─────────┐
  │ s │ exp(4bit) │ mant(3) │   →  value = ± 2^(exp-7) · (1 + mant/8)
  └───┴───────────┴─────────┘        (exp==0 → 次正规 2^-6·mant/8)

E2M1 (weight, 半个字节 / nibble):
  ┌───┬──────┬────┐
  │ s │exp(2)│m(1)│   →  value = ± kMag[ {exp,m} ]
  └───┴──────┴────┘         kMag = {0, 0.5, 1, 1.5, 2, 3, 4, 6}
   打包: 1 byte 存 2 个元素，偶 k 在低 nibble，奇 k 在高 nibble

E8M0 (scale, 1 byte):
  ┌────────────────┐
  │   exp(8bit)    │   →  scale = 2^(code - 127)
  └────────────────┘         code=127 → 1.0；code=255 → NaN(拒绝)
```

---

## 7. 组件依赖关系总图

```
L1:  forward/backward/act_quant  (对外 API)
        │ 调用
        ▼
L2:  moe_launch_*_strict  ──校验──>  moe_check_* / moe_copy_offsets
        │ <<<grid,128,0,stream>>>      （moe_prepare_experts / moe_check_h100_sm90 目前未接线）
        ▼
L3:  moe_grouped_gemm_fwd_strict / bwd_strict
        │                    │
        ├─ 路由 ── moe_find_expert (二分查找)
        │
        └─ 解码 ── moe_e4m3_to_f32 (仅 fwd)
                 ├ moe_e2m1_nibble_to_f32 (fwd+bwd)
                 └ moe_e8m0_to_f32      (fwd+bwd)
```

---

## 8. 关键语法点速查

| 语法 | 含义 |
|---|---|
| `__global__` | 从 host 启动、device 执行的 kernel |
| `__device__ __forceinline__` | device 函数，强制内联（消除调用开销） |
| `__restrict__` | 告知编译器指针不重叠，便于优化 |
| `__fmul_rn(x,y)` | FP32 乘法 + round-to-nearest-even，**不融合** |
| `__fadd_rn(x,y)` | FP32 加法 + RNE，**不融合** |
| `__bfloat162float` / `__float2bfloat16` | BF16 ↔ FP32 转换 |
| `ldexpf(x,e)` | 精确计算 `x · 2^e` |
| `<<<grid, block, shmem, stream>>>` | kernel 启动配置 |
| `TORCH_CHECK(cond, msg)` | PyTorch 侧断言，失败抛 C++ 异常 |
| `data_ptr<T>()` | 取 tensor 的设备端裸指针（类型 T） |
| `[[noreturn]]` | 标记函数永不返回（总是抛异常） |
| `constexpr` | 编译期常量 |
| `static_cast<T>(x)` | 显式类型转换 |
| `#pragma unroll` | 提示编译器展开循环 |

---

## 9. 一句话总结

- **L1** 三个 PyTorch 入口负责「收 tensor → 校验 → 分发」，其中 act_quant 是 fail-closed 占位。
- **L2** launch 函数负责「算 grid/block 并启动 kernel」，另外 host 端还预留了
  `moe_prepare_experts`（vLLM prep kernel 的镜像）但未接入 hot path。
- **L3** 两个 strict kernel 用「一线程一输出元素 + 二分路由 + 软件解码 + 串行升序累加」
  实现 bit-exact 的分组 GEMM，前向复用 E4M3/E2M1/E8M0 三种解码器，反向复用后两种。
