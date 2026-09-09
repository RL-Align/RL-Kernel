# SM90 WGMMA 高性能 Grouped GEMM 设计文档

| 项 | 内容 |
| --- | --- |
| 版本 | v1.0（设计稿） |
| 目标算子 | `mxfp8_mxfp4_grouped_gemm_fwd`（P5-4） |
| 目标硬件 | NVIDIA H100 / SM90（Hopper） |
| 数值立场 | 本方案为独立性能 profile（`p5-wgmma-sm90-v1`），不顶替 strict 路径的字节级一致性 |

---

## 目录

1. [设计目标与约束](#0-设计目标与约束)
2. [总体架构：Persistent Kernel + 全局 Work Queue](#1-总体架构persistent-kernel--全局-work-queue)
3. [线程组织：CTA 内两个 Warpgroup](#2-线程组织cta-内两个-warpgroup)
4. [TMA 描述符（Host 端）](#3-tma-描述符host-端)
5. [软件流水线（多级 mbarrier）](#4-软件流水线多级-mbarrier)
6. [fp4 → fp8 解码（Producer 侧，smem 内）](#5-fp4--fp8-解码producer-侧smem-内)
7. [WGMMA 指令形态（SM90 fp8）](#6-wgmma-指令形态sm90-fp8)
8. [Epilogue：E8M0 scale 后移](#7-epiloguee8m0-scale-后移)
9. [内存布局总结](#8-内存布局总结)
10. [Tile 分级](#9-tile-分级)
11. [优化清单（按收益排序）](#10-优化清单按收益排序)
12. [主循环伪代码骨架](#11-主循环伪代码骨架)
13. [与 prep kernel 的最终分工](#12-与-prep-kernel-的最终分工)
14. [风险与偏离记录](#13-风险与偏离记录)

---

## 0. 设计目标与约束

| 项目 | 内容 |
| --- | --- |
| 语义 | MoE 分组 GEMM：每个 expert 用自己的权重 `W[e]` 对自己的 token 组做矩阵乘 |
| 输入 | `A` = MXFP8 激活（E4M3，按 token 已按 expert 重排）；`W` = MXFP4 冻结权重（E2M1，nibble 打包）；`expert_offsets` = 每个 expert 的 token 区间 |
| 输出 | `C` = BF16 结果 `[M, N]` |
| 块粒度 | 32 元素一 block，E8M0 scale |
| 核心约束 | 权重冻结（无 dW），backward 为 BF16，FP32 累加 |

### 三条不可违背的数值语义

> 虽然本 profile 会偏离，但需显式记录偏离点：

1. K 方向升序串行累加
2. mul/add 分开 round（无 FMA 融合）
3. 括号 `(partial × sa) × sw`

**偏离说明**：WGMMA 路径在 epilogue 里把 scale 后移、用 fp22 累加，这两点偏离 strict，故只能作为独立 profile 存在。

---

## 1. 总体架构：Persistent Kernel + 全局 Work Queue

不采用「每 expert 一次 launch」，而是一次 launch 处理所有 expert：

```
launch：gridDim = num_SMs × 2（固定），blockDim = 256（固定）
每个 CTA 常驻，循环抢任务：
    e = atomicAdd(&g_work_counter, 1)
    if e >= E: break
    m_e = expert_m[e]
    if m_e == 0: continue          // 空 expert 跳过
    // 本 CTA 处理 expert e，M 大则内部循环多个 M-tile
    for (mt = 0; mt < m_e; mt += TM):
        mainloop_one_tile(...)
```

**为什么这样**：

- grid/block 大小在 launch 时固定，不随 `m_e` 变化而重算，避免几百次微型 launch 的启动开销与 tail 效应。
- `atomicAdd` 抢任务天然负载均衡，大 expert 被不同 CTA 分块处理。
- 空 expert 直接 `continue`，零开销跳过。

---

## 2. 线程组织：CTA 内两个 Warpgroup

```
CTA (256 线程 = 8 warps)
├── Warpgroup 0 — Producer (4 warps = 128 线程)
│     职责：TMA 异步搬运（gmem→smem）
│           nibble → E4M3 解码（smem 内）
│           mbarrier.arrive 通知 Consumer
│
└── Warpgroup 1 — Consumer (4 warps = 128 线程)
      职责：mbarrier.try_wait 等数据就绪
           wgmma.mma_async 算 GEMM
           读寄存器 D 片段做 epilogue（乘 scale + 写回）
```

**关键点**：

- Producer/Consumer 是同一 CTA 内的两个 warpgroup，**不是两个 block**。
- 用 `setmaxnreg` 分别设寄存器上限：Consumer 需要大寄存器存 D 累加器，Producer 需要更多线程带宽。
- Consumer 至少 1 warp 可发 wgmma，多 warp 让多个 tile 同时在飞以隐藏延迟。

---

## 3. TMA 描述符（Host 端）

用 prep kernel（`__get_group_gemm_starts` 的类比）算出每个 expert 的 `a_off / b_off / scale_off / m_e` 后，host 端用 `cuTensorMapEncodeTiled` 为每个 expert 编描述符：

```cpp
// 权重 B：每 expert 一张描述符，uint8 打包态（不解码）
CUtensorMap desc_b[E];
for (int e = 0; e < E; ++e) {
  cuTensorMapEncodeTiled(
    &desc_b[e],
    CU_TENSOR_MAP_DATA_TYPE_UINT8,   // fp4 nibble 打包态按 uint8 搬
    2,                                // 2D
    b_base + expert_b_off[e],
    size, stride, box, estride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_NONE,       // 不解码，swizzle 关
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
```

**要点**：

- TMA 搬的是打包 nibble（uint8，2 元素/字节），**不做解码**，解码留给 Producer。
- 地址/偏移已 bake 进描述符，per-expert 描述符最直接。
- 必须满足 128 字节对齐（对应 prep 的 `% 128` 断言）。

---

## 4. 软件流水线（多级 mbarrier）

重叠靠 N-stage 流水线，smem 开 2~4 份缓冲，mbarrier 用 phase 奇偶翻转同步：

```
smem 布局（per stage）：
    sB_packed[STAGES][...]   — TMA 直接落地（打包态 uint8）
    sB_fp8   [STAGES][...]   — 解码后的 E4M3
    mbar[STAGES]             — 每 stage 一个 mbarrier

Producer 循环：
    cp.async.bulk.tensor.2d ... (sB_packed[stage], desc_b, {x,y}, mbar[stage])
    mbarrier.arrive.expect_tx(mbar[stage], bytes)
    // TMA 完成 → 读 sB_packed 解码 → 写 sB_fp8[stage]
    mbarrier.arrive(mbar[stage])         // 通知 Consumer

Consumer 循环：
    mbarrier.try_wait.parity(mbar[stage], phase)
    wgmma.mma_async(... sB_fp8[stage] ...)
    phase ^= 1
```

Producer 在 stage `i+1` 上 TMA+解码时，Consumer 在 stage `i` 上 WGMMA，二者并行，隐藏 TMA 与解码延迟。

---

## 5. fp4 → fp8 解码（Producer 侧，smem 内）

分两段，解码只发生在 smem，**绝不写回 gmem**：

1. **TMA 搬运**：gmem 的打包 nibble → `sB_packed`（纯搬运，无解码）。
2. **Producer 解码**：读 `sB_packed`，用 E2M1→E4M3 解码（`kMag[8]={0, 0.5, 1, 1.5, 2, 3, 4, 6}` 映射），在寄存器里解出值，再 `st.shared` 写 `sB_fp8`（E4M3）。

> E2M1 → E4M3 是无损精确（E2M1 的 16 个值全在 E4M3 表示范围内），此解码不引入误差。Consumer 读 `sB_fp8`，完全不接触 nibble。

---

## 6. WGMMA 指令形态（SM90 fp8）

Consumer 用 fp8 `wgmma.mma_async`（A 在 smem，B 在 smem，累加到 FP32 寄存器）：

```
wgmma.mma_async.sync.aligned.m64n128k16.f32.e4m3.e4m3
  {d0..d7},                    // D 累加器在寄存器
  desc_a, desc_b,              // A/B 的 smem 描述符
  scale_a, scale_b,            // fp8 WGMMA 可选乘性 scale
  1;                           // imm
```

- `A` = 激活（E4M3，P5-1 量化输出，直接 TMA 进 smem）。
- `B` = 解码后的权重（E4M3，来自 `sB_fp8`）。
- tile 形状 `m64n128k16` 由架构约束，128/256 对齐到 warp。

**SM90 硬件事实（必须记住）**：

- SM90 **没有 TMEM**（那是 SM100 的东西），累加器在寄存器。
- SM90 fp8 WGMMA 内部累加是 ≈fp22 降精度，因此本路径不能顶替 strict。
- WGMMA **无 FP4 操作数**，所以必须先把 E2M1 解码成 E4M3（第 5 节）。

---

## 7. Epilogue：E8M0 scale 后移

block scale（每 32 元素一个 E8M0）不进 WGMMA 主循环，放在 epilogue：

```
D 在寄存器（fp32）
  → 每个 32-block 对应的 (sa × sw) 乘到对应列/块
  → round 成 BF16
  → 写回 gmem
```

- scale 按 K 的 32-block 分，K-tile（256 = 8×32）正好对齐 8 个 scale-block，epilogue 按列查 scale 乘回。
- 此步偏离 strict 的 `(partial×sa)×sw` 顺序，故属 perf profile。

---

## 8. 内存布局总结

| 数据 | gmem 存放 | smem 存放 | 谁负责转换 |
| --- | --- | --- | --- |
| A 激活 | E4M3 打包（每元素 1B） | E4M3（TMA 直搬） | TMA |
| W 权重 | E2M1 nibble 打包（2 元素/字节） | 先 `sB_packed` 再 `sB_fp8`（E4M3） | Producer 解码 |
| scale | E8M0（每 32 元素 1B） | 直接读 | epilogue |

---

## 9. Tile 分级

只有 M 需要按 `m_e` 分级（N 是模型常量，K 是编译期常量）：

| 条件 | M-tile | 处理方式 |
| --- | --- | --- |
| `m_e > 256` | 256 | 内部循环 `ceil(m_e/256)` 次 |
| `128 < m_e ≤ 256` | 128 | 单次 |
| `m_e ≤ 128` | 128（或直接一次性） | 单次 |

- **N tile** = 128 / 256（架构期常量，wgmma 指令形态决定），不随 `m_e` 变。
- **K tile** = 256 = 8×32，对齐 E8M0 scale-block，便于 epilogue 整齐落 scale。

---

## 10. 优化清单（按收益排序）

1. **B（权重）常驻 smem，跨 M-tile 复用** — per-expert 最大红利，`W[e]` 只搬一次，之后只有 A 在流。
2. **多级流水线 + 双缓冲** — 隐藏 TMA/解码延迟，比单纯加大 tile 更有效。
3. **解码放 Producer 侧** — Consumer 只读 fp8，不碰 nibble。
4. **空 expert 跳过** — `if (m_e == 0) continue;`
5. **work-queue chunking** — `atomicAdd` 一次抢一组，减少全局原子竞争。
6. **smem swizzle 匹配** — 解码写 fp8 时 swizzle 与 wgmma 要求一致，避免 bank conflict。
7. **cluster 广播（可选）** — 大 expert 多 CTA 分头算时用 TMA multicast 广播 `W[e]`。
8. **分级 tile 先简化后优化** — 先做固定 tile + 边界 tail 跑通，再决定是否按 `m_e` 分级。

---

## 11. 主循环伪代码骨架

```cpp
__device__ unsigned g_work = 0;

__global__ void grouped_gemm_sm90(
    const CUtensorMap* desc_a, const CUtensorMap* desc_b,
    const uint8_t* scales_a, const uint8_t* scales_w,
    float* out, const int* expert_m, const int* a_off, const int* b_off, ...)
{
  while (true) {
    unsigned e = atomicAdd(&g_work, 1);
    if (e >= E) break;
    int M_e = expert_m[e];
    if (M_e == 0) continue;

    // B 权重解码进 smem（只一次，复用）
    load_and_decode_W(desc_b[e]);          // TMA → sB_packed → sB_fp8

    for (int mt = 0; mt < M_e; mt += TM) {
      // Producer: TMA 搬 A tile → sA
      // Consumer: wgmma.mma_async(A, B) → D 寄存器
      // epilogue: D × (sa × sw) → round BF16 → out
    }
  }
}
```

---

## 12. 与 prep kernel 的最终分工

| 职责 | 位置 |
| --- | --- |
| tensor → data_ptr / shape 校验 | host 绑定层 |
| per-expert 偏移、M、stride、scale 布局、对齐校验 | prep kernel（`__get_group_gemm_starts`） |
| 用偏移生成 TMA 描述符 | host（`cuTensorMapEncodeTiled`） |
| nibble → E4M3 解码 | 主 kernel Producer，smem 内 |
| GEMM 数值计算 | 主 kernel Consumer，wgmma |
| scale 乘回 + 写 out | 主 kernel epilogue |

---

## 13. 风险与偏离记录

| 风险/偏离 | 说明 | 缓解 |
| --- | --- | --- |
| fp22 累加精度 | 非 strict | 独立 profile，注册独立 `numeric_profile` |
| scale 括号顺序 | epilogue 后移 `(sa×sw)` | 同上，禁止用于 strict 验收 |
| nibble 解码开销 | Producer 多一步 | 流水线重叠，E2M1→E4M3 无损 |
| 空 expert 尾效 | 大量小 expert | work-queue + `continue` + chunking |
| 描述符开销 | 每 expert 一张 | host 一次性生成，kernel 内复用 |

---

## 一句话总结

> SM90 上的高性能 grouped GEMM = **persistent CTA + atomic work-queue 调度**，CTA 内分 **Producer**（TMA 搬 nibble + smem 解码成 E4M3）与 **Consumer**（fp8 WGMMA 累加）两个 warpgroup，靠 **多级 mbarrier 流水线**重叠，epilogue 后移 E8M0 scale——prep kernel 只算 per-expert 地址/形状/scale 布局供 host 编 TMA 描述符，本身不解码不搬数。
