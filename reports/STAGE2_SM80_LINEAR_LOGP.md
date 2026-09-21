# A100/SM80 native `linear_logp` — Stage 2 性能优化报告

- 日期：2026-09-21
- 分支：`sm80-linear-logp`
- GPU：A100 80GB PCIe，SM80（只使用 GPU 1）
- 范围：BF16、D=4096、V=128256、forward-only、single GPU
- 不变项：online-softmax 数学路径、显式 PoC 调用、不接 registry、无 backward/TP/FP8

## 实现

`csrc/cuda/fused_linear_logp_sm80.cu` 从 Stage 1 的同步单缓冲改为：

- 16-byte `cp.async.cg.shared.global` 搬运；
- K 维两级 shared-memory ping-pong buffer；
- load 与当前 tile 的 WMMA compute 重叠；
- 每个 K step 的 CTA barrier 从 2 次降为 1 次；
- 通用 WMMA fragment 网格，每 warp 可持有多个 16x16 FP32 accumulator；
- 最终采用 BM=32、BN=128、BK=32、8 warps/CTA。

online max/sumexp、target-logit 记录和末尾 merge 的数学逻辑未改变。完整 `[N,V]` logits 仍不 materialize。

## 第一阶段：cp.async 流水本身

保持 BM/BN/BK=16/64/16 时，双缓冲 cp.async 的结果：

| N | Stage 1 (ms) | cp.async 16/64/16 (ms) | 加速 |
|---:|---:|---:|---:|
| 128 | 1017.85 | 262.82 | 3.87x |
| 256 | 1017.94 | 262.84 | 3.87x |
| 512 | 1013.41 | 262.64 | 3.86x |
| 1024 | 1011.30 | 263.27 | 3.84x |
| 2048 | 1011.36 | 264.39 | 3.82x |
| 4096 | 1419.12 | 460.46 | 3.08x |

结论：流水成功隐藏了相当一部分 global-load latency，但小 tile 和低 MMA issuing density 仍是主导限制。

## 第二阶段：tile 联合实验

所有候选都保留 cp.async 双缓冲；没有做“只放大 tile、不做异步流水”的实验。

| BM/BN/BK | warps | N=4096 (ms) | 结论 |
|---|---:|---:|---|
| 32/128/16 | 8 | 251.18 | barrier 次数偏多 |
| **32/128/32** | **8** | **246.12** | **最优** |
| 64/128/32 | 8 | 299.75 | CTA 数减少、accumulator/register pressure 增大 |
| 32/256/32 | 4 | 275.02 | 8-warp 版需 50 KB 静态 shared，超过 48 KB；4-warp 版寄存器压力较高 |
| 32/128/64 | 4 | 458.11 | shared/寄存器与低 warp 数造成严重退化 |

最优 cubin 静态资源：72 registers/thread、31,744 B shared/CTA、256 threads/CTA。按寄存器估算最多 3 CTA/SM，理论 resident-warp occupancy 上限约 37.5%；N=4096 仅 128 CTA，实际 grid 仍不足以在 108 个 SM 上持续维持该上限。

## 最终结果（warmup=5，iters=20）

| N | cuBLAS ms / tok/s | Triton ms / tok/s | Stage 2 ms / tok/s | max abs err | mean abs err | peak activation |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.983 / 130256 | 61.538 / 2080 | 167.373 / 765 | 1.717e-5 | 5.841e-6 | ~0.00 MB |
| 256 | 1.629 / 157169 | 61.339 / 4174 | 167.455 / 1529 | 2.670e-5 | 5.923e-6 | ~0.00 MB |
| 512 | 3.169 / 161577 | 61.392 / 8340 | 167.489 / 3057 | 2.098e-5 | 5.608e-6 | ~0.00 MB |
| 1024 | 6.380 / 160507 | 61.508 / 16648 | 167.512 / 6113 | 2.575e-5 | 5.683e-6 | 0.01 MB |
| 2048 | 12.789 / 160139 | 62.537 / 32749 | 167.608 / 12219 | 2.766e-5 | 5.666e-6 | 0.02 MB |
| 4096 | 25.514 / 160540 | 83.623 / 48982 | **246.116 / 16643** | 2.766e-5 | 5.814e-6 | **0.03 MB** |

所有 Stage 2 shape 均无 NaN/Inf。cuBLAS/materialized 在 N=4096 的峰值激活为 5010 MB；Triton 为 0.05 MB。

## 安全与正确性

- 精度保持在原有约 1e-5 量级；
- 未 materialize `[N,V]`；
- N=16 边界 CTA（含 cp.async zero-fill）通过 compute-sanitizer memcheck；
- `ERROR SUMMARY: 0 errors`。

## 结论与下一步

1. **cp.async 是否成功隐藏 load latency？** 是，孤立改动带来 3.1–3.9x 加速；但尚未完全隐藏。
2. **最优 tile？** BM=32、BN=128、BK=32，8 warps/CTA。
3. **1.4 s 降到多少？** N=4096 从 1419.12 ms 降到 246.12 ms，5.77x 加速。
4. **剩余差距？** 对 Triton 83.62 ms 仍慢 2.94x；对 cuBLAS 25.51 ms 仍慢 9.65x。
5. **瓶颈转移了吗？** 尚未转成纯 bandwidth/compute wall。按权重重读模型，N=4096 等效带宽约 547 GB/s，远低于 A100 峰值和 Triton 的约 1.6 TB/s；有效 BF16 算力约 17.5 TFLOP/s，也远未触及计算峰值。当前主要是 WMMA/load_matrix 指令开销、72-reg register pressure、低实际 occupancy、有限 grid 并行度，以及仍然存在的 K-chain/同步延迟。
6. **是否值得进入手写 mma.sync + swizzle？** 值得。前两阶段已拿到 5.77x，但距离 Triton 仍有接近 3x，且计数模型显示既未到带宽墙也未到计算墙。下一阶段应以 `mma.sync m16n8k16`、`ldmatrix`、shared swizzle/bank-conflict 控制和更紧凑的 accumulator layout 为主；目标是降低 WMMA API 生成的 load/寄存器开销并提高可驻留 warp 数。

