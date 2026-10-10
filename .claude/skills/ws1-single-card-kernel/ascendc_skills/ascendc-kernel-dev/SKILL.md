---
name: ascendc-kernel-dev
description: WS1 Ascend C operator development fundamentals (programming model / kernel structure / tiling design). Read before writing a new .asc kernel - SP/VEC/MTE2/MTE3 pipeline roles, TQue queue mechanics, the four tiling design elements, and how they land in this repo (bisheng build, consolidated pybind in npu_module.cpp, row-granular split for batch invariance). Trigger when writing a new Ascend C kernel, designing tiling, or planning UB buffers.
---

# WS1 Ascend C Operator Development Fundamentals

> Sub-skill of `ws1-single-card-kernel`, curated from cannbot-skills
> (see `ascendc-tiling-design` / `ascendc-api-best-practices` at
> https://gitcode.com/cann/cannbot-skills) and adapted to this repo
> (rl-kernel WS1). For the full methodology invoke the locally installed
> skills `ascendc-tiling-design` and `ascendc-whitebox-design`.

## Programming model (understand before writing any kernel)

Ascend C is SPMD: the same kernel program is replicated across AI Cores,
with `GetBlockIdx()/GetBlockNum()` distinguishing identity. Inside a single
core execution is an **asynchronous pipeline** — compute and data movement
run on separate hardware queues with no automatic synchronization:

| Pipe | Full name | Role | Typical ops |
|------|-----------|------|-------------|
| MTE2 | Memory Transfer Engine (in) | GM → UB/L1 load | `DataCopy` / `DataCopyPad` |
| V | Vector | vector compute | `Add/Mul/Exp/ReduceSum/Cast` |
| M | Matrix (Cube) | matrix compute | Matmul high-level API (WS1 vector ops here generally don't use it) |
| MTE3 | Memory Transfer Engine (out) | UB → GM store | `DataCopy` / `DataCopyPad` |
| S | Scalar | scalar control flow | loop counters, `SetValue`, flag checks |

Memory hierarchy: GM (HBM) → L1 → L0A/L0B/L0C (Cube) → UB (the Vector unit's
working memory; 192KB on DAV_2201, 248KB on DAV_3510). **The WS1
vector/reduction data path is GM→UB→V→UB→GM.**

Cross-pipe dependencies require explicit synchronization: `EnQue/DeQue`
(implicit via queues), `SetFlag/WaitFlag` (explicit), `PipeBarrier`
(coarse-grained last resort). The full 14-rule sync checklist lives in the
locally installed skill `ascendc-sync-audit`.

## Kernel skeleton (maps to this repo's .asc files)

```cpp
extern "C" __global__ __aicore__ void my_op_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR out,
                                                   MyOpTilingData tiling) {
    MyOpKernel op;
    op.Init(x, y, out, tiling);
    op.Process();
}

class MyOpKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR out, MyOpTilingData tiling) {
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileBytes);
        xGm.SetGlobalBuffer((__gm__ T *)x + offset, count);
    }
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t i) {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();
        // GM→UB uses the struct-based overload: DataCopyExtParams (block
        // count/len/strides) + DataCopyPadExtParams<T> (dst/valid sizes, in
        // elements) — a bare element count does not compile on current CANN.
        // Build both structs in Init from the tiling; field semantics per the
        // official DataCopyPad reference.
        DataCopyPad(x, xGm[i * tileLen], copyParams, padParams);
        inQueueX.EnQue(x);                            // MTE2 -> V sync
    }
    __aicore__ inline void Compute(int32_t i) {
        LocalTensor<T> x = inQueueX.DeQue<T>();       // waits for the copy
        LocalTensor<T> y = outQueue.AllocTensor<T>();
        /* vector compute */
        outQueue.EnQue<T>(y);                         // V -> MTE3 sync
        inQueueX.FreeTensor(x);
    }
    __aicore__ inline void CopyOut(int32_t i) {
        LocalTensor<T> out = outQueue.DeQue<T>();     // waits for compute
        // UB→GM takes only DataCopyExtParams (padding applies on load only)
        DataCopyPad(outGm[i * tileLen], out, copyParams);
        outQueue.FreeTensor(out);
    }
};
```

Queue lifecycles differ by direction — state them separately:
- **copy-in**: `AllocTensor → DataCopyPad → EnQue`, then the compute side
  `DeQue`s. Issuing the copy BEFORE `EnQue` is what publishes a filled tensor;
  following a copy-out-style order here lets compute read uninitialized input.
- **copy-out**: the compute side fills the tensor and `EnQue`s it; the store
  side runs `DeQue → DataCopyPad → FreeTensor`.
Never hand-roll `MTE2_MTE3`/`MTE3_MTE2` flags (random corruption or hangs on
real hardware — see the gotchas in the main SKILL.md).

**Host side**: torch bindings are consolidated in `csrc/ascend/npu_module.cpp`
(a single `PYBIND11_MODULE`; individual `.asc` files must not carry their own,
or you get duplicate `PyInit__C_npu` link errors). The build goes through
bisheng (`setup.py` exports `ASCEND_HOME_PATH` automatically).

## Tiling design: four mandatory elements

### 1. Multi-core split
- Core question: how is work distributed across AI Cores? Load balanced,
  reasonable granularity.
- **This repo's fixed pattern (the foundation of batch invariance)**:
  `MAX_BLOCKS=128`, `for (row = GetBlockIdx(); row < N; row += GetBlockNum())`
  with host-side `blockNum = min(N, MAX_BLOCKS)` — each row is processed
  end-to-end by one block, so the instruction sequence depends only on the
  shape, never on batch layout or block assignment.

### 2. UB split
- How much per trip: bounded by UB capacity (192KB on DAV_2201).
- Provide the chunk-size formula; loop over chunks when it doesn't fit.
- Vector op counts must be rounded up to a multiple of `32/sizeof(T)`
  (32B alignment), and every participating UB buffer must be allocated for
  that rounded count — round first, then size (see
  [ascendc-api-best-practices](../ascendc-api-best-practices/SKILL.md)).

### 3. Buffer planning
- List every buffer (inQueue / outQueue / tmpBuf / castBuf ...) with its
  size formula.
- **Reuse and fill**: steps with non-overlapping lifetimes share one `TBuf`;
  grow the tile against the **usable** UB budget — `GetCoreMemSize(UB)`
  reports raw hardware capacity and some architectures reserve part of it,
  so validate the allocatable size on the target arch and leave headroom.
- Double buffering (`BUFFER_NUM=2`) is the standard MTE2/V overlap; remember
  threshold computations scale ×2.

### 4. Branch coverage
- dtype branches: fp32 / bf16 / fp16 (this repo's pytest parametrizes all three).
- shape branches: large/small shapes, rows whose width is not 32B-aligned.
- boundaries: out-of-range targets, strided blocks (>MAX_BLOCKS), multi-tile.

## Operator class routing (common WS1 ops)

| Class | Trait | Typical ops in this repo | Notes |
|-------|-------|--------------------------|-------|
| Elementwise | same input/output shape | silu, swiglu, rmsnorm (elementwise part) | loop over 32B-aligned tiles |
| Reduction | reduce along an axis | rmsnorm, logp, fused linear logp, lm_head | reduction tree decides precision — see the bitwise-consistency classes in the main SKILL.md |
| Broadcast | shapes differ, align by broadcast | broadcast add | vector broadcast via `src1RepStride=0` |
| Copy/lookup | pure byte movement | embedding | must be `torch.equal` vs the golden |
| Sort/TopK | sorting | - | merge-sort based; none in this repo yet |

## Checklist

- [ ] Row-granular split (preserves batch invariance), host `blockNum = min(N, MAX_BLOCKS)`
- [ ] Correct queue order, no hand-rolled cross-pipe flags
- [ ] Vector counts rounded up to `32/sizeof(T)`; buffers sized for the rounded count; GM writeback only real bytes
- [ ] GM scalar reads/writes go through a 32B `DataCopyPad` window + flag sync
- [ ] Scalar math (no exp/log) uses the padded 8-element vector workaround
- [ ] UB buffers reused and filled; double-buffer thresholds ×2
- [ ] All three dtype branches (fp32/bf16/fp16) + unaligned tail coverage
- [ ] No `PYBIND11_MODULE` (consolidated in npu_module.cpp)
