# P5-4 Triton 版本实现计划（mxfp8×mxfp4 grouped GEMM，bit-exact strict）

## 背景（Context）

P5-4 grouped GEMM 现有两条链路：FP32 oracle（`oracle.py`，黄金字节）与 CUDA strict（`csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu`，H100/SM90 专用，`__fmul_rn`/`__fadd_rn` 逐位一致）。CUDA strict 在 prep kernel 启动前经 `moe_check_h100_sm90` 强制 SM90（非 SM90 设备 fail-closed）。

本次新增一条 **Triton 后端**：实现 `mxfp8_mxfp4_grouped_gemm_fwd/bwd`，数值语义与 oracle **逐位一致（bit-exact strict）**，从而能通过 `scripts/check_p5.py` 的 sha256 门禁。同样**仅适配 SM90（H100）**，不追求非 H100 / ROCm 可移植性。代码结构镜像 CUDA 文件（prep kernel / 传参 struct / 计算层），统一英文 banner 注释。

**已确认的两个决策**：
- Scope：**仅 P5-4 grouped GEMM**（fwd+bwd），其余算子（P5-1/2/3/5）留在 oracle 上。
- 数值：**bit-exact strict**（逐元素 mul/add 无 FMA，串行升序归约），不用 `tl.dot`。

## 范围与接入

- 新增文件 `rl_engine/moe/triton_grouped_gemm.py`（含 Triton 内核 + `TritonP5GemmProvider`）。
- 改 `rl_engine/moe/provider.py`：`resolve_provider` 的 `aliases` 增加 `"triton": "rl_engine.moe.triton_grouped_gemm:TritonP5GemmProvider"`（与 `"cuda"` 并列）。
- **不改**：`ops.cpp`、`MXTensor`、`oracle.py`、`mx_format.py`、`check_p5.py`、`.cu`。
- fail-closed：triton 未安装时 `TritonP5GemmProvider` 构造/调用必须 raise，禁止静默回退 oracle；与 `CudaP5GemmProvider` 对齐（独立 numeric profile，不顶替 strict path）。

## 数值契约（bit-exact，逐位对齐 oracle）

Triton 内核必须**精确复现** oracle 的归约顺序与舍入；任何一处 FMA 融合都会导致 sha256 失配。

### fwd（对齐 `oracle.mxfp8_mxfp4_grouped_gemm_fwd` + `_block_scaled_dot`）

对每个输出元素 `(m, n)`，按以下冻结顺序：

```
acc = 0
for j in 0 .. K/32-1:                      # 块升序
    partial = 0
    for kk in j*32 .. (j+1)*32-1:          # 块内升序 k
        partial = fadd_rn(partial, fmul_rn(a_elems[m,kk], w_elems[n,kk]))
    acc = fadd_rn(acc, fmul_rn(fmul_rn(partial, sa[m,j]), sw[n,j]))   # (partial*sa)*sw
```

- decode 全精确（e4m3/e2m1/e8m0 → fp32 无舍入）。
- 关键：块内 partial 先算完 32 次 mul/add，**再**乘 `sa`、`sw`，最后累加进 `acc`；括号顺序 `(partial*sa)*sw` 冻结。

### bwd（对齐 `oracle.mxfp8_mxfp4_grouped_gemm_bwd` + `_serial_dot`）

```
# 先 dequant W 到 BF16（每 expert，寄存器内，不写回 global）
w_full[n,k] = e2m1_decode(w_codes[n,k]) * e8m0_decode(w_scales[n, k//32])   # fp32, block-32 scale 折叠
w_bf16[n,k] = round_bf16(w_full[n,k])                                       # fp32 -> BF16 RNE

# 再归约（升序 n，无 block-scale 在归约中）
dx[m,k] = 0
for n in 0 .. N-1:                          # 升序 n
    dx = fadd_rn(dx, fmul_rn(to_fp32(dy_bf16[m,n]), to_fp32(w_bf16[n,k])))
```

- **注意**：bwd 与 fwd 归约语义不同 —— bwd 先把 scale 折进 W 并 round 到 BF16，归约只在 N 上做，无 per-32-block 结构。dy 输入为 BF16。

### 舍入手段

- 禁用 `tl.dot`（张量核 FFMA，非逐位）。用逐元素 `fmul_rn`/`fadd_rn`（Triton libdevice `__nv_fmul_rn`/`__nv_fadd_rn`，对应 CUDA 的 `__fmul_rn`/`__fadd_rn`）。
- 若 Triton 版本无显式 `fmul_rn`，退化为 `*`/`+` 并**关闭 FMA 收缩**；以 `check_p5.py` 的 sha256 为最终判定（一旦失配即视为收缩未被关闭）。
- decode 用 **算术解码**（位运算 + 指数缩放，无查表）：e4m3/e2m1/e8m0 → fp32 全精确无舍入，镜像 `.cu` 的 `moe_e4m3_to_f32`/`moe_e2m1_nibble_to_f32`/`moe_e8m0_to_f32`（`ldexp`/`2**k` 语义，避免 `powf` 引入误差）。e2m1 的 8 个幅值 {0,0.5,1,1.5,2,3,4,6} 在 fp32 全精确。
- BF16 舍入**必须用 `tl.cast(x, tl.bfloat16)`（RNE）**，对齐 `torch.Tensor.to(bfloat16)` 与 CUDA `__float2bfloat16`，禁止 RTZ（截断）；以 `check_p5.py` 的 sha256 为最终判定。

## 文件结构

```
rl_engine/moe/triton_grouped_gemm.py
├── module docstring（SPDX + 用途 + numeric profile + fail-closed 说明）
├── triton availability guard（try import triton / tl；失败置 _TRITON_AVAILABLE=False）
├── 常量：_KERNEL_FINGERPRINT（`"mxfp8-mxfp4-grouped-gemm-strict-triton-v1"`，与 CUDA 指纹不同，by design）、pinned BM/BN、block=32
├── launch descriptor struct（dataclass 镜像 C struct）
├── decode helpers（lookup table 生成 + @triton.jit decode）
├── prep kernel（@triton.jit）
├── fwd and bwd strict compute kernel（@triton.jit ×2）
├── host launcher（python：分配 descriptor buffer + launch prep + launch compute）
└── provider（TritonP5GemmProvider）
```

## 注释风格规范（统一，本次核心要求）

采用单行 banner，英文小写 section 名，镜像 `.cu` 的 `/* ************ section ************ */` 形式：

```python
# ********************** launch descriptor struct **********************
```

规则：
1. 每个功能段顶部一个 banner；`# ` 前缀，section 名两侧各 ≥22 个 `*`，左右对称。
2. section 名英文小写，与 `.cu` 对应段同名（见下方分节）。
3. 行内注释统一英文、祈使/陈述语气一致（与 `.cu` 现有注释一致）。
4. 保留字段（`weight_expert_stride`/`weight_scale_expert_stride`）照抄 `.cu` 的反删除注释。

## 分节结构（每节一个 banner）

1. **launch descriptor struct**
   - 定义 `MoeGroupedGemmExpert`（fwd，16 字段）与 `MoeGroupedGemmBwdExpert`（bwd，14 字段）两个 dataclass，字段名/顺序镜像 `.cu` 同名 struct。
   - Triton 无 C struct：descriptor 用 int64 tensor `[E, n_fields]` 表达，`n_fields`=字段数；「指针」→ int64 字节偏移，「stride」→ int64 常量列。每列即一个字段。
   - 保留 `weight_expert_stride`/`weight_scale_expert_stride` 两列 + 反删除注释（future WGMMA per-expert grouped GEMM，strict one-row 不读）。

2. **decode helpers**
   - `@triton.jit` 侧（算术解码，无 host 查表）：`decode_e4m3(code)` / `decode_e2m1(nibble)` / `decode_e8m0(scale)`，纯位运算 + 指数缩放，无舍入；镜像 `.cu` 的三个 codec。
   - BF16 舍入：`round_bf16(fp32)` 用 `tl.cast(_, tl.bfloat16)`（RNE），对齐 `torch.Tensor.to(bfloat16)` / CUDA `__float2bfloat16`，禁止 RTZ。

3. **prep kernel**
   - 镜像 CUDA 的 device-side prep：`@triton.jit`，grid `(E,)`，每个 program 读 `expert_offsets[e]`、`expert_offsets[e+1]`，计算 `row_begin/row_count` 及派生偏移，写入 descriptor tensor `[E, n_fields]`。
   - `expert_offsets` 直接从 device 读，**不 copy 到 host**（与 CUDA prep 一致，避免 host sync）。

4. **fwd and bwd strict compute kernel**
   - fwd：grid `(cdiv(M,BM), cdiv(N,BN))`；每个 program 算一个 BM×BN 输出 tile，每元素寄存器内按「fwd 数值契约」串行升序 k（含 32-block 结构）归约。
   - bwd：grid `(cdiv(M,BM), cdiv(K,BK))`；每元素按「bwd 数值契约」先 dequant W 到 BF16（寄存器内），再串行升序 n 归约。
   - 专家路由：`find_expert(m)` 用**固定 `ceil(log2 E)` 次展开的二分**（E 为 kernel 常量，迭代次数确定），避免数据依赖循环；空 group 天然被跳过（无 m 落进）。
   - BM/BN/BK pinned（如 16/16/16），**无 autotune**（保证确定性，对齐 det_gemm.py 的「autotune disabled」惯例）。
   - 空 group（`row_count==0`）直接跳过，不写输出。

5. **host launcher**
   - python 侧：`triton_grouped_gemm_fwd(a, w, expert_offsets)` / `_bwd(...)`。
   - 分配 descriptor buffer（`torch.empty([E, n_fields], int64, device)`）→ launch prep → launch compute，返回 `[M,N]` / `[M,K]` fp32。
   - 入参校验：`a/w` 为 `MXTensor`（e4m3/e2m1）、tensor is_cuda/contiguous、`expert_offsets` int32 且首 0 末 M 非递减（复用 `contract` 校验，不引入新逻辑）。

6. **provider**
   - `TritonP5GemmProvider(ReferenceProvider)`：只 override `mxfp8_mxfp4_grouped_gemm_fwd/bwd`，其余留在 oracle。
   - `name="triton-p5-gemm"`，`numeric_profile="p5-strict-triton-v1"`。
   - `capabilities()`：`backend="triton-p5-strict"`、`geometry=["one-row"]`、`devices=["cuda-sm90"]`、`implemented=["mxfp8_mxfp4_grouped_gemm_fwd", "mxfp8_mxfp4_grouped_gemm_bwd"]`。
   - `provenance()`：补 `geometry/tile/split_k/kernel_fingerprint/triton_version/cuda_version`；`kernel_fingerprint` 与模块内 `_KERNEL_FINGERPRINT` 常量逐字一致（单一事实源，人工同步），`_KERNEL_FINGERPRINT = "mxfp8-mxfp4-grouped-gemm-strict-triton-v1"`。
   - ⚠️ **此 Triton 指纹与 CUDA 指纹（`"mxfp8-mxfp4-grouped-gemm-strict-v1"`）不同，且这是有意为之（by design）**——两者是不同的后端 / numeric profile，绝不能复用同一指纹。
   - fail-closed：`_TRITON_AVAILABLE` 为假时 raise；设备非 SM90 时 raise（镜像 `moe_check_h100_sm90`，仅适配 SM90）。

## 验证

在有 GPU（CUDA/ROCm）的机器上：

```
python scripts/check_p5.py --provider triton --device cuda
```

- 所有 e2e case（含 `uneven_experts` 空 group）PASS，`y/dx` 与 oracle 逐字节一致（sha256 相等）。
- 顺带断言：base weight 无 dW（`w.codes`/`w.scales` sha256 前后不变）；bwd 只回 `dX`。
- 断言 `resolve_provider("triton").provenance()` 含上述字段；`capabilities()["geometry"]==["one-row"]`。

对拍输入复用 oracle/fixtures：`a=mx_quantize(x,"e4m3")`、`w`（e2m1 codes/scales）、`expert_offsets`；`y_triton` vs `oracle.mxfp8_mxfp4_grouped_gemm_fwd(a,w,offsets)` 逐位；bwd 同理。

## 备注（非本次范围）

- P5-1/2/3/5 不动；P5-1 激活量化仍 fail-closed。
- `tl.dot`（WGMMA/tensor-core）性能路径 = 独立 numeric profile + 与 oracle 的 max-diff 报告，本次不交付；不得静默顶替 strict path（与 `P5_4_plan.md` 备注一致）。
- 本次 Triton strict 为可移植参考实现（慢于 tuned GEMM，符合 one-row 参考定位）。
