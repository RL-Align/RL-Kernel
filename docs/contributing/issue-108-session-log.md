# ISSUE-108 Session Log

本文档记录本 session 中围绕 RL-Kernel 算子测试框架、CUDA 验证和 upstream 同步的所有关键修改。后续本 session 中每次代码修改都必须继续追加到本文档，记录目标、设计判断、修改文件、验证方式和结果。

## 记录原则

- 使用中文记录工程判断。
- 每个改动都需要说明为什么做，而不只记录改了什么。
- 保持最小增量：一次修改尽量只围绕一个明确问题。
- 每个子任务需要能独立验证；无法验证时必须明确说明原因。
- CUDA 验证必须如实记录环境、命令、误差和失败范围。
- 不把失败路径写成已支持能力。

## 总体目标

本 session 的目标是把算子验证从零散脚本推进为可复用、可扩展、可审查的工程化框架：

- 建立统一 tolerance contract，用于管理不同算子的误差阈值。
- 建立公共 operator check runner，替代单算子专用验证脚本。
- 建立统一 operator input 生成逻辑，覆盖后训练常见算子的基础输入。
- 将测试入口改造成可指定 `op`、`candidate`、`dtype`、`device`、shape 参数的 CLI。
- 同步 upstream/main，吸收 PR #122 中的 SM90 相关修复。
- 在 H20 机器上验证普通 CUDA `fused_logp` 路径。
- 明确 SM90 `fused_linear_logp` 在 CUDA 12.4 下仍未通过。

## 时间线

### 1. tolerance table 和 contract loader

目标：

- 将不同 dtype、op class 的误差容差从测试代码中抽离出来。
- 让误差阈值可以被审查和维护，而不是散落在测试断言中。

修改文件：

- `rl_engine/testing/tolerance.py`
- `rl_engine/testing/tolerance_contract.yaml`
- `tests/test_tolerance_contract.py`

设计判断：

- 使用 YAML 保存 contract，便于人工 review。
- 将容差按 `accuracy.default` 和可选硬件 override 组织。
- `default` 是通用 fallback，不等同于 CPU；CPU、SM90、SM100、ROCm、Ascend 等未来可作为明确 override key。

验证：

- `tests/test_tolerance_contract.py` 验证 contract 可读、结构正确。

结果：

- tolerance contract 框架建立完成。

### 2. operator check runner

目标：

- 建立类似 GoogleTest 思路的算子验证 runner。
- 一个 case 表示一组确定输入和 gold path，一个 candidate 表示被测实现。

修改文件：

- `rl_engine/testing/op_checks.py`
- `tests/test_op_checks.py`
- `rl_engine/testing/__init__.py`

设计判断：

- `OperatorCase` 表示测试对象：`name`、`op_class`、`dtype`、`inputs`、`gold_fn`。
- `CandidateSpec` 表示被测实现：`name`、`fn`、`backend`、`arch_key`。
- runner 负责：
  - 调用 candidate。
  - 调用 gold。
  - flatten 多输出。
  - 按 `op_class + dtype + arch_key` 解析容差。
  - 计算 `max_abs_error`、`mean_abs_error`、`max_rel_error`。
  - 返回结构化 report。

验证：

- `tests/test_op_checks.py` 覆盖 native logp、registry logp、失败 candidate、arch override 等场景。

结果：

- 公共 operator check runner 建立完成。

### 3. `check_operator.py` 从 logp 专用入口改为公共入口

目标：

- 让测试者通过 CLI 指定算子、candidate、dtype、device 和 shape。
- 避免后续每个算子都写一个独立测试脚本。

修改文件：

- `scripts/check_operator.py`

设计判断：

- `check_operator.py` 只负责：
  - 解析参数。
  - 选择 device/dtype。
  - 调用 `make_candidate`。
  - 调用 `make_operator_case`。
  - 调用 `run_operator_suite`。
  - 输出 summary 或 JSON。
- 不在入口中硬编码具体算子实现。

验证：

```bash
python scripts/check_operator.py --op logp --candidate pytorch --dtype fp32 --batch 1 --seq 4 --vocab 17
python scripts/check_operator.py --op logp --candidate registry --dtype bf16 --batch 2 --seq 16 --vocab 257 --json
python -m pytest tests/test_op_checks.py -q
```

结果：

- 公共 CLI 最小闭环通过。

### 4. 抽离 operator specs

目标：

- 避免新增算子时修改测试入口。
- 将算子元信息集中到专门文件。

修改文件：

- `rl_engine/testing/operator_specs.py`

设计判断：

- 每个算子通过 `OperatorSpec` 描述：
  - `name`
  - `op_class`
  - `gold_path`
  - `registry_name`
  - `candidate_paths`
- `check_operator.py` 不直接知道某个算子的 Python 类路径。
- `--candidate cuda` 明确选择 CUDA candidate。
- `--candidate registry` 仅用于测试 dispatcher 分发结果，不作为具体 CUDA correctness 的替代。

当前 logp 映射：

```text
pytorch      -> NativeLogpOp
cuda         -> FusedLogpGenericOp
cuda-generic -> FusedLogpGenericOp
cuda-sm90    -> FusedLogpSM90Op
registry     -> kernel_registry.get_op("logp")
```

结果：

- 后续新增算子主要扩展 `operator_specs.py`，不再修改公共入口。

### 5. 统一 operator input 工厂

目标：

- 用户指出不希望每个新算子都手写 `_make_xxx_inputs` 和 `_xxx_shape_name`。
- 统一准备 ISSUE #108 中所有算子的输入初始化。

修改文件：

- `rl_engine/testing/operator_inputs.py`
- `tests/test_operator_inputs.py`
- `rl_engine/testing/operator_specs.py`
- `scripts/check_operator.py`

设计判断：

- 新增 `make_operator_inputs(op_name, args, dtype, device)`。
- 新增 `operator_shape_name(op_name, args)`。
- 支持 `random` 和 `constant` 两种输入模式：
  - `random` 用 seed 控制可复现。
  - `constant` 用固定值便于 debug。
- 支持的算子输入：
  - `rms_norm`
  - `matmul`
  - `attention`
  - `logp`
  - `rope`
  - `silu`
  - `swiglu`
  - `embedding`
  - `lm_head`
  - `kv_cache_attention`

CLI 增加参数：

```text
--input-mode random|constant
--constant-value
--token-value
--normalized-dim
--k-dim
--n-dim
--theta
--eps
```

验证：

```bash
python -m pytest tests/test_operator_inputs.py -q
python -m pytest tests/test_op_checks.py -q
python scripts/check_operator.py --op logp --candidate pytorch --dtype fp32 --batch 1 --seq 4 --vocab 17 --input-mode constant --constant-value 0.5 --token-value 3
```

结果：

- 多算子输入生成能力建立。
- logp CLI 随机输入和固定输入均通过。

### 6. 删除模型命名和档位参数

目标：

- 用户指出测试框架不应绑定某个模型，也不需要冗余 `--size` 档位。
- 输入生成应表达为通用数据规模，而不是某个模型配置。

修改文件：

- `rl_engine/testing/operator_inputs.py`
- `scripts/check_operator.py`
- `tests/test_operator_inputs.py`

设计判断：

- 删除 `QWEN3_8B` 命名。
- 删除 `ModelShape` / `DEFAULT_MODEL_SHAPE` 抽象。
- 改为普通常量：

```python
DEFAULT_HIDDEN = 4096
DEFAULT_N_HEADS = 32
DEFAULT_N_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_INTERMEDIATE = 12288
DEFAULT_VOCAB = 151936
DEFAULT_ROPE_THETA = 1.0e6
DEFAULT_RMS_EPS = 1.0e-6
```

- 删除 `--size small|medium|large`。
- 仅保留显式 `--batch` 和 `--seq`。

验证：

```bash
rg -n "Qwen|QWEN|qwen" rl_engine/testing scripts tests
rg -n "ModelShape|DEFAULT_MODEL|model" rl_engine/testing/operator_inputs.py
rg -n "BATCH_SHAPES|BatchShape|--size|small|medium|large" scripts/check_operator.py rl_engine/testing/operator_inputs.py tests/test_operator_inputs.py
python -m pytest tests/test_operator_inputs.py tests/test_op_checks.py -q
```

结果：

- 模型耦合和档位参数均已删除。
- 相关测试通过。

### 7. 同步 upstream/main 并处理冲突

目标：

- 用户要求检查当前代码是否过旧，拉取最新代码，如有冲突则解决。

操作：

```bash
git fetch --all --prune
git stash push -u -m pre-upstream-main-sync
git rebase upstream/main
git stash pop
```

冲突文件：

- `csrc/cuda/fused_logp_sm90.cu`

设计判断：

- `upstream/main` 已包含 PR #122。
- PR #122 中已经包含 SM90 文件的两项修复：
  - `#include <math_constants.h>`
  - `reinterpret_cast<const nv_bfloat16*>`
- 因此冲突解决时采用 upstream/main 版本。
- 丢弃本地临时加入的 `#include <cuda_runtime.h>`。

验证：

```bash
python -m pytest tests/test_operator_inputs.py tests/test_op_checks.py -q
```

结果：

- rebase 到最新 `upstream/main` 成功。
- 冲突解决完成。
- 本地相对 `upstream/main` 为 `ahead 3, behind 0`。

### 8. H20 CUDA 环境和普通 CUDA logp 验证

目标：

- 将测试框架迁移到 H 系列 GPU 环境验证。
- 先确认普通 CUDA `fused_logp` 路径是否可用。

环境记录：

```text
GPU: NVIDIA H20
Driver: 565.57.01
Driver CUDA capability: 12.7
nvcc: 12.4
Python: 3.11.15
```

普通 CUDA 扩展检查：

```text
_EXT_AVAILABLE: True
has fused_logp: True
has fused_logp_sm90: False
```

验证命令：

```bash
python scripts/check_operator.py \
  --op logp \
  --candidate cuda \
  --device cuda \
  --dtype bf16 \
  --arch-key sm90 \
  --batch 2 \
  --seq 16 \
  --vocab 257
```

输出：

```text
INFO [RL-Kernel]: Successfully linked to precompiled _C.fused_logp fallback kernel.
suite=logp passed=True pass_rate=1.0000
candidate=cuda-logp backend=cuda passed=True pass_rate=1.0000
case=logp-torch.bfloat16-2x16x257 output=0 shape=(2, 16)
max_abs=1.49779320e-02
mean_abs=7.53845274e-03
max_rel=2.70811981e-03
tol=(atol=5.000e-02, rtol=0.000e+00)
passed=True
```

结论：

- 普通 CUDA `FusedLogpGenericOp -> _C.fused_logp` 路径通过。
- 这证明测试框架最小 GPU 闭环已经打通：

```text
CLI
-> operator_specs
-> operator_inputs
-> PyTorch gold
-> CUDA candidate
-> run_operator_suite
-> tolerance contract
-> compare_output
-> structured report
```

### 9. SM90 fused_linear_logp 当前状态

目标：

- 尝试编译和验证 SM90 路径。

结果：

- `fused_logp_sm90` 的旧 include 和 type 问题已由 upstream PR #122 解决。
- 但 `fused_linear_logp_sm90.cu` 在 CUDA 12.4 下仍未通过 ptxas。

错误摘要：

```text
ptxas error: State space incorrect for instruction 'cp.async.bulk.tensor'
ptxas fatal: Ptx assembly aborted due to errors
```

设计判断：

- PR #122 描述中提到相关 SM90 路径在 CUDA 13.1 下 assembled。
- 当前 H20 环境是 nvcc 12.4，不应将该路径写为已通过。
- 当前应先以普通 CUDA `fused_logp` 作为验证通过范围。

结论：

- 已通过：`--candidate cuda`
- 未通过：`--candidate cuda-sm90` / SM90 fused linear logp

## 当前文件状态摘要

本 session 产生或涉及的主要文件：

```text
rl_engine/testing/tolerance.py
rl_engine/testing/tolerance_contract.yaml
tests/test_tolerance_contract.py
rl_engine/testing/op_checks.py
tests/test_op_checks.py
rl_engine/testing/__init__.py
scripts/check_operator.py
rl_engine/testing/operator_specs.py
rl_engine/testing/operator_inputs.py
tests/test_operator_inputs.py
csrc/cuda/fused_logp_sm90.cu
docs/contributing/issue-108-session-log.md
```

说明：

- `AGENTS.md` 是未跟踪文件，未纳入本 session 的代码修改范围。
- `csrc/cuda/fused_logp_sm90.cu` 最终与 upstream PR #122 版本一致。

## 后续记录模板

之后每次代码修改都在本文档追加如下条目：

```markdown
### YYYY-MM-DD HH:MM - 变更标题

目标：

- 本次最小子任务要解决什么问题。

修改文件：

- `path/to/file.py`

设计决策：

- 为什么这样改。
- 为什么没有选择其他方案。

验证方式：

- 执行的测试命令。
- CUDA 环境，如 GPU、CUDA 版本、driver、arch。
- 关键输出指标。

结果：

- 通过 / 未通过 / 部分通过。
- 未通过时必须记录完整错误摘要。

后续：

- 是否需要继续拆分子任务。
- 是否影响 CI、benchmark 或其他算子。
```

CUDA 验证建议额外记录：

```markdown
GPU:
CUDA:
Driver:
Arch:
Candidate:
Backend:
Command:
max_abs:
mean_abs:
max_rel:
atol:
rtol:
Result:
Known issue:
```

### 2026-06-28 - CUDA 13 CUB reduce functor 兼容修复

目标：

- 修复 H100 + CUDA 13.0 环境下 SM90 编译失败的问题。

修改文件：

- `csrc/cuda/fused_logp_sm90.cu`
- `docs/contributing/issue-108-session-log.md`

错误摘要：

```text
csrc/cuda/fused_logp_sm90.cu(76): error: namespace "cub" has no member "Max"
csrc/cuda/fused_logp_sm90.cu(86): error: namespace "cub" has no member "Sum"
```

设计决策：

- 不继续依赖 CUB 内置 `cub::Max()` 和 `cub::Sum()` functor 名称。
- 在当前 SM90 文件内定义本地 `FloatMax` 和 `FloatSum`，传给 `cub::BlockReduce::Reduce`。
- 这样保留原有 reduction 语义，同时规避 CUDA 13 / CCCL 中 CUB functor API 变化。

验证方式：

- 本地只做源码修改；需要在 H100 + CUDA 13.0 机器上重新执行：

```bash
rm -rf build
find rl_engine -name "*.so" -delete

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
export MAX_JOBS=1

KERNEL_ALIGN_FORCE_SM90=1 pip install -v --no-build-isolation -e . 2>&1 | tee build_sm90.log
```

结果：

- 待 H100 机器重新编译确认。

后续：

- 如果继续失败，优先查看 `grep -nE "FAILED:|error:|ptxas|fatal" build_sm90.log | head -n 80`。

### 2026-06-28 - 对齐 SM90 LogP Python wrapper 输入接口

目标：

- 修复 `check_operator.py --candidate cuda-sm90` 调用失败的问题。

修改文件：

- `rl_engine/kernels/ops/cuda/loss/logp.py`
- `docs/contributing/issue-108-session-log.md`

错误摘要：

```text
TypeError: FusedLogpSM90Op.__call__() got an unexpected keyword argument 'token_ids'
```

设计决策：

- 测试框架统一通过 `case.inputs` 传递 `token_ids`。
- `FusedLogpGenericOp`、`NativeLogpOp` 都使用 `token_ids` 命名。
- 因此将 `FusedLogpSM90Op.__call__(logits, labels)` 改为 `__call__(logits, token_ids)`，让 candidate 接口与 gold/case 输入一致。
- 同时在 wrapper 内部把 `[B, S, V]` logits reshape 为 `[B*S, V]`，把 `[B, S]` token ids flatten 为 `[B*S]`，再把 `_C.fused_logp_sm90` 的 `[B*S]` 输出 reshape 回 `[B, S]`。

验证方式：

- 需要在 H100 + CUDA 13.0 机器上同步该 patch 后运行：

```bash
python scripts/check_operator.py \
  --op logp \
  --candidate cuda-sm90 \
  --device cuda \
  --dtype bf16 \
  --arch-key sm90 \
  --batch 2 \
  --seq 16 \
  --vocab 257
```

结果：

- 已撤回该方向。`rl_engine/kernels/ops/cuda/loss/logp.py` 属于被测 CUDA 算子实现，不应为了测试框架改动其接口。
- 后续适配应放在 testing 层，例如在 candidate adapter 中把测试框架统一的 `token_ids` 映射为 SM90 wrapper 需要的 `labels`，并处理 flatten/reshape。

### 2026-06-28 - 在 testing 层适配 SM90 LogP candidate

目标：

- 保持 `rl_engine/kernels/ops/cuda` 下被测实现不变。
- 让 `check_operator.py --candidate cuda-sm90` 可以使用测试框架统一的 `token_ids` 输入。

修改文件：

- `rl_engine/testing/operator_specs.py`
- `docs/contributing/issue-108-session-log.md`

设计决策：

- 新增 `_LogpSM90CandidateAdapter`，只在 `args.op == "logp"` 且 `candidate == "cuda-sm90"` 时使用。
- adapter 接收测试框架标准输入 `logits` 和 `token_ids`。
- adapter 内部把 `logits` 从 `[B, S, V]` flatten 为 `[B*S, V]`，把 `token_ids` 从 `[B, S]` flatten 为 `[B*S]`。
- adapter 调用原始 SM90 candidate：`self._candidate(logits_2d, labels_1d)`。
- adapter 将输出 reshape 回 `[B, S]`，以便 `compare_output` 按原始 case shape 比较。

验证方式：

- 本地执行 Python 测试和编译检查：

```bash
python -m py_compile rl_engine/testing/operator_specs.py
python -m pytest tests/test_op_checks.py tests/test_operator_inputs.py -q
```

- H100 机器需要重新运行：

```bash
python scripts/check_operator.py \
  --op logp \
  --candidate cuda-sm90 \
  --device cuda \
  --dtype bf16 \
  --arch-key sm90 \
  --batch 2 \
  --seq 16 \
  --vocab 257
```

结果：

- 本地待验证；H100 CUDA 结果待重新运行确认。

### 2026-06-28 - H100 CUDA generic 与 SM90 LogP 对照验证

目标：

- 记录 H100 + CUDA 13.0 环境下 `logp` 的 generic CUDA 和 SM90 candidate 行为差异。
- 明确测试框架已能区分“通过的 CUDA generic candidate”和“编译/运行存在问题的 SM90 candidate”。

环境：

```text
GPU: NVIDIA H100 80GB HBM3
Driver: 580.95.05
CUDA driver capability: 13.0
nvcc: 13.0
Python: 3.12.13
torch: 2.12.0+cu130
torch cuda: 13.0
compute capability: (9, 0)
```

修改文件：

- `docs/contributing/issue-108-session-log.md`

验证命令：

```bash
for v in 256 512 1024 2048 4096; do
  echo "=== vocab=$v ==="
  python scripts/check_operator.py \
    --op logp \
    --candidate cuda \
    --device cuda \
    --dtype bf16 \
    --arch-key sm90 \
    --batch 1 \
    --seq 1 \
    --vocab $v
done
```

generic CUDA 结果：

```text
vocab=256:  passed=True, max_abs=5.77497482e-03
vocab=512:  passed=True, max_abs=8.04328918e-03
vocab=1024: passed=True, max_abs=1.80721283e-04
vocab=2048: passed=True, max_abs=1.77164078e-02
vocab=4096: passed=True, max_abs=2.69813538e-02
atol=5.000e-02, rtol=0.000e+00
```

结论：

- `FusedLogpGenericOp -> _C.fused_logp` 在 H100 + CUDA 13.0 上多 vocab correctness 全部通过。
- 这进一步确认测试框架、input 生成、gold path、candidate 调用和 compare_output 链路是通的。

SM90 对照现象：

```text
TILE_V=4096:
  vocab=257/4096/151936 均在 cuTensorMapEncodeTiled 失败。
  错误：CUDA_ERROR_INVALID_VALUE。

TILE_V=256:
  vocab=256 返回结果，但 passed=False，max_abs≈1.04094028e+00。
  vocab=512/1024/2048/4096 在 20s timeout 下没有输出 report，表现为 hang/timeout。
```

结论：

- `cuda-sm90` 已能编译和加载，但当前 SM90 TMA kernel 仍不能标记为通过。
- 当前通过范围只包括 `--candidate cuda` generic CUDA logp。
- SM90 问题应作为独立 CUDA kernel bugfix 处理，不归因于测试框架。

### 2026-06-28 - 最终整理：不提交 CUDA 源码改动

目标：

- 用户明确要求本阶段不修改 `csrc` 下 CUDA/TMA 源码。
- 本阶段只提交算子测试框架和文档，不把 SM90 kernel 实验 patch 混入测试框架 PR。

本地处理：

- 已还原 `csrc/cuda/fused_logp_sm90.cu`。
- 本地 `csrc/utils/tma_utils.cuh` 没有 diff。
- 因此本地最终不会提交任何 `csrc` 改动。

服务器状态对照：

用户在 H100 服务器上看到：

```text
Changes not staged for commit:
  modified:   csrc/cuda/fused_logp_sm90.cu
  modified:   csrc/utils/tma_utils.cuh
```

本地当前状态不同：

```text
csrc/cuda/fused_logp_sm90.cu: no diff after restore
csrc/utils/tma_utils.cuh: no diff locally
```

结论：

- H100 服务器上的 `csrc/utils/tma_utils.cuh` 改动不是本地当前工作区的一部分。
- 如果服务器要回到与本地一致的测试框架提交状态，需要在服务器上还原两个 CUDA/TMA 文件：

```bash
git restore csrc/cuda/fused_logp_sm90.cu csrc/utils/tma_utils.cuh
```

保留记录的 CUDA 现象：

- H100 环境：

```text
GPU: NVIDIA H100 80GB HBM3
Driver: 580.95.05
CUDA driver capability: 13.0
nvcc: 13.0.88
Python: 3.12.13
torch: 2.12.0+cu130
torch cuda: 13.0
compute capability: (9, 0)
```

- `--candidate cuda` generic logp 在 vocab 256/512/1024/2048/4096 上通过。
- `--candidate cuda-sm90` 可以编译和加载，但：
  - `TILE_V=4096` 触发 `cuTensorMapEncodeTiled failed`。
  - `TILE_V=256` 时 vocab=256 返回但数值不通过，vocab>=512 出现 timeout/hang。
- 所以 SM90 fused logp 目前记录为 CUDA kernel 问题，不作为测试框架失败。

### 2026-06-28 - 测试框架目录归位

目标：

- 用户指出测试框架文件放在通用 `rl_engine/testing` 下过于分散。
- 本阶段将 operator correctness checking 代码移动到 kernel 相关目录，避免和 RL batch/reference testing 混在一起。

最终目录：

```text
rl_engine/kernels/gtest/
  __init__.py
  op_checks.py
  operator_inputs.py
  operator_specs.py
  tolerance.py
  tolerance_contract.yaml
```

职责划分：

- `op_checks.py`
  - 定义 `OperatorCase`、`CandidateSpec`、report dataclass。
  - 调用 gold 和 candidate。
  - flatten 输出。
  - 解析 tolerance。
  - 计算 `max_abs_error`、`mean_abs_error`、`max_rel_error`。
  - 返回通过率和结构化 report。
- `operator_inputs.py`
  - 统一构造标准语义输入。
  - 支持 `random` 和 `constant`。
  - 支持 `batch`、`seq`、`vocab` 等 CLI 参数。
  - 当前覆盖 ISSUE-108 相关算子的输入初始化骨架。
- `operator_specs.py`
  - 注册每个算子的 gold path 和 candidate path。
  - gold path 必须来自 `rl_engine.kernels.ops.pytorch`。
  - candidate path 来自 `cuda`、`triton`、`rocm` 或未来 backend。
- `tolerance.py` 和 `tolerance_contract.yaml`
  - 加载 dtype/operator-class 容差表。
  - 供 `op_checks.py` 在 compare output 时解析 `atol` 和 `rtol`。
- `scripts/check_operator.py`
  - 命令行入口。
  - 不直接硬编码具体算子实现。

导入边界：

- operator checking 框架只从 `rl_engine.kernels.gtest` 导入。
- `rl_engine/testing/__init__.py` 不导出 `CandidateSpec`、`OperatorCase`、`run_operator_suite`。
- 这样可以避免 kernel correctness checking 和通用 RL testing helper 混在一起。

### 2026-06-28 - 添加新算子的傻瓜式流程

目标：

- 新增算子时不修改测试主逻辑。
- 新增算子只改注册信息、输入工厂和必要测试。
- gold 永远使用 `rl_engine/kernels/ops/pytorch` 下实现。

步骤 1：确认算子标准接口

先确定这个算子的标准语义输入。例如：

```text
logp:
  inputs:
    logits: [B, S, V]
    token_ids: [B, S]
  output:
    selected_logp: [B, S]
```

要求：

- PyTorch gold、CUDA、Triton、ROCm wrapper 都应尽量使用同一套 Python 接口。
- 不同 backend 不应要求测试框架长期维护 shape/参数名 adapter。
- 当前 `_LogpSM90CandidateAdapter` 只是为了验证框架最小闭环的临时例外，不作为长期模式。

步骤 2：在 `operator_inputs.py` 添加输入构造

文件：

```text
rl_engine/kernels/gtest/operator_inputs.py
```

需要做三件事：

1. 在 `make_operator_inputs()` 的 `builders` 中加入算子名。
2. 在 `operator_shape_name()` 的 `names` 中加入 shape 描述。
3. 新增 `_make_xxx_inputs(args, dtype, device)`。

要求：

- 输入必须是标准语义输入，不是某个 CUDA kernel 的私有格式。
- `random` 模式必须可由 `--seed` 复现。
- `constant` 模式必须便于 debug。
- 多 batch 情况默认保留 `[B, S, ...]` 语义形状，不提前为某个 backend flatten。

步骤 3：在 `operator_specs.py` 注册 gold 和 candidate

文件：

```text
rl_engine/kernels/gtest/operator_specs.py
```

添加：

```python
"new_op": OperatorSpec(
    name="new_op",
    op_class="...",
    gold_path="rl_engine.kernels.ops.pytorch....NativeNewOp",
    registry_name="new_op",
    candidate_paths={
        "pytorch": "rl_engine.kernels.ops.pytorch....NativeNewOp",
        "cuda": "rl_engine.kernels.ops.cuda....CudaNewOp",
        "triton": "rl_engine.kernels.ops.triton....TritonNewOp",
    },
)
```

硬性规则：

- `gold_path` 必须来自 `rl_engine.kernels.ops.pytorch`。
- `candidate_paths["pytorch"]` 只能用于框架自检，不代表高性能算子通过。
- `candidate_paths["cuda"]`、`candidate_paths["triton"]` 等必须对应实际被测 backend。
- 不允许用实现了不同数学功能的算子互相比较，例如不能用 `linear_logp` 测普通 `logp`。

步骤 4：确认 gold 调用方法

当前 `make_operator_case()` 对 logp 使用：

```python
gold_fn=gold_op.forward_fp32
```

这对 `NativeLogpOp` 是正确的。新增算子时必须确认 PyTorch gold 是否有对应方法。

如果新算子没有 `forward_fp32`，不要在测试主逻辑中硬编码临时分支；应在 `operator_specs.py` 中显式补充 gold 调用策略，作为一个独立小改动提交。

步骤 5：新增输入和 runner 单测

至少补两类测试：

```text
tests/test_operator_inputs.py:
  确认 make_operator_inputs("new_op", ...) 能生成输入。
  确认 random seed 可复现。
  确认 constant 模式值正确。

tests/test_op_checks.py 或新测试文件:
  用 pytorch candidate vs pytorch gold 验证框架能跑通。
  用 bad candidate 验证失败报告符合预期。
```

步骤 6：本地验证

CPU 框架验证：

```bash
python -m pytest tests/test_op_checks.py tests/test_operator_inputs.py -q
```

CUDA candidate 验证：

```bash
python scripts/check_operator.py \
  --op new_op \
  --candidate cuda \
  --device cuda \
  --dtype bf16 \
  --arch-key sm90 \
  --batch 2 \
  --seq 16
```

如果 CUDA candidate 不通过，先判断：

```text
1. gold 和 candidate 是否真的是同一个数学函数。
2. candidate Python wrapper 是否使用标准接口。
3. 输入 dtype / shape 是否符合 candidate 声明。
4. 误差是否超过 tolerance。
5. 是否是 kernel 编译或运行错误。
```

不要为了让测试通过去修改 gold，也不要把不同功能的算子混在一起比较。

### 2026-06-28 - `check_operator.py` 支持参数

入口：

```bash
python scripts/check_operator.py [options]
```

核心参数：

```text
--op
  算子名。当前最小版本支持 logp。

--candidate
  被测实现。当前 logp 支持 registry、pytorch、native、cuda、cuda-generic、cuda-sm90。

--dtype
  fp32、bf16、fp16。

--device
  auto、cpu、cuda 或 torch 可识别的 device 字符串。

--arch-key
  tolerance override key，例如 sm90。为空时使用 default tolerance。
```

shape 参数：

```text
--batch
  batch size，默认 2。

--seq
  sequence length，默认 16。

--vocab
  vocabulary size，默认 257。

--normalized-dim
  norm 类算子的 hidden/normalized dimension。

--k-dim
  matmul K dimension。

--n-dim
  matmul N dimension。
```

输入初始化参数：

```text
--input-mode
  random 或 constant。

--constant-value
  constant 模式下浮点 tensor 的基础值。

--token-value
  constant 模式下 token id 的基础值，会对 vocab 取模。

--seed
  random 模式下的随机种子。
```

其他参数：

```text
--theta
  RoPE theta。

--eps
  norm epsilon。

--json
  输出完整 JSON report。
```

当前最小可运行示例：

```bash
python scripts/check_operator.py \
  --op logp \
  --candidate cuda \
  --device cuda \
  --dtype bf16 \
  --arch-key sm90 \
  --batch 1 \
  --seq 1 \
  --vocab 4096
```

预期：

- 在 H100 + CUDA 13.0 环境中，generic CUDA logp 已观察到通过。
- `cuda-sm90` 当前不应作为通过路径使用。
