# 阶段 5：最终交付审查（2026-10-02）

> 本文保留 2026-10-02 的历史状态及源码指纹；后续 sm90 入口和草稿 PR 状态见
> [当前审查状态](07-draft-pr-status.md)。下文“未提交”等表述属于当时记录。

## 结论与范围

已完成本次授权的独立实现、官方 gtest 接入、A100 算子级正确性/逐位配置验证、
trace、性能测量及中文报告。代码保持未提交，供审查；没有 commit/push/PR/评论。
这不是全模型 WS1 EXIT，也不代表维护者已经验收或所有形状都有性能优势。

工作区 `/home/hlb/.codex/worktrees/105a/RL-Kernel`，分支 `codex/timestep-official-astra`，
基线 `32b765ec992cd1206517104ec66506881203c91c`。正式 thread ID
`01a0f7ef-7063-71e1-b087-24915728539e`，host `local`。

完整源码指纹：`1cf074e9048f3a8c310b289f511d7303dd14398a1bc2153ac1ef820efc583f9c`。
逐文件 SHA 在 [final-source-manifest.json](evidence/final-source-manifest.json)。
算子和 gtest 文件与生产矩阵记录的源码 SHA 全部一致；验证脚本最后仅修复 runpy
相对路径记录错误，已在最终小宽度矩阵和本地定向矩阵验证。

## 合同和数学审查

- [维护者回复](https://github.com/RL-Align/RL-Kernel/issues/386#issuecomment-5926837071)：标准256/3072前后向、reduction gtest、CPU/GPU 容差、以 test-qwenimage 为基线；SM90可选。
- [参考 PR #204](https://github.com/RL-Align/RL-Kernel/pull/204)：头 SHA ddf237fbda67dbd5ab8a1eeb592a033472edab08；仅参考注册/trace/fallback范式。
- main `1968a87114c331513f9d320c9bd526f0d91f3a52` 与本基线的 gtest、CLI、使用文档无差异。
- Diffusers `031b2798addadd1652db7cfba50eacc1079245cf` 的标准模块已核对，cos-first、频率分母128、内部scale1000；不增加conditioning。
- 前后向公式见阶段1；最终补偿顺序和独立参考修正见阶段3。所有 gate 取未修改的官方合同，FP64只作诊断。
- 参数梯度的不变性范围是同一完整逻辑样本集的canonical顺序；外部独立microbatch的BF16 .grad加和不承诺逐位一致。

## 实际验收

| 验证 | 结果与证据 |
|---|---|
| 生产 H3072、两后端两精度、B1/3/16、两种子 | 24/24通过；[交付矩阵](evidence/a100/a100-delivery-matrix.json) |
| 长样本归约 K>32，H17 B33/64 | 8/8通过；[额外矩阵](evidence/a100/a100-long-sample-reduction.json) |
| 官方 check_operator CLI | cuda/triton × fp32/bf16 四条全通过；evidence/a100/a100-gtest-*.json |
| 相关pytest | 60 passed in 6.61s；[日志](evidence/a100/a100-delivery-pytest.log) |
| 配置不变性 | 重复、chunk1/2、重排、padding、strided、singleton行；按字节比较，全部通过 |
| 真实后端 | 四份Profiler trace捕获embedding/mm/silu/dt CUDA事件，actual_backend无fallback |
| 生命周期 | 空批次、仅部分输入需梯度、多活动图、重复后向、非默认stream通过 |
| 正式预编译模块 | 从真实setup.py提取新增扩展配置构建成功并实际加载；本机6项回归、B16 seed9386两后端两精度4/4通过 |
| 静态检查 | 修改的Python文件 Ruff check通过；git diff --check通过 |

本地打包验证只构建新增 `rl_engine._timestep_cuda`，不是全仓 `_C` 的完整编译。
A100测试使用同一CUDA源的独立JIT扩展。原始路径、失败与构建日志均保留，没有以
历史研究线通过记录充当新实现证据。首次 Ninja/Python.h 环境失败、两次数值方案
失败、runpy路径失败均可在 evidence/ 中追溯。

## 复现

远端独立根 `/home/linux/timestep-official-astra-01a0f7ef`，代码在 src，缓存在 cache，
原始证据在 evidence；本地已取回。下列只读环境复用不修改研究目录：

```bash
cd /home/linux/timestep-official-astra-01a0f7ef/src
export PATH=/home/linux/timestep-a100.ydZtoB/venv/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR=/home/linux/timestep-official-astra-01a0f7ef/cache/torch
export TRITON_CACHE_DIR=/home/linux/timestep-official-astra-01a0f7ef/cache/triton
export CPATH=/home/linux/timestep-a100.ydZtoB/python-headers/usr/include/python3.10:/home/linux/timestep-a100.ydZtoB/python-headers/usr/include/x86_64-linux-gnu/python3.10:/home/linux/timestep-a100.ydZtoB/python-headers/usr/include
python scripts/validate_timestep_official.py --trace --benchmark --output /absolute/new-output/matrix.json
python -m pytest tests/test_timestep_official.py tests/test_tolerance_contract.py tests/test_operator_inputs.py -q
python scripts/check_operator.py --op timestep_embed_mlp --candidate cuda --device cuda --dtype fp32 --batch 3 --seq 1 --seed 386 --check-grad --json
# 后两参数组合：--candidate cuda|triton，--dtype fp32|bf16。
python scripts/validate_timestep_official.py --hidden 17 --batches 33 64 --seeds 386 --output /absolute/new-output/long-reduction.json
```

生产矩阵种子386/9386，上游种子分别+1；CLI seed386，上游默认seed123。
输入值在CPU产生并复制，但gtest上游随机数在候选设备产生；不同GPU架构的RNG
launch配置可能影响同seed实际序列，CPU gold在每次运行中复用候选的同一上游值。
因此本机GTX1650Ti证据与A100结果独立记录，不把同seed误称为跨卡完全相同输入。

## 限制与后续

SM90/H100、多GPU、全模型训练、全仓原生扩展CI未验证；不支持autocast或二阶梯度。
BF16采用融合FP32中间值，非Diffusers每层BF16舍入的逐位重放。
CUDA小批次有加速，Triton多数形状仍慢于普通批量PyTorch；完整实测见阶段4。
报告均已保存；当前无原生上下文压缩工具，不声称已执行原生压缩。
远端最后检查0%GPU、14MiB、无compute process；不关闭服务器或删除任何证据。
下一步为人工审查；若后续授权发布，再进行针对目标仓库的完整CI和PR流程。
