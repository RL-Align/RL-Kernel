# 2026-11-20 MI300X：真实 V4.1 四层模型证据

本轮新增了真实 checkpoint 的原始 **0/1/2/3** 层 Forward/Backward、FP32 logits/raw token
logprobs、attention TP、PP 边界和 DP 梯度/更新后参数同步。下表不是上一轮组件结果。
总体状态仍为 **BLOCKED / incomplete**：8 卡占用、发布认证/授权、独立官方后端对照仍未解决。

模型 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，revision
`08c8657f02291e11b6f6b63e0e16014f8e748518`。原始 hidden5120、64 attention heads、384 experts/top6、
HC4、层压缩率0/0/2/2全部保留。约31GB真实 tensors/scales 按原始名称加载；所有384个专家的
原始权重均加载，每个 token 依实际路由选择6个。Engram 只装载三组输入实际引用的原始行，
缺行直接报错，没有随机补权重或缩小专家数。下载范围和 SHA256 见
[主体清单](checkpoint-manifest.json)、[额外 source attention 清单](additional-attention-checkpoint-manifest.json)。

## 实测矩阵

基础输入为真实 tokenizer 的 `[0,53091,4374,1465,721,22,16,19]`，batch1，seed1234，
连续 positions、全有效 token mask、BF16 hidden、FP32 logits/logprobs。
所有 PASS 同时完成 Forward、Backward、deterministic replay 和 **shared-kernel** 训练/推理比较。
数值正确性证据是逐算子 reference 测试，以及 PP 与相同 TP 契约非流水基准的内部等价验证；
没有把运行成功视为已经取得独立官方模型 oracle 的证明。

| Configuration | GPU数 / 物理GPU | 原始层 | F/B | Replay / shared-kernel parity | 比较元素 / mismatch | Status / Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| TP1 PP1 DP1 | 1 / 6 | 0,1,2,3 | PASS | PASS / PASS | 161,869,548 / 0 | [PASS](tp1-pp1-dp1/summary.json) |
| TP2 PP1 DP1 | 2 / 5,7 | 0,1,2,3 | PASS | PASS / PASS | 185,326,808 / 0 | [PASS](tp2-pp1-dp1/tp-rank0/summary.json)、[rank1](tp2-pp1-dp1/tp-rank1/summary.json) |
| TP1 PP2 DP1 | 2 / 1,2 | 0,1 / 2,3 | PASS | PASS / PASS | 475,091,376 / 0 | [PASS](tp1-pp2-dp1/summary-rank0.json)、[rank1](tp1-pp2-dp1/summary-rank1.json) |
| TP2 PP2 DP1 | 4 / 3,4,5,7 | 0,1 / 2,3 | PASS | PASS / PASS | 620,323,304 / 0 | [PASS](tp2-pp2-dp1/summary-rank0.json)、[逐rank结果](tp2-pp2-dp1/) |
| TP1 PP2 DP2 | 4 / 1,2,3,4 | 0,1 / 2,3 | PASS | PASS / PASS | 4,043,086,008 / 0 | [PASS，含连续更新与同步](tp1-pp2-dp2/summary-rank0.json)、[逐rank结果](tp1-pp2-dp2/) |
| TP4 PP2 DP1 | 8 / 0…7 | 未执行 | NOT RUN | NOT RUN | 未比较 | [BLOCKED](tp4-pp2-dp1/summary.json) |
| TP2 PP2 DP2 | 8 / 0…7 | 未执行 | NOT RUN | NOT RUN | 未比较 | [BLOCKED](tp2-pp2-dp2/summary.json) |

8 卡 launcher 实际执行了空闲检查，GPU0 被 PID3131740 占用，见
[预检](tp4-pp2-dp1/gpu-preflight.log)。没有终止其他任务或把多个 rank 共用少量卡称为8 GPU通过。
TP 分片 attention heads/output groups/wo_b 输入维，FFN 复制执行。
TP1 loss=15.523721694946289，TP2 loss=15.483124732971191；分片 BF16 求和改变数值契约，
没有跨 TP1/TP2 的 bitwise 结论。PP cut 保留层2→3 source/consumer 在同一 stage，
传递真实 residual、delayed-pre 及其反向梯度，非 collective-only smoke。

TP1 PP2 与 TP1 单卡、TP2 PP2 与 TP2 非流水基准的中间张量和参数梯度逐 bit 一致。
DP2 使用相同真实 token 微批次的两个副本，实测平均梯度、参数同步、更新后输出和整段重放；
尚未比较不同 DP 样本划分与单卡全批次归约的位级等价。

## 新增模型级 stress

- [奇数 prefill + decode](prefill-decode/summary.json)：三组真实 prompt，先 prefill3、再逐 token
  decode 至8个位置，真实 C2 partial state 和层2→3缓存复用，跨步骤 Backward。
  493,547,412 个元素比较，0 mismatch；未宣称 chunked 与 monolithic 逐 bit 相同。
- [梯度累积与 optimizer](optimizer-stress/summary.json)：seeds17/1234，两个不同 prompt 的
  NLL means 累积，batch1/2、长度7/8，每轮两次真实 SGD(lr1e-5, foreach=False)更新，整段重复两次。
  38个参数张量实际改变，比较梯度、更新后参数和更新后 train/inference 输出，
  1,422,081,556 个元素，0 mismatch。没有 dropout；改变 seed 不引入额外随机算子。
- [PP2 DP2 optimizer](tp1-pp2-dp2/container.log)：两次更新、两轮完整重放；两个 stage 分别有
  19/20 个参数张量改变。实际同步的是模型 `.grad` 和更新后的参数，非只对独立示例张量做 RCCL。

[summary.json](summary.json) 汇总模型执行；[consistency.csv](consistency.csv) 有8668条实际张量比较。
[category-totals.json](category-totals.json) 将以下结果分开：

| 类别 | 比较元素 | mismatch |
| --- | --- | --- |
| Deterministic replay，含 gradient/optimizer replay | 4,024,773,394 | 0 |
| Train/inference shared-kernel parity | 183,827,246 | 0 |
| 同 TP 契约跨 PP / replica consistency | 1,980,071,100 | 0 |
| 更新后 DP weight synchronization | 1,212,654,272 | 0 |

合计7,401,326,012是跨 rank、运行和权重/梯度的重复存储元素比较，不是独立 token 数或40层覆盖。
raw token logprobs 每个位置是一个FP32值；基础case每次7个 next-token logprobs。
[原始 FP32 值及 uint32 bits](tp1-pp1-dp1/outputs-raw-logprobs.json) 取自实际保存的输出张量。
全部位级比较读取原 dtype storage bytes，区分+0/-0及NaN payload；没有用容差、rounding或输出
再量化制造0。每条 comparisons.json 均保留 first_mismatch 字段，本轮均为null。

## 算子、后续 source 与回归

[最终 V4.1 日志](logs/v41-model-committed-regression.log)：85 PASS、0 skip，
包含 config/checkpoint/bit-pattern CPU逻辑、GPU reference、input/route/非量化参数 VJP、
新 delayed mHC、RoPE、FP8/FP4 codecs、384-way top6、固定顺序 expert combine 和 source tests。
算子 reference correctness 的明确容差保留于测试源码；与bitwise结果分开。

[真实权重隔离 source 证据](targeted-source-layers-final.json) 覆盖 Engram14、KV sources8/14/20、
index sources24/28/32/36、C1/C2、candidate mask；使用真实四层的上游激活作为隔离输入，
不是这些后续完整 block 的真实40层上下文。其256,874,432次元素比较单列，不计入模型总数。
短序列的真实 source20 candidate mask不会触发2048个block上限的裁剪；partial/newest/unreachable
候选块逻辑另有单测，长上下文 pruning 仍未验收。

- [V4 contracts/model逻辑](logs/v4-final-contract-regression.log)：1004 PASS、11 skip；skip原因包含
  单卡容器下的2/4 GPU用例以及既有缺失 `rl_engine.p3.recordings`，完整原因在日志。
- [V4 GPU kernels](logs/v4-model-committed-gpu-regression.log)：103 PASS、4个 CPU Triton interpreter
  专用case skip。没有把这些结果推广为重新跑过 V4 完整四层 GRPO。
- [PR497目录兼容测试](logs/layout-model-final.log)：16 PASS、0 skip。
- 保留了[新增测试初始fixture调用错误](logs/v41-targeted-source-layers-initial-fixture-failure.log)和
  [V4初始缺少 pinned-source 环境变量](logs/v4-contract-regression-initial-missing-source-env.log)。
  已修复测试调用/环境并重跑通过，未删除失败测试。
- 修复 TP model mismatch diagnostics 的 rank 默认0问题，提交 `3d2fa817`，随后真实 TP2重跑通过。
  本轮未发现需通过降低精度或改模型语义隐藏的数值 mismatch。

## 实现、环境和边界

代码位于 `rl_engine/models/deepseek_v41/{model,attention,loading,parallel}.py` 与
`rl_engine/backends/rocm/{mhc/v41_delayed,attention/v41_transforms,moe/v41_router,moe/v41_combine}.py`。
复用已有 RMSNorm、Engram hash/gate/lookup、C1/C2 compressor、FP8/FP4 projection、AITER sparse
forward/CSR backward 和 V4 mHC post；没有修改 V4 或外部框架代码。

[runtime.json](runtime.json)：MI300X VF gfx942，PyTorch2.13.0+rocm10.0.0、HIP7.15.26333、
Triton3.8.0、AITER0.1.29、RCCL2.30.4、TileLang0.1.10。
[GPU配置](logs/model-gpu-configuration.log) 与每个配置的 image-id/provenance 保留实际环境。
镜像内 transformers 为5.22.0，与 pyproject 的5.13.1 pin不同；没有静默修改依赖声明。

当前是冻结 packed weights/scales、embedding、head 的训练 profile，激活量化 backward 为 identity
STE；非量化 dense/FP32 controller/router/gain 参数实际训练。Indexer离散 top-k没有伪造梯度，
没有实现独立辅助损失。不是 full-parameter training；没有 packed weight/master-weight 学习路径。
BF16/FP32 dense GEMM 来自 PyTorch ROCm library，专用变换来自 RLK Triton，明确区分。

训练和 inference_mode 共享 kernels，只能称shared-kernel parity。
独立官方后端探测并非假设缺包：TileLang实际存在，但原始官方E8M0 quant kernel在gfx942编译时
报 `unknown type name fp8_e8_t`，见[完整实际编译失败](logs/official-tilelang-kernel-probe.log)。
未修改第三方源码或替换官方kernel后冒称官方后端通过。独立模型oracle为BLOCKED。

完整40层、任意padding/非连续positions、EP/CP、packed-weight训练、长上下文以及8 GPU尚未验收。
四层 builder 不会补造缺失 checkpoint 或跨 stage source state。

## 复现、提交与发布

[主入口及compatibility matrix](../../README.md) 提供模型/依赖和精确命令。
`run_model_gpu.sh` 需要干净工作树，保存实际源码commit、GPU guard、image ID与checkpoint清单。
基础矩阵逐配置 provenance 记录了 `fa96c028`、`e8b79ce0`、`a2f92262`、`3d2fa817` 等已提交源码。
报告生成器对应commit记录在summary.json；[最终源码文件SHA256](source-sha256.json)用于审计。
补充 prefill/decode 在对应提交前的工作树执行，后续仅做closure-binding lint修改，不能将其
表述为当时已经由clean-commit launcher运行；基础矩阵则已从已提交源码重跑。

在同一GPU依赖容器，挂载repo为/work、artifacts为/artifacts并设PYTHONPATH=/work：

```bash
python -m examples.operators.dsv41_4l.run_model_state \
  --checkpoint /artifacts/model-4l --output /artifacts/new-state-run
python -m examples.operators.dsv41_4l.run_training_stress \
  --checkpoint /artifacts/model-4l --output /artifacts/new-training-run
```

全部大 checkpoint、baseline.pt、outputs.pt 保留在本机 `/home/rl-kernel/dsv41-artifacts/`，
Git仅提交约6MB的可审计摘要/比较/短日志，不提交大型权重、编译缓存或完整梯度。
现有merge、之前所有commits均保留。fetch后origin/dsv4-rocm仍为`71c7cb2d`且已被当前分支包含，
无需重复merge。分支始终是`dsv4.1-rocm`。

GitHub发布 **BLOCKED**：[gh未登录](logs/model-gh-auth-status.log)、[SSH publickey失败](logs/model-ssh-auth-status.log)。
用户明确选择“暂时保留本地提交”。一次再次尝试push被自动审批以“尚未重新授权向GitHub外发代码”
拒绝，命令未执行；没有绕过审批或force push。后续push需用户重新授权并完成认证。
先继续解决8 GPU空闲窗口及独立后端/全参数训练边界，再按对应契约补测剩余项。
