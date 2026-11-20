# 2026-11-20：限定物理 GPU4–7 的真实 V4.1 四层验收

本轮指定的1/2/4卡模型矩阵全部 **PASS**。全部运行使用原始0/1/2/3层真实checkpoint，
保留之前全部commits和V4功能。八卡测试已由用户暂时取消，不作为本轮blocker；未等待GPU0空闲，
未终止、抢占或修改GPU0–3进程。GitHub推送仍因认证未恢复而BLOCKED。

## 实际可见性与设备映射

先检查了旧容器：`ROCR_VISIBLE_DEVICES=6`、`HIP_VISIBLE_DEVICES=0`。
没有叠加该容器的HIP过滤，而是创建新的测试容器；host/Docker/Python/torchrun/RCCL子进程均
使用当前测试的ROCR子集，并清除HIP_VISIBLE_DEVICES、CUDA_VISIBLE_DEVICES、GPU_DEVICE_ORDINAL、
HSA_VISIBLE_DEVICES。新launcher拒绝GPU0–3、重复/反序IDs和八卡请求。

[原容器与新容器配置](logs/container-visibility-inspection.json)；
[四卡实际PyTorch数量、PCI与GPU整数roundtrip](logs/container-mapping-all.json)。
Docker传递`/dev/kfd`和`/dev/dri`，GPU选择通过ROCR及运行时PCI验证实现；没有宣称DRM节点级硬隔离。
每个容器先验证映射并做真实GPU读回，每个rank在RCCL初始化前再次检查数量、PCI及二次过滤。

| 逻辑GPU（四卡请求） | 物理GPU | 实测PCI | ROCm unique ID |
| --- | --- | --- | --- |
| 0 | 4 | 0000:a3:00.0 | d51f48da8b8b8703 |
| 1 | 5 | 0000:ab:00.0 | 99c1795a4a1184cb |
| 2 | 6 | 0000:b3:00.0 | 62f26f8642cc65a5 |
| 3 | 7 | 0000:bb:00.0 | e07794f155d12122 |

单卡请求设ROCR=4，仅逻辑0→物理4；两卡请求设ROCR=4,5，逻辑0/1→物理4/5。
每个run的`gpu-mapping.json`保存容器验证，各`summary*.json`的`gpu_assignment`保存rank实际映射。
GPU idle guard只对分配卡做显存检查，不抢占任务。

## 新跑的真实模型矩阵

模型：`deepseek-ai/DeepSeek-V4.1-Flash-1017`，revision
`08c8657f02291e11b6f6b63e0e16014f8e748518`。
[真实权重下载清单](checkpoint-manifest.json)保留精确range与SHA256。
没有重编号原始层、缩小专家数、随机补权重或伪造source state。
层1 Engram、层2 compressor/indexer、层3跨层状态、384-way top6 MoE、delayed mHC都实际执行。

所有PASS包括Forward、Backward、gradient consistency、deterministic replay、raw FP32 logprobs
和 **shared-kernel** 训练/inference_mode比较。PP还与相同TP契约的非流水单卡/两卡基准比较。

| 配置 | 物理IDs | 逻辑IDs | 原始层 | F/B | 重放 / shared-kernel parity | 比较元素 / mismatch | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TP1 PP1 DP1 | 4 | 0 | 0,1,2,3 | PASS | PASS / PASS | 161,869,548 / 0 | [summary](tp1-pp1-dp1/summary.json)、[log](tp1-pp1-dp1/container.log) |
| TP2 PP1 DP1 | 4,5 | 0,1 | 0,1,2,3 | PASS | PASS / PASS | 185,326,808 / 0 | [rank0](tp2-pp1-dp1/tp-rank0/summary.json)、[rank1](tp2-pp1-dp1/tp-rank1/summary.json) |
| TP1 PP2 DP1 | 4,5 | 0,1 | 0,1 / 2,3 | PASS | PASS / PASS | 475,091,376 / 0 | [rank0](tp1-pp2-dp1/summary-rank0.json)、[rank1](tp1-pp2-dp1/summary-rank1.json) |
| TP2 PP2 DP1 | 4,5,6,7 | 0,1,2,3 | 0,1 / 2,3 | PASS | PASS / PASS | 620,323,304 / 0 | [逐rank](tp2-pp2-dp1/)、[log](tp2-pp2-dp1/container.log) |
| TP1 PP2 DP2，不同prompt | 4,5,6,7 | 0,1,2,3 | 0,1 / 2,3 | PASS | PASS / PASS | 4,043,086,008 / 0 | [逐rank](tp1-pp2-dp2/)、[log](tp1-pp2-dp2/container.log) |
| 第二prompt单卡基准 | 4 | 0 | 0,1,2,3 | PASS | PASS / PASS | 161,869,548 / 0 | [summary](tp1-pp1-dp1-case1/summary.json) |

TP2 PP2：rank0/1在物理4/5执行层0/1，rank2/3在物理6/7执行层2/3。
PP2 DP2：replica0为物理4/5，replica1为物理6/7；DP组分别为物理4/6、5/7。

**本轮新增了不同DP样本验证**，不只重复上一轮相同prompt的两个副本：

- prompt0输入 `[0,53091,4374,1465,721,22,16,19]`，单卡loss=15.523721694946289。
- prompt1输入 `[0,45588,261,3617,6058,305,34555,377]`，单卡loss=14.796404838562012。
- 每个replica先与各自单卡基准的中间张量、FP32 logprobs和参数梯度逐bit比较。
- 实际DP模型梯度平均后，再与两份独立单卡梯度的`(g0+g1)/2`逐bit比较。
  契约是各参数原梯度dtype的求和/除二，不把它表述为任意batch布局的全参数FP32归约等价。
- 两次真实SGD更新、整段重复两轮、同步模型`.grad`与更新后的参数；两个stage分别有18/20个
  参数张量实际改变。更新后的各自prompt训推输出、梯度及参数重放均为0 mismatch。

[真实FP32值与原始uint32 bits，prompt0](tp1-pp1-dp1/outputs-raw-logprobs.json)、
[prompt1](tp1-pp1-dp1-case1/outputs-raw-logprobs.json)、
[DP replica0](tp1-pp2-dp2/logprobs-rank1-raw-logprobs.json)、
[DP replica1](tp1-pp2-dp2/logprobs-rank3-raw-logprobs.json)。
这些JSON直接从实际保存的FP32输出tensor导出，包含物理/逻辑GPU身份；不是舍入后的比较结果。

## GPU4补充模型stress及算子回归

[GPU4 prefill/decode](prefill-decode/summary.json)：逻辑0，三组真实prompt，prefill3后连续单token
解码至8个位置，真实partial compressor、跨层KV/index cache和跨步Backward，
493,547,412个元素比较，0 mismatch。未宣称chunked与monolithic逐bit等价。

[GPU4 optimizer stress](optimizer-stress/summary.json)：逻辑0，seed17/1234、两个不同prompt的
梯度累积、batch1/2、长度7/8、两次SGD更新及整段重放，38个参数张量实际更新；
1,422,081,556个元素比较，0 mismatch。没有dropout，不声称seed变化引入随机计算。

[summary.json](summary.json)汇总所有本轮结果；[consistency.csv](consistency.csv)包含8840条实际比较，
每条标明physical/logical IDs、assigned GPU、rank、operator、dtype和mismatch。

| 类别 | 比较元素 | mismatch |
| --- | --- | --- |
| Deterministic replay，含梯度/更新后参数 | 4,181,384,308 | 0 |
| Train/inference shared-kernel parity | 189,085,880 | 0 |
| 跨PP、TP replica、DP梯度/参数一致性 | 3,192,725,372 | 0 |

合计7,563,195,560是跨运行/rank/权重/梯度的重复存储元素比较，不是独立token数或40层覆盖。
比较直接读取原dtype storage bytes，区分signed zero/NaN payload，不使用allclose、rounding或输出
降精度替代bit-pattern。没有发现数值mismatch，本轮没有修改算子数值契约。

GPU回归均限制在物理6、逻辑0，并另存[容器PCI验证](logs/regression-gpu-mapping.json)：

- [V4.1完整当前测试目录](logs/v41-regression.log)：95 PASS、0 skip，含可信reference correctness、
  Forward/Backward、真实source-layer组件，以及新增10项资源策略检查。
- [V4 GPU kernels](logs/v4-gpu-regression.log)：103 PASS、4个CPU Triton interpreter专用case skip。
- [V4 contracts/model逻辑](logs/v4-contract-regression.log)：1004 PASS、11 skip；缺少额外GPU和
  既有`rl_engine.p3.recordings`的详细skip原因原样保留。没有声称重新验收V4完整GRPO。
- [目录兼容](logs/layout-regression.log)：16 PASS。
- [GPU选择策略CPU单测](logs/visibility-policy-tests.log)：10 PASS，确认非法GPU选择在设备查询前
  被拒绝、重排PCI可被识别、空值HIP/CUDA等二次过滤也必须unset。

## 数值与训练范围

当前仍是冻结packed FP8/FP4权重/scales、embedding、head的profile；激活量化采用identity STE。
非量化dense、router、mHC和norm/gain参数有真实梯度与更新。Indexer离散top-k无伪造梯度。
训练与inference_mode共享ROCm kernels，属于shared-kernel parity，没有独立后端或跨框架结论。
TP分片attention heads/output groups，FFN复制；TP1与TP2不同BF16归约契约不可混为位级等价。
独立官方TileLang oracle的历史gfx942编译阻塞未在本轮解决。全参数量化训练、完整40层、EP/CP、
任意padding/非连续positions和长上下文candidate pruning仍未验收。

## 复现与Git

见[当前主说明](../../README.md)。在干净且已提交的`dsv4.1-rocm`上顺序执行：

```bash
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=4,5,6,7
image=rlk/dsv4-4l-rocm:pr717-ropefix-aade119d-mi300x
weights=/home/rl-kernel/dsv41-artifacts/model-4l
out=/home/rl-kernel/dsv41-new-gpu4567-reproduction
mkdir "$out"
run=examples/operators/dsv41_4l/run_model_gpu.sh
"$run" "$image" 4 "$weights" "$out/tp1" 1 1 1 -
"$run" "$image" 4,5 "$weights" "$out/tp2" 2 1 1 -
"$run" "$image" 4,5 "$weights" "$out/pp2" 1 2 1 "$out/tp1/baseline.pt"
DSV41_MODEL_CASE=1 "$run" "$image" 4 "$weights" "$out/case1" 1 1 1 -
"$run" "$image" 4,5,6,7 "$weights" "$out/tp2pp2" 2 2 1 "$out/tp2"
DSV41_DP_SECOND_BASELINE="$out/case1/baseline.pt" DSV41_MODEL_OPTIMIZER_STEPS=2 \
  "$run" "$image" 4,5,6,7 "$weights" "$out/pp2dp2" 1 2 2 "$out/tp1/baseline.pt"
```

本机原始输出及大baseline.pt保存在`/home/rl-kernel/dsv41-artifacts/gpu4567-20261120/`。
Git只提交约6.4MB的短日志、JSON/CSV和权重来源清单，没有checkpoint或编译缓存。
每次run的`provenance.json`保留真实source commit/image ID，[源码SHA256](source-sha256.json)可核对。
所有先前merge、算子和commits都保留，始终只在`dsv4.1-rocm`开发，无新分支/PR/force push。

[gh认证状态](logs/gh-auth-status.log)仍为未登录，[SSH](logs/ssh-auth-status.log)仍publickey denied。
本轮正常push的授权已由最新用户指令恢复，但认证条件未满足，因此没有再次发起无认证push。
如需恢复发布，在服务器人工执行`gh auth login --hostname github.com --git-protocol https --web`，
完成浏览器OAuth后运行`gh auth setup-git`；随后可正常`git push -u origin dsv4.1-rocm`，不需要提供token。
