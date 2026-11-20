# DeepSeek V4.1 ROCm：组件开发与验收入口

**状态：BLOCKED，尚未完成完整 text-only 模型集成。** 本目录当前执行隔离算子、
官方权重组件、连续 optimizer 更新和组件级 RCCL 检查，不构造四层模型。
不能把这些结果用于宣称四层 Forward/Backward、模型 logits/logprobs 或模型级训推一致性通过。

唯一开发分支为 `dsv4.1-rocm`；不创建 PR，不修改 `main` / `dsv4-rocm`。
V4 合并提交为 `4b41ae76d2b4cd3f87ba71c7b56868d888ee15bd`。

## 官方来源与数值契约

固定 checkpoint revision：`08c8657f02291e11b6f6b63e0e16014f8e748518`。

- [官方 composite config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/config.json)
- [官方 inference config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/config.json)
- [官方计算定义](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/model.py)
- [官方量化与 mHC 定义](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/kernel.py)
- [官方 Engram 哈希定义](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/engram.py)

配置来自 `text_config`；不套用 V4 的默认值。原始层 0/1/2/3 的压缩率为 0/0/2/2，
layer 1 包含 Engram，layer 2 产生被 layer 3 复用的 KV/index 状态。
配置规划器拒绝缺失 source 的层选择，不重编号原始层。
layer 14 Engram、KV sources 8/14/20 和 index sources 的配置依赖已经覆盖；
这些 source 层的真实计算尚未全部覆盖。

V4.1 激活 FP8 scale 是 `2**ceil(log2(max(amax, 1e-4)/448))`，
与现有 V4 packed-MX quantizer 的规则不同。FP8 权重 scale 为 32×32；
FP4 expert 是按 K 每 32 个元素一个 E8M0 scale，原始 checkpoint 以 I8 保存打包字节。
expert 每次 projection 先舍入 BF16，再以 FP32 进入 clamp/SwiGLU。
RMSNorm 使用配置中的 `1e-20`；Engram 按每个 HC copy 独立归一化。

当前 packed 权重和量化 scale 均冻结，激活量化 backward 明确采用 identity STE。
已验证 input / routing / norm / Engram gain / compressor dense-weight 梯度；
没有实现或声称 packed FP8/FP4 权重的可学习量化梯度。

## Compatibility matrix

`Action` 是适配动作；最后一列另列当前实现和实际覆盖，不能据此推断完整模型通过。
路径均相对仓库根目录。

| Module | Existing V4 path | V4.1 change | Action | Test coverage / current status |
| --- | --- | --- | --- | --- |
| Model config | `rl_engine/models/deepseek_v4/` component contracts | composite text config、40 层、5120 hidden | IMPLEMENT | parser、原始层/source 规划与 topology 算术 PASS；model builder BLOCKED |
| Checkpoint loading | `rl_engine/backends/rocm/gemm/checkpoint_fp8.py` | 原名 layer keys、FP8 block32、Engram 大表 | MODIFY | 固定 revision 按 tensor/按行加载 PASS；完整模型绑定 BLOCKED |
| RMSNorm | `rl_engine/backends/rocm/norm/rmsnorm_residual.py` | epsilon 1e-20，5120/1280 等维度 | IMPLEMENT | 新 Triton forward/dx/dweight、dtype/shape/replay PASS |
| Q projection / LoRA | `rl_engine/backends/rocm/gemm/checkpoint_fp8.py` | q rank 1280，block32 | IMPLEMENT | 官方 layer 2 wq_a forward/input STE PASS；V4.1 LoRA 未接入 |
| RoPE | `rl_engine/models/deepseek_v4/partial_rope.py` | source 共享且无 V4 indexer Hadamard | MODIFY | 尚未接入 V4.1；BLOCKED |
| Attention | `rl_engine/models/deepseek_v4/joint_attention.py` | 共享压缩 KV，source-owned indexer | MODIFY | 完整 attention forward/backward BLOCKED |
| KV compressor | `rl_engine/models/deepseek_v4/kv_compressor_c4.py`、`kv_compressor_c128.py` | C1/C2，无 V4 overlap/position bias | IMPLEMENT | C2 pooling及官方 layer 2 compressor 全梯度、partial/decode/reset PASS；C1完整层未验证 |
| Indexer | `rl_engine/models/deepseek_v4/indexer_training.py` | 32 heads、共享 key、candidate blocks | IMPLEMENT | 参数/source 规划 PASS；计算路径 BLOCKED |
| Sparse attention | `rl_engine/backends/rocm/attention/native_sparse.py` | 新 KV/index operand binding | MODIFY | V4 后端保留；V4.1 operand 接线/数值验收 BLOCKED |
| KV cache/state | `rl_engine/models/deepseek_v4/compressor_state.py` | 跨层压缩 KV/index/candidate 生命周期 | IMPLEMENT | compressor 与 hash 自身 state PASS；跨层 attention state BLOCKED |
| MoE router | `rl_engine/models/deepseek_v4/gate.py`、`hash_router.py` | 384 experts，全部采用 V4.1 learned routing | MODIFY | V4.1 router/dispatch BLOCKED |
| Expert forward | `rl_engine/backends/rocm/moe/packed_mx.py` | hidden5120/ffn2304，ceil-scale，BF16边界 | MODIFY | 原 FP4 GEMM 复用；官方 layer 2 expert 0 PASS |
| Expert backward | `rl_engine/models/deepseek_v4/expert_training.py` | 保留新 projection 边界的 input/routing VJP | MODIFY | 官方 expert input/routing 梯度 PASS；384-expert整体未验证 |
| mHC | `rl_engine/backends/rocm/mhc/triton_mhc.py` | pre-mix 由前一 sublayer 传入，RMS eps 改变 | MODIFY | 不直接冒用 V4 layer semantics；V4.1 delayed-mix 集成 BLOCKED |
| Engram | V4 无对应组件 | tokenizer 压缩、ngram、FP8表、gated residual | IMPLEMENT | 真实 tokenizer、layer1/14 hash、真实4行 lookup、gate forward/backward PASS；完整 wkv/table 绑定未验证 |
| Residual/output merge | `rl_engine/models/deepseek_v4/layer_end.py` | delayed pre-mix、Engram前置、终层collapse | MODIFY | V4.1 模型 merge/head 接线 BLOCKED |
| PP/TP/EP/CP | V4 parallel/PP/CP modules | source state 跨 stage/rank 所有权 | IMPLEMENT | 仅配置算术 PASS；模型各 topology BLOCKED |
| Model autograd | V4 component autograd / integrations | 完整 V4.1 text-only 图 | IMPLEMENT | 已实现组件有 autograd；四层模型图 BLOCKED |
| Quantization scales | `rl_engine/backends/rocm/gemm/checkpoint_fp8.py`、`moe/packed_mx.py` | FP8 block32、FP4专家、两种FP4 activation scale | MODIFY | FP8激活/权重和FP4 expert PASS；attention KV FP4 scale16/E4M3与index FP4路径 BLOCKED |
| Distributed communication | `rl_engine/models/deepseek_v4/parallel_collectives.py` | 模型 rank/state 与更新后权重同步 | REUSE | 2/4 rank RCCL gradient smoke/replay PASS；模型通信与权重同步 BLOCKED |

## 可复现运行

使用原测试机已存在的镜像：
`rlk/dsv4-4l-rocm:pr717-ropefix-aade119d-mi300x`。
该标签是本地依赖镜像，当前不是公共 registry 下载地址。
镜像 ID 已保存于结果目录 `1gpu/image.json`；复现需要取得相同依赖环境。
运行时通过只读 mount 使用已提交的 RLK 源码，不使用镜像中较旧的 RLK 实现。
不修改镜像内的 vime、verl 或 vLLM 源码。

从仓库根目录，在干净且已提交的 `dsv4.1-rocm` 工作树执行：

```bash
# 新目录；GPU CSV 必须全部空闲。脚本逐个检查，不会终止其他任务。
examples/operators/dsv41_4l/reproduce.sh \
  rlk/dsv4-4l-rocm:pr717-ropefix-aade119d-mi300x \
  7 /home/rl-kernel/dsv41-reproduce-single
```

已有权重切片时，单独执行记录式验收：

```bash
examples/operators/dsv41_4l/run_operator_acceptance.sh \
  rlk/dsv4-4l-rocm:pr717-ropefix-aade119d-mi300x \
  2,4 /home/rl-kernel/dsv41-artifacts/checkpoint-slice \
  /home/rl-kernel/dsv41-reproduce-two /home/rl-kernel/dsv41-reproduce-cache
```

`fetch_operator_weights.py` 只下载当前组件需要的约 35 MiB tensor 和 4 个真实 Engram 行，
不会生成伪造状态，也不会把该切片命名为四层 checkpoint。
`PinnedCheckpointReader` 要求固定 SHA、HTTP 206 与精确 Content-Range；
服务端忽略 range 时，在读取大型 body 前报错。缺失 tensor 或 Engram 行必须报错。

`operator_acceptance.py` 分别报告容差 correctness、storage-bit replay、
operator autograd-forward/inference-mode parity。两种 mode 共享相同算子 kernel，
没有跨后端或跨框架一致性结论。2/4 rank 还执行 RCCL 梯度重复归约和数值检查，
不是 TP/PP/EP/CP 模型执行。没有跨 topology bitwise equality 结论。

## 已测结果及剩余工作

见 [2026-11-20 实测报告](results/mi300x_components_20261120/README.md)、
[summary.json](results/mi300x_components_20261120/summary.json) 和
[consistency.csv](results/mi300x_components_20261120/consistency.csv)。

下一阶段必须实现模型 construction、attention/indexer/candidate 与跨层状态、
delayed mHC、384-expert 路由/合并、shared expert 和最终 norm/head，
然后按原始 layers 0/1/2/3 实测 Forward/Backward、logits/raw FP32 logprobs 和独立模型级训推路径。
这些是**尚未完成的开发**，不是由于外部训练框架不支持而免除的验收项目。
完整模型不在当前已验证范围内。
