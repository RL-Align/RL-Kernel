# DeepSeek V4.1 ROCm：真实四层模型与 GPU 验收

原始层 **0/1/2/3** 的 text-only 图已接入并执行真实 checkpoint Forward/Backward。
已运行 TP1 PP1、PP2、TP2、TP2 PP2、PP2 DP2；逐项证据见
[本轮模型报告](results/mi300x_model_20261120/README.md)。当前仅允许物理 GPU **4/5/6/7**，八卡测试已由用户暂时取消，不再等待 GPU0。
本轮指定资源的证据见 [GPU4–7 报告](results/mi300x_gpu4567_20261120/README.md)。
[上一轮组件报告](results/mi300x_components_20261120/README.md) 保持原样，不能作为模型证据。

唯一开发分支 `dsv4.1-rocm`，保留 V4 merge `4b41ae76d2b4cd3f87ba71c7b56868d888ee15bd`。
没有修改 `main`、`dsv4-rocm` 或外部框架，没有创建 PR。

## 来源、范围和数值契约

模型 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，固定 revision
`08c8657f02291e11b6f6b63e0e16014f8e748518`。
[composite config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/config.json)、
[inference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/model.py)、
[quantization/mHC](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/kernel.py)、
[Engram](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/engram.py)。

四层保留 hidden5120、64 heads、384 experts/top6、HC4、原始层号与压缩率 0/0/2/2。
Engram 在层 1，层 2 生成 compressed KV/index，层 3 消费同一 source state。
额外的真实权重隔离测试覆盖 Engram14、KV sources8/14/20、index sources24/28/32/36、C1/C2。
这些隔离测试不构成后续完整 block 或完整40层验证。

当前训练 profile 为 `dsv41-text-frozen-quant-embedding-head-input-ste-shared-kernels-v1`：

- 原始 packed FP8/FP4 权重及 scales、embedding、head 冻结；其余 dense/FP32 参数可求梯度和更新。
  尚未实现 packed 权重的可学习量化或全参数训练。
- 激活量化 VJP 使用明确的 identity STE。离散 index top-k 不制造梯度；没有新增 indexer 辅助训练损失。
- RMS epsilon `1e-20`；mHC epsilon `1e-6`、20 次 Sinkhorn；mHC pre 延迟至下一 sublayer 使用。
- FP8 权重 block32×32，激活 scale `2**ceil(log2(max(amax,1e-4)/448))`。
  Indexer FP4 group32/E8M0；compressed KV FP4 group16/E4M3；不添加 V4 Hadamard。
- sparse attention 复用固定 AITER forward/CSR backward；codec、RMSNorm、mHC、路由/合并使用 RLK Triton。
  非量化 dense/grouped projection 和控制器明确使用 PyTorch ROCm library GEMM。
- head 将原 BF16 权重精确提升 FP32，输出完整 FP32 logits 和未舍入的 FP32 token logprobs。
  默认输出相邻 token 的 logprobs；`target_ids` 可为 decode 的每个位置指定真实目标。
- 训练 autograd 与 `torch.inference_mode()` 共享 kernels：只能称 **shared-kernel parity**。
  没有独立训练/推理后端或跨框架 bitwise 结论。
- TP 目前分片 attention Q heads、输出 groups、wo_b 输入维，复制 FFN 和 source caches。
  分片投影按 BF16 求和，与 TP1 的未分片 GEMM 是不同数值契约。
- PP2 cut 位于层1和层2之间，P2P 传递 residual 与 delayed-pre 及各自 VJP。
  不支持将同一 source 与其消费者任意拆到不同 pipeline stage；EP/CP 未接入此模型。
- 支持全有效 text tokens、连续 positions、prefill 后单 token decode。
  任意 padding mask/非连续 position 会明确报错。Engram 只加载本次输入实际需要的真实行；缺行报错。

## Compatibility matrix

路径相对仓库根目录。PASS 只覆盖报告记录的形状、权重和 topology。

| Module | Existing V4 path | V4.1 change | Action | Test coverage |
| --- | --- | --- | --- | --- |
| Model config | `models/deepseek_v4/` | composite config、原始 source 依赖 | IMPLEMENT | parser/source planning PASS |
| Checkpoint | `backends/rocm/gemm/checkpoint_fp8.py` | 原始 tensor/scales、精确 HTTP ranges | MODIFY | 原始四层完整绑定 PASS |
| RMSNorm | `backends/rocm/norm/` | eps1e-20 | IMPLEMENT | forward/input/gain VJP PASS |
| Q / LoRA | `models/deepseek_v4/lora.py` | rank1280、block32 | MODIFY | Q forward/input VJP PASS；V4.1 LoRA NOT RUN |
| RoPE | `models/deepseek_v4/partial_rope.py` | YaRN、无 index Hadamard | IMPLEMENT | GPU forward/transpose VJP PASS |
| Attention | `backends/rocm/attention/native_sparse.py` | C1/C2 source-owned operands | MODIFY | 四层及后续 source 隔离 Forward/Backward PASS |
| Compressor | `models/deepseek_v4/kv_compressor_c4.py` | C1/C2、partial group | IMPLEMENT | 2/8/14/20、decode、dense weight VJP PASS |
| Indexer | `models/deepseek_v4/indexer_training.py` | group32 FP4、shared key、candidate blocks | IMPLEMENT | 2/8/14/20/24/28/32/36 前向 PASS；无辅助训练损失 |
| Sparse attention | `backends/rocm/attention/native_sparse.py` | 新 KV/index binding | REUSE | AITER forward/CSR VJP 实际执行 PASS |
| KV state | `models/deepseek_v4/compressor_state.py` | 原始 owner、跨层复用 | IMPLEMENT | 四层/后续source、reset/partial/decode PASS |
| MoE router | `models/deepseek_v4/gate.py` | 384-way sqrtsoftplus/top6 | IMPLEMENT | selection、route VJP、重放 PASS |
| Expert forward | `backends/rocm/moe/packed_mx.py` | ceil-scale、BF16 boundaries | MODIFY | 真实 FP4 routed + FP8 shared expert PASS |
| Expert backward | `models/deepseek_v4/expert_training.py` | input/route VJP | MODIFY | 选中 experts VJP PASS；packed weight grads 未实现 |
| mHC | `backends/rocm/mhc/triton_mhc.py` | delayed-pre、FP32 controller | IMPLEMENT | controller/gains/Sinkhorn/collapse/post VJP PASS |
| Engram | V4 无 | compressed tokenizer/hash/real-row lookup/gate | IMPLEMENT | 1/14 实际 weights、gain/input VJP PASS |
| Residual / head | `models/deepseek_v4/layer_end.py` | delayed pre、final collapse、FP32 head | IMPLEMENT | 四层 hidden/logits/raw logprobs PASS |
| TP/PP/DP/EP/CP | V4 parallel modules | attention TP、PP边界、DP梯度/权重同步 | IMPLEMENT | TP1/2、PP2、DP2 已测；EP/CP 不支持；8卡已取消 |
| Autograd | V4 component autograd | 四层连接图、状态保留 graph | IMPLEMENT | 四层/跨 decode backward PASS；非全参数训练 |
| Quantization | `backends/rocm/gemm/`、`moe/` | FP8 block32、两种 FP4 activation scales | MODIFY | GPU codec/reference/STE/replay PASS |
| Communication | V4 distributed components | 模型 activation/VJP 和 DP 参数 | MODIFY | 实际模型 P2P/RCCL、重放及同步见逐项报告 |

## 模型复现

测试镜像 `rlk/dsv4-4l-rocm:pr717-ropefix-aade119d-mi300x` 是本机已有依赖镜像，
不是公共下载地址。实际 image ID 和源码 commit 保存在每次运行的 provenance 中。
源码只读挂载，未修改镜像内 vime/verl/vLLM。GPU CSV 必须为空闲的4–7子集，launcher 拒绝0–3和8卡请求，不终止任务。
所有子进程以ROCR选择物理卡，并清除HIP/CUDA二次过滤。每个容器和rank验证实际PCI映射，
映射不符则在RCCL初始化和模型执行前报错。`gpu-mapping.json` 和逐rank summary记录物理/逻辑IDs。
以下命令在仓库根目录执行，每个输出目录必须尚不存在。

先在同镜像的 CPU 容器内下载精确权重；`--metadata` 指向已保存的官方 config 目录：

```bash
python -m examples.operators.dsv41_4l.fetch_model_weights /artifacts/model-4l \
  --metadata /artifacts/official/DeepSeek-V4.1-Flash-1017
python -m examples.operators.dsv41_4l.fetch_attention_weights /artifacts/attention-sources
```

主体四层约31GB，包含384个专家的全部原始 tensors/scales；不会用随机权重补缺。
Engram 仅获取 `cases.json` 中三组真实 tokenizer 输入所需行，缺失原始行必须重新准备。
checkpoint 与 baseline/outputs 大文件保存在 `/home/rl-kernel/dsv41-artifacts/`，不提交到 Git。

```bash
image=rlk/dsv4-4l-rocm:pr717-ropefix-aade119d-mi300x
weights=/home/rl-kernel/dsv41-artifacts/model-4l
out=/home/rl-kernel/dsv41-new-reproduction
mkdir "$out"
run=examples/operators/dsv41_4l/run_model_gpu.sh
"$run" "$image" 4 "$weights" "$out/tp1" 1 1 1 -
"$run" "$image" 4,5 "$weights" "$out/tp2" 2 1 1 -
"$run" "$image" 4,5 "$weights" "$out/pp2" 1 2 1 "$out/tp1/baseline.pt"
"$run" "$image" 4,5,6,7 "$weights" "$out/tp2pp2" 2 2 1 "$out/tp2"
DSV41_MODEL_OPTIMIZER_STEPS=2 "$run" "$image" 4,5,6,7 "$weights" \
  "$out/pp2dp2" 1 2 2 "$out/tp1/baseline.pt"
```

`run_model_state.py` 运行真实四层奇数 prefill 和连续 decode Backward；
`run_training_stress.py` 覆盖 seeds17/1234、两组不同 prompt 梯度累积、batch1/2、长度7/8、
两次 SGD 更新、整段重放和更新后 shared-kernel parity。
这些脚本也要求 `DSV41_EXPECTED_GPU_MAPPING` 指向GPU host mapping文件，且ROCR与其一致。
`DSV41_MODEL_CASE=1` 可生成第二个真实prompt的单卡基准；为PP2 DP2另设
`DSV41_DP_SECOND_BASELINE=/path/to/case1/baseline.pt`，验证不同prompt的梯度平均和更新。详细命令及实际状态见本轮报告。

`compare_bits` 比较原 dtype 的 storage bytes，区分 signed zero/NaN payload，不使用 allclose 或舍入。
Operator correctness 的 reference 数值容差测试单列；其 PASS 不等于独立模型后端 bitwise PASS。
所有 mismatch 保存 first operator/layer-key/rank/bit patterns。数量是包含重复比较的存储元素数。

组件级旧入口 `reproduce.sh` / `run_operator_acceptance.sh` 继续保留；它们执行的是隔离组件，
不是上述模型 launcher。40层、全参数训练、独立后端、长上下文 candidate pruning 和全部 topology
仍不能由当前结果推断通过。
