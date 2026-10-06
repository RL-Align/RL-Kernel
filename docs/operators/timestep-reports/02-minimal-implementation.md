# 阶段 2：最小实现与 gtest（2026-10-01）

## 目标与实现

在阶段 1 的固定合同下接入官方 gtest，保持参考与候选实现独立。
新增 NativeTimestepEmbedMLPOp（逐行 torch.mv + autograd）、CUDA 固定 tile FP32
FMA 原语、Triton 固定 tile IEEE FP32 dot，以及共用的显式前后向调度。
固定 256 通道，生产 hidden=3072；小 hidden 只为边界测试。
CUDA 延迟 JIT 构建，缓存路径由调用环境设置；Triton 无 autotune/atomic。

gtest 新增 reduction spec 与生产形状输入工厂；五个可微输入全部注册。
不修改 tolerance_contract.json。运行时入口直接导入 TimestepEmbedMLPOp；未改变全仓
registry 的既有调度优先级。默认后端失败即抛错，allow_fallback=True 才允许
PyTorch fallback；返回独立 trace，后向向同一个 trace 追加实际请求的原语。
launch_record 只是调度证据，后续必须用 profiler 记录真实 CUDA kernel。

梯度推导和归约顺序见阶段 1。参数梯度在完整逻辑样本集上归约，不把分块后的
BF16 梯度相加。sample_ids 排序、active_mask 去 padding，输出映射回物理布局。
非连续输入在内部规范化；没有数值身份哈希、MKL/CPU 库指纹或历史缓存依赖。

## 已执行

环境：本机 Python 3.10，PyTorch 2.6.0+cu124，Triton 3.2.0，GTX 1650 Ti。
源代码基线 32b765e；新增代码未提交，实际证据文件保存在 evidence/。

- `OMP_NUM_THREADS=4 python3 scripts/check_operator.py --op timestep_embed_mlp --candidate pytorch --device cpu --dtype fp32 --batch 2 --seq 1 --check-grad`：6/6 输出与梯度比较通过；cpu-gtest-fp32.log。
- 上述 `--dtype bf16`：6/6 通过；前向最大绝对误差 .003863573；cpu-gtest-bf16.log。
- `OMP_NUM_THREADS=4 python3 -m unittest discover -s tests -p test_timestep_official.py -v`：初始 5 项 CPU 测试通过，1.460 秒；cpu-unit-initial.log。后续新增 GPU 边界测试尚未在该日志中运行。
- 本机 Triton 生产 H=3072 FP32 官方 gtest，batch=2, seq=1, 默认 seed=123：6/6 通过；前向 max_abs=2.980232e-6，dt max_abs=.0029296875（满足官方相对误差）；local-triton-fp32-initial.log。
- `python3 scripts/validate_timestep_official.py --backends triton --hidden 17 --batches 3 --seeds 386 --output .../local-small-initial.json`：FP32 与 BF16 两行全部通过，耗时 6.963 秒 / .077 秒。含 CPU 独立参考、重复、chunk 1/2、重排、padding、strided、singleton 行逐字节比较。实际源码 SHA 存于 JSON。

CPU gold 的自比较只证明接入。小宽度不替代生产宽度。launch 标签不替代 profiler。

## A100 状态与失败保留

用户直接批准上传 rl_engine/csrc/scripts/tests 到
/home/linux/timestep-official-astra-01a0f7ef/src；缓存与输出在同根目录下独立。
A100-SXM4-40GB 初始空闲（14 MiB，0%，无 compute process）。
现有 venv /home/linux/timestep-a100.ydZtoB/venv 只读复用，PyTorch 2.6.0+cu124、Triton 3.2.0。
初始密钥认证失败；用户提供交互认证后成功。密码未写入源码/报告/命令行。
首次上传被自动审批拒绝，用户明确批准后重试；旧控制连接失联，确认远端 src
仍为空且无运行作业后，仅关闭本任务旧连接，用带 keepalive 的新连接继续。
没有杀死远端进程或修改研究目录。

下一阶段：生产宽度 A100 FP32/BF16 + CPU gold + 不变性 + 真实 trace；保留失败
而不调整容差。当前没有受支持的原生压缩入口，继续执行。
