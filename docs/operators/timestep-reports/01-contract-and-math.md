# 阶段 1：合同与数学规格（2026-10-01）

## 身份、基线与证据

正式任务 01a0f7ef-7063-71e1-b087-24915728539e；host local；工作区
/home/hlb/.codex/worktrees/105a/RL-Kernel；分支 codex/timestep-official-astra。
初始 detached HEAD 干净，32b765ec992cd1206517104ec66506881203c91c。
`git ls-remote origin refs/heads/test-qwenimage refs/heads/main` 确认目标分支未移动；
主线固定 1968a87114c331513f9d320c9bd526f0d91f3a52。
两 SHA 的 rl_engine/kernels/gtest、docs/contributing/gtest-usage.md、
scripts/check_operator.py 完全相同（git diff --stat 输出为空）。不导入 main。

已通过 gh api 重新读取 issue comment 5926837071；维护者要求标准 256-channel
sinusoidal → Linear 256→3072 → SiLU → Linear 3072→3072 前后向，reduction 类，
CPU/GPU 在 gtest 容差内；SM90 可选。#204 为已合并 PR，头提交
ddf237fbda67dbd5ab8a1eeb592a033472edab08。
参考其注册、trace 和 fallback 范式，不复制实现或证据。

上游 Diffusers 固定 031b2798addadd1652db7cfba50eacc1079245cf，
transformer_qwenimage.py 的 QwenTimestepProjEmbeddings 已经 gh api 核对：
num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000。
参数 nn.Linear 布局 [out,in]。输入边界为模块 timestep，不再做 pipeline 缩放。

## 数学与精度选择

i=0..127，f_i=exp((-log(10000)*i)/128)，p_ni=(t_n*f_i)*1000。
E=[cos(p),sin(p)]；Z=E W1^T+b1；H=Z sigmoid(Z)；Y=H W2^T+b2。
G 为显式上游梯度。dW2=G^T H，db2=sum_n G；
dZ=(G W2) * (s+Z*s*(1-s))；dW1=dZ^T E，db1=sum_n dZ；
dE=dZ W1；dt_n=sum_i ((-dE_ni*sin(p_ni)+dE_n,i+128*cos(p_ni))*1000)*f_i。

FP32/BF16 参数进入运算时提升 FP32，点积、激活和保存的中间值 FP32，最终输出
及参数梯度回到参数 dtype；timestep 保持 FP32。融合路径不在每个中间节点额外
量化 BF16，与 gtest FP32 gold 对照；不声称逐位重放 Diffusers 原生 BF16 运算。
频率用普通 PyTorch FP32 CPU 生成后复制；无库哈希、MKL 派发或频率指纹门禁。

CUDA 固定 K 顺序 FP32 FMA，Triton 固定 tile 和 FP32 ieee dot；二者归约树可不同。
不使用原子归约或依据 batch 的 autotune。输出和 dt 的归约仅沿固定特征维。
参数梯度沿完整逻辑样本集固定顺序归约；chunk_size 仅改变前向分块，不拆分参数
梯度归约。layout API 通过 sample_ids 和 active_mask 恢复逻辑顺序再执行。
普通外部 microbatch backward 的 BF16 .grad 相加不承诺逐位不变，不能作为此
接口的配置变换；报告必须注明测试的是同一逻辑样本集。

## 合同与验证计划

合同 SHA 即目标基线（与上述 main 同内容），version ws1-c1-v2；容差文件不修改。
用 resolve_tolerance 分别解析 forward_accuracy、gradient_accuracy 与两种 invariance。
FP32 reduction atol/rtol=1e-4/1e-4；BF16 前向 .05/.02，梯度 .1/.02；不变性逐位。
独立 PyTorch row-mv gold + autograd，另用普通 F.linear 核验标准语义。
随机种子 386、9386；生产 H=3072，小 H 仅用于边界与梯度测试。
需保存运行源码 hash、环境、真实 profiler kernel trace、全部误差与失败。

## 实际结果与限制

本地 PyTorch 2.6.0+cu124、Triton 3.2.0、CUDA toolkit 12.8，GTX 1650 Ti 空闲。
A100 默认 SSH 认证失败：Permission denied (publickey,password)，已请求可用连接入口。
未执行 GPU 正确性或性能测试；当前阶段只有合同/基线只读核查通过。
未发现 AGENTS.md；已读仓库 .claude/skills/ws1-single-card-kernel/SKILL.md。
研究工作区只读，未复用旧代码，仅审查参考语义。

下一步：新增最小 gold、CUDA/Triton 前后向、gtest 注册，再运行本地 smoke。
当前没有可用的原生上下文压缩工具；此检查点不等于原生压缩，不因此暂停。
