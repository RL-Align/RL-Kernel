# V4.1 MI300X 实测报告（2026-11-20）

**整体目标尚未完成：BLOCKED。** 已合并 V4 并完成下列组件开发和真实 GPU 验证。
尚未构造或运行真实 V4.1 四层模型，模型级 Forward/Backward、logits、raw FP32 logprobs、
模型级 deterministic replay 和 train/inference parity 全部为 **NOT RUN**，mismatch 为 `null`。

## Git 与实现

- 分支：`dsv4.1-rocm`。
- main 基线：`e9d52e95226b62d826f0a224bafad2cab3ed7538`。
- 合入 V4：`71c7cb2d033806632cef1a3678939bf73dc581dd`。
- 独立 merge：`4b41ae76d2b4cd3f87ba71c7b56868d888ee15bd`。
- 本报告三组 GPU 测量的统一实现 SHA：`53c85b1d17de1d51765531b0d16516747e8f54ec`（下方 JSON 为权威完整 SHA）。
- [summary.json](summary.json) 与各 rank JSON 保存完整 SHA 和所导入源码 SHA256。
  后续报告提交会改变分支 HEAD，不能把新的 HEAD 冒认为当时测量 pin。
- V4 原有模型组件、packed-MX 后端和 `examples/operators/dsv4_4l/` 相对合入的 V4 分支无改动。
  两处文档导航冲突保留两侧条目。历史 CSV 的 CRLF 字节保留；整个 merge 的 whitespace 检查
  使用 `git -c core.whitespace=cr-at-eol diff --check`，新增文件正常 `git diff --check` 通过。

实现了 config/source-layer 规划、按 tensor/行读取的 checkpoint 接口、block32 FP8 quant/GEMM/VJP、
FP4 expert 与路由权重梯度、epsilon-driven RMSNorm、C2 pooling 和 compressor partial state、
Engram tokenizer 映射/hash/真实行 lookup/gated residual 及显式组件组装接口。
[compatibility matrix](../../README.md#compatibility-matrix) 列出 20 项动作、复用边界和缺口。

未完成：完整 model builder/权重绑定、attention/indexer/candidate/cross-layer state、
V4.1 delayed mHC、384-expert 路由/dispatch/combine/shared-expert 模型绑定、最终输出 head、
模型级 autograd、TP/PP/DP/EP/CP 与权重同步。
Engram 完整 wkv/table 模块没有端到端验收；真实 source layers 8/14/20 等仍需 targeted GPU tests。
不能将 layer 1/14 哈希测试称为 layer 14 模型测试。

## 环境与输入

实测 8 张 AMD Instinct MI300X VF，gfx942，每卡约 191.7 GiB；不是 MI355X。
外部 PID 3131740 占用物理 GPU 0，测试未终止或抢占它。
最初 4 卡列表包含 GPU 0 时 idle guard 拒绝启动；随后使用 1/2/4/6。
8 GPU 验收为 **NOT RUN**，见 [idle guard](logs/8gpu-required-gpu0-preflight.log)。

Host ROCm header 为 10.1.0；镜像 PyTorch 为 `2.13.0+rocm10.0.0`，
HIP `7.15.26333`、Triton `3.8.0+git669b31ac`、amd-aiter `0.1.29`、RCCL `2.30.4`。
transformers `5.22.0`、safetensors `0.8.0`。这不是 pyproject 依赖安装的全新环境验收。
使用现有镜像及其 AITER/CK；新组件显式走 RLK Triton，没有修改外部仓库。
镜像 pin 见 [image.json](1gpu/image.json)。

官方 checkpoint revision 为 `08c8657f02291e11b6f6b63e0e16014f8e748518`。
[权重 manifest](checkpoint-slice-manifest.json) 记录每个精确 byte range 与 SHA256。
只下载约 35 MiB 的官方组件权重和 4 个真实 Engram 行，未下载完整约 510 GB checkpoint。
layer 1 Engram shard 的约 101.5 GB 大表采用精确行读取，不替换为随机小表。
[真实 tokenizer manifest](tokenizer-manifest.json) 验证 129280 raw IDs 压缩为 99092 IDs；
对应 map SHA256 为 `26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496`。

## 测试矩阵

三个判断分别报告：A=对 reference 的容差 correctness；B=精确 storage-bit replay；
C=组件 autograd-forward/inference-mode parity。C 两侧共享相同 kernel，
不代表模型级、跨后端或跨框架一致性。

| Configuration | GPU Count | Model Layers | Forward / Backward | A | B | C（仅组件） | Compared elements | Mismatch | Status | Evidence |
| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| 隔离算子，物理 GPU 7 | 1 | 无模型；layer1/2组件 | PASS / PASS | PASS | PASS | PASS | 6,055,471 | 0 | PASS | [rank0](1gpu/operator-rank-0.json)、[log](1gpu/container.log) |
| 隔离算子 + RCCL，物理 2/4 | 2 | 无模型；layer1/2组件 | PASS / PASS | PASS | PASS | PASS | 12,356,702 | 0 | PASS | [rank0](2gpu/operator-rank-0.json)、[rank1](2gpu/operator-rank-1.json)、[log](2gpu/container.log) |
| 隔离算子 + RCCL，物理 1/2/4/6 | 4 | 无模型；layer1/2组件 | PASS / PASS | PASS | PASS | PASS | 24,713,404 | 0 | PASS | [rank0](4gpu/operator-rank-0.json)、[rank1](4gpu/operator-rank-1.json)、[rank2](4gpu/operator-rank-2.json)、[rank3](4gpu/operator-rank-3.json)、[log](4gpu/container.log) |
| TP1 PP1 DP1 模型 | 1 | 原始 0/1/2/3 | NOT RUN / NOT RUN | NOT RUN | NOT RUN | NOT RUN | — | — | BLOCKED | [summary](summary.json) |
| PP2 / TP2 模型 | 2 | 原始 0/1/2/3 | NOT RUN / NOT RUN | NOT RUN | NOT RUN | NOT RUN | — | — | BLOCKED | [summary](summary.json) |
| TP2 PP2 / PP2 DP2 模型 | 4 | 原始 0/1/2/3 | NOT RUN / NOT RUN | NOT RUN | NOT RUN | NOT RUN | — | — | BLOCKED | [summary](summary.json) |
| TP4 PP2 / TP2 PP2 DP2 模型 | 8 | 原始 0/1/2/3 | NOT RUN / NOT RUN | NOT RUN | NOT RUN | NOT RUN | — | — | BLOCKED | [summary](summary.json)、[GPU占用](logs/8gpu-required-gpu0-preflight.log) |

共比较 **43,125,577 个 storage elements，0 mismatch**。
数字包含重复 tensor/token 比较、梯度、optimizer snapshots 和部分 RCCL 重放比较，
不是唯一 prompt token 数；逐项 token 计数见 rank JSON。
FP4 比较按打包 storage elements 计数；比较器区分 +0/-0 和不同 NaN payload，检查 dtype/shape，
首差异包含 layer/operator/rank/index/实际 bit patterns。没有用 allclose 代替 bitwise。
allclose/assert_close 仅用于 A；A 的非零参考误差及容差分别保存在每个 case 中。

每个 rank 执行 12 组记录式 case：seed 11/29/53，batch 1/2/2，sequence length 3/5/17，
覆盖官方 wq_a、官方 expert0、官方 gain 的 Engram gate，以及 C2 pooling。
Engram gate 的 stress 为两条独立执行的 3-step SGD 轨迹，每步累积两个 microbatches；
检查真实梯度、更新后参数和输出重复性，以及更新后组件 parity。
2/4 rank 对不同 rank 数据的梯度执行 RCCL reduction，检查数值和同 topology 重放。
没有执行模型权重同步、模型 prompt rollout、raw FP32 logprob 或跨 topology bitwise 对比。

## 回归、发现及修复

- V4 contracts：264 PASS，见 [最终日志](logs/v4-contracts-final-rocm-runtime.log)。
  逻辑测试在可见 GPU 的 ROCm 依赖环境运行。
- V4 GPU kernels：91 PASS，见 [最终日志](logs/v4-gpu-final.log)。
- V4.1 tests：71 PASS，0 skip，见 [最终日志](logs/v41-tests-final.log)。
- Layout：33 PASS、1 个现有 CI-only extension enforcement skip；
  [原因日志](logs/layout-final-with-reasons.log)。这没有验收编译 CUDA extension。
- 额外 CPU-only 容器尝试为 **FAIL**：既有 rollout-config test 在 collection 时经 verl/vLLM
  初始化 ROCm platform，因无 GPU 失败，见 [原始错误](logs/v4-cpu-final.log)。
  未删除或跳过该测试；相同 264 项在 ROCm runtime 通过。
- 未重新运行 V4 完整四层 GRPO；保留 PR #711/#717 的原始报告、历史 mismatch 与限制，
  没有把当前算子回归推广成历史问题全部解决。

修复并保留失败证据：

1. expert autograd 误把 dataclass state 当 iterable 保存；改为显式保存 tensor 字段。
   [首次失败日志](logs/v41-official-tests.log) 与 [修复后日志](logs/v41-official-tests-fix1.log)。
2. 容器 worktree `.git` 指向 host 路径，runner 无法读取 SHA；改为由干净 host 工作树显式提供
   SHA，同时记录实际导入文件 hashes。见 [失败日志](resolved-failures/acceptance-1gpu/container.log)。
3. `torchrun --standalone` 在离线容器解析 hostname，未启动 ranks；改用静态 loopback rendezvous。
   仅停止了本任务自己的挂起容器。见 [原始日志](resolved-failures/acceptance-2gpu-committed/container.log)。
4. 两卡真实运行暴露 `safe_open(device="cuda")` 把 rank1 权重放到 GPU0；
   改为显式解析当前 CUDA device index。见 [rank1原始失败](resolved-failures/acceptance-2gpu-final/operator-rank-1.json)。

这些是接口/设备/launcher 修复，未通过 rounding、改变输入或跳过计算消除数值 mismatch。
当前记录式 bitwise 用例没有观测到数值 mismatch；模型级 mismatch 未测，不能记为 0。

## 复现与后续验收

一键复现脚本已经在新目录完整执行：12 个记录式 case 和 71 项测试通过，见
[复现记录](reproduce-check.json) 与 [日志](logs/reproduce-entrypoint-check.log)。
该额外复现未加入上面的 43,125,577 计数。

复现命令见 [入口 README](../../README.md#可复现运行) 和
[reproduce.sh](../../reproduce.sh)。所有输出必须使用新目录。
汇总原始 rank JSON 的命令写在 [summarize_acceptance.py](../../summarize_acceptance.py) 的 CLI 中；
本报告是固定日期实验，含历史失败与环境约束，不是通用模型验收器。
[consistency.csv](consistency.csv) 提供每个请求 topology 的独立状态。
[evidence-sha256.json](evidence-sha256.json) 校验原始实测 JSON/log/manifest；README 不在实测 hash 清单中。

后续仍需完成上述模型接口和算子接线，加载真实 layers 0/1/2/3 及所需 source state，
分别执行模型级 A/B/C，然后补齐 layer14/source8/14/20、不同压缩率、FP4 attention/indexer、
全部合法多卡 topology、多步训练和更新后权重同步。8 GPU 需等待外部 GPU0 任务释放。
以上未完成项均保留在 `dsv4.1-rocm`，不能标记 Definition of Done 全部通过。

## 发布状态

代码和结果已在本机 `dsv4.1-rocm` 提交，**尚未 push**。HTTPS push 缺少 Git 凭据，
SSH 为 `Permission denied (publickey)`；见 [publication.json](publication.json)。
远端 `dsv4.1-rocm` 仍为 `e9d52e95`，未创建 PR，未更新 main/dsv4-rocm。
GitHub app 的账号连接不能替代本机 Git transport 认证；未重建提交或改写测量 pin 来绕过。
配置本机 Git/SSH 认证后，从此工作树执行 `git push origin dsv4.1-rocm`。

新增源码分别位于 `rl_engine/models/deepseek_v41/`、`rl_engine/backends/rocm/engram/`，
以及 ROCm attention/norm/gemm/moe 下的 `v41_*` 文件；测试位于
`tests/models/deepseek_v41/`，复现及结果位于 `examples/operators/dsv41_4l/`。
`.gitattributes` 仅对本目录 raw logs 禁止换行转换和 whitespace lint，保留日志原始字节；
代码 whitespace/lint 检查正常执行。
