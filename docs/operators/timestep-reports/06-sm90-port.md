# sm90 兼容迁移（待实机验收）

> 本文保留迁移/草稿时的历史状态。2026-10-07 已完成 H100 实机验收和参考独立审查，见[最新报告](08-h100-reference-audit.md)。

本次新增 sm90 验证入口，不改变算子数学、FP32 补偿求和、参考实现或误差阈值。
当前 CUDA 内核使用通用 FP32/FMA/warp shuffle，Triton 使用固定归约；尚未实现
Hopper WGMMA/TMA 专用优化。源码适配不能代替 H100/H200 实机正确性与性能验收。

## 在目标机器运行

使用支持 sm90 的 CUDA 工具链（建议沿用 CUDA 12.4、PyTorch 2.6.0+cu124、
Triton 3.2.0 环境），激活环境后在仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.4 \
TIMESTEP_RUN_ROOT="$PWD/.timestep-cache/sm90" \
bash scripts/run_timestep_sm90.sh
```

入口要求当前 GPU 的 compute capability 为 9.0，并设置 TORCH_CUDA_ARCH_LIST=9.0。
TIMESTEP_CUDA_JIT_ONLY=1 跳过已安装的 rl_engine._timestep_cuda，以当前源码重新
JIT 编译；每次运行使用独立 CUDA/Triton 缓存，防止复用 A100 编译产物。
此变量需在进程第一次加载扩展前设置。默认 A100 入口行为保持不变。
不需要设置其他算子的 KERNEL_ALIGN_FORCE_SM90；本入口只构建 timestep 独立扩展。

验证沿用原有 pytest、4 组官方 gtest、24 组生产形状矩阵、profiler 和 benchmark。
报告写入指定目录下的 validation.*，任何阶段失败都会停止。
与 A100 的逐位一致性没有额外保证；须在目标卡检查输出、五类梯度及既有不变性。
原参考实现独立准确性审查的证据缺口仍然存在，迁移不会自动消除它。

## 当前验证边界

本地 GPU 为 GTX 1650 Ti（sm75），没有 sm90 实机；仅完成脚本语法与架构路由检查。
没有宣称 sm90 编译、运行、精度或性能通过。既有 A100 报告属于先前版本；本次
新增入口/加载配置后应生成新的源文件清单与目标机报告。原恢复包没有被修改。

专用性能优化应单独评估固定 tile、数据复用与融合，再考虑 WGMMA/TMA。
BF16/TF32 Tensor Core 路径可能改变乘法精度与归约顺序，必须重新检查所有梯度，
尤其历史敏感的 dt；不能以放宽阈值或再次修改参考来代替验证。

参考：
- https://docs.nvidia.com/cuda/archive/12.9.2/hopper-compatibility-guide/index.html
- https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
