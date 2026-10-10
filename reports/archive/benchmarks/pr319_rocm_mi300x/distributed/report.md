# PR #319 — strict ROCm CP attention, multi-rank acceptance (MI300X)

`torchrun --standalone --nproc-per-node=N scripts/ws2_p2p_nccl_attention_reference_check.py \
  --transport rccl_ag_rs --strict-shared-core`

Requires the native extension built for the active GPU platform
(`PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace`); without it the strict path
fails closed on the ROCm deterministic RoPE operator rather than running a different one.

| world | TP | CP | replicas | transport | ranks passed | out | lse | dQ | dK | dV |
|---:|---:|---:|---:|---|---|---|---|---|---|---|
| 2 | 1 | 2 | 1 | rccl_ag_rs | 2/2 | bitwise | bitwise | bitwise | bitwise | bitwise |
| 4 | 2 | 2 | 1 | rccl_ag_rs | 4/4 | bitwise | bitwise | bitwise | bitwise | bitwise |
| 8 | 2 | 2 | 2 | rccl_ag_rs | 8/8 | bitwise | bitwise | bitwise | bitwise | bitwise |

Every rank reports `strict_shared_core.executed=true` and `passed=true`.
