# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""NCCL transport for expert parallelism on CUDA."""

from __future__ import annotations

import torch
import torch.distributed as dist

from rl_engine.distributed.ep.exchange import EpExchangeError, TorchAllToAll

SUPPORTED_EP_SIZES = (1, 2, 4, 8)


class CudaNcclAllToAll(TorchAllToAll):
    """``all_to_all_single`` on an NCCL group. Anything else fails instead of falling back."""

    backend_id = "nccl"

    def __init__(self, group) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            raise EpExchangeError("process group required")
        if "nccl" not in str(dist.get_backend(group)).lower():
            raise EpExchangeError("CUDA expert exchange requires an NCCL process group")
        if torch.version.hip or not torch.cuda.is_available():
            raise EpExchangeError("CUDA expert exchange requires a CUDA build and device")
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        if self.size not in SUPPORTED_EP_SIZES:
            raise EpExchangeError(f"EP size {self.size} is outside {SUPPORTED_EP_SIZES}")
        self.device = torch.device("cuda", torch.cuda.current_device())


__all__ = ["SUPPORTED_EP_SIZES", "CudaNcclAllToAll"]
