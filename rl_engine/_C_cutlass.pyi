# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from typing import Any

import torch

def gemm_fp32_tn(a: torch.Tensor, b_transposed: torch.Tensor) -> torch.Tensor: ...
def gemm_fp32_partials_strided(
    a: torch.Tensor,
    b_transposed: torch.Tensor,
    workspace: torch.Tensor,
    starts: torch.Tensor,
    nodes: torch.Tensor,
    partial_ids: torch.Tensor,
    length: int,
) -> None: ...
def build_info() -> dict[str, Any]: ...
