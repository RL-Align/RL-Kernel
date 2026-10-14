# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from typing import Any

import torch

def gemm_fp32_tn(a: torch.Tensor, b_transposed: torch.Tensor) -> torch.Tensor: ...
def build_info() -> dict[str, Any]: ...
