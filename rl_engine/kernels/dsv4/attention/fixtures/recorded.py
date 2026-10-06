# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import hashlib

import torch
from torch import Tensor


def tensor_checksum(tensor: Tensor) -> str:
    blob = tensor.detach().cpu().contiguous()
    data = blob.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()
