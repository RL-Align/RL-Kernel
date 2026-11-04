# SPDX-License-Identifier: Apache-2.0
"""Reuse aiter's fixed per-row reduction. No new score or head-sum kernel."""

import hashlib
import importlib
from pathlib import Path

import torch

from rl_engine.contracts.operators.indexer_score import (
    ROW_DOT_MAX_WIDTH,
    SCORE_BACKEND_ID,
    IndexerScoreError,
    require,
)

ROW_DOT_MODULE = "aiter.ops.triton._triton_kernels.attention.mha_fused_bwd"
ROW_DOT_LAUNCH = {"BLOCK_M": 1, "num_warps": 4, "IS_VARLEN": False, "IS_FP8": False}


class AiterIndexerScoreBackend:
    backend_id = SCORE_BACKEND_ID
    device_type = "cuda"

    def __init__(self):
        require(torch.version.hip is not None and torch.cuda.is_available(), "requires a ROCm GPU")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        require(arch == "gfx950", f"unvalidated architecture {arch}")
        try:
            self.triton = importlib.import_module("triton")
            module = importlib.import_module(ROW_DOT_MODULE)
            self._row_dot = module._bwd_preprocess
        except (ImportError, AttributeError) as exc:
            raise IndexerScoreError("required aiter row reduction unavailable") from exc
        self.provenance = {
            "backend": self.backend_id,
            "arch": arch,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": self.triton.__version__,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sources": {
                ROW_DOT_MODULE: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            },
        }

    @staticmethod
    def plan(contract):
        return dict(ROW_DOT_LAUNCH)

    def row_dot(self, a, b):
        """sum(a * b, -1) with one fixed reduction per [row, column]; b may broadcast."""
        require(a.ndim == b.ndim == 3 and a.shape == b.shape, "row-dot shape mismatch")
        require(a.dtype == b.dtype == torch.float32, "row-dot operands must be FP32")
        require(a.device == b.device == self.device, "row-dot device mismatch")
        m, h, d = a.shape
        require(d <= ROW_DOT_MAX_WIDTH, "row-dot feature budget exceeded")
        require(h <= 65535, "row-dot columns exceed the grid z limit")
        with torch.cuda.device(self.device):
            if not m or not h or not d:
                return torch.zeros((m, h), dtype=torch.float32, device=self.device)
            out = torch.empty((m, h), dtype=torch.float32, device=self.device)
            self._row_dot[(m, 1, h)](
                a,
                b,
                out,
                0,
                a.stride(1),
                a.stride(0),
                a.stride(2),
                0,
                b.stride(1),
                b.stride(0),
                b.stride(2),
                0,
                out.stride(1),
                out.stride(0),
                0,
                None,
                m,
                None,
                BLOCK_D_MODEL=d,
                BLOCK_D_MODEL_POW2=self.triton.next_power_of_2(d),
                **ROW_DOT_LAUNCH,
            )
            return out
