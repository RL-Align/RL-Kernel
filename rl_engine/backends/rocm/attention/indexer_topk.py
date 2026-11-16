# SPDX-License-Identifier: Apache-2.0
"""Stable Top-K by packed (score, ~id) keys, the item 31 scheme with aiter's fpval_to_key."""

import hashlib
import importlib
from pathlib import Path

import torch
import triton
import triton.language as tl

from rl_engine.contracts.operators.indexer_topk import TOPK_BACKEND_ID, require

KEY_MODULE = "aiter.ops.triton._triton_kernels.moe.moe_routing.topk"
LAUNCH = {"num_warps": 8, "num_stages": 1}
_fpval_to_key = importlib.import_module(KEY_MODULE).fpval_to_key


@triton.jit
def _pack(value, column, valid):
    raw = value.to(tl.uint32, bitcast=True)
    raw = tl.where((raw & 0x7FFFFFFF) == 0, 0, raw).to(tl.uint32)
    packed = (_fpval_to_key(raw).to(tl.uint64) << 32) | (0x7FFFFFFF - column).to(tl.uint64)
    return tl.where(valid & (value != float("-inf")), packed, 0)


@triton.jit
def _indexer_topk(SCORES, LENGTHS, OUT, stride, K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(LENGTHS + row)
    offsets = tl.arange(0, K)
    candidates = tl.zeros((K,), tl.uint64)
    for start in range(0, length, K):
        column = start + offsets
        valid = column < length
        value = tl.load(SCORES + row * stride + column, mask=valid, other=0.0)
        merged = tl.cat(candidates, _pack(value, column, valid), can_reorder=True)
        candidates = tl.reshape(tl.topk(merged[None, :], K, dim=1), (K,))
    column = 0x7FFFFFFF - (candidates & 0x7FFFFFFF).to(tl.int64)
    column = tl.sort(tl.where(candidates != 0, column, -1), descending=True)
    tl.store(OUT + row * K + offsets, column.to(tl.int32))


class TritonIndexerTopKBackend:
    backend_id = TOPK_BACKEND_ID
    device_type = "cuda"

    def __init__(self):
        require(torch.version.hip is not None and torch.cuda.is_available(), "requires a ROCm GPU")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        require(arch == "gfx942", f"unvalidated architecture {arch}")
        key = importlib.import_module(KEY_MODULE)
        self.provenance = {
            "backend": self.backend_id,
            "arch": arch,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "launch": LAUNCH,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sources": {KEY_MODULE: hashlib.sha256(Path(key.__file__).read_bytes()).hexdigest()},
        }

    @staticmethod
    def plan(contract):
        return dict(LAUNCH, block=contract.topk)

    def select(self, scores, lengths, k):
        rows = scores.shape[0]
        out = torch.empty((rows, k), dtype=torch.int32, device=self.device)
        if rows:
            with torch.cuda.device(self.device):
                _indexer_topk[(rows,)](scores, lengths, out, scores.stride(0), K=k, **LAUNCH)
        return out
