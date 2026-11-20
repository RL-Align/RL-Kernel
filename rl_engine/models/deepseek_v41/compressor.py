# SPDX-License-Identifier: Apache-2.0
"""Source-owned V4.1 C1/C2 compressor, including real partial-group state.

Dense BF16/FP32 projections use PyTorch's ROCm GEMM; pooling and normalization
use the explicit RLK Triton kernels. These are library GEMMs, not V4 C4 overlap
kernels. An incomplete group produces None, exactly as the released model.
"""

import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.attention.v41_compressor_pool import compressor_pool
from rl_engine.backends.rocm.norm.v41_rmsnorm import rmsnorm


class Compressor(torch.nn.Module):
    def __init__(self, config, original_layer_id, wkv, norm_weight, wgate=None):
        super().__init__()
        t = config.text
        plan = config.layer_plan(original_layer_id)
        if not plan.owns_kv:
            raise ValueError("only a KV source layer owns a compressor")
        self.layer_id = original_layer_id
        self.ratio = plan.compress_ratio
        self.eps = t["rms_norm_eps"]
        self.dim = t["hidden_size"]
        self.head_dim = t["head_dim"]
        if wkv.shape != (self.head_dim, self.dim) or norm_weight.shape != (self.head_dim,):
            raise ValueError("compressor checkpoint shape mismatch")
        if wkv.device.type != "cuda" or not torch.version.hip or norm_weight.device != wkv.device:
            raise ValueError("compressor weights must share an AMD ROCm GPU")
        if self.ratio == 2 and (
            wgate is None or wgate.shape != wkv.shape or wgate.device != wkv.device
        ):
            raise ValueError("C2 needs the original gate projection")
        if self.ratio == 1 and wgate is not None:
            raise ValueError("C1 has no gate projection")
        dtype = torch.float32 if self.ratio == 2 else torch.bfloat16
        self.wkv = torch.nn.Parameter(wkv.to(dtype))
        self.wgate = torch.nn.Parameter(wgate.float()) if wgate is not None else None
        self.norm_weight = torch.nn.Parameter(norm_weight)
        self.reset()

    def reset(self):
        self.partial_kv = None
        self.partial_score = None
        self._length = 0
        self._batch = None

    def forward(self, x, start_pos=0):
        if (
            x.ndim != 3
            or x.shape[-1] != self.dim
            or x.dtype != torch.bfloat16
            or x.device != self.wkv.device
        ):
            raise ValueError("expected BF16 compressor input[B,L,hidden] on the source GPU")
        batch, length, _ = x.shape
        if batch == 0 or length == 0:
            raise ValueError("compressor requests must be nonempty")
        if type(start_pos) is not int or start_pos < 0:
            raise ValueError("start_pos must be a nonnegative integer")
        if start_pos == 0:
            self.reset()
            self._batch = batch
        if start_pos != self._length or batch != self._batch or (start_pos > 0 and length != 1):
            raise ValueError("decode requires one token at a contiguous position and fixed batch")
        if self.ratio == 1:
            out = rmsnorm(F.linear(x, self.wkv), self.norm_weight, eps=self.eps)
        else:
            kv = F.linear(x.float(), self.wkv)
            score = F.linear(x.float(), self.wgate)
            if self.partial_kv is not None:
                kv = torch.cat((self.partial_kv, kv), dim=1)
                score = torch.cat((self.partial_score, score), dim=1)
            groups = kv.shape[1] // 2
            cutoff = groups * 2
            # Keep real projected tokens attached to their training graph.
            self.partial_kv = kv[:, cutoff:].clone() if cutoff < kv.shape[1] else None
            self.partial_score = score[:, cutoff:].clone() if cutoff < score.shape[1] else None
            out = None
            if groups:
                pooled = compressor_pool(
                    kv[:, :cutoff].reshape(batch, groups, 2, self.head_dim),
                    score[:, :cutoff].reshape(batch, groups, 2, self.head_dim),
                )
                out = rmsnorm(pooled.bfloat16(), self.norm_weight, eps=self.eps)
        self._length = start_pos + length
        return out
