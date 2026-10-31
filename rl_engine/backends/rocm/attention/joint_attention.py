# SPDX-License-Identifier: Apache-2.0
"""Launch the unmodified pinned aiter training forward for *every* runtime mode."""

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys

import torch

from rl_engine.contracts.operators.joint_attention import JOINT_PROFILE, require

KERNEL_SHA256 = "56db3b46c58f8fb30912d4564189b214e2c9a37fa2138a155b45a4ce51b5c844"
KERNEL_COMMIT = "326cffa38c9eda16bef19375ba55334e4d3acd59"


def load_pinned_kernel(path):
    path = Path(path).resolve()
    require(path.is_file(), "set RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE to the pinned upstream file")
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == KERNEL_SHA256,
        "aiter sparse source hash mismatch; no fallback or runtime download",
    )
    name = "_rlk_aiter_sparse_" + KERNEL_SHA256
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


class AiterJointAttentionBackend:
    profile = JOINT_PROFILE
    device_type = "cuda"

    def __init__(self, *, kernel_source=None):
        require(torch.version.hip is not None and torch.cuda.is_available(), "ROCm is required")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        require(arch == "gfx942", "only the validated MI300X/gfx942 profile is enabled")
        source = kernel_source or os.environ.get("RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE", "")
        require(bool(source), "pinned aiter sparse kernel source is required")
        self.module = load_pinned_kernel(source)
        self.triton = importlib.import_module("triton")
        cache = importlib.import_module("vllm.models.deepseek_v4.common.ops.cache_utils")
        self._gather_cache = cache.dequantize_and_gather_k_cache_triton
        self.provenance = {
            "profile": self.profile,
            "arch": arch,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": self.triton.__version__,
            "aiter_commit": KERNEL_COMMIT,
            "kernel_sha256": KERNEL_SHA256,
            "cache_source_sha256": hashlib.sha256(Path(cache.__file__).read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def forward_into(self, q, kv, sink, indices, out, lse, contract):
        m, h, d = q.shape
        if m == 0:
            return
        # Bypass upstream environment/config lookup, not the upstream kernel.
        # One CTA owns a query/head-group; no split-K softmax merge.
        self.module._sparse_mla_fwd_kernel[(m, self.triton.cdiv(h, 32))](
            q,
            kv,
            indices,
            sink,
            out,
            lse,
            q.stride(0),
            q.stride(1),
            kv.stride(0),
            out.stride(0),
            out.stride(1),
            lse.stride(0),
            indices.stride(0),
            h,
            d,
            kv.shape[0],
            indices.shape[1],
            contract.softmax_scale,
            HAS_ATTN_SINK=True,
            BLOCK_D=self.triton.next_power_of_2(d),
            BLOCK_H=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )

    def gather_packed_pages(self, cache, block_table, block_size, out, lengths, use_fnuz):
        self._gather_cache(out, cache, lengths, None, block_table, block_size, 0, use_fnuz=use_fnuz)
