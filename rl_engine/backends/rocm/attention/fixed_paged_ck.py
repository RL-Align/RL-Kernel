# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RL-Kernel-owned, fixed-schedule CK paged attention implementation."""

from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=2)
def load_fixed_paged_ck(tile_m: int = 128):
    from torch.utils.cpp_extension import load

    if tile_m not in (64, 128):
        raise ValueError("fixed CK tile must be 64 or 128")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm fixed CK attention before HIP Graph capture")
    spec = importlib.util.find_spec("aiter_meta")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("fixed CK attention requires the installed aiter_meta headers")
    meta = Path(next(iter(spec.submodule_search_locations)))
    ck = meta / "3rdparty/composable_kernel"
    source = Path(__file__).resolve().parents[4] / "csrc/rocm/attention/fixed_paged_ck.hip"
    fingerprint = hashlib.sha256(source.read_bytes())
    for header in (
        ck / "example/ck_tile/01_fmha/fmha_fwd.hpp",
        ck
        / "include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp",
        ck / "include/ck_tile/core/tensor/tile_scatter_gather.hpp",
    ):
        fingerprint.update(header.read_bytes())
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    flags = [
        # In an all-masked KV tile, m_old == m must give an exact rescale of
        # one. Contracting scale*m_old - rounded(scale*m) into FMA breaks that
        # identity and makes results depend on the surrounding query chunk.
        # Explicit CK MFMA instructions are unaffected by this scalar flag.
        "-O3",
        "-std=c++20",
        "-ffp-contract=off",
        f"--offload-arch={arch}",
        f"-DRLK_CK_TILE_M={tile_m}",
        "-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2",
        "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
        "-DCK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT=0",
        "-DCK_TILE_ATTENTION_USE_SOFTSIGN_ASM=1",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-fbracket-depth=1024",
        "-fgpu-flush-denormals-to-zero",
        "-fno-offload-uniform-block",
        "-mllvm",
        "--amdgpu-kernarg-preload-count=16",
        "-mllvm",
        "--lsr-drop-solution=1",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-mllvm",
        "-enable-post-misched=0",
        "-fno-gpu-rdc",
    ]
    fingerprint.update(" ".join(flags).encode())
    return load(
        name=f"rlk_fixed_paged_ck_m{tile_m}_{fingerprint.hexdigest()[:16]}",
        sources=[str(source)],
        extra_include_paths=[
            str(ck / "include"),
            str(ck / "library/include"),
            str(ck / "example/ck_tile/01_fmha"),
            str(meta / "3rdparty/ck_helper"),
        ],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=flags,
        with_cuda=True,
    )


def fixed_paged_prefill(
    q,
    k,
    v,
    cuq,
    indptr,
    flat_pages,
    maxq,
    maxk,
    dropout,
    scale,
    softcap,
    zero_tensors,
    causal,
    window_left,
    window_right,
    sink,
    return_lse,
    return_dropout,
    *,
    block_table,
    seqlen_k,
    out=None,
    tile_m=128,
):
    """Subset of the AITER entrypoint required by strict Qwen3 BF16 attention."""
    if (
        q.dtype != torch.bfloat16
        or q.shape[-1] != 128
        or k.shape[1] != 16
        or dropout
        or softcap
        or sink
        or return_dropout
        or zero_tensors
        or window_left != -1
        or window_right != -1
    ):
        raise ValueError(
            "fixed CK entrypoint requires BF16/D128/page16 without dropout, bias or windows"
        )
    tensors = (q, k, v, cuq, block_table, seqlen_k)
    if any(t.device != q.device for t in tensors) or not q.is_cuda:
        raise ValueError("fixed CK inputs must share one ROCm device")
    if k.dtype != q.dtype or v.dtype != q.dtype or v.shape != k.shape:
        raise ValueError("fixed CK K/V must match Q dtype and each other")
    if not q.is_contiguous() or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("fixed CK requires packed Q and contiguous K/V head dimensions")
    batch = block_table.shape[0]
    if (
        block_table.ndim != 2
        or block_table.stride(-1) != 1
        or cuq.shape != (batch + 1,)
        or seqlen_k.shape != (batch,)
        or any(t.dtype != torch.int32 for t in (cuq, block_table, seqlen_k))
        or not cuq.is_contiguous()
        or not seqlen_k.is_contiguous()
    ):
        raise ValueError("fixed CK requires packed int32 query/length metadata and a 2D page table")
    module = load_fixed_paged_ck(tile_m)
    guard_columns = (-block_table.shape[1]) % 8
    if guard_columns:
        block_table = torch.nn.functional.pad(block_table, (0, guard_columns), value=0)
    if out is None:
        out = torch.empty_like(q)
    elif (
        out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError("fixed CK output must match packed Q")
    lse = torch.empty(
        (q.shape[1], q.shape[0]) if return_lse else (0,), dtype=torch.float32, device=q.device
    )
    module.forward(
        [
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            out.data_ptr(),
            lse.data_ptr(),
            cuq.data_ptr(),
            block_table.data_ptr(),
            seqlen_k.data_ptr(),
            torch.cuda.current_stream(q.device).cuda_stream,
        ],
        [
            q.shape[0],
            block_table.shape[0],
            maxq,
            q.shape[1],
            k.shape[2],
            k.shape[0],
            block_table.stride(0),
            q.stride(0),
            k.stride(1),
            v.stride(1),
            k.stride(2),
            v.stride(2),
            k.stride(0),
            v.stride(0),
            q.stride(1),
            0,
        ],
        scale,
        causal,
        return_lse,
    )
    # Dropout is disabled; AITER backward accepts the unused, correctly typed
    # state buffer just as for the native forward entrypoint.
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, lse, torch.empty((0,), dtype=q.dtype, device=q.device), rng_state
