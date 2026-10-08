"""Synchronous p3-op-abi.v4 CUDA Stable Top-6, with actual status/echo readback."""

import ctypes
import hashlib
import struct
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .bitmath import NATIVE
from .contract import (
    PROVENANCE_SCHEMA,
    TIE_POLICY,
    TOPK_ABI,
    UNSET,
    E,
    P3Error,
    P3OpResult,
    P3Verdict,
)
from .oracle import require
from .provider import readback_verdict

CUDA_FLAGS = (
    "-std=c++17",
    "-O2",
    "-shared",
    "-Xcompiler",
    "-fPIC",
    "-arch=sm_90",
    "--ftz=false",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)


def cuda_source_fingerprint():
    return hashlib.sha256(
        b"".join(
            (NATIVE / name).read_bytes()
            for name in ("bitmath.h", "stable_topk6.cuh", "stable_topk6.cu")
        )
    ).hexdigest()


@lru_cache(maxsize=1)
def cuda_library():
    directory = tempfile.TemporaryDirectory(prefix="p3-cuda-")
    out = Path(directory.name) / "p3.so"
    sha = cuda_source_fingerprint()
    subprocess.run(
        [
            "nvcc",
            *CUDA_FLAGS,
            f'-DP3_SOURCE_SHA="{sha}"',
            str(NATIVE / "stable_topk6.cu"),
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(out))
    lib._directory = directory
    lib.p3_launch_topk.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.p3_launch_math.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.p3_source_sha.restype = ctypes.c_char_p
    lib.p3_build_flags.restype = ctypes.c_char_p
    return lib


class CudaTopKProvider:
    def stable_topk6_fwd(self, ctx, q, *, block_size=32):
        try:
            require(
                isinstance(q, torch.Tensor)
                and q.dtype == torch.float32
                and q.ndim == 2
                and q.shape[1] == E
                and q.is_contiguous(),
                P3Verdict.SCHEMA_MISMATCH,
                "q must be contiguous FP32[T,256]",
            )
            require(
                isinstance(ctx.row_active, np.ndarray)
                and ctx.row_active.dtype == np.bool_
                and ctx.row_active.shape == (len(q),),
                P3Verdict.SCHEMA_MISMATCH,
                "row_active schema",
            )
            if not ctx.row_active.any():
                return P3OpResult(P3Verdict.ZERO_ACTIVE_TOKENS)
            require(
                ctx.backend_tag == "cuda"
                and q.is_cuda
                and torch.version.hip is None
                and torch.cuda.is_available(),
                P3Verdict.UNSUPPORTED_CAPABILITY,
                "CUDA required",
            )
            require(
                torch.cuda.get_device_capability(q.device) == (9, 0)
                and "H100" in torch.cuda.get_device_name(q.device),
                P3Verdict.UNSUPPORTED_CAPABILITY,
                "reference profile requires SM90 H100",
            )
            require(
                block_size in (1, 32, 64, 128),
                P3Verdict.UNSUPPORTED_CAPABILITY,
                "unsupported launch block",
            )
            require(
                ctx.allocator is not None, P3Verdict.CORRUPT_ARTIFACT, "durable allocator required"
            )
            require(
                ctx.allocator.scope == [ctx.run_id, ctx.engine_id, ctx.rank],
                P3Verdict.IDENTITY_DRIFT,
                "allocator context scope mismatch",
            )
            lib = cuda_library()
            require(
                lib.p3_source_sha().decode() == cuda_source_fingerprint()
                and lib.p3_binary_arch() == 90,
                P3Verdict.UNSUPPORTED_CAPABILITY,
                "actual binary/source arch mismatch",
            )
            flags = lib.p3_build_flags().decode()
            stream = ctx.stream or torch.cuda.current_stream(q.device)
            require(
                stream.device == q.device,
                P3Verdict.UNSUPPORTED_CAPABILITY,
                "stream device mismatch",
            )
            with torch.cuda.device(q.device), torch.cuda.stream(stream):
                safe_q = q.clone()
                if not ctx.row_active.all():
                    safe_q[torch.as_tensor(~ctx.row_active, device=q.device)] = 0
                ids = torch.empty((len(q), 6), dtype=torch.int32, device=q.device)
                initial = np.frombuffer(struct.pack("<iiQ", UNSET, 0, 0), dtype=np.uint8).copy()
                status = torch.as_tensor(initial, device=q.device)
                host_status = torch.empty(16, dtype=torch.uint8, pin_memory=True)
                ctx.invocation_id = ctx.allocator.reserve()  # Durable before any kernel launch.
                ctx.status_record = status
                launch = lib.p3_launch_topk(
                    safe_q.data_ptr(),
                    ids.data_ptr(),
                    len(q),
                    status.data_ptr(),
                    ctx.invocation_id,
                    stream.cuda_stream,
                    block_size,
                )
                host_status.copy_(status, non_blocking=True)  # D2H on the same stream.
                stream.synchronize()
                verdict = readback_verdict(
                    host_status.numpy().tobytes(), ctx.invocation_id, launched=launch == 0
                )
            provenance = {
                "schema_version": PROVENANCE_SCHEMA,
                "requested_backend": ctx.backend_tag,
                "actual_backend": "cuda",
                "backend_profile": "cuda.sm90.h100.v1",
                "kernel": "p3_topk_kernel",
                "source_fingerprint": cuda_source_fingerprint(),
                "build_flags": flags,
                "target_arch": "sm90",
                "actual_binary_arch": 90,
                "fast_math": False,
                "fallback_reason": None,
                "device": str(q.device),
                "gpu": torch.cuda.get_device_name(q.device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": str(q.dtype),
                "shape": list(q.shape),
                "stride": list(q.stride()),
                "event": "same_stream_D2H_then_synchronize",
                "block": block_size,
                "grid": (len(q) + block_size - 1) // block_size,
                "topk_abi": TOPK_ABI,
                "tie_policy": TIE_POLICY,
                "stream": stream.cuda_stream,
                "invocation_id": ctx.invocation_id,
                "run_id": ctx.run_id,
                "engine_id": ctx.engine_id,
                "rank": ctx.rank,
                "attempt_id": ctx.allocator.attempt_id,
                "status_readback": host_status.numpy().tobytes().hex(),
                "capabilities": {
                    "int32_atomicMin": True,
                    "uint64_atomicExch": True,
                    "device_fence": True,
                },
            }
            return P3OpResult(
                verdict, {"ids": ids} if verdict == P3Verdict.PASS else None, provenance
            )
        except P3Error as exc:
            return P3OpResult(exc.verdict)
        except (OSError, subprocess.CalledProcessError):
            return P3OpResult(P3Verdict.UNSUPPORTED_CAPABILITY)
        except RuntimeError:
            return P3OpResult(P3Verdict.INCOMPLETE_ARTIFACT)


def device_bitmath(values, operation):
    """Diagnostic host/device parity slice using the exact shared header."""
    require(
        values.is_cuda and values.dtype == torch.float32 and values.is_contiguous(),
        P3Verdict.SCHEMA_MISMATCH,
        "CUDA contiguous FP32 required",
    )
    lib = cuda_library()
    out = torch.empty_like(values)
    with torch.cuda.device(values.device):
        stream = torch.cuda.current_stream(values.device)
        error = lib.p3_launch_math(
            values.data_ptr(),
            out.data_ptr(),
            values.numel(),
            ("exp", "log1p", "softplus", "sigmoid", "sqrt", "bf16").index(operation),
            stream.cuda_stream,
        )
        stream.synchronize()
    require(error == 0, P3Verdict.INCOMPLETE_ARTIFACT, "math diagnostic launch failed")
    return out
