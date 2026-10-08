"""Compile the same header consumed by CUDA; never substitute NumPy/libm exp/log."""

import ctypes
import hashlib
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .contract import P3Error, P3Verdict

NATIVE = Path(__file__).with_name("native")
HOST_FLAGS = ("-std=c++17", "-O2", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math")


def source_fingerprint():
    return hashlib.sha256(
        b"".join((NATIVE / x).read_bytes() for x in ("bitmath.h", "bitmath_host.cpp"))
    ).hexdigest()


@lru_cache(maxsize=1)
def _library():
    # Per-process temporary build avoids shared-cache binary substitution and races.
    directory = tempfile.TemporaryDirectory(prefix="p3-bitmath-")
    out = Path(directory.name) / "bitmath.so"
    subprocess.run(
        ["c++", *HOST_FLAGS, str(NATIVE / "bitmath_host.cpp"), "-o", str(out)],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(out))
    lib._directory = directory
    lib.p3_math.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return lib


def apply(values, operation):
    x = np.ascontiguousarray(values, dtype=np.float32)
    out = np.empty_like(x)
    lib = _library()
    if lib.p3_host_environment() != 1:
        raise P3Error(
            P3Verdict.UNSUPPORTED_CAPABILITY,
            "bitmath requires round-to-nearest and gradual underflow",
        )
    lib.p3_math(
        x.ctypes.data,
        out.ctypes.data,
        x.size,
        ("exp", "log1p", "softplus", "sigmoid", "sqrt", "bf16").index(operation),
    )
    return out


def sum6(a):
    a = np.asarray(a, dtype=np.float32)
    return ((a[..., 0] + a[..., 1]) + (a[..., 2] + a[..., 3])) + (a[..., 4] + a[..., 5])
