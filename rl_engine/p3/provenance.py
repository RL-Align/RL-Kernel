"""Versioned actual execution metadata shared by references and validators."""

import numpy as np

from . import bitmath
from .contract import PROVENANCE_SCHEMA, REDUCTION_TREE, TIE_POLICY
from .serialization import fingerprint


def tensor_metadata(value):
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "stride": list(value.strides),
            "device": "cpu",
        }
    if isinstance(value, dict):
        return {k: tensor_metadata(v) for k, v in value.items()}
    if hasattr(value, "payload"):
        return tensor_metadata(value.payload)
    return {"scalar": str(value)}


def cpu_provenance(inputs=(), *, round_policy="operator-input-or-sealed-forward"):
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "requested_backend": "recorded-cpu",
        "actual_backend": "recorded-cpu",
        "backend_profile": "synthetic-cpu.v1",
        "fast_math": False,
        "fallback_reason": None,
        "device": "cpu",
        "certifies_cuda": False,
        "kernel": "recorded-cpu.oracle.v1",
        "target_arch": "host",
        "source_fingerprint": bitmath.source_fingerprint(),
        "build_flags": list(bitmath.HOST_FLAGS),
        "math_path": "p3-bitmath.v1",
        "reduction_tree_id": REDUCTION_TREE,
        "tie_policy": TIE_POLICY,
        "round_policy": round_policy,
        "stream": "synchronous",
        "event": "completed-host-call",
        "launch": {"block": None, "warps": None, "stages": None},
        "input_tensors": [tensor_metadata(v) for v in inputs],
        "inputs_fingerprint": fingerprint(
            [v.payload if hasattr(v, "payload") else v for v in inputs]
        ),
    }
