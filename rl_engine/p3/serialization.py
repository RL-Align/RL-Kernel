"""Versioned, typed little-endian serialization; floats retain raw FP32 bits."""

import hashlib
import struct

import numpy as np


def canonical(value):
    if value is None:
        return b"n"
    if isinstance(value, (bool, np.bool_)):
        return b"b" + bytes([bool(value)])
    if isinstance(value, (int, np.integer)):
        if int(value) >= 2**63:
            return b"u" + struct.pack("<Q", int(value))
        return b"i" + struct.pack("<q", int(value))
    if isinstance(value, (float, np.floating)):
        return b"f" + struct.pack("<f", value)
    if isinstance(value, str):
        data = value.encode("utf-8")
        return b"s" + struct.pack("<I", len(data)) + data
    if isinstance(value, bytes):
        return b"r" + struct.pack("<I", len(value)) + value
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
        return b"t" + canonical(a.dtype.str) + canonical(list(a.shape)) + canonical(a.tobytes())
    if isinstance(value, dict):
        # Ordered field maps are intentional: callers must use contract schema order.
        return (
            b"d"
            + struct.pack("<I", len(value))
            + b"".join(canonical(k) + canonical(v) for k, v in value.items())
        )
    if isinstance(value, (list, tuple)):
        return b"l" + struct.pack("<I", len(value)) + b"".join(map(canonical, value))
    raise TypeError(type(value))


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def encode(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
        return {
            "__tensor__": True,
            "dtype": a.dtype.str,
            "shape": list(a.shape),
            "bytes": a.tobytes().hex(),
        }
    if isinstance(value, (float, np.floating)):
        return {"__f32__": struct.pack("<f", value).hex()}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode(value):
    if isinstance(value, dict):
        if "__tensor__" in value:
            return (
                np.frombuffer(bytes.fromhex(value["bytes"]), dtype=value["dtype"])
                .reshape(value["shape"])
                .copy()
            )
        if "__f32__" in value:
            return np.frombuffer(bytes.fromhex(value["__f32__"]), dtype="<f4")[0]
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value
