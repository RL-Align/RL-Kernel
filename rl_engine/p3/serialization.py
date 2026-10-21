"""Versioned, typed little-endian serialization; floats retain raw FP32 bits."""

import hashlib
import struct
from functools import wraps

import numpy as np

from .contract import FIELD_TYPES, P3Error, P3Verdict, schema_fields


def checked_structure(fn):
    @wraps(fn)
    def checked(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except P3Error:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
            raise P3Error(P3Verdict.SCHEMA_MISMATCH, f"malformed {fn.__name__} input") from exc

    return checked


def ordered_record(version, values):
    """Project and type-check records using the sole contract field definitions."""
    fields = schema_fields(version)
    if set(values) != set(fields):
        raise P3Error(P3Verdict.SCHEMA_MISMATCH, "record field set mismatch")
    result = {key: values[key] for key in fields}
    for key, value in result.items():
        kind = FIELD_TYPES.get(key, "str")
        integer = isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        valid = {
            "str": isinstance(value, str),
            "int": integer and -(2**63) <= int(value) < 2**63,
            "bool": isinstance(value, (bool, np.bool_)),
            "f32": type(value) in (float, np.float32),
            "dict": isinstance(value, dict),
            "list": isinstance(value, list),
            "int_or_str": isinstance(value, str) or (integer and 0 <= int(value) < 2**64),
        }[kind]
        if not valid:
            raise P3Error(P3Verdict.SCHEMA_MISMATCH, f"invalid {key} field type")
        if kind == "f32" and np.isfinite(value) and float(np.float32(value)) != float(value):
            raise P3Error(P3Verdict.SCHEMA_MISMATCH, f"{key} must retain FP32 precision")
    return result


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
