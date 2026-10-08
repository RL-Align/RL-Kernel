"""Bit-defined CPU oracle. It produces raw intermediates, never CUDA certification."""

import numpy as np

from . import bitmath
from .contract import EPSILON, ROUND_POLICIES, SCALE, E, K, P3Error, P3Verdict


def require(condition, verdict, detail):
    if not condition:
        raise P3Error(verdict, detail)


def tensor(x, dtype, shape):
    require(
        isinstance(x, np.ndarray) and x.dtype == np.dtype(dtype) and x.shape == shape,
        P3Verdict.SCHEMA_MISMATCH,
        f"expected {dtype}{shape}",
    )


def finite(x, verdict=P3Verdict.NON_FINITE):
    require(np.isfinite(x).all(), verdict, "non-finite active tensor")


def stable_topk6(q):
    require(
        isinstance(q, np.ndarray) and q.dtype == np.float32 and q.ndim == 2,
        P3Verdict.SCHEMA_MISMATCH,
        "q must be FP32[T,256]",
    )
    require(
        q.shape[1] == E,
        P3Verdict.FORBIDDEN_LOCAL_SHARD_TOPK,
        "Top-6 requires all 256 logical experts",
    )
    finite(q)
    # Stable sorting of the original ascending expert axis implements the total order,
    # including +/-zero equality. T09 must independently cross-check the golden.
    return np.argsort(-q, axis=1, kind="stable")[:, :K].astype(np.int32)


def router_sqrt_softplus_fwd(z, logit_round_point):
    tensor(z, "float32", (len(z), E))
    require(
        logit_round_point in ROUND_POLICIES,
        P3Verdict.LOGIT_ROUND_POINT_MISMATCH,
        "unknown round policy",
    )
    finite(z, P3Verdict.UPSTREAM_NON_FINITE)
    zp = z.copy() if logit_round_point == "fp32_direct" else bitmath.apply(z, "bf16")
    s = bitmath.apply(bitmath.apply(zp, "softplus"), "sqrt")
    finite(zp)
    finite(s)
    return {"s": s, "saved_score": {"z_prime": zp, "s": s.copy()}}


def normalize(ids, s):
    a = np.take_along_axis(s, ids, axis=1)
    Z = bitmath.sum6(a)[:, None] + EPSILON
    p = a / Z
    w = p * SCALE
    for x in (a, Z, p, w):
        finite(x)
    return {
        "ids": ids.copy(),
        "weights": w,
        "saved_route": {"ids": ids.copy(), "a": a, "Z": Z, "p": p},
    }


def hash_route_fwd(input_token_id, s, tid2eid):
    tensor(s, "float32", (len(s), E))
    tensor(input_token_id, "int64", (len(s),))
    require(
        tid2eid.dtype == np.int32
        and tid2eid.ndim == 2
        and tid2eid.shape[1] == K
        and len(tid2eid) > 0
        and ((tid2eid >= 0) & (tid2eid < E)).all(),
        P3Verdict.HASH_TABLE_MISMATCH,
        "invalid table or sentinel",
    )
    require(
        ((input_token_id >= 0) & (input_token_id < len(tid2eid))).all(),
        P3Verdict.HASH_TABLE_INDEX_OUT_OF_RANGE,
        "table index out of range",
    )
    finite(s)
    return normalize(tid2eid[input_token_id], s)


def learned_route_fwd(s, b):
    tensor(s, "float32", (len(s), E))
    tensor(b, "float32", (E,))
    finite(s)
    finite(b)
    q = s + b  # Independent post-bias buffer; weights always gather pre-bias s.
    finite(q)
    return {**normalize(stable_topk6(q), s), "q": q}


def route_bwd(dweights, saved):
    ids, Z, p = saved["ids"], saved["Z"], saved["p"]
    tensor(dweights, "float32", (len(ids), K))
    tensor(ids, "int32", (len(ids), K))
    require(
        ((ids >= 0) & (ids < E)).all(), P3Verdict.SCHEMA_MISMATCH, "invalid saved logical expert"
    )
    for value in (saved["a"], Z, p):
        finite(value)
    require((Z > 0).all(), P3Verdict.NON_FINITE, "invalid saved normalization denominator")
    finite(dweights, P3Verdict.UPSTREAM_NON_FINITE)
    c = bitmath.sum6(dweights * p)[:, None]
    da = (SCALE / Z) * (dweights - c)
    finite(da)
    ds = np.zeros((len(ids), E), dtype=np.float32)
    for slot in range(K):  # Duplicate experts accumulate strictly in slot 0 -> 5 order.
        for row in range(len(ids)):
            ds[row, ids[row, slot]] += da[row, slot]
    finite(ds)
    return {"ds": ds, "da": da, "c": c}


def router_sqrt_softplus_bwd(ds, saved):
    z, s = saved["z_prime"], saved["s"]
    tensor(ds, "float32", z.shape)
    nz = ds != np.float32(0)
    finite(ds[nz])
    finite(z[nz])
    finite(s[nz])
    require((s[nz] > 0).all(), P3Verdict.NON_FINITE, "selected score underflow")
    dz = np.zeros_like(ds)  # Includes canonical positive zero for signed-zero ds.
    dsp = np.where(z[nz] > np.float32(20), np.float32(1), bitmath.apply(z[nz], "sigmoid"))
    dz[nz] = (ds[nz] * dsp) / (np.float32(2) * s[nz])
    finite(dz)
    return {"dz": dz}
