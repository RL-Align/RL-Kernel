# SPDX-License-Identifier: Apache-2.0
"""INK-59: aligned selected-token FP32 bit export for CUDA and ROCm."""

import hashlib

import torch


def export_selected_logprob(
    logprob: torch.Tensor,
    token_keys: torch.Tensor,
    token_ids: torch.Tensor,
    routes: torch.Tensor,
    *,
    weight_version: str,
    weights_sha256: str,
    input_sha256: str,
    platform: str,
    runtime_revision: str,
    engine: str,
    kernel_id: str,
    visit_trace: dict | None = None,
    fixture: bool = False,
) -> dict:
    """Export actual selected-token values and 48-visit ordered expert IDs.

    All tensors are already aligned by the caller's teacher-forced token rows.
    Capture is synchronous, diagnostic only. CPU is accepted only for fixtures.
    """
    if platform not in ("cuda", "rocm") or engine not in ("miles", "sglang"):
        raise ValueError("invalid platform or engine")
    if not all((weight_version, runtime_revision, kernel_id)):
        raise ValueError("missing version or kernel identity")
    for digest in (weights_sha256, input_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid SHA256 provenance")
    hardware = "fixture"
    if not fixture:
        actual = "rocm" if torch.version.hip is not None else "cuda"
        if not logprob.is_cuda or actual != platform:
            raise RuntimeError("real export requires matching accelerator tensors")
        props = torch.cuda.get_device_properties(logprob.device)
        hardware = (
            props.gcnArchName.split(":")[0] if actual == "rocm" else f"sm{props.major}{props.minor}"
        )
        if actual == "rocm" and hardware != "gfx950":
            raise RuntimeError("initial ROCm export requires gfx950")
    if logprob.ndim != 1 or not logprob.numel() or logprob.dtype != torch.float32:
        raise ValueError("expected nonempty FP32 logprob vector")
    n = logprob.numel()
    if token_keys.shape != (n,) or token_ids.shape != (n,) or routes.shape != (n, 48, 8):
        raise ValueError("misaligned token keys, IDs or 48-visit routes")
    if any(t.dtype not in (torch.int32, torch.int64) or t.device != logprob.device
           for t in (token_keys, token_ids, routes)):
        raise ValueError("identity tensors must be integers on the logprob device")
    keys, order = token_keys.long().sort()
    if bool((keys[1:] == keys[:-1]).any()):
        raise ValueError("duplicate logical token keys")
    if not bool(torch.isfinite(logprob).all()):
        raise ValueError("nonfinite logprob")
    if bool(((token_ids < 0) | (token_ids >= 200058)).any()):
        raise ValueError("invalid vocabulary ID")
    if bool(((routes < 0) | (routes >= 128)).any()):
        raise ValueError("invalid route expert ID")
    sorted_routes = routes.sort(-1).values
    if bool((sorted_routes[..., 1:] == sorted_routes[..., :-1]).any()):
        raise ValueError("duplicate route expert ID")
    ordered = routes[order].to(torch.int32).cpu().contiguous()
    # One hash per logical visit retains order and localizes routing drift.
    route_hashes = [hashlib.sha256(ordered[:, j].contiguous().numpy().tobytes()).hexdigest()
                    for j in range(48)]
    bits = logprob[order].detach().cpu().contiguous().view(torch.int32).tolist()
    visit_hashes = None
    if visit_trace is not None:
        meta = visit_trace["metadata"]
        if meta.get("weights_sha256") != weights_sha256 or meta.get("input_sha256") != input_sha256:
            raise ValueError("trace and logprob provenance disagree")
        rows = [r for r in visit_trace["records"] if r["direction"] == "forward"
                and r["role"] == "partial"]
        if sorted(r["logical"] for r in rows) != list(range(48)):
            raise ValueError("incomplete 48-visit forward trace")
        visit_hashes = [r["sha256"] for r in sorted(rows, key=lambda r: r["logical"])]
        if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
               for h in visit_hashes):
            raise ValueError("invalid visit hash")
    return {
        "schema": "inkling2-selected-logp-v1", "fixture": fixture,
        "platform": platform, "hardware": hardware, "engine": engine, "runtime_revision": runtime_revision,
        "kernel_id": kernel_id, "weight_version": weight_version,
        "weights_sha256": weights_sha256, "input_sha256": input_sha256,
        "token_keys": keys.cpu().tolist(), "token_ids": token_ids[order].cpu().tolist(),
        "route_hashes": route_hashes, "visit_hashes": visit_hashes, "logprob_bits": bits,
    }


def compare_exports(train: dict, rollout: dict) -> dict:
    """Compare ordered routes and exact logprob bits; reject incompatible inputs."""
    identity = ("schema", "fixture", "platform", "hardware", "weight_version", "weights_sha256",
                "input_sha256", "token_keys", "token_ids")
    if train.get("schema") != "inkling2-selected-logp-v1" or any(
        train[key] != rollout[key] for key in identity
    ):
        raise ValueError("incompatible logprob exports")
    if train["engine"] != "miles" or rollout["engine"] != "sglang":
        raise ValueError("expected Miles actor and SGLang rollout")
    n = len(train["token_keys"])
    for report in (train, rollout):
        if n == 0 or len(report["logprob_bits"]) != n or len(report["route_hashes"]) != 48:
            raise ValueError("incomplete export")
        for bit in report["logprob_bits"]:
            if type(bit) is not int or not -(2**31) <= bit < 2**31 or (
                bit & 0x7F800000
            ) == 0x7F800000:
                raise ValueError("invalid FP32 bit pattern")
    mismatches = [key for key, a, b in zip(train["token_keys"], train["logprob_bits"],
                                          rollout["logprob_bits"], strict=True) if a != b]
    drift = [j for j, (a, b) in enumerate(zip(train["route_hashes"],
                                             rollout["route_hashes"], strict=True)) if a != b]
    a_visits, b_visits = train["visit_hashes"], rollout["visit_hashes"]
    if (a_visits is None) != (b_visits is None):
        raise ValueError("trace availability differs")
    if a_visits is not None and (len(a_visits) != 48 or len(b_visits) != 48):
        raise ValueError("incomplete visit hashes")
    boundary = [] if a_visits is None else [
        j for j, (a, b) in enumerate(zip(a_visits, b_visits, strict=True)) if a != b
    ]
    return {"token_count": n, "bit_mismatch_count": len(mismatches),
            "first_logical_token": mismatches[0] if mismatches else None,
            "first_route_visit": drift[0] if drift else None,
            "first_boundary_visit": boundary[0] if boundary else None,
            "trace_available": a_visits is not None,
            "equal": not mismatches and not drift and not boundary,
            "fixture": train["fixture"], "ws1_certified": False}
