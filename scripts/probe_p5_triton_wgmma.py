#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SM90 gate for the experimental P5-4 Triton WGMMA kernels.

Run on an H100/SM90 host after installing this repository and Triton:
    python scripts/probe_p5_triton_wgmma.py
    compute-sanitizer --tool racecheck python scripts/probe_p5_triton_wgmma.py

This checks the actual generated PTX and compares real GPU results with the
serial FP32 oracle. It is deliberately separate from provider registration.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import shutil
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_engine.moe import oracle  # noqa: E402
from rl_engine.moe.mx_format import (  # noqa: E402
    MXTensor,
    e4m3_encode,
    pack_nibbles,
)
from rl_engine.moe import triton_grouped_gemm_wgmma as impl  # noqa: E402

REPORT = {
    "numeric_profile_candidate": "p5-wgmma-triton-v1",
    "status": "pending",
    "errors": [],
    "kernels": {},
}
ARTIFACTS = None


def metadata_reference(offsets, n, k, block_m, backward=False):
    """CPU expected values for the probe only; runtime prep stays in impl."""
    desc, prefix = [], [0]
    for expert, (begin, end) in enumerate(zip(offsets, offsets[1:], strict=False)):
        desc.append(
            [
                begin,
                end,
                end - begin,
                expert * n * (k // 2),
                expert * n * (k // 32),
                begin * (n if backward else k),
                0 if backward else begin * (k // 32),
                begin * (k if backward else n),
            ]
        )
        prefix.append(prefix[-1] + (end - begin + block_m - 1) // block_m)
    return desc, prefix


if impl.triton is not None:
    triton, tl = impl.triton, impl.tl

    @triton.jit
    def _minimal_fp8(A, B, C):
        r = tl.arange(0, 64)
        k = tl.arange(0, 32)
        a = tl.load(A + r[:, None] * 32 + k[None, :]).to(tl.float8e4nv, bitcast=True)
        b = tl.load(B + k[:, None] * 64 + r[None, :]).to(tl.float8e4nv, bitcast=True)
        tl.store(C + r[:, None] * 64 + r[None, :], tl.dot(a, b, max_num_imprecise_acc=32))


def _record_kernel(label, compiled):
    ptx = compiled.asm.get("ptx", "")
    if "wgmma.mma_async" not in ptx:
        raise AssertionError(f"{label}: expected WGMMA in compiled PTX")
    info = {
        "wgmma_ptx": True,
        "registers": getattr(compiled, "n_regs", None),
        "spills": getattr(compiled, "n_spills", None),
        "shared_bytes": getattr(compiled.metadata, "shared", None),
    }
    REPORT["kernels"][label] = info
    if ARTIFACTS is not None:
        (ARTIFACTS / f"{label}.ptx").write_text(ptx)
        cubin = compiled.asm.get("cubin")
        if cubin is not None:
            target = ARTIFACTS / f"{label}.cubin"
            target.write_bytes(cubin)
            disassembler = shutil.which("cuobjdump")
            if disassembler:
                sass = subprocess.run(
                    [disassembler, "--dump-sass", str(target)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                (ARTIFACTS / f"{label}.sass").write_text(sass)
                info["sass_hgmma"] = "HGMMA" in sass.upper()
                if not info["sass_hgmma"]:
                    raise AssertionError(f"{label}: no HGMMA in disassembled cubin")
            else:
                info["sass_status"] = "cuobjdump unavailable; inspect saved cubin on CUDA host"
    print(f"{label}: {info}")


def _error(label: str, got: torch.Tensor, want: torch.Tensor) -> None:
    if got.shape != want.shape or got.dtype != torch.float32:
        raise AssertionError(f"{label}: wrong output shape/dtype")
    if got.numel() == 0:
        return
    if not bool(torch.isfinite(got).all()) or not bool(torch.isfinite(want).all()):
        raise AssertionError(f"{label}: nonfinite output")
    delta = (got - want).abs()
    absolute = float(delta.max())
    normalized = float((delta / want.abs().clamp_min(1.0)).max())
    print(f"{label}: max_abs={absolute:.6g}, max_normalized={normalized:.6g}")
    REPORT["errors"].append(
        {"label": label, "max_abs": absolute, "max_normalized": normalized, "threshold": 2e-2}
    )
    if normalized > 2e-2:
        raise AssertionError(f"{label}: exceeded profile tolerance")


def _inputs(offsets: list[int], n: int, k: int, one_hot: bool):
    m, e = offsets[-1], len(offsets) - 1
    if one_hot:
        a_values = torch.zeros((m, k), dtype=torch.float32)
        dy = torch.zeros((m, n), dtype=torch.bfloat16)
        for row in range(m):
            # For one expert's 16 consecutive rows, 7*row walks all 16
            # nibble positions modulo 16 and alternates low/high bytes.
            a_values[row, (row * 7) % k] = 1.0
            if n:
                dy[row, (row * 11 + 2) % n] = 1.0
    else:
        generator = torch.Generator().manual_seed(m * 10000 + n * 100 + k)
        a_values = torch.randn((m, k), generator=generator)
        dy = torch.randn((m, n), generator=generator).to(torch.bfloat16)

    # Every nibble (including negative zero) appears; scale varies by row and
    # by 32-K block, exposing swapped nibble and scale-block indexing.
    nibbles = (
        (
            torch.arange(e)[:, None, None] * 3
            + torch.arange(n)[None, :, None] * 5
            + torch.arange(k)[None, None, :] * 7
        )
        .remainder(16)
        .to(torch.uint8)
    )
    as_codes = (124 + torch.arange(m)[:, None] % 3 + torch.arange(k // 32)[None, :] % 3).to(
        torch.uint8
    )
    ws_codes = (
        123
        + torch.arange(e)[:, None, None] % 3
        + torch.arange(n)[None, :, None] % 3
        + torch.arange(k // 32)[None, None, :] % 4
    ).to(torch.uint8)
    a = MXTensor(e4m3_encode(a_values).cuda(), as_codes.cuda(), "e4m3", (m, k))
    w = MXTensor(pack_nibbles(nibbles).cuda(), ws_codes.cuda(), "e2m1", (e, n, k))
    return a, w, dy.cuda(), torch.tensor(offsets, dtype=torch.int32, device="cuda")


def _ptx_probe(a: MXTensor, w: MXTensor, dy: torch.Tensor, offsets: torch.Tensor) -> None:
    m, k = a.shape
    _, n, _ = w.shape
    y = torch.empty((m, n), dtype=torch.float32, device="cuda")
    dx = torch.empty((m, k), dtype=torch.float32, device="cuda")
    desc, prefix = impl._prepare_metadata(offsets, n, k)
    fwd = impl._launch_fwd(a, w, desc, prefix, y, 1)
    desc_b, prefix_b = impl._prepare_metadata(offsets, n, k, backward=True)
    bwd = impl._launch_bwd(dy, w, desc_b, prefix_b, dx, 1)
    for label, compiled in (("forward", fwd), ("backward", bwd)):
        _record_kernel(label, compiled)
    torch.cuda.synchronize()
    _error("direct_persistent.Y", y, impl.grouped_gemm_fwd(a, w, offsets))
    _error("direct_persistent.dX", dx, impl.grouped_gemm_bwd(dy, w, offsets))


def _metadata_probe():
    cases = [[0], [0, 0, 0], [0, 0, 63, 127, 127, 192], [0, 10000, 10001, 10001, 10002]]
    cases += [[0, *itertools.accumulate(i % 5 for i in range(e))] for e in (1023, 1024, 1025, 2049)]
    # A nondefault stream catches accidental default-stream metadata launches.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for offsets in cases:
            gpu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
            e, m, q = len(offsets) - 1, offsets[-1], 3
            for backward in (False, True):
                desc, prefix = impl._prepare_metadata(gpu, 65, 96, backward=backward)
                want_desc, want_prefix = metadata_reference(offsets, 65, 96, impl.BM, backward)
                self_desc = torch.tensor(want_desc, dtype=torch.int64).reshape(e, impl.DESC_FIELDS)
                if not torch.equal(desc.cpu(), self_desc) or prefix.cpu().tolist() != want_prefix:
                    raise AssertionError(f"metadata mismatch E={e} backward={backward}")
                total = want_prefix[-1] * q
                for c in sorted({1, 3, max(1, total), total + 3}):
                    # Do not synchronize between prep and schedule. Re-prep also
                    # checks allocator reuse and stream ordering across repeats.
                    desc, prefix = impl._prepare_metadata(gpu, 65, 96, backward=backward)
                    ids = torch.full((m, q), -1, dtype=torch.int64, device="cuda")
                    impl._schedule_probe_kernel[(c,)](
                        desc,
                        prefix,
                        ids,
                        E=e,
                        Q=q,
                        BLOCK_M=impl.BM,
                    )
                    expected = torch.empty((m, q), dtype=torch.int64)
                    for expert, (begin, end) in enumerate(zip(offsets, offsets[1:], strict=False)):
                        for row in range(begin, end):
                            mt = want_prefix[expert] + (row - begin) // impl.BM
                            expected[row] = mt * q + torch.arange(q)
                    if not torch.equal(ids.cpu(), expected):
                        raise AssertionError(f"schedule mismatch E={e} C={c}")
        # Exercise more than one level of _inclusive_scan independently, without
        # allocating million-expert GEMM weights or descriptors.
        x = torch.arange(impl.SCAN_BLOCK**2 + 1, device="cuda", dtype=torch.int64) % 7
        if not torch.equal(impl._inclusive_scan(x), x.cumsum(0)):
            raise AssertionError("recursive inclusive scan mismatch")
    stream.synchronize()
    REPORT["metadata_schedule"] = "pass"


def _graph_probe():
    a, w, dy, offsets = _inputs([0, 1, 1, 70], 65, 96, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            impl.grouped_gemm_fwd(a, w, offsets, validate_contents=False, programs=2)
            impl.grouped_gemm_bwd(dy, w, offsets, validate_contents=False, programs=2)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = impl.grouped_gemm_fwd(a, w, offsets, validate_contents=False, programs=2)
        dx = impl.grouped_gemm_bwd(dy, w, offsets, validate_contents=False, programs=2)
    # Same storage/shapes, changing contents: a cached CPU-derived prefix fails.
    for values in ([0, 1, 1, 70], [0, 0, 69, 70], [0, 70, 70, 70]):
        offsets.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        graph.replay()
        _error("graph.Y", y, oracle.mxfp8_mxfp4_grouped_gemm_fwd(a, w, offsets))
        _error("graph.dX", dx, oracle.mxfp8_mxfp4_grouped_gemm_bwd(dy, w, offsets))
    REPORT["cuda_graph_dynamic_offsets"] = "pass"


def _case(label: str, offsets: list[int], n: int, k: int, one_hot: bool) -> None:
    a, w, dy, expert_offsets = _inputs(offsets, n, k, one_hot)
    codes_before, scales_before = w.codes.clone(), w.scales.clone()
    y = impl.grouped_gemm_fwd(a, w, expert_offsets)
    dx = impl.grouped_gemm_bwd(dy, w, expert_offsets)
    _error(f"{label}.Y", y, oracle.mxfp8_mxfp4_grouped_gemm_fwd(a, w, expert_offsets))
    _error(f"{label}.dX", dx, oracle.mxfp8_mxfp4_grouped_gemm_bwd(dy, w, expert_offsets))
    for programs in (1, 2, 132):
        trusted_y = impl.grouped_gemm_fwd(
            a, w, expert_offsets, validate_contents=False, programs=programs
        )
        trusted_dx = impl.grouped_gemm_bwd(
            dy, w, expert_offsets, validate_contents=False, programs=programs
        )
        if not torch.equal(y, trusted_y) or not torch.equal(dx, trusted_dx):
            raise AssertionError(f"{label}: checked/trusted or C={programs} changed results")
    if not torch.equal(y, impl.grouped_gemm_fwd(a, w, expert_offsets)):
        raise AssertionError(f"{label}: forward repeat changed")
    if not torch.equal(dx, impl.grouped_gemm_bwd(dy, w, expert_offsets)):
        raise AssertionError(f"{label}: backward repeat changed")
    if not torch.equal(codes_before, w.codes) or not torch.equal(scales_before, w.scales):
        raise AssertionError(f"{label}: packed weight was modified")
    if label == "one_hot_middle_empty":
        _ptx_probe(a, w, dy, expert_offsets)
        # A packed row and the same row submitted alone must use the same
        # expert and return the same FP32 bytes (no cross-row interaction).
        for expert in range(len(offsets) - 1):
            if offsets[expert] == offsets[expert + 1]:
                continue
            for row in {offsets[expert], offsets[expert + 1] - 1}:
                single_a = MXTensor(a.codes[row : row + 1], a.scales[row : row + 1], "e4m3", (1, k))
                single_offsets = torch.tensor(
                    [0 if i <= expert else 1 for i in range(len(offsets))],
                    dtype=torch.int32,
                    device="cuda",
                )
                single_y = impl.grouped_gemm_fwd(single_a, w, single_offsets)
                if not torch.equal(y[row : row + 1], single_y):
                    raise AssertionError(f"{label}: packed/one-row mismatch at row {row}")
                single_dx = impl.grouped_gemm_bwd(dy[row : row + 1], w, single_offsets)
                if not torch.equal(dx[row : row + 1], single_dx):
                    raise AssertionError(f"{label}: packed/one-row dX mismatch at row {row}")

        invalid_scales = w.scales.clone()
        invalid_scales[0, 0, 0] = 255
        invalid_w = MXTensor(w.codes, invalid_scales, "e2m1", w.shape)
        try:
            impl.grouped_gemm_fwd(a, invalid_w, expert_offsets)
        except ValueError as exc:
            if "255" not in str(exc):
                raise
        else:
            raise AssertionError("invalid E8M0 code 255 was accepted")


def _scale_edge_case(code: int) -> None:
    """Check valid E8M0 extremes without an overflowing reduction."""
    a_values = torch.zeros((1, 32), dtype=torch.float32)
    a_values[0, 0] = 1.0
    a = MXTensor(
        e4m3_encode(a_values).cuda(),
        torch.full((1, 1), 127, dtype=torch.uint8, device="cuda"),
        "e4m3",
        (1, 32),
    )
    nibbles = torch.ones((1, 64, 32), dtype=torch.uint8)
    w = MXTensor(
        pack_nibbles(nibbles).cuda(),
        torch.full((1, 64, 1), code, dtype=torch.uint8, device="cuda"),
        "e2m1",
        (1, 64, 32),
    )
    dy = torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda")
    dy[0, 0] = 1.0
    offsets = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    y = impl.grouped_gemm_fwd(a, w, offsets)
    want_y = oracle.mxfp8_mxfp4_grouped_gemm_fwd(a, w, offsets)
    dx = impl.grouped_gemm_bwd(dy, w, offsets)
    want_dx = oracle.mxfp8_mxfp4_grouped_gemm_bwd(dy, w, offsets)
    _error(f"scale_{code}.Y", y, want_y)
    _error(f"scale_{code}.dX", dx, want_dx)
    # A single nonzero exact power-of-two product cannot hide BF16 subnormal
    # flush or exponent overflow behind the profile's absolute floor of 1.
    if not torch.equal(y, want_y) or not torch.equal(dx, want_dx):
        raise AssertionError(f"scale_{code}: exact one-hot scale edge mismatch")


def _invalid_input_probe():
    a, w, dy, _ = _inputs([0, 1, 2, 3], 65, 96, True)
    for values in ([1, 1, 2, 3], [0, 1, 2, 2], [0, 2, 1, 3], [0, -1, 2, 3]):
        bad = torch.tensor(values, dtype=torch.int32, device="cuda")
        for call in (
            lambda bad=bad: impl.grouped_gemm_fwd(a, w, bad),
            lambda bad=bad: impl.grouped_gemm_bwd(dy, w, bad),
        ):
            try:
                call()
            except ValueError as exc:
                if "expert_offsets" not in str(exc):
                    raise
            else:
                raise AssertionError(f"invalid offsets accepted: {values}")
    offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    for activation in (False, True):
        src = a if activation else w
        bad_scales = src.scales.clone()
        bad_scales.view(-1)[0] = 255
        bad = MXTensor(src.codes, bad_scales, src.elem_format, src.shape)
        calls = [
            lambda bad=bad, activation=activation: impl.grouped_gemm_fwd(
                bad if activation else a, w if activation else bad, offsets
            )
        ]
        if not activation:
            calls.append(lambda bad=bad: impl.grouped_gemm_bwd(dy, bad, offsets))
        for call in calls:
            try:
                call()
            except ValueError as exc:
                if "255" not in str(exc):
                    raise
            else:
                raise AssertionError("invalid E8M0 code accepted")
    REPORT["checked_invalid_inputs"] = "pass"


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This probe requires an SM90 CUDA device")
    if impl.triton is None:
        raise RuntimeError("Triton is unavailable")
    print(f"torch={torch.__version__} triton={impl.triton.__version__} cuda={torch.version.cuda}")
    REPORT["environment"] = {
        "torch": torch.__version__,
        "triton": impl.triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
    }
    REPORT["source_sha256"] = {
        name: hashlib.sha256((ROOT / "rl_engine/moe" / name).read_bytes()).hexdigest()
        for name in (
            "triton_grouped_gemm_wgmma.py",
            "mx_format.py",
        )
    }
    REPORT["configuration"] = {
        "BM": impl.BM,
        "BN_FWD": impl.BN_FWD,
        "BK_FWD": 32,
        "BN_BWD": impl.BN_BWD,
        "BK_BWD": impl.BK_BWD,
        "num_warps": impl.NUM_WARPS,
        "num_stages": impl.NUM_STAGES,
        "enable_fp_fusion": False,
        "max_num_imprecise_acc": 32,
        "split_k": 1,
    }
    # Minimum FP8 lowering gate before complex scheduling or packed decode.
    ones_a = torch.full((64, 32), 0x38, dtype=torch.uint8, device="cuda")
    ones_b = torch.full((32, 64), 0x38, dtype=torch.uint8, device="cuda")
    out = torch.empty((64, 64), dtype=torch.float32, device="cuda")
    _record_kernel(
        "minimal_fp8",
        _minimal_fp8[(1,)](
            ones_a,
            ones_b,
            out,
            num_warps=4,
            num_stages=2,
        ),
    )
    if not torch.equal(out, torch.full_like(out, 32)):
        raise AssertionError("minimal FP8 product mismatch")
    _metadata_probe()
    _invalid_input_probe()
    _case("one_row", [0, 0, 1], n=65, k=32, one_hot=True)
    _case("one_hot_middle_empty", [0, 1, 1, 70], n=65, k=96, one_hot=True)
    _case("large_expert", [0, 129, 130, 130], n=65, k=64, one_hot=False)
    _case("one_large_many_small", [0, 10000, 10001, 10001, 10002], 1, 32, True)
    _case("bm_boundaries", [0, 63, 127, 192], 129, 96, False)
    _case("all_empty", [0, 0, 0], n=65, k=64, one_hot=True)
    _case("no_experts", [0], n=65, k=32, one_hot=True)
    _case("zero_n", [0, 1, 1, 3], n=0, k=96, one_hot=True)
    _graph_probe()
    _scale_edge_case(0)
    _scale_edge_case(253)
    _scale_edge_case(254)
    REPORT["status"] = "probe-pass; external memcheck/racecheck and provider gates still required"
    print("P5-4 Triton WGMMA probe PASS (not provider qualification)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=pathlib.Path, default=ROOT / "runs/triton-wgmma-probe")
    args = parser.parse_args()
    ARTIFACTS = args.artifacts
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as exc:
        REPORT["status"] = "failed"
        REPORT["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (ARTIFACTS / "report.json").write_text(json.dumps(REPORT, indent=2) + "\n")
