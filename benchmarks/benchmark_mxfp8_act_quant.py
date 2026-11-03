# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-1 ``mxfp8_act_quant``: torch-native vs Triton vs CUDA vs torchao.

Baselines
---------
``torch-native``     the start-kit oracle ``rl_engine.moe.mx_format.mx_quantize``:
                     reshape / amax / divide / cast as separate eager ops.
``torchao``          ``torchao.prototype.mx_formats.mx_tensor.to_mx`` in FLOOR
                     scaling mode — the reference OCP-MX cast in PyTorch land,
                     measured eager and under ``torch.compile`` (how torchao is
                     actually deployed).
``triton_kernels``   ``triton_kernels.numerics_details.mxfp.downcast_to_mxfp``
                     from the Triton repo — the MX quantizer vLLM runs for its
                     MXFP4/GPT-OSS path, and the closest production kernel to
                     this operator (same block-32 E8M0 layout).
``vllm``             ``per_token_group_quant_fp8`` — vLLM's (and, through the
                     same algorithm in sgl-kernel, SGLang's) production FP8
                     activation quantizer, dispatching to its hand-written CUDA
                     kernel ``torch.ops._C.per_token_group_fp8_quant``. Note the
                     *different format*: group 128 with one FP32 scale (the
                     DeepSeek-V3 recipe), so it is a cost reference, not a
                     drop-in alternative. It is also measured at group 32 with
                     ``use_ue8m0=True``, the configuration closest to MX.
``roofline``         ``x.to(torch.float8_e4m3fn)``: the same memory traffic minus
                     the block reduction and the scale bytes, i.e. the fastest a
                     correct kernel could possibly be on this hardware.

Both MX-format baselines (torchao FLOOR, triton_kernels ROUND_DOWN) emit
*byte-identical* output to our kernels, which independently validates the P5
recipe against two outside implementations.

torchao's own fused MX kernels (``triton_to_mxfp8_dim0``, ``mxfp8_quantize_cuda``)
are gated on ``is_sm_at_least_100()`` — MXFP8 is Blackwell-native — so on Hopper
they raise ``AssertionError("needs triton")``. NVIDIA Transformer Engine gates
its MXFP8 quantizer the same way.

The operator is memory bound: it reads ``numel * itemsize`` bytes and writes
``numel`` element bytes plus ``numel / 32`` E8M0 scale bytes, so the
effective-bandwidth column is the number to look at.

The fail-closed non-finite check costs one device sync per call (plus a flag
memset). It is off by default here and reported separately with
``--check-finite``; the providers expose the same switch as a ``check_finite``
attribute so a caller can hoist the check to once per step.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_act_quant.py
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_mxfp8_act_quant.py \
        --dtype float32 --iters 200 --check-finite --backward
"""

from __future__ import annotations

import argparse

import torch

from rl_engine.kernels.ops.cuda.moe import mxfp8_act_quant_bwd_cuda, mxfp8_act_quant_fwd_cuda
from rl_engine.moe.mx_format import MX_BLOCK, mx_quantize
from rl_engine.moe.oracle import mxfp8_act_quant_bwd

try:
    from rl_engine.kernels.ops.triton.moe import (
        mxfp8_act_quant_bwd_triton,
        mxfp8_act_quant_fwd_triton,
    )

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is an optional dependency
    _HAS_TRITON = False

try:
    from torchao.prototype.mx_formats.mx_tensor import to_mx as _torchao_to_mx

    _HAS_TORCHAO = True
except ImportError:  # pragma: no cover - torchao is an optional baseline
    _HAS_TORCHAO = False

try:
    from triton_kernels.numerics_details.mxfp import DequantScaleRoundingMode, downcast_to_mxfp

    _HAS_TRITON_KERNELS = True
except ImportError:  # pragma: no cover - triton_kernels is an optional baseline
    _HAS_TRITON_KERNELS = False

try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    _HAS_VLLM = True
except ImportError:  # pragma: no cover - vllm is an optional baseline
    _HAS_VLLM = False

DEV = "cuda"
WARMUP, ITERS = 20, 100

# (name, tokens, hidden) — routed-expert activation tiles at the DeepSeek-V3
# hidden size 7168 and the packed FFN width 2048, plus the one-row decode
# geometry the P5 contract calls out.
SHAPES = [
    ("decode_1x7168", 1, 7168),
    ("small_128x7168", 128, 7168),
    ("prefill_4096x7168", 4096, 7168),
    ("prefill_16384x7168", 16384, 7168),
    ("ffn_8192x2048", 8192, 2048),
]

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _time(fn, x, iters: int) -> float:
    """Milliseconds per call, CUDA-event timed."""
    for _ in range(WARMUP):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bandwidth_gbps(x: torch.Tensor, ms: float) -> float:
    moved = x.numel() * x.element_size() + x.numel() + x.numel() // MX_BLOCK
    return moved / (ms * 1e-3) / 1e9


def _fmt(ms: float) -> str:
    return "n/a" if ms != ms else f"{ms:.4f}"


def build_candidates(check_finite: bool):
    """(label, note, callable) triples, ordered slowest-family first."""
    candidates = [("torch-native", "P5 oracle", lambda t: mx_quantize(t, "e4m3"))]
    if _HAS_TORCHAO:
        compiled = torch.compile(_torchao_to_mx, dynamic=False)
        candidates += [
            ("torchao", "MX, eager", lambda t: _torchao_to_mx(t, torch.float8_e4m3fn, MX_BLOCK)),
            (
                "torchao+compile",
                "MX, inductor",
                lambda t: compiled(t, torch.float8_e4m3fn, MX_BLOCK),
            ),
        ]
    if _HAS_TRITON_KERNELS:
        round_down = DequantScaleRoundingMode.ROUND_DOWN
        candidates.append(
            (
                "triton_kernels",
                "MX, vLLM MXFP4 path",
                lambda t: downcast_to_mxfp(
                    t, torch.float8_e4m3fn, axis=-1, DEQUANT_SCALE_ROUNDING_MODE=round_down
                ),
            )
        )
    if _HAS_VLLM:
        candidates += [
            (
                "vllm-g128",
                "fp8 g128 + fp32 scale",
                lambda t: per_token_group_quant_fp8(t, 128, use_ue8m0=False),
            ),
            (
                "vllm-g32-ue8m0",
                "fp8 g32 + e8m0 scale",
                lambda t: per_token_group_quant_fp8(t, MX_BLOCK, use_ue8m0=True),
            ),
        ]
    if _HAS_TRITON:
        candidates.append(
            (
                "ours-triton",
                "MX",
                lambda t: mxfp8_act_quant_fwd_triton(t, check_finite=check_finite),
            )
        )
    candidates.append(
        ("ours-cuda", "MX", lambda t: mxfp8_act_quant_fwd_cuda(t, check_finite=check_finite))
    )
    candidates.append(("roofline-cast", "lower bound", lambda t: t.to(torch.float8_e4m3fn)))
    return candidates


def verify_baselines_are_byte_equal(dtype: torch.dtype) -> None:
    """The comparison is only meaningful if every MX path emits the same bytes."""
    x = (torch.randn(64, 256, device=DEV) * 3.0).to(dtype)
    ref = mx_quantize(x, "e4m3")

    def same(codes: torch.Tensor, scales: torch.Tensor) -> bool:
        return torch.equal(codes.view(torch.uint8), ref.codes) and torch.equal(
            scales.view(torch.uint8).reshape(ref.scales.shape), ref.scales
        )

    if _HAS_TORCHAO:
        scales, codes = _torchao_to_mx(x, torch.float8_e4m3fn, MX_BLOCK)
        print(f"torchao to_mx(FLOOR)       byte-identical to the P5 oracle: {same(codes, scales)}")
    if _HAS_TRITON_KERNELS:
        codes, scales = downcast_to_mxfp(
            x,
            torch.float8_e4m3fn,
            axis=-1,
            DEQUANT_SCALE_ROUNDING_MODE=DequantScaleRoundingMode.ROUND_DOWN,
        )
        print(f"triton_kernels(ROUND_DOWN) byte-identical to the P5 oracle: {same(codes, scales)}")
    if _HAS_TRITON:
        got = mxfp8_act_quant_fwd_triton(x)
        assert same(got.codes, got.scales)
    got = mxfp8_act_quant_fwd_cuda(x)
    assert same(got.codes, got.scales)


def run(dtype: torch.dtype, iters: int, check_finite: bool) -> None:
    """One row per implementation, one column per shape (ms/call)."""
    torch.manual_seed(0)
    candidates = build_candidates(check_finite)
    check = "on" if check_finite else "off"
    print(f"\nforward  dtype={dtype}  iters={iters}  fail-closed check={check}")
    print(f"device={torch.cuda.get_device_name()}\n")

    inputs = [
        (name, (torch.randn(rows, cols, device=DEV) * 3.0).to(dtype)) for name, rows, cols in SHAPES
    ]
    header = f"{'implementation':<18}{'format':<24}" + "".join(
        f"{name.split('_')[-1]:>13}" for name, _ in inputs
    )
    print(header)
    print("-" * len(header))
    ours = {}
    for label, note, fn in candidates:
        cells = []
        for name, x in inputs:
            ms = _time(fn, x, iters)
            cells.append(ms)
            if label.startswith("ours"):
                ours[name] = min(ours.get(name, float("inf")), ms)
        print(f"{label:<18}{note:<24}" + "".join(f"{ms:>13.4f}" for ms in cells))

    print()
    print(
        f"{'ours, effective bandwidth':<42}"
        + "".join(f"{_bandwidth_gbps(x, ours[name]):>12.0f}G" for name, x in inputs)
    )


def run_backward(dtype: torch.dtype, iters: int) -> None:
    """STE backward (dX = dY): a pure copy, so ``clone()`` is the right baseline."""
    print(f"\nbackward (STE, dX = dY)  dtype={dtype}")
    header = f"{'shape':<20}{'torch clone (ms)':>20}{'triton (ms)':>14}{'cuda (ms)':>12}"
    print(header)
    print("-" * len(header))
    for name, rows, cols in SHAPES:
        dy = (torch.randn(rows, cols, device=DEV) * 3.0).to(dtype)
        native = _time(mxfp8_act_quant_bwd, dy, iters)
        cuda = _time(mxfp8_act_quant_bwd_cuda, dy, iters)
        triton_ms = _time(mxfp8_act_quant_bwd_triton, dy, iters) if _HAS_TRITON else float("nan")
        print(f"{name:<20}{native:>20.4f}{_fmt(triton_ms):>14}{cuda:>12.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument(
        "--check-finite",
        action="store_true",
        help="keep the fail-closed non-finite check (one device sync per call)",
    )
    parser.add_argument("--backward", action="store_true", help="also benchmark the STE backward")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("benchmark requires a CUDA device")
    if not _HAS_TRITON:
        print("warning: triton is unavailable; the triton column will be n/a")
    if not _HAS_TORCHAO:
        print("warning: torchao is unavailable; the SOTA baseline columns are skipped")

    dtype = DTYPES[args.dtype]
    verify_baselines_are_byte_equal(dtype)
    run(dtype, args.iters, args.check_finite)
    if args.backward:
        run_backward(dtype, args.iters)


if __name__ == "__main__":
    main()
