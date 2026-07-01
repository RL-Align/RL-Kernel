from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_engine.kernels.ops.pytorch.norm.rmsnorm_native import NativeRMSNormOp  # noqa: E402
from rl_engine.kernels.ops.pytorch.norm.rmsnorm_ref import rmsnorm_ref_custom  # noqa: E402
from rl_engine.kernels.ops.triton.rmsnorm_triton import rmsnorm_triton  # noqa: E402

try:
    from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_cuda

    HAS_CUDA_EXT = True
except Exception as exc:
    CUDA_IMPORT_ERROR = exc
    HAS_CUDA_EXT = False


def parse_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def get_impl(name: str):
    if name == "pytorch":
        return rmsnorm_ref_custom
    if name == "triton":
        return rmsnorm_triton
    if name == "cuda":
        if not HAS_CUDA_EXT:
            raise RuntimeError(f"CUDA extension import failed: {CUDA_IMPORT_ERROR}")
        return rmsnorm_cuda
    raise ValueError(f"Unknown impl: {name}")


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def max_rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    denom = b_f.abs().clamp_min(1e-6)
    return ((a_f - b_f).abs() / denom).max().item()


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-5, 2e-5
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    raise ValueError(f"Unsupported dtype: {dtype}")


def assert_close_with_report(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    abs_diff = max_abs_diff(actual, expected)
    rel_diff = max_rel_diff(actual, expected)
    print(
        f"    {name:<8} "
        f"abs_diff={abs_diff:.6e}, "
        f"rel_diff={rel_diff:.6e}, "
        f"atol={atol}, rtol={rtol}"
    )
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch: abs_diff={abs_diff}, rel_diff={rel_diff}, "
            f"atol={atol}, rtol={rtol}"
        )


def assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=0.0, rtol=0.0):
        diff = max_abs_diff(actual, expected)
        raise AssertionError(f"{name} mismatch, max diff = {diff}")


def native_rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = x.float()
    w_f = weight.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    y = x_f * rstd * w_f
    return y.to(x.dtype), rstd.squeeze(-1)


def native_dw(
    x: torch.Tensor,
    dy: torch.Tensor,
    rstd: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    contrib = dy.float() * x.float() * rstd.float().unsqueeze(-1)
    return contrib.masked_fill(~mask[:, None], 0.0).sum(dim=0)


def run_forward_backward(fn, x: torch.Tensor, w: torch.Tensor, dy: torch.Tensor):
    x_req = x.detach().clone().contiguous().requires_grad_(True)
    w_req = w.detach().clone().contiguous().requires_grad_(True)
    dy_req = dy.detach().clone().contiguous()
    y = fn(x_req, w_req)
    y.backward(dy_req)
    return y.detach(), x_req.grad.detach(), w_req.grad.detach()


def run_cuda_dw(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_req = x.detach().clone().contiguous().requires_grad_(True)
    w_req = weight.detach().clone().contiguous().requires_grad_(True)
    y = rmsnorm_cuda(x_req, w_req, eps=eps, mask=mask)
    y.backward(dy.detach().clone().contiguous())
    return w_req.grad.detach()


def build_padded_layout(
    x_real: torch.Tensor,
    dy_real: torch.Tensor,
    total_rows: int,
    real_positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x_real.device
    dtype = x_real.dtype
    t_real, hidden = x_real.shape

    assert len(real_positions) == t_real
    assert total_rows >= t_real

    x_pad = torch.randn((total_rows, hidden), device=device, dtype=torch.float32).to(dtype)
    dy_pad = torch.randn((total_rows, hidden), device=device, dtype=torch.float32).to(dtype)
    mask = torch.zeros((total_rows,), device=device, dtype=torch.bool)

    for src_t, dst_t in enumerate(real_positions):
        x_pad[dst_t] = x_real[src_t]
        dy_pad[dst_t] = dy_real[src_t]
        mask[dst_t] = True

    return x_pad, dy_pad, mask


def test_correctness_case(
    *,
    impl_name: str,
    dtype: torch.dtype,
    total_rows: int,
    hidden: int,
    eps: float,
) -> None:
    torch.manual_seed(0)
    native = NativeRMSNormOp()

    x_cpu = torch.randn(total_rows, hidden, device="cpu", dtype=torch.float32)
    w_cpu = torch.randn(hidden, device="cpu", dtype=torch.float32)
    dy_cpu = torch.randn(total_rows, hidden, device="cpu", dtype=torch.float32)

    x_ref = x_cpu.to(dtype).float().detach().requires_grad_(True)
    w_ref = w_cpu.to(dtype).float().detach().requires_grad_(True)
    dy_ref = dy_cpu.to(dtype).float()
    y_ref = native.forward_fp32(x_ref, w_ref, eps=eps)
    y_ref.backward(dy_ref)

    x_gpu = x_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    w_gpu = w_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    dy_gpu = dy_cpu.to(device="cuda", dtype=dtype)

    fn = get_impl(impl_name)
    y_gpu = fn(x_gpu, w_gpu, eps=eps)
    y_gpu.backward(dy_gpu)

    got_y = y_gpu.detach().cpu()
    got_dx = x_gpu.grad.detach().cpu()
    got_dw = w_gpu.grad.detach().cpu()

    atol, rtol = tolerances(dtype)
    dw_scale = max(1.0, math.sqrt(total_rows) / 4.0)

    print(f"[case] correctness impl={impl_name}, T={total_rows}, H={hidden}, dtype={dtype}")
    assert_close_with_report("y", got_y, y_ref.detach(), atol=atol, rtol=rtol)
    assert_close_with_report("dx", got_dx, x_ref.grad.detach(), atol=atol, rtol=rtol)
    assert_close_with_report(
        "dw",
        got_dw,
        w_ref.grad.detach(),
        atol=atol * dw_scale,
        rtol=rtol * dw_scale,
    )
    print("    passed\n")


def test_deterministic_case(
    *,
    impl_name: str,
    dtype: torch.dtype,
    total_rows: int,
    hidden: int,
    repeat: int,
) -> None:
    torch.manual_seed(0)
    fn = get_impl(impl_name)
    x = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
    w = torch.randn(hidden, device="cuda", dtype=dtype)
    dy = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)

    y0, dx0, dw0 = run_forward_backward(fn, x, w, dy)
    torch.cuda.synchronize()

    for i in range(repeat):
        y, dx, dw = run_forward_backward(fn, x, w, dy)
        torch.cuda.synchronize()
        assert_exact(f"{impl_name} y repeat={i}", y0, y)
        assert_exact(f"{impl_name} dx repeat={i}", dx0, dx)
        assert_exact(f"{impl_name} dw repeat={i}", dw0, dw)

    print(
        f"[PASS] deterministic impl={impl_name}, T={total_rows}, "
        f"H={hidden}, dtype={dtype}, repeat={repeat}"
    )


def test_forward_dx_batch_position_case(
    *,
    impl_name: str,
    dtype: torch.dtype,
    hidden: int,
) -> None:
    torch.manual_seed(1)
    fn = get_impl(impl_name)

    target_x = torch.randn(1, hidden, device="cuda", dtype=dtype)
    target_dy = torch.randn(1, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)

    y_single, dx_single, _ = run_forward_backward(fn, target_x, weight, target_dy)

    placements = [(16, 0), (16, 7), (64, 63)]
    for total_rows, row_id in placements:
        x = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
        dy = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
        x[row_id : row_id + 1] = target_x
        dy[row_id : row_id + 1] = target_dy
        y, dx, _ = run_forward_backward(fn, x, weight, dy)
        assert_exact(f"{impl_name} y row={row_id}", y_single[0], y[row_id])
        assert_exact(f"{impl_name} dx row={row_id}", dx_single[0], dx[row_id])

    print(f"[PASS] batch-position invariant impl={impl_name}, H={hidden}, dtype={dtype}")


def test_forward_dx_padding_layout_case(
    *,
    impl_name: str,
    dtype: torch.dtype,
    hidden: int,
) -> None:
    torch.manual_seed(2)
    fn = get_impl(impl_name)

    valid_rows = 4
    total_rows = 16
    positions = [1, 5, 9, 14]

    valid_x = torch.randn(valid_rows, hidden, device="cuda", dtype=dtype)
    valid_dy = torch.randn(valid_rows, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)

    x_a = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
    dy_a = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
    x_a[0:valid_rows] = valid_x
    dy_a[0:valid_rows] = valid_dy
    y_a, dx_a, _ = run_forward_backward(fn, x_a, weight, dy_a)

    x_b = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
    dy_b = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
    for i, pos in enumerate(positions):
        x_b[pos] = valid_x[i]
        dy_b[pos] = valid_dy[i]
    y_b, dx_b, _ = run_forward_backward(fn, x_b, weight, dy_b)

    for i, pos in enumerate(positions):
        assert_exact(f"{impl_name} padding y row={i}", y_a[i], y_b[pos])
        assert_exact(f"{impl_name} padding dx row={i}", dx_a[i], dx_b[pos])

    print(f"[PASS] padding-layout invariant impl={impl_name}, H={hidden}, dtype={dtype}")


def test_dw_padding_layout_case(
    *,
    dtype: torch.dtype,
    total_rows: int,
    hidden: int,
    eps: float,
    strict_bitwise: bool,
) -> None:
    if not HAS_CUDA_EXT:
        raise RuntimeError(f"CUDA extension import failed: {CUDA_IMPORT_ERROR}")

    torch.manual_seed(0)
    x_real = torch.randn((total_rows, hidden), device="cuda", dtype=torch.float32).to(dtype)
    dy_real = torch.randn((total_rows, hidden), device="cuda", dtype=torch.float32).to(dtype)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.float32).to(dtype)

    x1 = x_real.clone()
    dy1 = dy_real.clone()
    mask1 = torch.ones((total_rows,), device="cuda", dtype=torch.bool)

    x2, dy2, mask2 = build_padded_layout(
        x_real=x_real,
        dy_real=dy_real,
        total_rows=2 * total_rows,
        real_positions=[2 * i + 1 for i in range(total_rows)],
    )

    _, rstd1 = native_rmsnorm_forward(x1, weight, eps)
    _, rstd2 = native_rmsnorm_forward(x2, weight, eps)
    dw1 = run_cuda_dw(x1, dy1, weight, mask1, eps)
    dw2 = run_cuda_dw(x2, dy2, weight, mask2, eps)
    ref_dw1 = native_dw(x1, dy1, rstd1, mask1)
    ref_dw2 = native_dw(x2, dy2, rstd2, mask2)

    atol, rtol = tolerances(dtype)
    print(f"[case] dw padding invariant dtype={dtype}, T={total_rows}, H={hidden}")
    print("    max |cuda dw1 - cuda dw2|:", max_abs_diff(dw1, dw2))
    print("    max |ref  dw1 - ref  dw2|:", max_abs_diff(ref_dw1, ref_dw2))
    print("    max |cuda dw1 - ref  dw1|:", max_abs_diff(dw1, ref_dw1))
    print("    max |cuda dw2 - ref  dw2|:", max_abs_diff(dw2, ref_dw2))
    assert_close_with_report("dw12", dw1, dw2, atol=atol, rtol=rtol)
    assert_close_with_report("dw1ref", dw1, ref_dw1, atol=atol, rtol=rtol)
    assert_close_with_report("dw2ref", dw2, ref_dw2, atol=atol, rtol=rtol)
    if strict_bitwise:
        assert_exact("cuda dw1 - cuda dw2", dw1.float(), dw2.float())

    x3, dy3, mask3 = build_padded_layout(
        x_real=x_real,
        dy_real=dy_real,
        total_rows=2 * total_rows + 1,
        real_positions=[2 * i for i in range(total_rows)],
    )
    _, rstd3 = native_rmsnorm_forward(x3, weight, eps)
    dw3 = run_cuda_dw(x3, dy3, weight, mask3, eps)
    ref_dw3 = native_dw(x3, dy3, rstd3, mask3)

    print("    max |cuda dw2 - cuda dw3|:", max_abs_diff(dw2, dw3))
    print("    max |cuda dw3 - ref  dw3|:", max_abs_diff(dw3, ref_dw3))
    assert_close_with_report("dw23", dw2, dw3, atol=atol, rtol=rtol)
    assert_close_with_report("dw3ref", dw3, ref_dw3, atol=atol, rtol=rtol)
    if strict_bitwise:
        assert_exact("cuda dw2 - cuda dw3", dw2.float(), dw3.float())

    print("[PASS] backward dw is invariant under masked random-padding layout.")


def impls_from_arg(impl: str, *, include_pytorch: bool) -> list[str]:
    if impl != "all":
        return [impl]
    impls = ["triton", "cuda"]
    if include_pytorch:
        impls.insert(0, "pytorch")
    return impls


def kernel_impls_from_arg(impl: str) -> list[str]:
    if impl == "pytorch":
        raise ValueError(
            "correctness suite compares kernels against PyTorch; " "use cuda, triton, or all"
        )
    return impls_from_arg(impl, include_pytorch=False)


def dtypes_from_arg(dtype: str) -> list[torch.dtype]:
    if dtype == "all":
        return [torch.float32, torch.float16, torch.bfloat16]
    return [parse_dtype(dtype)]


def hidden_sizes_from_args(hidden: int, sweep_hidden: bool) -> list[int]:
    if not sweep_hidden:
        return [hidden]
    sizes = [63, 64, 65, 127, 128, 129, 255, 256, 257, hidden]
    return list(dict.fromkeys(sizes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["all", "correctness", "deterministic", "batch", "dw"],
        default="all",
    )
    parser.add_argument("--impl", choices=["pytorch", "triton", "cuda", "all"], default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "all"], default="bf16")
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--H", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--sweep-hidden", action="store_true")
    parser.add_argument("--strict-bitwise", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    run_all = args.suite == "all"

    if run_all or args.suite == "correctness":
        shapes = [
            (1, 128),
            (2, 256),
            (8, 768),
            (16, 1024),
            (32, 2048),
            (64, 4096),
            (128, 4096),
        ]
        for impl in kernel_impls_from_arg(args.impl):
            for dtype in dtypes_from_arg(args.dtype):
                for total_rows, hidden in shapes:
                    test_correctness_case(
                        impl_name=impl,
                        dtype=dtype,
                        total_rows=total_rows,
                        hidden=hidden,
                        eps=args.eps,
                    )
        print("All native-vs-kernel tests passed.")

    if run_all or args.suite == "deterministic":
        for impl in impls_from_arg(args.impl, include_pytorch=True):
            for dtype in dtypes_from_arg(args.dtype):
                test_deterministic_case(
                    impl_name=impl,
                    dtype=dtype,
                    total_rows=args.T,
                    hidden=args.H,
                    repeat=args.repeat,
                )
        print("all deterministic tests passed")

    if run_all or args.suite == "batch":
        for impl in impls_from_arg(args.impl, include_pytorch=True):
            for dtype in dtypes_from_arg(args.dtype):
                for hidden in hidden_sizes_from_args(args.H, args.sweep_hidden):
                    test_forward_dx_batch_position_case(impl_name=impl, dtype=dtype, hidden=hidden)
                    test_forward_dx_padding_layout_case(impl_name=impl, dtype=dtype, hidden=hidden)
        print("all batch-invariant tests passed")

    if run_all or args.suite == "dw":
        for dtype in dtypes_from_arg(args.dtype):
            test_dw_padding_layout_case(
                dtype=dtype,
                total_rows=args.T,
                hidden=args.H,
                eps=args.eps,
                strict_bitwise=args.strict_bitwise,
            )

    print("all RMSNorm tests passed")


if __name__ == "__main__":
    main()
