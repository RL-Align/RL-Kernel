# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Distributed TP validation for the fused linear_logp operator.

Launch with torchrun, for example:

    torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py

The correctness phase compares the tensor-parallel path against a materialized
full-vocab reference. The optional stress phase skips the full reference and only
checks that larger vocab-sharded runs complete with finite outputs/gradients.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _dtype_default_atol(dtype: torch.dtype, reference_mode: str) -> float:
    if dtype == torch.float32:
        return 1e-4
    if reference_mode == "fp32":
        return 8e-2
    return 3e-2


def _dtype_default_rtol(dtype: torch.dtype, reference_mode: str) -> float:
    if dtype == torch.float32:
        return 1e-4
    if reference_mode == "fp32":
        return 8e-2
    return 3e-2


def _rank_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _init_distributed() -> tuple[int, int, int, torch.device, str]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    rank, local_rank, world_size = _rank_env()
    if world_size < 2:
        raise RuntimeError("Run this script with torchrun and at least 2 processes.")

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices exist."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, device, backend


def _print_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _make_boundaries(vocab_size: int, world_size: int, uneven: bool) -> list[int]:
    if vocab_size < world_size:
        raise ValueError("vocab_size must be >= world_size so every rank has a shard.")

    sizes = [vocab_size // world_size for _ in range(world_size)]
    sizes[-1] += vocab_size % world_size

    if uneven and world_size > 1:
        for rank in range(world_size - 1):
            move = min(rank + 1, sizes[rank] - 1)
            sizes[rank] -= move
            sizes[-1] += move

    boundaries = [0]
    for size in sizes:
        boundaries.append(boundaries[-1] + size)
    if boundaries[-1] != vocab_size:
        raise AssertionError("internal shard boundary construction failed")
    return boundaries


def _materialized_logp(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    reference_mode: str,
) -> torch.Tensor:
    if reference_mode == "fp32":
        logits = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
    else:
        logits = F.linear(hidden, weight, bias).float()
    target_1d = target.reshape(-1).to(device=logits.device, dtype=torch.long)
    selected = torch.gather(
        torch.log_softmax(logits.float(), dim=-1),
        dim=-1,
        index=target_1d.unsqueeze(1),
    ).squeeze(1)
    return selected.reshape(target.shape)


def _load_op(source: str) -> Any:
    if source == "registry":
        from rl_engine.kernels.registry import kernel_registry

        return kernel_registry.get_op("linear_logp")
    if source == "native":
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

        return NativeLinearLogpOp()
    if source == "triton":
        from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

        return TritonLinearLogpOp()
    if source == "sm90":
        from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

        return FusedLinearLogpSM90Op()
    raise ValueError(f"unknown op source: {source}")


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _max_rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-8)
    return float((diff / denom).max().item())


def _reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _reduce_min_int(value: bool, device: torch.device) -> bool:
    tensor = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def _time_block(device: torch.device, fn) -> float:
    _synchronize(device)
    start = time.perf_counter()
    fn()
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def _check_metric(
    *,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    device: torch.device,
) -> tuple[bool, str]:
    local_abs = _max_abs(actual, expected)
    local_rel = _max_rel(actual, expected)
    local_ok = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
    max_abs = _reduce_max(local_abs, device)
    max_rel = _reduce_max(local_rel, device)
    ok = _reduce_min_int(local_ok, device)
    return ok, f"{name}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"


def run_correctness(args, rank: int, world_size: int, device: torch.device, op: Any) -> bool:
    dtype = _parse_dtype(args.dtype)
    boundaries = _make_boundaries(args.vocab_size, world_size, args.uneven_shards)
    start, end = boundaries[rank], boundaries[rank + 1]

    gen = _generator(device, args.seed)
    hidden = torch.randn(
        args.tokens,
        args.hidden_size,
        generator=gen,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(
        args.vocab_size,
        args.hidden_size,
        generator=gen,
        device=device,
        dtype=dtype,
    )
    bias = (
        torch.randn(args.vocab_size, generator=gen, device=device, dtype=dtype)
        if not args.no_bias
        else None
    )
    target = torch.randint(0, args.vocab_size, (args.tokens,), generator=gen, device=device)
    grad_out = torch.randn(args.tokens, generator=gen, device=device, dtype=torch.float32)

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True) if bias is not None else None
    ref_out = _materialized_logp(
        ref_hidden,
        ref_weight,
        target,
        ref_bias,
        reference_mode=args.reference_mode,
    )
    (ref_out * grad_out).sum().backward()

    tp_hidden = hidden.detach().clone().requires_grad_(True)
    local_weight = weight[start:end].detach().clone().requires_grad_(True)
    local_bias = bias[start:end].detach().clone().requires_grad_(True) if bias is not None else None

    tp_out = op(
        tp_hidden,
        local_weight,
        target,
        local_bias,
        tp_group=dist.group.WORLD,
        vocab_start_index=start,
        global_vocab_size=args.vocab_size,
    )
    (tp_out * grad_out).sum().backward()

    atol = args.atol if args.atol is not None else _dtype_default_atol(dtype, args.reference_mode)
    rtol = args.rtol if args.rtol is not None else _dtype_default_rtol(dtype, args.reference_mode)
    local_bias_ref = ref_bias.grad[start:end] if ref_bias is not None else None

    checks = [
        _check_metric(
            name="output",
            actual=tp_out,
            expected=ref_out,
            atol=atol,
            rtol=rtol,
            device=device,
        ),
        _check_metric(
            name="hidden_grad",
            actual=tp_hidden.grad,
            expected=ref_hidden.grad,
            atol=atol,
            rtol=rtol,
            device=device,
        ),
        _check_metric(
            name="weight_grad",
            actual=local_weight.grad,
            expected=ref_weight.grad[start:end],
            atol=atol,
            rtol=rtol,
            device=device,
        ),
    ]
    if local_bias is not None and local_bias_ref is not None:
        checks.append(
            _check_metric(
                name="bias_grad",
                actual=local_bias.grad,
                expected=local_bias_ref,
                atol=atol,
                rtol=rtol,
                device=device,
            )
        )

    if rank == 0:
        print("\n[correctness]")
        print(f"  dtype={dtype}, reference_mode={args.reference_mode}, atol={atol}, rtol={rtol}")
        print(f"  tokens={args.tokens}, hidden={args.hidden_size}, vocab={args.vocab_size}")
        print(f"  shard_boundaries={boundaries}")
        for ok, line in checks:
            print(f"  {'PASS' if ok else 'FAIL'} {line}")

    return all(ok for ok, _ in checks)


def run_stress(args, rank: int, world_size: int, device: torch.device, op: Any) -> bool:
    dtype = _parse_dtype(args.dtype)
    boundaries = _make_boundaries(args.stress_vocab_size, world_size, args.uneven_shards)
    start, end = boundaries[rank], boundaries[rank + 1]

    hidden_gen = _generator(device, args.seed + 1000)
    local_gen = _generator(device, args.seed + 2000 + rank)
    target_gen = _generator(device, args.seed + 3000)

    hidden = torch.randn(
        args.stress_tokens,
        args.stress_hidden_size,
        generator=hidden_gen,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    local_weight = torch.randn(
        end - start,
        args.stress_hidden_size,
        generator=local_gen,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    local_bias = None
    if not args.no_bias:
        local_bias = torch.randn(
            end - start,
            generator=local_gen,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
    target = torch.randint(
        0,
        args.stress_vocab_size,
        (args.stress_tokens,),
        generator=target_gen,
        device=device,
    )

    def step() -> torch.Tensor:
        out = op(
            hidden,
            local_weight,
            target,
            local_bias,
            tp_group=dist.group.WORLD,
            vocab_start_index=start,
            global_vocab_size=args.stress_vocab_size,
        )
        loss = out.float().mean()
        loss.backward()
        return out

    stress_out = None

    def timed_step() -> None:
        nonlocal stress_out
        stress_out = step()

    _reset_peak_memory(device)
    elapsed_ms = _time_block(device, timed_step)

    finite_tensor = (
        torch.isfinite(stress_out).all()
        & torch.isfinite(hidden.grad).all()
        & torch.isfinite(local_weight.grad).all()
    )
    if local_bias is not None:
        finite_tensor = finite_tensor & torch.isfinite(local_bias.grad).all()
    finite = _reduce_min_int(bool(finite_tensor.item()), device)

    peak_gb = _reduce_max(_peak_memory_gb(device), device)
    elapsed_ms = _reduce_max(elapsed_ms, device)
    if rank == 0:
        print("\n[stress]")
        print(
            "  tokens=%d, hidden=%d, vocab=%d, dtype=%s"
            % (args.stress_tokens, args.stress_hidden_size, args.stress_vocab_size, dtype)
        )
        print(f"  shard_boundaries={boundaries}")
        print(f"  finite={'PASS' if finite else 'FAIL'}")
        print(f"  max_rank_elapsed_ms={elapsed_ms:.3f}")
        if device.type == "cuda":
            print(f"  max_rank_peak_memory_gb={peak_gb:.3f}")
    return finite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op-source",
        choices=["registry", "native", "triton", "sm90"],
        default="registry",
    )
    parser.add_argument("--dtype", default="bf16", help="bf16, fp16, or fp32")
    parser.add_argument("--reference-mode", choices=["matching", "fp32"], default="matching")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--uneven-shards", action="store_true")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--run-stress", action="store_true")
    parser.add_argument("--stress-tokens", type=int, default=4096)
    parser.add_argument("--stress-hidden-size", type=int, default=2048)
    parser.add_argument("--stress-vocab-size", type=int, default=32768)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rank, local_rank, world_size, device, backend = _init_distributed()
    try:
        _print_rank0(rank, "[env]")
        _print_rank0(rank, f"  backend={backend}, world_size={world_size}")
        _print_rank0(rank, f"  torch={torch.__version__}, cuda={torch.version.cuda}")
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device)
            cc = torch.cuda.get_device_capability(device)
            _print_rank0(rank, f"  rank0_device={name}, capability=sm_{cc[0]}{cc[1]}")
        _print_rank0(rank, f"  op_source={args.op_source}")

        op = _load_op(args.op_source)
        dist.barrier()
        ok = run_correctness(args, rank, world_size, device, op)
        if args.run_stress:
            dist.barrier()
            ok = run_stress(args, rank, world_size, device, op) and ok

        ok = _reduce_min_int(ok, device)
        dist.barrier()
        if rank == 0:
            print("\n[result]")
            print("  PASS" if ok else "  FAIL")
        if not ok:
            raise SystemExit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        _ = local_rank


if __name__ == "__main__":
    main()
