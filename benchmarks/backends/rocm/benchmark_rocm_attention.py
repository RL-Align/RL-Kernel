# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Operator-only benchmark for the strict ROCm attention path.

Seeded Q/K/V only: no checkpoint, tokenizer, or serving engine.  Defaults to the
Qwen3-8B dense head layout (``Hq=32``, ``Hkv=8``, ``D=128``), BF16, causal
prefill.

Three backends are compared:

``native``
    ``torch.nn.functional.scaled_dot_product_attention`` with the KV heads
    expanded to the Q head count.  Not batch-invariant; present as the
    throughput reference every ROCm deployment already has.
``triton``
    ``flash_attn`` with the ROCm Triton backend enabled.
``strict``
    ``aiter.rocm.ck_dense_mha`` through the WS2 contract dispatch, i.e. the path
    this PR adds.

Three extra sections quantify what the strict contract actually costs and
where it stops holding.  All three are properties of the vendor kernel rather
than of the integration:

* ``determinism_cost`` — AITER ``mha_bwd`` with ``deterministic`` on vs off.
* ``batch_composition`` — whether raw AITER returns the same bits for a batch
  and for the same rows submitted one at a time, swept over shapes because the
  answer varies with shape.
* ``tp_head_count_sensitivity`` — whether a head shard depends on how many
  heads shared its launch, i.e. whether the path is TP-degree invariant.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

DEFAULT_Q_HEADS = 32
DEFAULT_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128


class _ContextParallel:
    rank = 0
    world_size = 1
    layout = "single"


class _Request:
    """Structural request understood by the Vime attention provider."""

    def __init__(self, query, key, value, metadata):
        self.query = query
        self.key = key
        self.value = value
        self.metadata = metadata
        self.context_parallel = _ContextParallel()
        self.tensor_parallel_group = None
        self.key_padding_mask = None


def _metadata(q_heads: int, kv_heads: int) -> dict[str, Any]:
    return {
        "global_q_heads": q_heads,
        "global_kv_heads": kv_heads,
        "tp_rank": 0,
        "tp_world_size": 1,
        "attention_mode": "prefill",
        "role": "train",
        "causal": True,
    }


def _tensors(batch, q_heads, kv_heads, seq_len, head_dim, dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        batch, q_heads, seq_len, head_dim, generator=generator, device="cuda", dtype=dtype
    )
    key = torch.randn(
        batch, kv_heads, seq_len, head_dim, generator=generator, device="cuda", dtype=dtype
    )
    value = torch.randn(
        batch, kv_heads, seq_len, head_dim, generator=generator, device="cuda", dtype=dtype
    )
    return query, key, value


def _measure(run: Callable[[bool], None], *, backward: bool, warmup: int, iters: int):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        run(backward)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(backward)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 4),
        "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }


def _native_runner(query, key, value, q_heads, kv_heads):
    repeats = q_heads // kv_heads

    def run(backward: bool) -> None:
        q = query.detach().requires_grad_(backward)
        k = key.detach().requires_grad_(backward)
        v = value.detach().requires_grad_(backward)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.repeat_interleave(repeats, dim=1),
            v.repeat_interleave(repeats, dim=1),
            is_causal=True,
        )
        if backward:
            out.backward(torch.ones_like(out))

    return run


def _triton_runner(query, key, value, head_dim):
    import os

    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
    from flash_attn import flash_attn_func

    scale = 1.0 / math.sqrt(head_dim)

    def run(backward: bool) -> None:
        q = query.detach().transpose(1, 2).contiguous().requires_grad_(backward)
        k = key.detach().transpose(1, 2).contiguous().requires_grad_(backward)
        v = value.detach().transpose(1, 2).contiguous().requires_grad_(backward)
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        if backward:
            out.backward(torch.ones_like(out))

    return run


def _strict_runner(query, key, value, metadata):
    from rl_engine.integrations.orchestrators.vime.providers import attention_provider

    def run(backward: bool) -> None:
        q = query.detach().requires_grad_(backward)
        k = key.detach().requires_grad_(backward)
        v = value.detach().requires_grad_(backward)
        result = attention_provider(_Request(q, k, v, metadata))
        if backward:
            result.out.backward(torch.ones_like(result.out))

    return run


def _determinism_cost(seq_lens, q_heads, kv_heads, head_dim, dtype, warmup, iters):
    """AITER deterministic backward vs the non-deterministic one."""

    from aiter.ops.mha import mha_bwd, mha_fwd

    scale = 1.0 / math.sqrt(head_dim)
    rows = []
    for seq_len in seq_lens:
        query, key, value = _tensors(1, q_heads, kv_heads, seq_len, head_dim, dtype)
        # AITER consumes [B, S, H, D].
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        out, lse, _mask, rng_state = mha_fwd(q, k, v, 0.0, scale, True, -1, -1, 0, True, False)
        grad_out = torch.ones_like(out)
        entry: dict[str, Any] = {"seq_len": seq_len}
        for deterministic in (True, False):

            # Bind the tensors as defaults: they are released below, and the
            # closure must not depend on the enclosing names still existing.
            def run(
                _backward: bool,
                deterministic=deterministic,
                grad_out=grad_out,
                q=q,
                k=k,
                v=v,
                out=out,
                lse=lse,
                rng_state=rng_state,
            ) -> None:
                mha_bwd(
                    grad_out,
                    q,
                    k,
                    v,
                    out,
                    lse,
                    0.0,
                    scale,
                    True,
                    -1,
                    -1,
                    deterministic,
                    rng_state=rng_state,
                )

            key_name = "deterministic" if deterministic else "non_deterministic"
            entry[key_name] = _measure(run, backward=False, warmup=warmup, iters=iters)
        entry["time_ratio"] = round(
            entry["deterministic"]["median_ms"] / entry["non_deterministic"]["median_ms"], 2
        )
        entry["memory_ratio"] = round(
            entry["deterministic"]["peak_mib"] / entry["non_deterministic"]["peak_mib"], 1
        )
        rows.append(entry)
        del query, key, value, q, k, v, out, lse, grad_out
        torch.cuda.empty_cache()
    return rows


def _batch_composition(q_heads, kv_heads, head_dim, dtype, shapes):
    """Whether raw AITER is batch-composition invariant, swept over shapes.

    The strict core sidesteps this by executing one logical row at a time; this
    section records what the vendor kernel does without that constraint.  The
    sweep matters: invariance holds for most shapes and breaks for a few, so a
    single-shape probe would report whichever answer it happened to land on.
    """

    from aiter.ops.mha import mha_fwd

    scale = 1.0 / math.sqrt(head_dim)

    def forward(query, key, value):
        out, lse, _mask, _rng = mha_fwd(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            0.0,
            scale,
            True,
            -1,
            -1,
            0,
            True,
            False,
        )
        return out.transpose(1, 2).contiguous(), lse

    rows = []
    for batch, seq_len in shapes:
        if batch < 2:
            continue
        query, key, value = _tensors(batch, q_heads, kv_heads, seq_len, head_dim, dtype, seed=11)
        batched_out, batched_lse = forward(query, key, value)
        worst_out = worst_lse = 0.0
        for row in range(batch):
            row_out, row_lse = forward(
                query[row : row + 1], key[row : row + 1], value[row : row + 1]
            )
            worst_out = max(worst_out, (batched_out[row : row + 1] - row_out).abs().max().item())
            worst_lse = max(worst_lse, (batched_lse[row : row + 1] - row_lse).abs().max().item())
        rows.append(
            {
                "batch": batch,
                "seq_len": seq_len,
                "raw_aiter_out_max_abs": worst_out,
                "raw_aiter_lse_max_abs": worst_lse,
                "raw_aiter_is_batch_invariant": worst_out == 0.0 and worst_lse == 0.0,
            }
        )
        print(
            f"batch-composition B={batch} S={seq_len:5d} "
            f"out {worst_out:.6e} lse {worst_lse:.6e} "
            f"{'invariant' if rows[-1]['raw_aiter_is_batch_invariant'] else 'NOT INVARIANT'}",
            flush=True,
        )
        del query, key, value, batched_out, batched_lse
        torch.cuda.empty_cache()
    return rows


def _tp_head_count_sensitivity(q_heads, kv_heads, head_dim, dtype, seq_lens, tp_degrees):
    """Does a head shard depend on how many heads shared its launch?

    TP shards attention by head and performs no cross-rank reduction, so a rank
    computing its own head slice ought to match the corresponding slice of an
    unsharded run.  Where it does not, the strict path is not TP-degree
    invariant and training/rollout must be pinned to one degree.
    """

    from aiter.ops.mha import mha_fwd

    scale = 1.0 / math.sqrt(head_dim)

    def forward(query, key, value):
        out, lse, _mask, _rng = mha_fwd(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            0.0,
            scale,
            True,
            -1,
            -1,
            0,
            True,
            False,
        )
        return out.transpose(1, 2).contiguous(), lse

    rows = []
    for seq_len in seq_lens:
        query, key, value = _tensors(1, q_heads, kv_heads, seq_len, head_dim, dtype, seed=3)
        full_out, full_lse = forward(query, key, value)
        for tp in tp_degrees:
            if q_heads % tp or kv_heads % tp:
                continue
            local_q, local_kv = q_heads // tp, kv_heads // tp
            worst_out = worst_lse = 0.0
            for rank in range(tp):
                shard_out, shard_lse = forward(
                    query[:, rank * local_q : (rank + 1) * local_q],
                    key[:, rank * local_kv : (rank + 1) * local_kv],
                    value[:, rank * local_kv : (rank + 1) * local_kv],
                )
                ref_out = full_out[:, rank * local_q : (rank + 1) * local_q]
                ref_lse = full_lse[:, rank * local_q : (rank + 1) * local_q]
                worst_out = max(worst_out, (shard_out - ref_out).abs().max().item())
                worst_lse = max(worst_lse, (shard_lse - ref_lse).abs().max().item())
            invariant = worst_out == 0.0 and worst_lse == 0.0
            rows.append(
                {
                    "seq_len": seq_len,
                    "tp": tp,
                    "local_q_heads": local_q,
                    "local_kv_heads": local_kv,
                    "out_max_abs": worst_out,
                    "lse_max_abs": worst_lse,
                    "tp_degree_invariant": invariant,
                }
            )
            print(
                f"tp-sensitivity S={seq_len:5d} TP={tp} (Hq={local_q},Hkv={local_kv}) "
                f"out {worst_out:.6e} lse {worst_lse:.6e} "
                f"{'invariant' if invariant else 'NOT INVARIANT'}",
                flush=True,
            )
        del query, key, value, full_out, full_lse
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("attention_results.json"))
    parser.add_argument("--q-heads", type=int, default=DEFAULT_Q_HEADS)
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--shapes",
        default="1x1024,1x2048,1x4096,2x2048,4x2048",
        help="comma-separated BATCHxSEQ pairs",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires a ROCm (or CUDA) device")

    dtype = torch.bfloat16
    metadata = _metadata(args.q_heads, args.kv_heads)
    shapes = [tuple(int(part) for part in pair.split("x")) for pair in args.shapes.split(",")]

    rows = []
    for batch, seq_len in shapes:
        query, key, value = _tensors(
            batch, args.q_heads, args.kv_heads, seq_len, args.head_dim, dtype
        )
        factories = {
            "native": lambda q=query, k=key, v=value: _native_runner(
                q, k, v, args.q_heads, args.kv_heads
            ),
            "triton": lambda q=query, k=key, v=value: _triton_runner(q, k, v, args.head_dim),
            "strict": lambda q=query, k=key, v=value: _strict_runner(q, k, v, metadata),
        }
        for name, factory in factories.items():
            try:
                runner = factory()
                forward = _measure(runner, backward=False, warmup=args.warmup, iters=args.iters)
                combined = _measure(runner, backward=True, warmup=args.warmup, iters=args.iters)
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                print(f"B={batch} S={seq_len} {name}: FAILED {type(exc).__name__}: {exc}")
                continue
            rows.append(
                {
                    "batch": batch,
                    "seq_len": seq_len,
                    "backend": name,
                    "forward": forward,
                    "forward_backward": combined,
                }
            )
            print(
                f"B={batch} S={seq_len:5d} {name:8s} "
                f"fwd {forward['median_ms']:8.4f} p95 {forward['p95_ms']:8.4f} "
                f"peak {forward['peak_mib']:9.1f} | "
                f"fwd+bwd {combined['median_ms']:8.4f} peak {combined['peak_mib']:9.1f}",
                flush=True,
            )
        del query, key, value
        torch.cuda.empty_cache()

    seq_lens = sorted({seq for _batch, seq in shapes})
    determinism = _determinism_cost(
        seq_lens, args.q_heads, args.kv_heads, args.head_dim, dtype, args.warmup, args.iters
    )
    composition_shapes = [
        (batch, seq_len) for batch in (2, 4) for seq_len in sorted({128, 256, 512, *seq_lens})
    ]
    composition = _batch_composition(
        args.q_heads, args.kv_heads, args.head_dim, dtype, composition_shapes
    )
    tp_sensitivity = _tp_head_count_sensitivity(
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        dtype,
        sorted({512, *seq_lens}),
        (2, 4, 8),
    )

    properties = torch.cuda.get_device_properties(0)
    payload = {
        "environment": {
            "gpu": properties.name,
            "arch": getattr(properties, "gcnArchName", "unknown"),
            "device_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "dtype": "bf16",
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
        },
        "latency": rows,
        "determinism_cost": determinism,
        "batch_composition": composition,
        "tp_head_count_sensitivity": tp_sensitivity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
