# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prior-art comparison for the Qwen3-Next TP4 full-attention core (D=256, GQA 4:1).

Every candidate computes causal attention for one rank of Qwen3-Next's TP4
layout: 4 query heads over 1 KV head, head dim 256, scale 1/16, queries aligned
to the end of the KV sequence. For each candidate the report records

* **batch invariance** - a target sequence's output computed alone, and inside
  a batch with three other sequences (placed first and last), must be bitwise
  equal; for candidates with a backward, the same for its dq/dk/dv;
* **prefill/decode invariance** - the last 64 query rows, and the last single
  row, computed against the full KV must equal the same rows of the full
  prefill. This is the replay-vs-rollout boundary of RFC #428;
* **repeatability** - two identical calls are bitwise equal;
* **accuracy** - error against an FP64 evaluation;
* **performance** - median CUDA-event latency of prefill, decode and backward.

A candidate that cannot be imported or launched in the pinned environment is
recorded as ``unavailable`` with the reason instead of failing the run.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

import torch
import torch.nn.functional as F

HQ, HKV, D = 4, 1, 256
SCALE = 1.0 / 16
TARGET = 1000
COMPANIONS = (777, 1500, 64)
CHUNK = 64
PREFILL_TOKENS = (512, 2048, 8192)
DECODE_KV = 8192
BACKWARD_TOKENS = 2048


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #


def _sample_us(fn: Callable[[], Any]) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0


def time_interleaved(calls: dict[str, Callable[[], Any]], warmup: int = 5, iters: int = 30):
    keys = list(calls)
    orders = (keys, list(reversed(keys)))
    for i in range(warmup):
        for key in orders[i % 2]:
            calls[key]()
    samples = {key: [] for key in keys}
    for i in range(iters):
        for key in orders[i % 2]:
            samples[key].append(_sample_us(calls[key]))
    return {key: statistics.median(values) for key, values in samples.items()}


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def _guard(key: str, factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"key": key, **factory()}
    except Exception as exc:  # noqa: BLE001 - record and continue
        return {"key": key, "unavailable": f"{type(exc).__name__}: {str(exc)[:300]}"}


def make_sequence(length: int, seed: int, *, kv_length: int | None = None):
    """q [Lq, HQ, D], k/v [Lk, HKV, D] in BF16 (token-major, as engines store them)."""
    kv_length = length if kv_length is None else kv_length
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, device="cuda", generator=g).to(torch.bfloat16)

    return randn(length, HQ, D), randn(kv_length, HKV, D), randn(kv_length, HKV, D)


def fp64_reference(q, k, v):
    lq, lk = q.shape[0], k.shape[0]
    qd, kd, vd = (t.double().transpose(0, 1) for t in (q, k, v))
    kd, vd = kd.expand(HQ, -1, -1), vd.expand(HQ, -1, -1)
    scores = qd @ kd.transpose(1, 2) * SCALE
    rows = torch.arange(lq, device=q.device)[:, None] + (lk - lq)
    scores = scores.masked_fill(torch.arange(lk, device=q.device)[None, :] > rows, float("-inf"))
    return (scores.softmax(-1) @ vd).transpose(0, 1)


# --------------------------------------------------------------------------- #
# Candidates
#
# ``run(seqs) -> outputs``: ``seqs`` is a list of (q, k, v) with queries aligned
# to the end of each KV sequence; every candidate computes the whole list in the
# way its engine batches it (one varlen launch, or one launch per sequence).
# ``grad(q, k, v, dy) -> (dq, dk, dv)`` when the candidate has a backward.
# --------------------------------------------------------------------------- #


def _rl_kernel():
    from rl_engine.integrations.qwen3_next_forward import shared_attention

    def one(q, k, v):
        out = shared_attention(*(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)))
        return out.squeeze(0).transpose(0, 1)

    def run(seqs):
        with torch.no_grad():
            return [one(*s) for s in seqs]

    def grad(q, k, v, dy):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        return torch.autograd.grad(one(*leaves), leaves, dy)

    return {
        "name": "RL-Kernel deterministic attention (one launch per sequence)",
        "source": "rl_engine.integrations.qwen3_next_forward.shared_attention",
        "run": run,
        "grad": grad,
    }


def _sdpa():
    def one(q, k, v):
        lq, lk = q.shape[0], k.shape[0]
        qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        if lq == lk:
            out = F.scaled_dot_product_attention(
                qh, kh, vh, is_causal=True, scale=SCALE, enable_gqa=True
            )
        else:
            rows = torch.arange(lq, device=q.device)[:, None] + (lk - lq)
            mask = torch.arange(lk, device=q.device)[None, :] <= rows
            out = F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=mask, scale=SCALE, enable_gqa=True
            )
        return out.squeeze(0).transpose(0, 1)

    def run(seqs):
        with torch.no_grad():
            return [one(*s) for s in seqs]

    def grad(q, k, v, dy):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        return torch.autograd.grad(one(*leaves), leaves, dy)

    return {
        "name": "torch SDPA (default backend, one launch per sequence)",
        "source": f"torch {torch.__version__} scaled_dot_product_attention",
        "run": run,
        "grad": grad,
    }


def _cu(lengths):
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return torch.tensor(out, device="cuda", dtype=torch.int32)


def _split(out, lengths):
    return list(out.split(list(lengths)))


def _vllm_fa(num_splits: int):
    import vllm
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    def run(seqs):
        lq = [s[0].shape[0] for s in seqs]
        lk = [s[1].shape[0] for s in seqs]
        q, k, v = (torch.cat([s[i] for s in seqs]) for i in range(3))
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=max(lq),
            cu_seqlens_q=_cu(lq),
            max_seqlen_k=max(lk),
            cu_seqlens_k=_cu(lk),
            softmax_scale=SCALE,
            causal=True,
            num_splits=num_splits,
            fa_version=2,
        )
        return _split(out, lq)

    return {
        "name": f"vLLM FlashAttention-2 varlen, num_splits={num_splits}",
        "source": f"vllm {vllm.__version__} vllm_flash_attn.flash_attn_varlen_func",
        "run": run,
    }


def _vllm_triton():
    import vllm
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    block = 16

    def run(seqs):
        lq = [s[0].shape[0] for s in seqs]
        lk = [s[1].shape[0] for s in seqs]
        blocks = [(n + block - 1) // block for n in lk]
        k_cache = torch.zeros(sum(blocks), block, HKV, D, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        table = torch.zeros(len(seqs), max(blocks), device="cuda", dtype=torch.int32)
        first = 0
        for i, (s, n) in enumerate(zip(seqs, blocks)):
            pad = n * block - s[1].shape[0]
            k_cache[first : first + n] = F.pad(s[1], (0, 0, 0, 0, 0, pad)).view(n, block, HKV, D)
            v_cache[first : first + n] = F.pad(s[2], (0, 0, 0, 0, 0, pad)).view(n, block, HKV, D)
            table[i, :n] = torch.arange(first, first + n, device="cuda")
            first += n
        q = torch.cat([s[0] for s in seqs])
        out = torch.empty_like(q)
        unified_attention(
            q,
            k_cache,
            v_cache,
            out,
            cu_seqlens_q=_cu(lq),
            max_seqlen_q=max(lq),
            seqused_k=torch.tensor(lk, device="cuda", dtype=torch.int32),
            max_seqlen_k=max(lk),
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )
        return _split(out, lq)

    return {
        # Without split-softmax buffers the 2D kernel runs for every batch, which
        # is also the only kernel vLLM's batch-invariant mode allows.
        "name": "vLLM Triton unified attention (2D kernel)",
        "source": f"vllm {vllm.__version__} v1/attention/ops/triton_unified_attention",
        "run": run,
    }


def _flashinfer():
    import flashinfer

    workspace = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

    def run(seqs):
        lq = [s[0].shape[0] for s in seqs]
        lk = [s[1].shape[0] for s in seqs]
        wrapper.plan(
            _cu(lq),
            _cu(lk),
            HQ,
            HKV,
            D,
            causal=True,
            sm_scale=SCALE,
            q_data_type=torch.bfloat16,
        )
        q, k, v = (torch.cat([s[i] for s in seqs]) for i in range(3))
        return _split(wrapper.run(q, k, v), lq)

    return {
        "name": "FlashInfer BatchPrefillWithRaggedKVCache",
        "source": f"flashinfer {flashinfer.__version__}",
        "run": run,
    }


def candidates() -> list[dict[str, Any]]:
    return [
        _guard("rl_kernel_cuda", _rl_kernel),
        _guard("torch_sdpa", _sdpa),
        _guard("vllm_fa2_auto", lambda: _vllm_fa(0)),
        _guard("vllm_fa2_split1", lambda: _vllm_fa(1)),
        _guard("vllm_triton_2d", _vllm_triton),
        _guard("flashinfer", _flashinfer),
    ]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _batch_bi(cand, target, others) -> dict[str, bool]:
    alone = cand["run"]([target])[0]
    first = cand["run"]([target, *others])[0]
    last = cand["run"]([*others, target])[-1]
    return {"first": _bitwise(first, alone), "last": _bitwise(last, alone)}


def _prefill_decode(cand, target) -> dict[str, bool]:
    q, k, v = target
    full = cand["run"]([target])[0]
    result = {}
    for rows in (CHUNK, 1):
        part = cand["run"]([(q[-rows:], k, v)])[0]
        result[f"last_{rows}"] = _bitwise(part, full[-rows:])
    return result


def _grad_bi(cand, target, others) -> dict[str, bool]:
    g = torch.Generator(device="cuda").manual_seed(5)
    dy = torch.randn(target[0].shape, device="cuda", generator=g).to(torch.bfloat16)
    alone = cand["grad"](*target, dy)
    # A backward is per sequence in every engine here; what can change is the
    # kernel the shape selects, so compare against a second target placed after
    # other sequences have been differentiated.
    for other in others:
        cand["grad"](*other, torch.ones_like(other[0]))
    again = cand["grad"](*target, dy)
    return {name: _bitwise(a, b) for name, a, b in zip(("dq", "dk", "dv"), again, alone)}


def _one(cand, target, others) -> dict[str, Any]:
    if "unavailable" in cand:
        return cand
    out = {"key": cand["key"], "name": cand["name"], "source": cand["source"]}
    try:
        out["batch_bitwise"] = _batch_bi(cand, target, others)
        out["prefill_decode_bitwise"] = _prefill_decode(cand, target)
        out["repeatable"] = _bitwise(cand["run"]([target])[0], cand["run"]([target])[0])
        if "grad" in cand:
            out["backward_repeat_bitwise"] = _grad_bi(cand, target, others)
        ref = fp64_reference(*target)
        got = cand["run"]([target])[0].double()
        out["accuracy"] = {
            "rel_l2_vs_fp64": float((got - ref).norm() / ref.norm()),
            "max_abs_vs_fp64": float((got - ref).abs().max()),
        }
    except Exception as exc:  # noqa: BLE001 - a crash is a result, not a harness failure
        out["failed"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out["batch_invariant"] = bool(
        "failed" not in out
        and all(out["batch_bitwise"].values())
        and all(out["prefill_decode_bitwise"].values())
        and all(out.get("backward_repeat_bitwise", {"": True}).values())
    )
    return out


def _latency(cands) -> dict[str, Any]:
    ready = [c for c in cands if "unavailable" not in c]

    def calls_for(seqs):
        calls = {}
        for cand in ready:
            try:
                cand["run"](seqs)
            except Exception:  # noqa: BLE001 - recorded by the BI section
                continue
            calls[cand["key"]] = lambda c=cand: c["run"](seqs)
        return calls

    prefill = {
        str(n): time_interleaved(calls_for([make_sequence(n, 300 + n)])) for n in PREFILL_TOKENS
    }
    decode = time_interleaved(calls_for([make_sequence(1, 401, kv_length=DECODE_KV)]))
    q, k, v = make_sequence(BACKWARD_TOKENS, 402)
    dy = torch.randn_like(q)
    backward = time_interleaved(
        {c["key"]: (lambda c=c: c["grad"](q, k, v, dy)) for c in ready if "grad" in c},
        warmup=3,
        iters=15,
    )
    return {
        "prefill_us": prefill,
        f"decode_us_kv{DECODE_KV}": decode,
        f"forward_plus_backward_us_{BACKWARD_TOKENS}": backward,
    }


def attention_report() -> dict[str, Any]:
    target = make_sequence(TARGET, 1)
    others = [make_sequence(n, 10 + i) for i, n in enumerate(COMPANIONS)]
    cands = candidates()
    return {
        "op": "qwen3_next_tp4_attention",
        "shape": {"q_heads": HQ, "kv_heads": HKV, "head_dim": D, "scale": SCALE},
        "target_tokens": TARGET,
        "companion_tokens": list(COMPANIONS),
        "candidates": [_one(c, target, others) for c in cands],
        "latency": _latency(cands),
    }
