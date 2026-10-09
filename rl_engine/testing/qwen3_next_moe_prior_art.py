# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prior-art comparison for the Qwen3-Next routed MoE (RFC #428 ``moe_route_combine_contract``).

Every candidate computes the routed half of ``Qwen3NextSparseMoeBlock`` at the
official TP1 shape (H=2048, 512 experts, top-10, expert width 512): router
logits, top-10 selection with renormalised weights, the selected SwiGLU
experts and the weighted combine. The shared expert is left out because every
candidate evaluates it as the same dense MLP.

For each candidate the report records

* **routing batch invariance** - the expert ids and weights of a fixed block
  of probe tokens computed alone, and embedded first and last in larger
  batches, must be bitwise equal;
* **output batch invariance** - the same for the routed output;
* **dx / dW batch invariance** (candidates with a backward) - the probe rows'
  input gradient must not change when unrelated rows join the batch, and the
  expert weight gradient must not change when rows whose output gradient is
  zero join it (mathematically neither can);
* **repeatability** - two identical calls are bitwise equal;
* **accuracy** - error against an FP64 evaluation of the HF formula with FP64
  routing, plus the fraction of tokens whose selected expert set differs;
* **performance** - median CUDA-event latency, candidates interleaved.

A candidate that cannot be imported or launched in the pinned environment is
recorded as ``unavailable`` with the reason instead of failing the run.
"""

from __future__ import annotations

import os
import statistics
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn.functional as F

HIDDEN = 2048
EXPERTS = 512
TOPK = 10
WIDTH = 512
PROBE = 8
BI_SIZES = (16, 64, 256, 1024)
PERF_TOKENS = (1, 8, 64, 256, 1024, 4096)
BACKWARD_TOKENS = (64, 1024)


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
    """Median latency per candidate; the order alternates so none always runs first."""
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


def _rel(value: torch.Tensor, reference: torch.Tensor) -> float:
    reference = reference.double()
    return float((value.double() - reference).norm() / reference.norm().clamp_min(1e-300))


def _guard(key: str, factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"key": key, **factory()}
    except Exception as exc:  # noqa: BLE001 - record and continue
        return {"key": key, "unavailable": f"{type(exc).__name__}: {str(exc)[:300]}"}


@contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


# --------------------------------------------------------------------------- #
# Inputs and the FP64 reference
# --------------------------------------------------------------------------- #


def make_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(shape, device="cuda", generator=g) * scale).to(torch.bfloat16)

    return {
        "router": randn(EXPERTS, HIDDEN, scale=0.02),
        "gate_up": randn(EXPERTS, 2 * WIDTH, HIDDEN, scale=0.02),
        "down": randn(EXPERTS, HIDDEN, WIDTH, scale=0.02),
    }


def make_tokens(count: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(count, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)


def fp64_reference(x: torch.Tensor, w: dict[str, torch.Tensor]):
    """HF formula in FP64 with FP64 routing; returns (output, expert ids)."""
    x64 = x.double()
    probs = (x64 @ w["router"].double().T).softmax(-1)
    weights, ids = probs.topk(TOPK, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    out = torch.zeros_like(x64)
    gate_up, down = w["gate_up"], w["down"]
    for expert in ids.unique().tolist():
        token, slot = torch.where(ids == expert)
        gate, up = (x64[token] @ gate_up[expert].double().T).chunk(2, dim=-1)
        y = (F.silu(gate) * up) @ down[expert].double().T
        out.index_add_(0, token, y * weights[token, slot, None])
    return out, ids


# --------------------------------------------------------------------------- #
# Candidates
#
# ``route(x) -> (ids [T,10], weights [T,10])`` exposes the routing decision;
# ``fwd(x) -> [T, H]`` is the routed output; ``graph(x) -> (y, leaves)`` (when
# the candidate is differentiable) returns the output and [x, gate_up, down].
# --------------------------------------------------------------------------- #


def _rl_kernel(w):
    from rl_engine.integrations import qwen3_next_forward as provider

    def route(x):
        routes = provider.stable_top10_routes(provider.shared_router(x, w["router"]))
        return routes.indices, routes.weights

    def fwd(x):
        with torch.no_grad():
            return provider.shared_moe(x, w["router"], w["gate_up"], w["down"])[0]

    gate_up = w["gate_up"].detach().clone().requires_grad_(True)
    down = w["down"].detach().clone().requires_grad_(True)

    def graph(x):
        leaf = x.detach().clone().requires_grad_(True)
        y, _ = provider.shared_moe(leaf, w["router"], gate_up, down)
        return y, [leaf, gate_up, down]

    return {
        "name": "rl-kernel shared_moe",
        "source": "rl_engine.integrations.qwen3_next_forward.shared_moe",
        "route": route,
        "fwd": fwd,
        "graph": graph,
    }


def _hf_modules(w, device="cuda"):
    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next import modeling_qwen3_next as hf

    config = Qwen3NextConfig(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=TOPK,
        moe_intermediate_size=WIDTH,
        norm_topk_prob=True,
    )
    router = hf.Qwen3NextTopKRouter(config).to(device=device, dtype=torch.bfloat16)
    experts = hf.Qwen3NextExperts(config).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        router.weight.copy_(w["router"])
        experts.gate_up_proj.copy_(w["gate_up"])
        experts.down_proj.copy_(w["down"])
    return router, experts


def _hf(w):
    import transformers

    router, experts = _hf_modules(w)

    def route(x):
        _, weights, ids = router(x)
        return ids, weights

    def call(x, mod):
        _, weights, ids = router(x)
        return mod(x, ids, weights)

    def fwd(x):
        with torch.no_grad():
            return call(x, experts)

    def graph(x):
        leaf = x.detach().clone().requires_grad_(True)
        return call(leaf, experts), [leaf, experts.gate_up_proj, experts.down_proj]

    return {
        "name": "HF transformers Qwen3NextExperts (eager)",
        "source": f"transformers {transformers.__version__} modeling_qwen3_next",
        "route": route,
        "fwd": fwd,
        "graph": graph,
    }


def _vllm(w, batch_invariant: bool):
    import vllm
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    flag = "1" if batch_invariant else "0"
    linear = F.linear
    if batch_invariant:
        from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant

        linear = linear_batch_invariant

    def route(x):
        with _env("VLLM_BATCH_INVARIANT", flag):
            # vLLM's Qwen3-Next gate is a BF16 ReplicatedLinear; batch-invariant
            # mode replaces its aten GEMM with vLLM's persistent Triton matmul.
            logits = linear(x, w["router"])
            weights, ids, _ = fused_topk(x, logits, TOPK, renormalize=True)
        return ids.long(), weights

    def fwd(x):
        with torch.no_grad(), _env("VLLM_BATCH_INVARIANT", flag):
            ids, weights = route(x)
            return fused_experts(x, w["gate_up"], w["down"], weights, ids.int())

    return {
        "name": f"vLLM fused_topk + fused_experts (Triton), VLLM_BATCH_INVARIANT={flag}",
        "source": f"vllm {vllm.__version__} model_executor/layers/fused_moe",
        "route": route,
        "fwd": fwd,
    }


def _flashinfer(w):
    import flashinfer
    from flashinfer.fused_moe import cutlass_fused_moe

    # CUTLASS SwiGLU takes fc1 as [up; gate] (the opposite half order of HF).
    gate, up = w["gate_up"].chunk(2, dim=1)
    fc1 = torch.cat((up, gate), dim=1).contiguous()

    def route(x):
        probs = F.linear(x, w["router"]).float().softmax(-1)
        weights, ids = probs.topk(TOPK, dim=-1)
        return ids, weights / weights.sum(-1, keepdim=True)

    def fwd(x):
        with torch.no_grad():
            ids, weights = route(x)
            out = cutlass_fused_moe(
                x, ids.int(), weights, fc1, w["down"], torch.bfloat16, quant_scales=[]
            )
            return out[0] if isinstance(out, (list, tuple)) else out

    return {
        "name": "FlashInfer cutlass_fused_moe (routes from torch.topk)",
        "source": f"flashinfer {flashinfer.__version__} fused_moe.cutlass_fused_moe",
        "route": route,
        "fwd": fwd,
    }


def _megatron_layer(w):
    """Megatron-core MoELayer configured as VIME runs Qwen3-Next.

    VIME's ``scripts/models/qwen3-next-80B-A3B.sh``:

    Softmax router computed in FP32, top-10 of 512, all-to-all dispatcher, TE
    grouped GEMM and TE fused permute, no auxiliary loss. Single process, TP1/EP1.
    """
    import os

    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        device = torch.device("cuda", torch.cuda.current_device())
        dist.init_process_group("nccl", rank=0, world_size=1, device_id=device)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=16,
        ffn_hidden_size=WIDTH,
        moe_ffn_hidden_size=WIDTH,
        num_moe_experts=EXPERTS,
        moe_router_topk=TOPK,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    spec = get_moe_module_spec(use_te=True, num_experts=EXPERTS, moe_grouped_gemm=True)
    layer = build_module(spec, config=config).cuda()
    with torch.no_grad():
        layer.router.weight.copy_(w["router"])
        for expert in range(EXPERTS):
            getattr(layer.experts.linear_fc1, f"weight{expert}").copy_(w["gate_up"][expert])
            getattr(layer.experts.linear_fc2, f"weight{expert}").copy_(w["down"][expert])
    return layer


def _megatron(w):
    import megatron.core
    import transformer_engine

    layer = _megatron_layer(w)
    gate_up = [getattr(layer.experts.linear_fc1, f"weight{e}") for e in range(EXPERTS)]
    down = [getattr(layer.experts.linear_fc2, f"weight{e}") for e in range(EXPERTS)]

    def route(x):
        with torch.no_grad():
            probs, routing_map = layer.router(x)
        weights, ids = probs.topk(TOPK, dim=-1)
        if not bool(routing_map.gather(1, ids).all()):
            raise RuntimeError("Megatron routing map disagrees with its routed probabilities")
        return ids, weights

    def call(x):
        return layer(x.unsqueeze(1))[0].squeeze(1)

    def fwd(x):
        with torch.no_grad():
            return call(x)

    def grads(x, dy):
        leaf = x.detach().clone().requires_grad_(True)
        dx, *dw = torch.autograd.grad(call(leaf), [leaf, *gate_up, *down], dy)
        return dx, torch.stack(dw[:EXPERTS]), torch.stack(dw[EXPERTS:])

    return {
        "name": "Megatron-core MoELayer + TE grouped GEMM (VIME's Qwen3-Next config)",
        "source": f"megatron-core {megatron.core.__version__}, "
        f"transformer-engine {transformer_engine.__version__}",
        "route": route,
        "fwd": fwd,
        "grads": grads,
    }


def _sglang_shims():
    """Make SGLang's Triton MoE path importable next to torch 2.13.

    ``sgl_kernel`` 0.3.21 is built against another libtorch ABI and cannot load
    here. On CUDA, SGLang's Triton MoE calls two of its kernels; both are
    replaced by SGLang's own implementations of the same operation: the Triton
    ``moe_sum_reduce_triton`` and the JIT ``moe_align_block_size`` that SGLang
    registers with the same signature. Every other ``sgl_kernel`` symbol only
    has to import; calling one raises.
    """
    import importlib.abc
    import importlib.machinery
    import sys
    import types

    class Missing:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return Missing(f"{self.name}.{name}")

        def __call__(self, *args, **kwargs):
            raise RuntimeError(f"{self.name} called, but sgl_kernel cannot load next to torch 2.13")

    class Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return Missing(f"{self.__name__}.{name}")

    class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            if name in ("sgl_kernel", "gguf") or name.startswith("sgl_kernel."):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)

        def create_module(self, spec):
            module = Stub(spec.name)
            module.__path__ = []
            return module

        def exec_module(self, module):
            pass

    if not any(type(f).__name__ == "Finder" for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    import sgl_kernel
    import sglang.kernels.ops.moe as moe_ops
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import moe_sum_reduce_triton
    from sglang.kernels.spec import KernelBackend

    sgl_kernel.moe_sum_reduce = moe_sum_reduce_triton
    lookup = moe_ops.get_kernel
    moe_ops.get_kernel = lambda op, backend: lookup(
        op, KernelBackend.JIT if op == "moe.moe_align_block_size" else backend
    )


def _sglang(w, deterministic: bool):
    import sglang

    _sglang_shims()
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as fm
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config as fc
    from sglang.srt.layers.moe.topk import StandardTopKOutput, fused_topk_torch_native

    class Bag:
        def __init__(self, **values):
            self.__dict__.update(values)

        def __getattr__(self, name):
            return False

    # SGLang reads these switches from its published server config; publishing
    # needs a full server (and sgl_kernel). Only the deterministic switch matters.
    settings = Bag(deterministic=Bag(enable_deterministic_inference=deterministic), moe=Bag())
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as bio

    def linear(x, weight):
        # Deterministic mode: what enable_batch_invariant_mode() installs for aten::mm,
        # called directly so the override does not leak into other candidates.
        return bio.mm_batch_invariant(x, weight.t()) if deterministic else F.linear(x, weight)

    config = MoeRunnerConfig(
        num_experts=EXPERTS,
        num_local_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=WIDTH,
        top_k=TOPK,
        params_dtype=torch.bfloat16,
        inplace=False,
    )

    def set_mode():
        fm.get_exec = fc.get_exec = lambda: settings
        fm.is_batch_invariant_mode_enabled = lambda: deterministic
        # One process, no TP group: symmetric memory is disabled, the group unused.
        fm.get_parallel = lambda: Bag(tp_group=None)
        fm.is_allocation_symmetric = lambda: False

    def route(x):
        set_mode()
        logits = linear(x, w["router"])
        weights, ids = fused_topk_torch_native(x, logits, TOPK, renormalize=True)[:2]
        return ids.long(), weights

    def fwd(x):
        with torch.no_grad():
            ids, weights = route(x)
            topk = StandardTopKOutput(weights, ids.int(), None)
            return fm.fused_experts(x, w["gate_up"], w["down"], topk, config)

    mode = "deterministic inference" if deterministic else "default"
    return {
        "name": f"SGLang fused_moe (Triton), {mode}",
        "source": f"sglang {sglang.__version__} moe_runner/triton_utils",
        "route": route,
        "fwd": fwd,
    }


def candidates(w) -> list[dict[str, Any]]:
    return [
        _guard("rl_kernel_cuda", lambda: _rl_kernel(w)),
        _guard("hf_transformers", lambda: _hf(w)),
        _guard("vllm_bi0", lambda: _vllm(w, False)),
        _guard("vllm_bi1", lambda: _vllm(w, True)),
        _guard("flashinfer_cutlass", lambda: _flashinfer(w)),
        _guard("megatron_te", lambda: _megatron(w)),
        _guard("sglang_triton", lambda: _sglang(w, False)),
        _guard("sglang_deterministic", lambda: _sglang(w, True)),
    ]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _embedded(x: torch.Tensor, size: int):
    """Probe rows first in a batch of ``size``, and the same probe rows last."""
    first = x[:size]
    last = x[:size].clone()
    last[size - PROBE :] = x[:PROBE]
    return first, last


def _rows_bi(fn, x) -> dict[str, bool]:
    alone = fn(x[:PROBE])
    result = {}
    for size in BI_SIZES:
        first, last = _embedded(x, size)
        a, b = fn(first), fn(last)
        if isinstance(alone, tuple):
            ok = all(
                _bitwise(p[:PROBE], q) and _bitwise(r[size - PROBE :], q)
                for p, r, q in zip(a, b, alone)
            )
        else:
            ok = _bitwise(a[:PROBE], alone) and _bitwise(b[size - PROBE :], alone)
        result[str(size)] = bool(ok)
    return result


def has_backward(cand) -> bool:
    return "graph" in cand or "grads" in cand


def cand_grads(cand, rows, dy_rows):
    """``(dx, d gate_up, d down)`` of one backward."""
    if "grads" in cand:
        return cand["grads"](rows, dy_rows)
    y, leaves = cand["graph"](rows)
    return torch.autograd.grad(y, leaves, dy_rows)


def _grad_bi(cand, x) -> dict[str, Any]:
    """dx of the probe rows, and dW with zero-gradient rows appended."""
    g = torch.Generator(device="cuda").manual_seed(7)
    dy = torch.randn(x.shape, device="cuda", generator=g).to(torch.bfloat16)

    def grads(rows, dy_rows):
        return cand_grads(cand, rows, dy_rows)

    alone = grads(x[:PROBE], dy[:PROBE])
    dx, dw = {}, {}
    for size in BI_SIZES:
        full = grads(x[:size], dy[:size])
        dx[str(size)] = bool(_bitwise(full[0][:PROBE], alone[0]))
        padded = dy[:size].clone()
        padded[PROBE:] = 0
        masked = grads(x[:size], padded)
        dw[str(size)] = bool(all(_bitwise(m, a) for m, a in zip(masked[1:], alone[1:])))
    return {"dx_rows_bitwise": dx, "dweight_zero_rows_bitwise": dw}


def _accuracy(cand, x, w) -> dict[str, Any]:
    ref, ref_ids = fp64_reference(x, w)
    ids, _ = cand["route"](x)
    same = ids.long().sort(-1).values == ref_ids.sort(-1).values
    out = cand["fwd"](x)
    return {
        "rel_l2_vs_fp64": _rel(out, ref),
        "max_abs_vs_fp64": float((out.double() - ref).abs().max()),
        "tokens_with_different_expert_set": int((~same.all(-1)).sum()),
        "tokens": int(x.shape[0]),
    }


def _one(cand, x, w) -> dict[str, Any]:
    if "unavailable" in cand:
        return cand
    out = {"key": cand["key"], "name": cand["name"], "source": cand["source"]}
    try:
        out["route_rows_bitwise"] = _rows_bi(cand["route"], x)
        out["output_rows_bitwise"] = _rows_bi(cand["fwd"], x)
        out["repeatable"] = bool(_bitwise(cand["fwd"](x[:256]), cand["fwd"](x[:256])))
        if has_backward(cand):
            out.update(_grad_bi(cand, x))
        out["accuracy"] = _accuracy(cand, x[:256], w)
    except Exception as exc:  # noqa: BLE001 - a crash is a result, not a harness failure
        out["failed"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out["batch_invariant"] = bool(
        "failed" not in out
        and all(out["route_rows_bitwise"].values())
        and all(out["output_rows_bitwise"].values())
        and all(out.get("dx_rows_bitwise", {"": True}).values())
        and all(out.get("dweight_zero_rows_bitwise", {"": True}).values())
    )
    return out


def _latency(cands, tokens) -> dict[str, Any]:
    ready = [c for c in cands if "unavailable" not in c]
    forward = {}
    for count in tokens:
        x = make_tokens(count, seed=100 + count)
        calls = {}
        for cand in ready:
            try:
                cand["fwd"](x)
                calls[cand["key"]] = lambda c=cand: c["fwd"](x)
            except Exception:  # noqa: BLE001 - recorded by the BI section
                continue
        forward[str(count)] = time_interleaved(calls)
    backward = {}
    for count in BACKWARD_TOKENS:
        x = make_tokens(count, seed=200 + count)
        dy = torch.randn_like(x)
        calls = {}
        for cand in ready:
            if not has_backward(cand):
                continue

            def step(c=cand):
                cand_grads(c, x, dy)

            calls[cand["key"]] = step
        backward[str(count)] = time_interleaved(calls, warmup=2, iters=10)
    return {"forward_us": forward, "forward_plus_backward_us": backward}


def moe_report(only=None) -> dict[str, Any]:
    """The report for every candidate, or only for the keys in ``only``.

    Some candidates need another Python environment (SGLang's pinned torch ABI,
    Megatron's training stack); run each environment with ``only`` and merge.
    """
    torch.manual_seed(0)
    w = make_weights()
    x = make_tokens(max(BI_SIZES), seed=1)
    keys = [
        "rl_kernel_cuda",
        "hf_transformers",
        "vllm_bi0",
        "vllm_bi1",
        "flashinfer_cutlass",
        "megatron_te",
        "sglang_triton",
        "sglang_deterministic",
    ]
    if only is not None and set(only) - set(keys):
        raise ValueError(f"Unknown candidates: {sorted(set(only) - set(keys))}")
    cands = [c for c in candidates(w) if only is None or c["key"] in only]
    return {
        "op": "qwen3_next_routed_moe",
        "shape": {"hidden": HIDDEN, "experts": EXPERTS, "top_k": TOPK, "expert_width": WIDTH},
        "probe_rows": PROBE,
        "candidates": [_one(c, x, w) for c in cands],
        "latency": _latency(cands, PERF_TOKENS),
    }
