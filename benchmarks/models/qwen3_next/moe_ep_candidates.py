# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prior art for the Qwen3-Next expert-parallel routed MoE (layer 0, real weights).

Every candidate receives the same BF16 hidden states, replicated on every rank,
and the official layer-0 router and 512 experts. At EP4 rank r owns experts
``[128 r, 128 r + 128)`` at full width (expert tensor parallel 1); at EP1 one
GPU holds all 512. The shared expert is the same TP4 path for every candidate
and is left out, so only routing, expert compute and the cross-rank combine
differ.

Candidates (``CANDIDATES``):

* ``rl_kernel_ep`` -- RL-Kernel's EP4 path (``qwen3_next_ep``): replicated FP32
  router and stable top-10, local dispatch, byte-only all-gather of route rows,
  FP32 combine in route order on every rank. EP1 is ``shared_moe``.
* ``vllm_ep`` -- vLLM 0.30's expert-parallel ``FusedMoE`` path without data
  parallelism: BF16 gate, ``fused_topk`` (softmax, renormalised top-10),
  ``fused_experts`` with the rank's ``expert_map``, then
  ``tensor_model_parallel_all_reduce`` -- vLLM's own all-reduce dispatch
  (FlashInfer, custom, symmetric-memory or PyNCCL, chosen per message).
* ``vllm_ep_bi`` -- the same with ``VLLM_BATCH_INVARIANT=1``.
* ``vllm_ep_nccl`` -- the same expert kernel, combined with
  ``torch.distributed.all_reduce`` on NCCL.
* ``flashinfer_cutlass_ep`` -- FlashInfer's ``cutlass_fused_moe`` (CUTLASS
  grouped GEMM) with ``ep_size``/``ep_rank`` on vLLM's routes, combined with
  vLLM's all-reduce, as vLLM and SGLang integrate it.
* ``megatron_alltoall`` / ``megatron_allgather`` -- Megatron-core ``MoELayer``
  with ``SequentialMLP`` experts, FP32 router (softmax over the top-10 logits,
  which equals renormalised top-10 probabilities), TP4 x EP4 x ETP1 without
  sequence parallelism, and the ``alltoall`` or ``allgather`` token dispatcher.
* ``megatron_alltoall_det`` / ``megatron_allgather_det`` -- the same under
  ``torch.use_deterministic_algorithms(True)``, which makes Megatron's token
  unpermute use ``index_add_`` instead of ``scatter_add_``.

Forward: output digest, repeat, a batch-invariance grid (probe rows inside
batches of several sizes at the first, middle and last position), error
against an FP64 evaluation, latency over several token counts. Backward, for
the candidates that train (RL-Kernel and Megatron): input and owned-expert
weight gradient digests, repeat, the same batch grid on the input gradient and
latency.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import statistics
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

EXPERTS, LOCAL, HIDDEN, INTERMEDIATE, TOP_K = 512, 128, 2048, 512, 10
PROBE = 8
BATCHES = (8, 64, 256, 1024)
LATENCY_TOKENS = (64, 256, 1024, 4096)
CANDIDATES = (
    "rl_kernel_ep",
    "vllm_ep",
    "vllm_ep_bi",
    "vllm_ep_nccl",
    "flashinfer_cutlass_ep",
    "megatron_alltoall",
    "megatron_allgather",
    "megatron_alltoall_det",
    "megatron_allgather_det",
)


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def expert_weights(weights, experts) -> tuple[torch.Tensor, torch.Tensor]:
    """``[E, 1024, 2048]`` (gate then up) and ``[E, 2048, 512]`` BF16 for ``experts``."""
    gate_up = torch.empty(len(experts), 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    down = torch.empty(len(experts), HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
    for local, expert in enumerate(experts):
        gate_up[local, :INTERMEDIATE].copy_(weights[f"experts.{expert}.gate_proj.weight"])
        gate_up[local, INTERMEDIATE:].copy_(weights[f"experts.{expert}.up_proj.weight"])
        down[local].copy_(weights[f"experts.{expert}.down_proj.weight"])
    return gate_up, down


@dataclass
class Candidate:
    forward: Callable[[torch.Tensor], torch.Tensor]
    # (x, dy) -> (dx, d_gate_up_owned, d_down_owned); None for inference-only stacks
    backward: Callable[[torch.Tensor, torch.Tensor], tuple] | None = None


def _rl_kernel(router, gate_up, down, ep, group):
    from rl_engine.models.qwen3_next.qwen3_next_ep import _EPRoutedExperts
    from rl_engine.models.qwen3_next.qwen3_next_forward import (
        combine_routes,
        shared_moe,
        shared_router,
        stable_top10_routes,
    )

    rank = dist.get_rank(group) if ep > 1 else 0

    def run(x, w13, w2):
        if ep == 1:
            return shared_moe(x, router, w13, w2)[0]
        routes = stable_top10_routes(shared_router(x, router))
        rows = _EPRoutedExperts.apply(x, w13, w2, routes.indices, rank, group)
        return combine_routes(rows, routes.weights)

    def backward(x, dy):
        x = x.detach().requires_grad_(True)
        w13 = gate_up.detach().requires_grad_(True)
        w2 = down.detach().requires_grad_(True)
        dx, dw13, dw2 = torch.autograd.grad(run(x, w13, w2), (x, w13, w2), dy)
        return dx, dw13, dw2

    return Candidate(lambda x: run(x, gate_up, down), backward)


@contextlib.contextmanager
def _vllm_batch_invariant(enabled: bool):
    """Run with ``VLLM_BATCH_INVARIANT`` set to ``enabled``.

    vLLM reads the variable on every call but caches the tile config per
    shape, so the cache is cleared only when the setting actually changes;
    clearing it on every call would add a config-file lookup to each timed call.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_configs

    value = "1" if enabled else "0"
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != value:
        os.environ["VLLM_BATCH_INVARIANT"] = value
        get_moe_configs.cache_clear()
    yield


def _vllm_routes(x, router):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    weights, ids, _ = fused_topk(x, F.linear(x, router), TOP_K, renormalize=True)
    return weights, ids


def _combine(out, ep, group, backend):
    if ep == 1:
        return out
    if backend == "nccl":
        dist.all_reduce(out, group=group)
        return out
    from vllm.distributed import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(out)


def _vllm(router, gate_up, down, ep, group, *, batch_invariant=False, backend="vllm"):
    from vllm.model_executor.layers.fused_moe import fused_experts

    mapping = None
    if ep > 1:
        from rl_engine.models.qwen3_next.qwen3_next_ep import expert_map

        mapping = expert_map(dist.get_rank(group), gate_up.device)

    def run(x):
        with _vllm_batch_invariant(batch_invariant):
            weights, ids = _vllm_routes(x, router)
            out = fused_experts(
                x, gate_up, down, weights, ids, global_num_experts=EXPERTS, expert_map=mapping
            )
        return _combine(out, ep, group, backend)

    return Candidate(run)


def _flashinfer(router, gate_up, down, ep, group):
    from flashinfer.fused_moe import cutlass_fused_moe

    rank = dist.get_rank(group) if ep > 1 else 0
    # FlashInfer's SwiGLU expects [up; gate] in fc1.
    fc1 = torch.cat((gate_up[:, INTERMEDIATE:], gate_up[:, :INTERMEDIATE]), dim=1).contiguous()
    fc2 = down.contiguous()

    def run(x):
        with _vllm_batch_invariant(False):
            weights, ids = _vllm_routes(x, router)
        out = cutlass_fused_moe(
            x,
            ids.int(),
            weights.float(),
            fc1,
            fc2,
            torch.bfloat16,
            quant_scales=[],
            ep_size=ep,
            ep_rank=rank,
        )
        out = out[0] if isinstance(out, (list, tuple)) else out
        return _combine(out, ep, group, "vllm")

    return Candidate(run)


def _megatron(router, gate_up, down, ep, dispatcher, deterministic=False):
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=16,
        ffn_hidden_size=INTERMEDIATE,
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=INTERMEDIATE,
        moe_router_topk=TOP_K,
        moe_router_pre_softmax=False,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type=dispatcher,
        moe_grouped_gemm=False,
        moe_permute_fusion=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=ep,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        perform_initialization=False,
    )
    spec = get_moe_module_spec(use_te=False, num_experts=EXPERTS, moe_grouped_gemm=False)
    layer = MoELayer(config, submodules=spec.submodules).cuda().eval()
    experts = layer.experts.local_experts
    if len(experts) != gate_up.shape[0]:
        raise RuntimeError("Megatron built an unexpected number of local experts")
    with torch.no_grad():
        layer.router.weight.copy_(router)
        for local, expert in enumerate(experts):
            expert.linear_fc1.weight.copy_(gate_up[local])
            expert.linear_fc2.weight.copy_(down[local])

    @contextlib.contextmanager
    def mode():
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(deterministic)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(previous)

    def run(x):
        with mode():
            out, _ = layer(x[:, None, :])
        return out[:, 0, :]

    def backward(x, dy):
        layer.zero_grad(set_to_none=True)
        x = x.detach().requires_grad_(True)
        with mode():
            out, _ = layer(x[:, None, :])
            out[:, 0, :].backward(dy)
        dw13 = torch.stack([e.linear_fc1.weight.grad for e in experts])
        dw2 = torch.stack([e.linear_fc2.weight.grad for e in experts])
        return x.grad, dw13, dw2

    return Candidate(run, backward)


def build(name: str, router, gate_up, down, *, ep: int, group) -> Candidate:
    if name == "rl_kernel_ep":
        return _rl_kernel(router, gate_up, down, ep, group)
    if name == "vllm_ep":
        return _vllm(router, gate_up, down, ep, group)
    if name == "vllm_ep_bi":
        return _vllm(router, gate_up, down, ep, group, batch_invariant=True)
    if name == "vllm_ep_nccl":
        return _vllm(router, gate_up, down, ep, group, backend="nccl")
    if name == "flashinfer_cutlass_ep":
        return _flashinfer(router, gate_up, down, ep, group)
    if name.startswith("megatron_"):
        dispatcher = name.split("_")[1]
        return _megatron(router, gate_up, down, ep, dispatcher, name.endswith("_det"))
    raise KeyError(name)


def fp64_reference(x, router, all_gate_up, all_down) -> torch.Tensor:
    """FP64 router, renormalised top-10 and SwiGLU experts; FP64 [T, H]."""
    xd = x.double()
    probs = torch.softmax(xd @ router.double().t(), dim=-1)
    weights, indices = probs.topk(TOP_K, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    out = torch.zeros_like(xd)
    for expert in indices.unique().tolist():
        token, slot = torch.nonzero(indices == expert, as_tuple=True)
        gate_up = all_gate_up[expert].to(x.device, torch.float64)
        down = all_down[expert].to(x.device, torch.float64)
        gate, up = (xd[token] @ gate_up.t()).chunk(2, dim=-1)
        rows = (F.silu(gate) * up) @ down.t()
        out.index_add_(0, token, rows * weights[token, slot, None])
    return out


def _median_us(fn, repeats=20) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def _batch_grid(fn, probe, filler) -> dict[str, bool]:
    """Probe rows alone vs inside batches of each size, first / middle / last."""
    alone = fn(probe)
    grid = {}
    for size in BATCHES:
        for position in sorted({0, (size - PROBE) // 2, size - PROBE}):
            rows = torch.cat((filler[:position], probe, filler[position : size - PROBE]))
            grid[f"{size}@{position}"] = torch.equal(fn(rows)[position : position + PROBE], alone)
    return grid


def candidate_report(
    x_all, dy_all, router, gate_up, down, *, ep: int, group, reference
) -> dict[str, Any]:
    """Run every candidate; ``x_all``/``dy_all`` hold ``max(LATENCY_TOKENS)`` rows.

    The first 1024 rows are the measured batch; rows 1024.. are fillers for
    the batch grid; the probe is ``x_all[:PROBE]``.
    """
    names = os.environ.get("MOE_EP_CANDIDATES")
    names = names.split(",") if names else CANDIDATES
    x, dy = x_all[:1024], dy_all[:1024]
    probe, filler = x_all[:PROBE], x_all[1024:]
    dprobe, dfiller = dy_all[:PROBE], dy_all[1024:]
    report: dict[str, Any] = {}
    outputs = {}
    for name in names:
        try:
            cand = build(name, router, gate_up, down, ep=ep, group=group)
            with torch.no_grad():
                out = cand.forward(x)
        except Exception as error:  # an unavailable stack is recorded, not fatal
            report[name] = {"unavailable": f"{type(error).__name__}: {error}"}
            continue
        with torch.no_grad():
            again = cand.forward(x)
            grid = _batch_grid(cand.forward, probe, filler)
            latency = {
                str(n): _median_us(lambda n=n: cand.forward(x_all[:n])) for n in LATENCY_TOKENS
            }
        outputs[name] = out
        error = (out.double() - reference).abs()
        row = {
            "output_sha256": digest(out),
            "repeat_bitwise": torch.equal(out, again),
            "batch_grid": grid,
            "batch_bitwise": all(grid.values()),
            "max_abs_vs_fp64": float(error.max()),
            "mean_abs_vs_fp64": float(error.mean()),
            "latency_us": latency,
        }
        if cand.backward is not None:
            dx, dw13, dw2 = cand.backward(x, dy)
            dx2, dw13b, dw2b = cand.backward(x, dy)
            alone = cand.backward(probe, dprobe)[0]
            bgrid = {}
            for size in BATCHES:
                for position in sorted({0, (size - PROBE) // 2, size - PROBE}):
                    rows = torch.cat((filler[:position], probe, filler[position : size - PROBE]))
                    grads = torch.cat(
                        (dfiller[:position], dprobe, dfiller[position : size - PROBE])
                    )
                    got = cand.backward(rows, grads)[0][position : position + PROBE]
                    bgrid[f"{size}@{position}"] = torch.equal(got, alone)
            row["backward"] = {
                "dx_sha256": digest(dx),
                "d_gate_up_sha256": digest(dw13),
                "d_down_sha256": digest(dw2),
                "d_gate_up_slice_sha256": (
                    [digest(dw13[r * LOCAL : (r + 1) * LOCAL]) for r in range(4)]
                    if ep == 1
                    else None
                ),
                "d_down_slice_sha256": (
                    [digest(dw2[r * LOCAL : (r + 1) * LOCAL]) for r in range(4)]
                    if ep == 1
                    else None
                ),
                "repeat_bitwise": torch.equal(dx, dx2)
                and torch.equal(dw13, dw13b)
                and torch.equal(dw2, dw2b),
                "batch_grid": bgrid,
                "batch_bitwise": all(bgrid.values()),
                "latency_us": _median_us(lambda: cand.backward(x, dy), repeats=5),
            }
        report[name] = row
    for name, out in outputs.items():
        report[name]["max_abs_vs"] = {
            other: float((out.float() - peer.float()).abs().max())
            for other, peer in outputs.items()
            if other != name
        }
    return report
