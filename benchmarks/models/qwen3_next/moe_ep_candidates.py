# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prior art for the Qwen3-Next expert-parallel routed MoE (layer 0, real weights).

Every candidate receives the same BF16 hidden states, replicated on every rank,
and the official layer-0 router and 512 experts. At EP4 rank r owns experts
``[128 r, 128 r + 128)`` at full width (expert tensor parallel 1); at EP1 one
GPU holds all 512. The shared expert is the same TP4 path for every candidate
and is left out, so only routing, expert compute and the cross-rank combine
differ.

Candidates:

* ``rl_kernel_ep`` -- RL-Kernel's EP4 path (``qwen3_next_ep``): replicated FP32
  router and stable top-10, local dispatch, byte-only all-gather of route rows,
  FP32 combine in route order on every rank. EP1 is ``shared_moe``.
* ``vllm_fused_ep`` -- vLLM 0.30's expert-parallel ``FusedMoE`` path without
  data parallelism: BF16 gate, ``fused_topk`` (softmax, renormalised top-10),
  ``fused_experts`` with the rank's ``expert_map``, then the all-reduce that
  vLLM applies over the TP/EP group (here ``torch.distributed.all_reduce`` on
  NCCL). EP1 is the same kernel with all experts and no all-reduce.
* ``vllm_fused_ep_bi`` -- the same with ``VLLM_BATCH_INVARIANT=1`` (fixed
  ``fused_moe`` tile config, ``SPLIT_K`` 1), vLLM's batch-invariant mode.
* ``megatron_alltoall`` / ``megatron_allgather`` -- Megatron-core ``MoELayer``
  with ``SequentialMLP`` experts, FP32 router (softmax over the top-10 logits,
  which equals renormalised top-10 probabilities), TP4 x EP4 x ETP1 without
  sequence parallelism, and the ``alltoall`` or ``allgather`` token dispatcher.

For each candidate a launch records the output digest, whether the first rows
computed alone equal the same rows inside the full batch, whether a second
call repeats the bits, the largest error against an FP64 evaluation of the same
routes and a median latency.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import statistics
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

EXPERTS, LOCAL, HIDDEN, INTERMEDIATE, TOP_K = 512, 128, 2048, 512, 10
ALONE_ROWS = 16
CANDIDATES = (
    "rl_kernel_ep",
    "vllm_fused_ep",
    "vllm_fused_ep_bi",
    "megatron_alltoall",
    "megatron_allgather",
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


def _rl_kernel(router, gate_up, down, ep, group):
    from rl_engine.models.qwen3_next.qwen3_next_ep import _EPRoutedExperts
    from rl_engine.models.qwen3_next.qwen3_next_forward import (
        combine_routes,
        shared_moe,
        shared_router,
        stable_top10_routes,
    )

    if ep == 1:
        return lambda x: shared_moe(x, router, gate_up, down)[0]
    rank = dist.get_rank(group)

    def run(x):
        routes = stable_top10_routes(shared_router(x, router))
        rows = _EPRoutedExperts.apply(x, gate_up, down, routes.indices, rank, group)
        return combine_routes(rows, routes.weights)

    return run


@contextlib.contextmanager
def _vllm_batch_invariant(enabled: bool):
    """Set ``VLLM_BATCH_INVARIANT`` for one call; vLLM reads it per call but caches
    the tuned tile configs, so the cache is cleared on both sides of the switch."""
    from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_configs

    previous = os.environ.get("VLLM_BATCH_INVARIANT")
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if enabled else "0"
    get_moe_configs.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VLLM_BATCH_INVARIANT")
        else:
            os.environ["VLLM_BATCH_INVARIANT"] = previous
        get_moe_configs.cache_clear()


def _vllm(router, gate_up, down, ep, group, batch_invariant=False):
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    mapping = None
    if ep > 1:
        from rl_engine.models.qwen3_next.qwen3_next_ep import expert_map

        mapping = expert_map(dist.get_rank(group), gate_up.device)

    def run(x):
        logits = F.linear(x, router)
        with _vllm_batch_invariant(batch_invariant):
            weights, ids, _ = fused_topk(x, logits, TOP_K, renormalize=True)
            out = fused_experts(
                x, gate_up, down, weights, ids, global_num_experts=EXPERTS, expert_map=mapping
            )
        if ep > 1:
            dist.all_reduce(out, group=group)
        return out

    return run


def _megatron(router, gate_up, down, ep, group, dispatcher):
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
    with torch.no_grad():
        layer.router.weight.copy_(router)
        experts = layer.experts.local_experts
        if len(experts) != gate_up.shape[0]:
            raise RuntimeError("Megatron built an unexpected number of local experts")
        for local, expert in enumerate(experts):
            expert.linear_fc1.weight.copy_(gate_up[local])
            expert.linear_fc2.weight.copy_(down[local])

    def run(x):
        out, _ = layer(x[:, None, :])
        return out[:, 0, :]

    return run


def build(name: str, router, gate_up, down, *, ep: int, group) -> Callable:
    if name == "rl_kernel_ep":
        return _rl_kernel(router, gate_up, down, ep, group)
    if name == "vllm_fused_ep":
        return _vllm(router, gate_up, down, ep, group)
    if name == "vllm_fused_ep_bi":
        return _vllm(router, gate_up, down, ep, group, batch_invariant=True)
    if name == "megatron_alltoall":
        return _megatron(router, gate_up, down, ep, group, "alltoall")
    if name == "megatron_allgather":
        return _megatron(router, gate_up, down, ep, group, "allgather")
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


def _latency_us(fn, x, repeats=20) -> float:
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def candidate_report(x, router, gate_up, down, *, ep: int, group, reference=None) -> dict[str, Any]:
    """Run every candidate on the same inputs; ``reference`` is FP64 [T, H] or None."""
    report: dict[str, Any] = {}
    outputs = {}
    for name in CANDIDATES:
        try:
            fn = build(name, router, gate_up, down, ep=ep, group=group)
        except Exception as error:  # an unavailable stack is recorded, not fatal
            report[name] = {"unavailable": f"{type(error).__name__}: {error}"}
            continue
        with torch.no_grad():
            out = fn(x)
            again = fn(x)
            alone = fn(x[:ALONE_ROWS].clone())
            latency = _latency_us(fn, x)
        outputs[name] = out
        row = {
            "output_sha256": digest(out),
            "repeat_bitwise": torch.equal(out, again),
            "batch_bitwise": torch.equal(alone, out[:ALONE_ROWS]),
            "latency_us": latency,
        }
        if reference is not None:
            row["max_abs_vs_fp64"] = float((out.double() - reference).abs().max())
        report[name] = row
    for name, out in outputs.items():
        report[name]["max_abs_vs"] = {
            other: float((out.float() - peer.float()).abs().max())
            for other, peer in outputs.items()
            if other != name
        }
    return report
