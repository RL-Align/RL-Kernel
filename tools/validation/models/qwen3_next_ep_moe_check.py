#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""EP4 MoE gate on the official checkpoint's layer-0 weights; four GPUs, torchrun.

    TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 \\
        tools/validation/models/qwen3_next_ep_moe_check.py \\
        --checkpoint <Qwen3-Next-80B-A3B-Instruct> --output <dir>

Each rank writes ``rank-<r>.json``. The gate requires, bitwise:

* every one of the 512 experts loaded and exported by exactly one rank, equal to
  the checkpoint (no missing or duplicate expert);
* identical routes on every rank;
* the EP4 routed output, its input gradient and every owned expert's weight
  gradients equal to the same experts evaluated on one GPU with all 512
  experts at full width (``shared_moe``, EP1), for 8, 64, 256 and 1024 tokens;
* full batch == concatenated chunks, reordered tokens == reordered output;
* a batch in which one rank owns no route (empty-rank participation);
* the training forward (autograd on) == the no-grad forward, on every rank.

It records the EP4 block output's digest and its largest difference from the
TP4 (expert tensor parallel 4) block, which is expected to be non-zero.
"""

import hashlib
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from rl_engine.models.qwen3_next.qwen3_next_ep import (  # noqa: E402
    EP_PROFILE_ID,
    LOCAL_EXPERTS,
    EP4MoE,
    _EPRoutedExperts,
)
from rl_engine.models.qwen3_next.qwen3_next_forward import (  # noqa: E402
    _RoutedExperts,
    shared_router,
    stable_top10_routes,
)
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import TP4MoE, moe_hf_shapes  # noqa: E402
from rl_engine.validation.common.tensor_identity import (  # noqa: E402
    assert_tensor_bitwise_equal as exact,
)
from tools.validation.models.qwen3_next_gate_common import (  # noqa: E402
    CheckpointWeights,
    gather,
    run_gate,
)

TOKEN_COUNTS = (8, 64, 256, 1024)


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def _full_width_experts(weights, device):
    """All 512 experts at full width on one GPU (EP1 reference)."""
    gate_up = torch.empty(512, 1024, 2048, device=device, dtype=torch.bfloat16)
    down = torch.empty(512, 2048, 512, device=device, dtype=torch.bfloat16)
    for expert in range(512):
        gate_up[expert, :512].copy_(weights[f"experts.{expert}.gate_proj.weight"])
        gate_up[expert, 512:].copy_(weights[f"experts.{expert}.up_proj.weight"])
        down[expert].copy_(weights[f"experts.{expert}.down_proj.weight"])
    return gate_up, down


def _routed_pair(module, gate_up_ref, down_ref, hidden, indices, dy):
    """EP4 rows/grads and the EP1 reference on the same routes."""
    group, rank = module.group, module.rank
    x = hidden.detach().requires_grad_(True)
    rows = _EPRoutedExperts.apply(
        x, module.experts.gate_up, module.experts.down, indices, rank, group
    )
    dx, dgu, ddown = torch.autograd.grad(rows, (x, module.experts.gate_up, module.experts.down), dy)
    xr = hidden.detach().requires_grad_(True)
    gu = gate_up_ref.detach().requires_grad_(True)
    dn = down_ref.detach().requires_grad_(True)
    ref = _RoutedExperts.apply(xr, gu, dn, indices)
    rdx, rdgu, rddown = torch.autograd.grad(ref, (xr, gu, dn), dy)
    owned = slice(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
    return (rows, dx, dgu, ddown), (ref, rdx, rdgu[owned], rddown[owned])


def ep_gate(checkpoint, generator):
    group = dist.group.WORLD
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    module = EP4MoE(group=group, device=device)
    tp = TP4MoE(group=group, device=device)
    cases = []
    with ExitStack() as stack:
        weights = CheckpointWeights(checkpoint, "model.layers.0.mlp.", moe_hf_shapes(), stack)
        module.load_hf_weights(weights)
        tp.load_hf_weights(weights)
        exported = module.export_local_hf_weights()
        owners = {}
        for peer, names in enumerate(gather_names(sorted(exported))):
            for name in names:
                if name.startswith("experts."):
                    owners.setdefault(name, []).append(peer)
        expert_names = [n for n in moe_hf_shapes() if n.startswith("experts.")]
        if sorted(owners) != sorted(expert_names) or any(len(v) != 1 for v in owners.values()):
            raise AssertionError("An expert is missing or held by more than one rank")
        for name, value in exported.items():
            if name.startswith("experts."):
                exact(value, weights[name], name=f"EP4 export {name}")
        cases.append("hf-load-export-each-expert-once")
        gate_up_ref, down_ref = _full_width_experts(weights, device)
    digests, ep_vs_tp = {}, {}
    for length in TOKEN_COUNTS:
        hidden = torch.randn(length, 2048, device=device, dtype=torch.bfloat16, generator=generator)
        hidden = hidden * 0.1
        for peer in gather(hidden):
            exact(peer, hidden, name="replicated input")
        routes = stable_top10_routes(shared_router(hidden, module.gate.weight))
        for peer in gather(routes.indices):
            exact(peer, routes.indices, name=f"routes across ranks {length}")
        dy = torch.randn(length, 10, 2048, device=device, dtype=torch.bfloat16, generator=generator)
        ep, ref = _routed_pair(module, gate_up_ref, down_ref, hidden, routes.indices, dy)
        for name, a, b in zip(("rows", "dx", "dgate_up", "ddown"), ep, ref):
            exact(a, b, name=f"EP4 vs EP1 {name} {length}")
        with torch.no_grad():
            whole = module(hidden)
            cut = min(65, length - 1)
            exact(torch.cat((module(hidden[:cut]), module(hidden[cut:]))), whole, name="chunks")
            order = torch.randperm(length, device=device, generator=generator)
            exact(module(hidden[order]), whole[order], name="reorder")
            exact(module(hidden[:1]), whole[:1], name="first row alone")
            ep_vs_tp[str(length)] = float((whole.float() - tp(hidden).float()).abs().max())
        trained = module(hidden.detach().requires_grad_(True))
        exact(trained.detach(), whole, name="training forward == no-grad")
        for peer in gather(whole):
            exact(peer, whole, name="EP4 output replicated")
        digests[str(length)] = digest(whole)
        cases.append(f"ep4-vs-ep1-chunk-reorder-train-{length}")
    # Empty-rank participation: routes that avoid rank 3's experts entirely.
    hidden = torch.randn(16, 2048, device=device, dtype=torch.bfloat16, generator=generator) * 0.1
    indices = torch.stack(
        [torch.randperm(384, device=device, generator=generator)[:10] for _ in range(16)]
    )
    dy = torch.randn(16, 10, 2048, device=device, dtype=torch.bfloat16, generator=generator)
    ep, ref = _routed_pair(module, gate_up_ref, down_ref, hidden, indices, dy)
    for name, a, b in zip(("rows", "dx", "dgate_up", "ddown"), ep, ref):
        exact(a, b, name=f"empty rank 3 {name}")
    cases.append("empty-rank-participation")
    return {
        "profile": EP_PROFILE_ID,
        "ep": {
            "cases": cases,
            "output_sha256": digests,
            "max_abs_vs_tp4": ep_vs_tp,
            "local_experts": [rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS],
        },
    }


def gather_names(names):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, names)
    return gathered


if __name__ == "__main__":
    run_gate(__doc__, "ep4_moe_block_layer0", ep_gate)
