# SPDX-License-Identifier: Apache-2.0
"""TP-invariant shared expert: actor ops on I/TP and H/TP slices plus exact all-gathers.

At TP>1 every rank returns the full [M, H] BF16 shared output, bitwise equal to the TP1 actor
(`_GateUp(fp8(w1), fp8(w3))` -> activation -> `fp8(w2)`, all `CheckpointFP8Linear`). Each output
element keeps its full-K accumulation inside one rank's kernel; ranks only exchange finished
BF16 columns, so no cross-rank floating-point reduction exists.
"""

import json
from pathlib import Path

import torch

# N tile of the fixed actor CK kernel (shared_down_fp8.KERNEL: 256x16x64x256 = block/M/N/K).
_CK_N_TILE = 64
_CK_K_TILE = 256


def shared_expert_partition(intermediate, hidden, tp, rank):
    """Original-checkpoint row ranges owned by `rank`; fail closed on any misalignment.

    Returns dict with gate/up rows (w1/w3: [I, H]) and down rows (w2: [H, I]), each as
    (row_start, row_stop, scale_row_start, scale_row_stop).
    """
    if tp < 1 or not 0 <= rank < tp:
        raise ValueError("invalid TP rank")
    if intermediate % tp or hidden % tp:
        raise ValueError("shared expert I and H must split evenly across TP")
    i_rows, h_rows = intermediate // tp, hidden // tp
    for rows in (i_rows, h_rows):
        if rows % 128 or rows % _CK_N_TILE:
            raise ValueError("shared expert slice rows must align with block128 and the CK N tile")
    if hidden % 128 or hidden % _CK_K_TILE or intermediate % 128 or intermediate % _CK_K_TILE:
        raise ValueError("shared expert K must align with block128 and the CK K tile")
    i0, h0 = rank * i_rows, rank * h_rows
    return {
        "gate_up": (i0, i0 + i_rows, i0 // 128, (i0 + i_rows) // 128),
        "down": (h0, h0 + h_rows, h0 // 128, (h0 + h_rows) // 128),
    }


def _checkpoint_file(model_path, key):
    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        return root / json.loads(index.read_text())["weight_map"][key]
    paths = list(root.glob("*.safetensors"))
    if len(paths) != 1:
        raise ValueError("shared expert importer requires an index or one safetensors file")
    return paths[0]


def build_shared_expert_slices(model_path, layer_index, tp, rank, device):
    """Actor CheckpointFP8Linear modules on this rank's slices of the ORIGINAL tensors."""
    from safetensors import safe_open

    from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Linear

    prefix = f"layers.{layer_index}.ffn.shared_experts"
    handles = {}

    def slices(name):
        path = _checkpoint_file(model_path, f"{prefix}.{name}.weight")
        if path not in handles:
            handles[path] = safe_open(str(path), framework="pt", device="cpu")
        source = handles[path]
        return (
            source.get_slice(f"{prefix}.{name}.weight"),
            source.get_slice(f"{prefix}.{name}.scale"),
        )

    w1, s1 = slices("w1")
    w3, s3 = slices("w3")
    w2, s2 = slices("w2")
    intermediate, hidden = w1.get_shape()
    if w3.get_shape() != [intermediate, hidden] or w2.get_shape() != [hidden, intermediate]:
        raise ValueError("shared expert w1/w2/w3 geometry mismatch")
    part = shared_expert_partition(intermediate, hidden, tp, rank)

    def rows(weight, scale, span):
        r0, r1, c0, c1 = span
        n, k = weight.get_shape()
        if scale.get_shape() != [n // 128, k // 128]:
            raise ValueError("original block128 scale geometry differs")
        return CheckpointFP8Linear(weight[r0:r1].to(device), scale[c0:c1].to(device))

    gate = rows(w1, s1, part["gate_up"])
    up = rows(w3, s3, part["gate_up"])
    down = rows(w2, s2, part["down"])
    return gate, up, down, part, hidden


def bind_tp_exact_shared_expert(ffn, model_path, layer_index):
    """Replace the shared expert forward of one vLLM DeepseekV4MoE at TP>1 (instance-only)."""
    from rl_engine.integrations.engines.rollout.vllm.tp_exact import gather_last_dim, tp_info

    rank, size, _ = tp_info()
    if size <= 1:
        raise ValueError("TP-exact shared expert is for TP>1 only; TP1 keeps the native path")
    shared = ffn.shared_experts
    if shared is None:
        raise ValueError("MoE layer has no unfused shared expert (fused shared expert unsupported)")
    if getattr(shared, "_rlk_tp_exact", None):
        raise ValueError("shared expert already bound")
    device = shared.down_proj.weight.device
    gate, up, down, part, hidden = build_shared_expert_slices(
        model_path, layer_index, size, rank, device
    )
    # Keep the pinned vLLM activation instance (SiluAndMulWithClamp(swiglu_limit)): elementwise per
    # intermediate column over the local [gate | up] layout, the same op as the TP1 rollout/actor.
    act_fn = shared.act_fn

    @torch.no_grad()
    def forward(x):
        if x.dim() != 2 or x.shape[-1] != hidden or x.dtype != torch.bfloat16:
            raise ValueError("TP-exact shared expert expects [M, H] BF16 input")
        if x.shape[0] == 0:
            return torch.empty((0, hidden), device=x.device, dtype=x.dtype)
        x = x.contiguous()
        gate_up = torch.cat((gate(x)[0], up(x)[0]), dim=-1)
        activated = gather_last_dim(act_fn(gate_up).contiguous())
        return gather_last_dim(down(activated)[0].contiguous())

    # nn.Module.__call__ resolves self.forward, so the runner's SharedExperts._layer(...) call
    # (and any direct call) reaches this; the vLLM column/row-parallel linears never run.
    shared.forward = forward
    shared._rlk_tp_exact = dict(tp=size, rank=rank, **part)
    shared._rlk_tp_exact_modules = (gate, up, down)  # keep slices resident and inspectable
    return shared
