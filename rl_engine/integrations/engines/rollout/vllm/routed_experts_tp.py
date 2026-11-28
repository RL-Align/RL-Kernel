# SPDX-License-Identifier: Apache-2.0
"""TP-invariant rollout routed experts (EP=1, TP>1) built from the actor's own kernels.

vLLM at EP=1/TP>1 gives every rank all 256 experts with the intermediate dim split, so
the down projection is a K-split partial sum that would need a cross-rank float
reduction. This module instead shards the *output* columns of each actor projection and
moves data only with rank-ordered all-gathers:

    gate_up : packed_w4a16 over this rank's I/TP rows of gate and up (fused gate|up)
    act     : swiglu_limit_func (elementwise; gate_j pairs with up_j inside the slice)
    gather  : act slices -> full [rows, I] (exact concat)
    down    : packed_w4a16 over this rank's H/TP rows of w2, full-K accumulation, gammas
    combine : actor combine_rows on the [M, H/TP] slice (per-column elementwise)
    gather  : [M, H/TP] -> [M, H]

packed_w4a16 accumulates every output element over the full K in fixed 32-wide blocks
and is independent of N (a tile covers 32 output columns), so sliced results equal the
corresponding columns of the TP1 result bit for bit.
"""

import json
import os
from contextlib import ExitStack
from dataclasses import dataclass

import torch

NUM_EXPERTS = 256
_STATE = {"calls": 0, "fallbacks": 0}


def slice_bounds(total, size, rank, what="dimension"):
    """Contiguous [start, stop) of ``rank`` when ``total`` is split evenly over ``size``.

    Each slice must be a multiple of 32 so slice boundaries coincide with the
    kernel's 32-column output tiles.
    """
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid TP rank {rank} of {size}")
    if total % size or (total // size) % 32:
        raise ValueError(f"{what} {total} must split into multiples of 32 over TP{size}")
    width = total // size
    return rank * width, (rank + 1) * width


def expert_key(layer_index, expert, name):
    return f"layers.{layer_index}.ffn.experts.{expert}.{name}"


class SafetensorsSource:
    """Row-sliced reads of one-byte original expert tensors from a checkpoint dir."""

    def __init__(self, model_path, stack):
        from safetensors import safe_open

        self._safe_open = safe_open
        self._stack = stack
        self._root = model_path
        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as handle:
                self._files = json.load(handle)["weight_map"]
        else:
            self._files = None
        self._handles = {}

    def _handle(self, name):
        fname = "model.safetensors" if self._files is None else self._files[name]
        if fname not in self._handles:
            path = os.path.join(self._root, fname)
            self._handles[fname] = self._stack.enter_context(
                self._safe_open(path, framework="pt", device="cpu")
            )
        return self._handles[fname]

    def shape(self, name):
        return tuple(self._handle(name).get_slice(name).get_shape())

    def rows(self, name, start, stop):
        value = self._handle(name).get_slice(name)[start:stop]
        if value.element_size() != 1:
            raise ValueError(f"original one-byte expert code/scale required: {name}")
        return value.view(torch.uint8)


@dataclass
class TpExpertSlices:
    layer_index: int
    rank: int
    size: int
    gate_up_codes: torch.Tensor  # [E, 2*I/TP, K/2]  gate rows then up rows
    gate_up_scales: torch.Tensor  # [E, 2*I/TP, K/32]
    down_codes: torch.Tensor  # [E, H/TP, I/2]
    down_scales: torch.Tensor  # [E, H/TP, I/32]
    intermediate: int
    hidden: int


def load_tp_expert_slices(source, layer_index, rank, size, device, num_experts=NUM_EXPERTS):
    """Stack the original codes/scales of all experts, restricted to this rank's rows.

    ``source`` provides ``shape(name)`` and ``rows(name, start, stop) -> uint8 [rows, cols]``.
    gate_up = cat(w1 slice, w3 slice) along N per expert (same order as the actor's fused
    gate|up); down = w2 rows [rank*H/TP, (rank+1)*H/TP).
    """
    w1_shape = source.shape(expert_key(layer_index, 0, "w1.weight"))
    w2_shape = source.shape(expert_key(layer_index, 0, "w2.weight"))
    intermediate, hidden = w1_shape[0], w1_shape[1] * 2
    if w2_shape != (hidden, intermediate // 2):
        raise ValueError(f"original w1/w2 geometry mismatch: {w1_shape} vs {w2_shape}")
    i0, i1 = slice_bounds(intermediate, size, rank, "intermediate size")
    h0, h1 = slice_bounds(hidden, size, rank, "hidden size")

    def stack(parts):
        return torch.stack(parts).to(device).contiguous()

    gu_codes, gu_scales, dn_codes, dn_scales = [], [], [], []
    for e in range(num_experts):
        gu_codes.append(
            torch.cat(
                (
                    source.rows(expert_key(layer_index, e, "w1.weight"), i0, i1),
                    source.rows(expert_key(layer_index, e, "w3.weight"), i0, i1),
                )
            )
        )
        gu_scales.append(
            torch.cat(
                (
                    source.rows(expert_key(layer_index, e, "w1.scale"), i0, i1),
                    source.rows(expert_key(layer_index, e, "w3.scale"), i0, i1),
                )
            )
        )
        dn_codes.append(source.rows(expert_key(layer_index, e, "w2.weight"), h0, h1))
        dn_scales.append(source.rows(expert_key(layer_index, e, "w2.scale"), h0, h1))
    slices = TpExpertSlices(
        layer_index,
        rank,
        size,
        stack(gu_codes),
        stack(gu_scales),
        stack(dn_codes),
        stack(dn_scales),
        intermediate,
        hidden,
    )
    width_i, width_h = i1 - i0, h1 - h0
    expected = (
        (slices.gate_up_codes, (num_experts, 2 * width_i, hidden // 2)),
        (slices.gate_up_scales, (num_experts, 2 * width_i, hidden // 32)),
        (slices.down_codes, (num_experts, width_h, intermediate // 2)),
        (slices.down_scales, (num_experts, width_h, intermediate // 32)),
    )
    for tensor, shape in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != torch.uint8:
            raise ValueError(f"TP expert slice shape {tuple(tensor.shape)} != {shape}")
    return slices


def find_fused_experts(ffn):
    """The object vLLM passes as ``self`` to ``UnfusedOAITritonExperts.apply``."""
    return ffn.experts.routed_experts.quant_method.moe_kernel.fused_experts


def bind_tp_exact_routed_experts(ffn, model_path, layer_index):
    """Load this rank's original expert slices and attach them for the patched apply.

    Call once per layer at TP>1 after ``bind_rollout_fixed_routed_combine``.
    """
    from .routed_combine import bind_rollout_fixed_routed_combine
    from .tp_exact import tp_info

    rank, size, _ = tp_info()
    if size <= 1:
        raise RuntimeError("bind_tp_exact_routed_experts is for TP>1 only")
    bind_rollout_fixed_routed_combine()
    backend = find_fused_experts(ffn)
    from vllm.model_executor.layers.fused_moe.experts import gpt_oss_triton_kernels_moe as mod

    if not isinstance(backend, mod.UnfusedOAITritonExperts):
        raise RuntimeError(f"unsupported routed expert backend: {type(backend).__name__}")
    device = torch.device("cuda", torch.cuda.current_device())
    with ExitStack() as stack:
        slices = load_tp_expert_slices(
            SafetensorsSource(model_path, stack), layer_index, rank, size, device
        )
    backend._rlk_tp_experts = slices
    print(
        "RLK_ROLLOUT_TP_EXACT_ROUTED_EXPERTS_BOUND",
        "layer", layer_index, "rank", rank, "tp", size,
        "gate_up", tuple(slices.gate_up_codes.shape), "down", tuple(slices.down_codes.shape),
        flush=True,
    )
    return slices


def expert_offsets(experts, num_experts=NUM_EXPERTS):
    counts = torch.bincount(experts, minlength=num_experts)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int64)


def tp_exact_unfused_apply(backend, output, hidden_states, routing_data, activation, mod):
    """Replacement body for ``UnfusedOAITritonExperts.apply`` at TP>1 (no vLLM matmuls)."""
    from rl_engine.backends.rocm.moe.packed_w4a16 import packed_w4a16
    from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func

    from .routed_combine import fixed_combine_rows
    from .tp_exact import gather_last_dim, tp_info

    slices = getattr(backend, "_rlk_tp_experts", None)
    if slices is None:
        raise RuntimeError("TP>1 routed experts were not bound (bind_tp_exact_routed_experts)")
    rank, size, _ = tp_info()
    if (rank, size) != (slices.rank, slices.size):
        raise RuntimeError("TP expert slices were loaded for a different TP layout")
    if activation != mod.MoEActivation.SILU:
        raise RuntimeError("TP-exact routed experts support the SILU clamp activation only")
    limit = backend.quant_config.gemm1_clamp_limit if backend.quant_config else None
    if limit is None or float(limit) != 10.0:
        raise RuntimeError("TP-exact routed experts require gemm1_clamp_limit == 10.0")
    if hidden_states.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise RuntimeError("TP-exact routed experts support BF16 only")
    M, K = hidden_states.shape[-2:]
    if K != slices.hidden:
        raise RuntimeError("hidden size differs from the checkpoint")

    tokens = routing_data._rlk_row_tokens
    experts = routing_data._rlk_row_experts
    valid = routing_data._rlk_row_valid
    gammas = routing_data.gate_scal.to(torch.float32).contiguous()
    offsets = expert_offsets(experts)
    _STATE["calls"] += 1
    run_id = f"vllm-tp-exact-routed-L{slices.layer_index}-{_STATE['calls']}"

    with torch.no_grad():
        x = hidden_states.reshape(M, K)[tokens].contiguous()
        z = packed_w4a16(x, slices.gate_up_codes, slices.gate_up_scales, offsets)
        act = torch.empty((z.shape[0], z.shape[1] // 2), device=z.device, dtype=z.dtype)
        # Same call as the actor's _SwiGLU; gate_j/up_j pair inside the I/TP slice.
        swiglu_limit_func(act, z, 10.0)
        act = gather_last_dim(act.contiguous())
        down = packed_w4a16(act, slices.down_codes, slices.down_scales, offsets, gammas)
        width = down.shape[1]
        local = torch.empty((M, width), device=down.device, dtype=torch.bfloat16)
        if routing_data._rlk_all_valid:
            fixed_combine_rows(down, tokens, experts, valid, M, run_id, local, True)
        else:
            # Warm-up/dummy batches with -1 slots: same per-slice pipeline, FP32 index_add_
            # over valid rows restricted to the local column slice (no cross-rank reduction).
            _STATE["fallbacks"] += 1
            print("RLK_ROLLOUT_TP_ROUTED_COMBINE_FALLBACK", _STATE["calls"], "M", M,
                  "invalid", int((~valid).sum()), flush=True)
            acc = torch.zeros((M, width), dtype=torch.float32, device=down.device)
            acc.index_add_(0, tokens[valid], down[valid].to(torch.float32))
            local.copy_(acc.to(torch.bfloat16))
        full = gather_last_dim(local)
    output.view(M, K).copy_(full)
    if not _STATE.get("executed"):
        print("RLK_ROLLOUT_TP_EXACT_ROUTED_EXPERTS_EXECUTED", run_id, "tp", size,
              tuple(x.shape), flush=True)
        _STATE["executed"] = True
    return output
