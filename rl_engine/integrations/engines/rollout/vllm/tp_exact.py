# SPDX-License-Identifier: Apache-2.0
"""TP-invariant rollout helpers: exact data movement only, no cross-rank float reduction."""

from types import MethodType

import torch

# Dtypes the collective moves natively; 1-byte dtypes (FP8, bool) travel as uint8.
_NATIVE_GATHER_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.int16)
_INT_BY_SIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32}


def tp_info():
    """(rank, size, device_group) of the vLLM tensor-parallel group."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )

    return (
        get_tensor_model_parallel_rank(),
        get_tensor_model_parallel_world_size(),
        get_tp_group().device_group,
    )


def last_dim_from_gathered(gathered, size, lead_shape):
    """[size * M, C] rank-ordered dim-0 gather -> [*lead_shape, size * C] rank-ordered last dim."""
    rows, cols = gathered.shape
    if rows % size != 0:
        raise ValueError(f"gathered rows {rows} not divisible by TP size {size}")
    m = rows // size
    out = gathered.view(size, m, cols).permute(1, 0, 2).reshape(m, size * cols)
    return out.view(*lead_shape, size * cols)


def gather_wire_view(x):
    """Same-shape view of x with a dtype the collective accepts (bit-exact)."""
    if x.dtype in _NATIVE_GATHER_DTYPES:
        return x
    if x.element_size() == 1:
        return x.view(torch.uint8)
    raise TypeError(f"gather_last_dim does not support {x.dtype}")


def capacity_bytes(max_num_batched_tokens, tp_size):
    """Collective payload capacity for the largest TP gather."""
    return int(max_num_batched_tokens) * 4096 * 2 * int(tp_size)


def prepare_tp_collective(vllm_config):
    """Create the TP collective with enough capacity before the first gather."""
    _, size, group = tp_info()
    if size == 1:
        return None
    from rl_engine.distributed.algorithms.collectives import collective_for_group

    return collective_for_group(
        group,
        min_size_bytes=capacity_bytes(vllm_config.scheduler_config.max_num_batched_tokens, size),
    )


def gather_last_dim(x):
    """Rank-ordered all-gather of x concatenated along the last dim; identity at TP1."""
    _, size, group = tp_info()
    if size == 1:
        return x
    from rl_engine.distributed.algorithms.collectives import collective_for_group

    x = x.contiguous()
    lead, cols = x.shape[:-1], x.shape[-1]
    wire = gather_wire_view(x).reshape(-1, cols)
    nbytes = wire.numel() * wire.element_size()
    gathered = collective_for_group(group, min_size_bytes=nbytes).all_gather(wire)
    return last_dim_from_gathered(gathered, size, lead).view(x.dtype)


def _is_none_mode(value):
    if value is None:
        return True
    name = getattr(value, "name", None)
    return name == "NONE" or (name is None and value in (0, "NONE"))


def assert_eager(vllm_config):
    """Fail closed unless the rollout runs eager: no CUDA graphs, no torch.compile."""
    if not getattr(vllm_config.model_config, "enforce_eager", False):
        raise RuntimeError("TP-exact rollout requires enforce_eager=True")
    compilation = vllm_config.compilation_config
    if not _is_none_mode(getattr(compilation, "mode", None)):
        raise RuntimeError(f"TP-exact rollout forbids torch.compile (mode={compilation.mode})")
    if not _is_none_mode(getattr(compilation, "cudagraph_mode", None)):
        raise RuntimeError(
            f"TP-exact rollout forbids CUDA graphs (cudagraph_mode={compilation.cudagraph_mode})"
        )


def int_bits(x):
    """Exact int32 widening of x's bit pattern (sign-extended for 1/2-byte dtypes)."""
    return x.view(_INT_BY_SIZE[x.element_size()]).to(torch.int32)


def bits_to_dtype(bits, dtype):
    """Inverse of int_bits for a value that fits the original width."""
    size = torch.empty((), dtype=dtype).element_size()
    return bits.to(_INT_BY_SIZE[size]).view(dtype)


def combine_owner_bits(parts, dtype):
    """Sum per-rank owner-or-zero int32 bit views and restore dtype (CPU reference)."""
    total = parts[0].clone()
    for part in parts[1:]:
        total += part
    return bits_to_dtype(total, dtype)


def bind_exact_embedding(embed_tokens):
    """Make the cross-rank vocab combine exact, including -0.0; no-op at TP1."""
    _, size, device_group = tp_info()
    if size == 1:
        return embed_tokens
    if getattr(embed_tokens, "_rlk_exact_embedding", False):
        return embed_tokens

    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    if not embed_tokens.reduce_results:
        raise RuntimeError("exact embedding expects reduce_results=True")

    def forward(self, input_):
        if self.use_fused_embedding:
            output_parallel = vpe.ops.vocab_parallel_embedding(
                input_ if input_.ndim == 1 else input_.reshape(-1),
                self.weight,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            if input_.ndim != 1:
                output_parallel = output_parallel.view(*input_.shape, self.embedding_dim)
            input_mask = None
        else:
            masked_input, input_mask = vpe.get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            output_parallel = self.quant_method.embedding(self, masked_input.long())
        dtype = output_parallel.dtype
        bits = output_parallel.contiguous().view(_INT_BY_SIZE[output_parallel.element_size()])
        if input_mask is not None:
            bits.masked_fill_(input_mask.unsqueeze(-1), 0)
        # One owner row + zeros: the integer sum is exact, -0.0 included.
        wide = bits.to(torch.int32)
        # Plain RCCL on the TP device group: vLLM's custom/quick all-reduce is float-only.
        torch.distributed.all_reduce(wide, group=device_group)
        return bits_to_dtype(wide, dtype)

    embed_tokens.forward = MethodType(forward, embed_tokens)
    embed_tokens._rlk_exact_embedding = True
    return embed_tokens


def bind_exact_moe_runner(runner):
    """Drop the MoE runner's TP all-reduces; routed and shared outputs are already full."""
    _, size, _ = tp_info()
    if size == 1:
        return runner
    if getattr(runner, "_rlk_exact_moe_runner", False):
        return runner
    config = runner.moe_config
    if config.is_sequence_parallel:
        raise RuntimeError("TP-exact MoE does not support sequence parallel")
    if runner.routed_output_transform is not None:
        raise RuntimeError("TP-exact MoE does not support routed_output_transform")
    if getattr(config, "ep_size", 1) != 1:
        raise RuntimeError("TP-exact MoE requires EP=1")
    if runner._fused_output_is_reduced:
        raise RuntimeError("TP-exact MoE requires an unreduced fused-output kernel")
    shared = runner.shared_experts
    if shared is not None and getattr(shared, "_stream", None) is not None:
        raise RuntimeError("TP-exact MoE requires VLLM_DISABLE_SHARED_EXPERTS_STREAM=1")

    def reduce_shared(shared_output, fused_output_is_reduced=None):
        return shared_output

    def reduce_final(states, trunc_size, output_is_reduced=None):
        return states[..., :trunc_size] if trunc_size is not None else states

    runner._maybe_reduce_shared_expert_output = reduce_shared
    runner._maybe_reduce_final_output = reduce_final
    runner._rlk_exact_moe_runner = True
    return runner
