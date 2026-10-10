# SPDX-License-Identifier: Apache-2.0
"""Canonical logical-token parameter reductions for packed Megatron batches."""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

import torch

_ACTIVE = ContextVar("rl_kernel_cp_gradient_layout", default=None)


def current_layout():
    return _ACTIVE.get()


def bind_layout(function, layout):
    """Keep the forward's token layout during activation recomputation."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _ACTIVE.set(layout)
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    return wrapped


def packed_gather_indices(lengths, local_rows, cp_world):
    """Map rank-major zigzag shards to unpadded sample/token order."""
    out = []
    offset = 0
    for length in lengths:
        if cp_world == 1:
            out.extend(range(offset, offset + length))
            offset += length
            continue
        width = (length + 2 * cp_world - 1) // (2 * cp_world)
        for pos in range(length):
            block, within = divmod(pos, width)
            rank = block if block < cp_world else 2 * cp_world - 1 - block
            local = offset + within + (width if block >= cp_world else 0)
            out.append(rank * local_rows + local)
        offset += 2 * width
    if offset > local_rows:
        raise ValueError("canonical CP layout exceeds packed token rows")
    return out


@dataclass
class CPLayout:
    order: torch.Tensor
    local_rows: int
    cp_world: int
    cp_group: object
    tp_world: int
    tp_group: object
    cp_rank: int = 0
    tp_rank: int = 0

    def ordered(self, value):
        return value.index_select(0, self.order)

    def gather_many(self, *values):
        values = tuple(v.contiguous() for v in values)
        if any(v.size(0) != self.local_rows for v in values):
            raise ValueError("canonical CP tensors must start with packed token rows")
        if self.cp_world > 1:
            from rl_engine.distributed.algorithms.collectives import collective_for_group

            capacity = sum(v.numel() * v.element_size() for v in values)
            collective = collective_for_group(self.cp_group, min_size_bytes=capacity)
            values = collective.all_gather_many(values)
        return tuple(self.ordered(v) for v in values)


def replica_parameter_gradient(gradient, replicas, rank):
    """Contribute a full canonical parameter sum from one replica only.

    The framework still averages CP gradients and sums shared TP gradients.
    Repeatedly adding identical FP32 values with a ring can round differently
    for eight replicas. A single nonzero contribution makes those reductions
    exact without changing activation gradients or the optimizer algorithm.
    Do not mutate the incoming gradient: autograd may share it with another edge.
    """
    if rank != 0:
        return torch.zeros_like(gradient)
    return gradient if replicas == 1 else gradient * replicas


class CanonicalLossGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, cp_world):
        ctx.cp_world = cp_world
        # Megatron rescales this scalar in place after the loss callback.
        return loss.clone()

    @staticmethod
    def backward(ctx, gradient):
        # Remove the framework's CP multiplier before any low-precision
        # model derivative, including activation gradients and underflow.
        return gradient / ctx.cp_world, None


def weight_gradient(x, dy, layout, chunks=1, column=True):
    from rl_engine.ops.gemm.det_gemm import det_gemm_linear_weight_gradient

    if layout is not None:
        x, dy = layout.gather_many(x, dy)
    source = dy if column else x
    if chunks != 1 and source.size(-1) % chunks:
        raise ValueError("canonical parameter shard does not divide TP")
    combine_columns = False
    if torch.version.hip is None and x.is_cuda:
        from rl_engine.backends.cuda.gemm.det_gemm import det_gemm_backend

        combine_columns = det_gemm_backend() == "cublaslt_nosplitk"
    if chunks == 1 or combine_columns:
        result = det_gemm_linear_weight_gradient(x, dy)
    else:
        source = dy if column else x
        if source.size(-1) % chunks:
            raise ValueError("canonical parameter shard does not divide TP")
        width = source.size(-1) // chunks
        parts = [
            det_gemm_linear_weight_gradient(
                x if column else x.narrow(1, i * width, width).contiguous(),
                dy.narrow(1, i * width, width).contiguous() if column else dy,
            )
            for i in range(chunks)
        ]
        result = torch.cat(parts, dim=0 if column else 1)
    if layout is not None:
        result = replica_parameter_gradient(result, layout.cp_world, layout.cp_rank)
    return result


class CPRowLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, linear, layout):
        ctx.save_for_backward(x, weight)
        ctx.layout = layout
        return linear(x, weight)

    @staticmethod
    def backward(ctx, dy):
        from rl_engine.ops.gemm.det_gemm import det_gemm_linear_input_gradient

        x, w = ctx.saved_tensors
        dy = dy.contiguous()
        return (
            det_gemm_linear_input_gradient(dy, w) if ctx.needs_input_grad[0] else None,
            weight_gradient(x, dy, ctx.layout) if ctx.needs_input_grad[1] else None,
            None,
            None,
        )


class CPRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, layout, sharded_heads):
        ctx.save_for_backward(x, weight)
        ctx.eps, ctx.layout, ctx.sharded_heads = eps, layout, sharded_heads
        return torch.nn.functional.rms_norm(x, (x.size(-1),), weight, eps)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        with torch.enable_grad():
            local = x.detach().requires_grad_(True)
            y = torch.nn.functional.rms_norm(local, (x.size(-1),), weight.detach(), ctx.eps)
            (dx,) = torch.autograd.grad(y, local, dy)
        full_x, full_dy = ctx.layout.gather_many(x, dy)
        if ctx.sharded_heads and ctx.layout.tp_world > 1:
            # Q/K gamma is shared across heads. Preserve token/head order
            # before its parameter sum. Only TP0 contributes the complete
            # gamma gradient to Megatron's subsequent TP SUM.
            gathered = []
            for value in (full_x, full_dy):
                pieces = [torch.empty_like(value) for _ in range(ctx.layout.tp_world)]
                torch.distributed.all_gather(pieces, value.contiguous(), group=ctx.layout.tp_group)
                gathered.append(torch.cat(pieces, dim=-2))
            full_x, full_dy = gathered
        with torch.enable_grad():
            gamma = weight.detach().requires_grad_(True)
            full_y = torch.nn.functional.rms_norm(full_x, (x.size(-1),), gamma, ctx.eps)
            (dw,) = torch.autograd.grad(full_y, gamma, full_dy)
        replicas = ctx.layout.cp_world
        owner = ctx.layout.cp_rank == 0
        if ctx.sharded_heads:
            owner = owner and ctx.layout.tp_rank == 0
        dw = replica_parameter_gradient(dw, replicas, 0 if owner else 1)
        return dx, dw, None, None, None


def rms_norm(x, weight, eps, sharded_heads=False):
    layout = current_layout()
    if layout is None or not torch.is_grad_enabled():
        return torch.nn.functional.rms_norm(x, (x.size(-1),), weight, eps)
    return CPRMSNorm.apply(x, weight, eps, layout, sharded_heads)


class CPEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight, start, layout):
        ctx.save_for_backward(ids)
        ctx.shape, ctx.dtype, ctx.start, ctx.layout = weight.shape, weight.dtype, start, layout
        local = ids - start
        valid = (local >= 0) & (local < weight.size(0))
        return torch.nn.functional.embedding(local.masked_fill(~valid, 0), weight).masked_fill(
            ~valid.unsqueeze(-1), 0
        )

    @staticmethod
    def backward(ctx, dy):
        from rl_engine.backends.cuda.embedding.embedding import _deterministic_embedding_grad_weight

        (ids,) = ctx.saved_tensors
        ids, dy = ctx.layout.gather_many(ids.reshape(-1, 1), dy.reshape(-1, dy.size(-1)))
        dw = _deterministic_embedding_grad_weight(
            ids.flatten() - ctx.start,
            dy.float(),
            weight_shape=ctx.shape,
            weight_dtype=ctx.dtype,
        )
        dw = replica_parameter_gradient(dw, ctx.layout.cp_world, ctx.layout.cp_rank)
        return None, dw, None, None


def install():
    """Install at the batch/model boundaries; no arithmetic changes in VIME."""
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
    from vime.backends.megatron_utils import data
    from vime.backends.megatron_utils import loss as loss_module
    from vime.backends.megatron_utils import model

    if getattr(GPTModel, "_rlk_cp_parameter_grads", False):
        return
    from rl_engine.ops.norm.reproducible import install as install_norm

    install_norm()
    original_loss = model.loss_function

    @wraps(original_loss)
    def loss_function(*args, **kwargs):
        value, normalizer, metrics = original_loss(*args, **kwargs)
        return (
            CanonicalLossGradient.apply(value, mpu.get_context_parallel_world_size()),
            normalizer,
            metrics,
        )

    model.loss_function = loss_function
    loss_module.loss_function = loss_function
    from megatron.core.tensor_parallel.random import CheckpointFunction

    checkpoint_forward = CheckpointFunction.forward

    @wraps(checkpoint_forward)
    def checkpoint(ctx, function, distribute_saved_activations, *args):
        layout = current_layout()
        if layout is not None:
            function = bind_layout(function, layout)
        return checkpoint_forward(ctx, function, distribute_saved_activations, *args)

    CheckpointFunction.forward = staticmethod(checkpoint)
    original_batch = data.get_batch

    @wraps(original_batch)
    def get_batch(*args, **kwargs):
        batch = original_batch(*args, **kwargs)
        batch["packed_seq_params"]._rlk_true_lengths = tuple(
            t.numel() for t in batch["unconcat_tokens"]
        )
        return batch

    data.get_batch = get_batch
    model.get_batch = get_batch
    original_forward = GPTModel.forward

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        packed = kwargs.get("packed_seq_params")
        lengths = getattr(packed, "_rlk_true_lengths", None)
        if lengths is None or not torch.is_grad_enabled():
            return original_forward(self, *args, **kwargs)
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids.size(0) != 1:
            raise ValueError("canonical packed CP backward requires batch dimension one")
        cp = mpu.get_context_parallel_world_size()
        layout = CPLayout(
            torch.tensor(packed_gather_indices(lengths, ids.size(1), cp), device=ids.device),
            ids.size(1),
            cp,
            mpu.get_context_parallel_group(),
            mpu.get_tensor_model_parallel_world_size(),
            mpu.get_tensor_model_parallel_group(),
            mpu.get_context_parallel_rank(),
            mpu.get_tensor_model_parallel_rank(),
        )
        token = _ACTIVE.set(layout)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    original_embedding = VocabParallelEmbedding.forward

    @wraps(original_embedding)
    def embedding(self, ids):
        layout = current_layout()
        if layout is None:
            return original_embedding(self, ids)
        if self.reduce_scatter_embeddings:
            raise ValueError("canonical CP embedding does not support sequence parallelism")
        local = CPEmbedding.apply(ids, self.weight, self.vocab_start_index, layout)
        return reduce_from_tensor_model_parallel_region(local, group=self.tp_group)

    VocabParallelEmbedding.forward = embedding
    GPTModel.forward = forward
    GPTModel._rlk_cp_parameter_grads = True
