# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bitwise TP regression tests for the v0.1.0 reduction tree and partial-gradient sync (#44).

Every check compares TP=2/4 against TP=1 on the same canonical data. Inputs span
many orders of magnitude so a different summation order changes the bits; each
test also asserts that a naive order would have been caught.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.integrations.common.ppo import (
    PPOConfig,
    PPORollout,
    PPOTrainingWorker,
    build_ppo_batch,
)
from rl_engine.integrations.common.tp_scalar import (
    TPModelInputs,
    balanced_tree_sum,
    canonical_feature_dot,
    clip_tp_grad_norm_,
    deterministic_tp_all_reduce_,
    sum_tp_contributions,
)

CANONICAL_TP = 8


def _bits(tensor):
    """Reinterpret a floating tensor as integers of the same width."""
    dtype = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
    }[tensor.dtype]
    return tensor.detach().contiguous().view(dtype)


def assert_bitwise(actual, expected):
    """Fail unless both tensors have identical shape, dtype and bits."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(_bits(actual), _bits(expected)), (actual, expected)


def wide_range(*shape, seed, device):
    """Values spanning ~1e-4 to ~1e4 in both signs, so rounding depends on summation order."""
    generator = torch.Generator().manual_seed(seed)
    scale = 10.0 ** torch.randint(-4, 5, shape, generator=generator).float()
    return (torch.randn(*shape, generator=generator) * scale).to(device)


def _init(rank, size, rendezvous, backend):
    """Join a file-rendezvous group; NCCL ranks use device ``rank``."""
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=size,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=60),
    )


def _groups(size):
    """Return (TP group of all ranks, this rank's singleton TP=1 group); all ranks create both."""
    group = dist.new_group(list(range(size)))
    singles = [dist.new_group([r]) for r in range(size)]
    return group, singles[dist.get_rank()]


def _shard(full, rank, size, dim, stride=1):
    """Megatron-style shard: each of ``stride`` blocks is split ``size`` ways along ``dim``."""
    return torch.cat([b.chunk(size, dim=dim)[rank] for b in full.chunk(stride, dim=dim)], dim=dim)


def _reduction_worker(rank, size, rendezvous, backend):
    """Local subtree + fixed-tree cross-rank reduction, head outputs and gradient norms."""
    _init(rank, size, rendezvous, backend)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    try:
        group, single = _groups(size)
        per_rank = CANONICAL_TP // size

        # 1. Canonical leaves: local subtree, then the cross-rank tree, equals TP=1.
        for dtype in (torch.float32, torch.float64):
            leaves = [wide_range(5, seed=10 + i, device=device).to(dtype) for i in range(8)]
            for i, value in enumerate((2.0**60, 1.0, -(2.0**60), 1.0) * 2):
                leaves[i][0] = value  # (big+1)+(-big+1) cancels; a left fold keeps the ones.
            reference = balanced_tree_sum(leaves)
            sequential = leaves[0]
            for leaf in leaves[1:]:
                sequential = sequential + leaf
            assert not torch.equal(_bits(sequential), _bits(reference)), "data cannot detect order"
            local = balanced_tree_sum(leaves[rank * per_rank : (rank + 1) * per_rank])
            assert_bitwise(deterministic_tp_all_reduce_(local.clone(), group), reference)

        # 2. Scalar head outputs: canonical FP32 leaves + fixed tree, forward and backward.
        hidden = 64
        features = wide_range(3, 4, hidden, seed=1, device=device).bfloat16()
        weight = wide_range(hidden, seed=2, device=device).bfloat16()
        full = features.clone().requires_grad_()
        reference = canonical_feature_dot(full, weight, tp_world=1, canonical_tp=CANONICAL_TP)
        flat = (full.float() * weight.float()).sum(-1)
        assert not torch.equal(_bits(flat), _bits(reference)), "data cannot detect leaf order"
        local = _shard(features, rank, size, dim=-1).requires_grad_()
        values = sum_tp_contributions(
            canonical_feature_dot(
                local,
                _shard(weight, rank, size, dim=0),
                tp_world=size,
                canonical_tp=CANONICAL_TP,
            ),
            group,
        )
        assert_bitwise(values, reference)
        upstream = wide_range(3, 4, seed=3, device=device)
        values.backward(upstream)
        reference.backward(upstream)
        assert_bitwise(local.grad, _shard(full.grad, rank, size, dim=-1))

        # 3. Gradient norm and clipping with column, fused gate/up, row and replicated params.
        specs = (("column", (16, 8), 0, 1), ("gate_up", (32, 8), 0, 2), ("row", (8, 16), 1, 1))
        random_grads = {
            name: wide_range(*shape, seed=20 + i, device=device)
            for i, (name, shape, _, _) in enumerate(specs)
        }
        random_grads["norm"] = wide_range(8, seed=30, device=device)
        # Canonical leaves [2**54, 2, 2, 2, 2, 2, 2, 2]: the balanced tree keeps the twos
        # (2**54 + 12) while a left fold rounds every two away (2**54), on any backend.
        crafted_grads = {name: torch.zeros(shape, device=device) for name, shape, _, _ in specs}
        crafted_grads["norm"] = torch.zeros(8, device=device)
        crafted_grads["column"][0, 0] = 2.0**27
        for leaf in range(1, CANONICAL_TP):
            crafted_grads["column"][2 * leaf, :2] = 1.0

        def parameters(full_grads, tp_rank, tp_size):
            out = []
            for name, shape, dim, stride in specs:
                grad = _shard(full_grads[name], tp_rank, tp_size, dim, stride).clone()
                parameter = torch.nn.Parameter(torch.zeros_like(grad))
                parameter.partition_dim, parameter.partition_stride = dim, stride
                parameter.grad = grad
                out.append((parameter, "sharded"))
            norm = torch.nn.Parameter(torch.zeros(8, device=device))
            norm.grad = full_grads["norm"].clone()
            out.append((norm, "replicated"))
            return out

        for full_grads in (random_grads, crafted_grads):
            reference_params = parameters(full_grads, 0, 1)
            reference_norm = clip_tp_grad_norm_(
                reference_params,
                group=single,
                max_norm=1.0,
                device=device,
                canonical_tp=CANONICAL_TP,
            )
            tp_params = parameters(full_grads, rank, size)
            tp_norm = clip_tp_grad_norm_(
                tp_params, group=group, max_norm=1.0, device=device, canonical_tp=CANONICAL_TP
            )
            assert_bitwise(tp_norm, reference_norm)
            for (parameter, _), (reference_parameter, _), (_, _, dim, stride) in zip(
                tp_params, reference_params, specs
            ):
                assert_bitwise(
                    parameter.grad, _shard(reference_parameter.grad, rank, size, dim, stride)
                )
            assert_bitwise(tp_params[-1][0].grad, reference_params[-1][0].grad)
        tree = torch.tensor(2.0**54 + 12, dtype=torch.float64, device=device).sqrt()
        left_fold = torch.tensor(2.0**54, dtype=torch.float64, device=device).sqrt()
        assert_bitwise(reference_norm, tree)
        assert not torch.equal(_bits(tree), _bits(left_fold)), "data cannot detect order"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
def test_reduction_tree_is_bitwise_tp_invariant(tmp_path, size):
    """TP=2/4 head outputs, head gradients, leaf sums and clipped norms match TP=1 bitwise."""
    mp.start_processes(
        _reduction_worker,
        args=(size, str(tmp_path / "init"), "gloo"),
        nprocs=size,
        start_method="spawn",
    )


@pytest.mark.skipif(os.environ.get("RUN_TP_CUDA") != "1", reason="opt-in multi-GPU validation")
def test_nccl_reduction_tree_is_bitwise_tp_invariant(tmp_path):
    """The same checks with CUDA tensors use DeterministicCollective on two GPUs."""
    mp.start_processes(
        _reduction_worker,
        args=(2, str(tmp_path / "nccl"), "nccl"),
        nprocs=2,
        start_method="spawn",
    )


VOCAB, HEADS, HEAD_DIM = 8, 4, 4


class HeadNormActor(torch.nn.Module):
    """Tiny head-sharded actor with a replicated per-head RMSNorm weight, like Qwen3 Q/K norm.

    ``embedding`` [V, heads*d] is sharded over heads, ``norm`` [d] is shared by every head,
    and ``head`` [heads*d, V] is row-sharded; partial logits are summed over TP. The norm
    gradient on each rank therefore only covers its local heads.
    """

    def __init__(self, *, group, size, rank, dtype, device):
        """Build the full seeded model and keep this rank's head shard."""
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        embedding = torch.randn(VOCAB, HEADS * HEAD_DIM, generator=generator) * 0.5
        head = torch.randn(HEADS * HEAD_DIM, VOCAB, generator=generator) * 0.5
        norm = 1 + 0.1 * torch.randn(HEAD_DIM, generator=generator)
        self.group, self.size = group, size
        self.embedding = torch.nn.Parameter(
            embedding.chunk(size, dim=1)[rank].clone().to(device, dtype)
        )
        self.norm = torch.nn.Parameter(norm.clone().to(device, dtype))
        self.head = torch.nn.Parameter(head.chunk(size, dim=0)[rank].clone().to(device, dtype))

    def forward(self, inputs: TPModelInputs) -> torch.Tensor:
        """Return FP32 selected-token logprobs [B, S-1]."""
        hidden = self.embedding[inputs.input_ids].unflatten(-1, (HEADS // self.size, HEAD_DIM))
        hidden = hidden * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6).to(
            hidden.dtype
        )
        local = (hidden * self.norm).flatten(-2) @ self.head
        logits = sum_tp_contributions(local.float(), self.group)
        return logits[:, :-1].log_softmax(-1).gather(-1, inputs.input_ids[:, 1:, None]).squeeze(-1)


def ppo_batch(device):
    """A fixed version-0 batch: one right- and one left-padded sequence."""
    ids = torch.tensor([[1, 2, 3, 4, 5, 0], [0, 6, 2, 7, 4, 3]], device=device)
    attention = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]], dtype=torch.bool, device=device
    )
    action = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1]], dtype=torch.bool, device=device)
    rollout = PPORollout(
        ids,
        attention,
        (attention.long().cumsum(1) - 1).clamp(min=0),
        action,
        torch.full((2, 5), -2.0, device=device),
        torch.tensor([True, False], device=device),
        torch.tensor([0, 1], device=device),
        torch.tensor(0, device=device),
        torch.tensor(0, device=device),
        torch.tensor([1.0, 0.0, 1.0], device=device),
    )
    return build_ppo_batch(
        rollout,
        state_values=torch.zeros(2, 6, device=device),
        rewards=torch.tensor([1.0, -0.5], device=device),
        reference_logps=torch.full((2, 5), -2.1, device=device),
        config=PPOConfig(),
    )


def _train(group, size, rank, precision, norm_layout, device):
    """Run one PPO update and return the actor plus its (FP32 master) norm weight."""
    from rl_engine.integrations.engines.train.megatron.ppo_precision import FP32MasterWeights

    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model = HeadNormActor(group=group, size=size, rank=rank, dtype=dtype, device=device)
    layouts = {"embedding": "sharded", "norm": norm_layout, "head": "sharded"}
    config = PPOConfig(epochs=1, minibatch_size=2, microbatch_size=1, max_grad_norm=0.05)
    factory = lambda params: torch.optim.SGD(params, lr=1.0)  # noqa: E731
    if precision == "bf16":
        master = FP32MasterWeights(model.parameters(), factory)
        worker = PPOTrainingWorker(
            role="actor",
            model=model,
            forward=model,
            optimizer=master.optimizer,
            parameter_layouts=layouts,
            group=group,
            config=config,
            gradient_state=master.gradient_state,
            backward=torch.Tensor.backward,
            step=master.step,
        )
        norm = lambda: master.masters[1]  # noqa: E731
    else:
        worker = PPOTrainingWorker(
            role="actor",
            model=model,
            forward=model,
            optimizer=factory(list(model.parameters())),
            parameter_layouts=layouts,
            group=group,
            config=config,
        )
        norm = lambda: model.norm  # noqa: E731
    initial = norm().detach().clone()
    result = worker.train(ppo_batch(device))
    assert result["updates"] == 1
    assert not torch.equal(norm(), initial), "the norm weight was not updated"
    return model, norm().detach().clone()


def _update_worker(rank, size, rendezvous, backend, precision):
    """TP=1 vs TP=2 PPO updates with a partial-gradient norm weight; replicas must agree."""
    _init(rank, size, rendezvous, backend)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    try:
        group, single = _groups(size)
        _, reference_norm = _train(single, 1, 0, precision, "partial", device)
        _, norm = _train(group, size, rank, precision, "partial", device)
        replicas = [torch.empty_like(norm) for _ in range(size)]
        dist.all_gather(replicas, norm, group=group)
        for replica in replicas[1:]:
            assert_bitwise(replica, replicas[0])
        # TP1 and TP2 differ only by BF16/FP32 rounding of the local head partials.
        tolerance = {"fp32": 1e-5, "bf16": 1e-3}[precision]
        torch.testing.assert_close(norm, reference_norm, rtol=tolerance, atol=tolerance / 10)

        # Negative control: declaring the partial gradient replicated lets replicas diverge.
        _, diverged = _train(group, size, rank, precision, "replicated", device)
        replicas = [torch.empty_like(diverged) for _ in range(size)]
        dist.all_gather(replicas, diverged, group=group)
        assert any(not torch.equal(r, replicas[0]) for r in replicas[1:])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_partial_gradients_keep_tp_replicas_identical(tmp_path, precision):
    """Q/K-norm-style partial gradients are summed before clipping, FP32 masters included."""
    mp.start_processes(
        _update_worker,
        args=(2, str(tmp_path / "init"), "gloo", precision),
        nprocs=2,
        start_method="spawn",
    )


@pytest.mark.skipif(os.environ.get("RUN_TP_CUDA") != "1", reason="opt-in multi-GPU validation")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_nccl_partial_gradients_keep_tp_replicas_identical(tmp_path, precision):
    """The TP1/TP2 update comparison on two GPUs through DeterministicCollective."""
    mp.start_processes(
        _update_worker,
        args=(2, str(tmp_path / "nccl"), "nccl", precision),
        nprocs=2,
        start_method="spawn",
    )


def _partial_sync_worker(rank, size, rendezvous, case):
    """Drive ``_synchronize_partial_gradients`` on gloo with stand-in worker state."""
    from types import SimpleNamespace

    _init(rank, size, rendezvous, "gloo")
    try:
        group = dist.new_group(list(range(size)))
        a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
        layouts = {"q_norm": "partial", "k_norm": "partial"}
        if case == "different-names" and rank == 1:
            layouts = {"q_norm": "partial", "other_norm": "partial"}
        names = list(layouts)
        if case == "different-order" and rank == 1:
            names = names[::-1]
        params = {"q_norm": a, "k_norm": b, "other_norm": b}
        state = SimpleNamespace(
            group=group,
            parameters=[(params[n], layouts[n]) for n in names],
            _partial=sorted((n, i) for i, n in enumerate(names)),
            _partial_names_checked=False,
        )
        sync = PPOTrainingWorker._synchronize_partial_gradients
        if case == "different-names":
            with pytest.raises(ValueError, match="differ across TP ranks"):
                sync(state, None, torch.device("cpu"))
            return
        if case == "unused":
            # No rank used k_norm; rank 0 alone used q_norm.
            a.grad = torch.full((3,), 2.0) if rank == 0 else None
            sync(state, None, torch.device("cpu"))
            assert b.grad is None
            assert torch.equal(a.grad, torch.full((3,), 2.0))
            return
        # different-order: each rank lists the two names in another order; gradients are
        # still summed name by name.
        a.grad = torch.full((3,), float(1 + rank))
        b.grad = torch.full((3,), float(10 * (1 + rank)))
        sync(state, None, torch.device("cpu"))
        assert torch.equal(a.grad, torch.full((3,), 3.0))
        assert torch.equal(b.grad, torch.full((3,), 30.0))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["unused", "different-order", "different-names"])
def test_partial_gradients_follow_names_and_global_presence(tmp_path, case):
    """Unused everywhere stays unused (as at TP1); order and names do not depend on the caller."""
    mp.start_processes(
        _partial_sync_worker,
        args=(2, str(tmp_path / "init"), case),
        nprocs=2,
        start_method="spawn",
    )


def test_canonical_square_sums_one_leaf_rank_matches_smaller_tp():
    """With canonical clipping a one-leaf rank sums stride blocks as TP1 does; whole-shard
    summing would differ in the last bit for this gradient."""
    from rl_engine.integrations.common.tp_scalar import _canonical_square_sums

    x = 1.5 * 2**27
    full = torch.tensor([x, 0, 0, 0, 1.75, 1.75, 0, 0], dtype=torch.float32)
    parameter = torch.nn.Parameter(torch.zeros(8))
    parameter.partition_dim, parameter.partition_stride = 0, 2
    tp1 = _canonical_square_sums(full, parameter, 2)  # canonical_tp=2 at TP1: two leaves
    for rank in range(2):
        (leaf,) = _canonical_square_sums(_shard(full, rank, 2, 0, stride=2), parameter, 1)
        assert_bitwise(leaf, tp1[rank])


def test_canonical_square_sums_count_a_leader_only_bias_in_leaf_zero():
    """A sharded gradient without ``partition_dim`` (leader-only bias) no longer raises."""
    from rl_engine.integrations.common.tp_scalar import _canonical_square_sums

    bias = torch.nn.Parameter(torch.zeros(1))
    sums = _canonical_square_sums(torch.tensor([3.0]), bias, 4)
    assert [float(s) for s in sums] == [9.0, 0.0, 0.0, 0.0]
