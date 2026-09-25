# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-8 (#67): multi-process TP gate (gloo transport, one shared CUDA device).

Each rank holds ONLY its own shard, computes its leaf partials with the WS1
strict CUDA kernels, and the merged bytes must equal the single-process
simulation. The transport moves bits; the mid-split tree is the contract, so
gloo here and the deterministic IPC collectives later must agree bitwise.
"""

from __future__ import annotations

import os

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rank_main(rank: int, world: int, rendezvous: str, out_path: str) -> None:
    import torch.distributed as dist

    from rl_engine.moe import fixtures
    from rl_engine.moe.backends.shared_expert import CudaSharedExpertProvider
    from rl_engine.moe.contract import tensor_sha256
    from rl_engine.moe.parallel import OrderedTreeReducer, shard_shared_batch
    from rl_engine.moe.parallel.shared_expert_tp import NUM_LEAVES

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        base = CudaSharedExpertProvider()
        batch = fixtures.make_shared_batch("shared_t16").to("cuda")
        local = shard_shared_batch(batch, world, rank)  # this rank's shard only
        ffn = batch.w_fc1.shape[0] // 2
        leaf = ffn // NUM_LEAVES

        z_l = base._gemm(batch.x.contiguous(), local.w_fc1, False)
        h_l = base._swiglu_fwd(z_l)
        mine: dict[int, torch.Tensor] = {}
        for j, tag in enumerate(local.metadata["leaf_tags"]):
            h_leaf = h_l[:, j * leaf : (j + 1) * leaf].contiguous()
            w2_leaf = local.w_fc2[:, j * leaf : (j + 1) * leaf].contiguous()
            mine[tag] = base._gemm(h_leaf, w2_leaf, False).cpu()

        gathered: list[dict[int, torch.Tensor] | None] = [None] * world
        dist.all_gather_object(gathered, mine)
        partials: dict[int, torch.Tensor] = {}
        for part in gathered:
            assert part is not None
            partials.update(part)
        # FP32 adds are IEEE round-to-nearest on CPU and GPU alike, so the
        # tree merge is bitwise transport- and device-independent.
        y = OrderedTreeReducer().reduce(partials).to(torch.bfloat16)
        if rank == 0:
            with open(out_path, "w") as fh:
                fh.write(tensor_sha256(y))
    finally:
        dist.destroy_process_group()


@requires_cuda
@pytest.mark.parametrize("world", [2, 4])
def test_multiprocess_tp_matches_simulation(tmp_path, world):
    import torch.multiprocessing as mp

    from rl_engine.moe import fixtures
    from rl_engine.moe.contract import tensor_sha256
    from rl_engine.moe.parallel import TPSimulatedSharedExpertProvider

    try:
        provider = TPSimulatedSharedExpertProvider(base="cuda", tp=world)
    except NotImplementedError as exc:
        pytest.skip(f"cuda backend unavailable: {exc}")
    batch = fixtures.make_shared_batch("shared_t16").to("cuda")
    y_sim, _ = provider.shared_expert_mlp_fwd(batch)

    rendezvous = tmp_path / "rdzv"
    out_path = tmp_path / "rank0_hash"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(
        _rank_main,
        args=(world, str(rendezvous), str(out_path)),
        nprocs=world,
        join=True,
    )
    assert out_path.read_text() == tensor_sha256(y_sim)
