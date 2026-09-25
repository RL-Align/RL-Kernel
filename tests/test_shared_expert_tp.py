# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-8 (#67): Shared Expert TP/SP invariance and shared-once gates.

All comparisons are byte-equality (sha256 over raw bytes) under the
``p5-shared-tp-tree8-v1`` profile; TP=1 of the same profile is the anchor.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe import fixtures, oracle
from rl_engine.moe.contract import SharedBatch, tensor_sha256
from rl_engine.moe.parallel import (
    SharedOnceLedger,
    TPSimulatedSharedExpertProvider,
    combine_shared_once,
    shard_shared_batch,
    shared_combine_key,
    sp_shard,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BASES = ("cuda", "triton")


def _provider(base: str, tp: int) -> TPSimulatedSharedExpertProvider:
    try:
        return TPSimulatedSharedExpertProvider(base=base, tp=tp)
    except NotImplementedError as exc:
        pytest.skip(f"{base} backend unavailable: {exc}")


def _run(provider, batch: SharedBatch, dy: torch.Tensor):
    y, saved = provider.shared_expert_mlp_fwd(batch)
    dx = provider.shared_expert_mlp_bwd(dy, batch, saved)
    return y, dx


def _batch_and_dy(t: int = 16):
    batch = fixtures.make_shared_batch("shared_t16").to("cuda")
    dy = fixtures.make_grad_output("shared_t16", (t, batch.x.shape[1])).to("cuda")
    return batch, dy


@requires_cuda
@pytest.mark.parametrize("base", BASES)
def test_tp_byte_equal_to_tp1(base):
    """TP=2/4/8 forward AND dX are byte-equal to TP=1 of the same profile."""
    batch, dy = _batch_and_dy()
    y_ref, dx_ref = _run(_provider(base, 1), batch, dy)
    for tp in (2, 4, 8):
        y, dx = _run(_provider(base, tp), batch, dy)
        assert tensor_sha256(y) == tensor_sha256(y_ref), f"fwd diverged at tp={tp}"
        assert tensor_sha256(dx) == tensor_sha256(dx_ref), f"dX diverged at tp={tp}"


@requires_cuda
def test_tp_cuda_triton_byte_equal():
    """The two base backends agree under the tree profile at every tp."""
    batch, dy = _batch_and_dy()
    for tp in (1, 4):
        y_c, dx_c = _run(_provider("cuda", tp), batch, dy)
        y_t, dx_t = _run(_provider("triton", tp), batch, dy)
        assert tensor_sha256(y_c) == tensor_sha256(y_t)
        assert tensor_sha256(dx_c) == tensor_sha256(dx_t)


@requires_cuda
@pytest.mark.parametrize("base", BASES)
def test_sp_reduce_scatter_equals_all_reduce_slice(base):
    """SP: the rank's row shard equals the slice of the full merged output."""
    batch, dy = _batch_and_dy()
    for tp in (2, 4):
        y, _ = _run(_provider(base, tp), batch, dy)
        gathered = torch.cat([sp_shard(y, tp, r) for r in range(tp)], dim=0)
        assert tensor_sha256(gathered) == tensor_sha256(y)


@requires_cuda
@pytest.mark.parametrize("base", BASES)
def test_tree_profile_close_to_ws1_oracle(base):
    """The tree profile differs from the WS1 serial oracle only in the fc2/dx
    reduction order; the fc1/SwiGLU/dh columns are byte-equal per shard by
    construction. Document the deviation instead of pretending byte parity."""
    batch, dy = _batch_and_dy()
    y_gold, saved_gold = oracle.shared_expert_mlp_fwd(batch)
    dx_gold = oracle.shared_expert_mlp_bwd(dy, batch, saved_gold)
    y, dx = _run(_provider(base, 1), batch, dy)
    torch.testing.assert_close(y.float(), y_gold.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dx, dx_gold, rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize("base", BASES)
def test_batch_invariance_under_tp(base):
    """fwd(x)[t] == fwd(x[t:t+1]) byte-for-byte also in the sharded config."""
    batch, _ = _batch_and_dy()
    provider = _provider(base, 4)
    y_full, _ = provider.shared_expert_mlp_fwd(batch)
    for t in range(0, batch.x.shape[0], 5):
        row = SharedBatch(x=batch.x[t : t + 1].contiguous(), w_fc1=batch.w_fc1, w_fc2=batch.w_fc2)
        y_row, _ = provider.shared_expert_mlp_fwd(row)
        assert tensor_sha256(y_row) == tensor_sha256(y_full[t : t + 1]), f"row {t}"


@requires_cuda
def test_shard_split_is_gate_up_paired():
    """The FC1 shard must pair each gate chunk with ITS up chunk (P5-8 s3)."""
    batch, _ = _batch_and_dy()
    ffn = batch.w_fc1.shape[0] // 2
    tp = 4
    for rank in range(tp):
        local = shard_shared_batch(batch, tp, rank)
        shard = ffn // tp
        assert tensor_sha256(local.w_fc1[:shard]) == tensor_sha256(
            batch.w_fc1[rank * shard : (rank + 1) * shard]
        )
        assert tensor_sha256(local.w_fc1[shard:]) == tensor_sha256(
            batch.w_fc1[ffn + rank * shard : ffn + (rank + 1) * shard]
        )
        assert local.placement == "tp-sharded"


@requires_cuda
@pytest.mark.parametrize("base", BASES)
def test_shared_once_under_ep_replication(base):
    """EP replicas compute identical bytes; the combine admits exactly one."""
    batch, dy = _batch_and_dy()
    provider = _provider(base, 2)
    replicas = [provider.shared_expert_mlp_fwd(batch)[0] for _ in range(4)]  # 4 EP ranks
    hashes = {tensor_sha256(y) for y in replicas}
    assert len(hashes) == 1, "EP replicas must be bitwise identical"

    routed = torch.zeros_like(replicas[0])
    ledger = SharedOnceLedger()
    key = shared_combine_key(batch, layer_tag="layer0")
    combined = combine_shared_once(routed, replicas[0], ledger, key)
    assert tensor_sha256(combined) == tensor_sha256(replicas[0])
    # Injection: a second EP replica trying to merge the same shared output
    # must trip the fail-closed gate.
    with pytest.raises(RuntimeError, match="shared-once violated"):
        combine_shared_once(combined, replicas[1], ledger, key)


@requires_cuda
def test_fail_closed_configs():
    batch, _ = _batch_and_dy()
    with pytest.raises(NotImplementedError):
        TPSimulatedSharedExpertProvider(base="cuda", tp=3)  # unsupported degree
    with pytest.raises(NotImplementedError):
        shard_shared_batch(batch, tp=4, rank=7)  # rank out of range
    provider = _provider("cuda", 2)
    cpu_batch = fixtures.make_shared_batch("shared_t1")
    with pytest.raises(NotImplementedError):
        provider.shared_expert_mlp_fwd(cpu_batch)


@requires_cuda
def test_provenance_declares_tree_and_placement():
    provider = _provider("cuda", 4)
    info = provider.provenance()
    assert info["numeric_profile"] == "p5-shared-tp-tree8-v1"
    assert info["placement"] == "tp-sharded"
    assert "midsplit" in info["reduction_tree"]
    assert "logical tag" in info["reduction_tree"]
