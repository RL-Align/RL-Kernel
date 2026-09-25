# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-8 (#67): Shared Expert TP column/row, SP reduce-scatter, shared-once.

Numeric profile ``p5-shared-tp-tree8-v1``
-----------------------------------------
FP32 addition is not associative, so no cross-rank merge of per-rank partial
sums can be byte-equal to a purely serial single-card reduction. Following the
precedent ``det_gemm`` set (a mid-split K tree whose children are exactly the
contiguous per-rank ranges, so "simulated TP=2 (a+b) matches TP=1"), this
profile anchors the TP=1 reference itself on a fixed reduction tree:

- FC1 is column-parallel: the gate block and the up block of ``w_fc1`` are
  EACH split into ``tp`` contiguous row ranges (never the raw ``[2F]`` rows,
  which would mispair gate/up across ranks). SwiGLU pairs gate/up locally;
  no cross-rank math. Per-column bytes equal the WS1 strict path.
- FC2 is row-parallel. Its K dimension (F) is reduced with a fixed
  ``NUM_LEAVES``-leaf mid-split tree: leaf j is the contiguous chunk
  ``F/NUM_LEAVES`` of columns, computed with the WS1 strict serial GEMM, and
  leaves are merged pairwise in ascending logical-tag order, all in FP32,
  with the single BF16 round at the end. A rank owning ``NUM_LEAVES/tp``
  adjacent leaves computes exactly one subtree, so every ``tp`` in
  {1, 2, 4, 8} reproduces identical bytes and the cross-rank merge is the
  upper part of the same tree.
- Backward: ``dh`` and the SwiGLU backward are rank-local (full-H serial,
  byte-equal to the WS1 path per column). ``dX`` sums per-leaf nodes
  ``node_j = gate_leaf_j + up_leaf_j`` with the same mid-split tree.
- SP: every output element's reduction tree is identical on every rank, so a
  reduce-scatter (merge + keep your row slice) is bitwise the same as
  all-reduce + slice; ``sp=True`` returns the rank's row shard of the same
  bytes.

The tree ("leaf = contiguous F/8 chunk, mid-split pairwise, FP32, leaf order
= ascending logical tag") is contract material and is reported in
``provenance()`` per P5-8 s2. Leaf order follows logical tags, never physical
rank ids, which is the ordered-collective-facade contract this module mocks
with :class:`OrderedTreeReducer` until #7 lands.

Shared-once: EP replicates the shared expert; :class:`SharedOnceLedger`
fail-closes the combine when the same shared output would be added twice.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from rl_engine.moe.backends.shared_expert import (
    CudaSharedExpertProvider,
    TritonSharedExpertProvider,
    _StrictSharedExpertProvider,
)
from rl_engine.moe.contract import SharedBatch, tensor_sha256

# Fixed leaf count = the maximum supported TP degree. Changing it changes the
# addition order, i.e. the numeric profile.
NUM_LEAVES = 8
TP_TREE_PROFILE = "p5-shared-tp-tree8-v1"
SUPPORTED_TP = (1, 2, 4, 8)

_BASE_BACKENDS = {
    "cuda": CudaSharedExpertProvider,
    "triton": TritonSharedExpertProvider,
}


def _fail(msg: str) -> None:
    raise NotImplementedError(f"{msg} (fail-closed, no fallback)")


class OrderedTreeReducer:
    """Mock of the #7 ordered collective facade for the mid-split tree merge.

    Leaves arrive keyed by logical tag; the merge is pairwise mid-split in
    ascending tag order, FP32, independent of physical rank ids or arrival
    order. The real facade replaces only the transport, never the tree.
    """

    def reduce(self, partials_by_tag: dict[int, torch.Tensor]) -> torch.Tensor:
        tags = sorted(partials_by_tag)
        if len(tags) == 0 or len(tags) & (len(tags) - 1):
            _fail(f"tree merge needs a power-of-two leaf count, got {len(tags)}")
        if tags != list(range(tags[0], tags[0] + len(tags))):
            _fail(f"tree merge needs contiguous logical tags, got {tags}")
        level = [partials_by_tag[t] for t in tags]
        while len(level) > 1:
            level = [level[i] + level[i + 1] for i in range(0, len(level), 2)]
        return level[0]


def shard_shared_batch(batch: SharedBatch, tp: int, rank: int) -> SharedBatch:
    """TP shard of a replicated SharedBatch (P5-8 s1 fixed split).

    FC1 column-parallel: gate rows and up rows are EACH split contiguously,
    and the rank's local ``w_fc1`` re-packs its gate shard above its up shard
    so the WS1 kernels see the usual packed layout. FC2 row-parallel: the
    rank keeps its contiguous F/tp columns.
    """
    if tp not in SUPPORTED_TP:
        _fail(f"tp={tp} not in {SUPPORTED_TP}")
    if not 0 <= rank < tp:
        _fail(f"rank={rank} out of range for tp={tp}")
    ffn = batch.w_fc1.shape[0] // 2
    if ffn % NUM_LEAVES:
        _fail(f"F={ffn} must be divisible by NUM_LEAVES={NUM_LEAVES}")
    shard = ffn // tp
    gate = batch.w_fc1[rank * shard : (rank + 1) * shard]
    up = batch.w_fc1[ffn + rank * shard : ffn + (rank + 1) * shard]
    return SharedBatch(
        x=batch.x,
        w_fc1=torch.cat([gate, up], dim=0).contiguous(),
        w_fc2=batch.w_fc2[:, rank * shard : (rank + 1) * shard].contiguous(),
        placement="tp-sharded",
        metadata={"tp": tp, "rank": rank, "leaf_tags": _rank_leaves(tp, rank)},
    )


def _rank_leaves(tp: int, rank: int) -> list[int]:
    per_rank = NUM_LEAVES // tp
    return list(range(rank * per_rank, (rank + 1) * per_rank))


class SharedOnceLedger:
    """Fail-closed guard: one shared contribution per combine key (P5-8 s3)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def register(self, key: str) -> None:
        if key in self._seen:
            raise RuntimeError(
                f"shared-once violated: shared output {key!r} was already merged "
                "into the combine (fail-closed; EP replication must contribute once)"
            )
        self._seen.add(key)


def combine_shared_once(
    routed_y: torch.Tensor,
    shared_y: torch.Tensor,
    ledger: SharedOnceLedger,
    key: str,
) -> torch.Tensor:
    """Gate helper for the P6 combine: adds ``shared_y`` exactly once per key."""
    ledger.register(key)
    return routed_y + shared_y


def shared_combine_key(batch: SharedBatch, layer_tag: str) -> str:
    digest = hashlib.sha256()
    digest.update(layer_tag.encode())
    digest.update(tensor_sha256(batch.x).encode())
    return digest.hexdigest()


class TPSimulatedSharedExpertProvider(_StrictSharedExpertProvider):
    """Single-process simulation of the TP/SP shared expert (WS2 semantics).

    Runs every rank's local math with the WS1 strict kernels of ``base`` and
    merges through :class:`OrderedTreeReducer`, so the bytes are exactly what
    the multi-process execution produces (the transport moves bits, the tree
    is the contract). ``tp=1`` is the profile's single-card anchor.
    """

    name = "shared-expert-tp-sim"
    numeric_profile = TP_TREE_PROFILE
    strict_profile = False

    def __init__(self, base: str = "cuda", tp: int = 1) -> None:
        if base not in _BASE_BACKENDS:
            _fail(f"unknown base backend {base!r}")
        if tp not in SUPPORTED_TP:
            _fail(f"tp={tp} not in {SUPPORTED_TP}")
        self._base = _BASE_BACKENDS[base]()
        self.tp = tp
        self._reducer = OrderedTreeReducer()
        self.name = f"shared-expert-tp-sim-{base}"

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
            "placement": "tp-sharded" if self.tp > 1 else "replicated",
            "tp": self.tp,
            "fc1_split": "column-parallel, gate/up each contiguous per rank",
            "fc2_split": "row-parallel",
            "reduction_tree": (
                f"midsplit {NUM_LEAVES}-leaf, leaf = contiguous F/{NUM_LEAVES} chunk, "
                "FP32 merge, leaf order = ascending logical tag"
            ),
            "collective": "OrderedTreeReducer (mock of the #7 facade)",
        }

    # -- forward ----------------------------------------------------------
    def shared_expert_mlp_fwd(self, batch: SharedBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        self._check_batch(batch)
        ffn = batch.w_fc1.shape[0] // 2
        if ffn % NUM_LEAVES:
            _fail(f"F={ffn} must be divisible by NUM_LEAVES={NUM_LEAVES}")
        leaf = ffn // NUM_LEAVES
        base = self._base
        partials: dict[int, torch.Tensor] = {}
        saved: dict[str, Any] = {"z_local": {}, "shards": {}}
        for rank in range(self.tp):
            local = shard_shared_batch(batch, self.tp, rank)
            z_l = base._gemm(batch.x.contiguous(), local.w_fc1, False)  # [T, 2F/tp] FP32
            h_l = base._swiglu_fwd(z_l)  # [T, F/tp] BF16
            saved["z_local"][rank] = z_l
            saved["shards"][rank] = local
            for j, tag in enumerate(local.metadata["leaf_tags"]):
                h_leaf = h_l[:, j * leaf : (j + 1) * leaf].contiguous()
                w2_leaf = local.w_fc2[:, j * leaf : (j + 1) * leaf].contiguous()
                partials[tag] = base._gemm(h_leaf, w2_leaf, False)  # [T, H] FP32
        y = self._reducer.reduce(partials).to(torch.bfloat16)
        return y, saved

    # -- backward ---------------------------------------------------------
    def shared_expert_mlp_bwd(
        self, dy: torch.Tensor, batch: SharedBatch, saved: dict[str, Any]
    ) -> torch.Tensor:
        self._check_batch(batch)
        ffn = batch.w_fc1.shape[0] // 2
        leaf = ffn // NUM_LEAVES
        base = self._base
        dy_bf16 = dy.to(torch.bfloat16).contiguous()
        nodes: dict[int, torch.Tensor] = {}
        for rank in range(self.tp):
            local: SharedBatch = saved["shards"][rank]
            z_l = saved["z_local"][rank]
            shard = ffn // self.tp
            dh_l = base._gemm(dy_bf16, local.w_fc2, True).to(torch.bfloat16)  # [T, F/tp]
            dz_l = base._swiglu_bwd(dh_l, z_l)  # [T, 2F/tp] BF16 (gate|up)
            for j, tag in enumerate(local.metadata["leaf_tags"]):
                dz_g = dz_l[:, j * leaf : (j + 1) * leaf].contiguous()
                dz_u = dz_l[:, shard + j * leaf : shard + (j + 1) * leaf].contiguous()
                w1_g = local.w_fc1[j * leaf : (j + 1) * leaf].contiguous()
                w1_u = local.w_fc1[shard + j * leaf : shard + (j + 1) * leaf].contiguous()
                # node_j = gate-leaf partial + up-leaf partial, in that order.
                nodes[tag] = base._gemm(dz_g, w1_g, True) + base._gemm(dz_u, w1_u, True)
        return self._reducer.reduce(nodes)


def sp_shard(y: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Rank's row shard of the merged output (reduce-scatter's local result).

    Every element's reduction tree is identical on every rank, so this equals
    all-reduce-then-slice bitwise; rows are padded to a multiple of tp on the
    caller's side if needed (fixtures and production T are multiples).
    """
    rows = y.shape[0]
    if rows % tp:
        _fail(f"SP needs T divisible by tp, got T={rows}, tp={tp}")
    shard = rows // tp
    return y[rank * shard : (rank + 1) * shard].contiguous()
