# SPDX-License-Identifier: Apache-2.0
"""Bind existing framework-owned Indexer parameters without changing their layout."""

from dataclasses import dataclass

from rl_engine.models.deepseek_v4.indexer_projection import IndexerProjection, require


@dataclass(frozen=True)
class BoundIndexerProjection:
    operator: IndexerProjection
    q_weight: object
    weights_weight: object
    packed_q: object
    q_weight_version: str | None

    def __call__(self, qr, hidden, *, logical_rows=None):
        return self.operator(
            qr,
            hidden,
            self.q_weight,
            self.weights_weight,
            logical_rows=logical_rows,
            packed_q=self.packed_q,
            q_weight_version=self.q_weight_version,
        )

    def q(self, qr, *, logical_rows=None):
        return self.operator.project_q(
            qr,
            self.q_weight,
            logical_rows=logical_rows,
            packed_q=self.packed_q,
            q_weight_version=self.q_weight_version,
        )

    def weights(self, hidden, *, logical_rows=None):
        return self.operator.project_weights(hidden, self.weights_weight, logical_rows=logical_rows)


def bind_framework_indexer(
    indexer,
    *,
    contract,
    layer_id,
    backend=None,
    packed_q=None,
    expected_implementation=None,
):
    require(hasattr(indexer, "wq_b") and hasattr(indexer, "weights_proj"), "unknown Indexer ABI")
    for name in ("wq_b", "weights_proj"):
        layer = getattr(indexer, name)
        require(getattr(layer, "bias", None) is None, "Indexer projections are bias-free")
        quant_method = getattr(layer, "quant_method", None)
        if name == "weights_proj" or contract.q_precision == "bf16":
            require(
                quant_method is None or type(quant_method).__name__ == "UnquantizedLinearMethod",
                "quantized framework weights require an explicit compatible packed-Q contract",
            )
            require(
                not getattr(layer, "_fp8_source_scales_by_parameter", {}),
                "BF16 master carries deployment FP8 metadata; use block-FP8 contract",
            )
    if contract.q_precision == "block_fp8_ue8m0":
        require(packed_q is not None, "native checkpoint loader must provide packed Q")
        q_weight = packed_q.master
        if q_weight is not None:
            require(
                q_weight is indexer.wq_b.weight, "prepared master differs from framework weight"
            )
        else:
            require(
                packed_q.source is indexer.wq_b.weight,
                "prepared FP8 source differs from the framework weight",
            )
    else:
        q_weight = indexer.wq_b.weight
    return BoundIndexerProjection(
        IndexerProjection(
            contract,
            layer_id=layer_id,
            backend=backend,
            expected_implementation=expected_implementation,
        ),
        q_weight,
        indexer.weights_proj.weight,
        packed_q,
        None if packed_q is None else packed_q.version,
    )
