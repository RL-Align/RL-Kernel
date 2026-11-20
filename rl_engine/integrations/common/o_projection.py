# SPDX-License-Identifier: Apache-2.0
"""Opt-in framework boundary. Never invoke a framework's second TP reduction."""

from dataclasses import dataclass

from rl_engine.models.deepseek_v4.o_projection import OProjection, OProjectionWeights


@dataclass(frozen=True)
class BoundOProjection:
    operator: OProjection
    weights: OProjectionWeights

    def __call__(self, attention_output, *, rows):
        # Return typed local FP32 partials, not a tensor masquerading as reduced.
        return self.operator(attention_output, self.weights, rows=rows)


def bind_output_projection(
    *,
    contract,
    layer_id,
    wo_a,
    wo_b,
    version,
    a_checkpoint=None,
    b_checkpoint=None,
    backend=None,
    expected_implementation=None,
):
    """Explicit unshuffled tensors; the engine loader owns checkpoint conversion.

    wo_a [local_groups,R,8*D], wo_b [hidden,local_groups*R] are BF16 masters
    (or None for FP8 inference). Checkpoints use native FP8 data/FP32 scales.
    Refuse to infer a shuffled framework-private format or silently patch it.
    """
    op = OProjection(
        contract,
        layer_id=layer_id,
        backend=backend,
        expected_implementation=expected_implementation,
    )
    weights = op.prepare_weights(
        wo_a, wo_b, version=version, a_checkpoint=a_checkpoint, b_checkpoint=b_checkpoint
    )
    return BoundOProjection(op, weights)
