# SPDX-License-Identifier: Apache-2.0
"""INK-53: valid-vocabulary, softcap and logical ownership negative controls."""

from rl_engine.models.inkling2.lm_head_logprob import selected_logprob, softcap_valid_logits


def logits_probe(raw, targets, *, valid_vocab_size, fault):
    if fault == "include_padding":
        logits = softcap_valid_logits(raw, valid_vocab_size=raw.shape[-1])
    elif fault == "omit_softcap":
        logits = raw[:, :valid_vocab_size]
    else:
        raise ValueError("unknown logits probe")
    return selected_logprob(logits, targets)


def validate_ownership(expected: dict, observed: dict) -> None:
    """Wrong logical owners are not a numerical comparison."""
    if set(expected) != {"tp", "ep", "cp"} or set(observed) != set(expected):
        raise ValueError("missing TP/EP/CP ownership metadata")
    for axis in ("tp", "ep", "cp"):
        if tuple(expected[axis]) != tuple(observed[axis]):
            raise ValueError(f"incomparable {axis} logical ownership")
