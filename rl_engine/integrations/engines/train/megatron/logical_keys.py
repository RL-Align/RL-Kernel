# SPDX-License-Identifier: Apache-2.0
"""Stable sample/position keys for actual ordinary native actor token rows."""

import hashlib

import torch


def logical_keys_for_input(input_ids, cases, *, valid_lengths=None):
    if input_ids.ndim != 2:
        raise ValueError("ordinary [batch,time] token IDs are required")
    actual = input_ids.detach().cpu().tolist()
    expected = [(c["id"], c["prompt_token_ids"] + c["response_token_ids"]) for c in cases]
    sample_ids = []
    for i, tokens in enumerate(actual):
        length = len(tokens) if valid_lengths is None else valid_lengths[i]
        matches = [
            name
            for name, sequence in expected
            if sequence == tokens[:length] and all(v == 0 for v in tokens[length:])
        ]
        if len(matches) != 1:
            raise ValueError("actual actor tokens must identify exactly one declared logical case")
        sample_ids.append(
            int.from_bytes(hashlib.sha256(matches[0].encode()).digest()[:8], "big")
            & ((1 << 63) - 1)
        )
    batch, time = input_ids.shape
    samples = torch.tensor(sample_ids, device=input_ids.device, dtype=torch.int64)[None, :].expand(
        time, batch
    )
    positions = torch.arange(time, device=input_ids.device, dtype=torch.int64)[:, None].expand(
        time, batch
    )
    return torch.stack((samples, positions), -1).reshape(time * batch, 2).contiguous()


def right_padded_lengths(input_ids, cases):
    """Match the real input prefix and only the pinned BSHD zero-padding suffix."""
    if input_ids.ndim != 2:
        raise ValueError("ordinary [batch,time] token IDs required")
    sequences = [c["prompt_token_ids"] + c["response_token_ids"] for c in cases]
    lengths = []
    for row in input_ids.detach().cpu().tolist():
        matches = [
            len(s)
            for s in sequences
            if len(s) <= len(row)
            and row[: len(s)] == s
            and all(token == 0 for token in row[len(s) :])
        ]
        if len(matches) != 1:
            raise ValueError(
                "actual right-padded input does not identify one complete logical case"
            )
        lengths.append(matches[0])
    return lengths
