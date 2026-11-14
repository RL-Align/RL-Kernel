# SPDX-License-Identifier: Apache-2.0
"""Stable sample/position keys for actual ordinary native actor token rows."""

import hashlib

import torch


def logical_keys_for_input(input_ids, cases):
    if input_ids.ndim != 2:
        raise ValueError("ordinary [batch,time] token IDs are required")
    actual = input_ids.detach().cpu().tolist()
    expected = [(c["id"], c["prompt_token_ids"] + c["response_token_ids"]) for c in cases]
    sample_ids = []
    for tokens in actual:
        matches = [name for name, sequence in expected if sequence == tokens]
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
