# SPDX-License-Identifier: Apache-2.0
"""Small shuffled Top-6 fixtures; never a substitute for P4 multi-rank testing."""

from dataclasses import replace

import torch

from rl_engine.contracts.operators.moe import MoEOutputCombineInputs, MoEOutputCombinePlan


def request(t=2, h=33, dtype=torch.bfloat16, shared=True, index_dtype=torch.int64):
    rng = torch.Generator().manual_seed(530)
    order = torch.randperm(t * 6, generator=rng)
    plan = MoEOutputCombinePlan(
        top_k=6,
        hidden_size=h,
        token_index_dtype=index_dtype,
        output_dtype=dtype,
        expert_weight_applied=True,
        shared_expert_mode="separate" if shared else "excluded",
    )
    # Non-contiguous hidden storage, including for the shared branch.
    values = torch.empty(t * 6, h * 2, dtype=dtype)[:, ::2]
    values.copy_(torch.randn(t * 6, h, generator=rng).to(dtype)[order])
    shared_out = None
    if shared:
        shared_out = torch.empty(t, h * 2, dtype=dtype)[:, ::2]
        shared_out.copy_(torch.randn(t, h, generator=rng).to(dtype))
    inputs = MoEOutputCombineInputs(
        expert_output=values,
        token_indices=(order // 6).to(index_dtype),
        expert_indices=(order % 6 + 10).to(index_dtype),
        slot_indices=(order % 6).to(index_dtype),
        shared_output=shared_out,
        num_tokens=t,
        mapping_run_id="run-53",
        producer_backend="fixture.P5",
        producer_build_fingerprint="fixture-v1",
    )
    return plan, inputs


def move(inputs, device):
    values = {}
    for name in (
        "expert_output",
        "token_indices",
        "expert_indices",
        "slot_indices",
        "shared_output",
    ):
        tensor = getattr(inputs, name)
        if tensor is None:
            continue
        if tensor.ndim == 2:
            copied = torch.empty(
                tensor.shape[0], tensor.shape[1] * 2, dtype=tensor.dtype, device=device
            )[:, ::2]
            copied.copy_(tensor)
        else:
            copied = tensor.to(device)
        values[name] = copied
    return replace(inputs, **values)
