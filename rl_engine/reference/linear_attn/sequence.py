# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU temporal recurrence using #659's corrected single-token reference from #469.

Correctness-oriented CPU reference. Reuses the single-step implementation and
copies the state bank per token; not intended for performance-sensitive execution.
"""

import torch

from rl_engine.contracts.operators.gdn import GDNContract
from rl_engine.reference.linear_attn.gated_delta_rule import GatedDeltaRuleRecurrentStepOp


def gated_delta_rule_sequence(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    contract: GDNContract,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed [T, HV, V] output and an independent complete state bank.

    State is [blocks, HV, V, K]. Pass the returned state into the same entry
    point for temporal chunk continuation; no detach or boundary cast occurs.
    Shared slots are zero-based, unique and nonnegative, with no null sentinel.
    """
    if mixed_qkv.device.type != "cpu":
        raise ValueError("gated_delta_rule_sequence is a CPU reference")
    boundaries = contract.validate(mixed_qkv, a, b, A_log, dt_bias, state, indices, cu_seqlens)
    # A private dummy slot preserves the legacy step's null rule while allowing
    # every logical shared-contract slot, including zero, to hold real state.
    bank = torch.cat((state.new_zeros((1, *state.shape[1:])), state))
    outputs = []
    step = GatedDeltaRuleRecurrentStepOp()
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        for token in range(start, end):
            out, bank = step(
                mixed_qkv[token : token + 1],
                a[token : token + 1],
                b[token : token + 1],
                A_log,
                dt_bias,
                bank,
                indices[seq : seq + 1] + 1,
                scale=contract.scale,
                num_k_heads=contract.num_k_heads,
                use_qk_l2norm=contract.use_qk_l2norm,
            )
            outputs.append(out[:, 0])
    output = (
        torch.cat(outputs)
        if outputs
        else mixed_qkv[:, :0].reshape(0, contract.num_v_heads, contract.value_dim)
    )
    return output, bank[1:].clone()
