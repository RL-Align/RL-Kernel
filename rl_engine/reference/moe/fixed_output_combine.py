# SPDX-License-Identifier: Apache-2.0
"""Task 53/54 reference with visible FP32 accumulation and one final cast."""

import torch

from rl_engine.contracts.operators.fixed_output_combine import OutputCombineTrace


@torch.no_grad()
def fixed_output_combine_reference(prepared, expert_output, shared_output=None, *, mapping_run_id):
    prepared.validate_execution(expert_output, shared_output, mapping_run_id)
    if expert_output.device.type != "cpu":
        raise RuntimeError("the fixed output-combine reference is CPU-only")
    t, h = prepared.num_tokens, prepared.plan.hidden_size
    restored = expert_output.index_select(0, prepared._rows.flatten()).reshape(t, 6, h)
    acc = torch.zeros((t, h), dtype=torch.float32)
    steps = []
    for slot in range(6):
        acc = acc + restored[:, slot].float()
        steps.append(acc.clone())
    if shared_output is not None:
        acc = acc + shared_output.float()
    return OutputCombineTrace(acc.to(expert_output.dtype), torch.stack(steps, dim=1), acc)
