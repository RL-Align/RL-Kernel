# SPDX-License-Identifier: Apache-2.0
"""Previous-pass attention-residual message capture for Inkling 2 Mini (INK-19).

Matches the Miles fixed-loop core: after 12 core layers, the message bank
contains the last completed core block and the final partial residual,
not the earliest completed core block. This reference never detaches/clones
sources: gradients must flow back through the preceding pass.
"""

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class PreviousPassBank:
    source_pass: int
    blocks: tuple[torch.Tensor, torch.Tensor]

    def for_target_pass(self, pass_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if pass_index != self.source_pass + 1:
            raise ValueError("message bank must come from the immediately preceding pass")
        return self.blocks


def capture_previous_pass_bank(
    completed_blocks: Sequence[torch.Tensor],
    partial: torch.Tensor,
    *,
    prelude_blocks: Sequence[torch.Tensor],
    source_pass: int,
) -> PreviousPassBank:
    """Capture two AttnRes sources without copying or severing autograd.

    The six-layer prelude has one completed block. The 12-layer core adds
    two completed blocks; Miles keeps the second core block and its final
    partial as the messages. Pass 0 feeds pass 1, and pass 1 feeds pass 2.
    """
    if source_pass not in (0, 1):
        raise ValueError("fixed three-pass model only captures passes 0 and 1")
    if len(prelude_blocks) != 1 or len(completed_blocks) != 3:
        raise ValueError("expected one prelude block and three completed blocks")
    if completed_blocks[0] is not prelude_blocks[0]:
        raise ValueError("completed blocks do not preserve the prelude snapshot")
    selected = (completed_blocks[2], partial)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 2 for t in selected):
        raise ValueError("message sources must be rank-2 [tokens, hidden] tensors")
    if selected[0].shape != selected[1].shape:
        raise ValueError("message sources must have the same token and hidden shape")
    if selected[0].dtype != selected[1].dtype or selected[0].device != selected[1].device:
        raise ValueError("message sources must share dtype and device")
    return PreviousPassBank(source_pass, selected)
