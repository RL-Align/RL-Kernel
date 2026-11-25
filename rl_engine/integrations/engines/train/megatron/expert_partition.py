# SPDX-License-Identifier: Apache-2.0
"""Original expert ownership and packed loader for the gated EP4 extension.

This loader is a component, not admission to N5. Native EP dispatch/gradient and
full independent acceptance must qualify before the engine EP1 guard changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OriginalExpertPartition:
    total: int
    ep_size: int
    ep_rank: int

    def __post_init__(self):
        if self.total != 256 or self.ep_size not in (1, 4) or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("original DSV4 requires 256 experts and explicit EP1/EP4 ownership")

    @property
    def expert_ids(self):
        count = self.total // self.ep_size
        return tuple(range(self.ep_rank * count, (self.ep_rank + 1) * count))

    def verify_native(self, native_ids, *, etp, dense_dp, expert_dp):
        if tuple(native_ids) != self.expert_ids:
            raise ValueError(
                "native local expert indices differ from original contiguous ownership"
            )
        if (etp, dense_dp, expert_dp) != (1, self.ep_size, 1):
            raise ValueError("expected ETP1, dense DP=EP and expert DP1 for this PP2 layout")


def load_original_experts(source, prefix, partition, device):
    """Read only locally owned original packed codes/scales, without dequantizing."""
    import torch

    from rl_engine.contracts.operators.expert_training import PackedExpertWeight

    names = []

    def matrix(stem):
        codes, scales = [], []
        for expert in partition.expert_ids:
            key = f"{prefix}.ffn.experts.{expert}.{stem}"
            pair = []
            for suffix in [".weight", ".scale"]:
                name = key + suffix
                value = source.get_tensor(name)
                if value.element_size() != 1 or value.requires_grad:
                    raise ValueError("original frozen one-byte expert code/scale required")
                names.append(name)
                pair.append(value.view(torch.uint8).to(device))
            codes.append(pair[0])
            scales.append(pair[1])
        code, scale = torch.stack(codes), torch.stack(scales)
        packed = PackedExpertWeight(code, scale, (code.shape[0], code.shape[1], code.shape[2] * 2))
        packed.validate(packed.shape, code.device)
        return packed

    gate, up, down = matrix("w1"), matrix("w3"), matrix("w2")
    if gate.shape != up.shape or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1]):
        raise ValueError("original gate/up/down expert geometry differs")
    fused = PackedExpertWeight(
        torch.cat((gate.codes, up.codes), dim=1),
        torch.cat((gate.scales, up.scales), dim=1),
        (gate.shape[0], 2 * gate.shape[1], gate.shape[2]),
    )
    return (
        fused,
        down,
        dict(
            ep_size=partition.ep_size,
            ep_rank=partition.ep_rank,
            original_expert_ids=partition.expert_ids,
            loaded_original_tensor_names=names,
            scope="local packed expert loader only; native EP communication not yet qualified",
        ),
    )
