# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Nemotron-H lm_head_logprob CUDA suites."""
from __future__ import annotations

import pytest
import torch

from rl_engine.models.nemotron_h.lm_head_contract import NEMOTRON_H_LM_HEAD_SPEC

HIDDEN = NEMOTRON_H_LM_HEAD_SPEC.hidden_size
VOCAB = NEMOTRON_H_LM_HEAD_SPEC.vocab_size
# Matches the HF initializer_range so logits have a realistic O(1) spread.
WEIGHT_STD = 0.02


def nemotron_sm90_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    from rl_engine.backends.cuda.logprob.nemotron_h_lm_head_logp import missing_strict_symbols

    return not missing_strict_symbols()


requires_nemotron_sm90 = pytest.mark.skipif(
    not nemotron_sm90_available(),
    reason="Nemotron-H strict lm_head_logprob needs a Hopper GPU and the extension built "
    "with KERNEL_ALIGN_FORCE_SM90=1",
)


def make_hidden(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)


def make_targets(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, VOCAB, (rows,), generator=gen, device="cuda")


def _make_weight(seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda", dtype=torch.float32)
    return weight.mul_(WEIGHT_STD).to(torch.bfloat16)


@pytest.fixture(scope="module")
def lm_head_weight() -> torch.Tensor:
    return _make_weight(1)


@pytest.fixture(scope="module")
def embed_weight() -> torch.Tensor:
    return _make_weight(2)


@pytest.fixture(scope="module")
def strict_op():
    from rl_engine.backends.cuda.logprob.nemotron_h_lm_head_logp import NemotronHLMHeadLogprobSM90Op

    return NemotronHLMHeadLogprobSM90Op()
