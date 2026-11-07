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


def strict_det_gemm_available() -> bool:
    """The backward GEMMs refuse to run unless a strict det-GEMM backend is ready."""

    from rl_engine.backends.cuda.gemm.det_gemm import det_gemm_backend
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if det_gemm_backend() == "cublaslt_nosplitk":
        return True
    marker = getattr(_C, "det_gemm_sm90_compiled", None) if _EXT_AVAILABLE else None
    return callable(marker) and bool(marker())


requires_strict_det_gemm = pytest.mark.skipif(
    not (nemotron_sm90_available() and strict_det_gemm_available()),
    reason="Nemotron-H lm_head_logprob backward needs a strict det-GEMM backend: build with "
    "KERNEL_ALIGN_DET_GEMM_SM90=1, or export CUBLAS_WORKSPACE_CONFIG=:16:8 and "
    "CUBLASLT_WORKSPACE_SIZE=1 before the process starts",
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
