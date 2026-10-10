# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import sys
import types

import torch

from rl_engine.platforms.constants import constants


def _install_fake_flashinfer(monkeypatch, capture):
    """Inject a stub ``flashinfer.sampling`` so the FlashInfer code path runs on CPU.

    ``top_p_sampling_from_probs`` records the probabilities it receives so the test
    can assert how temperature was applied before sampling.
    """
    fi = types.ModuleType("flashinfer")
    fi_sampling = types.ModuleType("flashinfer.sampling")

    def top_k_renorm_probs(probs, top_k):
        return probs

    def top_p_sampling_from_probs(probs, top_p, deterministic=True):
        capture["probs"] = probs.clone()
        return torch.zeros(probs.shape[0], dtype=torch.long)

    fi_sampling.top_k_renorm_probs = top_k_renorm_probs
    fi_sampling.top_p_sampling_from_probs = top_p_sampling_from_probs
    fi.sampling = fi_sampling

    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setitem(sys.modules, "flashinfer.sampling", fi_sampling)


def test_flashinfer_temperature_applied_once(monkeypatch):
    """Regression: the FlashInfer path must scale logits by 1/T exactly once.

    Previously temperature was divided once for all backends and a second time
    inside the FlashInfer branch, yielding softmax(logits / T**2).
    """
    capture = {}
    _install_fake_flashinfer(monkeypatch, capture)

    from rl_engine.ops.sampling.api import SamplerBackend

    sampler = SamplerBackend()
    # Force the FlashInfer path regardless of the detected hardware.
    sampler.backend = constants.BackendLib.FLASHINFER.value

    torch.manual_seed(0)
    logits = torch.randn(4, 16)
    temperature = 2.0

    sampler.sample(logits, top_p=0.9, temperature=temperature)

    expected = torch.softmax(logits.float() / temperature, dim=-1)
    assert torch.allclose(
        capture["probs"], expected, atol=1e-6
    ), "FlashInfer path must apply temperature exactly once (got T**2 scaling)"


def test_flashinfer_temperature_one_is_unchanged(monkeypatch):
    """temperature == 1.0 must leave logits unscaled (both divisions are skipped)."""
    capture = {}
    _install_fake_flashinfer(monkeypatch, capture)

    from rl_engine.ops.sampling.api import SamplerBackend

    sampler = SamplerBackend()
    sampler.backend = constants.BackendLib.FLASHINFER.value

    torch.manual_seed(0)
    logits = torch.randn(4, 16)

    sampler.sample(logits, top_p=0.9, temperature=1.0)

    expected = torch.softmax(logits.float(), dim=-1)
    assert torch.allclose(capture["probs"], expected, atol=1e-6)
