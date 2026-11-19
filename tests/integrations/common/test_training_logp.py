# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared training scoring without an orchestrator or accelerator runtime."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.common.training_logp import (
    LinearLogpProviderUnavailable,
    TrainingLogpInputs,
    _metric_entropy_from_strict_lse,
    compute_training_logp,
)


def _inputs(**overrides):
    values = dict(
        logits=torch.tensor(
            [[0.25, -0.5, 1.0, 0.1, -0.3, 0.6, -0.7, 0.4]] * 3,
            requires_grad=True,
        ),
        target_ids=torch.tensor([2, 5, 0]),
        real_vocab_size=7,
        padded_vocab_size=8,
        cp_rank=0,
        cp_world_size=1,
        cp_layout="single",
        num_vocab_tiles=4,
    )
    values.update(overrides)
    return TrainingLogpInputs(**values)


@pytest.mark.parametrize("with_entropy_grad", [False, True])
@pytest.mark.parametrize("cp_rank,cp_world_size,cp_layout", [(0, 1, "single"), (1, 2, "zigzag")])
def test_direct_scoring_and_gradients(with_entropy_grad, cp_rank, cp_world_size, cp_layout):
    inputs = _inputs(
        with_entropy=True,
        with_entropy_grad=with_entropy_grad,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
        cp_layout=cp_layout,
    )
    result = compute_training_logp(inputs)
    reference = inputs.logits.detach().clone().requires_grad_(True)
    log_probs = reference[:, :7].log_softmax(-1)
    logp = log_probs[torch.arange(3), inputs.target_ids]
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    torch.testing.assert_close(result.logp[:, 0], logp)
    torch.testing.assert_close(result.entropy, entropy)
    assert result.entropy.requires_grad == with_entropy_grad
    (result.logp.sum() + result.entropy.sum()).backward()
    (logp.sum() + (entropy.sum() if with_entropy_grad else entropy.detach().sum())).backward()
    torch.testing.assert_close(inputs.logits.grad, reference.grad)
    assert torch.count_nonzero(inputs.logits.grad[:, 7:]) == 0
    assert result.provenance["execution"]["role"] == "training_linear_logp"
    assert result.provenance["token_row_ownership"]["is_merge_axis"] is False
    assert result.provenance["token_row_ownership"]["rank"] == cp_rank


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"padded_vocab_size": 16}, "cover padded_vocab_size"),
        ({"tp_rank": 1}, "disagrees with TP group"),
        ({"tp_world_size": 2}, "disagrees with TP group"),
        ({"cp_rank": 2}, "invalid for CP"),
        ({"cp_layout": "zigzag"}, "local token ownership"),
        ({"vocab_start_index": 1}, "rank-contiguous"),
        ({"num_vocab_tiles": 3}, "must divide"),
        ({"real_vocab_size": 9}, "must not exceed"),
    ],
)
def test_rejects_incompatible_sharding(overrides, message):
    with pytest.raises(LinearLogpProviderUnavailable, match=message):
        compute_training_logp(_inputs(**overrides))


class _CPUScorer:
    """Exercise wrapper arguments and autograd without claiming hardware coverage."""

    backend_id = "test.cpu"
    provenance = {"actual_backend": backend_id}

    def __call__(self, hidden, weight, targets, bias, **kwargs):
        logits = torch.nn.functional.linear(hidden, weight, bias)
        return self.from_local_logits(logits, targets, **kwargs)

    def from_local_logits(self, logits, targets, **kwargs):
        values = logits[:, : kwargs["real_vocab_size"]].float()
        if kwargs.get("temperature") is not None:
            temperature = kwargs["temperature"]
            divisor = (
                temperature.reshape(-1, 1) if isinstance(temperature, torch.Tensor) else temperature
            )
            values = values / divisor
        logp = values.log_softmax(-1)[torch.arange(len(targets)), targets]
        return (logp, values.logsumexp(-1)) if kwargs.get("return_lse") else logp


@pytest.mark.parametrize("reuse_logits", [False, True])
@pytest.mark.parametrize(
    "temperature",
    [
        1.0,
        0.7,
        torch.tensor(0.7),
        torch.tensor([0.5, 0.75, 1.25]),
        torch.tensor([[0.5], [0.75], [1.25]]),
    ],
)
def test_projection_and_reused_logits_preserve_autograd(reuse_logits, temperature):
    generator = torch.Generator().manual_seed(35)
    hidden = torch.randn(3, 4, generator=generator, requires_grad=True)
    weight = torch.randn(8, 4, generator=generator, requires_grad=True)
    logits = hidden @ weight.T
    inputs = _inputs(
        logits=logits,
        hidden=hidden,
        weight=weight,
        vocab_start_index=0,
        local_logits=logits,
        reuse_local_logits=reuse_logits,
        temperature=temperature,
        local_logits_temperature=temperature,
        with_entropy=True,
        with_entropy_grad=True,
    )
    result = compute_training_logp(inputs, linear_logp=_CPUScorer())
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    divisor = temperature.reshape(-1, 1) if isinstance(temperature, torch.Tensor) else temperature
    values = (reference_hidden @ reference_weight.T)[:, :7] / divisor
    log_probs = values.log_softmax(-1)
    logp = log_probs[torch.arange(3), inputs.target_ids]
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    torch.testing.assert_close(result.logp[:, 0], logp)
    torch.testing.assert_close(result.entropy, entropy)
    (result.logp.sum() + result.entropy.sum()).backward()
    (logp.sum() + entropy.sum()).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(weight.grad, reference_weight.grad)
    assert result.provenance["execution"]["lm_head_result_reused"] == reuse_logits


def test_metric_entropy_reuses_lse_without_gradients():
    inputs = _inputs(
        hidden=torch.zeros(3, 4),
        weight=torch.zeros(8, 4),
        vocab_start_index=0,
        reuse_local_logits=True,
        with_entropy=True,
    )
    inputs = replace(inputs, local_logits=inputs.logits)
    before = inputs.logits.detach().clone()
    probabilities = inputs.logits[:, :7].log_softmax(-1)
    result = compute_training_logp(inputs, linear_logp=_CPUScorer())
    torch.testing.assert_close(result.entropy, -(probabilities.exp() * probabilities).sum(-1))
    assert torch.equal(inputs.logits, before)
    repeated = compute_training_logp(inputs, linear_logp=_CPUScorer())
    assert torch.equal(result.logp, repeated.logp)
    assert torch.equal(result.entropy, repeated.entropy)
    assert not result.entropy.requires_grad
    assert result.provenance["execution"]["entropy"]["strict_lse_reused"] is True
    result.logp.sum().backward()
    assert inputs.logits.grad is not None


def test_common_and_megatron_import_without_orchestrators():
    code = """
import importlib.abc
import sys
class RejectOrchestrators(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("rl_engine.integrations.orchestrators", "vime", "verl")):
            raise AssertionError("reverse dependency: " + fullname)
sys.meta_path.insert(0, RejectOrchestrators())
from rl_engine.integrations.common.training_logp import compute_training_logp
from rl_engine.integrations.engines.train.megatron.runtime import install_megatron_integration
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("group", [None, SimpleNamespace(rank=lambda: 0, size=lambda: 1)])
def test_metric_entropy_does_not_use_unrelated_distributed_ranks(monkeypatch, group):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)

    def unexpected_gather(*args, **kwargs):
        raise AssertionError("entropy must not gather over DP/CP ranks")

    monkeypatch.setattr(dist, "all_gather", unexpected_gather)
    logits = torch.tensor([[0.0, 1.0], [0.5, -0.5]])
    entropy = _metric_entropy_from_strict_lse(
        logits, logits.logsumexp(-1), vocab_start_index=0, real_vocab_size=2, tp_group=group
    )
    log_probs = logits.log_softmax(-1)
    torch.testing.assert_close(entropy, -(log_probs.exp() * log_probs).sum(-1))


def test_metric_entropy_reduces_only_the_explicit_tp_group(monkeypatch):
    import torch.distributed as dist

    group = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [1.0, -1.0, 0.0, 0.5]])
    log_probs = logits.log_softmax(-1)
    partials = [-(log_probs[:, i : i + 2].exp() * log_probs[:, i : i + 2]).sum(-1) for i in (0, 2)]

    def gather(outputs, local, *, group):
        assert group is expected_group
        torch.testing.assert_close(local, partials[1])
        for output, partial in zip(outputs, partials):
            output.copy_(partial)

    expected_group = group
    monkeypatch.setattr(dist, "all_gather", gather)
    result = _metric_entropy_from_strict_lse(
        logits[:, 2:], logits.logsumexp(-1), vocab_start_index=2, real_vocab_size=4, tp_group=group
    )
    assert torch.equal(result, partials[0] + partials[1])


def test_shared_scorer_uses_explicit_tiles_not_orchestrator_environment(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_LOGPROB_NUM_VOCAB_TILES", "3")
    result = compute_training_logp(_inputs(num_vocab_tiles=4))
    assert result.provenance["num_vocab_tiles"] == 4
