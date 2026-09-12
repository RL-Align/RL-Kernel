# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU coverage for the Vime ``linear_logp`` adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations import framework_operators
from rl_engine.integrations.framework_operators import MegatronLogpOperator
from rl_engine.integrations.ablation import IntegrationPlan
from rl_engine.integrations.megatron import MegatronIntegration
from rl_engine.integrations.state import (
    clear_active_integration,
    set_active_integration,
)
from rl_engine.integrations.vime.linear_logp_provider import (
    LinearLogpProviderUnavailable,
    LinearLogpResult,
    provider,
)


@pytest.fixture(autouse=True)
def _select_rlkernel_logp_route(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_LOGP_CASE", "R/R")


def _request(*, cp_rank: int = 0, with_entropy: bool = False, keep_mask=None):
    logits = torch.tensor(
        [[0.25, -0.5, 1.0, 0.1, -0.3, 0.6, -0.7, 0.4] for _ in range(3)],
        dtype=torch.float32,
        requires_grad=True,
    )
    return SimpleNamespace(
        logits=logits,
        target_ids=torch.tensor([2, 5, 0]),
        tensor_parallel_group=None,
        token_layout=SimpleNamespace(world_size=2, rank=cp_rank, layout="zigzag"),
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy,
        log_prob_keep_mask=keep_mask,
        metadata={
            "real_vocab_size": 7,
            "padded_vocab_size": 8,
            "tp_rank": 0,
            "tp_world_size": 1,
            "num_vocab_tiles": 4,
        },
    )


def _structural_request(*, with_entropy: bool = False):
    logits = torch.randn(3, 8, requires_grad=True)
    return SimpleNamespace(
        logits=logits,
        target_ids=torch.tensor([2, 5, 0]),
        tensor_parallel_group=None,
        token_layout=SimpleNamespace(world_size=1, rank=0, layout="single"),
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy,
        log_prob_keep_mask=None,
        temperature=1.0,
        context=SimpleNamespace(
            hidden=torch.randn(3, 4, requires_grad=True),
            projection=SimpleNamespace(weight=torch.randn(8, 4), bias=torch.randn(8)),
            vocab_partition=SimpleNamespace(
                local_start=0, local_size=8, real_size=7, padded_size=8
            ),
        ),
        metadata={
            "real_vocab_size": 7,
            "padded_vocab_size": 8,
            "tp_rank": 0,
            "tp_world_size": 1,
            "num_vocab_tiles": 4,
        },
    )


def test_provider_runs_locally_with_cp2_row_metadata():
    request = _request(cp_rank=1)
    result = provider(request)
    reference = torch.log_softmax(request.logits[:, :7], dim=-1)[
        torch.arange(request.logits.size(0)), request.target_ids
    ]

    assert result.logp.shape == (3, 1)
    torch.testing.assert_close(result.logp.squeeze(-1), reference)
    assert result.backend_id == "pytorch-vocab-parallel-logp-ws2"
    assert result.provenance["token_row_ownership"] == {
        "rank": 1,
        "world_size": 2,
        "layout": "zigzag",
        "local_token_rows": 3,
        "is_merge_axis": False,
    }


def test_provider_entropy_preserves_vime_semantics_and_autograd():
    request = _request(with_entropy=True)
    result = provider(request)
    reference_logits = request.logits.detach().clone().requires_grad_(True)
    log_probs = torch.log_softmax(reference_logits[:, :7], dim=-1)
    reference_logp = log_probs[torch.arange(reference_logits.size(0)), request.target_ids]
    reference_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

    torch.testing.assert_close(result.logp.squeeze(-1), reference_logp)
    torch.testing.assert_close(result.entropy, reference_entropy)
    (result.logp.sum() + result.entropy.sum()).backward()
    (reference_logp.sum() + reference_entropy.sum()).backward()
    torch.testing.assert_close(request.logits.grad[:, :7], reference_logits.grad[:, :7])
    assert bool((request.logits.grad[:, 7] == 0).all())


def test_provider_structural_path_returns_linear_logp_result(monkeypatch):
    monkeypatch.setenv("VIME_RL_KERNEL_STRICT", "1")

    class FakeLinearLogp:
        backend_id = "fake-linear-logp"
        provenance = {"actual_backend": "fake-linear-logp"}

        def __call__(self, hidden, weight, target_ids, bias, **_kwargs):
            logits = hidden @ weight.transpose(0, 1)
            if bias is not None:
                logits = logits + bias
            return torch.log_softmax(logits[:, :7], dim=-1)[
                torch.arange(target_ids.size(0)), target_ids
            ]

        def from_local_logits(self, local_logits, target_ids, **_kwargs):
            return torch.log_softmax(local_logits[:, :7], dim=-1)[
                torch.arange(target_ids.size(0)), target_ids
            ]

    import rl_engine.integrations.vime.linear_logp_provider as provider_module

    monkeypatch.setattr(provider_module, "_default_strict_linear_logp", lambda: FakeLinearLogp())
    request = _structural_request()
    result = provider(request)

    assert isinstance(result, LinearLogpResult)
    assert result.logp.shape == (3, 1)
    assert result.provenance["execution"]["role"] == "vime_training_linear_logp"


def test_megatron_adapter_forwards_structured_context(monkeypatch):
    request = _structural_request()
    observed = {}

    def provider(actual_request, *, linear_logp):
        observed["context"] = actual_request.context
        observed["wrapper"] = linear_logp
        return LinearLogpResult(
            logp=actual_request.logits[:, :1],
            entropy=None,
            backend_id="fake-linear-logp",
            contract_id="fake-linear-logp.v1",
            provenance={},
        )

    wrapper = SimpleNamespace(backend_id="fake-linear-logp", provenance={})
    monkeypatch.setattr(framework_operators, "_require_nvidia_cuda", lambda *_args: None)
    result = MegatronLogpOperator(provider, linear_logp=wrapper)(request)

    assert observed["context"] is request.context
    assert observed["wrapper"] is wrapper
    assert result.logp.shape == (3, 1)


def test_provider_rejects_top_p_replay_without_changing_its_semantics():
    request = _request(keep_mask=torch.ones((3, 8), dtype=torch.bool))

    with pytest.raises(LinearLogpProviderUnavailable, match="top-p replay"):
        provider(request)


def test_provider_rejects_local_vocab_metadata_that_cannot_describe_tp_ownership():
    request = _request()
    request.metadata["padded_vocab_size"] = 16

    with pytest.raises(LinearLogpProviderUnavailable, match="cover padded_vocab_size"):
        provider(request)


def test_production_route_rejects_rlkernel_provider_configuration(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_LOGP_CASE", "P/P")
    integration = MegatronIntegration(
        IntegrationPlan.from_case_ids(logp="P/P"),
        rl_kernel_operators={},
    )
    clear_active_integration("megatron")
    set_active_integration("megatron", integration)
    try:
        with pytest.raises(RuntimeError, match="omit --linear-logp-provider"):
            provider(_request())
    finally:
        clear_active_integration("megatron")

    assert "logp" not in integration.readback()["operators"]
