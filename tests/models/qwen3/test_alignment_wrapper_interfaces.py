# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Common invocation-surface tests for Attention, FFN, and logprob wrappers."""

from __future__ import annotations

import inspect

import pytest
import torch

from rl_engine.contracts.operators.logprob import (
    LogprobContract,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.ops.ffn import Qwen3FFNOp
from rl_engine.reference.attention import AttentionAblationOp
from rl_engine.reference.logprob.vocab_parallel_logp import VocabParallelLogprobOp
from rl_engine.runtime.operators import selected_logprobs_with_operator
from rl_engine.runtime.registry import KernelRegistry
from rl_engine.runtime.semantic_registry import OperatorRequirements


def _logprob_contract(
    *, active: tuple[bool, ...], determinism_scope: str = "cross_tp_bitwise"
) -> LogprobContract:
    return LogprobContract(
        role="train",
        dtype="fp32",
        mask=MaskSpec(num_tokens=len(active), active_mask=active),
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, 8),),
            real_vocab_size=7,
            padded_vocab_size=8,
        ),
        reduction=ReductionSpec(determinism_scope=determinism_scope),
    )


@pytest.mark.parametrize(
    "wrapper",
    [AttentionAblationOp, Qwen3FFNOp, VocabParallelLogprobOp],
)
def test_alignment_wrappers_share_the_deterministic_switch(wrapper):
    parameters = inspect.signature(wrapper.__call__).parameters

    assert "deterministic" in parameters
    assert parameters["deterministic"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["deterministic"].default in {None, True}


@pytest.mark.parametrize(
    ("wrapper", "backend_id"),
    [
        (AttentionAblationOp, "rlkernel.attention.deterministic.v1"),
        (Qwen3FFNOp, "rlkernel.ffn.qwen3.deterministic.v1"),
        (VocabParallelLogprobOp, "pytorch-vocab-parallel-logp-ws2"),
    ],
)
def test_alignment_wrapper_backend_ids_are_stable(wrapper, backend_id):
    assert wrapper.backend_id == backend_id


@pytest.mark.parametrize("deterministic", [True, False])
def test_contract_aware_logprob_bridge_accepts_logp_lse_result(deterministic):
    active = (True, False, True, True)
    contract = _logprob_contract(
        active=active,
        determinism_scope="cross_tp_bitwise" if deterministic else "fixed_topology",
    )
    logits = torch.tensor(
        [
            [[1.0, 2.0, 0.0, -1.0, 3.0, 0.5, -0.5, 100.0], [0.0] * 8],
            [[2.0, 0.0, 1.0, -2.0, 0.5, 3.0, -1.0, 100.0], [0.0] * 8],
        ]
    )
    targets = torch.tensor([[4, -100], [5, 0]])
    mask = torch.tensor(active).reshape_as(targets)

    actual = selected_logprobs_with_operator(
        VocabParallelLogprobOp(),
        logits,
        targets,
        active_mask=mask,
        contract=contract,
        num_vocab_tiles=4,
        deterministic=deterministic,
    )

    real_logits = logits[..., :7]
    safe_targets = targets.masked_fill(~mask, 0)
    expected = torch.gather(
        torch.log_softmax(real_logits, dim=-1),
        -1,
        safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    expected = expected.masked_fill(~mask, 0.0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_contract_aware_logprob_bridge_rejects_a_different_active_mask():
    contract = _logprob_contract(active=(True, False))

    with pytest.raises(ValueError, match="active_mask must match"):
        selected_logprobs_with_operator(
            VocabParallelLogprobOp(),
            torch.zeros((2, 8)),
            torch.tensor([0, 0]),
            active_mask=torch.tensor([True, True]),
            contract=contract,
            num_vocab_tiles=4,
        )


def test_semantic_catalog_exposes_exact_rlkernel_and_native_axes():
    catalog = KernelRegistry().semantic

    for semantic_op, backend_id in (
        ("attention", "rlkernel.attention.deterministic.v1"),
        ("ffn", "rlkernel.ffn.qwen3.deterministic.v1"),
        ("selected_logprob", "pytorch-vocab-parallel-logp-ws2"),
    ):
        rlkernel = catalog.backend_descriptor(semantic_op, backend_id)
        native = catalog.backend_descriptor(semantic_op, "native")
        assert rlkernel is not None
        assert rlkernel.determinism_or_alignment_properties["deterministic"] is True
        assert rlkernel.is_strictly_observable
        assert native is not None
        assert native.fallback_policy.value == "runtime_managed"


def test_semantic_ffn_and_logprob_wrappers_are_instantiable():
    catalog = KernelRegistry().semantic
    session = catalog.session()

    ffn = session.instantiate(
        session.resolve(
            semantic_op="ffn",
            requested_backend="rlkernel.ffn.qwen3.deterministic.v1",
            target="training",
            requirements=OperatorRequirements(device="cuda", dtype="bfloat16"),
        )
    )
    logprob = session.instantiate(
        session.resolve(
            semantic_op="selected_logprob",
            requested_backend="pytorch-vocab-parallel-logp-ws2",
            target="training",
            requirements=OperatorRequirements(device="cpu", dtype="float32"),
        )
    )

    assert isinstance(ffn, Qwen3FFNOp)
    assert isinstance(logprob, VocabParallelLogprobOp)
