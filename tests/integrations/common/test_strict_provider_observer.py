# SPDX-License-Identifier: Apache-2.0

import pytest

from rl_engine.contracts.strict_provider import CallContext, KernelPlan, StrictProviderError
from rl_engine.integrations.common.provider_observer import CompletionReceipt, ResolvedCall


def plan():
    return KernelPlan.from_dict(
        {
            "node": "conv",
            "phase": "decode",
            "implementation": "fake.Conv",
            "entrypoint": "fake.conv",
            "code": {
                "source_inventory_sha256": "a" * 64,
                "binary_inventory_sha256": "b" * 64,
                "compiled_kernel_id": "binary1",
                "compile_key": "key1",
            },
            "io": {
                "input_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "layout": "BHC",
                "shape_class": "token1",
                "workspace_policy": "private",
            },
            "numerics": {
                "accumulator_dtype": "float32",
                "split_policy": "disabled",
                "split_count": 1,
                "reduction_spec": "fixed",
                "cast_spec": "output",
                "atomic_policy": "forbidden",
            },
            "launch": {"tile_config": {"block": 64}, "graph_state": "eager", "tuning": "fixed"},
        }
    )


def context():
    return CallContext.from_dict(
        {
            "case_id": "case0",
            "call_id": "decode0",
            "node": "conv",
            "phase": "decode",
            "role": "rollout",
            "worker_id": "w0",
            "run_id": "run0",
            "attempt": "1",
            "weight_snapshot_id": "s0",
            "layer_id": 0,
        }
    )


def test_resolved_call_preserves_preparation_metadata_and_single_use():
    metadata = {"state_slot": 7, "split": 1}
    call = ResolvedCall(plan(), lambda: 1, call_context=context(), recheck=lambda: metadata)
    call.claim("session0")
    call.check_unchanged()
    with pytest.raises(StrictProviderError) as error:
        call.claim("session1")
    assert error.value.code == "call_reused"
    metadata["split"] = 4
    with pytest.raises(StrictProviderError) as error:
        call.check_unchanged()
    assert error.value.code == "metadata_changed"


def test_completion_requires_a_checked_receipt():
    call = ResolvedCall(
        plan(),
        lambda: "output",
        call_context=context(),
        recheck=lambda: {},
        completion=lambda output: CompletionReceipt(False),
    )
    with pytest.raises(StrictProviderError) as error:
        call.wait_completed("output")
    assert error.value.code == "incomplete_call"


def test_unknown_mutable_dispatch_cannot_be_hidden_as_diagnostic():
    runtime = {"entrypoint": "fake.conv", "shape": [1, 4], "state_generation": 1}
    call = ResolvedCall(plan(), lambda: "output", call_context=context(), recheck=lambda: runtime)
    runtime["state_generation"] = 2
    with pytest.raises(StrictProviderError):
        call.check_unchanged()
