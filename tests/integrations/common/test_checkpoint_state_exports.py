# SPDX-License-Identifier: Apache-2.0
"""Actual Module placement replaces buffers while preserving ordinary Parameters."""

import pytest
import torch

from rl_engine.integrations.engines.train.megatron.checkpoint_state import checkpoint_export_value


def model():
    root = torch.nn.Module()
    root.router = torch.nn.Module()
    root.router.weight = torch.nn.Parameter(torch.ones(2, 2))
    root.router.register_buffer("expert_bias", torch.tensor([0.125, -0.25]))
    return root


def test_export_tracks_buffer_replaced_by_module_conversion_and_updated_in_place():
    root = model()
    stale = root.router.expert_bias
    export = checkpoint_export_value(root, stale)
    root.to(dtype=torch.float64)
    assert root.router.expert_bias is not stale
    root.router.expert_bias.add_(0.001)
    assert export() is root.router.expert_bias
    assert export().dtype == torch.float64
    assert torch.equal(stale, torch.tensor([0.125, -0.25]))
    assert not torch.equal(export(), stale.double())


def test_parameter_export_preserves_the_optimizer_registered_identity():
    root = model()
    parameter = root.router.weight
    export = checkpoint_export_value(root, parameter)
    root.to(dtype=torch.float64)
    assert export is parameter is root.router.weight


def test_export_resolves_the_current_module_at_the_qualified_buffer_path():
    root = model()
    export = checkpoint_export_value(root, root.router.expert_bias)
    replacement = torch.nn.Module()
    replacement.register_buffer("expert_bias", torch.tensor([0.5, 0.75]))
    root.router = replacement
    assert export() is replacement.expert_bias


@pytest.mark.parametrize("alias", [False, True])
def test_unregistered_or_ambiguous_tensor_is_rejected(alias):
    root = model()
    tensor = torch.zeros(2)
    if alias:
        root.register_buffer("first", tensor)
        root.register_buffer("second", tensor)
    with pytest.raises(ValueError, match="one registered buffer"):
        checkpoint_export_value(root, tensor)


def test_removed_buffer_is_not_exported_as_stale_success():
    root = model()
    export = checkpoint_export_value(root, root.router.expert_bias)
    root.router.expert_bias = None
    with pytest.raises(RuntimeError, match="no longer available"):
        export()


@pytest.mark.parametrize("parameter", [False, True])
def test_original_i64_routing_survives_native_i32_destination_and_readback(parameter):
    from rl_engine.integrations.engines.train.megatron.checkpoint_state import copy_original_tensor

    root = model()
    destination = torch.zeros((2, 2), dtype=torch.int32)
    if parameter:
        root.router.tid2eid = torch.nn.Parameter(destination, requires_grad=False)
    else:
        root.router.register_buffer("tid2eid", destination)
    destination = root.router.tid2eid
    export = checkpoint_export_value(root, destination)
    source = torch.tensor([[1, 2**40], [-(2**40), 255]], dtype=torch.int64)
    copy_original_tensor(destination, source, "layers.0.ffn.gate.tid2eid")
    assert root.router.tid2eid is destination
    actual = export if parameter else export()
    assert actual.dtype == source.dtype
    assert actual.numpy().tobytes() == source.numpy().tobytes()
    source.zero_()
    assert actual[0, 1] == 2**40
    with pytest.raises(TypeError, match="original dtype mismatch"):
        copy_original_tensor(torch.zeros(2), torch.zeros(2, dtype=torch.float64), "norm.weight")


def test_native_rollout_router_keeps_the_registered_i64_parameter_reference():
    from types import SimpleNamespace
    from rl_engine.integrations.engines.rollout.vllm.hash_table import preserve_original_hash_dtype

    gate = torch.nn.Module()
    gate.tid2eid = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.int32), requires_grad=False)
    native_router_reference = gate.tid2eid
    preserve_original_hash_dtype(SimpleNamespace(gate=gate))
    assert gate.tid2eid is native_router_reference
    original = torch.tensor([[1, 2**40], [255, 0]], dtype=torch.int64)
    gate.tid2eid.copy_(original)
    assert native_router_reference.dtype == torch.int64
    assert torch.equal(native_router_reference, original)
