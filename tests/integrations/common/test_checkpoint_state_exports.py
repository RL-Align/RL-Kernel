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
