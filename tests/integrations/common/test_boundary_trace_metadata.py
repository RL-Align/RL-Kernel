# SPDX-License-Identifier: Apache-2.0
"""Trace only tensor/primitive state, never native objects with local hooks."""

import io

import torch

from examples.operators.dsv4_4l.boundary_trace import metadata_snapshot


def test_real_module_handle_with_trace_hooks_is_not_pickled():
    module = torch.nn.Linear(2, 3)
    module.register_forward_pre_hook(lambda m, a: None)
    values = torch.tensor([0.0, -0.0, 0.125], dtype=torch.float32)
    captured = metadata_snapshot(dict(args=(module, values), kwargs=dict(callback=lambda: None)))
    assert captured["args"][1] is values
    assert isinstance(captured["args"][0], str)
    stream = io.BytesIO()
    torch.save(captured, stream)
    stream.seek(0)
    actual = torch.load(stream, weights_only=True)
    assert actual["args"][1].numpy().tobytes() == values.numpy().tobytes()
