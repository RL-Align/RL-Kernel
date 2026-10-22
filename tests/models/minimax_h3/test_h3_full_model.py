# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from types import SimpleNamespace

import pytest
import torch

from rl_engine.validation.models import h3_block
from rl_engine.validation.models.h3_full_model import (
    FullModelTrace,
    full_model_execution,
    validate_full_structure,
)


class Block(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, x, *args):
        return x + self.weight


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_layers=50,
            num_refiner_layers=2,
            hidden_size=5376,
            num_attention_heads=56,
            attention_head_dim=128,
        )
        self.transformer_blocks = torch.nn.ModuleList([Block(i) for i in range(50)])
        for name in [
            "proj_in",
            "audio_proj_in",
            "context_embedder",
            "time_proj",
            "time_embedder",
            "token_refiner",
            "norm_out",
            "proj_out",
            "audio_proj_out",
        ]:
            setattr(self, name, torch.nn.Identity())
        self.token_refiner.refiner_blocks = torch.nn.ModuleList([Block(50), Block(51)])


def test_full_stack_uses_each_layers_own_parameters_and_keeps_gradients(monkeypatch):
    model = Model()
    seen = []

    def run(mode, ctx, ops):
        p = ctx["weights"][h3_block.BLOCK + ".weight"]
        seen.append(p.item())
        return {h3_block.OUTPUT: ctx["hidden"] + p}

    monkeypatch.setattr(h3_block, "run_graph", run)
    original = model.transformer_blocks[0].forward
    with full_model_execution(model, candidate=True, trace=FullModelTrace(), ops=object()):
        x = torch.ones(1, requires_grad=True)
        for block in model.transformer_blocks:
            x = block(x, None, None, None)
        x.sum().backward()
        assert x.item() == 1 + sum(range(50))
        assert seen == list(range(50))
        assert all(b.weight.grad.item() == 1 for b in model.transformer_blocks)
    assert model.transformer_blocks[0].forward == original


def test_repeated_block_zero_is_not_full_model():
    model = Model()
    model.transformer_blocks[49] = model.transformer_blocks[0]
    with pytest.raises(ValueError, match="distinct"):
        validate_full_structure(model)


def test_partial_execution_cannot_publish_full_model_pass():
    trace = FullModelTrace()
    trace.observe("block.0", torch.ones(1))
    with pytest.raises(ValueError, match="incomplete"):
        trace.validate()


def test_binding_restored_after_failure():
    model = Model()
    original = model.transformer_blocks[12].forward
    with pytest.raises(RuntimeError, match="injected"):
        with full_model_execution(model, candidate=True, trace=FullModelTrace(), ops=object()):
            raise RuntimeError("injected")
    assert model.transformer_blocks[12].forward == original


def test_candidate_execution_requires_every_node():
    trace = FullModelTrace()
    trace.candidate = True
    for name in [f"block.{i}" for i in range(50)] + [
        "refiner.0",
        "refiner.1",
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "token_refiner",
        "time_embedder",
        "norm_out",
        "proj_out",
        "audio_proj_out",
    ]:
        trace.calls[name] = 1
    with pytest.raises(ValueError, match="candidate nodes did not execute"):
        trace.validate()


def test_drift_order_follows_execution_instead_of_lexical_names():
    from tools.validation.models.h3_full_compare import drift_order

    report = {
        "trace": {
            "calls": {"block.2": 1, "block.10": 1},
            "backward_calls": {"block.10": 1, "block.2": 1},
        }
    }
    names = [
        "video",
        "nodes.block.10",
        "nodes.block.2",
        "nodes.block.2.grad",
        "nodes.block.10.grad",
    ]
    assert drift_order(report, names) == [
        "nodes.block.2",
        "nodes.block.10",
        "nodes.block.10.grad",
        "nodes.block.2.grad",
        "video",
    ]


def test_reference_inventory_rejects_tampered_tensor(tmp_path):
    import hashlib
    import json

    from tools.validation.models.h3_evidence_integrity import verify_inventory

    tensor = tmp_path / "tensor.pt"
    tensor.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": {"tensor.pt": hashlib.sha256(b"original").hexdigest()}})
    )
    verify_inventory(tmp_path, manifest)
    tensor.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_inventory(tmp_path, manifest)


def test_cuda_guard_rejects_one_fallback_among_native_bindings():
    from rl_engine.validation.models.h3_full_model import require_candidate_cuda

    native = "rl_engine.backends.cuda.model_specific.minimax_h3.NativeOp"
    ops = SimpleNamespace(
        backends=lambda: {"projection": native, "norm": "torch.nn.RMSNorm", "gate": native}
    )
    with pytest.raises(ValueError, match="candidate norm fell back"):
        require_candidate_cuda(ops)


def test_drift_order_interleaves_parameter_and_activation_backward_hooks():
    from tools.validation.models.h3_full_compare import drift_order

    trace = FullModelTrace()
    for name in (
        "nodes.block.2",
        "nodes.block.10",
        "nodes.block.10.grad",
        "parameter.block.10.weight",
        "nodes.block.2.grad",
    ):
        trace.record_event(name)
    trace.record_event("nodes.block.2")  # checkpoint recomputation is not a new first event
    names = list(reversed(trace.execution_order))
    assert drift_order({"trace": {"execution_order": trace.execution_order}}, names) == (
        trace.execution_order
    )


def test_strict_boundaries_apply_to_candidate_execution_only():
    from rl_engine.validation.models.h3_full_model import (
        PROVIDER_BOUNDARIES,
        STRICT_BOUNDARY_ROWS,
    )

    with pytest.raises(ValueError, match="candidate execution only"):
        with full_model_execution(
            Model(), candidate=False, trace=FullModelTrace(), strict_boundaries=True
        ):
            pass
    # every boundary module is either bound to a row or explicitly left on the provider
    assert set(STRICT_BOUNDARY_ROWS) | set(PROVIDER_BOUNDARIES) == {
        "proj_in", "audio_proj_in", "context_embedder", "token_refiner",
        "time_embedder", "norm_out", "proj_out", "audio_proj_out",
    }
