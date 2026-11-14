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
        "time_proj",
        "token_refiner",
        "time_embedder",
        "norm_out",
        "proj_out",
        "audio_proj_out",
    ]:
        trace.calls[name] = 1
    with pytest.raises(ValueError, match="candidate nodes did not execute"):
        trace.validate()


BOUNDARIES = [
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "time_proj",
    "token_refiner",
    "time_embedder",
    "norm_out",
    "proj_out",
    "audio_proj_out",
]


def _complete_trace(*, skip=(), backward_skip=("time_proj",)):
    trace = FullModelTrace()
    for name in [f"block.{i}" for i in range(50)] + ["refiner.0", "refiner.1"] + BOUNDARIES:
        if name not in skip:
            trace.calls[name] = 1
        if name not in backward_skip:
            trace.backward_calls[name] = 1
    return trace


def test_a_run_that_never_calls_time_proj_is_incomplete():
    with pytest.raises(ValueError, match="time_proj"):
        _complete_trace(skip=("time_proj",)).validate()


def test_time_proj_is_required_forward_only():
    trace = _complete_trace()  # no backward event for time_proj: it has no gradient
    assert trace.validate(backward=True)["calls"]["time_proj"] == 1
    with pytest.raises(ValueError, match=r"norm_out\.backward"):
        _complete_trace(backward_skip=("time_proj", "norm_out")).validate(backward=True)


def test_the_trace_reports_the_nodes_a_diagnostic_hybrid_replayed_with_the_provider():
    assert _complete_trace().validate()["diagnostic_provider_nodes"] == []
    trace = FullModelTrace()
    with full_model_execution(
        Model(), candidate=True, trace=trace, ops=object(), provider_nodes=("swiglu", "attention")
    ):
        assert trace.provider_nodes == ("attention", "swiglu")
    hybrid = _complete_trace()
    hybrid.provider_nodes = ("swiglu",)
    # still a valid trace (the diagnostic has to complete), but marked as not a candidate result
    assert hybrid.validate()["diagnostic_provider_nodes"] == ["swiglu"]


class _FakeBlas:
    """Stands in for ``torch.backends.cuda.preferred_blas_library`` (getter and setter)."""

    def __init__(self, current="default"):
        self.current, self.calls = current, []

    def __call__(self, backend=None):
        if backend is None:
            return self.current
        self.calls.append(backend)
        self.current = backend


def test_the_cublaslt_variant_puts_the_blas_library_back_on_exit(monkeypatch):
    from rl_engine.validation.models.h3_full_model import provider_variant_context

    blas = _FakeBlas("default")
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", blas)
    with provider_variant_context("blas-cublaslt"):
        assert blas.current == "cublaslt"
    assert blas.current == "default"
    assert blas.calls == ["cublaslt", "default"]


def test_the_blas_library_is_restored_also_when_the_body_raises(monkeypatch):
    from rl_engine.validation.models.h3_full_model import provider_variant_context

    blas = _FakeBlas("cublas")
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", blas)
    with pytest.raises(RuntimeError, match="boom"):
        with provider_variant_context("blas-cublaslt"):
            raise RuntimeError("boom")
    assert blas.current == "cublas"


def test_the_default_variant_leaves_the_blas_library_alone(monkeypatch):
    from rl_engine.validation.models.h3_full_model import provider_variant_context

    blas = _FakeBlas("default")
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", blas)
    with provider_variant_context("default"):
        pass
    assert blas.calls == []


def test_the_variant_context_is_also_correct_with_the_real_library():
    from rl_engine.validation.models.h3_full_model import provider_variant_context

    before = torch.backends.cuda.preferred_blas_library()
    try:  # a torch build without cuBLASLt (the CPU wheel in CI) refuses to select it
        torch.backends.cuda.preferred_blas_library("cublaslt")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    finally:
        torch.backends.cuda.preferred_blas_library(before)
    with provider_variant_context("blas-cublaslt"):
        assert "cublaslt" in str(torch.backends.cuda.preferred_blas_library()).lower()
    assert torch.backends.cuda.preferred_blas_library() == before


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
    """Strict boundaries need candidate execution, and every boundary module has a row."""
    from rl_engine.validation.models.h3_full_model import STRICT_BOUNDARY_ROWS

    with pytest.raises(ValueError, match="candidate execution only"):
        with full_model_execution(
            Model(), candidate=False, trace=FullModelTrace(), strict_boundaries=True
        ):
            pass
    # every boundary module is bound to a row
    assert set(STRICT_BOUNDARY_ROWS) == {
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "token_refiner",
        "time_embedder",
        "norm_out",
        "proj_out",
        "audio_proj_out",
    }


class _Embedder(torch.nn.Module):
    """The provider's ``TimestepEmbedding`` surface the binding reads."""

    def __init__(self, **extra):
        """A stand-in with the layers the binding reads; ``extra`` sets further attributes."""
        super().__init__()
        self.linear_1 = torch.nn.Linear(4, 6)
        self.linear_2 = torch.nn.Linear(6, 3)
        self.act = torch.nn.SiLU()
        self.cond_proj = None
        self.post_act = None
        for name, value in extra.items():
            setattr(self, name, value)

    def forward(self, sample, condition=None):
        raise AssertionError("the provider forward must not run in a strict run")


class _FinalNorm(torch.nn.Module):
    """The provider's ``MiniMaxH3AdaLayerNormOut`` surface the binding reads."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.RMSNorm(4, eps=1.25e-6)
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, hidden_states, temb, timestep_indices):
        raise AssertionError("the provider forward must not run in a strict run")


def _fake_rows(monkeypatch, called, keyed_calls=None):
    """Patch the registry with fake rows that record their calls in ``called``."""
    from rl_engine.runtime.registry import KernelRegistry

    keyed_calls = [] if keyed_calls is None else keyed_calls

    class Row:
        def __init__(self, row, method):
            self.row = row
            if method:
                self.forward = self._run

        def _run(self, x, weight, bias, **keyed):
            from rl_engine.ops.autograd.canonical_backward import active_session

            called.append((self.row, tuple(x.shape)))
            keyed_calls.append((self.row, keyed, active_session() is not None))
            return torch.nn.functional.linear(x, weight, bias)

        __call__ = _run

    class TimestepMLP:
        def forward(self, *args):
            called.append(("timestep_mlp_fp32", args))
            return torch.zeros(args[0].shape[0], 3)

    class FinalAdaLN:
        def forward(self, *args):
            called.append(("final_adaln_out", args))
            return args[0]

    def get_op(self, row, device=None):
        """The fake row for ``row``: an object with ``forward``, or a plain callable for video."""
        if row == "timestep_mlp_fp32":
            return TimestepMLP()
        if row == "final_adaln_out":
            return FinalAdaLN()
        # the audio/text rows expose ``forward``; the video rows are plain callables
        return Row(row, not row.startswith("video_"))

    monkeypatch.setattr(KernelRegistry, "get_op", get_op)
    # the fakes stand in for the CUDA ops strict mode requires
    from rl_engine.validation.models import h3_full_model as fm

    for row, fake in (("timestep_mlp_fp32", TimestepMLP), ("final_adaln_out", FinalAdaLN)):
        monkeypatch.setitem(fm._CUDA_ROW_OPS, row, f"{fake.__module__}.{fake.__qualname__}")


def _boundary_model():
    """A model whose eight boundary modules are small stand-ins."""
    model = Model()
    for name in ("proj_in", "audio_proj_in", "context_embedder", "proj_out", "audio_proj_out"):
        setattr(model, name, torch.nn.Linear(4, 3))
    model.time_embedder = _Embedder()
    model.norm_out = _FinalNorm()
    return model


def test_strict_boundaries_route_every_projection_through_its_row(monkeypatch):
    """Strict mode routes each boundary module through its row; video rows get keys, a session."""
    from rl_engine.validation.models.h3_full_model import STRICT_BOUNDARY_ROWS

    called, keyed_calls = [], []
    _fake_rows(monkeypatch, called, keyed_calls)
    model = _boundary_model()
    names = ["proj_in", "audio_proj_in", "context_embedder", "proj_out", "audio_proj_out"]
    with full_model_execution(
        model, candidate=True, trace=FullModelTrace(), ops=object(), strict_boundaries=True
    ) as trace:
        for name in names:
            out = getattr(model, name)(torch.ones(2, 5, 4))
            assert out.shape == (2, 5, 3)
        projections = [(row, shape) for row, shape in called if isinstance(shape, tuple)]
        assert sorted(row for row, _ in projections) == sorted(
            STRICT_BOUNDARY_ROWS[n] for n in names
        )
        assert {shape for _, shape in projections} == {(10, 4)}
        assert trace.boundary_bindings["proj_in"] == dict(
            row="video_input_projection_fp32", status="row"
        )
        assert trace.boundary_bindings["proj_out"]["status"] == "row"
        # every boundary module is bound to its row, none stays on the provider
        assert trace.boundary_bindings == {
            n: dict(row=r, status="row") for n, r in STRICT_BOUNDARY_ROWS.items()
        }
        assert trace.boundary_bindings["time_embedder"] == dict(
            row="timestep_mlp_fp32", status="row"
        )
        assert trace.boundary_bindings["norm_out"] == dict(row="final_adaln_out", status="row")
        # the video rows get the logical rows' identities, a parameter id and an open canonical
        # backward session (a trainable weight is refused without one); the others get no keys
        by_row = {row: (keyed, session) for row, keyed, session in keyed_calls}
        for name in ("proj_in", "proj_out"):
            keyed, session = by_row[STRICT_BOUNDARY_ROWS[name]]
            assert session and keyed["parameter_id"] == name
            assert keyed["logical_keys"].tolist() == [[b, s] for b in range(2) for s in range(5)]
        for name in ("audio_proj_in", "audio_proj_out", "context_embedder"):
            assert by_row[STRICT_BOUNDARY_ROWS[name]][0] == {}
    # leaving the context restores the provider forward and closes the session
    assert "forward" not in model.proj_in.__dict__
    from rl_engine.ops.autograd.canonical_backward import active_session

    assert active_session() is None


def test_strict_time_embedder_and_norm_out_pass_their_parameters_in_row_order(monkeypatch):
    """The two rows get the modules' parameters in the rows' argument order."""
    called = []
    _fake_rows(monkeypatch, called)
    model = _boundary_model()
    embedder, final = model.time_embedder, model.norm_out
    sample = torch.ones(2, 4)
    hidden = torch.ones(1, 5, 4)
    temb = torch.ones(2, 3)
    indices = torch.tensor([0, 1, 1, 0, 1])
    with full_model_execution(
        model, candidate=True, trace=FullModelTrace(), ops=object(), strict_boundaries=True
    ):
        model.time_embedder(sample)
        model.norm_out(hidden, temb, indices)
        # ``condition=None`` is the provider's default and is accepted explicitly too
        model.time_embedder(sample, condition=None)
    (name, args), _, (_, repeat) = called
    assert name == "timestep_mlp_fp32"
    assert args[0] is sample
    assert args[1:] == (
        embedder.linear_1.weight,
        embedder.linear_1.bias,
        embedder.linear_2.weight,
        embedder.linear_2.bias,
    )
    assert repeat[1] is embedder.linear_1.weight
    name, args = called[1]
    assert name == "final_adaln_out"
    assert args[0] is hidden
    assert args[1] is final.norm.weight
    assert args[2] is temb
    assert args[3] is final.linear.weight
    assert args[4] is final.linear.bias
    assert args[5] is indices
    assert args[6] == final.norm.eps == 1.25e-6
    assert len(args) == 7
    # leaving the context restores the provider forward of both modules
    assert "forward" not in embedder.__dict__
    assert "forward" not in final.__dict__


def test_strict_time_embedder_refuses_what_its_row_does_not_cover(monkeypatch):
    """The time_embedder binding refuses a condition, cond_proj, post_act and a non-SiLU act."""
    called = []
    _fake_rows(monkeypatch, called)
    model = _boundary_model()
    with full_model_execution(
        model, candidate=True, trace=FullModelTrace(), ops=object(), strict_boundaries=True
    ):
        with pytest.raises(ValueError, match="condition"):
            model.time_embedder(torch.ones(2, 4), condition=torch.ones(2, 4))
    assert not called
    for extra in ("cond_proj", "post_act"):
        model = _boundary_model()
        setattr(model.time_embedder, extra, torch.nn.Identity())
        with full_model_execution(
            model, candidate=True, trace=FullModelTrace(), ops=object(), strict_boundaries=True
        ):
            with pytest.raises(ValueError, match=extra):
                model.time_embedder(torch.ones(2, 4))
    assert not called
    for act in (torch.nn.GELU(), None):  # the row's activation is SiLU, nothing else
        model = _boundary_model()
        model.time_embedder.act = act
        with full_model_execution(
            model, candidate=True, trace=FullModelTrace(), ops=object(), strict_boundaries=True
        ):
            with pytest.raises(ValueError, match="SiLU"):
                model.time_embedder(torch.ones(2, 4))
    assert not called


def test_strict_boundaries_restore_time_embedder_and_norm_out_after_failure(monkeypatch):
    """Both modules get their provider forward back after an error inside the context."""
    called = []
    _fake_rows(monkeypatch, called)
    model = _boundary_model()
    with pytest.raises(RuntimeError, match="injected"):
        with full_model_execution(
            model, candidate=True, trace=FullModelTrace(), ops=object(), strict_boundaries=True
        ):
            assert "forward" in model.time_embedder.__dict__
            assert "forward" in model.norm_out.__dict__
            raise RuntimeError("injected")
    assert "forward" not in model.time_embedder.__dict__
    assert "forward" not in model.norm_out.__dict__


def test_native_time_embedder_and_norm_out_stay_on_the_provider(monkeypatch):
    """Named in ``native_boundaries``, the two modules stay on the provider."""
    called = []
    _fake_rows(monkeypatch, called)
    model = _boundary_model()
    with full_model_execution(
        model,
        candidate=True,
        trace=(trace := FullModelTrace()),
        ops=object(),
        strict_boundaries=True,
        native_boundaries=("time_embedder", "norm_out"),
    ):
        assert "forward" not in model.time_embedder.__dict__
        assert "forward" not in model.norm_out.__dict__
        assert trace.boundary_bindings["time_embedder"]["status"] == "provider"
        assert trace.boundary_bindings["norm_out"]["status"] == "provider"
        assert trace.boundary_bindings["proj_in"]["status"] == "row"
        with pytest.raises(AssertionError, match="provider forward"):
            model.time_embedder(torch.ones(2, 4))


@pytest.mark.parametrize("row", ["timestep_mlp_fp32", "final_adaln_out"])
def test_strict_refuses_the_reference_op_of_time_embedder_and_norm_out(monkeypatch, row):
    """If the registry falls back to the PyTorch reference, strict mode raises, binding nothing."""
    from rl_engine.runtime.registry import KernelRegistry

    called = []
    _fake_rows(monkeypatch, called)
    fake_get_op = KernelRegistry.get_op

    class Reference:
        """Stands in for the PyTorch reference op the registry falls back to."""

        def forward(self, *args):
            """Never reached: strict mode refuses this op before binding."""
            raise AssertionError("the reference op must not be bound")

    def get_op(self, name, device=None):
        """The fake rows, with ``row`` resolved to the reference."""
        return Reference() if name == row else fake_get_op(self, name, device)

    monkeypatch.setattr(KernelRegistry, "get_op", get_op)
    model = _boundary_model()
    with pytest.raises(RuntimeError, match=f"strict {row} resolved to .*Reference"):
        with full_model_execution(
            model, candidate=True, trace=FullModelTrace(), ops=object(), strict_boundaries=True
        ):
            pass
    # nothing stays bound after the refusal
    assert "forward" not in model.time_embedder.__dict__
    assert "forward" not in model.norm_out.__dict__
    assert "forward" not in model.proj_in.__dict__


def test_the_required_cuda_ops_are_the_registered_cuda_backends():
    """The class paths strict mode requires are the registry's CUDA backends of the two rows."""
    from rl_engine.runtime.registry import OpBackend
    from rl_engine.validation.models import h3_full_model as fm

    assert fm._CUDA_ROW_OPS == {
        "timestep_mlp_fp32": OpBackend.CUDA_H3_TIMESTEP_MLP.value,
        "final_adaln_out": OpBackend.CUDA_H3_FINAL_ADALN_OUT.value,
    }
