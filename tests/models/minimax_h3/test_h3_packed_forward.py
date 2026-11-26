# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The packed-sequence rows of the H3 forward: ``packed_forward`` against the provider's forward.

The provider forward runs unchanged on a stand-in model (identity projections and heads, no
blocks), so the test compares exactly the packing, the selection and the glue around them. Both
known provider variants are checked, forward and every input gradient, bit for bit.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from rl_engine.reference.embedding.index_select import NativeIndexSelectOp
from rl_engine.reference.packing.packed_index_copy import NativePackedIndexCopyOp
from rl_engine.validation.models import h3_packed_forward as hpf

transformer = pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3",
    reason="the Diffusers MiniMax-H3 transformer is the reference",
)
MODEL = transformer.MiniMaxH3Transformer3DModel


class _Identity(nn.Module):
    """An identity projection with a weight of ``dtype`` (what the forward reads the dtype from)."""

    def __init__(self, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        object.__setattr__(self, "linear_1", self)  # not a submodule (``.cuda()`` would recurse)

    def forward(self, x, *args):
        return x


class _Rope(nn.Module):
    def forward(self, position_ids):
        return position_ids.float()


class _AdaBlock(nn.Module):
    """A block whose output depends on every ``adaln_indices`` entry."""

    def __init__(self, width=16, entries=6):
        super().__init__()
        gen = torch.Generator().manual_seed(17)
        self.table = nn.Parameter(torch.randn(entries, width, generator=gen))

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb):
        assert adaln_indices.max() < self.table.shape[0]
        return hidden_states + self.table[adaln_indices].to(hidden_states.dtype)


class _Model(nn.Module):
    def __init__(self, text_dtype=torch.bfloat16):
        super().__init__()
        self.proj_in = _Identity(torch.float32)
        self.audio_proj_in = _Identity(torch.float32)
        self.context_embedder = _Identity(text_dtype)
        self.token_refiner = _Identity(text_dtype)
        self.time_proj = _Identity(torch.float32)
        self.time_embedder = _Identity(torch.float32)
        self.norm_out = _Identity(torch.float32)
        self.proj_out = _Identity(torch.float32)
        self.audio_proj_out = _Identity(torch.float32)
        self.rope = _Rope()
        self.transformer_blocks = nn.ModuleList([_AdaBlock()])
        self.gradient_checkpointing = False


def _inputs(batch=2, width=16, text=5, video=6, audio=3):
    gen = torch.Generator().manual_seed(7)
    seq = text + video + audio + 2  # two rows stay zero
    perm = torch.randperm(seq, generator=gen)
    t_idx, v_idx, a_idx = perm[:text], perm[text : text + video], perm[text + video : seq - 2]
    leaves = [
        torch.randn(batch, n, width, generator=gen).requires_grad_() for n in (video, audio, text)
    ]
    kwargs = dict(
        timestep=torch.tensor([0.25, 1.0]),
        timestep_indices=torch.arange(seq) % 2,  # two timesteps
        token_tags=torch.arange(seq) % 3,  # the three modality tags
        position_ids=torch.zeros(seq, 3),
        video_indices=v_idx,
        audio_indices=a_idx,
        text_indices=t_idx,
        return_dict=False,
    )
    return leaves, kwargs


def _run(forward, model):
    leaves, kwargs = _inputs()
    video, audio = forward(model, *leaves, **kwargs)
    upstream = torch.Generator().manual_seed(3)
    loss = (video * torch.randn(video.shape, generator=upstream)).sum() + (
        audio * torch.randn(audio.shape, generator=upstream)
    ).sum()
    grads = torch.autograd.grad(loss, leaves)
    return (video, audio, *grads)


def _bitwise(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def test_the_installed_provider_forward_is_known():
    assert hpf.forward_source_sha256(MODEL) in hpf.KNOWN_FORWARDS


def _variant() -> str:
    """The installed provider's own dtype-reading variant."""

    digest = hpf.forward_source_sha256(MODEL)
    if digest not in hpf.KNOWN_FORWARDS:
        pytest.skip("unknown provider forward")
    return hpf.KNOWN_FORWARDS[digest]


@pytest.mark.parametrize("text_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bind_pack", [False, True], ids=["select_only", "pack_and_select"])
def test_packed_forward_is_bitwise_the_providers(text_dtype, bind_pack):
    variant = _variant()
    model = _Model(text_dtype)
    provider = inspect.unwrap(MODEL.forward)
    pack = NativePackedIndexCopyOp() if bind_pack else None
    ours = inspect.unwrap(hpf._packed_forward(MODEL, pack, variant, select=NativeIndexSelectOp()))
    for got, want in zip(_run(ours, model), _run(provider, model), strict=True):
        assert _bitwise(got, want)


def test_bind_swaps_the_forward_and_restores_it():
    model = _Model()
    calls = []
    selections = []

    def pack(*args):
        calls.append(len(args))
        return NativePackedIndexCopyOp()(*args)

    def select(x, index, *, dim, unique_indices):
        assert dim == 1 and unique_indices is True
        selections.append(index)
        return NativeIndexSelectOp()(x, index, dim=dim, unique_indices=unique_indices)

    if hpf.forward_source_sha256(MODEL) not in hpf.KNOWN_FORWARDS:
        pytest.skip("unknown provider forward")
    model.__class__ = type("_Bound", (MODEL,), {})  # the provider's forward, on the stand-in
    leaves, kwargs = _inputs()
    with hpf.bind(model, pack, select=select):
        assert "forward" in model.__dict__
        model.forward(*leaves, **kwargs)
    assert "forward" not in model.__dict__
    assert calls == [7]
    assert len(selections) == 2
    assert selections[0] is kwargs["video_indices"] and selections[1] is kwargs["audio_indices"]


def test_bind_refuses_an_unknown_forward_and_changes_nothing():
    class Other(MODEL):
        def forward(self, *args, **kwargs):
            """Not a provider forward."""
            return None

    model = _Model()
    model.__class__ = type("_Other", (Other,), {})
    with pytest.raises(hpf.UnknownForwardError):
        with hpf.bind(model, NativePackedIndexCopyOp()):
            pass
    assert "forward" not in model.__dict__


def test_bind_refuses_a_forward_wrapped_on_the_instance():
    model = _Model()
    model.__class__ = type("_Wrapped", (MODEL,), {})
    model.forward = lambda *a, **k: None  # e.g. a record / replay hook
    with pytest.raises(hpf.UnknownForwardError, match="already wrapped"):
        with hpf.bind(model, NativePackedIndexCopyOp()):
            pass
    assert model.forward(1) is None


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="needs NVIDIA CUDA"
)


def _run_cuda(forward, model):
    leaves, kwargs = _inputs()
    leaves = [leaf.detach().cuda().requires_grad_() for leaf in leaves]
    kwargs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    video, audio = forward(model, *leaves, **kwargs)
    upstream = torch.Generator().manual_seed(3)
    loss = (video * torch.randn(video.shape, generator=upstream).cuda()).sum() + (
        audio * torch.randn(audio.shape, generator=upstream).cuda()
    ).sum()
    grads = torch.autograd.grad(loss, leaves)
    return (video, audio, *grads)


@needs_cuda
@pytest.mark.parametrize("text_dtype", [torch.bfloat16, torch.float32])
def test_the_cuda_row_is_bitwise_the_providers(text_dtype):
    """Compare the registry's CUDA scatter and Triton gather with the actual provider."""

    from rl_engine.backends.cuda.packing.packed_index_copy import PackedIndexCopyCudaOp
    from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp
    from rl_engine.runtime.registry import KernelRegistry

    variant = _variant()
    registry = KernelRegistry()
    pack = registry.get_op("packed_index_copy", device="cuda")
    select = registry.get_op("index_select", device="cuda")
    assert isinstance(pack, PackedIndexCopyCudaOp)
    assert isinstance(select, TritonIndexSelectOp)
    model = _Model(text_dtype).cuda()
    provider = inspect.unwrap(MODEL.forward)
    ours = inspect.unwrap(hpf._packed_forward(MODEL, pack, variant, select=select))
    for got, want in zip(_run_cuda(ours, model), _run_cuda(provider, model), strict=True):
        assert _bitwise(got, want)


@needs_cuda
@pytest.mark.parametrize("pack_available", [True, False], ids=["pack_and_select", "select_only"])
def test_the_full_model_binds_the_cuda_rows_and_reports_them(monkeypatch, pack_available):
    from rl_engine.runtime.registry import KernelRegistry
    from rl_engine.validation.models import h3_full_model as hfm

    if hpf.forward_source_sha256(MODEL) not in hpf.KNOWN_FORWARDS:
        pytest.skip("unknown provider forward")
    if not pack_available:
        # The Triton gather can run even when the optional CUDA scatter extension is absent.
        get_op = KernelRegistry.get_op

        def without_pack(registry, name, device):
            if name == "packed_index_copy":
                raise RuntimeError("no backend for packed_index_copy")
            return get_op(registry, name, device=device)

        monkeypatch.setattr(KernelRegistry, "get_op", without_pack)
    model = _Model().cuda()
    model.__class__ = type("_Bound", (MODEL,), {})
    want = _run_cuda(inspect.unwrap(MODEL.forward), model)
    trace = hfm.FullModelTrace()
    with hfm._packed_rows(model, trace):
        assert "forward" in model.__dict__
        got = _run_cuda(lambda m, *a, **k: m.forward(*a, **k), model)
    assert "forward" not in model.__dict__
    assert trace.packed_bindings == {
        "packed_index_copy": "row" if pack_available else "provider",
        "packed_index_select": "row",
    }
    for g, w in zip(got, want, strict=True):
        assert _bitwise(g, w)


def test_a_replay_wrapper_installed_after_bind_serves_the_packed_forward(tmp_path):
    """The generation tool's order: the packed forward is bound first, the resume wrapper goes on
    top of it (keeping its signature, which the pipeline reads), and comes off first."""

    from rl_engine.validation.models.h3_generation_resume import ResumableRun, StepStore

    if hpf.forward_source_sha256(MODEL) not in hpf.KNOWN_FORWARDS:
        pytest.skip("unknown provider forward")
    model = _Model()
    model.__class__ = type("_Bound", (MODEL,), {})
    want = _run(inspect.unwrap(MODEL.forward), model)
    calls = []

    def pack(*args):
        calls.append(len(args))
        return NativePackedIndexCopyOp()(*args)

    identity = {"commit": "abc", "seed": 1, "candidate": True, "preflight": False}
    with hpf.bind(model, pack):
        bound = model.forward
        run = ResumableRun(StepStore(tmp_path / "steps"), identity, resume=False)
        uninstall = run.install(model, lambda values, kwargs: values)
        assert inspect.signature(model.forward) == inspect.signature(bound)
        got = _run(lambda m, *a, **k: m.forward(*a, **k), model)
        uninstall()
        assert model.__dict__["forward"] == bound
    assert "forward" not in model.__dict__
    assert calls == [7]  # the one forward ran on the packed row, beneath the replay wrapper
    for g, w in zip(got, want, strict=True):
        assert _bitwise(g, w)


def _generated_forward_class():
    namespace: dict = {}
    exec("def forward(self, *args, **kwargs):\n    return None\n", namespace)  # no source file
    return type("_Generated", (MODEL,), {"forward": namespace["forward"]})


def test_an_unreadable_forward_source_is_an_unknown_forward():
    cls = _generated_forward_class()
    with pytest.raises(hpf.UnknownForwardError, match="cannot read"):
        hpf.forward_source_sha256(cls)
    model = _Model()
    model.__class__ = cls
    with pytest.raises(hpf.UnknownForwardError):
        with hpf.bind(model, NativePackedIndexCopyOp()):
            pass
    assert "forward" not in model.__dict__


class _Registry:
    error: BaseException | None = None
    op: object = None

    def get_op(self, name, device):
        assert (name, device) == ("packed_index_copy", "cuda")
        if self.error is not None:
            raise self.error
        return self.op


@pytest.mark.parametrize(
    "error, op",
    [
        (RuntimeError("no backend for packed_index_copy"), None),
        (ImportError("no extension"), None),
        (None, NativePackedIndexCopyOp()),  # resolved, but not the CUDA row
    ],
)
def test_the_pack_falls_back_only_for_a_missing_row(monkeypatch, error, op):
    from rl_engine.runtime import registry
    from rl_engine.validation.models import h3_full_model as hfm

    monkeypatch.setattr(_Registry, "error", error)
    monkeypatch.setattr(_Registry, "op", op)
    monkeypatch.setattr(registry, "KernelRegistry", _Registry)
    model, trace = _Model(), hfm.FullModelTrace()
    with hfm._packed_rows(model, trace):
        assert "forward" not in model.__dict__
    assert trace.packed_bindings["packed_index_copy"] == "provider"


def test_an_out_of_memory_error_while_binding_the_pack_raises(monkeypatch):
    from rl_engine.runtime import registry
    from rl_engine.validation.models import h3_full_model as hfm

    monkeypatch.setattr(_Registry, "error", torch.cuda.OutOfMemoryError("out of memory"))
    monkeypatch.setattr(registry, "KernelRegistry", _Registry)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with hfm._packed_rows(_Model(), hfm.FullModelTrace()):
            pass


def test_strided_index_views_reach_the_pack_contiguous():
    """The provider's ``index_copy`` takes any index view; the row validates contiguous ones."""

    digest = hpf.forward_source_sha256(MODEL)
    if digest not in hpf.KNOWN_FORWARDS:
        pytest.skip("unknown provider forward")
    seen = []

    def pack(*args):
        seen.append([a.is_contiguous() for a in args[3:6]])
        return NativePackedIndexCopyOp()(*args)

    model = _Model()
    leaves, kwargs = _inputs()
    want = inspect.unwrap(MODEL.forward)(model, *leaves, **kwargs)
    for key in ("text_indices", "video_indices", "audio_indices"):
        index = kwargs[key]
        kwargs[key] = torch.stack([index, index], dim=1)[:, 0]  # same values, stride 2
        assert not kwargs[key].is_contiguous()
    ours = inspect.unwrap(hpf._packed_forward(MODEL, pack, hpf.KNOWN_FORWARDS[digest]))
    got = ours(model, *leaves, **kwargs)
    assert seen == [[True, True, True]]
    assert all(_bitwise(g, w) for g, w in zip(got, want, strict=True))
