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
        self.transformer_blocks = nn.ModuleList()
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
        timestep_indices=torch.zeros(seq, dtype=torch.int64),
        token_tags=torch.zeros(seq, dtype=torch.int64),
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
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(-1), b.view(-1))


def test_the_installed_provider_forward_is_known():
    assert hpf.forward_source_sha256(MODEL) in hpf.KNOWN_FORWARDS


@pytest.mark.parametrize("variant", ["weight", "parameter"])
@pytest.mark.parametrize("text_dtype", [torch.bfloat16, torch.float32])
def test_packed_forward_is_bitwise_the_providers(variant, text_dtype):
    model = _Model(text_dtype)
    provider = inspect.unwrap(MODEL.forward)
    ours = inspect.unwrap(hpf._packed_forward(MODEL, NativePackedIndexCopyOp(), variant))
    for got, want in zip(_run(ours, model), _run(provider, model), strict=True):
        assert _bitwise(got, want)


def test_bind_swaps_the_forward_and_restores_it():
    model = _Model()
    calls = []

    def pack(*args):
        calls.append(len(args))
        return NativePackedIndexCopyOp()(*args)

    if hpf.forward_source_sha256(MODEL) not in hpf.KNOWN_FORWARDS:
        pytest.skip("unknown provider forward")
    model.__class__ = type("_Bound", (MODEL,), {})  # the provider's forward, on the stand-in
    leaves, kwargs = _inputs()
    with hpf.bind(model, pack):
        assert "forward" in model.__dict__
        model.forward(*leaves, **kwargs)
    assert "forward" not in model.__dict__
    assert calls == [7]


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


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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
@pytest.mark.parametrize("variant", ["weight", "parameter"])
@pytest.mark.parametrize("text_dtype", [torch.bfloat16, torch.float32])
def test_the_cuda_row_is_bitwise_the_providers(variant, text_dtype):
    """The registry's CUDA ``packed_index_copy`` (what ``h3_full_model`` binds), on a GPU."""

    from rl_engine.backends.cuda.packing.packed_index_copy import PackedIndexCopyCudaOp
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("packed_index_copy", device="cuda")
    assert isinstance(op, PackedIndexCopyCudaOp)
    model = _Model(text_dtype).cuda()
    provider = inspect.unwrap(MODEL.forward)
    ours = inspect.unwrap(hpf._packed_forward(MODEL, op, variant))
    for got, want in zip(_run_cuda(ours, model), _run_cuda(provider, model), strict=True):
        assert _bitwise(got, want)


@needs_cuda
def test_the_full_model_binds_the_cuda_row_and_reports_it():
    from rl_engine.validation.models import h3_full_model as hfm

    if hpf.forward_source_sha256(MODEL) not in hpf.KNOWN_FORWARDS:
        pytest.skip("unknown provider forward")
    model = _Model().cuda()
    model.__class__ = type("_Bound", (MODEL,), {})
    want = _run_cuda(inspect.unwrap(MODEL.forward), model)
    trace = hfm.FullModelTrace()
    with hfm._packed_rows(model, trace):
        assert "forward" in model.__dict__
        got = _run_cuda(lambda m, *a, **k: m.forward(*a, **k), model)
    assert "forward" not in model.__dict__
    assert trace.packed_bindings == {"packed_index_copy": "row", "packed_index_select": "reference"}
    for g, w in zip(got, want, strict=True):
        assert _bitwise(g, w)
