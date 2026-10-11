# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 packed rows: independent FP32 accuracy, Diffusers parity and invariance."""

import argparse
import importlib
from types import SimpleNamespace

import pytest
import torch

from build_tools import envs
from rl_engine.backends.cuda.packing.packed_index_copy import PackedIndexCopyCudaOp
from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.reference.packing.packed_index_copy import NativePackedIndexCopyOp
from rl_engine.runtime.registry import KernelRegistry, OpBackend
from rl_engine.validation.operators import run_operator_suite
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_HAS_CUDA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and _EXT_AVAILABLE
    and all(
        hasattr(_C, name) for name in ("packed_index_copy_forward", "packed_index_copy_backward")
    )
)
requires_cuda_kernel = pytest.mark.skipif(
    not _HAS_CUDA, reason="native packed_index_copy CUDA build required"
)
_DEVICES = ("cpu", pytest.param("cuda", marks=requires_cuda_kernel))


def _assert_bitwise_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _inputs(batch=2, width=17, lengths=(3, 2, 4), dtypes=None, device="cpu"):
    generator = torch.Generator().manual_seed(420)
    if dtypes is None:
        dtypes = (torch.float32,) * 3
    sources = [
        torch.randn(batch, length, width, generator=generator).to(device=device, dtype=dtype)
        for length, dtype in zip(lengths, dtypes, strict=True)
    ]
    sequence_length = sum(lengths) + 2
    permutation = torch.randperm(sequence_length, generator=generator)
    indices = []
    offset = 0
    for length in lengths:
        indices.append(permutation[offset : offset + length].to(device))
        offset += length
    grad = torch.randn(batch, sequence_length, width, generator=generator).to(
        device=device, dtype=dtypes[0]
    )
    return sources, indices, sequence_length, grad


def _fp32_gold(sources, indices, sequence_length, grad, *, promote=False):
    # Explicit row assignment is independent of the candidate's index_copy or
    # fused CUDA scatter. FP32 leaves also keep the gold gradients in FP32.
    leaves = [
        (source if promote else source.to(sources[0].dtype)).float().detach().requires_grad_()
        for source in sources
    ]
    output = torch.zeros(
        sources[0].shape[0], sequence_length, sources[0].shape[-1], dtype=torch.float32
    )
    for source, positions in zip(leaves, indices, strict=True):
        for row, destination in enumerate(positions.tolist()):
            output[:, destination, :] = source[:, row, :]
    gradients = torch.autograd.grad(output, leaves, grad.float(), allow_unused=True)
    gradients = tuple(
        value if value is not None else torch.zeros_like(leaf)
        for value, leaf in zip(gradients, leaves, strict=True)
    )
    assert output.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in gradients)
    return output, gradients


def _run(op, sources, indices, sequence_length, grad):
    leaves = [source.detach().clone().requires_grad_() for source in sources]
    output = op(*leaves, *indices, sequence_length)
    return (output.detach(), *torch.autograd.grad(output, leaves, grad))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("width", [1, 17, 257, 5376])
@pytest.mark.parametrize("device", _DEVICES)
def test_forward_and_gradients_against_independent_fp32_gold(dtype, width, device):
    sources, indices, length, grad = _inputs(width=width, dtypes=(dtype,) * 3)
    expected, gradients = _fp32_gold(sources, indices, length, grad)
    op = KernelRegistry().get_op("packed_index_copy", device=device)
    actual = _run(
        op,
        [x.to(device) for x in sources],
        [i.to(device) for i in indices],
        length,
        grad.to(device),
    )
    for got, want in zip(actual, (expected, *gradients), strict=True):
        _assert_bitwise_equal(got.cpu(), want.to(dtype))
    used = torch.cat(indices)
    holes = torch.ones(length, dtype=torch.bool)
    holes[used] = False
    _assert_bitwise_equal(actual[0].cpu()[:, holes], torch.zeros(2, 2, width, dtype=dtype))


@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, torch.float32, torch.float16),
        (torch.float16, torch.bfloat16, torch.float32),
    ],
)
@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize("device", _DEVICES)
def test_mixed_input_dtypes_and_fp32_promotion(dtypes, method, device):
    sources, indices, length, grad = _inputs(dtypes=dtypes)
    promote = method == "forward_fp32"
    expected, gradients = _fp32_gold(sources, indices, length, grad, promote=promote)
    output_dtype = torch.float32 if promote else dtypes[0]
    op = KernelRegistry().get_op("packed_index_copy", device=device)
    leaves = [x.to(device).requires_grad_() for x in sources]
    actual = getattr(op, method)(*leaves, *[i.to(device) for i in indices], length)
    actual_grads = torch.autograd.grad(actual, leaves, grad.to(device=device, dtype=output_dtype))
    _assert_bitwise_equal(actual.cpu(), expected.to(output_dtype))
    for got, want, dtype in zip(actual_grads, gradients, dtypes, strict=True):
        _assert_bitwise_equal(got.cpu(), want.to(dtype))
    if promote:
        ordinary = op(*leaves, *[i.to(device) for i in indices], length)
        assert not torch.equal(actual, ordinary.float()), "fixture must expose the input casts"


@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize("device", _DEVICES)
def test_mixed_backward_respects_output_cast_boundary(method, device):
    dtypes = (torch.bfloat16, torch.float32, torch.float16)
    sources, indices, length, _ = _inputs(dtypes=dtypes, device=device)
    sources = [source.requires_grad_() for source in sources]
    # FP32 upstream values distinguish a bf16 intermediate from a direct cast
    # into each source dtype. Autograd first aligns this gradient to the output.
    grad = torch.linspace(0.999, 1.019, 2 * length * 17, device=device).reshape(2, length, 17)
    op = KernelRegistry().get_op("packed_index_copy", device=device)
    output = getattr(op, method)(*sources, *indices, length)
    actual = torch.autograd.grad(output, sources, grad)
    aligned = grad.to(output.dtype)
    for got, positions, dtype in zip(actual, indices, dtypes, strict=True):
        _assert_bitwise_equal(got, aligned.index_select(1, positions).to(dtype))
    assert not torch.equal(grad.to(torch.bfloat16).float(), grad)


@pytest.mark.parametrize("modality", [0, 1, 2])
@pytest.mark.parametrize("device", _DEVICES)
def test_modified_saved_indices_raise_before_backward(modality, device):
    sources, indices, length, grad = _inputs(device=device)
    sources = [source.requires_grad_() for source in sources]
    output = KernelRegistry().get_op("packed_index_copy", device=device)(*sources, *indices, length)
    indices[modality].copy_(indices[modality].roll(1))
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output, sources, grad)


@pytest.mark.parametrize("lengths", [(0, 2, 4), (3, 0, 4), (3, 2, 0), (0, 0, 4)])
@pytest.mark.parametrize("device", _DEVICES)
def test_empty_modalities_keep_empty_gradients(lengths, device):
    sources, indices, length, grad = _inputs(lengths=lengths)
    expected, gradients = _fp32_gold(sources, indices, length, grad)
    op = KernelRegistry().get_op("packed_index_copy", device=device)
    actual = _run(
        op,
        [x.to(device) for x in sources],
        [i.to(device) for i in indices],
        length,
        grad.to(device),
    )
    for got, want in zip(actual, (expected, *gradients), strict=True):
        _assert_bitwise_equal(got.cpu(), want)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("device", _DEVICES)
def test_copy_preserves_special_values_and_aliased_sources(dtype, device):
    tiny = torch.finfo(dtype).tiny
    data = torch.tensor(
        [0.0, -0.0, tiny, -tiny, tiny / 2, float("inf"), float("nan")],
        dtype=dtype,
        device=device,
    ).reshape(1, 1, 7)
    data.requires_grad_()
    indices = [torch.tensor([position], device=device) for position in (3, 0, 2)]
    op = KernelRegistry().get_op("packed_index_copy", device=device)
    actual = op(data, data, data, *indices, 5)
    for position in (3, 0, 2):
        _assert_bitwise_equal(actual[:, position : position + 1], data)
    _assert_bitwise_equal(actual[:, 1:2], torch.zeros_like(data))
    grad = torch.arange(35, device=device, dtype=dtype).reshape(1, 5, 7)
    expected = grad[:, 3:4] + grad[:, 0:1] + grad[:, 2:3]
    _assert_bitwise_equal(torch.autograd.grad(actual, data, grad)[0], expected)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("input_mode", ["random", "constant"])
@pytest.mark.parametrize(
    "device, backend",
    [("cpu", "pytorch"), pytest.param("cuda", "cuda", marks=requires_cuda_kernel)],
)
def test_gtest_forward_backward(dtype, input_mode, device, backend):
    args = argparse.Namespace(
        op="packed_index_copy",
        candidate=backend,
        arch_key=device,
        batch=2,
        seq=9,
        normalized_dim=7,
        seed=420,
        input_mode=input_mode,
    )
    case = make_operator_case(args, dtype, torch.device(device))
    assert case.grad_input_names == ("text", "video", "audio")
    assert case.inputs["text"].shape == (2, 3, 7)
    report = run_operator_suite(
        args.op,
        candidates=[make_candidate(args)],
        cases=[case],
        check_grad=True,
        grad_mode="random",
    )
    assert report.passed, report.to_dict()


@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "batch",
        "width",
        "double",
        "integer",
        "strided",
        "zero_batch",
        "zero_width",
        "all_empty",
        "index_rank",
        "index_dtype",
        "index_count",
        "index_strided",
        "source_device",
        "index_device",
        "zero_length",
        "negative_length",
        "bool_length",
        "float_length",
    ],
)
def test_invalid_metadata(case, device, method):
    sources, indices, length, _ = _inputs(device=device)
    if case == "rank":
        sources[1] = sources[1][0]
    elif case == "batch":
        sources[1] = sources[1][:1]
    elif case == "width":
        sources[1] = sources[1][:, :, :-1].contiguous()
    elif case == "double":
        sources[1] = sources[1].double()
    elif case == "integer":
        sources[1] = sources[1].long()
    elif case == "strided":
        sources[1] = torch.zeros(2, 2, 34, device=device)[:, :, ::2]
    elif case == "zero_batch":
        sources = [x[:0] for x in sources]
    elif case == "zero_width":
        sources = [x[:, :, :0] for x in sources]
    elif case == "all_empty":
        sources = [x[:, :0] for x in sources]
        indices = [i[:0] for i in indices]
    elif case == "index_rank":
        indices[1] = indices[1].unsqueeze(0)
    elif case == "index_dtype":
        indices[1] = indices[1].int()
    elif case == "index_count":
        indices[1] = indices[1][:1]
    elif case == "index_strided":
        indices[1] = torch.arange(4, device=device)[::2]
    elif case == "source_device":
        sources[1] = torch.empty_like(sources[1], device="meta")
    elif case == "index_device":
        indices[1] = torch.empty_like(indices[1], device="meta")
    elif case == "zero_length":
        length = 0
    elif case == "negative_length":
        length = -1
    elif case == "bool_length":
        length = True
    elif case == "float_length":
        length = float(length)
    op = KernelRegistry().get_op("packed_index_copy", device=device)
    with pytest.raises((ValueError, TypeError)):
        getattr(op, method)(*sources, *indices, length)


@pytest.mark.parametrize("case", ["negative", "upper_bound", "duplicate", "overlap"])
@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize("device", _DEVICES)
def test_invalid_index_contents(case, method, device):
    sources, indices, length, _ = _inputs(device=device)
    if case == "negative":
        indices[0][0] = -1
    elif case == "upper_bound":
        indices[0][0] = length
    elif case == "duplicate":
        indices[0][0] = indices[0][1]
    else:
        indices[0][0] = indices[1][0]
    op = KernelRegistry().get_op("packed_index_copy", device=device)
    with pytest.raises((ValueError, RuntimeError), match="range, unique and disjoint"):
        getattr(op, method)(*sources, *indices, length)


class _IdentityProjection:
    def __init__(self, dtype):
        self.weight = torch.empty(0, dtype=dtype)
        self.linear_1 = self

    def __call__(self, value):
        return value


class _CapturePacked:
    def __call__(self, hidden_states, *_):
        self.packed = hidden_states
        return hidden_states


@pytest.fixture(scope="module")
def diffusers_forward():
    module = pytest.importorskip(
        "diffusers.models.transformers.transformer_minimax_h3",
        reason="Diffusers MiniMax-H3 transformer is required for upstream parity",
    )
    # Run the actual upstream forward with identity projections and no blocks.
    # Capture its packed tensor before the output heads, without model weights
    # or extracting/recompiling source text. Only the LoRA decorator is bypassed.
    return module.MiniMaxH3Transformer3DModel.forward.__wrapped__


@pytest.mark.parametrize(
    "dtypes",
    [(dtype,) * 3 for dtype in _DTYPES] + [(torch.bfloat16, torch.float32, torch.float16)],
)
@pytest.mark.parametrize("promote", [False, True])
@pytest.mark.parametrize("device", _DEVICES)
def test_forward_and_gradients_match_diffusers(diffusers_forward, dtypes, promote, device):
    sources, indices, length, grad = _inputs(dtypes=dtypes, device=device)
    sources = [x.requires_grad_() for x in sources]
    upstream_sources = [x.float() for x in sources] if promote else sources
    capture = _CapturePacked()
    text_projection = _IdentityProjection(upstream_sources[0].dtype)
    model = SimpleNamespace(
        proj_in=_IdentityProjection(upstream_sources[1].dtype),
        audio_proj_in=_IdentityProjection(upstream_sources[2].dtype),
        context_embedder=text_projection,
        token_refiner=lambda x: x,
        rope=lambda _: None,
        time_proj=lambda x: x,
        time_embedder=text_projection,
        transformer_blocks=(),
        norm_out=capture,
        proj_out=text_projection,
        audio_proj_out=text_projection,
    )
    diffusers_forward(
        model,
        hidden_states=upstream_sources[1],
        audio_hidden_states=upstream_sources[2],
        encoder_hidden_states=upstream_sources[0],
        timestep=torch.zeros(1, device=device),
        timestep_indices=torch.zeros(length, dtype=torch.long, device=device),
        token_tags=torch.zeros(length, dtype=torch.long, device=device),
        position_ids=torch.zeros(length, 3, device=device),
        video_indices=indices[1],
        audio_indices=indices[2],
        text_indices=indices[0],
        return_dict=False,
    )
    expected = capture.packed
    grad = grad.to(expected.dtype)
    expected_grads = torch.autograd.grad(expected, sources, grad)
    op = KernelRegistry().get_op("packed_index_copy", device=device)
    method = op.forward_fp32 if promote else op.forward
    actual = method(*sources, *indices, length)
    actual_grads = torch.autograd.grad(actual, sources, grad)
    _assert_bitwise_equal(actual, expected)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        _assert_bitwise_equal(got, want)


def test_registry_cpu_reference():
    assert isinstance(
        KernelRegistry().get_op("packed_index_copy", device="cpu"),
        NativePackedIndexCopyOp,
    )


def test_cuda_registry_fails_closed(monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_get_or_create_backend", lambda backend: None)
    assert registry._priority_map["cuda"]["packed_index_copy"] == [OpBackend.CUDA_PACKED_INDEX_COPY]
    with pytest.raises(RuntimeError, match="No functional backend"):
        registry.get_op("packed_index_copy", device="cuda")
    for device in ("rocm", "musa", "npu"):
        assert registry._priority_map[device]["packed_index_copy"] == []


def test_missing_extension_fails_closed(monkeypatch):
    module = importlib.import_module("rl_engine.backends.cuda.packing.packed_index_copy")
    monkeypatch.setattr(module, "_EXT_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="compiled CUDA"):
        module.PackedIndexCopyCudaOp()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA CUDA GPU",
)
@pytest.mark.skipif(
    not envs.env_flag(envs.RL_KERNEL_REQUIRE_EXT),
    reason="native extension enforcement is opt-in via RL_KERNEL_REQUIRE_EXT=1",
)
def test_required_cuda_extension_has_operator_symbols():
    assert _EXT_AVAILABLE and all(
        hasattr(_C, name) for name in ("packed_index_copy_forward", "packed_index_copy_backward")
    ), "rl_engine._C must provide both packed_index_copy kernels"


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_backward_invariance(dtype):
    op = PackedIndexCopyCudaOp()
    sources, indices, length, grad = _inputs(batch=1, dtypes=(dtype,) * 3, device="cuda")
    single = _run(op, sources, indices, length, grad)
    for batch, position in [(3, 0), (4, 2), (7, 6)]:
        values, _, length, upstream = _inputs(batch=batch, dtypes=(dtype,) * 3, device="cuda")
        positions = indices
        for source, baseline in zip(values, sources, strict=True):
            source[position] = baseline[0]
        upstream[position] = grad[0]
        first = _run(op, values, positions, length, upstream)
        for _ in range(3):
            for got, want in zip(_run(op, values, positions, length, upstream), first, strict=True):
                _assert_bitwise_equal(got, want)
        for got, want in zip(first, single, strict=True):
            _assert_bitwise_equal(got[position], want[0])
        permuted = _run(op, [x.flip(0) for x in values], positions, length, upstream.flip(0))
        for got, want in zip(permuted, first, strict=True):
            _assert_bitwise_equal(got.flip(0), want)
        split = batch // 2
        chunks = [
            _run(
                op,
                [x[start:end] for x in values],
                positions,
                length,
                upstream[start:end],
            )
            for start, end in ((0, split), (split, batch))
        ]
        for index, want in enumerate(first):
            _assert_bitwise_equal(torch.cat([chunk[index] for chunk in chunks]), want)
        other = (position + 1) % batch
        for source in values:
            source[other].fill_(99)
        for got, want in zip(_run(op, values, positions, length, upstream), first, strict=True):
            _assert_bitwise_equal(got[position], want[position])


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
def test_packed_row_permutation_and_extra_holes(dtype):
    op = PackedIndexCopyCudaOp()
    sources, indices, length, grad = _inputs(dtypes=(dtype,) * 3, device="cuda")
    expected = _run(op, sources, indices, length, grad)
    reordered = _run(
        op,
        [source.flip(1) for source in sources],
        [index.flip(0) for index in indices],
        length,
        grad,
    )
    _assert_bitwise_equal(reordered[0], expected[0])
    for got, want in zip(reordered[1:], expected[1:], strict=True):
        _assert_bitwise_equal(got.flip(1), want)
    permutation = torch.arange(length - 1, -1, -1, device="cuda")
    # permutation maps old destinations to new destinations.
    moved = _run(
        op,
        sources,
        [permutation[i] for i in indices],
        length,
        grad[:, permutation].contiguous(),
    )
    _assert_bitwise_equal(moved[0][:, permutation], expected[0])
    for got, want in zip(moved[1:], expected[1:], strict=True):
        _assert_bitwise_equal(got, want)
    expanded_grad = torch.cat((grad, torch.full_like(grad[:, :3], 99)), dim=1)
    expanded = _run(op, sources, indices, length + 3, expanded_grad)
    _assert_bitwise_equal(expanded[0][:, :length], expected[0])
    _assert_bitwise_equal(expanded[0][:, length:], torch.zeros_like(expanded[0][:, length:]))
    for got, want in zip(expanded[1:], expected[1:], strict=True):
        _assert_bitwise_equal(got, want)
    changed = [source.clone() for source in sources]
    changed[1][:, 0].fill_(-999)
    again = _run(op, changed, indices, length, grad)
    unrelated_rows = torch.arange(length, device="cuda") != indices[1][0]
    _assert_bitwise_equal(again[0][:, unrelated_rows], expected[0][:, unrelated_rows])
    for got, want in zip(again[1:], expected[1:], strict=True):
        _assert_bitwise_equal(got, want)


@requires_cuda_kernel
@pytest.mark.parametrize(
    "requires_grad",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, True),
    ],
)
def test_partial_input_gradients(requires_grad):
    sources, indices, length, grad = _inputs(device="cuda")
    for source, required in zip(sources, requires_grad, strict=True):
        source.requires_grad_(required)
    leaves = [source for source in sources if source.requires_grad]
    expected = NativePackedIndexCopyOp()(*sources, *indices, length)
    actual = PackedIndexCopyCudaOp()(*sources, *indices, length)
    for got, want in zip(
        torch.autograd.grad(actual, leaves, grad),
        torch.autograd.grad(expected, leaves, grad),
        strict=True,
    ):
        _assert_bitwise_equal(got, want)


@requires_cuda_kernel
def test_current_stream_and_noncontiguous_upstream_gradient():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        sources, indices, length, _ = _inputs(device="cuda")
        for source in sources:
            source.requires_grad_()
        actual = PackedIndexCopyCudaOp()(*sources, *indices, length)
        grad = torch.arange(2 * length * 17, device="cuda", dtype=torch.float32).reshape(
            17, length, 2
        )
        grad = grad.permute(2, 1, 0)
        assert not grad.is_contiguous()
        gradients = torch.autograd.grad(actual, sources, grad)
    stream.synchronize()
    expected, gold_gradients = _fp32_gold(
        [x.detach().cpu() for x in sources],
        [i.cpu() for i in indices],
        length,
        grad.cpu(),
    )
    _assert_bitwise_equal(actual.cpu(), expected)
    for got, want in zip(gradients, gold_gradients, strict=True):
        _assert_bitwise_equal(got.cpu(), want)


@requires_cuda_kernel
def test_cuda_backward_is_first_order_only():
    sources, indices, length, grad = _inputs(device="cuda")
    sources = [source.requires_grad_() for source in sources]
    grad.requires_grad_()
    output = PackedIndexCopyCudaOp()(*sources, *indices, length)
    first_gradients = torch.autograd.grad(output, sources, grad, create_graph=True)
    with pytest.raises(RuntimeError, match="once_differentiable"):
        sum(value.sum() for value in first_gradients).backward()


@requires_cuda_kernel
@pytest.mark.parametrize("negative_view", ["text", "video", "audio", "indices", "grad"])
def test_negative_views_match_reference(negative_view):
    sources, indices, length, grad = _inputs(device="cuda")
    if negative_view in ("text", "video", "audio"):
        index = ("text", "video", "audio").index(negative_view)
        sources[index] = torch._neg_view(sources[index])
        assert sources[index].is_neg()
    elif negative_view == "indices":
        indices[0] = torch._neg_view(-indices[0])
        assert indices[0].is_neg()
    else:
        grad = torch._neg_view(grad)
        assert grad.is_neg()
    sources = [source.requires_grad_() for source in sources]
    expected = NativePackedIndexCopyOp()(*sources, *indices, length)
    actual = PackedIndexCopyCudaOp()(*sources, *indices, length)
    _assert_bitwise_equal(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual, sources, grad),
        torch.autograd.grad(expected, sources, grad),
        strict=True,
    ):
        _assert_bitwise_equal(got, want)


@requires_cuda_kernel
@pytest.mark.parametrize("case", ["negative", "upper_bound", "duplicate", "overlap"])
def test_native_bindings_validate_index_contents(case):
    sources, indices, length, grad = _inputs(device="cuda")
    if case == "negative":
        indices[0][0] = -1
    elif case == "upper_bound":
        indices[0][0] = length
    elif case == "duplicate":
        indices[0][0] = indices[0][1]
    else:
        indices[0][0] = indices[1][0]
    with pytest.raises(RuntimeError, match="range, unique and disjoint"):
        _C.packed_index_copy_forward(*sources, *indices, length)
    with pytest.raises(RuntimeError, match="range, unique and disjoint"):
        _C.packed_index_copy_backward(grad, *indices)


@requires_cuda_kernel
def test_native_bindings_validate_metadata_and_negative_views():
    sources, indices, length, grad = _inputs(device="cuda")
    with pytest.raises(RuntimeError, match="dtype"):
        _C.packed_index_copy_forward(sources[0], sources[1].half(), sources[2], *indices, length)
    with pytest.raises(RuntimeError, match="contiguous"):
        _C.packed_index_copy_backward(grad.transpose(1, 2), *indices)
    with pytest.raises(RuntimeError, match="negative view bit"):
        _C.packed_index_copy_forward(torch._neg_view(sources[0]), *sources[1:], *indices, length)
    with pytest.raises(RuntimeError, match="negative view bit"):
        _C.packed_index_copy_backward(torch._neg_view(grad), *indices)
