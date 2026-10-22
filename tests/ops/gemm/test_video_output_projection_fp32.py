# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 FP32 output projection accuracy, logical-row BI, and video selection."""

import hashlib
import math

import pytest
import torch

from rl_engine.contracts.numerical import load_contract
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.ops.gemm.video_output_projection_fp32 import VideoOutputProjectionFP32Op
from rl_engine.reference.gemm.video_output_projection_fp32 import (
    NativeVideoOutputProjectionFP32Op,
    _reference_matmul_fp32,
)

HAS_SM90 = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() == (9, 0)
)
requires_sm90 = pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")


def _op():
    from rl_engine.backends.shared.triton.gemm.video_output_projection_fp32 import (
        TritonVideoOutputProjectionFP32Op,
    )

    return TritonVideoOutputProjectionFP32Op()


def _keys(rows, device):
    return torch.stack(
        (torch.zeros(rows, dtype=torch.int64, device=device), torch.arange(rows, device=device)),
        dim=1,
    )


def _data(rows, device="cuda", *, seed=420, scaled=True):
    generator = torch.Generator(device=device).manual_seed(seed)
    shapes = ((rows, 5376), (96, 5376), (96,), (rows, 96))
    values = [torch.randn(shape, generator=generator, device=device) for shape in shapes]
    if scaled:
        # In-place scaling avoids a second full hidden-state allocation in large cases.
        for value, scale in zip(values, (0.125, 0.0625, 0.125, 0.0625), strict=True):
            value.mul_(scale)
    return values


def _run(op, data, keys, chunks=None, *, train_weight=True, train_bias=True):
    x, weight, bias, grad = [None if t is None else t.detach().clone() for t in data]
    x.requires_grad_(True)
    weight.requires_grad_(train_weight)
    if bias is not None:
        bias.requires_grad_(train_bias)
    cuts = chunks if chunks is not None else [torch.arange(x.shape[0], device=x.device)]
    with canonical_backward_session() as session:
        outputs = [
            op(
                x.index_select(0, ix),
                weight,
                bias,
                logical_keys=keys.index_select(0, ix),
                parameter_id="proj_out",
            )
            for ix in cuts
        ]
        torch.autograd.backward(outputs, [grad.index_select(0, ix) for ix in cuts])
        session.validate_complete()
    y = torch.empty(x.shape[0], 96, dtype=torch.float32, device=x.device)
    for ix, out in zip(cuts, outputs, strict=True):
        y.index_copy_(0, ix, out.detach())
    return y, x.grad, weight.grad, None if bias is None else bias.grad


def _reference(data):
    x, weight, bias, grad = data
    x, weight = [t.detach().cpu().requires_grad_() for t in (x, weight)]
    bias = None if bias is None else bias.detach().cpu().requires_grad_()
    output = NativeVideoOutputProjectionFP32Op()(x, weight, bias)
    output.backward(grad.cpu())
    return output, x.grad, weight.grad, None if bias is None else bias.grad


def _assert_accuracy(actual, expected, kind):
    threshold = load_contract()["judgments"][kind]["by_op_class"]["reduction"]["float32"]
    assert actual.shape == expected.shape
    for a, b in zip(actual.split(4096), expected.split(4096), strict=True):
        torch.testing.assert_close(a.cpu(), b.cpu(), atol=threshold["atol"], rtol=threshold["rtol"])


def _assert_result_accuracy(actual, expected):
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if a is None or b is None:
            assert a is b
        else:
            _assert_accuracy(a, b, "forward_accuracy" if i == 0 else "gradient_accuracy")


def _assert_bytes(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
        return
    assert torch.equal(
        actual.detach().contiguous().view(torch.int32),
        expected.detach().contiguous().view(torch.int32),
    )


class _CPUAdapter(VideoOutputProjectionFP32Op):
    family = "pytorch"

    def _check_backend(self, x):
        pass


@pytest.mark.parametrize("with_bias", [False, True])
def test_cpu_reference_and_input_only_gradient(with_bias):
    data = _data(3, "cpu")
    x, weight, bias, grad = data
    if not with_bias:
        bias = data[2] = None
    x.requires_grad_(True)
    output = _CPUAdapter()(x, weight, bias)
    output.backward(grad)
    expected = _reference(data)
    _assert_accuracy(output, expected[0], "forward_accuracy")
    _assert_accuracy(x.grad, expected[1], "gradient_accuracy")
    assert weight.grad is None
    assert bias is None or bias.grad is None


@pytest.mark.parametrize("frozen", ["weight", "bias", "both"])
def test_frozen_parameter_registration(frozen):
    result = _run(
        _CPUAdapter(),
        _data(2, "cpu"),
        _keys(2, "cpu"),
        train_weight=frozen not in ("weight", "both"),
        train_bias=frozen not in ("bias", "both"),
    )
    assert (result[2] is None) == (frozen in ("weight", "both"))
    assert (result[3] is None) == (frozen in ("bias", "both"))


@pytest.mark.parametrize("bias_uses", ["all", "partial", "distinct"])
def test_canonical_bias_sharing_requires_matching_contributions(bias_uses, monkeypatch):
    from rl_engine.ops.autograd.canonical_backward import CanonicalBackwardSession
    from rl_engine.ops.gemm.canonical_linear import canonical_linear_fp32

    torch.manual_seed(421)
    x = torch.randn(7, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    biases = [torch.randn(5, requires_grad=True) for _ in range(2)]
    grad = torch.randn(7, 5)
    indices = [torch.tensor([4, 1, 6]), torch.tensor([0, 5, 2, 3])]
    keys = _keys(7, "cpu")
    keys[2, 0] = -1
    normalized = []
    original = CanonicalBackwardSession._normalize

    def observe(self, parameter_id, values):
        result = original(self, parameter_id, values)
        normalized.append((parameter_id, result[1]))
        return result

    monkeypatch.setattr(CanonicalBackwardSession, "_normalize", observe)
    with canonical_backward_session() as session:
        outputs = []
        for use, ix in enumerate(indices):
            bias = (
                None
                if bias_uses == "partial" and use == 1
                else biases[use if bias_uses == "distinct" else 0]
            )
            outputs.append(
                canonical_linear_fp32(
                    x[ix],
                    weight,
                    keys[ix],
                    parameter_id="weight",
                    family="pytorch",
                    bias=bias,
                    bias_parameter_id=f"bias.{use if bias_uses == 'distinct' else 0}",
                )
            )
        torch.autograd.backward(outputs, [grad[ix] for ix in indices])
        session.validate_complete()
        assert not session._normalized
    active = keys[:, 0] >= 0
    torch.testing.assert_close(weight.grad, grad[active].t() @ x.detach()[active])
    torch.testing.assert_close(x.grad, grad @ weight.detach())
    for use, bias in enumerate(biases):
        if use == 1 and bias_uses != "distinct":
            assert bias.grad is None
            continue
        ix = torch.arange(7) if bias_uses == "all" else indices[use]
        ix = ix[active[ix]]
        torch.testing.assert_close(bias.grad, grad[ix].sum(0))
    if bias_uses == "all":
        assert len(normalized) == 2
        assert normalized[0][1] is normalized[1][1]
    else:
        assert all(a[1] is not b[1] for i, a in enumerate(normalized) for b in normalized[i + 1 :])


@pytest.mark.parametrize(
    "invalid,error_type,message",
    [
        ("dtype", TypeError, "x must have dtype torch.float32"),
        ("weight_dtype", TypeError, "weight must have dtype torch.float32"),
        ("bias_dtype", TypeError, "bias must have dtype torch.float32"),
        ("shape", ValueError, "x must have non-empty shape"),
        ("rank", ValueError, "x must have non-empty shape"),
        ("empty", ValueError, "x must have non-empty shape"),
        ("weight_shape", ValueError, "weight must have shape"),
        ("bias_shape", ValueError, "bias must have shape"),
        ("noncontiguous", ValueError, "x must be contiguous"),
        ("keys_dtype", ValueError, "logical_keys must be contiguous int64"),
        ("keys_shape", ValueError, "logical_keys must have shape"),
        ("missing_keys", TypeError, "parameter gradients require logical_keys"),
        ("parameter_id", ValueError, "parameter gradients require a non-empty parameter_id"),
        ("duplicates", ValueError, "active logical keys must be unique"),
        ("duplicates_across_uses", ValueError, "active logical keys must be unique"),
        ("session", RuntimeError, "parameter gradients require canonical_backward_session"),
    ],
)
def test_invalid_inputs(invalid, error_type, message):
    x, weight, bias, _ = _data(2, "cpu")
    keys, parameter_id = _keys(2, "cpu"), "proj_out"
    if invalid == "dtype":
        x = x.bfloat16()
    elif invalid == "weight_dtype":
        weight = weight.bfloat16()
    elif invalid == "bias_dtype":
        bias = bias.bfloat16()
    elif invalid == "shape":
        x = x[:, :5375].contiguous()
    elif invalid == "rank":
        x = x.reshape(1, 1, 2, 5376)
    elif invalid == "empty":
        x = x[:0]
    elif invalid == "weight_shape":
        weight = weight[:95]
    elif invalid == "bias_shape":
        bias = bias[:95]
    elif invalid == "noncontiguous":
        x = torch.empty(2, 10752)[:, ::2]
    elif invalid == "keys_dtype":
        keys = keys.int()
    elif invalid == "keys_shape":
        keys = keys[:1]
    elif invalid == "missing_keys":
        keys = None
    elif invalid == "parameter_id":
        parameter_id = " "
    elif invalid == "duplicates":
        keys.zero_()
    weight.requires_grad_(True)
    op = _CPUAdapter()
    if invalid == "session":
        with pytest.raises(error_type, match=message):
            op(x, weight, bias, logical_keys=keys, parameter_id=parameter_id)
        return
    with canonical_backward_session():
        if invalid == "duplicates_across_uses":
            op(x, weight, bias, logical_keys=keys, parameter_id=parameter_id)
            keys = keys.clone()
            keys[:, 1] += 1
        with pytest.raises(error_type, match=message):
            op(x, weight, bias, logical_keys=keys, parameter_id=parameter_id)


@requires_sm90
@pytest.mark.parametrize("rows", [1, 9, 17, 65, 96, 97, 128, 129, 257])
@pytest.mark.parametrize("with_bias", [False, True])
def test_forward_and_actual_gradients(rows, with_bias):
    data = _data(rows)
    if not with_bias:
        data[2] = None
    _assert_result_accuracy(_run(_op(), data, _keys(rows, "cuda")), _reference(data))


@requires_sm90
@pytest.mark.parametrize("rows", [9, 65, 96, 97, 127, 128, 129, 257])
def test_parameter_bi_permutation_chunk_padding(rows):
    _check_parameter_bi(_data(rows))


def _check_parameter_bi(data):
    rows = data[0].shape[0]
    op, keys = _op(), _keys(rows, "cuda")
    baseline = _run(op, data, keys)
    generator = torch.Generator(device="cuda").manual_seed(421)
    order = torch.randperm(rows, generator=generator, device="cuda")
    permuted = _run(
        op,
        [t.index_select(0, order) if i in (0, 3) else t for i, t in enumerate(data)],
        keys.index_select(0, order),
    )
    inverse = order.argsort()
    for i, (actual, expected) in enumerate(zip(permuted, baseline, strict=True)):
        _assert_bytes(actual.index_select(0, inverse) if i < 2 else actual, expected)
    chunked = _run(op, data, keys, list(torch.tensor_split(order, 3)))
    for actual, expected in zip(chunked, baseline, strict=True):
        _assert_bytes(actual, expected)
    padded = [
        torch.cat((data[0], torch.randn(7, 5376, device="cuda"))),
        data[1],
        data[2],
        torch.cat((data[3], torch.randn(7, 96, device="cuda"))),
    ]
    padded_keys = torch.cat((keys, torch.full((7, 2), -1, device="cuda", dtype=torch.int64)))
    result = _run(op, padded, padded_keys)
    for i, (actual, expected) in enumerate(zip(result, baseline, strict=True)):
        _assert_bytes(actual[:rows] if i < 2 else actual, expected)
    for actual, expected in zip(_run(op, data, keys), baseline, strict=True):
        _assert_bytes(actual, expected)


@requires_sm90
@pytest.mark.parametrize(
    "seed,rows",
    [(seed, rows) for seed in (420, 421, 422, 423) for rows in (129, 13104)] + [(420, 5376)],
)
def test_standard_normal_long_reduction(rows, seed):
    # Output projection moves the 5376-term dot into forward; dW reduces over rows.
    # At 5376 rows, dbias shares the one-row forward GEMM shape.
    data = _data(rows, seed=seed, scaled=False)
    _assert_result_accuracy(_run(_op(), data, _keys(rows, "cuda")), _reference(data))
    _check_parameter_bi(data)


@requires_sm90
@pytest.mark.parametrize("rows,hidden,outputs", [(17, 4096, 128), (4096, 64, 32)])
def test_existing_canonical_linear_bf16_call_preserves_accuracy_and_bi(rows, hidden, outputs):
    from rl_engine.ops.gemm.canonical_linear import canonical_linear_fp32

    generator = torch.Generator(device="cuda").manual_seed(420)
    x = (torch.randn(rows, hidden, generator=generator, device="cuda") * 0.125).bfloat16()
    weight = (torch.randn(outputs, hidden, generator=generator, device="cuda") * 0.0625).bfloat16()
    grad = torch.randn(rows, outputs, generator=generator, device="cuda") * 0.0625
    keys = _keys(rows, "cuda")

    def run(cuts):
        inputs, parameter = [value.detach().clone().requires_grad_() for value in (x, weight)]
        with canonical_backward_session() as session:
            # Preserve the existing call shape without bias or a custom matmul.
            values = [
                canonical_linear_fp32(
                    inputs.index_select(0, indices),
                    parameter,
                    keys.index_select(0, indices),
                    parameter_id="existing_linear.weight",
                    family="triton",
                )
                for indices in cuts
            ]
            torch.autograd.backward(values, [grad.index_select(0, indices) for indices in cuts])
            session.validate_complete()
        output = torch.empty(rows, outputs, dtype=torch.float32, device="cuda")
        for indices, value in zip(cuts, values, strict=True):
            output.index_copy_(0, indices, value.detach())
        return output, inputs.grad, parameter.grad

    baseline = run([torch.arange(rows, device="cuda")])
    order = torch.randperm(rows, generator=generator, device="cuda")
    for cuts in ([order], list(torch.tensor_split(order, 3))):
        reordered = run(cuts)
        for actual, expected in zip(reordered, baseline, strict=True):
            _assert_bytes(actual, expected)
    assert baseline[0].dtype == torch.float32
    assert baseline[1].dtype == baseline[2].dtype == torch.bfloat16

    cx, cw, cg = [value.detach().cpu().float() for value in (x, weight, grad)]
    reference = (
        _reference_matmul_fp32(cx, cw.t()),
        _reference_matmul_fp32(cg, cw),
        _reference_matmul_fp32(cg.t(), cx),
    )
    _assert_accuracy(baseline[0], reference[0], "forward_accuracy")
    threshold = load_contract()["judgments"]["gradient_accuracy"]["by_op_class"]["reduction"][
        "bfloat16"
    ]
    for actual, expected in zip(baseline[1:], reference[1:], strict=True):
        torch.testing.assert_close(
            actual.cpu().float(), expected, atol=threshold["atol"], rtol=threshold["rtol"]
        )


@requires_sm90
@pytest.mark.parametrize("k_size", [129, 257])
def test_chunked_gemm_workspace_slices_preserve_strided_inputs(monkeypatch, k_size):
    from rl_engine.backends.shared.triton.gemm import det_gemm as shared_gemm

    generator = torch.Generator(device="cuda").manual_seed(420)
    rows, columns = 17, 33
    a = torch.randn(k_size, 2 * rows, generator=generator, device="cuda").mul_(0.125)[:, ::2].t()
    b = torch.randn(k_size, 2 * columns, generator=generator, device="cuda").mul_(0.0625)[:, ::2]
    assert not a.is_contiguous() and not b.is_contiguous()
    baseline = shared_gemm._triton_chunked_gemm_fp32(a, b)
    chunks = math.ceil(k_size / shared_gemm._FP32_CHUNK_K)
    # Allow five output rows per workspace, so all 17 rows require four slices.
    budget = chunks * columns * 5
    assert budget < chunks * rows * columns < shared_gemm._MAX_TREE_WORKSPACE_ELEMENTS
    monkeypatch.setattr(shared_gemm, "_MAX_TREE_WORKSPACE_ELEMENTS", budget)
    sliced = shared_gemm._triton_chunked_gemm_fp32(a, b)
    _assert_bytes(sliced, baseline)
    _assert_accuracy(sliced, _reference_matmul_fp32(a, b), "forward_accuracy")


@requires_sm90
@pytest.mark.parametrize("frozen", ["weight", "bias", "both"])
def test_backend_frozen_parameters_preserve_trainable_gradients(frozen):
    data = _data(17)
    actual = _run(
        _op(),
        data,
        _keys(17, "cuda"),
        train_weight=frozen not in ("weight", "both"),
        train_bias=frozen not in ("bias", "both"),
    )
    expected = list(_reference(data))
    if frozen in ("weight", "both"):
        expected[2] = None
    if frozen in ("bias", "both"):
        expected[3] = None
    _assert_result_accuracy(actual, expected)


@requires_sm90
def test_batch_placement_and_unrelated_rows():
    op = _op()
    x, weight, bias, grad = _data(39)
    x = x.reshape(3, 13, 5376)
    grad = grad.reshape(3, 13, 96)
    single = _run(op, (x[1], weight, bias, grad[1]), _keys(13, "cuda"))
    x.requires_grad_(True)
    # Frozen parameters exercise ordinary autograd without a parameter session.
    output = op(x, weight, bias)
    output.backward(grad)
    _assert_bytes(output[1], single[0])
    _assert_bytes(x.grad[1], single[1])
    with torch.no_grad():
        changed = x.detach().clone()
        changed[0].add_(5)
        changed[2].sub_(7)
        _assert_bytes(op(changed, weight, bias)[1], output[1])


@requires_sm90
@pytest.mark.parametrize("rows", [17, 129])
@pytest.mark.parametrize("expanded_upstream", [False, True])
def test_noncontiguous_upstream_and_forward_graph(expanded_upstream, rows):
    op = _op()
    x, weight, bias, grad = _data(rows)
    grad = (
        torch.ones((), device="cuda").expand_as(grad)
        if expanded_upstream
        else grad.t().contiguous().t()
    )
    assert not grad.is_contiguous()
    with canonical_backward_session() as session:
        output = op(
            x.requires_grad_(),
            weight.requires_grad_(),
            bias.requires_grad_(),
            logical_keys=_keys(rows, "cuda"),
            parameter_id="proj_out",
        )
        output.backward(grad)
        session.validate_complete()
    _assert_result_accuracy(
        (output, x.grad, weight.grad, bias.grad), _reference((x, weight, bias, grad))
    )
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                expected = op(x, weight, bias)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(x, weight, bias)
        graph.replay()
        _assert_bytes(captured, expected)


@requires_sm90
@pytest.mark.parametrize("with_bias", [False, True])
def test_projection_then_video_index_select(with_bias):
    # PyTorch selection is a test-only integration reference for packed_index_select.
    x, weight, bias, _ = _data(51)
    x = x.reshape(3, 17, 5376).requires_grad_()
    weight.requires_grad_(True)
    bias = bias.requires_grad_() if with_bias else None
    video_indices = torch.tensor([1, 8, 3, 16, 10], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(422)
    selected_grad = torch.randn(3, 5, 96, generator=generator, device="cuda") * 0.0625
    with canonical_backward_session() as session:
        full = _op()(
            x,
            weight,
            bias,
            logical_keys=_keys(51, "cuda"),
            parameter_id="proj_out",
        )
        selected = full.index_select(1, video_indices)
        selected.backward(selected_grad)
        session.validate_complete()
    cx, cw = [t.detach().cpu().requires_grad_() for t in (x, weight)]
    cb = None if bias is None else bias.detach().cpu().requires_grad_()
    reference = NativeVideoOutputProjectionFP32Op()(cx, cw, cb)
    reference_selected = reference.index_select(1, video_indices.cpu())
    reference_selected.backward(selected_grad.cpu())
    _assert_result_accuracy(
        (selected, x.grad, weight.grad, None if bias is None else bias.grad),
        (reference_selected, cx.grad, cw.grad, None if cb is None else cb.grad),
    )
    unused = torch.ones(17, dtype=torch.bool, device="cuda")
    unused[video_indices] = False
    assert torch.count_nonzero(x.grad[:, unused]) == 0
    # An explicit full gradient gives the same projection VJP as unique row selection.
    full_grad = torch.zeros_like(full)
    full_grad.index_copy_(1, video_indices, selected_grad)
    explicit = _run(
        _op(),
        (x.detach().reshape(51, 5376), weight, bias, full_grad.reshape(51, 96)),
        _keys(51, "cuda"),
    )
    _assert_bytes(x.grad.reshape(51, 5376), explicit[1])
    _assert_bytes(weight.grad, explicit[2])
    _assert_bytes(None if bias is None else bias.grad, explicit[3])


@requires_sm90
def test_default_cuda_dispatch():
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("video_output_projection_fp32", device="cuda")
    assert op.backend_info()["backend_id"] == "triton-video-output-projection-fp32"
    x, weight, bias, _ = _data(3)
    with torch.no_grad():
        _assert_bytes(op(x, weight, bias), _op()(x, weight, bias))


@requires_sm90
def test_projection_rejects_mixed_devices():
    x, weight, bias, _ = _data(2)
    with pytest.raises(ValueError, match="projection tensors must be on the same device"):
        _op()(x, weight.cpu(), bias)


def _layout(variant, frames, height, width, context_rows):
    """Build row counts and ranges from the pinned Diffusers 0.40.0 layout."""
    rows_per_frame = (height // 32) * (width // 32)
    latent_frames = (frames - 5) // 17 * 5 + 2
    target_audio = 2 * round(frames / 24 * 40)
    target_video = latent_frames * rows_per_frame
    cursor, video_ranges = context_rows, []
    if variant == "fl2va":
        video_ranges.append((cursor, cursor + 2 * rows_per_frame))
        cursor += 2 * rows_per_frame
    elif variant == "ref2va":
        image_rows = 9 * (2048 // 32) * (8192 // 32)
        video_ranges.append((cursor, cursor + image_rows))
        cursor += image_rows
        # The final 248-frame clip encodes 243 video frames, but its full audio.
        # Three clips total 15 seconds; each waveform is padded to 800 samples.
        for reference_frames in (56, 56, 248):
            cursor += 2 * math.ceil(int(reference_frames / 24 * 32000) / 800)
            reference_latents = (reference_frames - 5) // 17 * 5 + 2
            count = reference_latents * rows_per_frame
            video_ranges.append((cursor, cursor + count))
            cursor += count
    cursor += target_audio
    video_ranges.append((cursor, cursor + target_video))
    cursor += target_video
    return cursor, video_ranges


def _tensor_digest(tensor, row_order=None):
    """Hash all FP32 bytes in logical order without retaining a full CPU copy."""
    digest = hashlib.sha256()
    value = tensor.detach()
    for start in range(0, value.shape[0], 4096):
        chunk = (
            value[start : start + 4096]
            if row_order is None
            else value.index_select(0, row_order[start : start + 4096])
        )
        digest.update(chunk.cpu().contiguous().view(torch.int32).numpy().tobytes())
    return digest.hexdigest()


def _sampled_layout_accuracy(data, actual):
    """Check sampled y/dX/dW and all dbias with independent full-K FP32 sums."""
    x, weight, bias, grad = data
    rows = x.shape[0]
    sample_rows = torch.tensor([0, 1, rows // 2, rows - 2, rows - 1], device=x.device)
    input_columns = torch.tensor([0, 31, 95, 127, 1024, 4095, 5375], device=x.device)
    output_columns = torch.tensor([0, 47, 95], device=x.device)
    cx = x.detach().index_select(0, sample_rows).cpu()
    cw, cb = weight.detach().cpu(), bias.detach().cpu()
    cg = grad.index_select(0, sample_rows).cpu()
    expected_y = _reference_matmul_fp32(cx, cw.t()) + cb
    expected_dx = _reference_matmul_fp32(cg, cw)
    expected_dw = _reference_matmul_fp32(
        grad.index_select(1, output_columns).t(), x.detach().index_select(1, input_columns)
    )
    expected_dbias = grad.cpu().sum(dim=0)
    _assert_accuracy(actual[0].index_select(0, sample_rows), expected_y, "forward_accuracy")
    _assert_accuracy(actual[1].index_select(0, sample_rows), expected_dx, "gradient_accuracy")
    _assert_accuracy(
        actual[2].index_select(0, output_columns).index_select(1, input_columns),
        expected_dw,
        "gradient_accuracy",
    )
    _assert_accuracy(actual[3], expected_dbias, "gradient_accuracy")


def _run_layout(rows, video_ranges, *, permuted_chunks):
    # Regeneration keeps only one pass resident on the GPU. Outputs are represented
    # by complete byte digests on the host, not by another 12.6 GiB dX tensor.
    data = _data(rows)
    x, weight, bias, grad = data
    mask = torch.zeros(rows, dtype=torch.bool, device="cuda")
    for start, end in video_ranges:
        mask[start:end] = True
    grad[~mask] = 0
    keys = _keys(rows, "cuda")
    order = None
    if permuted_chunks:
        generator = torch.Generator(device="cuda").manual_seed(421)
        order = torch.randperm(rows, generator=generator, device="cuda")
        x = x.index_select(0, order)
        data[0] = x
        grad = grad.index_select(0, order)
        data[3] = grad
        keys = keys.index_select(0, order)
    x.requires_grad_(True)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    outputs = []
    with canonical_backward_session() as session:
        # Slices are views of x; concatenating cloned hidden states is unnecessary.
        cuts = (0, rows // 3, 2 * rows // 3, rows) if permuted_chunks else (0, rows)
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            outputs.append(
                _op()(
                    x[start:end],
                    weight,
                    bias,
                    logical_keys=keys[start:end],
                    parameter_id="proj_out",
                )
            )
        torch.autograd.backward(
            outputs, [grad[start:end] for start, end in zip(cuts[:-1], cuts[1:], strict=True)]
        )
        session.validate_complete()
    actual = (torch.cat(outputs), x.grad, weight.grad, bias.grad)
    if not permuted_chunks:
        _sampled_layout_accuracy(data, actual)
    inverse = None if order is None else order.argsort()
    return tuple(
        _tensor_digest(value, inverse if index < 2 else None) for index, value in enumerate(actual)
    )


@requires_sm90
@pytest.mark.parametrize(
    "variant,frames,height,width,context_rows,expected_rows,expected_video_rows",
    [
        pytest.param("t2va", 124, 768, 1344, 1024, 38734, 37296, id="124f-768p"),
        pytest.param("t2va", 243, 1344, 768, 1024, 74410, 72576, id="243f-768p"),
        pytest.param("t2va", 345, 768, 1344, 1024, 104990, 102816, id="345f-768p"),
        pytest.param("fl2va", 345, 576, 1856, 262144, 371870, 108576, id="fl2va-context-limit"),
        pytest.param(
            "ref2va", 345, 576, 1856, 262144, 629106, 364608, id="largest_supported_layout"
        ),
    ],
)
def test_task_layout_sampled_accuracy_and_full_bi(
    variant, frames, height, width, context_rows, expected_rows, expected_video_rows
):
    # L includes all conditioner rows, including reference vision tokens. The
    # largest profiles use the checkpoint's declared 262144-token context limit;
    # they are operator shape tests, not full-model inference or encoder tests.
    rows, video_ranges = _layout(variant, frames, height, width, context_rows)
    assert rows == expected_rows
    assert sum(end - start for start, end in video_ranges) == expected_video_rows
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    # Five hidden-state-sized allocations cover input, dX, canonical payloads,
    # and sorting/indexing temporaries. Keep an additional 4 GiB headroom.
    required_bytes = 5 * rows * 5376 * 4 + 4 * 1024**3
    if free_bytes < required_bytes:
        pytest.skip(
            f"layout {rows} needs approximately {required_bytes / 1024**3:.1f} GiB free; "
            f"device has {free_bytes / 1024**3:.1f} GiB"
        )
    baseline = _run_layout(rows, video_ranges, permuted_chunks=False)
    torch.cuda.empty_cache()
    reordered = _run_layout(rows, video_ranges, permuted_chunks=True)
    assert reordered == baseline
