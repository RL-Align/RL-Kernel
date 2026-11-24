# SPDX-License-Identifier: Apache-2.0
"""INK-49 attention ablations run against the native ROCm kernel (wave64 / MI300X)."""

import pytest
import torch

from rl_engine.models.inkling2.full_attention import full_attention, full_attention_rocm

pytestmark = pytest.mark.skipif(
    getattr(torch.version, "hip", None) is None or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)


@pytest.fixture(autouse=True)
def _binding():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_full_attention_forward"):
        pytest.skip("Inkling 2 full attention extension is unavailable")


def _attention_inputs():
    torch.manual_seed(47)
    query = torch.randn(1, 4, 2, 4, device="cuda", dtype=torch.float32)
    key = torch.randn(1, 4, 1, 4, device="cuda", dtype=torch.float32)
    value = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0, 0.0]],
                [[0.0, 0.0, 3.0, 0.0]],
                [[0.0, 0.0, 0.0, 4.0]],
            ]
        ],
        device="cuda",
    )
    return query, key, value


def _control_kwargs():
    # The "clean" configuration every ablation is compared against: full feature set.
    documents = torch.tensor([[0, 0, 1, 1]], device="cuda")
    bias = torch.zeros(2, 4, 4, device="cuda")
    bias[:, :, 0] = 3.0
    return dict(
        relative_bias=bias,
        sink=torch.tensor([1.0, 2.0], device="cuda"),
        output_gate=torch.tensor([0.5, -0.5], device="cuda"),
        query_documents=documents,
        key_documents=documents,
        causal=False,
    )


def _both(query, key, value, **kwargs):
    return (
        full_attention_rocm(query, key, value, **kwargs),
        full_attention(query, key, value, **kwargs),
    )


def test_clean_control_matches_reference_and_is_bitwise_repeatable():
    query, key, value = _attention_inputs()
    kwargs = _control_kwargs()
    native, reference = _both(query, key, value, **kwargs)
    torch.testing.assert_close(native, reference, atol=1e-6, rtol=1e-5)
    for _ in range(3):
        assert torch.equal(native, full_attention_rocm(query, key, value, **kwargs))


def test_document_mask_ablation_blocks_cross_document_attention():
    query, key, value = _attention_inputs()
    documents = torch.tensor([[0, 0, 1, 1]], device="cuda")
    changed_value = value.clone()
    changed_value[:, 2:] = changed_value[:, 2:] + 20.0
    docs = dict(query_documents=documents, key_documents=documents, causal=False)

    global_output = full_attention_rocm(query, key, value, causal=False)
    changed_global = full_attention_rocm(query, key, changed_value, causal=False)
    local_output = full_attention_rocm(query, key, value, **docs)
    changed_local = full_attention_rocm(query, key, changed_value, **docs)

    # Ablated (mask removed): document 1 values leak into document 0 queries.
    assert not torch.allclose(global_output, local_output)
    assert not torch.allclose(global_output[:, :2], changed_global[:, :2])
    # Intact mask: document 0 queries are bit-for-bit isolated from document 1 values.
    assert torch.equal(local_output[:, :2], changed_local[:, :2])
    torch.testing.assert_close(local_output, full_attention(query, key, value, **docs))


def test_document_mask_ablation_is_detected_in_control_configuration():
    query, key, value = _attention_inputs()
    kwargs = _control_kwargs()
    control = full_attention_rocm(query, key, value, **kwargs)
    ablated = dict(kwargs, query_documents=None, key_documents=None)
    no_mask = full_attention_rocm(query, key, value, **ablated)
    mismatch = (control != no_mask).any(dim=-1).any(dim=-1)[0]
    assert mismatch.any()
    # Drift is first visible in the leaking document-0 queries (positions 0/1).
    assert int(mismatch.nonzero()[0]) == 0


def test_relative_bias_ablation_changes_only_biased_attention():
    query, key, value = _attention_inputs()
    bias = torch.zeros(2, 4, 4, device="cuda")
    bias[:, :, 0] = 8.0

    without_bias = full_attention_rocm(query, key, value, causal=False)
    with_bias = full_attention_rocm(query, key, value, relative_bias=bias, causal=False)
    assert not torch.allclose(with_bias, without_bias)
    assert torch.all(with_bias[..., 0] > without_bias[..., 0])
    # An all-zero bias is the clean control: identical bits to no bias.
    zero = full_attention_rocm(
        query, key, value, relative_bias=torch.zeros_like(bias), causal=False
    )
    assert torch.equal(zero, without_bias)
    torch.testing.assert_close(
        with_bias, full_attention(query, key, value, relative_bias=bias, causal=False)
    )

    kwargs = _control_kwargs()
    control = full_attention_rocm(query, key, value, **kwargs)
    removed = full_attention_rocm(query, key, value, **dict(kwargs, relative_bias=None))
    assert not torch.equal(control, removed)


def test_sink_ablation_changes_probability_mass_without_value_keys():
    query, key, value = _attention_inputs()
    sink = torch.full((2,), 8.0, device="cuda")

    without_sink = full_attention_rocm(query, key, value, causal=False)
    with_sink = full_attention_rocm(query, key, value, sink=sink, causal=False)
    assert not torch.allclose(with_sink, without_sink)
    assert torch.linalg.vector_norm(with_sink) < torch.linalg.vector_norm(without_sink)
    # A very negative sink is the clean control: it contributes no mass.
    negligible = full_attention_rocm(
        query, key, value, sink=torch.full((2,), -1e4, device="cuda"), causal=False
    )
    torch.testing.assert_close(negligible, without_sink, atol=1e-6, rtol=1e-6)

    kwargs = _control_kwargs()
    control = full_attention_rocm(query, key, value, **kwargs)
    removed = full_attention_rocm(query, key, value, **dict(kwargs, sink=None))
    assert not torch.equal(control, removed)


def test_log_scaling_ablation_changes_softmax_temperature():
    query, key, value = _attention_inputs()

    default_scale = full_attention_rocm(query, key, value, causal=False)
    sharp_scale = full_attention_rocm(query, key, value, scale=4.0, causal=False)
    assert not torch.allclose(sharp_scale, default_scale)
    assert torch.linalg.vector_norm(sharp_scale) > torch.linalg.vector_norm(default_scale)
    # Passing the default scale explicitly is the clean control: identical bits.
    explicit = full_attention_rocm(query, key, value, scale=4**-0.5, causal=False)
    assert torch.equal(explicit, default_scale)
    torch.testing.assert_close(
        sharp_scale, full_attention(query, key, value, scale=4.0, causal=False)
    )

    kwargs = _control_kwargs()
    control = full_attention_rocm(query, key, value, **kwargs)
    rescaled = full_attention_rocm(query, key, value, **dict(kwargs, scale=1.0))
    assert not torch.equal(control, rescaled)
