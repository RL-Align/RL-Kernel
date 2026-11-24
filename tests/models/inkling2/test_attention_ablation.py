# SPDX-License-Identifier: Apache-2.0
"""INK-49 attention-mask, bias, sink, and scaling CUDA ablations."""

import pytest
import torch

from rl_engine.models.inkling2.full_attention import full_attention_cuda


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


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


def _require_extension():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_full_attention_forward"):
        pytest.skip("Inkling 2 full attention CUDA extension is unavailable")


def test_document_mask_ablation_blocks_cross_document_attention():
    _require_extension()
    query, key, value = _attention_inputs()
    documents = torch.tensor([[0, 0, 1, 1]], device="cuda")

    changed_value = value.clone()
    changed_value[:, 2:] = changed_value[:, 2:] + 20.0
    global_output = full_attention_cuda(query, key, value, causal=False)
    changed_global_output = full_attention_cuda(query, key, changed_value, causal=False)
    local_output = full_attention_cuda(
        query,
        key,
        value,
        query_documents=documents,
        key_documents=documents,
        causal=False,
    )
    changed_local_output = full_attention_cuda(
        query,
        key,
        changed_value,
        query_documents=documents,
        key_documents=documents,
        causal=False,
    )

    assert not torch.allclose(global_output, local_output)
    assert not torch.allclose(global_output[:, :2], changed_global_output[:, :2])
    torch.testing.assert_close(local_output[:, :2], changed_local_output[:, :2])


def test_relative_bias_ablation_changes_only_biased_attention():
    _require_extension()
    query, key, value = _attention_inputs()
    bias = torch.zeros(2, 4, 4, device="cuda")
    bias[:, :, 0] = 8.0

    without_bias = full_attention_cuda(query, key, value, causal=False)
    with_bias = full_attention_cuda(query, key, value, relative_bias=bias, causal=False)

    assert not torch.allclose(with_bias, without_bias)
    assert torch.all(with_bias[..., 0] > without_bias[..., 0])


def test_sink_ablation_changes_probability_mass_without_value_keys():
    _require_extension()
    query, key, value = _attention_inputs()
    sink = torch.full((2,), 8.0, device="cuda")

    without_sink = full_attention_cuda(query, key, value, causal=False)
    with_sink = full_attention_cuda(query, key, value, sink=sink, causal=False)

    assert not torch.allclose(with_sink, without_sink)
    assert torch.linalg.vector_norm(with_sink) < torch.linalg.vector_norm(without_sink)


def test_log_scaling_ablation_changes_softmax_temperature():
    _require_extension()
    query, key, value = _attention_inputs()

    default_scale = full_attention_cuda(query, key, value, causal=False)
    sharp_scale = full_attention_cuda(query, key, value, scale=4.0, causal=False)

    assert not torch.allclose(sharp_scale, default_scale)
    assert torch.linalg.vector_norm(sharp_scale) > torch.linalg.vector_norm(default_scale)
