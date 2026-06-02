# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp
from rl_engine.kernels.registry import kernel_registry
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger


def _fused_logp_op(op_type: str = "logp"):
    return kernel_registry.get_op(op_type)


def _reference_selected_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    ref_logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(ref_logp, dim=-1, index=token_ids.long().unsqueeze(-1)).squeeze(-1)


def test_native_logp_op_exposes_full_fused_logp_api():
    op = NativeLogpOp()

    for method_name in (
        "apply",
        "out",
        "apply_fp32",
        "indexed_out",
        "indexed_fp32",
        "online_out",
        "online_fp32",
        "online_indexed_out",
        "online_indexed_fp32",
    ):
        assert callable(getattr(op, method_name))


def test_native_fused_logp_out_reuses_output_storage_cpu():
    logits = torch.randn(2, 3, 17)
    token_ids = torch.randint(0, logits.size(-1), (2, 3))
    output = torch.empty(logits.shape[:-1])

    result = NativeLogpOp().out(logits, token_ids, output)

    assert result is output
    assert result.data_ptr() == output.data_ptr()
    assert torch.allclose(result, _reference_selected_logp(logits, token_ids))


def test_native_fused_logp_indexed_out_preserves_inactive_rows_cpu():
    logits = torch.randn(3, 4, 19)
    token_ids = torch.randint(0, logits.size(-1), (3, 4))
    mask = torch.zeros((3, 4), dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 3] = True
    mask[2, 0] = True
    row_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    output = torch.full(logits.shape[:-1], 123.0)

    result = NativeLogpOp().indexed_out(logits, token_ids, row_indices, output)
    ref_logp = _reference_selected_logp(logits, token_ids)

    assert result is output
    assert result.data_ptr() == output.data_ptr()
    assert torch.allclose(result[mask], ref_logp[mask])
    assert torch.equal(result[~mask], torch.full_like(result[~mask], 123.0))


def test_native_fused_logp_online_out_matches_reference_cpu():
    logits = torch.randn(2, 5, 23)
    token_ids = torch.randint(0, logits.size(-1), (2, 5))
    output = torch.empty(logits.shape[:-1])

    result = NativeLogpOp().online_out(logits, token_ids, output)

    assert result is output
    assert torch.allclose(result, _reference_selected_logp(logits, token_ids))


def test_native_fused_logp_online_indexed_out_preserves_inactive_rows_cpu():
    logits = torch.randn(2, 6, 29)
    token_ids = torch.randint(0, logits.size(-1), (2, 6))
    mask = torch.zeros((2, 6), dtype=torch.bool)
    mask[:, 1::2] = True
    row_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    output = torch.full(logits.shape[:-1], -77.0)

    result = NativeLogpOp().online_indexed_out(logits, token_ids, row_indices, output)
    ref_logp = _reference_selected_logp(logits, token_ids)

    assert result is output
    assert torch.allclose(result[mask], ref_logp[mask])
    assert torch.equal(result[~mask], torch.full_like(result[~mask], -77.0))


@pytest.mark.parametrize("method_name", ("indexed_fp32", "online_indexed_fp32"))
def test_native_fused_logp_indexed_fp32_zero_fills_inactive_rows_cpu(method_name: str):
    logits = torch.randn(2, 5, 31)
    token_ids = torch.randint(0, logits.size(-1), (2, 5))
    mask = torch.zeros((2, 5), dtype=torch.bool)
    mask[:, ::2] = True
    row_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

    result = getattr(NativeLogpOp(), method_name)(logits, token_ids, row_indices)
    ref_logp = _reference_selected_logp(logits, token_ids)

    assert result.dtype == torch.float32
    assert torch.allclose(result[mask], ref_logp[mask])
    assert torch.equal(result[~mask], torch.zeros_like(result[~mask]))


def test_native_fused_logp_rejects_out_of_range_row_indices_cpu():
    logits = torch.randn(2, 3, 11)
    token_ids = torch.randint(0, logits.size(-1), (2, 3))
    output = torch.empty(logits.shape[:-1])

    with pytest.raises(ValueError, match="out-of-range"):
        NativeLogpOp().indexed_out(logits, token_ids, torch.tensor([0, 6]), output)


def test_accuracy():
    device = device_ctx.device
    dtype = device_ctx.get_preferred_dtype()

    logger.info(f"Running Accuracy Test on: {device} | Dtype: {dtype}")

    G, L, V = 16, 128, 4096

    logits = torch.randn(G * L, V, device=device, dtype=dtype)
    token_ids = torch.randint(0, V, (G * L,), device=device, dtype=torch.int32)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        ref_logp = torch.log_softmax(logits.float(), dim=-1)
        ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1).long()).squeeze(-1)
        ref_logp = ref_logp.to(dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()

    try:
        logp_operator = _fused_logp_op()
        custom_logp = logp_operator(logits, token_ids)
    except Exception as e:
        logger.error(f"Failed to execute FusedLogp: {e}")
        raise

    diff = torch.abs(ref_logp.float() - custom_logp.float()).max().item()
    threshold = 1e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-5

    print("\n" + "=" * 50)
    print(f"RESULTS FOR {str(device).upper()}")
    print("-" * 50)
    print(f"Dispatched Operator Class: {logp_operator.__class__.__name__}")
    print(f"Max Difference: {diff:.8e}")

    if diff < threshold:
        print("Status: Accuracy Check Passed!")
    else:
        print("Status: Accuracy Check Failed! (Check your CUDA reduction logic)")
    print("=" * 50 + "\n")

    assert diff < threshold


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_logp_out_reuses_output_storage():
    device = torch.device("cuda")
    dtype = torch.float16
    logits = torch.randn(2, 5, 257, device=device, dtype=dtype)
    token_ids = torch.randint(0, logits.size(-1), (2, 5), device=device)
    output = torch.empty(logits.shape[:-1], device=device, dtype=dtype)

    ref_logp = torch.log_softmax(logits.float(), dim=-1)
    ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).to(dtype)
    result = _fused_logp_op().out(logits, token_ids, output)

    assert result.data_ptr() == output.data_ptr()
    assert torch.allclose(result, ref_logp, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_logp_fp32_output():
    device = torch.device("cuda")
    logits = torch.randn(2, 4, 129, device=device, dtype=torch.float16)
    token_ids = torch.randint(0, logits.size(-1), (2, 4), device=device)

    ref_logp = torch.log_softmax(logits.float(), dim=-1)
    ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    result = _fused_logp_op().apply_fp32(logits, token_ids)

    assert result.dtype == torch.float32
    assert torch.allclose(result, ref_logp, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_logp_indexed_out_only_updates_valid_rows():
    device = torch.device("cuda")
    dtype = torch.float16
    logits = torch.randn(3, 4, 131, device=device, dtype=dtype)
    token_ids = torch.randint(0, logits.size(-1), (3, 4), device=device)
    mask = torch.zeros((3, 4), device=device, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 3] = True
    mask[2, 0] = True
    row_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    output = torch.zeros(logits.shape[:-1], device=device, dtype=dtype)

    ref_logp = torch.log_softmax(logits.float(), dim=-1)
    ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).to(dtype)
    result = _fused_logp_op("logp_indexed").indexed_out(logits, token_ids, row_indices, output)

    assert result.data_ptr() == output.data_ptr()
    assert torch.allclose(result[mask], ref_logp[mask], atol=1e-3, rtol=1e-3)
    assert torch.equal(result[~mask], torch.zeros_like(result[~mask]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_logp_indexed_fp32_output():
    device = torch.device("cuda")
    logits = torch.randn(2, 5, 137, device=device, dtype=torch.float16)
    token_ids = torch.randint(0, logits.size(-1), (2, 5), device=device)
    mask = torch.zeros((2, 5), device=device, dtype=torch.bool)
    mask[:, ::2] = True
    row_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

    ref_logp = torch.log_softmax(logits.float(), dim=-1)
    ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    result = _fused_logp_op("logp_indexed").indexed_fp32(logits, token_ids, row_indices)

    assert result.dtype == torch.float32
    assert torch.allclose(result[mask], ref_logp[mask], atol=1e-3, rtol=1e-3)
    assert torch.equal(result[~mask], torch.zeros_like(result[~mask]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_logp_online_out_matches_reference():
    device = torch.device("cuda")
    dtype = torch.float16
    logits = torch.randn(2, 4, 263, device=device, dtype=dtype)
    token_ids = torch.randint(0, logits.size(-1), (2, 4), device=device)
    output = torch.empty(logits.shape[:-1], device=device, dtype=dtype)

    ref_logp = torch.log_softmax(logits.float(), dim=-1)
    ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).to(dtype)
    result = _fused_logp_op("logp_online").online_out(logits, token_ids, output)

    assert result.data_ptr() == output.data_ptr()
    assert torch.allclose(result, ref_logp, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_logp_online_indexed_out_only_updates_valid_rows():
    device = torch.device("cuda")
    dtype = torch.float16
    logits = torch.randn(3, 3, 251, device=device, dtype=dtype)
    token_ids = torch.randint(0, logits.size(-1), (3, 3), device=device)
    mask = torch.zeros((3, 3), device=device, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 2] = True
    mask[2, 1] = True
    row_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    output = torch.zeros(logits.shape[:-1], device=device, dtype=dtype)

    ref_logp = torch.log_softmax(logits.float(), dim=-1)
    ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).to(dtype)
    result = _fused_logp_op("logp_online_indexed").online_indexed_out(
        logits,
        token_ids,
        row_indices,
        output,
    )

    assert result.data_ptr() == output.data_ptr()
    assert torch.allclose(result[mask], ref_logp[mask], atol=1e-3, rtol=1e-3)
    assert torch.equal(result[~mask], torch.zeros_like(result[~mask]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_logp_online_indexed_fp32_output():
    device = torch.device("cuda")
    logits = torch.randn(2, 6, 193, device=device, dtype=torch.float16)
    token_ids = torch.randint(0, logits.size(-1), (2, 6), device=device)
    mask = torch.zeros((2, 6), device=device, dtype=torch.bool)
    mask[:, 1::2] = True
    row_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

    ref_logp = torch.log_softmax(logits.float(), dim=-1)
    ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    result = _fused_logp_op("logp_online_indexed").online_indexed_fp32(
        logits,
        token_ids,
        row_indices,
    )

    assert result.dtype == torch.float32
    assert torch.allclose(result[mask], ref_logp[mask], atol=1e-3, rtol=1e-3)
    assert torch.equal(result[~mask], torch.zeros_like(result[~mask]))


if __name__ == "__main__":
    test_accuracy()
