# SPDX-License-Identifier: Apache-2.0
"""The LM-head dgrad must retain BF16 rounding at canonical TP leaves."""
import pytest
import torch

from rl_engine.integrations import megatron_runtime as runtime
from rl_engine.kernels.ops.matmul import det_gemm


def tree(parts):
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    return parts[0]


@pytest.mark.parametrize('physical_tp', [1, 2, 4, 8])
def test_column_dgrad_matches_canonical_bf16_tree(monkeypatch, physical_tp):
    monkeypatch.setenv('RL_KERNEL_STRICT_CANONICAL_TP', '8')
    monkeypatch.setattr(det_gemm, 'det_gemm_linear_input_gradient', lambda dy, w: dy @ w)
    gen = torch.Generator().manual_seed(1729)
    dy = torch.randn(7, 256, generator=gen).bfloat16()
    weight = torch.randn(256, 32, generator=gen).bfloat16()
    # Include vocabulary padding in the physical and canonical shard widths.
    dy[:, -17:] = 0
    reference = tree([dy[:, i:i+32] @ weight[i:i+32] for i in range(0, 256, 32)])
    width = 256 // physical_tp
    actual = tree([
        runtime._canonical_column_input_gradient(
            dy[:, i:i+width].contiguous(), weight[i:i+width].contiguous(), physical_tp,
        ) for i in range(0, 256, width)
    ])
    assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
    # This fixture exposes the original unchunked-GEMM regression.
    assert not torch.equal(dy @ weight, reference)


@pytest.mark.parametrize(('canonical', 'tp', 'width'), [(3, 2, 12), (6, 2, 12), (8, 2, 6), (0, 2, 8)])
def test_column_dgrad_rejects_invalid_tree(monkeypatch, canonical, tp, width):
    monkeypatch.setenv('RL_KERNEL_STRICT_CANONICAL_TP', str(canonical))
    with pytest.raises(ValueError):
        runtime._canonical_column_input_gradient(torch.zeros(2, width), torch.zeros(width, 4), tp)


@pytest.mark.parametrize('shape', [(6, 8), (3, 2, 8)])
def test_lm_head_autograd_uses_canonical_subtree(monkeypatch, shape):
    from rl_engine.kernels.ops.cuda.loss import linear_logp
    monkeypatch.setenv('RL_KERNEL_STRICT_CANONICAL_TP', '4')
    monkeypatch.setattr(runtime, '_tp_world_size', lambda group: 2)
    monkeypatch.setattr(det_gemm, 'det_gemm_linear', lambda x, w: x @ w.t())
    monkeypatch.setattr(det_gemm, 'det_gemm_linear_input_gradient', lambda dy, w: dy @ w)
    monkeypatch.setattr(det_gemm, 'det_gemm_linear_weight_gradient', lambda x, dy: dy.t() @ x)
    reduced = []
    monkeypatch.setattr(linear_logp, '_deterministic_tp_all_reduce_', lambda dx, group: reduced.append(dx.clone()))
    gen = torch.Generator().manual_seed(41)
    x = torch.randn(shape, generator=gen).bfloat16().requires_grad_()
    w = torch.randn(32, 8, generator=gen).bfloat16().requires_grad_()
    b = torch.randn(32, generator=gen).bfloat16().requires_grad_()
    output = runtime._DeterministicTPOutputProjection.apply(x, w, b, object())
    dy = torch.randn(output.shape, generator=gen).bfloat16()
    output.backward(dy)
    rows = lambda t: (t.transpose(0, 1).contiguous() if t.ndim == 3 else t).reshape(-1, t.shape[-1])
    expected = rows(dy)[:, :16] @ w[:16] + rows(dy)[:, 16:] @ w[16:]
    assert len(reduced) == 1
    assert torch.equal(reduced[0], expected)
    assert torch.equal(rows(x.grad), expected)
    assert torch.equal(w.grad, rows(dy).t() @ rows(x))
    assert torch.equal(b.grad, rows(dy).float().sum(0).bfloat16())


@pytest.mark.parametrize('te', [False, True])
def test_qkv_installed_forward_uses_canonical_backward(monkeypatch, te):
    gen = torch.Generator().manual_seed(91)
    weight = torch.randn(32, 8, generator=gen).bfloat16().requires_grad_()
    class Column:
        def __init__(self):
            self.weight = weight
            if te:
                self.layer_norm_weight = torch.ones(8)
        def _forward_impl(self, x, w, **kwargs):
            return x @ w.t()
        def forward(self, x):
            return self._forward_impl(x, self.weight), None
    class Row:
        def __init__(self):
            self.weight = weight.t()
        def _forward_impl(self, x, w, **kwargs):
            return x @ w.t()
    class Attention:
        def __init__(self):
            self.linear_qkv = Column()
            self.linear_proj = Row()
    monkeypatch.setenv('RL_KERNEL_STRICT_CANONICAL_TP', '4')
    monkeypatch.setattr(runtime, '_tp_world_size', lambda group: 2)
    monkeypatch.setattr(runtime, '_fused_rms_norm_input', lambda module, x, name: x)
    monkeypatch.setattr(det_gemm, 'det_gemm_linear_input_gradient', lambda dy, w: dy @ w)
    monkeypatch.setattr(det_gemm, 'det_gemm_linear_weight_gradient', lambda x, dy: dy.t() @ x)
    runtime._patch_strict_attention_projections(
        self_attention_cls=Attention, column_linear_cls=Column, row_linear_cls=Row,
        det_gemm=lambda a, b: a @ b, copy_to_tp=lambda x: x, reduce_from_tp=lambda x: x,
    )
    x = torch.randn(3, 2, 8, generator=gen).bfloat16().requires_grad_()
    output, bias = Attention().linear_qkv.forward(x)
    dy = torch.randn(output.shape, generator=gen).bfloat16()
    output.backward(dy)
    flat = dy.reshape(-1, 32)
    expected = flat[:, :16] @ weight[:16] + flat[:, 16:] @ weight[16:]
    assert bias is None
    assert torch.equal(x.grad.reshape(-1, 8), expected)
    assert torch.equal(weight.grad, flat.t() @ x.detach().reshape(-1, 8))
