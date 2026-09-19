from types import SimpleNamespace
import pytest
import torch
from rl_engine.integrations.canonical_cp import packed_gather_indices, weight_gradient, CPRMSNorm
from rl_engine.kernels.ops.matmul import det_gemm


def test_checkpoint_recompute_retains_forward_layout():
    from torch.utils.checkpoint import checkpoint
    from rl_engine.integrations.canonical_cp import bind_layout, current_layout
    layout=object()
    seen=[]
    def function(x):
        seen.append((torch.is_grad_enabled(),current_layout()))
        return x.square()
    x=torch.ones(3,requires_grad=True)
    output=checkpoint(bind_layout(function,layout),x,use_reentrant=True)
    assert current_layout() is None
    output.sum().backward()
    assert seen==[(False,layout),(True,layout)]
    assert current_layout() is None
    assert torch.equal(x.grad,torch.full_like(x,2))


@pytest.mark.parametrize('cp', [1, 2, 4, 8])
def test_packed_rows_restore_token_order_without_padding(cp):
    sequences=[torch.arange(13), torch.arange(101, 128), torch.arange(200, 218)]
    ranks=[]
    for rank in range(cp):
        local=[]
        for seq in sequences:
            if cp==1:
                local.append(seq)
            else:
                width=(seq.numel()+2*cp-1)//(2*cp)
                padded=torch.nn.functional.pad(seq,(0,2*cp*width-seq.numel()),value=-1)
                pieces=padded.chunk(2*cp)
                local.extend([pieces[rank],pieces[2*cp-rank-1]])
        ranks.append(torch.nn.functional.pad(torch.cat(local),(0,7),value=-2))
    order=packed_gather_indices([len(s) for s in sequences],len(ranks[0]),cp)
    assert torch.equal(torch.cat(ranks)[order],torch.cat(sequences))


@pytest.mark.parametrize('cp', [1, 2, 4])
@pytest.mark.parametrize('chunks',[1,2,4])
def test_canonical_parameter_gradient_cancels_cp_loss_scale(monkeypatch,cp,chunks):
    monkeypatch.setattr(det_gemm,'det_gemm_linear_weight_gradient',lambda x,dy:dy.t()@x)
    gen=torch.Generator().manual_seed(18)
    x=torch.randn(21,16,generator=gen).bfloat16()
    dy=torch.randn(21,32,generator=gen).bfloat16()
    layout=SimpleNamespace(cp_world=cp,gather_many=lambda *args:(x,dy*cp))
    actual=weight_gradient(x[:1],dy[:1]*cp,layout,chunks)
    assert torch.equal(actual,dy.t()@x)


@pytest.mark.parametrize('cp',[1,2,4])
def test_norm_parameter_gradient_uses_all_logical_rows(cp):
    gen=torch.Generator().manual_seed(2)
    full_x=torch.randn(16,1,32,generator=gen).bfloat16()
    full_dy=torch.randn(16,1,32,generator=gen).bfloat16()
    w=torch.randn(32,generator=gen).bfloat16().requires_grad_()
    reference_w=w.detach().clone().requires_grad_()
    torch.nn.functional.rms_norm(full_x,(32,),reference_w,1e-6).backward(full_dy)
    layout=SimpleNamespace(cp_world=cp,gather_many=lambda *args:(full_x,full_dy*cp))
    x=full_x[:16//cp].detach().requires_grad_()
    CPRMSNorm.apply(x,w,1e-6,layout,False).backward(full_dy[:16//cp]*cp)
    assert torch.equal(w.grad,reference_w.grad)
