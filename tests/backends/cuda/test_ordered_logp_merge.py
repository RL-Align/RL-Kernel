import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("summaries", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("tokens", [1, 4, 127, 4096, 65536])
def test_ordered_merge_matches_torch_bytes(summaries, tokens):
    from rl_engine.backends.cuda.logprob.ordered_merge import ordered_logp_merge

    torch.manual_seed(1234)
    lse = torch.randn(summaries, tokens, device="cuda") * 20
    target = torch.randn_like(lse)
    lse[:, ::19] = float("-inf")
    lse[-1, ::19] = 0
    maximum = lse[0].clone()
    for s in range(1, summaries):
        maximum = torch.maximum(maximum, lse[s])
    total = torch.exp(lse[0] - maximum)
    zt = target[0].clone()
    for s in range(1, summaries):
        total = total + torch.exp(lse[s] - maximum)
        zt = zt + target[s]
    reference_lse = maximum + torch.log(total)
    reference_logp = zt - reference_lse
    logp, result_lse = ordered_logp_merge(lse, target)
    assert torch.equal(logp.view(torch.int32), reference_logp.view(torch.int32))
    assert torch.equal(result_lse.view(torch.int32), reference_lse.view(torch.int32))
