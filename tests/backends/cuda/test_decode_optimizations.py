import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("batch", [1, 4, 64])
def test_strided_linear_forward_backward_matches_contiguous(batch, monkeypatch):
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "cublaslt_nosplitk")
    from rl_engine.backends.cuda.gemm.det_gemm import DetGemmOp

    torch.manual_seed(1234)
    x = (
        torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)[:, 512:]
        .detach()
        .requires_grad_()
    )
    w = (
        torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)[:, 512:]
        .detach()
        .requires_grad_()
    )
    xc, wc = x.detach().contiguous().requires_grad_(), w.detach().contiguous().requires_grad_()
    op = DetGemmOp()
    result, reference = op.linear(x, w), op.linear(xc, wc)
    assert torch.equal(result, reference)
    dy = torch.randn_like(result)
    result.backward(dy)
    reference.backward(dy)
    assert torch.equal(x.grad, xc.grad)
    assert torch.equal(w.grad, wc.grad)


@pytest.mark.parametrize("batch", [1, 4, 64])
@pytest.mark.parametrize("length", [513, 4097, 7168])
@pytest.mark.parametrize("heads", [(4, 1), (8, 2), (16, 4)])
def test_small_query_tile_is_bitwise_equal_with_shuffled_pages(batch, length, heads):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper query tile")
    from rl_engine.backends.cuda.attention.flash_attn import StrictFlashAttention4Core

    torch.manual_seed(1234 + length)
    page = 16
    pages = (length + page - 1) // page
    q = torch.randn(batch, 1, heads[0], 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * pages, page, heads[1], 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kwargs = dict(
        page_table=torch.randperm(batch * pages, device="cuda", dtype=torch.int32).reshape(
            batch, pages
        ),
        seqused_k=torch.randint(
            max(1, length - 200), length + 1, (batch,), device="cuda", dtype=torch.int32
        ),
        max_seqlen_k=pages * page,
    )
    core = StrictFlashAttention4Core()
    assert core._paged_decode_fwd is not None
    result = core.forward_paged_bshd_with_lse(q, k, v, **kwargs)
    core._paged_decode_fwd = None
    reference = core.forward_paged_bshd_with_lse(q, k, v, **kwargs)
    assert torch.equal(result.out, reference.out)
    assert torch.equal(result.lse, reference.lse)


def _restore_dispatchers_after_test(monkeypatch):
    """install_sort_contract / install_capture rebind apply_top_k_top_p in every
    loaded module that imported it; restore all of them after the test."""
    import sys

    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict) and "apply_top_k_top_p" in namespace:
            monkeypatch.setattr(module, "apply_top_k_top_p", namespace["apply_top_k_top_p"])


@pytest.mark.parametrize(
    ("batch", "top_p"),
    [(1, (0.8, 1.0)), (4, (0.8, 1.0)), (64, (0.8, 1.0)), (64, (0.95, 0.95)), (8, (0.95, 0.95))],
)
def test_captured_full_support_matches_independent_policy(batch, top_p, monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler as native

    from rl_engine.integrations.engines.rollout.vllm import sampling_support
    from rl_engine.ops.sampling.policy import sampling_keep_mask

    _restore_dispatchers_after_test(monkeypatch)
    # vLLM's sort, read before anything is installed (vLLM 0.16 and older:
    # the dispatcher is the sort).
    sort = getattr(native, "apply_top_k_top_p_pytorch", native.apply_top_k_top_p)

    torch.manual_seed(1234)
    logits = torch.randn(batch, 151936, device="cuda", dtype=torch.float32)
    p = torch.linspace(*top_p, batch, device="cuda")
    k = torch.full((batch,), logits.size(1), device="cuda", dtype=torch.int32)
    expected = sampling_keep_mask(logits, temperature=0.7, top_p=p, top_k=k)
    reference = sort(logits.clone() / 0.7, k, p)
    sampling_support.install_capture()
    with sampling_support.capture_support(True) as support:
        result = native.apply_top_k_top_p(logits.clone() / 0.7, k, p)
    assert torch.equal(result, reference)
    assert torch.equal(support["mask"], expected)
    assert torch.equal(torch.isfinite(result), expected)
    with sampling_support.capture_support(False) as disabled:
        native.apply_top_k_top_p(logits.clone(), k, p)
    assert not disabled


def test_sort_contract_reaches_the_gpu_worker_sampler(monkeypatch):
    """The worker sampler imports apply_top_k_top_p by name."""

    import sys

    states = pytest.importorskip("vllm.v1.worker.gpu.sample.states")
    worker = pytest.importorskip("vllm.v1.worker.gpu.sample.sampler")
    from vllm.v1.sample.ops import topk_topp_sampler as native

    from rl_engine.integrations.engines.rollout.vllm import sampling_support

    if not hasattr(native, "apply_top_k_top_p_pytorch"):
        pytest.skip("this vLLM has no Triton dispatcher; apply_top_k_top_p is the sort")
    try:  # another by-name importer in vLLM 0.30
        import vllm.v1.watermarking.gpu_sampler  # noqa: F401
    except ImportError:
        pass
    _restore_dispatchers_after_test(monkeypatch)
    dispatcher = native.apply_top_k_top_p
    bound = [
        module
        for module in list(sys.modules.values())
        if isinstance(getattr(module, "__dict__", None), dict)
        and module.__dict__.get("apply_top_k_top_p") is dispatcher
    ]
    assert sampling_support.install_sort_contract()
    assert states.apply_top_k_top_p is native.apply_top_k_top_p
    if hasattr(worker, "apply_top_k_top_p"):  # vLLM 0.17 calls it through states only
        assert worker.apply_top_k_top_p is native.apply_top_k_top_p
    assert all(module.apply_top_k_top_p is native.apply_top_k_top_p for module in bound)

    torch.manual_seed(7)
    logits = torch.randn(64, 151936, device="cuda") / 0.7
    p = torch.full((64,), 0.95, device="cuda")
    expected = torch.isfinite(native.apply_top_k_top_p_pytorch(logits.clone(), None, p))
    assert torch.equal(torch.isfinite(states.apply_top_k_top_p(logits.clone(), None, p)), expected)
