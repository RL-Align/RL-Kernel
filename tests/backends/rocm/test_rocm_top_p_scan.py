# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
import pytest
import torch

IS_GFX942 = (
    torch.version.hip is not None
    and torch.cuda.is_available()
    and str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).startswith("gfx942")
)
pytestmark = pytest.mark.skipif(not IS_GFX942, reason="ROCm gfx942 scan contract")
if IS_GFX942:
    from rl_engine.backends.rocm.sampling.top_p_scan import apply_top_k_top_p, probability_cumsum


@pytest.mark.parametrize("rows,n", [(2, 4096), (4, 152064), (7, 152064), (3, 32001), (5, 65537)])
@pytest.mark.parametrize("kind", ["random", "uniform", "wide"])
def test_cumsum_raw_bits(rows, n, kind):
    torch.manual_seed(n + rows)
    x = torch.randn(rows, n, device="cuda")
    if kind == "uniform":
        x.zero_()
    elif kind == "wide":
        x.mul_(40)
    probabilities = x.sort().values.softmax(-1)
    expected = probabilities.cumsum(-1)
    actual = probability_cumsum(probabilities.clone())
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("temperature", [0.2, 0.7, 1.0, 1.8])
@pytest.mark.parametrize("use_k", [False, True])
def test_nucleus_matches_native_at_boundary(temperature, use_k):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch as native

    torch.manual_seed(512)
    logits = torch.randn(4, 152064, device="cuda") / temperature
    # Include ties and rows retaining nearly the entire vocabulary.
    logits[0].zero_()
    k = torch.tensor([152064, 70000, 512, 17], device="cuda") if use_k else None
    values = logits.sort().values
    if k is not None:
        threshold = values.gather(1, (values.size(1) - k).unsqueeze(1))
        values.masked_fill_(values < threshold, float("-inf"))
    cumulative = values.softmax(-1).cumsum(-1)
    boundary = 1 - cumulative[:, -5]
    p = torch.tensor([1.0, 0.999, 0.95, 0.1], device="cuda")
    for selected_p in [p, boundary, torch.nextafter(boundary, torch.ones_like(boundary))]:
        expected = native(logits.clone(), k, selected_p)
        actual = apply_top_k_top_p(logits.clone(), k, selected_p)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_scan_graph_replay_uses_current_probabilities():
    x = torch.randn(4, 32001, device="cuda").softmax(-1)
    source = x.clone()
    probability_cumsum(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.copy_(source)
        actual = probability_cumsum(x)
    for scale in [0.2, 1.0, 5.0]:
        source.copy_((torch.randn_like(source) * scale).softmax(-1))
        graph.replay()
        assert torch.equal(actual.view(torch.int32), source.cumsum(-1).view(torch.int32))


@pytest.mark.parametrize(
    "rows,n,dtype,has_p,accelerated",
    [
        (1, 152064, torch.float32, True, False),
        (4, 1024, torch.float32, True, False),
        (4, 32000, torch.float16, True, False),
        (4, 32000, torch.float32, False, False),
        (8, 32000, torch.float32, True, False),
        (2, 32000, torch.float32, True, True),
        (7, 32000, torch.float32, True, True),
    ],
)
def test_sampler_hook_preserves_native_fallbacks(monkeypatch, rows, n, dtype, has_p, accelerated):
    from types import SimpleNamespace

    from vllm.v1.sample.ops import topk_topp_sampler as native

    from rl_engine.backends.rocm.sampling import top_p_scan
    from rl_engine.integrations.engines.rollout.vllm.runtime import _patch_rocm_top_p_scan

    original = native.apply_top_k_top_p_pytorch
    calls = []

    def recorded(*args):
        calls.append(True)
        return apply_top_k_top_p(*args)

    monkeypatch.setattr(top_p_scan, "apply_top_k_top_p", recorded)
    monkeypatch.setattr(native, "apply_top_k_top_p_pytorch", original)
    integration = SimpleNamespace(record_installed_hook=lambda *args: None)
    _patch_rocm_top_p_scan(integration)
    patched = native.apply_top_k_top_p_pytorch
    _patch_rocm_top_p_scan(integration)
    assert native.apply_top_k_top_p_pytorch is patched
    x = torch.randn(rows, n, device="cuda", dtype=dtype)
    p = torch.full((rows,), 0.95, device="cuda") if has_p else None
    actual = patched(x.clone(), None, p)
    expected = original(x.clone(), None, p)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(bits), expected.view(bits))
    assert bool(calls) == accelerated
