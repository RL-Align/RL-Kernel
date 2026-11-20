# SPDX-License-Identifier: Apache-2.0
"""Opt-in real checkpoint component tests; never labeled as four-layer execution."""

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from rl_engine.models.deepseek_v41.checkpoint import LayerCheckpoint
from rl_engine.models.deepseek_v41.expert import V41Expert, V41FP4Linear
from rl_engine.models.deepseek_v41.reference import dequantize_fp4, quantized_linear_reference
from rl_engine.models.deepseek_v41.bitwise import compare_bits


@pytest.fixture(scope="module")
def checkpoint():
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("ROCm GPU required")
        pytest.skip("ROCm GPU required")
    path = os.environ.get("DSV41_OPERATOR_WEIGHTS")
    if not path:
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("DSV41_OPERATOR_WEIGHTS is required for official-weight component tests")
        pytest.skip("official operator-weight slice not configured")
    return LayerCheckpoint(path)


def test_official_fp4_expert_forward_input_and_route_gradients(checkpoint):
    layers = []
    dense = []
    for name in ("w1", "w2", "w3"):
        w, s = checkpoint.projection("layers.2.ffn.experts.0." + name, device="cuda")
        layers.append(V41FP4Linear(w, s))
        dense.append(dequantize_fp4(w, s))
    expert = V41Expert(*layers)
    torch.manual_seed(42)
    x = (torch.randn(3, 5120, device="cuda") * 0.1).bfloat16().requires_grad_()
    p = torch.tensor([0.01, 0.2, 0.9], device="cuda", requires_grad=True)
    dy = torch.randn_like(x)

    def reference(x, p):
        g = quantized_linear_reference(x, dense[0]).float().clamp_max(10)
        u = quantized_linear_reference(x, dense[2]).float().clamp(-10, 10)
        h = (torch.nn.functional.silu(g) * u * p[:, None]).bfloat16()
        return quantized_linear_reference(h, dense[1])

    y = expert(x, p)
    dx, dp = torch.autograd.grad(y, (x, p), dy)
    rx, rp = x.detach().clone().requires_grad_(), p.detach().clone().requires_grad_()
    ry = reference(rx, rp)
    rdx, rdp = torch.autograd.grad(ry, (rx, rp), dy)
    torch.testing.assert_close(y, ry, rtol=0.03, atol=0.03)
    torch.testing.assert_close(dx, rdx, rtol=0.05, atol=0.08)
    torch.testing.assert_close(dp, rdp, rtol=0.05, atol=0.08)
    x2, p2 = x.detach().clone().requires_grad_(), p.detach().clone().requires_grad_()
    y2 = expert(x2, p2)
    grads2 = torch.autograd.grad(y2, (x2, p2), dy)
    for a, b in zip((y, dx, dp), (y2, *grads2), strict=True):
        assert compare_bits(a, b)["mismatched_elements"] == 0
    with torch.inference_mode():
        assert compare_bits(y, expert(x.detach(), p.detach()))["mismatched_elements"] == 0


def test_four_real_engram_rows_dequantized_without_missing_row_substitution(checkpoint):
    from rl_engine.backends.rocm.engram.embedding import ResidentEngramRows

    root = checkpoint.directory
    data = load_file(str(root / "engram-rows.safetensors"), device="cuda")
    ids = data["original_row_ids"]
    w, s = data["layers.1.engram.embed.weight"], data["layers.1.engram.embed.scale"]
    mod = ResidentEngramRows(ids, w, s)
    request = ids[torch.tensor([[3, 0, 1], [2, 0, 3]], device="cuda")]
    expected = (w.float() * s.float().repeat_interleave(32, -1)).bfloat16()
    actual = mod(request)
    torch.testing.assert_close(actual, expected[torch.searchsorted(ids, request)], rtol=0, atol=0)
    assert compare_bits(actual, mod(request))["status"] == "PASS"
    with pytest.raises(KeyError):
        mod(torch.tensor([ids[1].item() + 1], device="cuda"))


def test_official_token_map_prefill_decode_and_layers_1_14(checkpoint):
    from rl_engine.backends.rocm.engram.ngram_hash import EngramHash
    from rl_engine.models.deepseek_v41.config import DeepSeekV41Config

    if not os.environ.get("DSV41_TOKENIZER"):
        pytest.fail("DSV41_TOKENIZER is required for official token map validation")
    root = Path(os.environ["DSV41_TOKENIZER"])
    mapping = load_file(str(root / "token-map.safetensors"), device="cuda")["token_map"]
    config = DeepSeekV41Config.from_json(Path(__file__).parent / "fixtures/config.json")
    mod = EngramHash(config, mapping, max_batch_size=2, max_seq_len=32)
    ids = torch.tensor([[0, 100, 200, 300, 1], [0, 1, 129279, 1000, 2]], device="cuda")
    full = mod(ids)
    pieces = [mod(ids[:, :2])]
    for pos in range(2, 5):
        pieces.append(mod(ids[:, pos : pos + 1], pos))
    assert full.shape == (2, 5, 2, 24)
    assert compare_bits(full, torch.cat(pieces, 1))["status"] == "PASS"
    for layer, rows in enumerate(config.text["engram_num_embeddings"]):
        assert bool((full[:, :, layer] >= 0).all() & (full[:, :, layer] < rows).all())


def test_official_c2_compressor_forward_backward_partial_state(checkpoint):
    from rl_engine.models.deepseek_v41.compressor import Compressor
    from rl_engine.models.deepseek_v41.config import DeepSeekV41Config
    from rl_engine.models.deepseek_v41.reference import compressor_pool_reference

    config = DeepSeekV41Config.from_json(Path(__file__).parent / "fixtures/config.json")
    prefix = "layers.2.attn.compressor."
    w = checkpoint.tensor(prefix + "wkv.weight", device="cuda")
    g = checkpoint.tensor(prefix + "wgate.weight", device="cuda")
    n = checkpoint.tensor(prefix + "norm.weight", device="cuda")
    mod = Compressor(config, 2, w, n, g)
    torch.manual_seed(35)
    x = torch.randn(2, 5, 5120, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = mod(x)
    assert y.shape == (2, 2, 512) and mod.partial_kv.shape == (2, 1, 512)
    dy = torch.randn_like(y)
    gs = torch.autograd.grad(y, (x, *mod.parameters()), dy)
    xr = x.detach().clone().requires_grad_()
    wr, gr, nr = (
        w.float().requires_grad_(),
        g.float().requires_grad_(),
        n.detach().clone().requires_grad_(),
    )
    kv = torch.nn.functional.linear(xr.float(), wr)[:, :4].reshape(2, 2, 2, 512)
    score = torch.nn.functional.linear(xr.float(), gr)[:, :4].reshape(2, 2, 2, 512)
    z = compressor_pool_reference(kv, score).bfloat16().float()
    ref = ((z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-20)) * nr.float()).bfloat16()
    rgs = torch.autograd.grad(ref, (xr, wr, gr, nr), dy)
    torch.testing.assert_close(y, ref, rtol=0.02, atol=0.02)
    for a, b in zip(gs, rgs, strict=True):
        torch.testing.assert_close(a, b, rtol=0.04, atol=0.06)
    mod.reset()
    y2 = mod(x)
    gs2 = torch.autograd.grad(y2, (x, *mod.parameters()), dy)
    for a, b in zip((y, *gs), (y2, *gs2), strict=True):
        assert compare_bits(a, b)["status"] == "PASS"
    with torch.inference_mode():
        assert compare_bits(y, mod(x.detach()))["status"] == "PASS"
        pieces = []
        for pos in range(5):
            piece = mod(x[:, pos : pos + 1], pos)
            if piece is not None:
                pieces.append(piece)
        torch.testing.assert_close(torch.cat(pieces, 1), y, rtol=0.02, atol=0.02)
        mod.reset()
        assert mod.partial_kv is mod.partial_score is None
