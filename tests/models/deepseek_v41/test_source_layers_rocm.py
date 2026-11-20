# SPDX-License-Identifier: Apache-2.0
"""Real-weight isolated later-layer mechanisms; not a 40-layer model test."""

import json
import os
from pathlib import Path

import pytest
import torch
from rl_engine.backends.rocm.engram.ngram_hash import EngramHash
from rl_engine.models.deepseek_v41.attention import Attention
from rl_engine.models.deepseek_v41.bitwise import compare_bits
from rl_engine.models.deepseek_v41.checkpoint import LayerCheckpoint
from rl_engine.models.deepseek_v41.config import DeepSeekV41Config
from rl_engine.models.deepseek_v41.loading import engram
from safetensors.torch import load_file


@pytest.fixture(scope="module")
def context():
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("real ROCm required")
        pytest.skip("ROCm required")
    names = (
        "DSV41_MODEL_WEIGHTS",
        "DSV41_ATTENTION_WEIGHTS",
        "DSV41_BASELINE",
        "DSV41_TARGET_RESULTS",
    )
    if any(not os.environ.get(k) for k in names):
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("real source-layer checkpoint paths required")
        pytest.skip("real source-layer weights not configured")
    root, extra, baseline, result = (Path(os.environ[k]) for k in names)
    config = DeepSeekV41Config.from_json(root / "config.json")
    data = torch.load(baseline, map_location="cuda", weights_only=True)
    rows = []

    def check(category, actual, expected):
        assert actual.keys() == expected.keys()
        for name in actual:
            r = compare_bits(actual[name], expected[name], operator=name)
            r["category"] = category
            rows.append(r)
            assert r["status"] == "PASS", r

    yield config, LayerCheckpoint(root), LayerCheckpoint(extra), data, check
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "scope": (
                    "real-weight isolated Engram14 and later attention mechanisms; "
                    "upstream activations from real four-layer baseline"
                ),
                "comparisons": rows,
                "compared_elements": sum(r["compared_elements"] for r in rows),
                "mismatched_elements": sum(r["mismatched_elements"] for r in rows),
            },
            indent=2,
        )
        + "\n"
    )


def test_real_engram14_forward_backward_and_mode_parity(context):
    config, checkpoint, _, baseline, check = context
    mod = engram(checkpoint, config, 14, torch.device("cuda", torch.cuda.current_device()))
    token_map = load_file(str(checkpoint.directory / "token-map.safetensors"), device="cuda")[
        "token_map"
    ]
    hasher = EngramHash(config, token_map, max_batch_size=1, max_seq_len=32)
    hashes = hasher(baseline["input_ids"])[:, :, 1]
    x = baseline["trace"]["layers.1.hidden"].detach().clone().requires_grad_()
    values = []
    for _ in range(2):
        y = mod(x, hashes)
        grads = torch.autograd.grad(y.float().square().mean(), (x, mod.q_weight, mod.k_weight))
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        values.append(dict(output=y, dx=grads[0], dq=grads[1], dk=grads[2]))
    check("real_engram14_replay", values[0], values[1])
    with torch.inference_mode():
        check(
            "real_engram14_shared_kernel_modes",
            {"output": mod(x, hashes)},
            {"output": values[0]["output"]},
        )


def test_real_kv_sources_and_later_index_sources_forward_backward(context):
    config, _, checkpoint, baseline, check = context
    layers = (8, 14, 20, 24, 28, 32, 36)
    modules = {
        i: Attention(config, checkpoint, i, torch.device("cuda", torch.cuda.current_device()))
        for i in layers
    }
    data = baseline["trace"]["layers.2.attention.input"]

    def run(training):
        for mod in modules.values():
            mod.reset()
            mod.zero_grad(set_to_none=True)
        state = {}
        traces = {}
        outputs = []
        for i in layers:
            if i in (8, 14, 20):
                state = {}
            x = data.detach().clone().requires_grad_(training)

            def trace(name, value):
                traces[name] = value.detach().clone()

            y = modules[i](x, 0, state, trace)
            traces[f"layer.{i}.output"] = y.detach().clone()
            for name, (owner, value) in state.items():
                traces[f"layer.{i}.state.{name}"] = value.detach().clone()
                if name == "kv":
                    assert owner == (i if i in (8, 14, 20) else 20)
            outputs.append((x, y))
        if training:
            sum(y.float().square().mean() for _, y in outputs).backward()
            for i, (x, _) in zip(layers, outputs, strict=True):
                assert x.grad is not None and bool(torch.isfinite(x.grad).all())
                traces[f"layer.{i}.dx"] = x.grad.detach().clone()
            for i, mod in modules.items():
                for name, p in mod.named_parameters():
                    if p.grad is not None:
                        assert bool(torch.isfinite(p.grad).all())
                        traces[f"layer.{i}.grad.{name}"] = p.grad.detach().clone()
        return traces

    first = run(True)
    second = run(True)
    check("real_later_attention_replay", first, second)
    with torch.inference_mode():
        inference = run(False)
    check("real_later_attention_shared_kernel_modes", inference, {k: first[k] for k in inference})
