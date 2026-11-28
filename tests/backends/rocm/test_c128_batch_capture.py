# SPDX-License-Identifier: Apache-2.0
"""Real 8x16 DAPO failure: C128 B1/B2 and full/prefix cache bytes."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rl_engine.backends.rocm.attention.native_compressor import compressed_kv
from rl_engine.backends.rocm.attention.qkv_preprocess import DSV4QKVPreprocess

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_c128_double_sample_cache_bits_and_gradients():
    if not os.getenv("RLK_DSV4_LONG_C128_TRACES") or not os.getenv(
        "RLK_DSV4_CHECKPOINT"
    ):
        pytest.skip("set the original full 8x16 failure traces and checkpoint")
    from examples.operators.dsv4_4l.trace_store import load_capture
    from vllm.transformers_utils.config import get_config

    root = Path(os.environ["RLK_DSV4_LONG_C128_TRACES"])

    def tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        for item in value:
            result = tensor(item)
            if result is not None:
                return result

    def load(case, name):
        folder = root / f"actor-rank-1-failure-math-{case}"
        entry = next(
            e
            for e in json.loads((folder / "manifest.json").read_text())
            if e["name"] == f"layer.3.attn.core_attention.compressor.{name}"
        )
        assert entry["weight_version"] == 0
        return tensor(load_capture(folder / entry["file"]))

    checkpoint = Path(os.environ["RLK_DSV4_CHECKPOINT"])
    with safe_open(
        str(next(checkpoint.glob("*.safetensors"))), framework="pt", device="cuda"
    ) as f:
        ape = f.get_tensor("layers.3.attn.compressor.ape").requires_grad_()
        weight = f.get_tensor("layers.3.attn.compressor.norm.weight").requires_grad_()
    cache = DSV4QKVPreprocess(
        get_config(str(checkpoint), trust_remote_code=False), 128
    ).cos_sin_cache
    outputs, operands = {}, {}
    for case in ("R2-b1", "R36-R2-b2", "R36-b1", "R36-prefix786"):
        kv = load(case, "linear_wkv.output").cuda().contiguous().requires_grad_()
        score = load(case, "linear_wgate.output").cuda().contiguous().requires_grad_()
        operands[case] = (kv, score)
        outputs[case] = compressed_kv(kv, score, ape, weight, cache, 128)

    # The captured B1/B2 projection operands are exact; the old occupancy
    # splits differed at compressed key24/channel496 before attention row3199.
    for i in (0, 1):
        assert torch.equal(operands["R2-b1"][i][:, 0], operands["R36-R2-b2"][i][:, 1])
    a = outputs["R2-b1"][:, 0].contiguous().view(torch.int16)
    b = outputs["R36-R2-b2"][:, 1].contiguous().view(torch.int16)
    assert torch.equal(a, b)
    n = outputs["R36-prefix786"].shape[0]
    assert torch.equal(
        outputs["R36-b1"][:n, 0].contiguous().view(torch.int16),
        outputs["R36-prefix786"][:, 0].contiguous().view(torch.int16),
    )
    out = outputs["R36-R2-b2"]
    gradient = torch.linspace(-0.5, 0.5, out.numel(), device=out.device).reshape_as(out)
    (out.float() * gradient).sum().backward()
    for value in (*operands["R36-R2-b2"], ape, weight):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all()) and bool(value.grad.ne(0).any())
