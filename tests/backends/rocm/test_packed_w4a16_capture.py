# SPDX-License-Identifier: Apache-2.0
"""Original active FP4 expert against the actual ROCm W4A16 rollout capture."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rl_engine.backends.rocm.moe.packed_w4a16 import packed_w4a16

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


@pytest.mark.parametrize("projection", ("gate_up", "down"))
def test_actual_first_expert_w4a16_projection_and_input_vjp(tmp_path, projection):
    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and actual routed matmul trace")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"]) / "generation-trace"
    manifest = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]

    def load(name, occurrence=0):
        e = [e for e in manifest if e["name"] == name and e["layer"] == 0 and e["forward"] == 0][
            occurrence
        ]
        return torch.load(root / e["file"], weights_only=True, map_location="cpu")

    _, ids = load("layer.0.ffn.router.select.output")
    order = ids.flatten().argsort(stable=True)
    sorted_experts = ids.flatten()[order]
    expert = int(sorted_experts[0])
    mask = sorted_experts == expert
    stage = 0 if projection == "gate_up" else 1
    payload = load("layer.0.ffn.packed.matmul.input", stage)
    operands = payload["x"][(order // 6)[mask]] if stage == 0 else payload["x"][mask]
    x = operands.cuda().contiguous().requires_grad_()
    gamma = None if stage == 0 else payload["gammas"][mask].cuda().requires_grad_()
    expected = load("layer.0.ffn.packed.matmul.output", stage)[mask]
    with safe_open(
        str(next(Path(os.environ["RLK_DSV4_CHECKPOINT"]).glob("*.safetensors"))), framework="pt"
    ) as source:
        stem = f"layers.0.ffn.experts.{expert}"
        codes = torch.cat(
            [
                source.get_tensor(stem + "." + n + ".weight").view(torch.uint8)
                for n in (("w1", "w3") if stage == 0 else ("w2",))
            ],
            0,
        )[None].cuda()
        scales = torch.cat(
            [
                source.get_tensor(stem + "." + n + ".scale").view(torch.uint8)
                for n in (("w1", "w3") if stage == 0 else ("w2",))
            ],
            0,
        )[None].cuda()
    offsets = torch.tensor([0, x.shape[0]], device="cuda", dtype=torch.int64)
    actual = packed_w4a16(x, codes, scales, offsets, gamma)
    bad = actual.detach().cpu().view(torch.int16) != expected.view(torch.int16)
    record = dict(
        expert=expert,
        projection=projection,
        rows=x.shape[0],
        elements=actual.numel(),
        mismatch_count=int(bad.sum()),
    )
    print("ORIGINAL_W4A16_PROJECTION", record, flush=True)
    torch.save(dict(actual=actual.detach().cpu(), expected=expected), tmp_path / "projection.pt")
    assert not bool(bad.any())
    actual.float().square().mean().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.ne(0).any())
