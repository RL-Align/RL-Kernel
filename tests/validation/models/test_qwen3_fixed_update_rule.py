"""Fixed-rollout update rule: an update over its bound passes only if AdamW explains it."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tools.validation.models.qwen3_ppo.compare_fixed import (
    assess,
    checkpoint_weight,
    explained_update,
)
from tools.validation.models.qwen3_ppo.fixed_contract import CONFIG, OPTIMIZER, write_tensor

NAME = "model.layers.0.post_attention_layernorm.weight"
NORMS = (76.9235, 74.1)


def _run(root, p0, grads, *, norms=NORMS, clip=True, **optimizer):
    """Train ``p0`` with torch AdamW on ``grads`` (one per step) and record evidence like a run."""
    root.mkdir(parents=True)
    weight = torch.nn.Parameter(p0.clone())
    opt = torch.optim.AdamW([weight], **{**OPTIMIZER, **optimizer}, foreach=False, fused=False)
    artifacts = {}
    for t, (g, norm) in enumerate(zip(grads, norms, strict=True)):
        artifacts[f"{t}/gradient/{NAME}"] = write_tensor(root, f"{t}/gradient/{NAME}", g)
        artifacts[f"{t}/grad_norm"] = write_tensor(
            root, f"{t}/grad_norm", torch.tensor(norm, dtype=torch.float64)
        )
        scale = min(1.0, CONFIG.max_grad_norm / (norm + 1e-6)) if clip else 1.0
        before = weight.detach().clone()
        weight.grad = g * scale
        opt.step()
        artifacts[f"{t}/update/{NAME}"] = write_tensor(
            root, f"{t}/update/{NAME}", weight.detach() - before
        )
    return dict(artifacts=artifacts)


def _setup(tmp_path, low=0.25, high=1.0, n=512):
    torch.manual_seed(0)
    p0 = torch.empty(n).uniform_(low, high)
    grads = [torch.randn(n) * 1e-3, torch.randn(n) * 1e-3]
    grads[0][7] = 2e-8  # clipped gradient far below eps: AdamW's step is linear in g here
    grads[1][9] = -1e-8
    return p0, grads


def _perturbed(grads, delta=6e-8):
    return [g + delta for g in grads]


def _explain(tmp_path, p0, left, right, step):
    return explained_update(
        f"{step}/update/{NAME}", p0.double(), tmp_path / "left", left, tmp_path / "right", right
    )


@pytest.mark.parametrize("step", [0, 1])
def test_eps_regime_difference_is_explained_on_both_steps(tmp_path, step):
    p0, grads = _setup(tmp_path)
    left = _run(tmp_path / "left", p0, grads)
    right = _run(tmp_path / "right", p0, _perturbed(grads))
    check = _explain(tmp_path, p0, left, right, step)
    assert check["passed"], check
    assert check["max_residual_ulps"] <= 2


def test_failing_round2_element_size_is_reproduced(tmp_path):
    """A 6e-8 gradient difference at a near-zero element moves its update by about 12 units."""
    p0, grads = _setup(tmp_path, low=0.5, high=1.0)
    left = _run(tmp_path / "left", p0, grads)
    right = _run(tmp_path / "right", p0, _perturbed(grads))
    check = _explain(tmp_path, p0, left, right, 0)
    u = torch.load(tmp_path / "right" / right["artifacts"][f"0/update/{NAME}"]["path"])
    v = torch.load(tmp_path / "left" / left["artifacts"][f"0/update/{NAME}"]["path"])
    assert abs(float(u[7] - v[7])) > 6e-7
    assert check["passed"] and check["max_residual_ulps"] <= 2


@pytest.mark.parametrize("low,high", [(2.0, 4.0), (4.0, 8.0), (16.0, 32.0)])
def test_large_weights_round_within_two_ulps(tmp_path, low, high):
    p0, grads = _setup(tmp_path, low=low, high=high)
    left = _run(tmp_path / "left", p0, grads)
    right = _run(tmp_path / "right", p0, _perturbed(grads))
    for step in (0, 1):
        assert _explain(tmp_path, p0, left, right, step)["passed"]


@pytest.mark.parametrize(
    "change", [dict(eps=1e-6), dict(lr=1.01e-5), dict(betas=(0.8, 0.999)), "clip"]
)
def test_a_different_optimizer_step_is_not_explained(tmp_path, change):
    """Weight decay is not among the cases: lr * weight_decay = 1e-7 is about one FP32 ulp of
    the weight per step, below what any elementwise FP32 update comparison resolves (the frozen
    bounds' 1e-7 floor does not see it either)."""
    p0, grads = _setup(tmp_path)
    left = _run(tmp_path / "left", p0, grads)
    if change == "clip":
        right = _run(tmp_path / "right", p0, grads, clip=False)
    else:
        right = _run(tmp_path / "right", p0, grads, **change)
    # betas only change the second step; everything else shows at the first.
    step = 1 if "betas" in change else 0
    assert not _explain(tmp_path, p0, left, right, step)["passed"]


def test_a_one_ulp_wrong_weight_is_not_explained(tmp_path):
    """An update off by more than rounding at one element fails even with equal gradients."""
    p0, grads = _setup(tmp_path, low=0.5, high=1.0)
    left = _run(tmp_path / "left", p0, grads)
    right = _run(tmp_path / "right", p0, grads)
    path = tmp_path / "right" / right["artifacts"][f"0/update/{NAME}"]["path"]
    u = torch.load(path)
    u[3] += 3 * 2**-24
    path.unlink()
    right["artifacts"][f"0/update/{NAME}"] = write_tensor(tmp_path / "right", f"0/update/{NAME}", u)
    check = _explain(tmp_path, p0, left, right, 0)
    assert not check["passed"] and check["worst_element"] == 3


def test_assess_only_lets_explained_updates_through():
    rows = {
        "0/update/a": dict(max_abs=3.0),
        "0/update/b": dict(max_abs=3.0),
        "0/update/c": dict(max_abs=3.0),
        "0/gradient/a": dict(max_abs=3.0),
        "0/update/d": dict(max_abs=0.5),
    }
    bounds = {name: dict(max_abs=1.0) for name in rows}
    verdict = {"0/update/a": dict(passed=True), "0/update/b": dict(passed=False)}
    failures, explained = assess(rows, bounds, verdict.get)
    assert failures == ["0/update/b", "0/update/c", "0/gradient/a"]
    assert set(explained) == {"0/update/a", "0/update/b"}
    assert explained["0/update/a"]["bound"] == 1.0


def test_checkpoint_weight_reads_the_index_and_skips_missing_names(tmp_path):
    w = torch.arange(4, dtype=torch.bfloat16)
    save_file({NAME: w}, str(tmp_path / "model-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map={NAME: "model-00001.safetensors"}))
    )
    assert torch.equal(checkpoint_weight(tmp_path, NAME), w.double())
    assert checkpoint_weight(tmp_path, "score.weight") is None
