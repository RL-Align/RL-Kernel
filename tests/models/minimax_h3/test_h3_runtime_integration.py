# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 integration rows: runtime modes, ablation overlay, Diffusers adapter, training
provider, LoRA gradient gate and the CPS transition contract."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

from rl_engine.models.minimax_h3 import runtime as rt
from rl_engine.runtime.policy import RLKernelMode

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --- runtime modes and overlay --------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "bind", "fail", "report", "case"),
    [
        ("strict", True, True, False, "R/R"),
        ("audit", True, False, True, "R/R"),
        ("auto", False, False, False, "P/P"),
        ("off", False, False, False, "P/P"),
    ],
)
def test_runtime_modes(mode, bind, fail, report, case):
    p = rt.h3_runtime_policy(mode)
    assert (p.bind, p.fail_on_fallback, p.report_only, p.case_id) == (bind, fail, report, case)


def test_runtime_mode_reads_environment(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_MODE", "off")
    assert rt.h3_runtime_policy().mode is RLKernelMode.OFF
    monkeypatch.setenv("RL_KERNEL_MODE", "bogus")
    with pytest.raises(RuntimeError, match="RL_KERNEL_MODE"):
        rt.h3_runtime_policy()


def test_overlay_parse_label_and_validation():
    """Overlay strings parse to node and boundary sets, label the run, and reject unknown names."""
    assert rt.AblationOverlay.parse("").label == "R/R"
    o = rt.AblationOverlay.parse("node:attention, boundary:token_refiner")
    assert o.native_nodes == {"attention"} and o.native_boundaries == {"token_refiner"}
    assert o.label == "ablation(native: node:attention,boundary:token_refiner)"
    for bad in ("node:nope", "boundary:time_proj", "layer:x"):
        with pytest.raises(ValueError):
            rt.AblationOverlay.parse(bad)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))


def test_adapter_off_binds_nothing_and_reports_provider():
    adapter = rt.H3DiffusersAdapter(_FakeModel(), "off")
    assert adapter._bind() is False
    rep = adapter.report()
    assert (rep["bound"], rep["label"], rep["case_id"]) == (False, "P/P", "P/P")


def test_adapter_strict_refuses_cpu_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        rt.H3DiffusersAdapter(_FakeModel(), "strict")._bind()


def test_adapter_audit_reports_the_fallback():
    adapter = rt.H3DiffusersAdapter(_FakeModel(), "audit")
    assert adapter._bind() is False and adapter.fell_back == "no CUDA device"
    assert adapter.report()["label"] == "P/P"


def _bindable(monkeypatch, mode, *, binding_fails):
    """An adapter whose model counts as CUDA-resident, with a stand-in for the full-model
    binding that records each ``candidate`` request and can refuse the candidate graph."""

    from contextlib import contextmanager

    from rl_engine.validation.models import h3_full_model

    entered = []

    @contextmanager
    def execution(model, *, candidate, trace, **_):
        """Stand-in for ``full_model_execution``: fails to bind the candidate when asked to."""
        if candidate and binding_fails:
            raise ValueError("candidate projection fell back from its CUDA implementation")
        entered.append(candidate)
        yield trace

    monkeypatch.setattr(h3_full_model, "full_model_execution", execution)
    adapter = rt.H3DiffusersAdapter(_FakeModel(), mode)
    monkeypatch.setattr(adapter, "_bind", lambda: True)
    return adapter, entered


@pytest.mark.parametrize("mode", ["strict", "audit"])
def test_an_error_inside_the_bound_step_is_not_a_fallback(monkeypatch, mode):
    """An error raised inside a bound step propagates and is not recorded as a fallback."""
    adapter, entered = _bindable(monkeypatch, mode, binding_fails=False)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        with adapter.active():
            raise RuntimeError("CUDA out of memory (raised by the caller's backward)")
    assert entered == [True] and adapter.fell_back is None
    assert adapter.report()["case_id"] == "R/R"


def test_audit_runs_the_provider_when_the_graph_cannot_bind(monkeypatch):
    """Audit mode runs the provider and records the fallback when the graph cannot bind."""
    adapter, entered = _bindable(monkeypatch, "audit", binding_fails=True)
    with adapter.active():
        ran = True
    assert ran and entered == [False]
    assert "fell back from its CUDA implementation" in adapter.fell_back
    rep = adapter.report()
    assert (rep["bound"], rep["case_id"], rep["label"]) == (False, "P/P", "P/P")


def test_strict_raises_when_the_graph_cannot_bind(monkeypatch):
    """Strict mode raises when the graph cannot bind and records no fallback."""
    adapter, entered = _bindable(monkeypatch, "strict", binding_fails=True)
    with pytest.raises(ValueError, match="fell back"):
        with adapter.active():
            pass
    assert entered == [] and adapter.fell_back is None


# --- CPS transition -----------------------------------------------------------------------


def _scheduler_cps(x, v, sigma, sigma_prev, noise, level):
    sin_noise = math.sin(level * math.pi / 2)
    std = sigma_prev * sin_noise
    pred = x - sigma * v
    est = x + v * (1 - sigma)
    mean = pred * (1 - sigma_prev) + est * torch.sqrt(sigma_prev**2 - std**2)
    prev = mean + std * noise
    return prev, mean, std, (-((prev - mean) ** 2)).mean(dim=tuple(range(1, prev.ndim)))


@needs_gpu
class TestTransition:
    def _case(self, batch=3, shape=(257, 96)):
        from rl_engine.models.minimax_h3 import transition as tr

        g = torch.Generator().manual_seed(1)
        x = torch.randn(batch, *shape, generator=g).cuda()
        v = torch.randn(batch, *shape, generator=g).cuda()
        noise = torch.stack([tr.step_noise(7, 2, i, "video", shape, "cuda") for i in range(batch)])
        return tr, x, v, noise

    def test_matches_the_scheduler_formula(self):
        pytest.importorskip("triton")
        tr, x, v, noise = self._case()
        sigma = torch.tensor(0.75, device="cuda")
        sigma_prev = torch.tensor(0.5, device="cuda")
        out = tr.cps_transition(x, v, sigma, sigma_prev, noise=noise)
        prev, mean, std, logp = _scheduler_cps(x, v, sigma, sigma_prev, noise, tr.NOISE_LEVEL)
        assert torch.equal(out.prev_sample, prev) and torch.equal(out.prev_sample_mean, mean)
        assert torch.equal(out.std_dev_t, std)
        assert torch.allclose(out.log_prob, logp, rtol=1e-4, atol=1e-12)

    def test_targets_cps_at_noise_level_08(self):
        from rl_engine.models.minimax_h3 import transition as tr

        assert (tr.SDE_TYPE, tr.NOISE_LEVEL) == ("cps", 0.8)

    def test_each_sample_is_independent_of_the_batch(self):
        pytest.importorskip("triton")
        tr, x, v, noise = self._case()
        whole = tr.cps_transition(x, v, 0.75, 0.5, noise=noise)
        for i in range(x.shape[0]):
            one = tr.cps_transition(x[i : i + 1], v[i : i + 1], 0.75, 0.5, noise=noise[i : i + 1])
            assert torch.equal(one.prev_sample, whole.prev_sample[i : i + 1])
            assert torch.equal(one.log_prob, whole.log_prob[i : i + 1])

    def test_noise_depends_on_sample_id_only(self):
        from rl_engine.models.minimax_h3 import transition as tr

        a = tr.step_noise(1, 3, 5, "video", (4, 8), "cuda")
        assert torch.equal(a, tr.step_noise(1, 3, 5, "video", (4, 8), "cpu").cuda())
        for other in (
            (2, 3, 5, "video"),
            (1, 4, 5, "video"),
            (1, 3, 6, "video"),
            (1, 3, 5, "audio"),
        ):
            assert not torch.equal(a, tr.step_noise(*other, (4, 8), "cuda"))

    def test_scoring_a_stored_rollout_reproduces_sampling_logprob(self):
        pytest.importorskip("triton")
        tr, x, v, noise = self._case()
        sampled = tr.cps_transition(x, v, 0.75, 0.5, noise=noise)
        scored = tr.cps_transition(x, v, 0.75, 0.5, prev_sample=sampled.prev_sample)
        assert torch.equal(scored.log_prob, sampled.log_prob)

    def test_zero_noise_level_is_the_mean_with_zero_logprob(self):
        pytest.importorskip("triton")
        tr, x, v, noise = self._case()
        out = tr.cps_transition(x, v, 0.75, 0.5, noise_level=0.0, noise=noise)
        assert torch.equal(out.prev_sample, out.prev_sample_mean)
        assert torch.count_nonzero(out.log_prob) == 0

    def test_gradient_reaches_the_velocity(self):
        pytest.importorskip("triton")
        tr, x, v, noise = self._case()
        v.requires_grad_(True)
        out = tr.cps_transition(x, v, 0.75, 0.5, prev_sample=(x + 0.1).detach())
        out.prev_sample_mean.sum().backward()
        assert v.grad is not None and torch.isfinite(v.grad).all()

    def test_log_prob_carries_the_policy_gradient(self):
        """The policy loss differentiates the scored log-prob: the gradient must reach the
        velocity, equal the formula's ``torch.mean`` autograd and not depend on the batch."""

        pytest.importorskip("triton")
        tr, x, v, noise = self._case()
        stored = tr.cps_transition(x, v, 0.75, 0.5, noise=noise).prev_sample
        weights = torch.tensor([0.5, -1.0, 2.0], device="cuda")  # per-sample advantages

        def grad_of(fn, rows):
            leaf = v[rows].clone().requires_grad_(True)
            (fn(x[rows], leaf, stored[rows]) * weights[rows]).sum().backward()
            return leaf.grad

        def scored(xs, vs, prev):
            """The differentiable CPS log-prob of ``prev`` from ``xs`` under velocities ``vs``."""
            out = tr.cps_transition(xs, vs, 0.75, 0.5, prev_sample=prev)
            assert out.log_prob.requires_grad
            return out.log_prob

        def formula(xs, vs, prev):  # the same log-prob with torch.mean, autograd end to end
            sigma, sigma_prev = torch.tensor(0.75).cuda(), torch.tensor(0.5).cuda()
            mean = _scheduler_cps(xs, vs, sigma, sigma_prev, torch.zeros_like(xs), 0.8)[1]
            return (-((prev - mean) ** 2)).mean(dim=tuple(range(1, prev.ndim)))

        batch = grad_of(scored, slice(None))
        assert torch.isfinite(batch).all() and batch.abs().sum() > 0
        assert torch.allclose(batch, grad_of(formula, slice(None)), rtol=1e-6, atol=0)
        for i in range(3):  # each sample's gradient is the bits it gets on its own
            assert torch.equal(batch[i : i + 1], grad_of(scored, slice(i, i + 1)))

    def test_matches_verl_omni_scheduler_when_available(self):
        root = os.environ.get("VERL_OMNI_PATH")
        if not root:
            pytest.skip("set VERL_OMNI_PATH to a verl-omni checkout")
        # load by path: putting the checkout on sys.path would shadow our own `tests` package
        path = Path(root) / "verl_omni/pipelines/schedulers/flow_match_sde.py"
        if not path.exists():
            pytest.skip(f"no scheduler at {path}")
        spec = importlib.util.spec_from_file_location("flow_match_sde_under_test", path)
        sched_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sched_mod)
        pytest.importorskip("triton")
        tr, x, v, noise = self._case(batch=2, shape=(64, 96))
        scheduler = sched_mod.FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        scheduler.sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        scheduler.timesteps = scheduler.sigmas[:-1] * 1000
        scheduler._step_index = 1
        # the scheduler draws its own noise: score the sampled rollout instead
        ours = tr.cps_transition(x, v, 0.75, 0.5, noise=noise)
        ref = scheduler.sample_previous_step(
            sample=x.cpu(),
            model_output=v.cpu(),
            prev_sample=ours.prev_sample.cpu(),
            noise_level=tr.NOISE_LEVEL,
            sde_type=tr.SDE_TYPE,
        )
        assert torch.equal(ref[2], ours.prev_sample_mean.cpu())
        assert torch.allclose(ref[1], ours.log_prob.cpu(), rtol=1e-4, atol=1e-12)


# --- LoRA gradient gate and vLLM-Omni adapter -------------------------------------------------


@needs_gpu
def test_lora_gradient_gate():
    pytest.importorskip("triton")
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(512, 384, generator=g) / 20).bfloat16().cuda()
    x = torch.randn(70, 384, generator=g).bfloat16().cuda()
    grad = torch.randn(70, 512, generator=g).bfloat16().cuda()
    report = rt.lora_gradient_gate(w, x, grad)
    assert report["repeatable"] and report["forward_rows_independent"] and report["base_frozen"]
    assert report["forward_rel_err_vs_fp64"] < 5e-3


@needs_gpu
def test_lora_gate_detects_a_gradient_reaching_the_base(monkeypatch):
    """``base_frozen`` is a real check: a backward that returns a base gradient fails it."""

    pytest.importorskip("triton")
    backward = rt._StrictLoRA.backward

    def leaky(ctx, grad):
        dx, _, d_a, d_b, _ = backward(ctx, grad)
        return dx, torch.zeros_like(ctx.saved_tensors[1]), d_a, d_b, None

    monkeypatch.setattr(rt._StrictLoRA, "backward", staticmethod(leaky))
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(64, 32, generator=g) / 20).bfloat16().cuda()
    x = torch.randn(8, 32, generator=g).bfloat16().cuda()
    grad = torch.randn(8, 64, generator=g).bfloat16().cuda()
    assert rt.lora_gradient_gate(w, x, grad)["base_frozen"] is False


def test_transition_refuses_sample_ids_that_do_not_match_the_batch():
    """``transition()`` refuses ``sample_ids`` whose length differs from the batch."""
    provider = rt.H3TrainingProvider(_FakeModel(), "off")
    x = torch.zeros(3, 4, 2)
    with pytest.raises(ValueError, match="sample_ids has 1 entries for a batch of 3"):
        provider.transition(x, x, 0.75, 0.5, seed=1, step=0, sample_ids=[0], modality="video")


@needs_gpu
def test_lora_gradients_match_fp64_autograd():
    pytest.importorskip("triton")
    g = torch.Generator().manual_seed(4)
    n, k, r, t = 256, 192, 4, 33
    w = (torch.randn(n, k, generator=g) / 14).bfloat16().cuda()
    x = torch.randn(t, k, generator=g).bfloat16().cuda().requires_grad_()
    a = (torch.randn(r, k, generator=g) / 14).cuda().requires_grad_()
    b = (torch.randn(n, r, generator=g) * 0.1).cuda().requires_grad_()
    grad = torch.randn(t, n, generator=g).bfloat16().cuda()
    rt.strict_lora_linear(x, w, a, b, 2.0).backward(grad)
    xd, ad, bd = (t_.detach().double().requires_grad_() for t_ in (x, a, b))
    ((xd @ w.double().t() + 2.0 * (xd @ ad.t()) @ bd.t()) * grad.double()).sum().backward()
    for got, ref in ((a.grad, ad.grad), (b.grad, bd.grad), (x.grad, xd.grad)):
        assert (got.double() - ref).norm() / ref.norm() < 5e-3
