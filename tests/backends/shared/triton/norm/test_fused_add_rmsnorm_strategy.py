# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU policy checks. Kernel execution is tested separately on a real GPU."""

import importlib.util
import inspect
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest
import torch


@pytest.fixture
def policy(monkeypatch):
    # Load the actual module under an isolated name. A no-op decorator lets
    # metadata tests run on hosts with no Triton wheels; no kernel is executed.
    triton = types.ModuleType("triton")
    triton.jit = lambda function: function
    language = types.ModuleType("triton.language")
    language.constexpr = object
    reduction = types.ModuleType("rl_engine.backends.shared.triton.reduction")

    def no_kernel(*args, **kwargs):
        raise AssertionError("Policy tests must not execute GPU reductions")

    reduction.sum_tile = no_kernel
    path = (
        Path(__file__).resolve().parents[5]
        / "rl_engine/backends/shared/triton/norm/fused_add_rmsnorm.py"
    )
    spec = importlib.util.spec_from_file_location("_rmsnorm_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "triton", triton)
        imports.setitem(sys.modules, "triton.language", language)
        imports.setitem(sys.modules, reduction.__name__, reduction)
        imports.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "device", [("cuda", "other GPU"), ("rocm", "AMD GPU"), ("xpu", ""), ("musa", "")]
)
def test_unmeasured_devices_use_shared_default(policy, device):
    select = policy.select_rmsnorm_weight_grad_plan
    assert select(_device(policy, *device), torch.bfloat16, 16384, 2688) == select(
        _device(policy, "cuda", "NVIDIA H100 80GB HBM3"), torch.bfloat16, 16384, 2688
    )


def _device(policy, backend, name=""):
    return policy.RMSNormDevice(backend=backend, gpu_name=name)


def test_device_dtype_and_width_override_precedence(policy):
    key = policy.RMSNormWeightGradKey
    device = _device(policy, "cuda", "test GPU")
    strategies = policy.RMSNormWeightGradStrategy
    sequential = policy._default_weight_grad_plan(strategies.SEQUENTIAL)
    tiled = policy._default_weight_grad_plan(strategies.TILED)
    # Broad overrides occur first: specificity must win regardless of registry order.
    for plan, rule_key in (
        (
            sequential,
            key(device=device, dtype=None, n_cols=None, min_rows=0, max_rows=None),
        ),
        (
            tiled,
            key(device=device, dtype=torch.bfloat16, n_cols=None, min_rows=0, max_rows=None),
        ),
        (
            sequential,
            key(device=device, dtype=torch.bfloat16, n_cols=2688, min_rows=100, max_rows=200),
        ),
    ):
        override = policy.RMSNormWeightGradPlan(
            key=rule_key,
            strategy=plan.strategy,
            config=plan.config,
            launch=plan.launch,
        )
        policy._WEIGHT_GRAD_PLANS[plan.strategy] += (override,)
    select = policy.select_rmsnorm_weight_grad_plan
    assert select(device, torch.float32, 65536, 2688).strategy == strategies.SEQUENTIAL
    assert select(device, torch.bfloat16, 150, 129).strategy == strategies.TILED
    assert select(device, torch.bfloat16, 150, 2688).strategy == strategies.SEQUENTIAL
    assert select(device, torch.bfloat16, 201, 2688).strategy == strategies.TILED
    # A device-specific rule must not match another backend or GPU model.
    for other in (_device(policy, "rocm", "test GPU"), _device(policy, "cuda", "other GPU")):
        assert select(other, torch.float32, 65536, 2688).strategy == strategies.SEQUENTIAL


def test_policy_cache_uses_all_metadata_and_is_bounded(policy):
    select = policy.select_rmsnorm_weight_grad_plan
    args = (_device(policy, "cuda", "H100"), torch.bfloat16, 16384, 2688)
    first = select(*args)
    assert select(*args) is first
    assert select.cache_info().hits == 1
    assert select.cache_info().maxsize == 1024
    for changed in [
        (_device(policy, "cuda", "other"), *args[1:]),
        (args[0], torch.float32, *args[2:]),
        (*args[:2], 16383, 2688),
        (*args[:3], 2689),
    ]:
        select(*changed)
    assert select.cache_info().misses == 5


def test_device_key_uses_input_index_and_resolves_current_before_caching(policy, monkeypatch):
    names = []
    current = [0]

    def name(index):
        names.append(index)
        return f"GPU {index}"

    monkeypatch.setattr(torch.cuda, "get_device_name", name)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.version, "hip", None)
    assert policy.rmsnorm_device_key(torch.device("cuda:1")) == _device(policy, "cuda", "GPU 1")
    assert current[0] == 0
    assert policy.rmsnorm_device_key(torch.device("cuda")) == _device(policy, "cuda", "GPU 0")
    current[0] = 1
    assert policy.rmsnorm_device_key(torch.device("cuda")) == _device(policy, "cuda", "GPU 1")
    assert names == [1, 0]
    monkeypatch.setattr(torch.version, "hip", "test")
    assert policy.rmsnorm_device_key(torch.device("cuda:1")) == _device(policy, "rocm", "GPU 1")
    assert names == [1, 0, 1]


@pytest.mark.parametrize("n_rows", [0, 65536])
def test_explicit_options_bypass_device_query_and_query_failures_propagate(
    policy, monkeypatch, n_rows
):
    def unavailable(device):
        raise RuntimeError("device query failed")

    monkeypatch.setattr(policy, "rmsnorm_device_key", unavailable)
    resolve = policy._resolve_weight_grad_plan
    args = (torch.device("cuda:1"), torch.bfloat16, n_rows, 2688)
    for strategy in policy.RMSNormWeightGradStrategy:
        default = policy._default_weight_grad_plan(strategy)
        policy._WEIGHT_GRAD_PLANS[strategy] = tuple(reversed(policy._WEIGHT_GRAD_PLANS[strategy]))
        plan = resolve(*args, strategy)
        assert plan.strategy == strategy
        assert plan is default
    with pytest.raises(RuntimeError, match="device query failed"):
        resolve(*args, None)


def test_policy_ranges_do_not_overlap_within_the_same_metadata_tier(policy):
    keys = [
        plan.key
        for plans in policy._WEIGHT_GRAD_PLANS.values()
        for plan in plans
        if plan.key is not None
    ]
    for i, lhs in enumerate(keys):
        assert lhs.min_rows >= 0
        assert lhs.max_rows is None or lhs.max_rows >= lhs.min_rows
        for rhs in keys[i + 1 :]:
            if (lhs.device, lhs.dtype, lhs.n_cols) != (
                rhs.device,
                rhs.dtype,
                rhs.n_cols,
            ):
                continue
            assert (lhs.max_rows is not None and lhs.max_rows < rhs.min_rows) or (
                rhs.max_rows is not None and rhs.max_rows < lhs.min_rows
            )


def test_resolved_plan_is_saved_per_autograd_call(policy, monkeypatch):
    # These stand-ins only exercise wrapper/ctx routing; they do not validate
    # RMSNorm numerics. GPU tests execute the real kernels at every boundary.
    def forward(x, residual, weight, *, eps):
        updated = x.float() + residual.float()
        return updated * weight, updated, torch.ones(x.numel() // x.shape[-1])

    calls = []

    def backward(
        updated_residual, inverse_rms, weight, grad_y, grad_updated_residual_output, **options
    ):
        calls.append(options)
        return (
            torch.zeros_like(updated_residual, dtype=options["x_dtype"]),
            torch.zeros_like(updated_residual, dtype=options["residual_dtype"]),
            torch.zeros_like(weight),
        )

    monkeypatch.setattr(policy, "_launch_fused_add_rmsnorm_fwd", forward)
    monkeypatch.setattr(policy, "_launch_fused_add_rmsnorm_bwd", backward)
    monkeypatch.setattr(
        policy, "rmsnorm_device_key", lambda device: _device(policy, "cuda", "H100")
    )
    tiled = policy.RMSNormWeightGradStrategy.TILED
    default = policy._default_weight_grad_plan(tiled)
    plans = [default]
    for lo, hi, block_rows in ((0, 32, 32), (33, None, 64)):
        plans.append(
            policy.RMSNormWeightGradPlan(
                key=policy.RMSNormWeightGradKey(
                    device=_device(policy, "cuda", "H100"),
                    dtype=None,
                    n_cols=None,
                    min_rows=lo,
                    max_rows=hi,
                ),
                strategy=tiled,
                config=policy._TiledConfig(block_rows=block_rows),
                launch=default.launch,
            )
        )
    monkeypatch.setitem(policy._WEIGHT_GRAD_PLANS, tiled, tuple(plans))
    op = policy.TritonFusedAddRMSNormOp()
    outputs = []
    for shape in [(1, 9, 7), (3, 11, 7)]:
        x = torch.ones(shape, dtype=torch.bfloat16, requires_grad=True)
        residual = torch.ones(shape, dtype=torch.float32, requires_grad=True)
        weight = torch.ones(7, requires_grad=True)
        outputs.append(op(x, residual, weight))
    assert [y.grad_fn.weight_grad_plan.config.block_rows for y, _ in outputs] == [
        32,
        64,
    ]
    saved_plans = [y.grad_fn.weight_grad_plan for y, _ in outputs]
    op.weight_grad_strategy = policy.RMSNormWeightGradStrategy.SEQUENTIAL
    tiled = policy.RMSNormWeightGradStrategy.TILED
    default = policy._default_weight_grad_plan(tiled)
    new_default = policy.RMSNormWeightGradPlan(
        key=None,
        strategy=tiled,
        config=policy._TiledConfig(block_rows=16),
        launch=default.launch,
    )
    monkeypatch.setitem(
        policy._WEIGHT_GRAD_PLANS,
        tiled,
        (new_default,),
    )
    policy.select_rmsnorm_weight_grad_plan.cache_clear()
    for y, _ in outputs:
        y.sum().backward()
    assert [c["plan"].config.block_rows for c in calls] == [32, 64]
    assert all(c["plan"] is saved for c, saved in zip(calls, saved_plans, strict=True))
    assert all(c["plan"].strategy == policy.RMSNormWeightGradStrategy.TILED for c in calls)
    assert all(
        c["x_dtype"] == torch.bfloat16 and c["residual_dtype"] == torch.float32 for c in calls
    )


@pytest.mark.parametrize(
    "strategy,fields",
    [
        ("sequential", {"block_cols", "num_warps"}),
        ("tiled", {"block_rows", "block_cols", "num_warps"}),
    ],
)
def test_configs_are_strategy_specific_and_immutable(policy, strategy, fields):
    from dataclasses import FrozenInstanceError

    plans = policy._WEIGHT_GRAD_PLANS[policy.RMSNormWeightGradStrategy(strategy)]
    assert sum(plan.key is None for plan in plans) == 1
    for plan in plans:
        assert plan.strategy.value == strategy
        assert set(asdict(plan.config)) == fields
        assert inspect.signature(plan.launch).parameters["config"].annotation is type(plan.config)
        with pytest.raises(FrozenInstanceError):
            setattr(plan.config, next(iter(fields)), 1)


def test_public_constructor_accepts_only_strategy(policy):
    assert set(inspect.signature(policy.TritonFusedAddRMSNormOp).parameters) == {
        "weight_grad_strategy"
    }
    with pytest.raises(TypeError, match="weight_grad_config"):
        policy.TritonFusedAddRMSNormOp(weight_grad_config=policy._TiledConfig())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 8, 9, 32, 33, 16383, 16384, 65537, 2**31 + 1])
@pytest.mark.parametrize("width", [1, 129, 2688, 8193])
def test_auto_uses_ordered_sequential_until_retuned(policy, dtype, rows, width):
    plan = policy.select_rmsnorm_weight_grad_plan(
        _device(policy, "cuda", "NVIDIA H100 80GB HBM3"), dtype, rows, width
    )
    assert plan.strategy == policy.RMSNormWeightGradStrategy.SEQUENTIAL
    assert plan.config == policy._SequentialConfig()


@pytest.mark.parametrize("removed", ["parallel", "fused"])
def test_reassociated_strategies_are_not_selectable(policy, removed):
    with pytest.raises(ValueError):
        policy.RMSNormWeightGradStrategy(removed)
    assert {s.value for s in policy.RMSNormWeightGradStrategy} == {"sequential", "tiled"}
