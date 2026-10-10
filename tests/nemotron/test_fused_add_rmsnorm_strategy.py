# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU policy checks. Kernel execution is tested separately on a real GPU."""

import importlib.util
import sys
import types
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
    path = (
        Path(__file__).resolve().parents[2]
        / "rl_engine/kernels/ops/triton/norm/fused_add_rmsnorm.py"
    )
    spec = importlib.util.spec_from_file_location("_rmsnorm_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "triton", triton)
        imports.setitem(sys.modules, "triton.language", language)
        imports.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rows,strategy,config",
    [
        (0, "sequential", (1, 128, 4)),
        (8, "sequential", (1, 128, 4)),
        (9, "tiled", (32, 64, 4)),
        (32, "tiled", (32, 64, 4)),
        (33, "tiled", (64, 64, 8)),
        (8193, "tiled", (64, 64, 8)),
        (16383, "tiled", (64, 64, 8)),
        (16384, "fused", (64, 64, 4)),
        (16385, "fused", (64, 64, 4)),
        (32768, "fused", (64, 64, 4)),
        (65536, "fused", (64, 64, 4)),
        (65537, "fused", (64, 64, 4)),
        (2**31 + 1, "fused", (64, 64, 4)),
    ],
)
def test_model_width_boundaries_and_large_row_fallback(policy, dtype, rows, strategy, config):
    plan = policy.select_rmsnorm_weight_grad_plan(
        ("cuda", "NVIDIA H100 80GB HBM3"), dtype, rows, 2688
    )
    assert plan.strategy.value == strategy
    assert (plan.config.block_rows, plan.config.block_cols, plan.config.num_warps) == config


@pytest.mark.parametrize("width", [1, 129, 2687, 2689, 4096, 8192, 8193, 65536])
@pytest.mark.parametrize("rows", [16384, 65537])
def test_other_widths_do_not_inherit_model_fused_cutoff(policy, width, rows):
    plan = policy.select_rmsnorm_weight_grad_plan(("cuda", "other GPU"), torch.float32, rows, width)
    assert plan.strategy == policy.RMSNormWeightGradStrategy.TILED
    assert plan.config == policy.RMSNormWeightGradConfig(block_rows=64, block_cols=64, num_warps=8)


@pytest.mark.parametrize(
    "device", [("cuda", "other GPU"), ("rocm", "AMD GPU"), ("xpu", ""), ("musa", "")]
)
def test_unmeasured_devices_use_shared_default(policy, device):
    select = policy.select_rmsnorm_weight_grad_plan
    assert select(device, torch.bfloat16, 16384, 2688) == select(
        ("cuda", "NVIDIA H100 80GB HBM3"), torch.bfloat16, 16384, 2688
    )


def test_device_dtype_and_width_override_precedence(policy):
    key = policy.RMSNormWeightGradKey
    device = ("cuda", "test GPU")
    sequential = policy.RMSNormWeightGradPlan(
        policy.RMSNormWeightGradStrategy.SEQUENTIAL, policy.RMSNormWeightGradConfig(block_rows=1)
    )
    tiled = policy.RMSNormWeightGradPlan(
        policy.RMSNormWeightGradStrategy.TILED, policy.RMSNormWeightGradConfig()
    )
    parallel = policy.RMSNormWeightGradPlan(
        policy.RMSNormWeightGradStrategy.PARALLEL, policy.RMSNormWeightGradConfig(block_rows=512)
    )
    # Insert broad overrides first: specificity must win independently of insertion order.
    policy.WEIGHT_GRAD_POLICY[key(device, None, None, 0, None)] = sequential
    policy.WEIGHT_GRAD_POLICY[key(device, torch.bfloat16, None, 0, None)] = tiled
    policy.WEIGHT_GRAD_POLICY[key(device, torch.bfloat16, 2688, 100, 200)] = parallel
    select = policy.select_rmsnorm_weight_grad_plan
    assert select(device, torch.float32, 65536, 2688) == sequential
    assert select(device, torch.bfloat16, 150, 129) == tiled
    assert select(device, torch.bfloat16, 150, 2688) == parallel
    assert select(device, torch.bfloat16, 201, 2688) == tiled


def test_policy_cache_uses_all_metadata_and_is_bounded(policy):
    select = policy.select_rmsnorm_weight_grad_plan
    args = (("cuda", "H100"), torch.bfloat16, 16384, 2688)
    first = select(*args)
    assert select(*args) is first
    assert select.cache_info().hits == 1
    assert select.cache_info().maxsize == 1024
    for changed in [
        (("cuda", "other"), *args[1:]),
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
    assert policy.rmsnorm_device_key(torch.device("cuda:1")) == ("cuda", "GPU 1")
    assert current[0] == 0
    assert policy.rmsnorm_device_key(torch.device("cuda")) == ("cuda", "GPU 0")
    current[0] = 1
    assert policy.rmsnorm_device_key(torch.device("cuda")) == ("cuda", "GPU 1")
    assert names == [1, 0]
    monkeypatch.setattr(torch.version, "hip", "test")
    assert policy.rmsnorm_device_key(torch.device("cuda:1")) == ("rocm", "GPU 1")
    assert names == [1, 0, 1]


def test_explicit_options_bypass_device_query_and_query_failures_propagate(policy, monkeypatch):
    def unavailable(device):
        raise RuntimeError("device query failed")

    monkeypatch.setattr(policy, "rmsnorm_device_key", unavailable)
    resolve = policy._resolve_weight_grad_plan
    args = (torch.device("cuda:1"), torch.bfloat16, 65536, 2688)
    for strategy in policy.RMSNormWeightGradStrategy:
        plan = resolve(*args, strategy, None)
        assert plan.strategy == strategy
        assert plan.config == policy._WEIGHT_GRAD_CONFIGS[strategy]
    config = policy.RMSNormWeightGradConfig(block_rows=16, block_cols=32, num_warps=4)
    plan = resolve(*args, None, config)
    assert plan.strategy == policy.RMSNormWeightGradStrategy.SEQUENTIAL
    assert plan.config == config
    with pytest.raises(RuntimeError, match="device query failed"):
        resolve(*args, None, None)
    assert resolve(args[0], args[1], 0, args[3], None, None).strategy.value == "sequential"


def test_policy_ranges_do_not_overlap_within_the_same_metadata_tier(policy):
    keys = list(policy.WEIGHT_GRAD_POLICY)
    for i, lhs in enumerate(keys):
        assert lhs.min_rows >= 0
        assert lhs.max_rows is None or lhs.max_rows >= lhs.min_rows
        for rhs in keys[i + 1 :]:
            if (lhs.device_key, lhs.dtype, lhs.n_cols) != (rhs.device_key, rhs.dtype, rhs.n_cols):
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

    def backward(updated, inverse, weight, grad_y, grad_u, **options):
        calls.append(options)
        return (
            torch.zeros_like(updated, dtype=options["x_dtype"]),
            torch.zeros_like(updated, dtype=options["residual_dtype"]),
            torch.zeros_like(weight),
        )

    monkeypatch.setattr(policy, "_launch_fused_add_rmsnorm_fwd", forward)
    monkeypatch.setattr(policy, "_launch_fused_add_rmsnorm_bwd", backward)
    op = policy.TritonFusedAddRMSNormOp()
    outputs = []
    for shape in [(1, 9, 7), (3, 11, 7)]:
        x = torch.ones(shape, dtype=torch.bfloat16, requires_grad=True)
        residual = torch.ones(shape, dtype=torch.float32, requires_grad=True)
        weight = torch.ones(7, requires_grad=True)
        outputs.append(op(x, residual, weight))
    assert [y.grad_fn.weight_grad_config.block_rows for y, _ in outputs] == [32, 64]
    op.weight_grad_strategy = policy.RMSNormWeightGradStrategy.SEQUENTIAL
    policy.select_rmsnorm_weight_grad_plan.cache_clear()
    for y, _ in outputs:
        y.sum().backward()
    assert [c["weight_grad_config"].block_rows for c in calls] == [32, 64]
    assert all(c["weight_grad_strategy"] == policy.RMSNormWeightGradStrategy.TILED for c in calls)
    assert all(
        c["x_dtype"] == torch.bfloat16 and c["residual_dtype"] == torch.float32 for c in calls
    )


def test_fused_allocates_only_grouped_workspace(policy, monkeypatch):
    strategy = policy.RMSNormWeightGradStrategy.FUSED
    plan = policy._resolve_weight_grad_plan(
        torch.device("cpu"), torch.bfloat16, 129, 7, strategy, None
    )
    assert plan.config.block_rows == 64
    calls = []

    class RecordingKernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((self.name, grid, args, kwargs))

            return launch

    monkeypatch.setattr(policy.triton, "cdiv", lambda x, y: (x + y - 1) // y, raising=False)
    monkeypatch.setattr(
        policy.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), raising=False
    )
    monkeypatch.setattr(policy, "_fused_add_rmsnorm_bwd_grouped_kernel", RecordingKernel("grouped"))
    monkeypatch.setattr(
        policy, "_fused_add_rmsnorm_bwd_weight_tiled_kernel", RecordingKernel("merge")
    )
    x = torch.empty((129, 7))
    weight = torch.empty(7)
    inverse = torch.empty(129)
    allocations = []
    allocate = torch.empty

    def track_empty(shape, **kwargs):
        allocations.append((shape, kwargs["dtype"]))
        return allocate(shape, **kwargs)

    monkeypatch.setattr(torch, "empty", track_empty)
    policy._BACKWARD_LAUNCHERS[strategy](
        updated_residual=x,
        inverse_rms=inverse,
        weight=weight,
        grad_y=x,
        grad_updated_residual_output=x,
        grad_x=x,
        grad_residual=x,
        grad_weight=weight,
        n_rows=129,
        n_cols=7,
        plan=plan,
    )
    assert allocations == [((3, 7), torch.float32)]
    assert [(name, grid) for name, grid, _, _ in calls] == [("grouped", (3,)), ("merge", (1,))]
    assert calls[0][2][7] is calls[1][2][0]  # Merge consumes the grouped scratch directly.
    assert calls[0][3]["ROWS_PER_GROUP"] == 64
    assert calls[0][3]["num_warps"] == 4
    assert all(options["enable_fp_fusion"] is False for _, _, _, options in calls)
