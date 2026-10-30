# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strategy selection checks, including on CPU hosts without a Triton runtime."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rl_engine.contracts.operators.softcapped_selected_logprob import (
    REDUCTION_CONFIG,
    SUPPORTED_DTYPES,
    TRITON_SUPPORTED_DEVICES,
    SoftcappedLogprobReductionConfig,
)

ROW = "row"
PARALLEL = "parallel"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Include both ends of a tile, unseen interior widths, model widths and values
# beyond the old measured maximum. Expected choices are independent of the map.
_RANGE_CASES = [
    (dtype, vocab, 4096)
    for dtype in _DTYPES
    for vocab in (31745, 32767, 32768, 32769, 49151, 128256, 202048, 262145, 2097153)
]


def _host_validation_and_dispatch_module():
    """Load production policy, validation and host dispatch when Triton is unavailable.

    Execute the pre-kernel definitions, pipelined resolver and public Op class
    without Triton/reduction imports. Policy and launchers share one module.
    Tests replace dispatch before any kernel can run. GPU hosts import the
    complete operator module instead.
    """
    path = (
        Path(__file__).resolve().parents[5]
        / "rl_engine/backends/shared/triton/logprob/softcapped_selected_logprob.py"
    )
    source = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton"
            and decorator.attr == "jit"
            for decorator in node.decorator_list
        ):
            break
        if isinstance(node, ast.Import):
            node.names = [name for name in node.names if name.name.split(".")[0] != "triton"]
            if not node.names:
                continue
        elif isinstance(node, ast.ImportFrom) and (
            (node.module or "").split(".")[0] == "triton"
            or (node.module or "").startswith("rl_engine.backends.shared.triton.reduction")
        ):
            continue
        body.append(node)
    else:
        raise AssertionError("Expected a Triton kernel after host validation")
    body.extend(
        node
        for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name
        in (
            "_ForwardResult",
            "_launch_softcapped_selected_logprob_fwd_pipelined",
            "TritonSoftcappedSelectedLogprobOp",
        )
    )
    module = ModuleType("_softcapped_logprob_host_validation_and_dispatch")
    sys.modules[module.__name__] = module
    try:
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), vars(module))
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


@pytest.fixture(scope="module")
def core():
    if not torch.cuda.is_available():
        try:
            import triton
        except ImportError:
            triton = None
        if triton is None or not hasattr(triton, "jit"):
            module = _host_validation_and_dispatch_module()
            try:
                yield module
            finally:
                sys.modules.pop(module.__name__, None)
            return
    # A GPU host must fail, not skip, if its Triton backend cannot be imported.
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob

    yield softcapped_selected_logprob


@pytest.fixture(scope="module")
def config(core):
    """Use the same module for tuning maps, cached selectors and kernel wrappers."""
    return core


@pytest.mark.parametrize("explicit", (False, True))
@pytest.mark.parametrize("invalid", ("shape", "ids_shape", "ids_dtype", "softcap"))
def test_public_input_errors_precede_device_lookup_and_dispatch(
    core, monkeypatch, explicit, invalid
):
    """Reject malformed metadata before auto selection or autograd, even without a GPU."""
    # Only host validation runs. Accept CPU here so real tensors exercise every
    # metadata check; neither hardware lookup nor autograd may be reached.
    monkeypatch.setattr(core, "TRITON_SUPPORTED_DEVICES", ("cpu",))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid inputs reached device lookup or autograd dispatch")

    monkeypatch.setattr(core, "softcapped_logprob_device_key", forbidden)
    monkeypatch.setattr(
        core,
        "_SoftcappedSelectedLogprobTritonFunction",
        SimpleNamespace(apply=forbidden),
        raising=False,
    )
    logits, ids, cap = torch.zeros((1, 4)), torch.tensor([0]), 30.0
    if invalid == "shape":
        logits = logits[0]
    elif invalid == "ids_shape":
        ids = ids.repeat(2)
    elif invalid == "ids_dtype":
        ids = ids.float()
    else:
        cap = 0.0
    strategy = core.SoftcappedLogprobStrategy.ROW if explicit else None
    op = core.TritonSoftcappedSelectedLogprobOp(forward_impl=strategy)
    error = TypeError if invalid == "ids_dtype" else ValueError
    message = {"shape": "shape", "ids_shape": "shape", "ids_dtype": "int64", "softcap": "softcap"}
    with pytest.raises(error, match=message[invalid]):
        op(logits, ids, softcap=cap)


@pytest.mark.parametrize("softcap", (0.0, -1.0, float("nan"), float("inf"), 1e-50, 100.0))
def test_softcap_rejects_unsafe_values(core, softcap):
    with pytest.raises(ValueError, match="softcap"):
        core._validate_softcap(softcap, 1)


def test_softcap_range_depends_on_vocabulary(core):
    for cap in (1.0, 17.3, 20.0, 30.0, 50.0):
        core._validate_softcap(cap, 262144)
    core._validate_softcap(80.0, 1)
    with pytest.raises(ValueError, match="V=32768.*FP32 exp/sum range"):
        core._validate_softcap(80.0, 32768)


@pytest.fixture(autouse=True)
def clear_strategy_cache(config):
    # A test may temporarily replace the map. Cached choices must not escape
    # that configuration or survive a failure during the test.
    config.select_softcapped_logprob_strategy.cache_clear()
    config.select_softcapped_logprob_pipeline_config.cache_clear()
    yield
    config.select_softcapped_logprob_strategy.cache_clear()
    config.select_softcapped_logprob_pipeline_config.cache_clear()


def _install_configs(config, monkeypatch, entries):
    monkeypatch.setattr(config, "FORWARD_STRATEGY_CONFIGS", entries)
    config.select_softcapped_logprob_strategy.cache_clear()


def test_backend_uses_shared_input_and_reduction_contract(core):
    """The backend does not maintain a second copy of reference metadata."""
    assert core.REDUCTION_CONFIG is REDUCTION_CONFIG
    assert core.SUPPORTED_DTYPES is SUPPORTED_DTYPES
    assert core.TRITON_SUPPORTED_DEVICES is TRITON_SUPPORTED_DEVICES


@pytest.fixture
def pipeline_ranges(config, monkeypatch):
    """Synthetic ranges exercise selection; they are not measured tuning recommendations."""
    device = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
    monkeypatch.setattr(
        config,
        "ROW_PIPELINE_CONFIGS",
        {
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=4,
                min_tiles=2,
                max_tiles=4,
            ): config.RowPipelineConfig(pipeline_stages=2),
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=5,
                max_rows=8,
                min_tiles=2,
                max_tiles=None,
            ): config.RowPipelineConfig(pipeline_stages=4),
            config.ForwardConfigKey(
                device=device,
                dtype=torch.bfloat16,
                min_rows=2,
                max_rows=3,
                min_tiles=2,
                max_tiles=2,
            ): config.RowPipelineConfig(pipeline_stages=1),
        },
    )
    return device


@pytest.mark.parametrize("stages", (0, -1))
def test_pipeline_config_rejects_nonpositive_depth(config, stages):
    with pytest.raises(ValueError, match="pipeline_stages must be positive"):
        config.RowPipelineConfig(pipeline_stages=stages)


@pytest.mark.parametrize(
    "backend,gpu_name,dtype,rows,vocab,expected_stages",
    (
        ("cuda", "test GPU", torch.bfloat16, 1, 1025, 2),
        ("cuda", "test GPU", torch.bfloat16, 4, 4096, 2),
        ("cuda", "test GPU", torch.bfloat16, 2, 1025, 1),
        ("cuda", "test GPU", torch.bfloat16, 3, 2048, 1),
        ("cuda", "test GPU", torch.bfloat16, 3, 2049, 2),
        ("cuda", "test GPU", torch.bfloat16, 5, 1025, 4),
        ("cuda", "test GPU", torch.bfloat16, 8, 1 << 40, 4),
        ("cuda", "other GPU", torch.bfloat16, 2, 1025, 2),
        ("rocm", "test GPU", torch.bfloat16, 2, 1025, 2),
        ("cuda", "test GPU", torch.float16, 2, 1025, None),
        ("cuda", "test GPU", torch.bfloat16, 0, 1025, None),
        ("cuda", "test GPU", torch.bfloat16, 1, 1024, None),
        ("cuda", "test GPU", torch.bfloat16, 4, 4097, None),
        ("cuda", "test GPU", torch.bfloat16, 9, 1025, None),
        ("cuda", "test GPU", torch.bfloat16, 1 << 40, 1 << 40, None),
    ),
)
def test_pipeline_ranges_match_shape_dtype_and_device(
    config, pipeline_ranges, backend, gpu_name, dtype, rows, vocab, expected_stages
):
    actual = config.select_softcapped_logprob_pipeline_config(
        config.SoftcappedLogprobDevice(backend=backend, gpu_name=gpu_name), dtype, rows, vocab
    )
    if expected_stages is None:
        assert actual is config.DEFAULT_ROW_PIPELINE_CONFIG
    else:
        assert actual.pipeline_stages == expected_stages


@pytest.mark.parametrize("tile_size", (16, 2048))
def test_pipeline_ranges_use_the_configured_tile_width(
    config, pipeline_ranges, monkeypatch, tile_size
):
    monkeypatch.setattr(
        config, "REDUCTION_CONFIG", SoftcappedLogprobReductionConfig(tile_size=tile_size)
    )
    select = config.select_softcapped_logprob_pipeline_config
    assert (
        select(pipeline_ranges, torch.bfloat16, 1, tile_size) is config.DEFAULT_ROW_PIPELINE_CONFIG
    )
    assert select(pipeline_ranges, torch.bfloat16, 1, tile_size + 1).pipeline_stages == 2


@pytest.mark.parametrize("device_specific", (False, True))
@pytest.mark.parametrize("reverse_order", (False, True))
def test_pipeline_overlapping_ranges_fail(config, monkeypatch, device_specific, reverse_order):
    device = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
    scope = device if device_specific else config._DEFAULT_DEVICE
    entries = [
        (
            config.ForwardConfigKey(
                device=scope,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=max_rows,
                min_tiles=2,
                max_tiles=4,
            ),
            config.RowPipelineConfig(pipeline_stages=stages),
        )
        for max_rows, stages in ((4, 2), (8, 4))
    ]
    monkeypatch.setattr(
        config, "ROW_PIPELINE_CONFIGS", dict(reversed(entries) if reverse_order else entries)
    )
    with pytest.raises(ValueError, match="overlapping row pipeline ranges"):
        config.select_softcapped_logprob_pipeline_config(device, torch.bfloat16, 4, 1025)


def test_pipeline_cache_separates_shapes_and_updates_after_clear(
    config, pipeline_ranges, monkeypatch
):
    select = config.select_softcapped_logprob_pipeline_config
    for context in (torch.enable_grad(), torch.no_grad(), torch.inference_mode()):
        with context:
            assert select(pipeline_ranges, torch.bfloat16, 4, 1025).pipeline_stages == 2
            assert select(pipeline_ranges, torch.bfloat16, 5, 1025).pipeline_stages == 4
    assert select.cache_info().misses == 2
    assert select.cache_info().hits == 4
    assert select.cache_info().maxsize == 1024
    monkeypatch.setattr(config, "ROW_PIPELINE_CONFIGS", {})
    monkeypatch.setattr(
        config, "DEFAULT_ROW_PIPELINE_CONFIG", config.RowPipelineConfig(pipeline_stages=5)
    )
    select.cache_clear()
    assert select(pipeline_ranges, torch.bfloat16, 4, 1025).pipeline_stages == 5


def test_pipelined_launcher_resolves_each_calls_shape(core, pipeline_ranges, monkeypatch):
    """Exercise real resolution and forwarding while replacing only the GPU launch."""
    seen = []
    sentinel = object()
    monkeypatch.setattr(core, "softcapped_logprob_device_key", lambda device: pipeline_ranges)

    def capture(logits, token_ids, softcap, *, pipeline_stages):
        seen.append((logits, token_ids, softcap, pipeline_stages))
        return sentinel

    monkeypatch.setattr(core, "_launch_softcapped_selected_logprob_fwd", capture, raising=False)
    for rows, stages in ((4, 2), (2, 1), (5, 4)):
        logits = torch.empty((rows, 1025), dtype=torch.bfloat16)
        ids = torch.zeros(rows, dtype=torch.int64)
        result = core._launch_softcapped_selected_logprob_fwd_pipelined(logits, ids, 17.3)
        assert result is sentinel
        assert seen[-1][0] is logits and seen[-1][1] is ids
        assert seen[-1][2:] == (17.3, stages)


@pytest.mark.parametrize(
    "device,hip_version,expected",
    (
        ("cuda", None, {"enable_fp_fusion": False, "enable_reflect_ftz": False}),
        ("cuda", "test-rocm", {"enable_fp_fusion": False, "allow_flush_denorm": False}),
        ("cpu", "test-rocm", {"enable_fp_fusion": False}),
        ("xpu", None, {"enable_fp_fusion": False}),
    ),
)
def test_math_options_preserve_subnormals_without_foreign_backend_flags(
    core, monkeypatch, device, hip_version, expected
):
    """Both GPU builds use CUDA devices; a ROCm build alone does not select GPU flags."""
    compiler = ModuleType("triton.backends.nvidia.compiler")

    class CUDAOptions:
        enable_reflect_ftz = True

    compiler.CUDAOptions = CUDAOptions
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    monkeypatch.setattr(torch.version, "hip", hip_version)
    assert core._math_launch_options(torch.device(device)) == expected


def test_older_cuda_backend_cannot_silently_ignore_subnormal_preservation(core, monkeypatch):
    """Older compilers must report the missing capability before launching kernels."""
    compiler = ModuleType("triton.backends.nvidia.compiler")
    compiler.CUDAOptions = type("CUDAOptions", (), {})
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="enable_reflect_ftz.*Upgrade Triton"):
        core._math_launch_options(torch.device("cuda"))


@pytest.mark.parametrize("tile_size", (16, 512, 1024, 2048))
def test_selector_counts_tiles_using_the_shared_config(config, monkeypatch, tile_size):
    """A changed arithmetic width must also move the 32-tile policy boundary."""
    monkeypatch.setattr(
        config, "REDUCTION_CONFIG", SoftcappedLogprobReductionConfig(tile_size=tile_size)
    )
    device = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="unmeasured GPU")
    select = config.select_softcapped_logprob_strategy
    assert select(device, torch.float32, 1, 31 * tile_size).value == ROW
    assert select(device, torch.float32, 1, 31 * tile_size + 1).value == PARALLEL


def test_auto_policy_excludes_explicit_experiments(config):
    # Pipelining did not beat PARALLEL; it remains explicit-only.
    assert config.SoftcappedLogprobStrategy.ROW_PIPELINED not in (
        config.FORWARD_STRATEGY_CONFIGS.values()
    )


def test_different_reduction_order_is_not_selectable(config):
    """Every retained strategy must share the same output and gradient bytes."""
    with pytest.raises(ValueError):
        config.SoftcappedLogprobStrategy("row_accumulate")


@pytest.mark.parametrize("dtype,vocab,max_rows", _RANGE_CASES)
def test_default_range_includes_boundaries_and_interior(config, dtype, vocab, max_rows):
    for rows in (1, 2, max_rows // 2, max_rows - 1, max_rows):
        assert (
            config.select_softcapped_logprob_strategy(
                config.SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100"),
                dtype,
                rows,
                vocab,
            )
            == PARALLEL
        )


@pytest.mark.parametrize("dtype,vocab,max_rows", _RANGE_CASES)
def test_rows_outside_default_range_use_row(config, dtype, vocab, max_rows):
    for rows in (0, max_rows + 1, max_rows * 2, 1 << 40):
        assert (
            config.select_softcapped_logprob_strategy(
                config.SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100"),
                dtype,
                rows,
                vocab,
            )
            == ROW
        )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "rows,vocab",
    (
        (16384, 262144),  # Rows above the configured maximum.
        (16, 31744),  # Last width below the minimum tile count.
        (32768, 2097152),  # Large vocabulary still respects the row limit.
        (1 << 40, 1 << 40),  # Metadata only: no allocation or integer truncation.
    ),
)
def test_all_out_of_map_dimensions_have_an_explicit_final_default(config, dtype, rows, vocab):
    assert config.DEFAULT_FORWARD_STRATEGY is config.SoftcappedLogprobStrategy.ROW
    assert (
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100"),
            dtype,
            rows,
            vocab,
        )
        is config.DEFAULT_FORWARD_STRATEGY
    )


def test_final_default_is_configurable_without_overriding_matched_rules(config, monkeypatch):
    key = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
    _install_configs(
        config,
        monkeypatch,
        {
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=1024,
                min_tiles=256,
                max_tiles=256,
            ): config.SoftcappedLogprobStrategy.ROW,
        },
    )
    monkeypatch.setattr(
        config, "DEFAULT_FORWARD_STRATEGY", config.SoftcappedLogprobStrategy.PARALLEL
    )
    select = config.select_softcapped_logprob_strategy
    assert select(key, torch.bfloat16, 16, 262144) == ROW
    assert select(key, torch.bfloat16, 2048, 524288) == PARALLEL


@pytest.mark.parametrize("vocab", (31745, 32766, 32770, 49151, 65534, 202048, 262147))
def test_unmeasured_vocabulary_reuses_a_tile_range(config, vocab):
    assert (
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA A40"),
            torch.bfloat16,
            16,
            vocab,
        )
        == PARALLEL
    )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_tile_count_boundary_uses_ceiling_without_power_of_two_rounding(config, dtype):
    select = config.select_softcapped_logprob_strategy
    device = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
    # 31 full tiles -> ROW; adding one element creates the 32nd tile -> PARALLEL.
    assert select(device, dtype, 16, 31744) == ROW
    assert select(device, dtype, 16, 31745) == PARALLEL
    # next_power_of_2 would incorrectly send this 17-tile width to PARALLEL.
    assert select(device, dtype, 16, 16385) == ROW


def test_dtype_must_match_an_entry(config):
    assert (
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100"),
            torch.float64,
            16,
            262144,
        )
        == ROW
    )


@pytest.mark.parametrize(
    "backend,gpu_name", [("cuda", "unlisted GPU"), ("rocm", "unlisted GPU"), ("xpu", "")]
)
def test_unlisted_devices_reuse_the_default_policy(config, backend, gpu_name):
    device = config.SoftcappedLogprobDevice(backend=backend, gpu_name=gpu_name)
    for vocab in (32768, 49152, 65536, 262143, 262144, 262145, 1048576):
        assert (
            config.select_softcapped_logprob_strategy(device, torch.bfloat16, 16, vocab) == PARALLEL
        )
    assert config.select_softcapped_logprob_strategy(device, torch.bfloat16, 16, 1025) == ROW


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("rows", (1, 16, 1024, 8192, 16384))
@pytest.mark.parametrize("vocab", (512, 1024, 8192, 16384))
def test_small_vocabularies_keep_row_even_when_rows_are_large(config, dtype, rows, vocab):
    assert (
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100"),
            dtype,
            rows,
            vocab,
        )
        == ROW
    )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_same_tile_neighbors_share_policy(config, dtype):
    select = config.select_softcapped_logprob_strategy
    for vocab in (32767, 32768, 32769, 202048):
        assert (
            select(
                config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
                dtype,
                4096,
                vocab,
            )
            == PARALLEL
        )
        assert (
            select(
                config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
                dtype,
                4097,
                vocab,
            )
            == ROW
        )


def test_production_ranges_are_valid_and_disjoint(config):
    keys = list(config.FORWARD_STRATEGY_CONFIGS)
    for index, key in enumerate(keys):
        assert 1 <= key.min_rows <= key.max_rows
        assert key.min_tiles > 0
        assert key.max_tiles is None or key.min_tiles <= key.max_tiles
        for other in keys[:index]:
            if (key.device, key.dtype) != (other.device, other.dtype):
                continue
            rows_overlap = key.min_rows <= other.max_rows and other.min_rows <= key.max_rows
            tiles_overlap = (key.max_tiles is None or other.min_tiles <= key.max_tiles) and (
                other.max_tiles is None or key.min_tiles <= other.max_tiles
            )
            assert not (rows_overlap and tiles_overlap)


def test_device_override_precedes_default_without_hiding_other_default_entries(config, monkeypatch):
    key = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
    entries = dict(config.FORWARD_STRATEGY_CONFIGS)
    entries[
        config.ForwardConfigKey(
            device=key, dtype=torch.bfloat16, min_rows=8, max_rows=32, min_tiles=256, max_tiles=256
        )
    ] = config.SoftcappedLogprobStrategy.ROW
    _install_configs(config, monkeypatch, entries)
    for rows in (8, 16, 32):
        assert config.select_softcapped_logprob_strategy(key, torch.bfloat16, rows, 262144) == ROW
    for rows in (7, 33):
        assert (
            config.select_softcapped_logprob_strategy(key, torch.bfloat16, rows, 262144) == PARALLEL
        )
    assert config.select_softcapped_logprob_strategy(key, torch.float16, 16, 262144) == PARALLEL
    # A model-name match alone must not select a CUDA-specific override on ROCm.
    assert (
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="rocm", gpu_name="test GPU"),
            torch.bfloat16,
            16,
            262144,
        )
        == PARALLEL
    )


@pytest.mark.parametrize("device_specific", (False, True))
@pytest.mark.parametrize("reverse_order", (False, True))
def test_overlapping_ranges_at_the_same_precedence_fail(
    config, monkeypatch, device_specific, reverse_order
):
    scope = (
        config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
        if device_specific
        else config._DEFAULT_DEVICE
    )
    entries = [
        (
            config.ForwardConfigKey(
                device=scope,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=16,
                min_tiles=256,
                max_tiles=256,
            ),
            config.SoftcappedLogprobStrategy.ROW,
        ),
        (
            config.ForwardConfigKey(
                device=scope,
                dtype=torch.bfloat16,
                min_rows=16,
                max_rows=32,
                min_tiles=256,
                max_tiles=256,
            ),
            config.SoftcappedLogprobStrategy.PARALLEL,
        ),
    ]
    if reverse_order:
        entries.reverse()
    _install_configs(config, monkeypatch, dict(entries))
    # Inclusive endpoints mean both rules match row count 16.
    with pytest.raises(ValueError, match="overlapping forward strategy ranges"):
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
            torch.bfloat16,
            16,
            262144,
        )
    assert (
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
            torch.bfloat16,
            15,
            262144,
        )
        == ROW
    )
    assert (
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
            torch.bfloat16,
            17,
            262144,
        )
        == PARALLEL
    )


def test_device_match_does_not_consult_ambiguous_default_ranges(config, monkeypatch):
    device = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
    _install_configs(
        config,
        monkeypatch,
        {
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=32,
                min_tiles=256,
                max_tiles=256,
            ): config.SoftcappedLogprobStrategy.ROW,
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=16,
                max_rows=64,
                min_tiles=256,
                max_tiles=256,
            ): config.SoftcappedLogprobStrategy.PARALLEL,
            config.ForwardConfigKey(
                device=device,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=64,
                min_tiles=256,
                max_tiles=256,
            ): config.SoftcappedLogprobStrategy.ROW,
        },
    )
    assert config.select_softcapped_logprob_strategy(device, torch.bfloat16, 16, 262144) == ROW
    with pytest.raises(ValueError, match="overlapping forward strategy ranges"):
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="rocm", gpu_name="test GPU"),
            torch.bfloat16,
            16,
            262144,
        )


def test_adjacent_tile_ranges_and_open_upper_bound(config, monkeypatch):
    """Finite and unbounded rules can share rows when their tile ranges are disjoint."""
    _install_configs(
        config,
        monkeypatch,
        {
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=16,
                min_tiles=32,
                max_tiles=64,
            ): config.SoftcappedLogprobStrategy.ROW,
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=16,
                min_tiles=65,
                max_tiles=None,
            ): config.SoftcappedLogprobStrategy.PARALLEL,
        },
    )
    select = config.select_softcapped_logprob_strategy
    for vocab in (32767, 32768, 65535, 65536):
        assert (
            select(
                config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
                torch.bfloat16,
                16,
                vocab,
            )
            == ROW
        )
    for vocab in (65537, 65538, 1 << 40):
        assert (
            select(
                config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
                torch.bfloat16,
                16,
                vocab,
            )
            == PARALLEL
        )
        assert (
            select(
                config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
                torch.bfloat16,
                17,
                vocab,
            )
            == ROW
        )


@pytest.mark.parametrize("upper", (64, None))
def test_overlapping_tile_ranges_fail_at_the_shared_endpoint(config, monkeypatch, upper):
    _install_configs(
        config,
        monkeypatch,
        {
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=16,
                min_tiles=32,
                max_tiles=upper,
            ): config.SoftcappedLogprobStrategy.ROW,
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=16,
                min_tiles=64,
                max_tiles=None,
            ): config.SoftcappedLogprobStrategy.PARALLEL,
        },
    )
    with pytest.raises(ValueError, match="overlapping forward strategy ranges"):
        config.select_softcapped_logprob_strategy(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
            torch.bfloat16,
            16,
            65536,
        )


def test_repeated_metadata_skips_configuration_matching(config, monkeypatch):
    scans = []

    class ObservedConfigs(dict):
        def items(self):
            scans.append(True)
            return super().items()

    _install_configs(config, monkeypatch, ObservedConfigs(config.FORWARD_STRATEGY_CONFIGS))
    args = (
        config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
        torch.bfloat16,
        17,
        262144,
    )
    assert config.select_softcapped_logprob_strategy(*args) == PARALLEL
    first_scans = len(scans)
    assert first_scans > 0
    # A newly constructed equivalent device must hit the same cached selection.
    equivalent_device = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
    assert equivalent_device is not args[0]
    assert config.select_softcapped_logprob_strategy(equivalent_device, *args[1:]) == PARALLEL
    assert len(scans) == first_scans
    assert config.select_softcapped_logprob_strategy.cache_info().hits == 1


def test_selector_cache_keeps_alternating_shapes(config):
    """Decode/prefill shape changes must not evict each other's warm selection."""
    select = config.select_softcapped_logprob_strategy
    device = config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU")
    shapes = ((16, 262144), (4097, 262144))
    for _ in range(2):
        for rows, vocab in shapes:
            select(device, torch.bfloat16, rows, vocab)
    assert select.cache_info().misses == 2
    assert select.cache_info().hits == 2
    assert select.cache_info().maxsize == 1024


@pytest.mark.parametrize(
    "backend,gpu_name,dtype,n_rows,vocab_size",
    [
        ("cuda", "other GPU", torch.bfloat16, 16, 262144),
        ("cuda", "test GPU", torch.float32, 16, 262144),
        ("cuda", "test GPU", torch.bfloat16, 1025, 262144),
        ("cuda", "test GPU", torch.bfloat16, 16, 262145),
    ],
)
def test_cache_distinguishes_each_metadata_field(
    config, monkeypatch, backend, gpu_name, dtype, n_rows, vocab_size
):
    _install_configs(
        config,
        monkeypatch,
        {
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=1024,
                min_tiles=256,
                max_tiles=256,
            ): config.SoftcappedLogprobStrategy.PARALLEL,
            config.ForwardConfigKey(
                device=config.SoftcappedLogprobDevice(backend="cuda", gpu_name="other GPU"),
                dtype=torch.bfloat16,
                min_rows=1,
                max_rows=1024,
                min_tiles=256,
                max_tiles=256,
            ): config.SoftcappedLogprobStrategy.ROW,
        },
    )
    select = config.select_softcapped_logprob_strategy
    assert (
        select(
            config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
            torch.bfloat16,
            16,
            262144,
        )
        == PARALLEL
    )
    assert (
        select(
            config.SoftcappedLogprobDevice(backend=backend, gpu_name=gpu_name),
            dtype,
            n_rows,
            vocab_size,
        )
        == ROW
    )
    assert select.cache_info().misses == 2


def test_clearing_cache_applies_a_replacement_policy(config, monkeypatch):
    select = config.select_softcapped_logprob_strategy
    args = (
        config.SoftcappedLogprobDevice(backend="cuda", gpu_name="test GPU"),
        torch.bfloat16,
        16,
        262144,
    )
    assert select(*args) == PARALLEL
    monkeypatch.setattr(config, "FORWARD_STRATEGY_CONFIGS", {})
    select.cache_clear()
    assert select(*args) == ROW


def test_selection_does_not_depend_on_grad_recording(config):
    args = (
        config.SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100"),
        torch.bfloat16,
        513,
        262144,
    )
    with torch.enable_grad():
        training = config.select_softcapped_logprob_strategy(*args)
    with torch.no_grad():
        assert config.select_softcapped_logprob_strategy(*args) is training
    with torch.inference_mode():
        assert config.select_softcapped_logprob_strategy(*args) is training


@pytest.fixture
def clear_device_cache(config):
    config._cuda_device_key.cache_clear()
    yield
    config._cuda_device_key.cache_clear()


def test_device_lookup_uses_tensor_device_and_caches_each_gpu(
    config, monkeypatch, clear_device_cache
):
    calls = []

    def name(index):
        calls.append(index)
        return {0: "NVIDIA H100", 1: "another GPU"}[index]

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_name", name)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert config.softcapped_logprob_device_key(
        torch.device("cuda:1")
    ) == config.SoftcappedLogprobDevice(backend="cuda", gpu_name="another GPU")
    assert config.softcapped_logprob_device_key(
        torch.device("cuda:1")
    ) == config.SoftcappedLogprobDevice(backend="cuda", gpu_name="another GPU")
    assert config.softcapped_logprob_device_key(
        torch.device("cuda")
    ) == config.SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert config.softcapped_logprob_device_key(
        torch.device("cuda")
    ) == config.SoftcappedLogprobDevice(backend="cuda", gpu_name="another GPU")
    assert calls == [1, 0]


@pytest.mark.parametrize("device", ("cuda", "cuda:2"))
def test_device_name_lookup_errors_propagate(config, monkeypatch, clear_device_cache, device):
    def failed_lookup(index):
        raise RuntimeError("device query failed")

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", failed_lookup)
    with pytest.raises(RuntimeError, match="device query failed"):
        config.softcapped_logprob_device_key(torch.device(device))


def test_current_device_lookup_errors_propagate(config, monkeypatch, clear_device_cache):
    def failed_lookup():
        raise RuntimeError("current device query failed")

    monkeypatch.setattr(torch.cuda, "current_device", failed_lookup)
    with pytest.raises(RuntimeError, match="current device query failed"):
        config.softcapped_logprob_device_key(torch.device("cuda"))


def test_rocm_uses_cuda_namespace_but_a_separate_config_key(
    config, monkeypatch, clear_device_cache
):
    monkeypatch.setattr(torch.version, "hip", "test HIP")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "AMD test GPU")
    assert config.softcapped_logprob_device_key(
        torch.device("cuda:2")
    ) == config.SoftcappedLogprobDevice(backend="rocm", gpu_name="AMD test GPU")


def test_other_backends_do_not_query_cuda(config, monkeypatch):
    def unexpected_query(*args):
        pytest.fail("A non-CUDA tensor must not query CUDA hardware")

    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_query)
    monkeypatch.setattr(torch.cuda, "current_device", unexpected_query)
    assert config.softcapped_logprob_device_key(
        torch.device("xpu:1")
    ) == config.SoftcappedLogprobDevice(backend="xpu", gpu_name="")
