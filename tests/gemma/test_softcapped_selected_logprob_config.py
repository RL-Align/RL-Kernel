# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strategy selection checks, including on CPU hosts without a Triton runtime."""

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROW = "row"
PARALLEL = "parallel"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Representative policy boundaries, including dtype-dependent choices and tails.
_RANGE_CASES = [
    (dtype, vocab, max_rows)
    for dtype in _DTYPES
    for vocab, max_rows in (
        (32767, 16),
        (32768, 2049),
        (32769, 1024),
        (49152, 2048),
        (262143, 1026),
        (262145, 1026),
        (524288, 2048),
        (1048576, 1024),
        (1048577, 16),
    )
] + [
    (torch.float16, 262144, 4096),
    (torch.bfloat16, 262144, 4096),
    (torch.float32, 262144, 2049),
    (torch.float16, 131072, 4096),
    (torch.bfloat16, 131072, 2048),
    (torch.float32, 131072, 2048),
]


def _host_configuration_module():
    """Execute the real host definitions, stopping before any GPU kernel.

    The operator keeps configuration beside its kernels. Reading this prefix
    lets CPU-only development check the production selector without importing
    Triton stubs, replacing kernels, or duplicating the selection algorithm.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "rl_engine/kernels/ops/triton/loss/softcapped_selected_logprob.py"
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
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "triton":
            continue
        body.append(node)
    else:
        raise AssertionError("Expected a Triton kernel after the host configuration")
    module = ModuleType("_softcapped_logprob_host_configuration")
    sys.modules[module.__name__] = module
    try:
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), vars(module))
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


@pytest.fixture(scope="module")
def config():
    if not torch.cuda.is_available():
        try:
            import triton
        except ImportError:
            triton = None
        if triton is None or not hasattr(triton, "jit"):
            module = _host_configuration_module()
            try:
                yield module
            finally:
                sys.modules.pop(module.__name__, None)
            return
    # A GPU host must fail, not skip, if its Triton backend cannot be imported.
    from rl_engine.kernels.ops.triton.loss import softcapped_selected_logprob

    yield softcapped_selected_logprob


@pytest.fixture(autouse=True)
def clear_strategy_cache(config):
    # A test may temporarily replace the map. Cached choices must not escape
    # that configuration or survive a failure during the test.
    config.select_softcapped_logprob_strategy.cache_clear()
    yield
    config.select_softcapped_logprob_strategy.cache_clear()


def _install_configs(config, monkeypatch, entries):
    monkeypatch.setattr(config, "FORWARD_STRATEGY_CONFIGS", entries)
    config.select_softcapped_logprob_strategy.cache_clear()


def test_auto_policy_excludes_explicit_experiments(config):
    # Accumulation changes rounding order; pipelining did not beat PARALLEL.
    # Neither experimental strategy is enabled by the default range policy.
    assert config.SoftcappedLogprobStrategy.ROW_ACCUMULATE not in (
        config.FORWARD_STRATEGY_CONFIGS.values()
    )
    assert config.SoftcappedLogprobStrategy.ROW_PIPELINED not in (
        config.FORWARD_STRATEGY_CONFIGS.values()
    )


@pytest.mark.parametrize("dtype,vocab,max_rows", _RANGE_CASES)
def test_default_range_includes_boundaries_and_interior(config, dtype, vocab, max_rows):
    for rows in (1, 2, max_rows // 2, max_rows - 1, max_rows):
        assert (
            config.select_softcapped_logprob_strategy(("cuda", "NVIDIA H100"), dtype, rows, vocab)
            == PARALLEL
        )


@pytest.mark.parametrize("dtype,vocab,max_rows", _RANGE_CASES)
def test_rows_outside_default_range_use_row(config, dtype, vocab, max_rows):
    for rows in (0, max_rows + 1, max_rows * 2, 1 << 40):
        assert (
            config.select_softcapped_logprob_strategy(("cuda", "NVIDIA H100"), dtype, rows, vocab)
            == ROW
        )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "rows,vocab",
    (
        (16384, 262144),  # Rows above the configured maximum.
        (16, 2097152),  # Vocabulary above the configured maximum.
        (32768, 2097152),  # Both above the configured maximum.
        (1 << 40, 1 << 40),  # Metadata only: no allocation or integer truncation.
    ),
)
def test_all_out_of_map_dimensions_have_an_explicit_final_default(config, dtype, rows, vocab):
    assert config.DEFAULT_FORWARD_STRATEGY is config.SoftcappedLogprobStrategy.ROW
    assert (
        config.select_softcapped_logprob_strategy(("cuda", "NVIDIA H100"), dtype, rows, vocab)
        is config.DEFAULT_FORWARD_STRATEGY
    )


def test_final_default_is_configurable_without_overriding_matched_rules(config, monkeypatch):
    key = ("cuda", "test GPU")
    _install_configs(
        config,
        monkeypatch,
        {
            config.ForwardConfigKey(
                "default", torch.bfloat16, 1, 1024, 262144
            ): config.SoftcappedLogprobStrategy.ROW,
        },
    )
    monkeypatch.setattr(
        config, "DEFAULT_FORWARD_STRATEGY", config.SoftcappedLogprobStrategy.PARALLEL
    )
    select = config.select_softcapped_logprob_strategy
    assert select(key, torch.bfloat16, 16, 262144) == ROW
    assert select(key, torch.bfloat16, 2048, 524288) == PARALLEL


@pytest.mark.parametrize("vocab", (1025, 28672, 32766, 32770, 49151, 65534, 262141, 262147))
def test_unmeasured_vocabulary_does_not_round_to_a_measured_neighbor(config, vocab):
    assert (
        config.select_softcapped_logprob_strategy(("cuda", "NVIDIA A40"), torch.bfloat16, 16, vocab)
        == ROW
    )


def test_dtype_must_match_an_entry(config):
    assert (
        config.select_softcapped_logprob_strategy(
            ("cuda", "NVIDIA H100"), torch.float64, 16, 262144
        )
        == ROW
    )


@pytest.mark.parametrize(
    "device", [("cuda", "unlisted GPU"), ("rocm", "unlisted GPU"), ("xpu", "")]
)
def test_unlisted_devices_reuse_the_default_policy(config, device):
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
        config.select_softcapped_logprob_strategy(("cuda", "NVIDIA H100"), dtype, rows, vocab)
        == ROW
    )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_neighbors_can_have_different_policy_at_the_same_row_count(config, dtype):
    select = config.select_softcapped_logprob_strategy
    device = ("cuda", "NVIDIA H100")
    assert select(device, dtype, 1024, 32767) == ROW
    assert select(device, dtype, 1024, 32768) == PARALLEL
    assert select(device, dtype, 1024, 32769) == PARALLEL


def test_large_gemma_batches_distinguish_storage_precision(config):
    select = config.select_softcapped_logprob_strategy
    device = ("cuda", "NVIDIA H100")
    for rows in (1025, 1026, 2048, 2049):
        for dtype in _DTYPES:
            assert select(device, dtype, rows, 262144) == PARALLEL
    for rows in (3072, 4096):
        assert select(device, torch.float16, rows, 262144) == PARALLEL
        assert select(device, torch.bfloat16, rows, 262144) == PARALLEL
        assert select(device, torch.float32, rows, 262144) == ROW


def test_production_ranges_are_valid_and_disjoint(config):
    groups = {}
    for key in config.FORWARD_STRATEGY_CONFIGS:
        assert 1 <= key.min_rows <= key.max_rows
        assert key.vocab_size > 0
        groups.setdefault((key.device_key, key.dtype, key.vocab_size), []).append(
            (key.min_rows, key.max_rows)
        )
    for ranges in groups.values():
        ordered = sorted(ranges)
        assert all(left[1] < right[0] for left, right in zip(ordered, ordered[1:], strict=False))


def test_device_override_precedes_default_without_hiding_other_default_entries(config, monkeypatch):
    key = ("cuda", "test GPU")
    entries = dict(config.FORWARD_STRATEGY_CONFIGS)
    entries[config.ForwardConfigKey(key, torch.bfloat16, 8, 32, 262144)] = (
        config.SoftcappedLogprobStrategy.ROW
    )
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
        config.select_softcapped_logprob_strategy(("rocm", "test GPU"), torch.bfloat16, 16, 262144)
        == PARALLEL
    )


@pytest.mark.parametrize("scope", ("default", ("cuda", "test GPU")))
@pytest.mark.parametrize("reverse_order", (False, True))
def test_overlapping_ranges_at_the_same_precedence_fail(config, monkeypatch, scope, reverse_order):
    entries = [
        (
            config.ForwardConfigKey(scope, torch.bfloat16, 1, 16, 262144),
            config.SoftcappedLogprobStrategy.ROW,
        ),
        (
            config.ForwardConfigKey(scope, torch.bfloat16, 16, 32, 262144),
            config.SoftcappedLogprobStrategy.PARALLEL,
        ),
    ]
    if reverse_order:
        entries.reverse()
    _install_configs(config, monkeypatch, dict(entries))
    # Inclusive endpoints mean both rules match row count 16.
    with pytest.raises(ValueError, match="overlapping forward strategy ranges"):
        config.select_softcapped_logprob_strategy(("cuda", "test GPU"), torch.bfloat16, 16, 262144)
    assert (
        config.select_softcapped_logprob_strategy(("cuda", "test GPU"), torch.bfloat16, 15, 262144)
        == ROW
    )
    assert (
        config.select_softcapped_logprob_strategy(("cuda", "test GPU"), torch.bfloat16, 17, 262144)
        == PARALLEL
    )


def test_device_match_does_not_consult_ambiguous_default_ranges(config, monkeypatch):
    device = ("cuda", "test GPU")
    _install_configs(
        config,
        monkeypatch,
        {
            config.ForwardConfigKey(
                "default", torch.bfloat16, 1, 32, 262144
            ): config.SoftcappedLogprobStrategy.ROW,
            config.ForwardConfigKey(
                "default", torch.bfloat16, 16, 64, 262144
            ): config.SoftcappedLogprobStrategy.PARALLEL,
            config.ForwardConfigKey(
                device, torch.bfloat16, 1, 64, 262144
            ): config.SoftcappedLogprobStrategy.ROW,
        },
    )
    assert config.select_softcapped_logprob_strategy(device, torch.bfloat16, 16, 262144) == ROW
    with pytest.raises(ValueError, match="overlapping forward strategy ranges"):
        config.select_softcapped_logprob_strategy(("rocm", "test GPU"), torch.bfloat16, 16, 262144)


def test_repeated_metadata_skips_configuration_matching(config, monkeypatch):
    scans = []

    class ObservedConfigs(dict):
        def items(self):
            scans.append(True)
            return super().items()

    _install_configs(config, monkeypatch, ObservedConfigs(config.FORWARD_STRATEGY_CONFIGS))
    args = (("cuda", "test GPU"), torch.bfloat16, 17, 262144)
    assert config.select_softcapped_logprob_strategy(*args) == PARALLEL
    first_scans = len(scans)
    assert first_scans > 0
    assert config.select_softcapped_logprob_strategy(*args) == PARALLEL
    assert len(scans) == first_scans
    assert config.select_softcapped_logprob_strategy.cache_info().hits == 1


@pytest.mark.parametrize(
    "different_metadata",
    [
        (("cuda", "other GPU"), torch.bfloat16, 16, 262144),
        (("cuda", "test GPU"), torch.float32, 16, 262144),
        (("cuda", "test GPU"), torch.bfloat16, 1025, 262144),
        (("cuda", "test GPU"), torch.bfloat16, 16, 262143),
    ],
)
def test_cache_distinguishes_each_metadata_field(config, monkeypatch, different_metadata):
    _install_configs(
        config,
        monkeypatch,
        {
            config.ForwardConfigKey(
                "default", torch.bfloat16, 1, 1024, 262144
            ): config.SoftcappedLogprobStrategy.PARALLEL,
            config.ForwardConfigKey(
                ("cuda", "other GPU"), torch.bfloat16, 1, 1024, 262144
            ): config.SoftcappedLogprobStrategy.ROW,
        },
    )
    select = config.select_softcapped_logprob_strategy
    assert select(("cuda", "test GPU"), torch.bfloat16, 16, 262144) == PARALLEL
    assert select(*different_metadata) == ROW
    assert select.cache_info().misses == 2


def test_clearing_cache_applies_a_replacement_policy(config, monkeypatch):
    select = config.select_softcapped_logprob_strategy
    args = (("cuda", "test GPU"), torch.bfloat16, 16, 262144)
    assert select(*args) == PARALLEL
    monkeypatch.setattr(config, "FORWARD_STRATEGY_CONFIGS", {})
    select.cache_clear()
    assert select(*args) == ROW


def test_selection_does_not_depend_on_grad_recording(config):
    args = (("cuda", "NVIDIA H100"), torch.bfloat16, 513, 262144)
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
    assert config.softcapped_logprob_device_key(torch.device("cuda:1")) == ("cuda", "another GPU")
    assert config.softcapped_logprob_device_key(torch.device("cuda:1")) == ("cuda", "another GPU")
    assert config.softcapped_logprob_device_key(torch.device("cuda")) == ("cuda", "NVIDIA H100")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert config.softcapped_logprob_device_key(torch.device("cuda")) == ("cuda", "another GPU")
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
    assert config.softcapped_logprob_device_key(torch.device("cuda:2")) == ("rocm", "AMD test GPU")


def test_other_backends_do_not_query_cuda(config, monkeypatch):
    def unexpected_query(*args):
        pytest.fail("A non-CUDA tensor must not query CUDA hardware")

    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_query)
    monkeypatch.setattr(torch.cuda, "current_device", unexpected_query)
    assert config.softcapped_logprob_device_key(torch.device("xpu:1")) == ("xpu", "")
