# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Host-side strategy checks; importing the operator requires a Triton runtime."""

import pytest
import torch

ROW = "row"
PARALLEL = "parallel"


@pytest.fixture(scope="module")
def config():
    if not torch.cuda.is_available():
        triton = pytest.importorskip("triton")
        if not hasattr(triton, "jit"):
            pytest.skip("A Triton runtime is required; type stubs alone cannot import the operator")
    # A GPU host must fail, not skip, if its Triton backend cannot be imported.
    from rl_engine.kernels.ops.triton.loss import softcapped_selected_logprob

    return softcapped_selected_logprob


@pytest.mark.parametrize(
    "dtype, rows, vocab, expected",
    [
        (torch.bfloat16, 16, 262144, PARALLEL),
        (torch.float16, 256, 28672, PARALLEL),
        (torch.float32, 64, 32768, PARALLEL),
        (torch.float32, 256, 32768, ROW),
        (torch.float32, 256, 49152, PARALLEL),
        (torch.float32, 256, 65536, ROW),  # Do not assume a monotonic crossover.
        (torch.float32, 256, 262144, ROW),  # Only ~1.03x: below the speedup screen.
        (torch.float16, 4, 32769, ROW),  # Independent runs disagreed.
        (torch.bfloat16, 16, 49152, PARALLEL),
        (torch.float32, 16, 49152, ROW),  # Same shape, different precision/noise.
        (torch.float32, 1, 262144, ROW),  # Large median gain, but run drift failed.
        (torch.bfloat16, 16, 16384, ROW),
    ],
)
def test_strategy_preserves_measured_winners_and_uncertain_fallbacks(
    config, dtype, rows, vocab, expected
):
    assert (
        config.select_softcapped_logprob_strategy(("cuda", "NVIDIA A40"), dtype, rows, vocab)
        == expected
    )


@pytest.mark.parametrize("rows, vocab", [(0, 262144), (8, 262144), (16, 262143), (16, 262145)])
def test_unmeasured_shapes_are_not_interpolated(config, rows, vocab):
    assert (
        config.select_softcapped_logprob_strategy(
            ("cuda", "NVIDIA A40"), torch.bfloat16, rows, vocab
        )
        == ROW
    )


@pytest.mark.parametrize(
    "device", [("cuda", "unlisted GPU"), ("rocm", "unlisted GPU"), ("xpu", "")]
)
def test_unlisted_devices_reuse_the_default_policy(config, device):
    assert config.select_softcapped_logprob_strategy(device, torch.bfloat16, 16, 262144) == PARALLEL
    assert config.select_softcapped_logprob_strategy(device, torch.bfloat16, 16, 1025) == ROW


def test_device_override_precedes_default_without_hiding_other_default_entries(config, monkeypatch):
    key = ("cuda", "test GPU")
    monkeypatch.setitem(
        config.FORWARD_STRATEGY_CONFIGS,
        config.ForwardConfigKey(key, torch.bfloat16, 16, 262144),
        config.SoftcappedLogprobStrategy.ROW,
    )
    assert config.select_softcapped_logprob_strategy(key, torch.bfloat16, 16, 262144) == ROW
    assert config.select_softcapped_logprob_strategy(key, torch.bfloat16, 16, 65536) == PARALLEL
    # A model-name match alone must not select a CUDA-specific override on ROCm.
    assert (
        config.select_softcapped_logprob_strategy(("rocm", "test GPU"), torch.bfloat16, 16, 262144)
        == PARALLEL
    )


def test_selection_does_not_depend_on_grad_recording(config):
    args = (("cuda", "NVIDIA A40"), torch.bfloat16, 16, 262144)
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
        return {0: "NVIDIA A40", 1: "another GPU"}[index]

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_name", name)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert config.softcapped_logprob_device_key(torch.device("cuda:1")) == ("cuda", "another GPU")
    assert config.softcapped_logprob_device_key(torch.device("cuda:1")) == ("cuda", "another GPU")
    assert config.softcapped_logprob_device_key(torch.device("cuda")) == ("cuda", "NVIDIA A40")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert config.softcapped_logprob_device_key(torch.device("cuda")) == ("cuda", "another GPU")
    assert calls == [1, 0]


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
