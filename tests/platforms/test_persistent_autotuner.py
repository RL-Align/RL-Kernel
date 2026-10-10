# SPDX-License-Identifier: Apache-2.0
import json
import multiprocessing
from pathlib import Path

import pytest

from rl_engine.platforms.autotuner import (
    PersistentAutotuner,
    TuneConfig,
    autotune_enabled,
    cache_key,
)

DEFAULT = TuneConfig()
FAST = TuneConfig((("num_warps", 8),))
CANDIDATES = (DEFAULT, FAST)


def _timing(config):
    return 2.0 if config == DEFAULT else 1.0


def _worker(directory, queue):
    result = PersistentAutotuner(Path(directory)).select({"op": "example"}, CANDIDATES, _timing)
    queue.put((result.config.to_dict(), result.source))


def test_memory_disk_and_key_isolation(tmp_path):
    tuner = PersistentAutotuner(tmp_path)
    sig = {"device": "gpu0", "shape": [4, 8], "dtype": "float32", "version": 1}
    first = tuner.select(sig, CANDIDATES, _timing)
    assert first.config == FAST and first.source == "tuned"

    def unexpected(_):
        pytest.fail("Cache hit must not benchmark")

    assert tuner.select(sig, CANDIDATES, unexpected).source == "memory"
    assert PersistentAutotuner(tmp_path).select(sig, CANDIDATES, unexpected).source == "disk"
    for field, value in [
        ("device", "gpu1"),
        ("shape", [8, 8]),
        ("dtype", "float16"),
        ("version", 2),
    ]:
        assert tuner.select({**sig, field: value}, CANDIDATES, _timing).source == "tuned"


def test_rejection_and_ties_keep_default(tmp_path):
    tuner = PersistentAutotuner(tmp_path)
    assert tuner.select({}, CANDIDATES, lambda c: 1 if c == DEFAULT else None).config == DEFAULT
    assert tuner.select({"tie": True}, CANDIDATES, lambda _: 1).config == DEFAULT
    with pytest.raises(RuntimeError, match="fixed"):
        tuner.select({"bad": True}, CANDIDATES, lambda _: None)
    with pytest.raises(ValueError, match="timing"):
        tuner.select({"nan": True}, CANDIDATES, lambda _: float("nan"))


@pytest.mark.parametrize(
    "contents", ["{bad json", "[]", '{"schema":0}', '{"schema":1,"config":{"num_warps":999}}']
)
def test_corrupt_cache_is_replaced(tmp_path, contents, caplog):
    path = tmp_path / f"{cache_key({}, CANDIDATES)}.json"
    path.write_text(contents)
    result = PersistentAutotuner(tmp_path).select({}, CANDIDATES, _timing)
    assert result.source == "tuned"
    assert json.loads(path.read_text())["config"] == FAST.to_dict()
    assert "Ignoring" in caplog.text


def test_unwritable_cache_and_no_identity(tmp_path, caplog):
    unavailable = tmp_path / "file"
    unavailable.write_text("not a directory")
    tuner = PersistentAutotuner(unavailable)
    assert tuner.select({}, CANDIDATES, _timing).config == FAST
    assert "using memory" in caplog.text
    other = tmp_path / "never-created"
    PersistentAutotuner(other).select({}, CANDIDATES, _timing, persistent=False)
    assert not other.exists()


def test_capture_does_not_access_disk_or_benchmark(tmp_path):
    tuner = PersistentAutotuner(tmp_path / "absent")

    def unexpected(_):
        pytest.fail("Capture cannot benchmark")

    assert tuner.select({}, CANDIDATES, unexpected, capture=True).source == "capture_default"
    assert not tuner.cache_dir.exists()
    tuner.select({}, CANDIDATES, _timing, persistent=False)
    assert tuner.select({}, CANDIDATES, unexpected, capture=True).config == FAST


def test_concurrent_processes_and_restart(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(str(tmp_path), queue)) for _ in range(3)]
    for worker in workers:
        worker.start()
    records = [queue.get(timeout=90) for _ in workers]
    for worker in workers:
        worker.join(timeout=90)
        assert worker.exitcode == 0
    assert all(config == FAST.to_dict() for config, _ in records)
    assert sorted(source for _, source in records) == ["disk", "disk", "tuned"]


@pytest.mark.parametrize("mode", ["strict", "audit", "auto", "off"])
@pytest.mark.parametrize("flag", ["0", "1"])
def test_opt_in_policy(monkeypatch, mode, flag):
    monkeypatch.setenv("RL_KERNEL_MODE", mode)
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", flag)
    assert autotune_enabled() == (mode == "auto" and flag == "1")


def test_programming_errors_propagate(tmp_path):
    def broken(_):
        raise OSError("kernel callback failed")

    with pytest.raises(OSError, match="kernel callback"):
        PersistentAutotuner(tmp_path).select({}, CANDIDATES, broken)
    assert not list(tmp_path.glob("*.json"))


def test_explicit_backend_dispatch_does_not_change_strict_routes(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    selected = []
    sentinel = object()

    def load(backend):
        selected.append(backend)
        return sentinel

    monkeypatch.setattr(registry, "_get_or_create_backend", load)
    for op, backend, expected in (
        ("det_gemm", "rlkernel.triton.tuned_gemm", OpBackend.TRITON_TUNED_GEMM),
        ("attn", "rlkernel.triton.tuned_attention", OpBackend.TRITON_TUNED_ATTENTION),
        ("logp", "rlkernel.triton.tuned_logp", OpBackend.TRITON_TUNED_LOGP),
    ):
        assert registry.get_op(op, device="cuda", backend=backend) is sentinel
        assert selected[-1] == expected
        registry.get_op(op, device="cuda")
        assert selected[-1] != expected
    with pytest.raises(ValueError, match="Unsupported"):
        registry.get_op("logp", device="cuda", backend="rlkernel.triton.tuned_gemm")
    with pytest.raises(ValueError, match="CUDA or ROCm"):
        registry.get_op("logp", device="cpu", backend="rlkernel.triton.tuned_logp")
