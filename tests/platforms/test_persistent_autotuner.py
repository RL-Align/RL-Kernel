# SPDX-License-Identifier: Apache-2.0
import json
import multiprocessing
import os
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
    """Fake timing: the default launch takes 2 ms and every other candidate 1 ms."""
    return 2.0 if config == DEFAULT else 1.0


def _worker(directory, queue):
    """Select in a child process and put the chosen config and its source on ``queue``."""
    result = PersistentAutotuner(Path(directory)).select({"op": "example"}, CANDIDATES, _timing)
    queue.put((result.config.to_dict(), result.source))


def test_memory_disk_and_key_isolation(tmp_path):
    """Hits come from memory, then disk; changing any signature field tunes again."""
    tuner = PersistentAutotuner(tmp_path)
    sig = {"device": "gpu0", "shape": [4, 8], "dtype": "float32", "version": 1}
    first = tuner.select(sig, CANDIDATES, _timing)
    assert first.config == FAST and first.source == "tuned"

    def unexpected(_):
        """Fail if a cache hit benchmarks a candidate."""
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
    """Rejections and ties keep the default; a rejected default or NaN timing raises."""
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
    """Corrupt or incompatible cache files are ignored, logged and rewritten."""
    path = tmp_path / f"{cache_key({}, CANDIDATES)}.json"
    path.write_text(contents)
    result = PersistentAutotuner(tmp_path).select({}, CANDIDATES, _timing)
    assert result.source == "tuned"
    assert json.loads(path.read_text())["config"] == FAST.to_dict()
    assert "Ignoring" in caplog.text


def test_unwritable_cache_and_no_identity(tmp_path, caplog):
    """An unusable cache path falls back to memory; nonpersistent mode creates nothing."""
    unavailable = tmp_path / "file"
    unavailable.write_text("not a directory")
    tuner = PersistentAutotuner(unavailable)
    assert tuner.select({}, CANDIDATES, _timing).config == FAST
    assert "using memory" in caplog.text
    other = tmp_path / "never-created"
    PersistentAutotuner(other).select({}, CANDIDATES, _timing, persistent=False)
    assert not other.exists()


def test_capture_does_not_access_disk_or_benchmark(tmp_path):
    """During capture, selection returns the default or a memory hit, never timing."""
    tuner = PersistentAutotuner(tmp_path / "absent")

    def unexpected(_):
        """Fail if graph capture benchmarks a candidate."""
        pytest.fail("Capture cannot benchmark")

    assert tuner.select({}, CANDIDATES, unexpected, capture=True).source == "capture_default"
    assert not tuner.cache_dir.exists()
    tuner.select({}, CANDIDATES, _timing, persistent=False)
    assert tuner.select({}, CANDIDATES, unexpected, capture=True).config == FAST


def test_concurrent_processes_and_restart(tmp_path):
    """Three concurrent processes agree; one tunes and the others read it from disk."""
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
    """Tuning is on only for ``RL_KERNEL_MODE=auto`` with ``RL_KERNEL_AUTOTUNE=1``."""
    monkeypatch.setenv("RL_KERNEL_MODE", mode)
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", flag)
    assert autotune_enabled() == (mode == "auto" and flag == "1")


def test_programming_errors_propagate(tmp_path):
    """Exceptions from ``evaluate`` propagate and leave no cache record behind."""

    def broken(_):
        """Simulate a kernel callback that raises ``OSError``."""
        raise OSError("kernel callback failed")

    with pytest.raises(OSError, match="kernel callback"):
        PersistentAutotuner(tmp_path).select({}, CANDIDATES, broken)
    assert not list(tmp_path.glob("*.json"))


def test_explicit_backend_dispatch_does_not_change_strict_routes(monkeypatch):
    """Explicit tuned IDs resolve; default routes and mismatched or CPU requests do not."""
    from rl_engine.runtime.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    selected = []
    sentinel = object()

    def load(backend):
        """Record the requested backend and return a sentinel."""
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

    class Drifted:
        """An op whose own backend_id no longer matches the ID it was requested by."""

        backend_id = "rlkernel.triton.renamed_logp"

    monkeypatch.setattr(registry, "_get_or_create_backend", lambda backend: Drifted())
    with pytest.raises(RuntimeError, match="resolved to 'rlkernel.triton.renamed_logp'"):
        registry.get_op("logp", device="cuda", backend="rlkernel.triton.tuned_logp")


def test_explicit_backend_ids_match_the_tuned_classes():
    """Each explicit ID routes to the class whose own ``backend_id`` is that ID."""
    pytest.importorskip("triton")
    import importlib

    from rl_engine.runtime.registry import OpBackend

    for backend_id, member in (
        ("rlkernel.triton.tuned_gemm", OpBackend.TRITON_TUNED_GEMM),
        ("rlkernel.triton.tuned_attention", OpBackend.TRITON_TUNED_ATTENTION),
        ("rlkernel.triton.tuned_logp", OpBackend.TRITON_TUNED_LOGP),
    ):
        module, _, name = member.value.rpartition(".")
        assert getattr(importlib.import_module(module), name).backend_id == backend_id


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership checks only")
def test_created_cache_directory_is_private(tmp_path):
    """A newly created cache directory gets mode 0o700 and stores the selection."""
    directory = tmp_path / "new" / "cache"
    assert PersistentAutotuner(directory).select({}, CANDIDATES, _timing).source == "tuned"
    assert directory.stat().st_mode & 0o777 == 0o700
    assert list(directory.glob("*.json"))


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership checks only")
@pytest.mark.parametrize("shared", ["group_writable", "foreign_owner"])
def test_untrusted_cache_directory_falls_back_to_memory(tmp_path, monkeypatch, shared):
    """A group-writable or foreign-owned cache directory is never read or written."""
    directory = tmp_path / "shared"
    directory.mkdir()
    if shared == "group_writable":
        directory.chmod(0o770)
    else:
        uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    first = PersistentAutotuner(directory).select({}, CANDIDATES, _timing)
    assert first.config == FAST and first.source == "tuned"
    assert not list(directory.iterdir())
    # A fresh process cannot reuse an untrusted record; it tunes again.
    assert PersistentAutotuner(directory).select({}, CANDIDATES, _timing).source == "tuned"


def _stub_kernel_body(x, out):
    """Stand-in for a Triton kernel body; only its module is used to hash the revision."""


def test_output_passed_by_keyword_launches_the_fixed_kernel(monkeypatch):
    """A launch whose declared output is not positional must not index past ``args``."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton import autotuning

    calls = []

    class Jit:
        fn = staticmethod(_stub_kernel_body)

        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, args, kwargs))

    monkeypatch.setattr(autotuning, "autotune_enabled", lambda: True)
    kernel = autotuning.TunableKernel(Jit(), outputs=(1,))
    x, out = torch.zeros(4), torch.zeros(4)
    kernel[(1,)](x, out=out)
    assert calls == [((1,), (x,), {"out": out})]


def test_launch_skips_tuning_when_profiling_copies_do_not_fit(monkeypatch, tmp_path):
    """Too little free memory for the private output copies launches the fixed kernel untuned."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm required")
    from rl_engine.backends.shared.triton import autotuning

    calls = []

    class Jit:
        fn = staticmethod(_stub_kernel_body)

        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, args, kwargs))

    monkeypatch.setattr(autotuning, "autotune_enabled", lambda: True)
    monkeypatch.setattr(autotuning, "_TUNER", PersistentAutotuner(tmp_path))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1024, 1 << 40))
    monkeypatch.setattr(
        autotuning, "_profile_candidates", lambda *a, **kw: pytest.fail("profiled without room")
    )
    kernel = autotuning.TunableKernel(Jit(), outputs=(1,))
    x = torch.zeros(4, device="cuda")
    out = torch.zeros(1 << 26, dtype=torch.uint8, device="cuda")
    kernel[(1,)](x, out)
    assert calls == [((1,), (x, out), {})]
    assert not list(tmp_path.glob("*.json"))
    assert not autotuning._TUNER._memory


def _oom_during_graph_capture(*args, **kwargs):
    """Stand-in for ``torch.cuda.graph`` whose capture allocation runs out of memory."""
    import torch

    raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected during graph capture)")


def test_launch_falls_back_when_profiling_runs_out_of_memory(monkeypatch, tmp_path, caplog):
    """An OOM inside profiling, after the estimate passed, launches the fixed kernel untuned.

    The real ``_profile_candidates`` runs: the private copies are made and every candidate is
    validated, then the timing-graph capture runs out of memory. The call must still produce the
    fixed launch on the caller's buffers, cache and persist nothing, and let a later call tune.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm required")
    from rl_engine.backends.shared.triton import autotuning

    calls = []

    class Jit:
        """Records every launch: the profiling launches and the final fixed launch."""

        fn = staticmethod(_stub_kernel_body)

        def __getitem__(self, grid):
            """Return a launcher that records its grid and arguments."""
            return lambda *args, **kwargs: calls.append((grid, args, kwargs))

    monkeypatch.setattr(autotuning, "autotune_enabled", lambda: True)
    monkeypatch.setattr(autotuning, "_TUNER", PersistentAutotuner(tmp_path))
    monkeypatch.setattr(autotuning, "_PROFILING_OOM_WARNED", set(), raising=False)
    monkeypatch.setattr(torch.cuda, "graph", _oom_during_graph_capture)
    kernel = autotuning.TunableKernel(Jit(), outputs=(1,))
    x, out = torch.zeros(4, device="cuda"), torch.zeros(4, device="cuda")

    with caplog.at_level("WARNING", logger=autotuning.__name__):
        kernel[(1,)](x, out)
        kernel[(1,)](x, out)
    # Profiling did start (reference + one validation launch per candidate) before the OOM.
    profiled = [call for call in calls if call[1][1] is not out]
    assert len(profiled) >= 1 + len(autotuning.launch_candidates(kernel))
    # Each call ends with the fixed launch on the caller's own buffers.
    fixed = [call for call in calls if call[1][1] is out]
    assert fixed == [((1,), (x, out), {}), ((1,), (x, out), {})]
    assert calls[-1] == fixed[-1]
    assert not list(tmp_path.glob("*.json"))
    assert not autotuning._TUNER._memory
    assert not kernel._launches
    assert len([r for r in caplog.records if "ran out of memory" in r.getMessage()]) == 1

    # No negative result was cached: once profiling succeeds, the same launch is tuned.
    candidates = autotuning.launch_candidates(kernel)
    timings = {config: 2.0 for config in candidates}
    timings[candidates[0]] = 1.0
    monkeypatch.setattr(autotuning, "_profile_candidates", lambda *a, **kw: timings)
    state = autotuning.TuneState(True)
    with autotuning.tuning_scope(state):
        kernel[(1,)](x, out)
    assert [record["source"] for record in state.records] == ["tuned"]
    assert kernel._launches


def test_launch_propagates_profiling_errors_that_are_not_out_of_memory(monkeypatch, tmp_path):
    """Only recoverable allocation failures fall back; other profiling errors still surface."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm required")
    from rl_engine.backends.shared.triton import autotuning

    class Jit:
        """A kernel whose launches do nothing."""

        fn = staticmethod(_stub_kernel_body)

        def __getitem__(self, grid):
            """Return a launcher that does nothing."""
            return lambda *args, **kwargs: None

    def broken(*args, **kwargs):
        """Stand-in for ``torch.cuda.graph`` that fails for a reason other than memory."""
        raise RuntimeError("graph capture failed")

    monkeypatch.setattr(autotuning, "autotune_enabled", lambda: True)
    monkeypatch.setattr(autotuning, "_TUNER", PersistentAutotuner(tmp_path))
    monkeypatch.setattr(torch.cuda, "graph", broken)
    kernel = autotuning.TunableKernel(Jit(), outputs=(1,))
    x, out = torch.zeros(4, device="cuda"), torch.zeros(4, device="cuda")
    with pytest.raises(RuntimeError, match="graph capture failed"):
        kernel[(1,)](x, out)
    assert not autotuning._TUNER._memory
    assert not kernel._launches


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), object(), [1, 2]])
def test_launch_with_an_unhashable_argument_runs_the_fixed_kernel(monkeypatch, tmp_path, scale):
    """A launch argument the cache key cannot encode must not fail the call or be tuned."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm required")
    from rl_engine.backends.shared.triton import autotuning

    calls = []

    class Jit:
        fn = staticmethod(_stub_kernel_body)

        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, args, kwargs))

    monkeypatch.setattr(autotuning, "autotune_enabled", lambda: True)
    monkeypatch.setattr(autotuning, "_TUNER", PersistentAutotuner(tmp_path))
    monkeypatch.setattr(
        autotuning,
        "_profile_candidates",
        lambda *a, **kw: pytest.fail("profiled an unkeyable launch"),
    )
    kernel = autotuning.TunableKernel(Jit(), outputs=(1,))
    x, out = torch.zeros(4, device="cuda"), torch.zeros(4, device="cuda")
    kernel[(1,)](x, out, scale)
    assert calls == [((1,), (x, out, scale), {})]
    assert not list(tmp_path.glob("*.json"))


def test_backward_without_a_pinned_forward_uses_the_environment_policy(monkeypatch):
    """Calling ``backward`` on a context that never ran the pinned forward must not crash."""
    pytest.importorskip("triton")
    from types import SimpleNamespace

    from rl_engine.backends.shared.triton import autotuning

    seen = []

    class Fn:
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, g):
            state = autotuning._STATE.get()
            seen.append((state.enabled, state.phase))
            return g

    autotuning.pin_autotune(Fn)
    for enabled in (True, False):
        monkeypatch.setattr(autotuning, "autotune_enabled", lambda enabled=enabled: enabled)
        assert Fn.backward(SimpleNamespace(), 1) == 1
        assert Fn.backward(SimpleNamespace(autotune_state=None), 1) == 1
    assert seen == [
        (True, "backward"),
        (True, "backward"),
        (False, "backward"),
        (False, "backward"),
    ]
