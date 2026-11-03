import importlib.util
from pathlib import Path


def load_benchmark():
    path = (
        Path(__file__).resolve().parents[1] / "benchmarks/benchmark_rmsnorm_residual.py"
    )
    spec = importlib.util.spec_from_file_location("rmsnorm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_loads_all_three_backends():
    assert list(load_benchmark().load_backends()) == ["torch-native", "triton", "cuda"]


def test_combined_measurement_keeps_dgamma_alive(monkeypatch):
    bench = load_benchmark()
    x, gamma, dy, dr, y, residual, saved, dx, dgamma = [object() for _ in range(9)]
    outputs = []

    def measure(fn, warmup, iterations):
        outputs.append(fn())
        return {"mean_ms": 1.0, "peak_extra_allocated_bytes": 0}

    monkeypatch.setattr(bench, "measure", measure)
    bench.benchmark_backend(
        lambda *args: (y, residual, saved),
        lambda *args: (dx, dgamma),
        (x, gamma, dy, dr),
        1,
        1,
    )
    assert outputs[-1] == (y, residual, dx, dgamma)


def test_measurement_fields_match_table(monkeypatch, capsys):
    bench = load_benchmark()

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 4.0

    monkeypatch.setattr(bench.torch.cuda, "Event", Event)
    monkeypatch.setattr(bench.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench.torch.cuda, "memory_allocated", lambda: 16)
    monkeypatch.setattr(bench.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(bench.torch.cuda, "max_memory_allocated", lambda: 48)
    result = bench.measure(lambda: None, warmup=1, iterations=2)
    assert result == {"mean_ms": 2.0, "peak_extra_allocated_bytes": 32}
    bench.print_table(
        [
            dict(
                T=1,
                backend="cuda",
                forward=result,
                backward=result,
                forward_backward=result,
            )
        ]
    )
    assert "| 1 | cuda |" in capsys.readouterr().out
