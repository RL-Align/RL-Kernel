from contextlib import contextmanager
from types import SimpleNamespace

from rl_engine.integrations.engines.rollout.vllm.memory import patch_weight_pool


def test_skipped_weight_pool_is_entered_and_restored():
    events = []

    @contextmanager
    def config_context():
        events.append("config-enter")
        yield
        events.append("config-exit")

    class Worker:
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=True))

        @contextmanager
        def _maybe_get_memory_pool_context(self, tag):
            assert tag == "weights"
            events.append("pool-enter")
            try:
                yield
            finally:
                events.append("pool-exit")

        def load_model(self, fail=False):
            with self._maybe_get_memory_pool_context(tag="weights") and config_context():
                events.append("load")
                if fail:
                    raise RuntimeError("load failed")
                return 42

    assert patch_weight_pool(Worker)
    assert not patch_weight_pool(Worker)
    assert Worker().load_model() == 42
    assert events == ["pool-enter", "config-enter", "load", "config-exit", "pool-exit"]
    events.clear()
    try:
        Worker().load_model(fail=True)
    except RuntimeError:
        pass
    assert events[-1] == "pool-exit"
    events.clear()
    Worker.vllm_config.model_config.enable_sleep_mode = False
    Worker().load_model()
    assert "pool-enter" not in events


def test_fixed_upstream_context_is_not_wrapped():
    class Worker:
        def load_model(self):
            with self._maybe_get_memory_pool_context(tag="weights"), self.config_context():
                return 42

    original = Worker.load_model
    assert not patch_weight_pool(Worker)
    assert Worker.load_model is original
