# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Each worker process installs into itself and leaves its own readback file."""

from __future__ import annotations

import multiprocessing

from rl_engine.integrations.orchestrators.miles.readback import audit_readbacks

_WORKERS = 3


def _install_in_worker(directory: str, case: str) -> None:
    from rl_engine.integrations.orchestrators.miles.config import MilesConfig
    from rl_engine.integrations.orchestrators.miles.install import install_from_config
    from rl_engine.runtime.plan import IntegrationPlan

    plan = IntegrationPlan.from_case_ids(attention=case, ffn="P/P", logp="P/P")
    install_from_config(
        MilesConfig(enabled=True, plan=plan, readback_dir=directory),
        attention_classes=[],
        ffn_classes=[],
    )


def _run_workers(directory: str, case: str, count: int) -> list:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_install_in_worker, args=(directory, case)) for _ in range(count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=600)
    return processes


def test_every_worker_writes_its_own_readback(tmp_path):
    processes = _run_workers(str(tmp_path), "P/P", _WORKERS)

    assert [process.exitcode for process in processes] == [0] * _WORKERS
    report = audit_readbacks(tmp_path, expected_workers=_WORKERS)
    assert report.healthy
    assert len({worker.pid for worker in report.workers}) == _WORKERS
    assert {worker.pid for worker in report.workers} == {process.pid for process in processes}


def test_missing_worker_is_detected(tmp_path):
    processes = _run_workers(str(tmp_path), "P/P", _WORKERS - 1)

    assert [process.exitcode for process in processes] == [0] * (_WORKERS - 1)
    report = audit_readbacks(tmp_path, expected_workers=_WORKERS)
    assert not report.healthy
    assert any("expected 3 worker" in problem for problem in report.problems)
