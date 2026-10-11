# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pytest plugin enforcing Qwen3-Next's declared mandatory case identities."""

import json
import time
from pathlib import Path

from rl_engine.validation.models.qwen3_next_cases import execution_verdict, load_cases


def pytest_addoption(parser):
    group = parser.getgroup("qwen3-next acceptance")
    group.addoption("--case-manifest", required=True)
    group.addoption("--case-suite", choices=("norm", "core", "all", "diagnostics"), required=True)
    group.addoption("--case-report", required=True)


def pytest_configure(config):
    plugin = CaseRecorder(
        load_cases(Path(config.getoption("--case-manifest")), config.getoption("--case-suite")),
        Path(config.getoption("--case-report")),
    )
    config.pluginmanager.register(plugin, "qwen3-next-case-recorder")


class CaseRecorder:
    def __init__(self, expected, output):
        self.expected = expected
        self.output = output
        self.collected = []
        self.outcomes = {}
        self.active = None

    def _progress(self, status="running"):
        """Observable progress is separate from the final acceptance artifact."""
        self.output.parent.mkdir(parents=True, exist_ok=True)
        target = self.output.with_suffix(".progress.json")
        partial = target.with_suffix(".partial")
        payload = json.dumps(
            {
                "status": status,
                "active_case": self.active,
                "observed_at": time.time(),
                "collected": len(self.collected),
                "required": len(self.expected),
                "finished_calls": sum("call" in phases for phases in self.outcomes.values()),
                "outcomes": self.outcomes,
            },
            indent=2,
        )
        partial.write_text(payload + "\n")
        partial.replace(target)

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]
        self._progress()

    def pytest_runtest_logstart(self, nodeid, location):
        self.active = nodeid
        self._progress()

    def pytest_runtest_logreport(self, report):
        outcome = "skipped" if hasattr(report, "wasxfail") else report.outcome
        self.outcomes.setdefault(report.nodeid, {})[report.when] = outcome
        self._progress()

    def pytest_sessionfinish(self, session, exitstatus):
        result = execution_verdict(self.expected, self.collected, self.outcomes)
        result["pytest_exit_code"] = int(exitstatus)
        result["passed"] = result["passed"] and exitstatus == 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        partial = self.output.with_suffix(self.output.suffix + ".partial")
        partial.write_text(json.dumps(result, indent=2) + "\n")
        partial.replace(self.output)
        self.active = None
        self._progress("complete")
        if not result["passed"] and exitstatus == 0:
            session.exitstatus = 1
