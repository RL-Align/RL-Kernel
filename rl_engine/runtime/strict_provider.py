# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fail-closed, single-worker manifest execution lifecycle.

This module imports neither engines nor integration adapters. It consumes the
trusted resolved-call protocol. An independent manifest validator checks
cross-worker provenance and coverage; the operator/model harness checks numerical
outputs. ``finish`` only seals this worker's execution evidence.

All binding/model/platform/execution fields participate in exact lock matching,
including extension fields. Observers put facts which are only diagnostic in
WorkerObservation.diagnostics, rather than adding unlocked binding fields.
"""

from __future__ import annotations

import json
from threading import RLock
from typing import Any, Mapping, Protocol
from uuid import uuid4

from rl_engine.contracts.strict_provider import (
    CallContext,
    KernelPlan,
    ProviderLock,
    SnapshotActivationReceipt,
    StrictProviderError,
    WorkerObservation,
    fingerprint,
)


class ResolvedDispatch(Protocol):
    """Interface supplied by a trusted observer of the actual dispatch."""

    @property
    def plan(self) -> KernelPlan: ...
    @property
    def runtime_id(self) -> str: ...
    @property
    def context_id(self) -> str: ...
    @property
    def snapshot_generation(self) -> int: ...
    @property
    def has_completion(self) -> bool: ...
    def claim(self, owner: str) -> None: ...
    def check_unchanged(self) -> None: ...
    def launch(self) -> Any: ...
    def wait_completed(self, output: Any) -> Mapping[str, Any]: ...


def _mismatch(expected: Any, observed: Any, path: str) -> tuple[str, Any, Any] | None:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        for key in expected:
            if key not in observed:
                return f"{path}.{key}", expected[key], "missing"
            difference = _mismatch(expected[key], observed[key], f"{path}.{key}")
            if difference is not None:
                return difference
        for key in observed:
            if key not in expected:
                return f"{path}.{key}", "absent", observed[key]
        return None
    if expected != observed:
        return path, expected, observed
    return None


class StrictProviderSession:
    """Own authorization, single-use launch and completion for one worker.

    A new session begins with inactive weights unless a worker-load receipt is
    supplied. ``bind`` may precede activation. Any rejected operation or failed
    completion makes the session terminal; a fresh session is needed to retry.
    ``finish`` returns evidence only, never a coverage or parity assertion.
    """

    def __init__(
        self,
        lock: ProviderLock,
        *,
        role: str,
        worker_id: str,
        run_id: str,
        attempt: str,
        weight_snapshot_id: str,
        rank: int = 0,
        snapshot_receipt: SnapshotActivationReceipt | None = None,
    ) -> None:
        if not isinstance(lock, ProviderLock):
            raise TypeError("lock must be a ProviderLock")
        for name, value in (
            ("role", role),
            ("worker_id", worker_id),
            ("run_id", run_id),
            ("attempt", attempt),
            ("weight_snapshot_id", weight_snapshot_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        if lock.execution["graph_mode"] != "eager":
            raise StrictProviderError(
                "unsupported_graph",
                "Initial strict session supports eager execution only",
                field="execution.graph_mode",
                expected="eager",
                observed=lock.execution["graph_mode"],
            )
        self.lock = lock
        self.role = role
        self.worker_id = worker_id
        self.run_id = run_id
        self.attempt = attempt
        self.rank = rank
        self.weight_snapshot_id = weight_snapshot_id
        self.snapshot_generation = 0
        self._active_snapshot = False
        self._snapshot_history: list[dict[str, Any]] = []
        self._state = "open"
        self._session_id = uuid4().hex
        self._bindings: dict[str, dict[str, Any]] = {}
        self._observations: dict[str, dict[str, Any]] = {}
        self._calls: list[dict[str, Any]] = []
        self._call_ids: set[tuple[Any, ...]] = set()
        self._inflight = False
        self._mutex = RLock()
        if snapshot_receipt is not None:
            self.activate_snapshot(snapshot_receipt)

    @property
    def state(self) -> str:
        return self._state

    def _require_open(self) -> None:
        if self._state != "open":
            raise StrictProviderError(
                "session_terminal",
                "Session no longer accepts operations",
                field="session.state",
                expected="open",
                observed=self._state,
            )
        if self._inflight:
            raise StrictProviderError("call_inflight", "Reentrant session operations are forbidden")

    def _reject(
        self,
        code: str,
        message: str,
        field: str | None = None,
        expected: Any = None,
        observed: Any = None,
    ) -> None:
        raise StrictProviderError(code, message, field=field, expected=expected, observed=observed)

    def _equal(self, expected: Any, observed: Any, path: str, code: str) -> None:
        mismatch = _mismatch(expected, observed, path)
        if mismatch is not None:
            field, wanted, actual = mismatch
            self._reject(code, "Observed execution does not match the lock", field, wanted, actual)

    def bind(self, node: str, role: str, observation: WorkerObservation) -> None:
        with self._mutex:
            try:
                self._require_open()
                if not isinstance(observation, WorkerObservation):
                    self._reject("binding_mismatch", "Expected a WorkerObservation")
                actual = observation.to_dict()
                self._equal(self.role, role, "role", "binding_mismatch")
                self._equal(self.role, actual["role"], "observation.role", "binding_mismatch")
                self._equal(
                    self.worker_id, actual["worker_id"], "observation.worker_id", "binding_mismatch"
                )
                self._equal(
                    self.rank, actual.get("rank", 0), "observation.rank", "binding_mismatch"
                )
                lock = self.lock.to_dict()
                for name in ("model", "platform", "execution"):
                    self._equal(lock[name], actual[name], f"observation.{name}", "binding_mismatch")
                if node in self._bindings:
                    self._reject(
                        "binding_reused", "Node already has a bound provider", "node", None, node
                    )
                expected = lock["nodes"].get(node, {}).get("bindings", {}).get(role)
                observed = actual["bindings"].get(node)
                if expected is None or observed is None:
                    self._reject(
                        "binding_mismatch", "Missing node/role binding", f"nodes.{node}.{role}"
                    )
                self._equal(expected, observed, f"bindings.{node}", "binding_mismatch")
                if "weight_snapshot_id" in actual:
                    self._equal(
                        self.weight_snapshot_id,
                        actual["weight_snapshot_id"],
                        "observation.weight_snapshot_id",
                        "binding_mismatch",
                    )
                self._bindings[node] = observed
                self._observations[node] = actual
            except BaseException:
                self._state = "failed"
                raise

    def begin_weight_update(self, snapshot_id: str) -> int:
        """Invalidate old authorization before any in-memory weight mutation."""
        with self._mutex:
            try:
                self._require_open()
                if not self._active_snapshot:
                    self._reject("snapshot_inactive", "Previous weight load/update is not complete")
                if not isinstance(snapshot_id, str) or not snapshot_id:
                    self._reject("snapshot_inactive", "A nonempty target snapshot is required")
                self._active_snapshot = False
                self.snapshot_generation += 1
                self.weight_snapshot_id = snapshot_id
                return self.snapshot_generation
            except BaseException:
                self._state = "failed"
                raise

    def activate_snapshot(self, receipt: SnapshotActivationReceipt) -> None:
        """Activate only completed worker-load and derived-cache refresh evidence."""
        with self._mutex:
            try:
                self._require_open()
                if self._active_snapshot or not isinstance(receipt, SnapshotActivationReceipt):
                    self._reject("snapshot_inactive", "Expected a new worker-load receipt")
                receipt.validate()
                for name, expected in (
                    ("snapshot_id", self.weight_snapshot_id),
                    ("generation", self.snapshot_generation),
                    ("worker_id", self.worker_id),
                    ("evidence_kind", "worker_load"),
                ):
                    self._equal(
                        expected, getattr(receipt, name), f"snapshot.{name}", "snapshot_inactive"
                    )
                if (
                    not receipt.completed
                    or not receipt.load_complete
                    or not receipt.cache_refresh_complete
                ):
                    self._reject(
                        "snapshot_inactive", "Weight loading and required refresh must complete"
                    )
                self._active_snapshot = True
                self._snapshot_history.append(receipt.to_dict())
            except BaseException:
                self._state = "failed"
                raise

    def execute(self, resolved_call: ResolvedDispatch, call_context: CallContext) -> Any:
        with self._mutex:
            record: dict[str, Any] | None = None
            try:
                self._require_open()
                # Consume before validating: a rejected dispatch cannot be retried elsewhere.
                resolved_call.claim(self._session_id)
                if not isinstance(call_context, CallContext):
                    self._reject("context_mismatch", "Expected a CallContext")
                context = call_context.to_dict()
                for name, expected in (
                    ("role", self.role),
                    ("worker_id", self.worker_id),
                    ("run_id", self.run_id),
                    ("attempt", self.attempt),
                    ("weight_snapshot_id", self.weight_snapshot_id),
                ):
                    self._equal(expected, context[name], f"context.{name}", "context_mismatch")
                if not self._active_snapshot:
                    self._reject(
                        "snapshot_inactive", "Weight snapshot has no completed activation receipt"
                    )
                self._equal(
                    fingerprint(context),
                    resolved_call.context_id,
                    "resolved_call.context_id",
                    "context_mismatch",
                )
                self._equal(
                    self.snapshot_generation,
                    resolved_call.snapshot_generation,
                    "resolved_call.snapshot_generation",
                    "snapshot_inactive",
                )
                call_key = tuple(
                    context[key] for key in ("case_id", "node", "phase", "layer_id", "call_id")
                )
                if call_key in self._call_ids:
                    self._reject("call_reused", "Duplicate logical call ID", "context.call_id")
                self._call_ids.add(call_key)
                plan = resolved_call.plan
                if not isinstance(plan, KernelPlan):
                    self._reject("unobservable_plan", "Observer must supply a validated KernelPlan")
                data = plan.to_dict()
                requires_completion = self.lock.platform["kind"] in ("rocm", "cuda") or (
                    data["launch"].get("device_kind") in ("rocm", "cuda")
                )
                if requires_completion and not resolved_call.has_completion:
                    self._reject(
                        "unobservable_completion",
                        "GPU calls require a completion/error check",
                        "resolved_call.has_completion",
                        True,
                        False,
                    )
                node = context["node"]
                if node not in self._bindings:
                    self._reject(
                        "unbound_provider", "Node has not passed bind", "context.node", None, node
                    )
                self._equal(node, data["node"], "plan.node", "plan_mismatch")
                self._equal(context["phase"], data["phase"], "plan.phase", "plan_mismatch")
                self._equal(
                    self._bindings[node]["implementation"],
                    data["implementation"],
                    "plan.implementation",
                    "plan_mismatch",
                )
                if data["launch"]["graph_state"] != "eager":
                    self._reject(
                        "unsupported_graph",
                        "Initial strict session supports eager execution only",
                        "plan.launch.graph_state",
                        "eager",
                        data["launch"]["graph_state"],
                    )
                allowed = self.lock.to_dict()["nodes"][node]["allowed_plans"].get(self.role, [])
                if plan.plan_id not in allowed:
                    self._reject(
                        "plan_mismatch",
                        "Execution plan is not sealed in the lock",
                        "plan.plan_id",
                        allowed,
                        plan.plan_id,
                    )
                record = {
                    "context": context,
                    "plan": data,
                    "plan_id": plan.plan_id,
                    "alignment_id": plan.alignment_id,
                    "runtime_id": resolved_call.runtime_id,
                    "context_id": resolved_call.context_id,
                    "call_ticket_id": fingerprint(
                        {
                            "session_id": self._session_id,
                            "context_id": resolved_call.context_id,
                            "nonce": uuid4().hex,
                        }
                    ),
                    "snapshot_generation": self.snapshot_generation,
                    "status": "prepared",
                    "states": ["prepared"],
                }
                self._calls.append(record)
                resolved_call.check_unchanged()
                record["states"].append("authorized")
                record["status"] = "authorized"
                # Recheck immediately before the trusted dispatch; no provider reselection.
                resolved_call.check_unchanged()
                self._inflight = True
                record["states"].append("launched")
                record["status"] = "launched"
                output = resolved_call.launch()
                completion = dict(resolved_call.wait_completed(output))
                if completion.get("status") != "completed":
                    self._reject("incomplete_call", "Provider completion is pending or failed")
                if requires_completion and completion.get("kind") in (None, "", "synchronous"):
                    self._reject(
                        "unobservable_completion", "GPU completion must report its explicit check"
                    )
                if self._state != "open":
                    self._reject("session_terminal", "Session failed during provider execution")
                record["completion"] = completion
                record["states"].append("completed")
                record["status"] = "completed"
                return output
            except BaseException:
                if record is not None:
                    record["status"] = "failed"
                    record["states"].append("failed")
                self._state = "failed"
                raise
            finally:
                self._inflight = False

    def finish(self) -> dict[str, Any]:
        """Seal a single-worker evidence summary, without declaring acceptance."""
        with self._mutex:
            try:
                self._require_open()
                if not self._active_snapshot:
                    self._reject("snapshot_inactive", "Cannot finish during a weight update")
                if any(call["status"] != "completed" for call in self._calls):
                    self._reject("incomplete_calls", "All submitted calls must complete")
                self._state = "finished"
                summary = {
                    "schema_version": "rl_kernel.strict_provider.summary.v1",
                    "lock_id": self.lock.lock_id,
                    "run_id": self.run_id,
                    "attempt": self.attempt,
                    "worker_id": self.worker_id,
                    "role": self.role,
                    "rank": self.rank,
                    "weight_snapshot_id": self.weight_snapshot_id,
                    "snapshot_generation": self.snapshot_generation,
                    "snapshot_history": self._snapshot_history,
                    "bindings": self._bindings,
                    "observations": self._observations,
                    "calls": self._calls,
                    "state": "finished",
                    "footer": {"complete": True, "call_count": len(self._calls)},
                }
                summary["summary_id"] = fingerprint(summary)
                # Return an independent JSON copy, not mutable session internals.
                return json.loads(json.dumps(summary))
            except BaseException:
                self._state = "failed"
                raise
