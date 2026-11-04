# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Trusted adapter boundary between actual provider dispatch and manifest checks.

An observer must resolve the real branch once, capture the dispatch arguments,
and expose the same selection through ``plan`` and ``launch``. A model forward
hook which guesses a branch from configuration does not satisfy this protocol.
These checks catch integration mistakes; they do not sandbox Python callbacks.
Operator-specific state layouts and metadata extraction belong to the adapter.
The common wrapper accepts the resolved plan and callbacks for any operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Mapping, Protocol

from rl_engine.contracts.strict_provider import (
    CallContext,
    KernelPlan,
    StrictProviderError,
    WorkerObservation,
    fingerprint,
)


@dataclass(frozen=True)
class CompletionReceipt:
    """Result of checking completion, including asynchronous provider errors.

    An asynchronous adapter must synchronize/check its event or future before
    returning ``completed=True``. Merely submitting a GPU launch is insufficient.
    """

    completed: bool
    kind: str = "callback"


class ResolvedCall:
    """A single-use resolved dispatch with preparation-time metadata checks.

    ``recheck`` returns JSON facts including every mutable argument/selection
    which could change the dispatch. The adapter must also validate shape-domain
    support and bind ``launch`` to the resolved configuration, with no internal
    unobserved reselection. Tensor values need not be hashed in the hot path.

    With no completion callback the adapter explicitly declares a synchronous
    operation. GPU adapters must supply a completion callback. The wrapper
    cannot infer stream semantics or inspect arbitrary native code.
    """

    __slots__ = (
        "_plan",
        "_launch_callback",
        "_recheck_callback",
        "_completion_callback",
        "_runtime_id",
        "_context_id",
        "_snapshot_generation",
        "_owner",
        "_claimed",
        "_lock",
    )

    def __init__(
        self,
        plan: KernelPlan,
        launch: Callable[[], Any],
        *,
        call_context: CallContext,
        snapshot_generation: int = 0,
        recheck: Callable[[], Mapping[str, Any]],
        completion: Callable[[Any], CompletionReceipt] | None = None,
    ) -> None:
        if not isinstance(plan, KernelPlan):
            raise TypeError("plan must be a KernelPlan")
        if not isinstance(call_context, CallContext):
            raise TypeError("call_context must be a CallContext")
        if type(snapshot_generation) is not int or snapshot_generation < 0:
            raise ValueError("snapshot_generation must be a nonnegative integer")
        if not callable(launch) or not callable(recheck):
            raise TypeError("launch and recheck must be callable")
        if completion is not None and not callable(completion):
            raise TypeError("completion must be callable")
        self._plan = plan
        self._launch_callback = launch
        self._recheck_callback = recheck
        self._completion_callback = completion
        self._runtime_id = fingerprint(recheck())
        self._context_id = fingerprint(call_context.to_dict())
        self._snapshot_generation = snapshot_generation
        self._owner: str | None = None
        self._claimed = False
        self._lock = Lock()

    @property
    def plan(self) -> KernelPlan:
        return self._plan

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def context_id(self) -> str:
        return self._context_id

    @property
    def snapshot_generation(self) -> int:
        return self._snapshot_generation

    @property
    def has_completion(self) -> bool:
        return self._completion_callback is not None

    def claim(self, owner: str) -> None:
        """Consume this dispatch exactly once, even if authorization later fails."""
        with self._lock:
            if self._claimed:
                raise StrictProviderError(
                    "call_reused",
                    "Resolved call was already consumed",
                    field="resolved_call.owner",
                    expected=owner,
                    observed=self._owner,
                )
            self._claimed = True
            self._owner = owner

    def check_unchanged(self) -> None:
        observed = fingerprint(self._recheck_callback())
        if observed != self._runtime_id:
            raise StrictProviderError(
                "metadata_changed",
                "Dispatch metadata changed after preparation",
                field="resolved_call.runtime_id",
                expected=self._runtime_id,
                observed=observed,
            )

    def launch(self) -> Any:
        return self._launch_callback()

    def wait_completed(self, output: Any) -> Mapping[str, Any]:
        if self._completion_callback is None:
            return {"kind": "synchronous", "status": "completed"}
        receipt = self._completion_callback(output)
        if not isinstance(receipt, CompletionReceipt) or receipt.completed is not True:
            raise StrictProviderError(
                "incomplete_call",
                "Provider did not confirm completed execution",
                field="completion",
                expected="completed",
                observed="pending_or_invalid",
            )
        return {"kind": receipt.kind, "status": "completed"}


class ProviderObserver(Protocol):
    """Implemented by engine adapters without selecting a second backend."""

    def observe_bound_provider(self, provider: Any) -> WorkerObservation:
        """Read actual loaded implementation and artifact identities."""
        ...

    def prepare_call(self, provider: Any, inputs: Any, metadata: Any) -> ResolvedCall:
        """Resolve supported real dispatch, preserving state/stream ownership."""
        ...
