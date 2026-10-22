# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict operator trace for MiniMax-H3 (RFC #420 ``h3_operator_trace``).

A strict run is only evidence if one can say, per graph node, *what actually ran*.
``OperatorTrace`` is that ledger: one ``OperatorCall`` per executed node of the
fingerprinted graph (``fingerprint.build_graph``), carrying the owning RFC row,
how the node is bound (``row`` / ``interim`` / ``reference``), the requested and
the actual backend, the implementation identity, dtype, accumulator, split policy,
topology and fallback state.

The trace is checked, not just stored:

* ``coverage`` -- every graph node ran exactly ``forwards`` times, nothing outside the graph ran;
* ``violations`` / ``assert_strict`` -- fail-closed gate: non-row bindings, fallbacks,
  requested != actual backend, a reduction row without an FP32 accumulator, any split
  policy other than ``none``, and missing/extra nodes are all reported;
* ``digest`` -- stable hash of the semantic content, so two runs (or two ranks) can be
  compared; ``first_difference`` names the first call at which two traces disagree.

It has no torch dependency. ``trace_call_from_resolution`` bridges the semantic operator
registry (``OperatorResolution`` / ``OperatorInstanceProvenance``), and
``block_observer`` plugs into ``h3_block.run_graph``'s ``node_observer`` hook so the
existing one-block replay and full-model replay can emit a trace without changing their
arithmetic.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterable, Mapping, Sequence

from rl_engine.models.minimax_h3 import fingerprint as fp

SCHEMA_VERSION = "rlkernel.h3.operator_trace.v1"

# How a node is bound today (``h3_block.Binding.status``).
STATUS_ROW = "row"  # the owning RFC row's operator
STATUS_INTERIM = "interim"  # another deterministic RL-Kernel operator stands in
STATUS_REFERENCE = "reference"  # provider replay; contributes no RL-Kernel evidence
STATUSES = (STATUS_ROW, STATUS_INTERIM, STATUS_REFERENCE)

SPLIT_NONE = "none"  # no split-K / stream-K / atomics: one fixed reduction order

# Rows whose arithmetic reduces over a dimension and so must accumulate in FP32.
# Everything else is element-wise or pure data movement ("none").
_REDUCING_ROWS = frozenset(
    {
        "adaln_projection_3mod",
        "audio_input_projection_fp32",
        "audio_output_projection_fp32",
        "final_adaln_out",
        "h3_attention_o_gemm",
        "h3_ffn_down_gemm",
        "h3_ffn_gate_up_gemm",
        "h3_full_attention",
        "h3_qk_rmsnorm_d128",
        "h3_qkv_gemm",
        "h3_rmsnorm",
        "text_context_projection",
        "timestep_mlp_fp32",
        "video_input_projection_fp32",
        "video_output_projection_fp32",
    }
)
ACCUMULATOR_FP32 = "fp32"
ACCUMULATOR_NONE = "none"


def expected_accumulator(row: str) -> str:
    """Accumulator the strict contract requires for a row's node."""

    return ACCUMULATOR_FP32 if row in _REDUCING_ROWS else ACCUMULATOR_NONE


@dataclass(frozen=True)
class OperatorCall:
    """One executed graph node."""

    path: str  # graph path, e.g. ``transformer_blocks.7.attention``
    row: str  # owning RFC row (checked against the graph)
    status: str  # row | interim | reference
    backend: str  # backend/operator that actually ran
    requested_backend: str = ""  # what dispatch was asked for; "" when not resolved by a registry
    implementation: str = ""  # concrete class/function reference
    instance_fingerprint: str = ""  # OperatorInstanceProvenance.instance_fingerprint, if known
    dtype: str = ""  # output dtype
    accumulator: str = ""  # fp32 | none | other
    split_policy: str = SPLIT_NONE
    topology: Mapping[str, Any] = field(default_factory=dict)  # e.g. {"tp": 4, "cp": 2, "sp": True}
    shapes: tuple[tuple[int, ...], ...] = ()  # input shapes, when known
    fallback: bool = False  # True if any fallback path was taken

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            out[item.name] = (
                [list(shape) for shape in value]
                if item.name == "shapes"
                else dict(value) if item.name == "topology" else value
            )
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> OperatorCall:
        known = {item.name for item in fields(OperatorCall)}
        extra = set(data) - known
        if extra:
            raise ValueError(f"unknown OperatorCall fields: {sorted(extra)}")
        values = dict(data)
        values["shapes"] = tuple(tuple(shape) for shape in values.get("shapes", ()))
        values["topology"] = dict(values.get("topology", {}))
        return OperatorCall(**values)


class OperatorTrace:
    """Ordered ledger of the nodes one run executed."""

    def __init__(
        self,
        *,
        mode: str = "candidate",
        forwards: int = 1,
        graph: Sequence[fp.GraphNode] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if forwards < 1:
            raise ValueError("forwards must be >= 1")
        self.mode = mode
        self.forwards = forwards
        self.graph = tuple(graph) if graph is not None else fp.build_graph()
        self.metadata = dict(metadata or {})
        self._by_path = {node.path: node for node in self.graph}
        self.calls: list[OperatorCall] = []

    # -- recording ---------------------------------------------------------------------------

    def record(self, call: OperatorCall) -> OperatorCall:
        """Append a call. Unknown paths and a row that disagrees with the graph raise at once."""

        node = self._by_path.get(call.path)
        if node is None:
            raise ValueError(f"{call.path!r} is not a node of the H3 graph")
        if call.row != node.row:
            raise ValueError(f"{call.path}: row {call.row!r} but the graph assigns {node.row!r}")
        if call.status not in STATUSES:
            raise ValueError(f"{call.path}: status {call.status!r} not in {STATUSES}")
        self.calls.append(call)
        return call

    def record_node(self, path: str, **fields_: Any) -> OperatorCall:
        """Record by path; the row is taken from the graph."""

        node = self._by_path.get(path)
        if node is None:
            raise ValueError(f"{path!r} is not a node of the H3 graph")
        fields_.setdefault("accumulator", expected_accumulator(node.row))
        return self.record(OperatorCall(path=path, row=node.row, **fields_))

    def merge(self, other: OperatorTrace) -> None:
        """Append another trace of the same graph (e.g. another rank's or stage's calls)."""

        if other.graph != self.graph:
            raise ValueError("cannot merge traces of different graphs")
        self.calls.extend(other.calls)

    # -- coverage ----------------------------------------------------------------------------

    def counts(self) -> Counter[str]:
        return Counter(call.path for call in self.calls)

    def coverage(self, paths: Iterable[str] | None = None) -> dict[str, list[str]]:
        """Nodes missing, repeated the wrong number of times, or outside the expectation.

        ``paths`` narrows the expectation to a subset of the graph (a one-block replay
        expects only that block); the default is the whole graph.
        """

        expected = list(paths) if paths is not None else [node.path for node in self.graph]
        counts = self.counts()
        wrong = [
            f"{path}: ran {counts.get(path, 0)}x, expected {self.forwards}x"
            for path in expected
            if counts.get(path, 0) != self.forwards
        ]
        unexpected = sorted(set(counts) - set(expected))
        return {"wrong_count": wrong, "unexpected": unexpected}

    # -- strictness --------------------------------------------------------------------------

    def violations(
        self,
        *,
        paths: Iterable[str] | None = None,
        allowed_statuses: Iterable[str] = (STATUS_ROW,),
        require_coverage: bool = True,
    ) -> list[str]:
        """Every reason this trace is not strict evidence; empty means it is."""

        allowed = set(allowed_statuses)
        problems: list[str] = []
        for call in self.calls:
            where = call.path
            if call.status not in allowed:
                problems.append(
                    f"{where}: bound as {call.status!r} ({call.backend}); allowed {sorted(allowed)}"
                )
            if call.fallback:
                problems.append(f"{where}: a fallback path was taken")
            if call.requested_backend and call.requested_backend != call.backend:
                problems.append(
                    f"{where}: requested {call.requested_backend!r} but {call.backend!r} ran"
                )
            if call.split_policy != SPLIT_NONE:
                problems.append(f"{where}: split policy {call.split_policy!r} is not 'none'")
            want = expected_accumulator(call.row)
            if call.accumulator != want:
                problems.append(
                    f"{where}: accumulator {call.accumulator!r}, strict contract needs {want!r}"
                )
            if not call.backend:
                problems.append(f"{where}: no backend recorded")
        if require_coverage:
            cov = self.coverage(paths)
            problems += cov["wrong_count"] + [
                f"{p}: not in the expected set" for p in cov["unexpected"]
            ]
        return problems

    def assert_strict(self, **kwargs: Any) -> None:
        """Raise ``ValueError`` listing every violation (see ``violations``)."""

        problems = self.violations(**kwargs)
        if problems:
            shown = problems[:40]
            more = f"\n  ... and {len(problems) - 40} more" if len(problems) > 40 else ""
            raise ValueError("operator trace is not strict:\n  " + "\n  ".join(shown) + more)

    def status_summary(self) -> dict[str, dict[str, int]]:
        """Per row, how many calls ran under each binding status."""

        summary: dict[str, Counter[str]] = {}
        for call in self.calls:
            summary.setdefault(call.row, Counter())[call.status] += 1
        return {row: dict(counter) for row, counter in sorted(summary.items())}

    # -- serialisation and comparison -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        graph = {"digest": fp.graph_digest(self.graph), "nodes": len(self.graph)}
        body = {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "forwards": self.forwards,
            "metadata": self.metadata,
            "graph": graph,
            "calls": [call.to_dict() for call in self.calls],
        }
        body["trace_sha256"] = _digest(body)
        return body

    @property
    def digest(self) -> str:
        return self.to_dict()["trace_sha256"]

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> OperatorTrace:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema {data.get('schema_version')!r}")
        trace = OperatorTrace(
            mode=data["mode"], forwards=data["forwards"], metadata=data.get("metadata", {})
        )
        if data["graph"]["digest"] != fp.graph_digest(trace.graph):
            raise ValueError("trace was recorded against a different H3 graph")
        for item in data["calls"]:
            trace.record(OperatorCall.from_dict(item))
        if data.get("trace_sha256") != trace.digest:
            raise ValueError("trace_sha256 does not match its content")
        return trace

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n"

    @staticmethod
    def loads(text: str) -> OperatorTrace:
        return OperatorTrace.from_dict(json.loads(text))


def first_difference(lhs: OperatorTrace, rhs: OperatorTrace) -> dict[str, Any] | None:
    """First call at which two traces disagree, or ``None`` if they are identical."""

    for index, (a, b) in enumerate(zip(lhs.calls, rhs.calls)):
        da, db = a.to_dict(), b.to_dict()
        if da != db:
            changed = sorted(key for key in da if da[key] != db[key])
            return {"index": index, "path": a.path, "fields": changed, "lhs": da, "rhs": db}
    if len(lhs.calls) != len(rhs.calls):
        index = min(len(lhs.calls), len(rhs.calls))
        longer = lhs if len(lhs.calls) > len(rhs.calls) else rhs
        return {"index": index, "path": longer.calls[index].path, "fields": ["length"]}
    return None


def _digest(body: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in body.items() if key != "trace_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --- Bridges --------------------------------------------------------------------------------


def trace_call_from_resolution(
    path: str,
    *,
    resolution: Any,
    provenance: Any | None = None,
    status: str = STATUS_ROW,
    dtype: str | None = None,
    accumulator: str | None = None,
    split_policy: str = SPLIT_NONE,
    shapes: Iterable[Iterable[int]] = (),
    row: str | None = None,
    graph: Sequence[fp.GraphNode] | None = None,
) -> OperatorCall:
    """Build a call from a semantic-registry ``OperatorResolution`` (+ instance provenance).

    ``backend`` is the resolved concrete backend, ``requested_backend`` what the caller
    asked for, the fingerprint is the session's ``instance_fingerprint``, and a
    non-empty ``fallback_attempts`` marks the call as a fallback. Unresolved
    resolutions raise: there is nothing that ran.
    """

    trace = resolution.trace
    if trace.status != "resolved" or trace.concrete_backend is None:
        raise ValueError(f"{path}: resolution status {trace.status!r}; nothing ran")
    graph_row = row or {n.path: n.row for n in (graph or fp.build_graph())}.get(path)
    if graph_row is None:
        raise ValueError(f"{path!r} is not a node of the H3 graph")
    return OperatorCall(
        path=path,
        row=graph_row,
        status=status,
        backend=trace.concrete_backend,
        requested_backend=trace.requested_backend,
        implementation=(
            provenance.concrete_implementation
            if provenance is not None
            else (trace.implementation_reference or "")
        ),
        instance_fingerprint=provenance.instance_fingerprint if provenance is not None else "",
        dtype=dtype or resolution.requirements.dtype,
        accumulator=accumulator if accumulator is not None else expected_accumulator(graph_row),
        split_policy=split_policy,
        topology=dict(resolution.requirements.topology),
        shapes=tuple(tuple(shape) for shape in shapes),
        fallback=bool(trace.fallback_attempts),
    )


# The one-block replay (h3_block.run_graph) names nodes by their block-local name;
# the full-model replay reports ``block.<i>.<node>``. Both map onto graph paths.
def graph_path(name: str, *, block_index: int | None = None) -> str:
    """Graph path of a ``block.<i>.<node>`` / ``refiner.<i>.<node>`` / bare block-node name."""

    parts = name.split(".")
    if len(parts) == 3 and parts[0] == "block":
        return f"transformer_blocks.{int(parts[1])}.{parts[2]}"
    if len(parts) == 3 and parts[0] == "refiner":
        return f"token_refiner.refiner_blocks.{int(parts[1])}.{parts[2]}"
    if block_index is not None and name in fp.BLOCK_NODE_NAMES:
        return f"transformer_blocks.{block_index}.{name}"
    return name


def block_observer(
    trace: OperatorTrace,
    *,
    block_index: int = 0,
    bindings: Mapping[str, Any] | None = None,
    topology: Mapping[str, Any] | None = None,
    dtype: str = "bfloat16",
    stage: str = "transformer_blocks",
) -> Callable[[str], None]:
    """``node_observer`` for ``h3_block.run_graph`` / ``h3_refiner.run_graph``.

    Records each node as the block runs it. ``stage`` is ``"transformer_blocks"`` (the 17-node
    block) or ``"token_refiner"`` (the 14-node refiner block). ``bindings`` maps a node name
    to a ``Binding`` (status, backend); by default it is taken from the stage's ``NODES`` so
    the trace states exactly how every node is bound today. Node order and row ownership are
    verified by ``record``.
    """

    if stage not in ("transformer_blocks", "token_refiner"):
        raise ValueError(f"unknown stage {stage!r}")
    if bindings is None:
        if stage == "token_refiner":
            from rl_engine.validation.models import h3_refiner as source  # lazy: needs torch
        else:
            from rl_engine.validation.models import h3_block as source  # lazy: needs torch

        bindings = {node.name: node.binding for node in source.NODES}

    def observe(name: str) -> None:
        binding = bindings[name]
        path = (
            f"token_refiner.refiner_blocks.{block_index}.{name}"
            if stage == "token_refiner"
            else graph_path(name, block_index=block_index)
        )
        trace.record_node(
            path,
            status=binding.status,
            backend=binding.backend,
            dtype=dtype,
            topology=dict(topology or {}),
        )

    return observe
