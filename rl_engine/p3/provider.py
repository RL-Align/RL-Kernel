"""Recorded producer, durable invocation allocator and sealed-saved validation.

T02/T03/T04/T06 use reference.py entry points and retain this shared wrapper contract.
Recorded CPU results are development evidence and cannot certify CUDA gates.
"""

import fcntl
import json
import os
import struct
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import oracle
from .contract import (
    OP_ABI,
    SAVED_FIELDS,
    SAVED_HEADER_FIELDS,
    SAVED_ROUTE_SCHEMA,
    SAVED_SCORE_SCHEMA,
    UNSET,
    E,
    K,
    P3Error,
    P3OpResult,
    P3Verdict,
    SavedRouteSealedV1,
    SavedScoreSealedV1,
)
from .serialization import fingerprint


class InvocationAllocator:
    """One locked durable journal per (run,engine,rank); reserve before launch."""

    def __init__(self, path, run_id, engine_id, rank, attempt_id=1):
        self.path = Path(path)
        self.scope = [run_id, engine_id, rank]
        self.attempt_id = attempt_id
        if not 0 < attempt_id <= 0xFFFFFFFF:
            raise P3Error(P3Verdict.CORRUPT_ARTIFACT, "invalid attempt id")

    @staticmethod
    def _commit(path, state):
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
            json.dump({"state": state, "checksum": fingerprint(state)}, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @classmethod
    def new_attempt(cls, path, run_id, engine_id, rank):
        """Durably allocate the next attempt under the same scope journal lock."""
        path = Path(path)
        scope = [run_id, engine_id, rank]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                attempt = 1
                if path.exists():
                    stored = json.loads(path.read_text())
                    state = stored["state"]
                    oracle.require(
                        stored["checksum"] == fingerprint(state)
                        and state["version"] == "p3-invocation-journal.v1"
                        and type(state["counter"]) is int
                        and 0 <= state["counter"] <= 0xFFFFFFFF
                        and type(state["attempt_id"]) is int
                        and 0 < state["attempt_id"] < 0xFFFFFFFF,
                        P3Verdict.CORRUPT_ARTIFACT,
                        "journal integrity or attempt exhaustion",
                    )
                    oracle.require(
                        state["scope"] == scope, P3Verdict.IDENTITY_DRIFT, "journal scope"
                    )
                    attempt = state["attempt_id"] + 1
                cls._commit(
                    path,
                    {
                        "version": "p3-invocation-journal.v1",
                        "scope": scope,
                        "attempt_id": attempt,
                        "counter": 0,
                    },
                )
                return cls(path, run_id, engine_id, rank, attempt)
        except P3Error:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise P3Error(P3Verdict.CORRUPT_ARTIFACT, "attempt durable commit failed") from exc

    def reserve(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix(".lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = {
                    "version": "p3-invocation-journal.v1",
                    "scope": self.scope,
                    "attempt_id": self.attempt_id,
                    "counter": 0,
                }
                if self.path.exists():
                    stored = json.loads(self.path.read_text())
                    state = stored["state"]
                    if (
                        stored["checksum"] != fingerprint(state)
                        or state["version"] != "p3-invocation-journal.v1"
                    ):
                        raise P3Error(P3Verdict.CORRUPT_ARTIFACT, "journal checksum/version")
                    if state["scope"] != self.scope:
                        raise P3Error(P3Verdict.IDENTITY_DRIFT, "allocator scope mismatch")
                    if state["attempt_id"] > self.attempt_id:
                        raise P3Error(P3Verdict.STALE_RUN_METADATA, "stale attempt")
                    if state["attempt_id"] < self.attempt_id:
                        state.update(attempt_id=self.attempt_id, counter=0)
                if type(state["counter"]) is not int or not 0 <= state["counter"] < 0xFFFFFFFF:
                    raise P3Error(P3Verdict.CORRUPT_ARTIFACT, "counter exhaustion; no wrap")
                state["counter"] += 1
                self._commit(self.path, state)
                return self.attempt_id << 32 | state["counter"]
        except P3Error:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise P3Error(P3Verdict.CORRUPT_ARTIFACT, "journal durable commit failed") from exc


def readback_verdict(record, invocation_id, *, launched=True, synchronized=True):
    if not launched or not synchronized:
        return P3Verdict.INCOMPLETE_ARTIFACT
    if len(record) != 16:
        return P3Verdict.CORRUPT_ARTIFACT
    status, reserved, echo = struct.unpack("<iiQ", record)
    if reserved or status not in (UNSET, 1, 2):
        return P3Verdict.CORRUPT_ARTIFACT
    if not invocation_id or echo != invocation_id:
        return P3Verdict.INCOMPLETE_ARTIFACT
    return P3Verdict.PASS if status == UNSET else P3Verdict(status)


def payload_checksum(version, payload):
    return fingerprint({name: payload[name] for name in SAVED_FIELDS[version]})


def seal_saved(version, payload, ctx, forward_result):
    oracle.require(
        forward_result.verdict == P3Verdict.PASS and ctx.invocation_id != 0,
        P3Verdict.INCOMPLETE_ARTIFACT,
        "only completed non-zero forward can seal",
    )
    oracle.require(
        bool(ctx.row_active.any()) and len(ctx.route_artifact_fingerprint) == 64,
        P3Verdict.IDENTITY_DRIFT,
        "missing active route artifact identity",
    )
    copied = {k: np.array(payload[k], copy=True) for k in SAVED_FIELDS[version]}
    header = {
        "version": version,
        "operator_abi": OP_ABI,
        "row_active": ctx.row_active.copy(),
        "route_artifact_fingerprint": ctx.route_artifact_fingerprint,
        "forward_invocation_id": ctx.invocation_id,
        "manifest_source": ctx.manifest_source,
        "saved_payload_checksum": payload_checksum(version, copied),
    }
    cls = SavedScoreSealedV1 if version == SAVED_SCORE_SCHEMA else SavedRouteSealedV1
    return cls(header, copied)


def validate_saved(saved, version, ctx):
    cls = SavedScoreSealedV1 if version == SAVED_SCORE_SCHEMA else SavedRouteSealedV1
    oracle.require(type(saved) is cls, P3Verdict.SCHEMA_MISMATCH, "raw or old saved forbidden")
    header, payload = saved.header, saved.payload
    required = set(SAVED_HEADER_FIELDS)
    oracle.require(
        set(header) == required
        and header["version"] == version
        and header["operator_abi"] == OP_ABI
        and isinstance(header["forward_invocation_id"], int)
        and 0 < header["forward_invocation_id"] < 2**64,
        P3Verdict.SCHEMA_MISMATCH,
        "sealed header ABI mismatch",
    )
    oracle.tensor(header["row_active"], "bool", ctx.row_active.shape)
    oracle.require(
        set(payload) == set(SAVED_FIELDS[version]), P3Verdict.SCHEMA_MISMATCH, "saved field set"
    )
    T = len(ctx.row_active)
    shapes = (
        {"z_prime": (T, E), "s": (T, E)}
        if version == SAVED_SCORE_SCHEMA
        else {"ids": (T, K), "a": (T, K), "Z": (T, 1), "p": (T, K)}
    )
    for name, shape in shapes.items():
        oracle.tensor(payload[name], "int32" if name == "ids" else "float32", shape)
    oracle.require(
        header["route_artifact_fingerprint"] == ctx.route_artifact_fingerprint
        and np.array_equal(header["row_active"], ctx.row_active),
        P3Verdict.IDENTITY_DRIFT,
        "forward route/mask identity drift",
    )
    oracle.require(
        header["manifest_source"] == ctx.manifest_source,
        P3Verdict.IDENTITY_DRIFT,
        "saved manifest source mismatch",
    )
    oracle.require(
        header["saved_payload_checksum"] == payload_checksum(version, payload),
        P3Verdict.CORRUPT_ARTIFACT,
        "saved bytes checksum mismatch",
    )


class RecordedProvider:
    def __init__(self, recording):
        self.recording = deepcopy(recording)

    @classmethod
    def from_artifact(cls, path, case_id):
        from .artifact import verify

        artifact = verify(path)
        matching = [r for r in artifact["recordings"] if r["case_id"] == case_id]
        oracle.require(len(matching) == 1, P3Verdict.IDENTITY_DRIFT, "case recording not found")
        return cls(matching[0])

    def invoke(self, operator, ctx, *inputs):
        from .checker import validate_operator_inputs, validate_recording
        from .provider import validate_saved

        try:
            validate_operator_inputs(operator, inputs, ctx.row_active)
            oracle.tensor(ctx.row_active, "bool", self.recording["row_active"].shape)
            oracle.require(
                np.array_equal(ctx.row_active, self.recording["row_active"]),
                P3Verdict.IDENTITY_DRIFT,
                "recording active mask drift",
            )
            if operator.endswith("_bwd"):
                version = (
                    SAVED_SCORE_SCHEMA if operator.startswith("router_sqrt") else SAVED_ROUTE_SCHEMA
                )
                validate_saved(inputs[-1], version, ctx)
                inputs = (*inputs[:-1], inputs[-1].payload)
            validate_recording(self.recording)
            if not ctx.row_active.any():
                return P3OpResult(P3Verdict.ZERO_ACTIVE_TOKENS)
            op = self.recording["operators"][operator]
            oracle.require(
                ctx.backend_tag == "recorded-cpu",
                P3Verdict.UNSUPPORTED_CAPABILITY,
                "no CUDA recorded fallback",
            )
            oracle.require(
                fingerprint(list(inputs)) == op["input_checksum"],
                P3Verdict.IDENTITY_DRIFT,
                "recording inputs drifted",
            )
            return P3OpResult(P3Verdict.PASS, deepcopy(op["payload"]), op["provenance"].copy())
        except KeyError:
            return P3OpResult(P3Verdict.SCHEMA_MISMATCH)
        except Exception as exc:
            if hasattr(exc, "verdict"):
                verdict = exc.verdict if exc.verdict < 50 else P3Verdict.CORRUPT_ARTIFACT
                return P3OpResult(verdict)
            raise

    def router_sqrt_softplus_fwd(self, ctx, z, logit_round_point):
        return self.invoke("router_sqrt_softplus_fwd", ctx, z, logit_round_point)

    def router_sqrt_softplus_bwd(self, ctx, ds, saved_score_sealed):
        return self.invoke("router_sqrt_softplus_bwd", ctx, ds, saved_score_sealed)

    def hash_route_fwd(self, ctx, input_token_id, s, tid2eid):
        return self.invoke("hash_route_fwd", ctx, input_token_id, s, tid2eid)

    def hash_route_bwd(self, ctx, dweights, saved_route_sealed):
        return self.invoke("hash_route_bwd", ctx, dweights, saved_route_sealed)

    def stable_topk6_fwd(self, ctx, q):
        return self.invoke("stable_topk6_fwd", ctx, q)

    def learned_route_fwd(self, ctx, s, b):
        return self.invoke("learned_route_fwd", ctx, s, b)

    def learned_route_bwd(self, ctx, dweights, saved_route_sealed):
        return self.invoke("learned_route_bwd", ctx, dweights, saved_route_sealed)
