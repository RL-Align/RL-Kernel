# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Append-only, crash-safe artifacts for cross-configuration runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from rl_engine.validation.cross_config._json import strict_json_loads

REQUIRED_CASE_ARTIFACTS = frozenset(
    {
        "requested.json",
        "materialized.json",
        "actual.json",
        "identity.json",
        "score_rollout.pt",
        "score_training.pt",
        "comparison.json",
        "token_diffs.pt",
    }
)
_JSON_SCHEMAS = {
    "requested.json": "cross_config.requested.v1",
    "materialized.json": "cross_config.materialized_envelope.v1",
    "actual.json": "cross_config.actual.v1",
    "identity.json": "cross_config.identity_envelope.v1",
    "comparison.json": "cross_config.alignment_result.v1",
}
_JSON_REQUIRED_KEYS = {
    "requested.json": frozenset({"case"}),
    "materialized.json": frozenset({"materialized_case"}),
    "actual.json": frozenset({"rollout", "training"}),
    "identity.json": frozenset({"identity"}),
    "comparison.json": frozenset({"status", "comparable", "passed"}),
}
_TENSOR_REQUIRED_KEYS = {
    "score_rollout.pt": frozenset({"selected_logprobs", "active_mask"}),
    "score_training.pt": frozenset({"selected_logprobs", "active_mask"}),
    "token_diffs.pt": frozenset(
        {
            "rollout_logprobs",
            "training_logprobs",
            "active_mask",
            "absolute_diff",
            "mismatch_mask",
        }
    ),
}


class ArtifactError(RuntimeError):
    """Raised when an artifact is incomplete, malformed, or would be overwritten."""


class ArtifactStore:
    """Persist immutable attempt directories and atomically mark completed attempts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def experiment_dir(self, experiment_id: str) -> Path:
        return self.root / _safe_component(experiment_id, "experiment_id")

    def initialize_experiment(
        self,
        experiment_id: str,
        *,
        experiment: Mapping[str, Any],
        plan: Iterable[Mapping[str, Any]],
    ) -> Path:
        """Create immutable experiment metadata, or verify an identical resume target."""

        directory = self.experiment_dir(experiment_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_or_verify_json(directory / "experiment.json", experiment)
        plan_text = "".join(_canonical_json(item) + "\n" for item in plan)
        self._write_or_verify_text(directory / "plan.jsonl", plan_text)
        return directory

    def create_attempt(
        self,
        experiment_id: str,
        case_id: str,
        *,
        attempt_id: Optional[str] = None,
    ) -> Path:
        """Allocate an append-only attempt directory for a case."""

        case_dir = (
            self.experiment_dir(experiment_id) / "cases" / _safe_component(case_id, "case_id")
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        if attempt_id is not None:
            attempt_dir = case_dir / _safe_component(attempt_id, "attempt_id")
            try:
                attempt_dir.mkdir()
            except FileExistsError as exc:
                raise ArtifactError(f"attempt already exists: {attempt_dir}") from exc
            _fsync_directory(case_dir)
            return attempt_dir

        # Another controller can win after _next_attempt_id() observes the
        # directory. mkdir is the atomic allocator; retry rather than aliasing or
        # overwriting the winning attempt.
        while True:
            resolved_attempt_id = self._next_attempt_id(case_dir)
            attempt_dir = case_dir / resolved_attempt_id
            try:
                attempt_dir.mkdir()
            except FileExistsError:
                continue
            _fsync_directory(case_dir)
            return attempt_dir

    def write_json(self, attempt_dir: str | Path, name: str, value: Mapping[str, Any]) -> Path:
        path = self._attempt_path(attempt_dir, name, suffix=".json")
        self._write_new_text(path, _canonical_json(value) + "\n")
        return path

    def write_tensor_bundle(
        self,
        attempt_dir: str | Path,
        name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Write CPU tensor payloads that can be loaded with ``weights_only=True``."""

        path = self._attempt_path(attempt_dir, name, suffix=".pt")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "tensors": {
                key: tensor.detach().to(device="cpu").contiguous()
                for key, tensor in tensors.items()
            },
            "metadata": dict(metadata or {}),
        }
        self._atomic_torch_save(path, payload)
        return path

    def load_tensor_bundle(self, path: str | Path) -> dict[str, Any]:
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ArtifactError(f"failed to load tensor artifact {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ArtifactError(f"unsupported tensor artifact schema: {path}")
        tensors = payload.get("tensors")
        if not isinstance(tensors, dict) or not all(
            isinstance(value, torch.Tensor) for value in tensors.values()
        ):
            raise ArtifactError(f"malformed tensor payload: {path}")
        return payload

    def complete_attempt(
        self,
        attempt_dir: str | Path,
        *,
        summary: Mapping[str, Any],
        required: Iterable[str] = REQUIRED_CASE_ARTIFACTS,
    ) -> Path:
        """Validate all payloads before publishing an atomic ``COMPLETE`` marker."""

        directory = Path(attempt_dir)
        required_names = frozenset(required)
        missing = sorted(name for name in required_names if not (directory / name).is_file())
        if missing:
            raise ArtifactError(f"cannot complete {directory}; missing artifacts: {missing}")
        self._validate_machine_artifacts(directory)
        marker_value = dict(summary)
        marker_value["artifact_sha256"] = {
            name: _sha256_file(directory / name) for name in sorted(required_names)
        }
        self._validate_complete_summary(directory, marker_value)
        marker = directory / "COMPLETE"
        self._write_new_text(marker, _canonical_json(marker_value) + "\n")
        return marker

    def completed_attempt(
        self,
        experiment_id: str,
        case_id: str,
        *,
        required: Iterable[str] = REQUIRED_CASE_ARTIFACTS,
    ) -> Optional[Path]:
        """Return the newest valid completed attempt, ignoring partial attempts."""

        case_dir = (
            self.experiment_dir(experiment_id) / "cases" / _safe_component(case_id, "case_id")
        )
        if not case_dir.is_dir():
            return None
        for attempt_dir in sorted(
            case_dir.iterdir(),
            key=_attempt_sort_key,
            reverse=True,
        ):
            if not attempt_dir.is_dir() or not (attempt_dir / "COMPLETE").is_file():
                continue
            try:
                self.validate_completed_attempt(
                    attempt_dir,
                    required=required,
                    expected_case_id=case_id,
                )
            except ArtifactError:
                continue
            return attempt_dir
        return None

    def validate_completed_attempt(
        self,
        attempt_dir: str | Path,
        *,
        required: Iterable[str] = REQUIRED_CASE_ARTIFACTS,
        expected_case_id: Optional[str] = None,
    ) -> None:
        directory = Path(attempt_dir)
        required_names = frozenset(required)
        marker = directory / "COMPLETE"
        if not marker.is_file():
            raise ArtifactError(f"missing COMPLETE marker: {directory}")
        try:
            marker_value = strict_json_loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ArtifactError(f"malformed COMPLETE marker: {marker}") from exc
        if not isinstance(marker_value, dict):
            raise ArtifactError(f"COMPLETE marker must contain a JSON object: {marker}")
        self._validate_complete_summary(
            directory,
            marker_value,
            expected_case_id=expected_case_id,
        )
        missing = sorted(name for name in required_names if not (directory / name).is_file())
        if missing:
            raise ArtifactError(f"completed attempt is missing artifacts: {missing}")
        self._validate_artifact_hashes(directory, marker_value, required_names)
        self._validate_machine_artifacts(directory, expected_case_id=expected_case_id)

    @staticmethod
    def _validate_artifact_hashes(
        directory: Path,
        marker: Mapping[str, Any],
        required: frozenset[str],
    ) -> None:
        recorded = marker.get("artifact_sha256")
        if not isinstance(recorded, Mapping) or set(recorded) != set(required):
            raise ArtifactError(f"COMPLETE marker has invalid artifact hashes: {directory}")
        for name in sorted(required):
            expected = recorded.get(name)
            if not isinstance(expected, str) or expected != _sha256_file(directory / name):
                raise ArtifactError(f"artifact hash does not match COMPLETE: {directory / name}")

    def _validate_machine_artifacts(
        self,
        directory: Path,
        *,
        expected_case_id: Optional[str] = None,
    ) -> None:
        tensor_payloads: dict[str, dict[str, Any]] = {}
        for name in ("score_rollout.pt", "score_training.pt", "token_diffs.pt"):
            path = directory / name
            if path.exists():
                payload = self.load_tensor_bundle(path)
                tensor_payloads[name] = payload
                tensors = payload["tensors"]
                missing_tensor_keys = sorted(_TENSOR_REQUIRED_KEYS[name].difference(tensors))
                if missing_tensor_keys:
                    raise ArtifactError(
                        f"tensor artifact {name} is missing keys: {missing_tensor_keys}"
                    )
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    raise ArtifactError(f"tensor artifact metadata must be an object: {path}")
                if expected_case_id is not None and metadata.get("case_id") != expected_case_id:
                    raise ArtifactError(
                        f"tensor artifact case_id does not match {expected_case_id!r}: {path}"
                    )
                if metadata.get("attempt_id") != directory.name:
                    raise ArtifactError(
                        f"tensor artifact attempt_id does not match {directory.name!r}: {path}"
                    )
        json_payloads: dict[str, dict[str, Any]] = {}
        for name in (
            "requested.json",
            "materialized.json",
            "actual.json",
            "identity.json",
            "comparison.json",
        ):
            path = directory / name
            if not path.exists():
                continue
            try:
                value = strict_json_loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ArtifactError(f"malformed JSON artifact: {path}") from exc
            if not isinstance(value, dict):
                raise ArtifactError(f"JSON artifact must contain an object: {path}")
            json_payloads[name] = value
            if expected_case_id is not None and value.get("case_id") != expected_case_id:
                raise ArtifactError(
                    f"JSON artifact case_id does not match {expected_case_id!r}: {path}"
                )
            if value.get("schema_version") != _JSON_SCHEMAS[name]:
                raise ArtifactError(f"JSON artifact has an unsupported schema: {path}")
            missing_json_keys = sorted(_JSON_REQUIRED_KEYS[name].difference(value))
            if missing_json_keys:
                raise ArtifactError(f"JSON artifact {name} is missing keys: {missing_json_keys}")
            if value.get("attempt_id") != directory.name:
                raise ArtifactError(
                    f"JSON artifact attempt_id does not match {directory.name!r}: {path}"
                )

        if expected_case_id is None and json_payloads:
            case_ids = {payload.get("case_id") for payload in json_payloads.values()}
            if len(case_ids) != 1 or None in case_ids:
                raise ArtifactError("JSON artifacts must declare one consistent case_id")
            inferred_case_id = next(iter(case_ids))
            for name, payload in tensor_payloads.items():
                if payload["metadata"].get("case_id") != inferred_case_id:
                    raise ArtifactError(
                        f"tensor artifact case_id does not match {inferred_case_id!r}: "
                        f"{directory / name}"
                    )

    @staticmethod
    def _validate_complete_summary(
        directory: Path,
        summary: Mapping[str, Any],
        *,
        expected_case_id: Optional[str] = None,
    ) -> None:
        if summary.get("schema_version") != "cross_config.complete.v1":
            raise ArtifactError(f"COMPLETE marker has an unsupported schema: {directory}")
        case_id = summary.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ArtifactError(f"COMPLETE marker is missing case_id: {directory}")
        if expected_case_id is not None and case_id != expected_case_id:
            raise ArtifactError(
                f"COMPLETE marker case_id does not match {expected_case_id!r}: {directory}"
            )
        if summary.get("attempt_id") != directory.name:
            raise ArtifactError(
                f"COMPLETE marker attempt_id does not match {directory.name!r}: {directory}"
            )
        if not isinstance(summary.get("status"), str):
            raise ArtifactError(f"COMPLETE marker is missing status: {directory}")
        if not isinstance(summary.get("artifact_sha256"), Mapping):
            raise ArtifactError(f"COMPLETE marker is missing artifact hashes: {directory}")

    def _write_or_verify_json(self, path: Path, value: Mapping[str, Any]) -> None:
        self._write_or_verify_text(path, _canonical_json(value) + "\n")

    def _write_or_verify_text(self, path: Path, text: str) -> None:
        if path.exists():
            self._verify_existing_text(path, text)
            return
        try:
            self._atomic_write_text(path, text)
        except ArtifactError:
            # A concurrent writer may have atomically published the same immutable
            # experiment metadata. Accept only byte-identical content.
            if not path.exists():
                raise
            self._verify_existing_text(path, text)

    @staticmethod
    def _verify_existing_text(path: Path, text: str) -> None:
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ArtifactError(f"failed to read existing artifact {path}: {exc}") from exc
        if existing != text:
            raise ArtifactError(f"resume metadata differs from existing artifact: {path}")

    def _write_new_text(self, path: Path, text: str) -> None:
        if path.exists():
            raise ArtifactError(f"refusing to overwrite artifact: {path}")
        self._atomic_write_text(path, text)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            _publish_new_file(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _atomic_torch_save(self, path: Path, payload: Mapping[str, Any]) -> None:
        if path.exists():
            raise ArtifactError(f"refusing to overwrite artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            torch.save(dict(payload), temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            _publish_new_file(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _next_attempt_id(case_dir: Path) -> str:
        indices: list[int] = []
        for child in case_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("attempt-"):
                continue
            suffix = child.name.removeprefix("attempt-")
            if suffix.isdigit():
                indices.append(int(suffix))
        return f"attempt-{max(indices, default=0) + 1:04d}"

    @staticmethod
    def _attempt_path(attempt_dir: str | Path, name: str, *, suffix: str) -> Path:
        directory = Path(attempt_dir)
        safe_name = _safe_component(name, "artifact name")
        if not safe_name.endswith(suffix):
            safe_name += suffix
        return directory / safe_name


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"artifact is not strict JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"failed to hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be a single non-empty path component")
    return value


def _attempt_sort_key(path: Path) -> tuple[int, int, str]:
    """Order standard attempt IDs numerically and retain nonstandard fallbacks."""

    prefix = "attempt-"
    suffix = path.name.removeprefix(prefix)
    if path.name.startswith(prefix) and suffix.isdigit():
        return (1, int(suffix), path.name)
    return (0, -1, path.name)


def _publish_new_file(temporary: Path, destination: Path) -> None:
    """Atomically publish without ever replacing an existing artifact."""

    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise ArtifactError(f"refusing to overwrite artifact: {destination}") from exc
    temporary.unlink()
    _fsync_directory(destination.parent)


def _fsync_directory(directory: Path) -> None:
    """Persist directory entry changes where the host filesystem supports it."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = ["ArtifactError", "ArtifactStore", "REQUIRED_CASE_ARTIFACTS"]
