# SPDX-License-Identifier: Apache-2.0
"""Content-based identity collectors without imports of GPU engines.

A current disk hash is not proof of which bytes are resident in a process.
Import observations explicitly preserve that distinction. Weight-load receipts
are produced only by reading actual loaded content through an adapter callback.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.machinery
import os
import platform
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import ModuleType

from rl_engine.contracts.strict_provider import (
    SnapshotActivationReceipt,
    StrictProviderError,
    fingerprint,
    parse_json,
)


def hash_file(path: str | os.PathLike) -> str:
    """Read all bytes, rejecting a file changed while it was being read."""
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise StrictProviderError(
            "identity_unavailable", "Cannot read identity artifact", str(path)
        ) from exc

    # mtime is only a concurrent-change guard; the identity itself hashes content.
    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    if identity(before) != identity(after) or identity(after) != identity(current):
        raise StrictProviderError(
            "artifact_changed", "Artifact changed during identity collection", str(path)
        )
    return digest.hexdigest()


def _relative_file(root: Path, name: str | os.PathLike) -> tuple[str, Path]:
    path = Path(name)
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        logical = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise StrictProviderError(
            "unsafe_artifact_path", "Artifact must be an existing file inside root", str(name)
        ) from exc
    if not resolved.is_file():
        raise StrictProviderError(
            "identity_unavailable", "Identity artifact is not a file", str(name)
        )
    return logical, resolved


def file_inventory(root: str | os.PathLike, paths: Iterable[str | os.PathLike]) -> dict:
    """Portable relative names and real content hashes; never absolute cache paths."""
    root = Path(root).resolve(strict=True)
    records = {}
    for name in paths:
        logical, resolved = _relative_file(root, name)
        if logical in records:
            raise StrictProviderError("invalid_inventory", "Duplicate artifact", logical)
        content_hash = hash_file(resolved)
        records[logical] = {
            "path": logical,
            "size": resolved.stat().st_size,
            "sha256": content_hash,
        }
    if not records:
        raise StrictProviderError("invalid_inventory", "Artifact inventory cannot be empty")
    files = [records[name] for name in sorted(records)]
    return {"files": files, "inventory_sha256": fingerprint(files)}


def source_identity(root: str | os.PathLike, paths: Iterable[str | os.PathLike]) -> dict:
    return file_inventory(root, paths)


def checkpoint_identity(
    root: str | os.PathLike,
    *,
    model_id: str,
    revision: str,
    config_name: str = "config.json",
    index_name: str = "model.safetensors.index.json",
) -> dict:
    """Hash every shard referenced by the checkpoint index, not its size/mtime."""
    root = Path(root).resolve(strict=True)
    _, config_path = _relative_file(root, config_name)
    _, index_path = _relative_file(root, index_name)
    config_digest = hash_file(config_path)
    index_digest = hash_file(index_path)
    try:
        index = parse_json(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise StrictProviderError("invalid_checkpoint", "Cannot parse checkpoint index") from exc
    if hash_file(index_path) != index_digest:
        raise StrictProviderError("artifact_changed", "Checkpoint index changed while being read")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise StrictProviderError(
            "invalid_checkpoint", "Nonempty checkpoint weight_map is required", "weight_map"
        )
    if any(not isinstance(name, str) for name in weight_map.values()):
        raise StrictProviderError("invalid_checkpoint", "Shard names must be strings", "weight_map")
    shards = file_inventory(root, sorted(set(weight_map.values())))
    return {
        "model_id": model_id,
        "revision": revision,
        "config_sha256": config_digest,
        "weight_index_sha256": index_digest,
        "shard_inventory_sha256": shards["inventory_sha256"],
        "shards": shards["files"],
        "evidence_kind": "disk_checkpoint",
    }


def observe_imported_extension(module: ModuleType | str) -> dict:
    """Observe an actually imported extension, without claiming an at-load hash.

    ExtensionFileLoader and sys.modules establish which module origin the
    interpreter selected. They cannot establish that its current disk bytes
    equal bytes loaded earlier. Enforce must require a stronger loader receipt
    when that guarantee is part of its profile.
    """
    if isinstance(module, str):
        name = module
        module = sys.modules.get(name)
    else:
        name = getattr(module, "__name__", "")
    if not isinstance(module, ModuleType) or sys.modules.get(name) is not module:
        raise StrictProviderError(
            "unloaded_artifact", "Expected an actual sys.modules instance", name
        )
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    origin = getattr(spec, "origin", None)
    if not isinstance(loader, importlib.machinery.ExtensionFileLoader) or not origin:
        raise StrictProviderError(
            "not_native_extension", "Module is not loaded by ExtensionFileLoader", name
        )
    path = Path(origin).resolve(strict=True)
    module_file = getattr(module, "__file__", "")
    if not module_file or Path(module_file).resolve(strict=True) != path:
        raise StrictProviderError(
            "loader_origin_mismatch", "Module file and loader origin differ", name
        )
    return {
        "module": name,
        "loader": f"{type(loader).__module__}.{type(loader).__qualname__}",
        "observed_path": str(path),
        "current_disk_sha256": hash_file(path),
        "size": path.stat().st_size,
        "evidence_kind": "imported_extension_current_disk",
        "loaded_binary_verified": False,
    }


def environment_identity(
    *, package_names: Iterable[str] = (), environment_names: Iterable[str] = ()
) -> dict:
    """Read an explicit allowlist; do not leak arbitrary environment secrets."""
    packages = {}
    for name in sorted(set(package_names)):
        try:
            packages[name] = {"version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"status": "not_installed"}
    environment = {}
    for name in sorted(set(environment_names)):
        environment[name] = (
            ({"value": os.environ[name]} if os.environ[name] else {"status": "set_empty"})
            if name in os.environ
            else {"status": "unset"}
        )
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": packages,
        "environment": environment,
    }


def _bytes_digest(content, field):
    if not isinstance(content, (bytes, bytearray, memoryview)) or not len(content):
        raise StrictProviderError(
            "invalid_load_evidence", "A nonempty loaded-content byte view is required", field
        )
    return hashlib.sha256(content).hexdigest()


def create_worker_load_receipt(
    *,
    snapshot_id: str,
    generation: int,
    worker_id: str,
    expected_shard_digests: Mapping[str, str],
    read_loaded_shard: Callable[[str], bytes],
    required_refresh: Iterable[str] = (),
    read_refreshed_component: Callable[[str], bytes] | None = None,
) -> SnapshotActivationReceipt:
    """Verify actual worker content at the adapter's completed load boundary.

    Callbacks must read the worker's loaded tensors/caches, using the canonical
    byte format expected by the adapter. Passing a version string, a completed
    flag or a disk path is insufficient. This does not itself attach to any
    engine's load/update lifecycle.
    """
    if not expected_shard_digests:
        raise StrictProviderError("invalid_load_evidence", "Expected shard inventory is empty")
    observed = {}
    for name in sorted(expected_shard_digests):
        expected = expected_shard_digests[name]
        actual = _bytes_digest(read_loaded_shard(name), f"shards.{name}")
        if actual != expected:
            raise StrictProviderError(
                "loaded_content_mismatch",
                "Loaded shard content differs",
                f"shards.{name}",
                expected,
                actual,
            )
        observed[name] = actual
    refresh = tuple(required_refresh)
    if len(set(refresh)) != len(refresh):
        raise StrictProviderError("invalid_load_evidence", "Duplicate required refresh component")
    refreshed = {}
    for name in sorted(refresh):
        if read_refreshed_component is None:
            raise StrictProviderError(
                "incomplete_refresh", "Refresh evidence reader is required", name
            )
        refreshed[name] = _bytes_digest(read_refreshed_component(name), f"refresh.{name}")
    return SnapshotActivationReceipt.from_dict(
        {
            "snapshot_id": snapshot_id,
            "generation": generation,
            "worker_id": worker_id,
            "expected_shards": sorted(expected_shard_digests),
            "verified_shards": sorted(observed),
            "required_refresh": sorted(refresh),
            "refreshed_components": sorted(refreshed),
            "content_digest": fingerprint({"shards": observed, "refresh": refreshed}),
            "completed": True,
            "evidence_kind": "worker_load",
        }
    )
