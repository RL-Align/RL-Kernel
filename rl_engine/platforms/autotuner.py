# SPDX-License-Identifier: Apache-2.0
"""Opt-in, process-safe persistent kernel configuration selection.

This module has no Triton dependency. Callers own candidate validation, timing,
resource recovery and buffers; only validated configurations may be cached.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

from rl_engine.runtime.policy import RLKernelMode, rl_kernel_mode

_LOG = logging.getLogger(__name__)
_SCHEMA = 1


def autotune_enabled() -> bool:
    """Return whether tuning is on: ``RL_KERNEL_AUTOTUNE=1`` in ``RL_KERNEL_MODE=auto``."""
    return os.getenv("RL_KERNEL_AUTOTUNE", "0") == "1" and rl_kernel_mode() == RLKernelMode.AUTO


@dataclass(frozen=True)
class TuneConfig:
    """Immutable launch overrides; an empty configuration is the original launch."""

    options: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, int]:
        """Return the overrides as Triton launch keyword arguments."""
        return dict(self.options)


@dataclass(frozen=True)
class TuneSelection:
    """A selected configuration and where it came from: tuned, disk, memory or capture_default."""

    config: TuneConfig
    source: str


def cache_key(signature: Mapping, candidates: tuple[TuneConfig, ...]) -> str:
    """Hash the launch signature and candidate list into a stable cache file name."""
    encoded = json.dumps(
        {
            "schema": _SCHEMA,
            "signature": signature,
            "candidates": [c.to_dict() for c in candidates],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _untrusted_reason(directory: Path) -> str | None:
    """Explain why other users could rewrite selections in ``directory``, if they could."""
    if not hasattr(os, "getuid"):
        return None
    status = directory.stat()
    if status.st_uid != os.getuid():
        return "owned by another user"
    if status.st_mode & 0o022:
        return "writable by group or others"
    return None


class PersistentAutotuner:
    """Cache configuration choices. ``evaluate`` returns milliseconds or rejects with None."""

    def __init__(self, cache_dir: Path | None = None):
        """Use ``cache_dir``, or resolve the environment/default directory on select."""
        self.cache_dir = cache_dir
        self._memory: dict[str, TuneConfig] = {}
        self._lock = threading.RLock()

    def select(
        self,
        signature: Mapping,
        candidates: tuple[TuneConfig, ...],
        evaluate: Callable[[TuneConfig], float | None],
        *,
        persistent: bool = True,
        capture: bool = False,
    ) -> TuneSelection:
        """Return the cached configuration for ``signature``, tuning it on first use.

        ``evaluate`` returns a timing in milliseconds, or ``None`` to reject a
        candidate. During graph capture this never evaluates or touches disk; it
        returns a memory hit or the unchanged launch.
        """
        if not candidates or candidates[0] != TuneConfig():
            raise ValueError("The first candidate must be the unchanged launch")
        key = cache_key(signature, candidates)
        if key in self._memory:
            return TuneSelection(self._memory[key], "memory")
        if capture:
            return TuneSelection(candidates[0], "capture_default")
        with self._lock:
            if key in self._memory:
                return TuneSelection(self._memory[key], "memory")
            directory = (
                self.cache_dir
                or Path(
                    os.getenv("RL_KERNEL_AUTOTUNE_CACHE_DIR", "~/.cache/rl_kernel/autotune")
                ).expanduser()
            )
            with self._disk_lock(directory, key, persistent) as disk:
                path = directory / f"{key}.json"
                config = self._read(path, key, candidates) if disk else None
                source = "disk"
                if config is None:
                    best = math.inf
                    config = candidates[0]
                    for candidate in candidates:
                        elapsed = evaluate(candidate)
                        if elapsed is None:
                            if candidate == candidates[0]:
                                raise RuntimeError("The fixed kernel configuration was rejected")
                            continue
                        if not math.isfinite(elapsed) or elapsed < 0:
                            raise ValueError(f"Invalid kernel timing: {elapsed}")
                        if elapsed < best:
                            best, config = elapsed, candidate
                    if disk:
                        self._write(path, key, config)
                    source = "tuned"
                self._memory[key] = config
                return TuneSelection(config, source)

    @staticmethod
    @contextmanager
    def _disk_lock(directory: Path, key: str, enabled: bool) -> Iterator[bool]:
        """Yield True while holding the per-key file lock, or False for memory only."""
        if not enabled:
            yield False
            return
        handle = None
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            reason = _untrusted_reason(directory)
            if reason is not None:
                _LOG.warning(
                    "Autotune cache %s is not private (%s); using memory", directory, reason
                )
                yield False
                return
            handle = (directory / f"{key}.lock").open("a")
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as exc:
            if handle is not None:
                handle.close()
            _LOG.warning("Autotune cache unavailable; using memory: %s", exc)
            yield False
            return
        try:
            yield True
        finally:
            handle.close()

    @staticmethod
    def _read(path: Path, key: str, candidates: tuple[TuneConfig, ...]) -> TuneConfig | None:
        """Return the candidate cached for ``key`` at ``path``, or None if absent/invalid."""
        try:
            record = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            _LOG.warning("Ignoring unreadable autotune cache %s: %s", path, exc)
            return None
        if (
            isinstance(record, dict)
            and record.get("schema") == _SCHEMA
            and record.get("key") == key
        ):
            for config in candidates:
                if record.get("config") == config.to_dict():
                    return config
        _LOG.warning("Ignoring incompatible autotune cache %s", path)
        return None

    @staticmethod
    def _write(path: Path, key: str, config: TuneConfig) -> None:
        """Atomically persist ``config`` for ``key``; on I/O errors log and keep memory."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"schema": _SCHEMA, "key": key, "config": config.to_dict()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            try:
                directory = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory = None
            if directory is not None:
                try:
                    os.fsync(directory)
                except OSError:
                    pass
                finally:
                    os.close(directory)
        except OSError as exc:
            _LOG.warning("Cannot persist autotune selection; using memory: %s", exc)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    _LOG.warning("Cannot remove temporary autotune record %s: %s", temporary, exc)
