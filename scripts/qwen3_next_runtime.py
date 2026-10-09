# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Scheduler-independent runtime for the Qwen3-Next stage runners.

Environment identity, failure classification, memory preconditions, owned child
processes, checkpoint verification, stage ordering and crash-recoverable
training storage. Nothing here talks to a batch scheduler: a run is identified
by the ``execution_id`` that ``scripts/qwen3_next_run.py`` writes into
``run.json``, and any allocation that provides four GPUs can host it.
"""

import errno
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import site
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

# --------------------------------------------------------------------------- #
# Environment identity
# --------------------------------------------------------------------------- #


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def interpreter_path(path):
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python interpreter is not executable: {path}")
    return str(path)


def source_identity(path):
    path = Path(path)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Editable Python source has no auditable Git identity: {path}")
    root = result.stdout.strip()
    tree = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    patch = subprocess.check_output(["git", "-C", root, "diff", "--binary", "HEAD"])
    return {"path": root, "tree": tree, "patch_sha256": hashlib.sha256(patch).hexdigest()}


def installed_provider_sources(distributions):
    """Pin provider code inside a wheel too; unchanged metadata is insufficient.

    Editable frameworks are separately identified by Git tree and patch. vLLM
    is an installed wheel in the shared environment, and local patches to its
    numerical kernels or actual scheduler must invalidate the environment lock.
    No framework import or GPU initialization is needed for this inventory.
    """
    prefixes = (
        "vllm/model_executor/determinism/",
        "vllm/model_executor/layers/mamba/",
        "vllm/model_executor/layers/fused_moe/",
        "vllm/model_executor/models/qwen3_next.py",
        "vllm/third_party/flash_linear_attention/",
        "vllm/v1/attention/",
        "vllm/v1/core/sched/",
        "vllm/v1/worker/",
        "vllm/v1/sample/",
        "vllm/config/",
        "vllm/compilation/",
        "vllm/distributed/",
    )
    records = {}
    for dist in distributions:
        if dist.metadata.get("Name", "").lower().replace("_", "-") != "vllm":
            continue
        files = dist.files
        if not files:
            raise ValueError("Installed vLLM has no auditable provider file inventory")
        selected = sorted(
            str(path)
            for path in files
            if str(path).endswith(".py") and str(path).startswith(prefixes)
        )
        required = "vllm/model_executor/determinism/batch_invariant.py"
        if required not in selected:
            raise ValueError("Installed vLLM omits the pinned shared GEMM provider")
        for name in selected:
            path = Path(dist.locate_file(name))
            records[name] = {"path": str(path.absolute()), "sha256": file_digest(path)}
    return records


def installation_roots():
    """The interpreter's site directories, independent of what has been imported.

    ``sys.path`` grows at runtime: importing setuptools appends its ``_vendor``
    directory, after which ``importlib.metadata.distributions()`` also lists the
    vendored packages. An identity read through ``sys.path`` therefore depends on
    import order, and a worker that imports the model before verifying would
    disagree with the manifest written by a fresh interpreter. Enumerate the fixed
    site roots instead; in a fresh interpreter they are exactly the ``site-packages``
    entries of ``sys.path``.
    """
    roots = {Path(sysconfig.get_path("purelib")), Path(sysconfig.get_path("platlib"))}
    roots.update(Path(p) for p in site.getsitepackages())
    if site.ENABLE_USER_SITE:
        roots.add(Path(site.getusersitepackages()))
    return sorted(root for root in roots if root.is_dir())


def current_identity():
    distributions = []
    sources = {}
    roots = installation_roots()
    installed = list(importlib.metadata.distributions(path=[str(root) for root in roots]))
    for dist in installed:
        if dist.metadata.get("Name", "").lower().replace("_", "-") in {"rl-engine", "rl-kernel"}:
            continue
        direct = dist.read_text("direct_url.json")
        if direct:
            direct_value = json.loads(direct)
            if direct_value.get("dir_info", {}).get("editable"):
                location = Path(unquote(urlparse(direct_value["url"]).path))
                # This project is loaded from the frozen source snapshot instead.
                if dist.metadata.get("Name", "").lower().replace("_", "-") != "rl-engine":
                    sources[str(location)] = source_identity(location)
        distributions.append(
            {
                "name": dist.metadata.get("Name"),
                "version": dist.version,
                "metadata_sha256": identity_digest(dist.read_text("METADATA")),
                "direct_url": direct,
            }
        )
    pth = {}
    for root in roots:
        for path in sorted(root.glob("*.pth")):
            pth[str(path)] = file_digest(path)
            for line in path.read_text().splitlines():
                location = Path(line)
                if (
                    line
                    and not line.startswith(("#", "import "))
                    and location.is_absolute()
                    and location.is_dir()
                ):
                    if (location / ".git").exists():
                        sources[str(location)] = source_identity(location)
    config = Path(sys.prefix) / "pyvenv.cfg"
    ninja = next(
        (
            Path(prefix) / "bin/ninja"
            for prefix in (sys.prefix, sys.base_prefix)
            if (Path(prefix) / "bin/ninja").is_file()
        ),
        None,
    )
    build_tools = {"ninja": {"path": str(ninja), "sha256": file_digest(ninja)} if ninja else None}
    identity = {
        "executable": sys.executable,
        "binary": str(Path(sys.executable).resolve()),
        "binary_sha256": file_digest(sys.executable),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "version": sys.version,
        "pyvenv_cfg_sha256": file_digest(config) if config.is_file() else None,
        "distributions": sorted(
            distributions, key=lambda item: (item["name"] or "", item["version"])
        ),
        "build_tools": build_tools,
        "pth": pth,
        "sources": sources,
        "installed_provider_sources": installed_provider_sources(installed),
    }
    return {"sha256": identity_digest(identity), "identity": identity}


def probe_environment(python):
    command = [interpreter_path(python), str(Path(__file__).absolute()), "--json"]
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    output = subprocess.check_output(command, text=True, env=env, timeout=180)
    return json.loads(output)


def verify_environment(expected, *, require_venv=False):
    actual = current_identity()
    if actual != expected:
        raise RuntimeError(
            "environment_mismatch: interpreter, dependencies or editable source changed"
        )
    if require_venv and sys.prefix == sys.base_prefix:
        raise RuntimeError("environment_mismatch: training/recompute requires the isolated venv")
    return actual


if __name__ == "__main__":
    print(json.dumps(current_identity(), sort_keys=True))


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #


FAILURE_KINDS = frozenset(
    {
        "environment_mismatch",
        "source_mismatch",
        "checkpoint_mismatch",
        "topology_mismatch",
        "insufficient_memory",
        "insufficient_disk",
        "out_of_memory",
        "time_limit",
        "unknown",
    }
)


class ProcessFailure(RuntimeError):
    def __init__(self, message, *, failure_kind):
        super().__init__(message)
        self.failure_kind = failure_kind


def failure_kind(error=None, *, text=""):
    explicit = getattr(error, "failure_kind", None)
    if explicit in FAILURE_KINDS:
        return explicit
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return "time_limit"
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "insufficient_disk"
    if type(error).__name__ == "OutOfMemoryError":
        return "out_of_memory"
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "environment_mismatch"
    detail = f"{type(error).__name__}: {error}\n{text}" if error is not None else text
    for kind in sorted(FAILURE_KINDS - {"unknown", "out_of_memory", "time_limit"}):
        if re.search(rf"\b{kind}\s*:", detail):
            return kind
    if re.search(r"\bOutOfMemoryError\s*:|CUDA out of memory", detail, re.IGNORECASE):
        return "out_of_memory"
    if re.search(r"\bENOSPC\b|\[Errno 28\]|No space left on device", detail):
        return "insufficient_disk"
    if re.search(r"\b(?:ModuleNotFoundError|ImportError)\s*:", detail):
        return "environment_mismatch"
    if re.search(r"\b(?:TimeoutExpired|TimeoutError|time_limit)\s*:", detail):
        return "time_limit"
    # Signals, exit 137, MemoryError and a bare 'Killed' are not proof of CUDA OOM.
    return "unknown"


def failure_fields(error):
    return {"failure_kind": failure_kind(error), "error": f"{type(error).__name__}: {error}"}


def subprocess_failure(code, log_path):
    with Path(log_path).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        tail = stream.read().decode("utf-8", errors="replace")
    return ProcessFailure(
        f"Native model process failed with exit {code}; see {log_path}\n{tail}",
        failure_kind=failure_kind(text=tail),
    )


# --------------------------------------------------------------------------- #
# Memory, checkpoint and atomic files
# --------------------------------------------------------------------------- #


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


_MEMORY_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_memory_request(text):
    """A memory size (e.g. ``1280G``) in bytes; a bare number is MB."""
    text = str(text).strip()
    if not text:
        raise RuntimeError("Empty memory request")
    scale = _MEMORY_SUFFIXES.get(text[-1].upper())
    digits = text[:-1] if scale else text
    if not digits.isdigit():
        raise RuntimeError(f"Unparseable memory request: {text!r}")
    return int(digits) * (scale if scale else 1024**2)


def available_memory_bytes():
    """MemAvailable, i.e. what the kernel thinks can be had without swapping."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def memory_limits(environ=None):
    """Every known memory ceiling, and what is free right now.

    ``effective_bytes`` is the smallest of the cgroup limit and the allocation
    the launcher declares in ``QWEN3_NEXT_ALLOCATED_MEMORY`` (``--memory``).
    It is None only when neither is known, which the stages treat as a hard
    failure: host free memory alone is never taken as a grant.
    ``available_bytes`` is a snapshot of MemAvailable; on a shared node a grant
    is not an availability guarantee, so both are checked.
    """
    environ = os.environ if environ is None else environ
    records = {}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if controllers == "":
            root, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):
            root, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        current = root / relative.lstrip("/")
        while current.is_relative_to(root):
            path = current / name
            if path.is_file():
                records[str(path)] = path.read_text().strip()
            if current == root:
                break
            current = current.parent
    finite = [int(value) for value in records.values() if value.isdigit() and int(value) < 2**60]
    declared = environ.get("QWEN3_NEXT_ALLOCATED_MEMORY")
    sources = {
        "cgroup_bytes": min(finite) if finite else None,
        "declared_bytes": parse_memory_request(declared) if declared else None,
    }
    known = [value for value in sources.values() if value is not None]
    return {
        "limits": records,
        **sources,
        "effective_bytes": min(known) if known else None,
        "available_bytes": available_memory_bytes(),
    }


def verify_checkpoint(checkpoint, identity):
    if checkpoint.name != identity["revision"]:
        raise RuntimeError("Checkpoint revision does not match the pinned manifest")
    config = json.loads((checkpoint / "config.json").read_text())
    for key, value in identity["config_fingerprint"].items():
        if config.get(key) != value:
            raise RuntimeError(f"Checkpoint architecture mismatch: {key}")
    snapshot = identity["weight_snapshot"]
    if digest(checkpoint / snapshot["index_file"]) != snapshot["index_sha256"]:
        raise RuntimeError("Checkpoint index checksum mismatch")
    for shard in snapshot["shards"]:
        path = checkpoint / shard["filename"]
        if path.stat().st_size != shard["size_bytes"] or digest(path) != shard["sha256"]:
            raise RuntimeError(f"Checkpoint shard checksum mismatch: {path.name}")
    return {
        "model_id": identity["model_id"],
        "revision": identity["revision"],
        "content_hash": snapshot["content_hash"],
        "shards_verified": len(snapshot["shards"]),
        "metadata_sha256": {p.name: digest(p) for p in sorted(checkpoint.glob("*.json"))},
    }


# --------------------------------------------------------------------------- #
# Owned child processes
# --------------------------------------------------------------------------- #


def signal_group(child, signum):
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def wait_owned_child(child, interrupted, *, grace_seconds=10):
    deadline = None
    while True:
        if interrupted() and deadline is None:
            signal_group(child, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
        if deadline is not None and time.monotonic() >= deadline:
            # Kill the group even if its leader exited, so grandchildren cannot linger.
            signal_group(child, signal.SIGKILL)
            return child.wait(timeout=10)
        code = child.poll()
        if code is not None and deadline is None:
            return code
        time.sleep(0.05)


# --------------------------------------------------------------------------- #
# Stage order
# --------------------------------------------------------------------------- #

STAGES = {
    "preflight": ("256G", "01:00:00", None, "scripts/qwen3_next_preflight.py"),
    "layers": ("512G", "04:00:00", "preflight", "scripts/qwen3_next_layers.py"),
    "l2": ("512G", "04:00:00", "layers", "scripts/qwen3_next_l2.py"),
    # The training acceptance is split so no single job holds a node for a day:
    # train1 = update1 + export + L2, train2 = update2 + export + L2 (continues
    # train1's durable state), train = reload + L2 + continuous/resumed controls.
    "train1": ("1280G", "06:00:00", "l2", "scripts/qwen3_next_training.py"),
    "train2": ("1280G", "07:00:00", "train1", "scripts/qwen3_next_training.py"),
    "train": ("1280G", "09:00:00", "train2", "scripts/qwen3_next_training.py"),
    # Batch invariance branches off a passed l2, under the same certified profile.
    "bi": ("512G", "06:00:00", "l2", "scripts/qwen3_next_batch_invariance.py"),
}


def require_predecessor(stage, previous, checkpoint, expected=None):
    """The previous stage's run passed, finished, and pinned the same identity."""
    from qwen3_next_contract import acceptance_identity, checkpoint_identity, validate_result

    predecessor = STAGES[stage][2]
    if predecessor is None:
        return
    if previous is None:
        raise ValueError(f"{stage} requires a passed {predecessor} run")
    manifest = json.loads((previous / "run.json").read_text())
    result = json.loads((previous / "result.json").read_text())
    if manifest["stage"] != predecessor or result.get("status") != "passed":
        raise ValueError("predecessor stage has not passed")
    if (previous / "RUNNING").exists():
        raise ValueError("predecessor run is still executing")
    validate_result(manifest, result, run_dir=previous)
    if manifest["checkpoint_identity"] != checkpoint_identity(checkpoint):
        raise ValueError("predecessor checkpoint identity mismatch")
    if expected is not None and acceptance_identity(manifest) != acceptance_identity(expected):
        raise ValueError("predecessor source/environment/provider/topology identity mismatch")


# --------------------------------------------------------------------------- #
# Training storage
# --------------------------------------------------------------------------- #


def _boundary(name):
    """CPU fault-injection seam; production deliberately performs no action."""


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or external storage ownership record: {path}")
    return json.loads(path.read_text())


def _write(path, value):
    if path.is_symlink() or path.with_suffix(path.suffix + ".partial").is_symlink():
        raise ValueError("Storage intent cannot follow a symlink")
    atomic_json(path, value)


def _root_record(root, identity):
    return {
        "schema_version": 1,
        "scope": "qwen3_next_owned_storage",
        "root": str(root),
        "identity": identity,
    }


def owned_root(root, identity):
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError("Training storage root cannot be a symlink")
    root = root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    parent_owner = root.parent / f".{root.name}.owner.json"
    expected = _root_record(root, identity)
    if parent_owner.exists():
        if _read(parent_owner) != expected:
            raise ValueError("Training storage belongs to another run")
    else:
        # Existing deployments can migrate only with their original owner.
        # An interrupted new creation has no directory until parent proof lands.
        if root.exists() and any(root.iterdir()):
            if _read(root / "RUN_OWNER.json") != identity:
                raise ValueError("Existing training files lack a matching run owner")
        _write(parent_owner, expected)
    _boundary("root_owner_persisted")
    root.mkdir(exist_ok=True)
    _sync(root.parent)
    _boundary("root_created")
    owner = root / "RUN_OWNER.json"
    if owner.exists() and _read(owner) != identity:
        raise ValueError("Internal training owner disagrees with durable parent proof")
    _write(owner, identity)
    return root


def require_owner(root, identity):
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Training storage must be a direct owned directory")
    root = root.resolve()
    if _read(root.parent / f".{root.name}.owner.json") != _root_record(root, identity):
        raise ValueError("Training storage lacks matching durable parent ownership")
    return root


def _child(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Storage intent must name one direct child")
    path = root / name
    if path.is_symlink():
        raise ValueError("Storage intents cannot follow target symlinks")
    return path


def create_staging(root, destination_name, metadata):
    if metadata.get("status") != "staging":
        raise ValueError("Staging creation needs an explicit unpublished status")
    root = require_owner(root, metadata["identity"])
    destination = _child(root, destination_name)
    if destination.name.startswith(".") or destination.suffix:
        raise ValueError("Published storage names must remain visible and have no suffix")
    staging = destination.with_suffix(".partial")
    if destination.exists() or staging.exists():
        raise ValueError("A staging intent cannot adopt an existing directory")
    intent = root / f".{destination.name}.staging.json"
    value = {
        "schema_version": 1,
        "scope": "qwen3_next_staging_intent",
        "identity": metadata["identity"],
        "staging": staging.name,
        "published": destination.name,
        "metadata": metadata,
    }
    if intent.exists():
        raise ValueError("Staging names cannot reuse an earlier durable intent")
    _write(intent, value)
    _boundary("staging_intent_persisted")
    staging.mkdir()
    _sync(root)
    _boundary("staging_created")
    _write(staging / "STAGING.json", metadata)
    _boundary("staging_marker_persisted")
    return staging, destination


def _finish_retirement(root, intent, record, *, recovering=False):
    target, tombstone = _child(root, record["target"]), _child(root, record["tombstone"])
    if target.exists() and tombstone.exists():
        raise ValueError("Retirement has conflicting original and tombstone directories")
    if record["status"] == "retired":
        if target.exists() or tombstone.exists():
            raise ValueError("Retired storage name was unexpectedly reused")
        return record
    if "successor" in record["proof"]:
        successor = Path(record["proof"]["successor"])
        if successor.parent != root or successor.is_symlink() or successor == target:
            raise ValueError("Retirement successor escaped its owning run")
        if recovering:
            from qwen3_next_checkpoint import validate_checkpoint

            if (
                validate_checkpoint(successor, record["identity"])
                != record["proof"]["successor_complete"]
            ):
                raise ValueError("Retirement successor no longer matches its verified publication")
    if target.exists():
        marker, expected, optional = _retirement_marker(record)
        marker = target / marker
        if marker.exists() or marker.is_symlink() or not optional:
            if _read(marker) != expected:
                raise ValueError("Retirement target differs from its durable ownership proof")
        target.rename(tombstone)
        _sync(root)
        _boundary("retirement_renamed")
    if tombstone.exists():
        shutil.rmtree(tombstone)
        _sync(root)
        _boundary("retirement_removed")
    record = {**record, "status": "retired"}
    _write(intent, record)
    _boundary("retirement_committed")
    return record


def retire_directory(path, root, identity, *, proof):
    root = require_owner(root, identity)
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError("Retirement cannot follow a target symlink")
    path = path.resolve()
    if path.parent != root or path.name.startswith("."):
        raise ValueError("Retirement target must be a visible direct child of this run")
    target = _child(root, path.name)
    intent = root / f".{target.name}.delete.json"
    recovering = intent.exists()
    if recovering:
        record = _read(intent)
        _validate_retirement(record, target.name, identity)
        if record["proof"] != proof:
            raise ValueError("Retirement retry changed its original ownership proof")
    else:
        if not target.is_dir():
            raise ValueError("Cannot authorize retirement of an unknown directory")
        record = {
            "schema_version": 1,
            "scope": "qwen3_next_retirement",
            "identity": identity,
            "target": target.name,
            "tombstone": f".{target.name}.deleting",
            "status": "retiring",
            "proof": proof,
        }
        _validate_retirement(record, target.name, identity)
        _write(intent, record)
        _boundary("retirement_intent_persisted")
    return _finish_retirement(root, intent, record, recovering=recovering)


def _validate_retirement(record, target, identity):
    if (
        record.get("schema_version") != 1
        or record.get("scope") != "qwen3_next_retirement"
        or record.get("identity") != identity
        or record.get("target") != target
        or target.startswith(".")
        or record.get("tombstone") != f".{target}.deleting"
        or record.get("status") not in ("retiring", "retired")
        or not isinstance(record.get("proof"), dict)
    ):
        raise ValueError("Retirement intent belongs to another target/run or is malformed")
    _retirement_marker(record)


def _retirement_marker(record):
    """Bind a destructive intent to one of the supported ownership proofs."""
    proof, identity = record["proof"], record["identity"]
    if proof.get("scope") == "qwen3_next_staging_intent":
        metadata = proof.get("metadata", {})
        published = proof.get("published")
        if (
            proof.get("schema_version") != 1
            or proof.get("identity") != identity
            or proof.get("staging") != record["target"]
            or not isinstance(published, str)
            or Path(published).name != published
            or published.startswith(".")
            or Path(published).suffix
            or str(Path(published).with_suffix(".partial")) != record["target"]
            or metadata.get("identity") != identity
            or metadata.get("status") != "staging"
        ):
            raise ValueError("Staging retirement proof does not own its target")
        return "STAGING.json", metadata, True
    if proof.get("status") == "staging" and proof.get("identity") == identity:
        if Path(record["target"]).suffix != ".partial":
            raise ValueError("Legacy staging proof cannot authorize a published directory")
        return "STAGING.json", proof, False
    complete, successor = proof.get("complete", {}), proof.get("successor_complete", {})
    if (
        complete.get("identity") == identity
        and complete.get("status") == "complete"
        and successor.get("identity") == identity
        and successor.get("status") == "complete"
        and type(complete.get("step")) is int
        and type(successor.get("step")) is int
        and complete["step"] < successor["step"]
        and isinstance(proof.get("successor"), str)
    ):
        return "COMPLETE.json", complete, False
    exported = proof.get("export", {})
    if exported.get("training_identity") == identity and exported.get("status") == "complete":
        return "EXPORT_COMPLETE.json", exported, False
    raise ValueError("Retirement lacks a supported same-run ownership proof")


def recover_storage(root, identity):
    root = require_owner(root, identity)
    recovered = []
    for intent in sorted(root.glob(".*.delete.json")):
        record = _read(intent)
        target = record.get("target")
        _child(root, target)
        _validate_retirement(record, target, identity)
        if intent.name != f".{target}.delete.json":
            raise ValueError("Retirement filename contradicts its target")
        if record["status"] == "retiring":
            recovered.append(_finish_retirement(root, intent, record, recovering=True))
        else:
            _finish_retirement(root, intent, record)
    for intent in sorted(root.glob(".*.staging.json")):
        record = _read(intent)
        if (
            record.get("schema_version") != 1
            or record.get("scope") != "qwen3_next_staging_intent"
            or record.get("identity") != identity
            or record.get("metadata", {}).get("identity") != identity
            or record.get("metadata", {}).get("status") != "staging"
        ):
            raise ValueError("Staging intent belongs to another run or is malformed")
        staging, published = _child(root, record["staging"]), _child(root, record["published"])
        if (
            published.name.startswith(".")
            or published.suffix
            or staging != published.with_suffix(".partial")
            or intent.name != f".{published.name}.staging.json"
        ):
            raise ValueError("Staging intent paths do not describe one publication")
        if staging.exists() and published.exists():
            raise ValueError("Staging and published copies cannot coexist")
        if staging.exists():
            marker = staging / "STAGING.json"
            if marker.exists() and _read(marker) != record["metadata"]:
                raise ValueError("Staging marker differs from its durable parent intent")
            recovered.append(retire_directory(staging, root, identity, proof=record))
    # A marker from the earlier protocol can prove ownership, but an arbitrary
    # directory with neither marker nor parent intent must never be deleted.
    for staging in sorted(root.glob("*.partial")):
        if not staging.is_dir():
            continue
        metadata = _read(staging / "STAGING.json")
        if metadata.get("identity") != identity or metadata.get("status") != "staging":
            raise ValueError("Interrupted state belongs to another run or lacks a matching owner")
        recovered.append(retire_directory(staging, root, identity, proof=metadata))
    return recovered


def training_storage_root(manifest, run_root):
    """The run's checkpoint and export root: ``--training-storage`` or ``<run>/training-state``."""
    shared = manifest.get("training_storage")
    return Path(shared) if shared else Path(run_root) / "training-state"
