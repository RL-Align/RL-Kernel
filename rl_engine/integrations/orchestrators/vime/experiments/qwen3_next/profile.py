# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Activate only the bounded CUSTOM profile after verifying native layer evidence.

Capability is process-local and false by default. Qualification runs use BI=0;
strict runs use BI=1 plus the immutable predecessor certificate in their model
identity. Generic vLLM GDN_ATTN registration is never modified.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_CERTIFIED = None
ROOT = Path(__file__).resolve().parents[6]


def profile_is_certified():
    return _CERTIFIED is not None and os.environ.get("VLLM_BATCH_INVARIANT") == "1"


def validate_certified_weights(identity, certified):
    """The base certificate is not a certificate for newly updated weight bytes."""
    from rl_engine.validation.models.qwen3_next_l2 import validate_weight_identity

    validate_weight_identity(identity)
    if identity.get("weight_version", "baseline") == "baseline":
        if any(
            identity.get(key) != certified.get(key) for key in ("checkpoint", "checkpoint_identity")
        ):
            raise ValueError("Baseline weights differ from the certified official checkpoint")
        return
    if any(
        (
            identity.get("base_checkpoint") != certified.get("checkpoint"),
            identity.get("base_checkpoint_identity") != certified.get("checkpoint_identity"),
        )
    ):
        raise ValueError("Updated export refers to a different certified official base")
    from qwen3_next_training_export import validate_training_export

    verified = validate_training_export(identity["checkpoint"], identity["checkpoint_identity"])
    if verified != identity["checkpoint_identity"]:
        raise ValueError("Actual updated weight files differ from the pinned export identity")
    training = verified["training_identity"]
    fields = (
        "source_commit",
        "source_tree",
        "environment_sha256",
        "provider_config",
        "topology",
        "checkpoint",
        "checkpoint_identity",
    )
    if any(training.get(key) != certified.get(key) for key in fields):
        raise ValueError(
            "Updated export training provenance differs from the certified base profile"
        )


def activate_for_identity(identity):
    global _CERTIFIED
    certificate = identity.get("scheduler_profile_certificate")
    mode = os.environ.get("VLLM_BATCH_INVARIANT")
    if certificate is None:
        if mode != "0" or _CERTIFIED is not None:
            raise ValueError("Uncertified shared providers require explicit qualification BI=0")
        return "qualification"
    if mode != "1":
        raise ValueError("A certified shared profile must execute strict BI=1")
    sys.path.insert(0, str(ROOT / "tools/validation/models"))
    from qwen3_next_contract import validate_result
    from qwen3_next_l2 import model_identity
    from qwen3_next_runtime import file_digest, identity_digest, verify_environment

    record_key = identity_digest({"certificate": certificate, "identity": identity})
    if _CERTIFIED == record_key:
        return "certified"
    if _CERTIFIED is not None:
        raise ValueError("A process cannot switch its certified shared provider identity")
    path = Path(certificate["path"])
    if (
        not path.is_absolute()
        or path.name != "result.json"
        or file_digest(path) != certificate["sha256"]
    ):
        raise ValueError("Shared profile certificate path/content identity mismatch")
    if os.environ.get("QWEN_SHARED_PROFILE_RESULT", str(path)) != str(path):
        raise ValueError("Scheduler certificate environment disagrees with the worker request")
    manifest = json.loads((path.parent / "run.json").read_text())
    result = json.loads(path.read_text())
    if manifest.get("stage") != "layers":
        raise ValueError("Only completed native layer/scheduler evidence can certify CUSTOM")
    certified = model_identity(manifest)
    fields = (
        "source_commit",
        "source_tree",
        "environment_sha256",
        "provider_config",
        "topology",
        "architecture_layers",
        "weights_dtype",
        "recurrent_state_dtype",
    )
    if any(identity.get(key) != certified.get(key) for key in fields):
        raise ValueError(
            "Scheduler certificate belongs to another source/environment/provider profile"
        )
    validate_certified_weights(identity, certified)
    provider = identity["provider_config"]
    if provider.get("profile") != "qwen3_next_tp4_shared_eager_v1":
        raise ValueError("Only the explicit Qwen3-Next TP4 eager profile can be certified")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
        != identity["source_tree"]
    ):
        raise ValueError("Current source tree differs from the certified provider")
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode:
        raise ValueError("Current provider source has uncommitted changes")
    for name, expected in provider["sources"].items():
        if file_digest(ROOT / name) != expected:
            raise ValueError(f"Certified provider source changed: {name}")
    for name, expected in provider["runtime_env"].items():
        if os.environ.get(name) != expected:
            raise ValueError(f"Certified runtime setting differs: {name}")
    cuda = manifest["cuda_identity"]
    if (
        os.environ.get("CUDA_HOME") != cuda["home"]
        or file_digest(Path(cuda["home"]) / "bin/nvcc") != cuda["nvcc_sha256"]
    ):
        raise ValueError("CUDA toolkit differs from the certified environment")
    verify_environment(manifest["environment"], require_venv=True)
    validate_result(manifest, result, run_dir=path.parent)
    _CERTIFIED = record_key
    return "certified"


def activate_profile_from_environment(identity):
    """Compatibility spelling; a path without its pinned request digest is insufficient."""
    return activate_for_identity(identity)
