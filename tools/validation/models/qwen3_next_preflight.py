#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Required four-GPU infrastructure checks, never full-model L2 acceptance."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from qwen3_next_contract import checkpoint_identity, evidence_record, result_envelope
from qwen3_next_runtime import (
    atomic_json,
    failure_fields,
    memory_limits,
    parse_memory_request,
    subprocess_failure,
    verify_checkpoint,
    verify_environment,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    progress = {
        "run_id": manifest["run_id"],
        "attempt": os.environ["QWEN3_NEXT_ATTEMPT"],
        "status": "running",
        "phase": "inventory",
        "full_model_l2": "not_evaluated",
    }

    def run(name, command, env=None):
        progress["phase"] = name
        atomic_json(output / "progress.json", progress)
        print(f"PREFLIGHT {name}: {command}", flush=True)
        log_path = output / f"{name}.log"
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if completed.returncode:
            raise subprocess_failure(completed.returncode, log_path)

    try:
        verify_environment(manifest["environment"])
        if checkpoint_identity(manifest["checkpoint"]) != manifest["checkpoint_identity"]:
            raise RuntimeError("checkpoint_mismatch: pinned metadata or shard inventory changed")
        import torch

        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "vllm",
                "triton",
                "transformers",
                "flashinfer-python",
                "safetensors",
            )
        }
        nvcc = subprocess.check_output(
            [f"{manifest['cuda_home']}/bin/nvcc", "--version"], text=True
        )
        inventory = {
            "source_commit": manifest["source_commit"],
            "environment_sha256": manifest["environment_sha256"],
            "visible_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "execution_gpus": 4,
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "versions": versions,
            "torch_cuda": torch.version.cuda,
            "nvcc": nvcc,
            "compiler": subprocess.check_output(["g++", "--version"], text=True),
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
            "gpu_topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
            "visible_gpus": [
                torch.cuda.get_device_properties(i).__repr__()
                for i in range(torch.cuda.device_count())
            ],
            "memory": memory_limits(),
            "memory_requested": manifest.get("memory"),
        }
        atomic_json(output / "environment.json", inventory)
        run("environment-lock", [sys.executable, "-m", "pip", "freeze", "--all"])
        if torch.cuda.device_count() != 4 or any(
            "B200" not in torch.cuda.get_device_name(i) for i in range(4)
        ):
            raise RuntimeError(
                "topology_mismatch: requires exactly four B200 GPUs in the execution step"
            )
        if "V13.0.88" not in nvcc or torch.version.cuda != "13.0":
            raise RuntimeError(
                "environment_mismatch: requires CUDA toolkit 13.0.88 and a CUDA 13.0 torch runtime"
            )
        for name, required in {
            "torch": "2.13.0",
            "vllm": "0.30.0",
            "triton": "3.7.1",
            "transformers": "5.17.0",
        }.items():
            if versions[name].split("+", 1)[0] != required:
                raise RuntimeError(
                    f"environment_mismatch: unverified runtime version: {name}={versions[name]}"
                )
        memory = inventory["memory"]
        quota = memory["effective_bytes"]
        # Compare against what THIS run requested, recorded by the submitter, so
        # the budget is not a second constant that has to be kept in sync -- and
        # so a trimmed allocation fails loudly instead of passing a stale bound.
        if "memory" not in manifest:
            raise RuntimeError("Manifest does not record the memory request")
        requested = parse_memory_request(manifest["memory"])
        if quota is None or quota < requested:
            raise RuntimeError(
                f"insufficient_memory: node granted less memory than this run requested "
                f"({manifest['memory']} = {requested} bytes): "
                f"effective={quota}, cgroup={memory['cgroup_bytes']}, "
                f"declared={memory['declared_bytes']}"
            )
        # A grant is not availability on a shared node. Fail in seconds rather
        # than being OOM-killed minutes into a 162 GB model load.
        available = memory["available_bytes"]
        if available is None or available < requested:
            raise RuntimeError(
                f"insufficient_memory: node does not currently have the requested memory free "
                f"({manifest['memory']} = {requested} bytes): "
                f"MemAvailable={available}. Co-tenants can hold host RAM that an "
                "allocation nominally grants; use an exclusive allocation or wait "
                "for the node to drain."
            )
        run("build", [sys.executable, "setup.py", "build_ext", "--inplace"])
        env = {
            **os.environ,
            "PY": sys.executable,
            "QWEN3_NEXT_EVIDENCE_DIR": str(output / "operators"),
            "QWEN3_NEXT_OPERATOR_SUITE": manifest.get("suite", "all"),
        }
        run("operators", ["bash", "ci/scripts/run_qwen3_next_operator_gates.sh"], env)
        if manifest["stage"] == "preflight" or manifest.get("suite") in ("norm", "all"):
            run(
                "device-guards",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/models/qwen3_next/check_qwen3_next_multigpu.py",
                    f"--junitxml={output}/device-guards.xml",
                ],
            )
            from xml.etree import ElementTree

            suites = list(
                ElementTree.parse(output / "device-guards.xml").getroot().iter("testsuite")
            )
            if not suites or any(
                int(s.get(k, 0)) for s in suites for k in ("skipped", "errors", "failures")
            ):
                raise RuntimeError("Device guards must pass without skips")
        checkpoint = None
        if manifest["stage"] == "preflight":
            run(
                "tp4",
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=4",
                    "tools/validation/models/qwen3_next_tp_probe.py",
                    "--output",
                    str(output / "tp4"),
                ],
            )
            ranks = [
                json.loads((output / "tp4" / f"rank-{rank}.json").read_text()) for rank in range(4)
            ]
            if any(
                row.get("status") != "passed"
                or row.get("rank") != rank
                or row.get("world_size") != 4
                for rank, row in enumerate(ranks)
            ):
                raise RuntimeError("All four TP ranks must pass")
            atomic_json(output / "tp4.json", {"status": "passed", "ranks": ranks})
            progress["phase"] = "checkpoint-integrity"
            atomic_json(output / "progress.json", progress)
            identity = json.loads(
                Path("rl_engine/validation/models/qwen3_next_norm_manifest.json").read_text()
            )
            checkpoint = verify_checkpoint(Path(manifest["checkpoint"]), identity["model_identity"])
            atomic_json(output / "checkpoint.json", checkpoint)
            run(
                "rollout-smoke",
                [
                    sys.executable,
                    "tools/validation/models/qwen3_next_rollout_smoke.py",
                    "--model",
                    manifest["checkpoint"],
                    "--output",
                    str(output / "rollout-smoke.json"),
                    "--tensor-parallel-size",
                    "4",
                    "--non-strict-smoke",
                ],
            )
        import rl_engine._C
        from rl_engine.integrations.engines.train.vllm.qwen3_next_gdn import _provider

        provider = _provider()
        names = {"operators": "operators/summary.json", "environment": "environment.json"}
        if manifest["stage"] == "preflight" or manifest.get("suite") in ("norm", "all"):
            names["device_guards"] = "device-guards.xml"
        if manifest["stage"] == "preflight":
            names.update(
                tp4="tp4.json", checkpoint="checkpoint.json", rollout_smoke="rollout-smoke.json"
            )
        evidence = {
            name: evidence_record(output / filename, args.manifest.parent)
            for name, filename in names.items()
        }
        result = result_envelope(manifest, os.environ["QWEN3_NEXT_ATTEMPT"], evidence=evidence)
        result.update(
            checkpoint=checkpoint,
            provider_paths={
                "norm": rl_engine._C.__file__,
                "gdn": f"{provider.__module__}.{provider.__name__}",
            },
        )
        atomic_json(output / "result.json", result)
    except Exception as exc:
        progress.update(
            status="failed", attempt=os.environ["QWEN3_NEXT_ATTEMPT"], **failure_fields(exc)
        )
        atomic_json(output / "result.json", progress)
        raise


if __name__ == "__main__":
    main()
