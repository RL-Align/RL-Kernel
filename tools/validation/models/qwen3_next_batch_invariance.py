#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Stage ``bi``: the Qwen3-Next batch-invariance matrix (RFC #428 ``batch_invariance_ci``).

Runs after a passed ``l2`` stage, under the same certified provider profile
(``VLLM_BATCH_INVARIANT=1``). The cells are ``rl_engine/validation/models/qwen3_next_bi_cases.py``
(pinned by ``qwen3_next_bi_manifest.json``); ``bi_suite: quick`` in the manifest
runs the CI subset. Two independent worker processes, as in ``l2``:

1. rollout: every rollout cell and its reference (the target alone, same chunk
   plan) in one native vLLM engine; the target generates its sixteen tokens;
2. replay: for every ``chunk`` cell, the VIME recompute of the target's tokens;
   for every ``packing`` cell, the target packed with its neighbours and alone.

Then, bitwise on response tokens, logits, logprobs, the embedding and every
layer, sub-node and state digest: concurrency and interference cells against
their reference, chunk cells against their replay, packing cells against the
replay alone. ``bi-matrix.json`` records every cell, the first divergence and
its layer family.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/validation/models"))

from qwen3_next_contract import evidence_record, result_envelope, validate_result  # noqa: E402
from qwen3_next_l2 import (  # noqa: E402
    check_ready,
    model_identity,
    run_process,
    verify_before_model_load,
    wait_gpu_release,
)
from qwen3_next_runtime import (  # noqa: E402
    atomic_json,
    failure_fields,
    file_digest,
    memory_limits,
    parse_memory_request,
    require_predecessor,
    verify_environment,
)

from rl_engine.validation.models import qwen3_next_bi_cases as bi  # noqa: E402
from rl_engine.validation.models.qwen3_next_binary import (  # noqa: E402
    archive_provider_binary,
    validate_binary_record,
    verify_imported_provider,
)
from rl_engine.validation.models.qwen3_next_chain_report import family_of  # noqa: E402
from rl_engine.validation.models.qwen3_next_l2 import (  # noqa: E402
    FROZEN_TEXTS,
    compare_artifacts,
    compare_batch_prefill,
    first_divergent_node,
    load_tensor_artifact,
    validate_model_artifact,
)


def suite_cells(manifest):
    suite = manifest.get("bi_suite", "full")
    if suite not in ("full", "quick"):
        raise ValueError(f"Unknown batch-invariance suite {suite!r}")
    bi.load_manifest()
    return bi.quick_cells() if suite == "quick" else bi.cells()


def encoded_texts(manifest):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["checkpoint"], local_files_only=True, trust_remote_code=False
    )
    return [tokenizer.encode(text, add_special_tokens=False) for text in FROZEN_TEXTS]


def packing_source(cell):
    """The rollout that defines a packing cell's target tokens: the target alone, 64/64."""
    return bi.Cell(
        cell_id=cell.cell_id,
        family=cell.family,
        target=cell.target,
        first_chunk=64,
        chunk_size=64,
    ).reference()


def rollout_requests(manifest, cells):
    """References first, each once; then every rollout cell."""
    texts, identity = encoded_texts(manifest), model_identity(manifest)
    requests, seen = [], set()
    for cell in cells:
        reference = packing_source(cell) if cell.engine == "replay" else cell.reference()
        for item in (reference, cell):
            if item.engine == "rollout" and item.cell_id not in seen:
                seen.add(item.cell_id)
                requests.append(bi.rollout_request(item, texts, identity))
    return requests


def replay_requests(manifest, cells, rollout_dir):
    """Chunk-cell replays and packing cells (each with its unpacked replay)."""
    texts, identity = encoded_texts(manifest), model_identity(manifest)
    requests, seen = [], set()
    for cell in cells:
        if cell.family not in ("chunk", "packing"):
            continue
        source = cell.cell_id if cell.family == "chunk" else packing_source(cell).cell_id
        artifact = load_tensor_artifact(rollout_dir / f"{source}.pt")
        prompt = artifact["prompt_tokens"][0].tolist()
        responses = artifact["response_tokens"][0].tolist()
        alone = f"replay-{source}"
        if alone not in seen:
            seen.add(alone)
            requests.append(bi.replay_request(alone, prompt, responses, identity))
        if cell.family == "packing":
            layout = bi.packing_layout(cell, texts, prompt + responses[:-1])
            requests.append(
                bi.replay_request(cell.cell_id, prompt, responses, identity, packing=layout)
            )
    return requests


def worker(role, manifest_path, output, provider_binary):
    """One independent process: run every request of ``role`` into ``output``."""
    manifest = json.loads(Path(manifest_path).read_text())
    module, _capabilities = check_ready()
    verify_environment(manifest["environment"], require_venv=True)
    output.mkdir(parents=True, exist_ok=True)
    verify_before_model_load(manifest, output)
    actual_binary = verify_imported_provider(provider_binary, source_tree=manifest["source_tree"])
    cells = suite_cells(manifest)
    if role == "rollout":
        from rl_engine.integrations.engines.rollout.vllm.qwen3_next_native_rollout import (
            run_batch_invariance_cases,
        )

        requests = rollout_requests(manifest, cells)
        run = run_batch_invariance_cases
    else:
        requests = replay_requests(manifest, cells, output.parent / "rollout")
        run = module.run_recompute_cases
    pending = []
    for request in requests:
        path = output / f"{request['case_id']}.pt"
        atomic_json(path.with_suffix(".request.json"), request)
        pending.append((request, path))
    run(pending)
    records = {}
    for request, path in pending:
        artifact = load_tensor_artifact(path)
        validate_model_artifact(artifact, request)
        if artifact.get("provider_binary") != actual_binary:
            raise ValueError("A batch-invariance case used a different CUDA provider")
        records[request["case_id"]] = {"path": str(path), "sha256": file_digest(path)}
    atomic_json(
        output / "worker-result.json",
        {"status": "passed", "role": role, "pid": os.getpid(), "cases": records},
    )


def judge(cells, output):
    rollout, replay = output / "rollout", output / "replay"
    load = {}

    def artifact(directory, case_id):
        key = (directory, case_id)
        if key not in load:
            load[key] = load_tensor_artifact(directory / f"{case_id}.pt")
        return load[key]

    results = {}
    for cell in cells:
        if cell.family == "chunk":
            lhs, rhs = artifact(rollout, cell.cell_id), artifact(replay, f"replay-{cell.cell_id}")
            paired = compare_artifacts(lhs, rhs)
            comparisons, against = paired["comparisons"], "replay"
        elif cell.family == "packing":
            source = packing_source(cell).cell_id
            lhs, rhs = artifact(replay, f"replay-{source}"), artifact(replay, cell.cell_id)
            comparisons, against = compare_batch_prefill(lhs, rhs), "replay alone"
        else:
            lhs = artifact(rollout, cell.reference().cell_id)
            rhs = artifact(rollout, cell.cell_id)
            comparisons, against = compare_batch_prefill(lhs, rhs), "reference alone"
        node = first_divergent_node(lhs, rhs)
        passed = all(comparisons.values())
        if node is None and not passed:
            node = {"layer": None, "node": next(k for k, v in comparisons.items() if not v)}
        results[cell.cell_id] = {
            "family": cell.family,
            "compared_with": against,
            "passed": passed,
            "comparisons": comparisons,
            "first_divergent_node": None if passed else node,
            "first_drift_family": (None if passed else family_of(node["layer"], node["node"])),
        }
    return results


def execute(manifest_path, output):
    manifest = json.loads(Path(manifest_path).read_text())
    output.mkdir(parents=True, exist_ok=True)
    previous = Path(manifest["previous"])
    require_predecessor("bi", previous, Path(manifest["checkpoint"]), manifest)
    check_ready()
    verify_environment(manifest["environment"], require_venv=True)
    memory, requested = memory_limits(), parse_memory_request(manifest["memory"])
    if any(
        value is None or value < requested
        for value in (memory["effective_bytes"], memory["available_bytes"])
    ):
        raise RuntimeError("insufficient_memory: batch-invariance stage budget unavailable")
    run_process(
        [manifest["python"], str(ROOT / "setup.py"), "build_ext", "--inplace"],
        log_path=output / "build.log",
        timeout=1800,
    )
    provider_binary = archive_provider_binary(ROOT, output, source_tree=manifest["source_tree"])
    validate_binary_record(provider_binary, source_tree=manifest["source_tree"])
    probe = output / "gpu-probe.json"
    run_process(
        [
            manifest["python"],
            str(ROOT / "tools/validation/models/qwen3_next_l2.py"),
            "--gpu-probe",
            "--output",
            str(probe),
        ],
        log_path=output / "gpu-probe.log",
        timeout=180,
    )
    uuids = json.loads(probe.read_text())["uuids"]
    wait_gpu_release(uuids)
    atomic_json(
        output / "environment.json",
        {"environment": manifest["environment"], "memory": memory, "gpu_uuids": uuids},
    )
    for role in ("rollout", "replay"):
        run_process(
            [
                manifest["python"],
                str(Path(__file__)),
                "--worker",
                role,
                "--manifest",
                str(manifest_path),
                "--output",
                str(output / role),
                "--provider-binary",
                json.dumps(provider_binary),
            ],
            log_path=output / f"{role}.log",
            timeout=8 * 3600,
        )
        wait_gpu_release(uuids)
    cells = suite_cells(manifest)
    results = judge(cells, output)
    failed = [cell_id for cell_id, item in results.items() if not item["passed"]]
    matrix = {
        "schema_version": 1,
        "scope": "qwen3_next_batch_invariance",
        "status": "passed" if not failed else "failed",
        "suite": manifest.get("bi_suite", "full"),
        "identity": model_identity(manifest),
        "cells_sha256": bi.manifest_payload()["cells_sha256"],
        "cell_ids": [cell.cell_id for cell in cells],
        "cells": results,
        "failed": failed,
        "first_failure": failed[0] if failed else None,
        "provider_binary": provider_binary,
    }
    atomic_json(output / "bi-matrix.json", matrix)
    prior = json.loads((previous / "result.json").read_text())
    shutil.copyfile(previous / prior["evidence"]["operators"]["path"], output / "operators.json")
    evidence = {
        name: evidence_record(output / filename, Path(manifest_path).parent)
        for name, filename in {
            "operators": "operators.json",
            "environment": "environment.json",
            "provider_binary": "provider-binary.json",
            "bi_matrix": "bi-matrix.json",
        }.items()
    }
    if failed:
        raise RuntimeError(f"Batch invariance failed in {len(failed)} cells; first: {failed[0]}")
    result = result_envelope(manifest, os.environ["QWEN3_NEXT_ATTEMPT"], evidence=evidence)
    validate_result(
        manifest,
        result,
        run_dir=Path(manifest_path).parent,
        attempt=os.environ["QWEN3_NEXT_ATTEMPT"],
    )
    atomic_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("rollout", "replay"))
    parser.add_argument("--provider-binary")
    parser.add_argument("--check-ready", action="store_true")
    args = parser.parse_args()
    if args.check_ready:
        _, capabilities = check_ready()
        print(json.dumps(capabilities))
        return
    if args.worker:
        worker(args.worker, args.manifest, args.output, json.loads(args.provider_binary))
        return
    try:
        execute(args.manifest, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / "result.json",
            {
                "status": "failed",
                "attempt": os.environ.get("QWEN3_NEXT_ATTEMPT"),
                "phase": "batch_invariance",
                **failure_fields(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
