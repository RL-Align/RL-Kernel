# SPDX-License-Identifier: Apache-2.0
"""Freeze actual vLLM TP2 sampler output with exact per-rank checkpoint checks."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from examples.post_training.tp_ppo.fixed_contract import identity, sha, verify_checkpoint
from tests.distributed.tp.vllm_ppo_client import VllmPPOClient


def main():
    """Install the verified checkpoint as version 0 in a vLLM TP2 sampler and freeze one rollout.

    Writes ``version0.pt``, ``rollout.pt`` and a ``manifest.json`` with provenance, hashes and
    sampler timings to ``--output``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vllm-python", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = identity()
    report["checkpoint_manifest_sha256"] = verify_checkpoint(a.checkpoint)
    thresholds = json.loads(a.calibration.read_text())
    weights = {}
    index = json.loads((a.checkpoint / "model.safetensors.index.json").read_text())
    for name in sorted(set(index["weight_map"].values())):
        weights.update(load_file(str(a.checkpoint / name)))
    if weights.keys() != index["weight_map"].keys():
        raise ValueError("checkpoint tensor inventory mismatch")
    path = a.output / "version0.pt"
    torch.save(dict(actor_version=0, weights=weights), path)
    del weights
    client = VllmPPOClient(
        a.vllm_python,
        a.output / "vllm",
        torch.device("cuda", 0),
        checkpoint=a.checkpoint,
        tp=2,
        cache_logprob_atol=thresholds["behavior_logprob"]["atol"],
    )
    try:
        response = client._request(dict(op="install", filename=str(path.resolve()), version=0))
        if response["installed"] != 0:
            raise ValueError("version zero installation failed")
        client.version = 0
        client.installed.append(0)
        rollout = client.sample(0, 0)
        rollout.validate(actor_version=0, critic_version=0)
        torch.save({k: v.cpu() for k, v in rollout.packet().items()}, a.output / "rollout.pt")
        report.update(
            rollout_sha256=sha(a.output / "rollout.pt"),
            weight_packet_sha256=sha(path),
            calibration_sha256=sha(a.calibration),
            installation=response,
            sampler_runtime=client.runtime,
            sampler_events=client.timings,
            status="fixed_rollout_complete",
            sampler="vllm",
            tp=2,
        )
        (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
