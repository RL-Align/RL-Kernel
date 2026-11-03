# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prepare a synthetic M10 handoff and export the real T05 training result.

CPU: python tests/megatron_training_pairing.py prepare /path/to/handoff
GPU: torchrun --standalone --nproc_per_node=1 tests/megatron_training_pairing.py \
         score /path/to/handoff --out /path/to/training

This example uses T01/T02 test doubles for inputs and sampling metadata; its
output uses the existing draft M10 exporter. It is not a T09 acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p7_doubles import t02_sampling  # noqa: E402
from p7_doubles.fixtures import fixture_digest, make_fixture  # noqa: E402

from rl_engine.alignment.cross_config.adapters.megatron_training import (  # noqa: E402
    PreUpdateWeightLock,
    ShardDigest,
    tensor_bytes_sha256,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_export import (  # noqa: E402
    export_m10,
    load_m10,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_gate import RowLayout  # noqa: E402
from rl_engine.alignment.cross_config.adapters.megatron_training_provider import (  # noqa: E402
    M10ProviderInput,
    RecordedTrainingProvider,
)

CASE_ID = "t05-m10-pairing-v1"
TENSOR_FIELDS = {
    "final_decoder_rows": "hidden",
    "lm_head_weight_local": "lm_head_weight",
    "target_token_ids": "target_ids",
    "active_token_mask": "active_mask",
}
STRICT_ENV = {
    "VIME_RL_KERNEL_STRICT": "1",
    "RL_KERNEL_LOGP_CASE": "R/R",
    "RL_KERNEL_STRICT_CANONICAL_TP": "4",
    "RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE": "512",
}


def _fixture():
    return replace(
        make_fixture(CASE_ID, sample_lengths=(64, 32), width=128, real_vocab=500, padded_vocab=512),
        canonical_tp=4,
    )


def _weight_lock(fixture):
    shard = ShardDigest.from_tensor(
        fixture.lm_head_weight, tp_rank=0, vocab_start=0, vocab_end=fixture.padded_vocab
    )
    return PreUpdateWeightLock(CASE_ID, tensor_bytes_sha256(fixture.lm_head_weight), (shard,))


def _manifest(fixture):
    lock = _weight_lock(fixture)
    return {
        "purpose": "T05/T06 M10 pairing example; draft, no certification authority",
        "case_id": CASE_ID,
        "fixture_schema_version": fixture.schema_version,
        "fixture_checksum": fixture_digest(fixture),
        "sample_ids": ["s0", "s1"],
        "sample_lengths": list(fixture.sample_lengths),
        "row_keys": [[f"s{s}", p] for s, p in fixture.row_identity()],
        "target_semantics": "already selected for each row; do not shift target_token_ids again",
        "real_vocab_size": fixture.real_vocab,
        "padded_vocab_size": fixture.padded_vocab,
        "physical_tp": 1,
        "physical_cp": 1,
        "tp_group_present": True,
        "environment": STRICT_ENV,
        "expected_training_entrypoint": "sm90_deterministic_linear_logp_tp",
        "weight_lock": lock.to_dict(),
        "weight_lock_fingerprint": lock.fingerprint,
        "sampling_record": t02_sampling.record(t02_sampling.IDENTITY_REQUEST),
        "sampling_record_source": "tests/p7_doubles/t02_sampling.py (test double)",
        "tensors": {
            key: {
                "shape": list(getattr(fixture, attr).shape),
                "dtype": str(getattr(fixture, attr).dtype),
                "sha256": tensor_bytes_sha256(getattr(fixture, attr)),
            }
            for key, attr in TENSOR_FIELDS.items()
        },
    }


def prepare_inputs(directory: Path):
    """Write CPU-only tensors and identities; refuse to overwrite any handoff."""
    fixture = _fixture()
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / "inputs.pt"
    torch.save({key: getattr(fixture, attr) for key, attr in TENSOR_FIELDS.items()}, inputs)
    manifest = _manifest(fixture)
    manifest["inputs_file_sha256"] = hashlib.sha256(inputs.read_bytes()).hexdigest()
    (directory / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_inputs(directory: Path):
    """Validate the shared inputs before allocating GPU memory or scoring."""
    manifest = json.loads((directory / "fixture.json").read_text())
    inputs = directory / "inputs.pt"
    file_digest = manifest.pop("inputs_file_sha256", None)
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != file_digest:
        raise ValueError("inputs.pt checksum mismatch")
    expected = _fixture()
    if manifest != _manifest(expected):
        raise ValueError("fixture identity/configuration differs from the pinned example")
    tensors = torch.load(inputs, map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or set(tensors) != set(TENSOR_FIELDS):
        raise ValueError("inputs.pt tensor fields differ from the pinned example")
    for key, spec in manifest["tensors"].items():
        value = tensors[key]
        if (
            not isinstance(value, torch.Tensor)
            or list(value.shape) != spec["shape"]
            or str(value.dtype) != spec["dtype"]
            or tensor_bytes_sha256(value) != spec["sha256"]
        ):
            raise ValueError(f"{key} shape/dtype/bytes differ from the pinned example")
    fixture = replace(expected, **{attr: tensors[key] for key, attr in TENSOR_FIELDS.items()})
    if fixture_digest(fixture) != manifest["fixture_checksum"]:
        raise ValueError("fixture checksum mismatch")
    return fixture, manifest


def score(directory: Path, output: Path):
    fixture, manifest = load_inputs(directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if int(os.environ.get("WORLD_SIZE", "0")) != 1:
        raise ValueError("launch with torchrun --standalone --nproc_per_node=1")
    for key, value in STRICT_ENV.items():
        if os.environ.get(key, value) != value:
            raise ValueError(f"{key} must be {value!r} for this pairing fixture")
        os.environ[key] = value
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("this example requires a CUDA SM90 device and the strict extension")

    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        lock = _weight_lock(fixture)
        with torch.no_grad():
            evidence = RecordedTrainingProvider().score(
                M10ProviderInput(
                    final_decoder_rows=fixture.hidden.cuda(),
                    lm_head_weight_local=fixture.lm_head_weight.cuda(),
                    target_token_ids=fixture.target_ids.cuda(),
                    active_token_mask=fixture.active_mask,
                    vocab_start=0,
                    real_vocab_size=fixture.real_vocab,
                    padded_vocab_size=fixture.padded_vocab,
                    token_layout_world_size=1,
                    token_layout_rank=0,
                    token_layout="single",
                    row_layout=RowLayout(("s0", "s1"), fixture.sample_lengths),
                    checkpoint_id=lock.checkpoint_id,
                    weight_id=lock.weight_id,
                    weight_lock=lock,
                    lm_head_lora=False,
                    sampling_record=manifest["sampling_record"],
                    tensor_parallel_group=dist.group.WORLD,
                    metadata={"tp_rank": 0, "tp_world_size": 1},
                )
            )
        if evidence.entrypoint != manifest["expected_training_entrypoint"]:
            raise RuntimeError(f"unexpected entrypoint: {evidence.entrypoint}")
        export_m10(evidence, output)
        loaded = load_m10(output)
        if loaded.active_raw_bytes != evidence.active_raw_bytes:
            raise RuntimeError("export round trip changed the active FP32 bytes")
        receipt = {
            "case_id": CASE_ID,
            "fixture_checksum": manifest["fixture_checksum"],
            "weight_lock_fingerprint": lock.fingerprint,
            "active_count": len(loaded.active_keys),
            "active_bytes_sha256": evidence.active_bytes_sha256,
            "evidence_fingerprint": evidence.fingerprint,
            "missing_evidence": [item.to_dict() for item in evidence.missing_evidence],
            "certified": False,
            "t06_t09_pairing": "not performed",
        }
        (output / "pairing_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="create synthetic input files on CPU")
    prepare.add_argument("directory", type=Path)
    scoring = commands.add_parser("score", help="export a real SM90 training result")
    scoring.add_argument("directory", type=Path)
    scoring.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_inputs(args.directory), indent=2))
    else:
        score(args.directory, args.out)


if __name__ == "__main__":
    main()
