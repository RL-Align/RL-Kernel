# SPDX-License-Identifier: Apache-2.0
"""Task GPU policy and measured physical/logical ROCm mapping, before collectives.

This module imports PyTorch only after checking the visibility environment.
Host preparation queries metadata for the selected physical GPUs only.
"""

import argparse
import ctypes
import json
import os
import subprocess
from pathlib import Path

ALLOWED_PHYSICAL_GPUS = (4, 5, 6, 7)
SECONDARY_FILTERS = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
)


def physical_ids(csv):
    parts = csv.split(",")
    if not parts or any(not p.isdecimal() for p in parts):
        raise ValueError("expected physical GPU CSV within 4,5,6,7")
    ids = [int(p) for p in parts]
    if ids != sorted(set(ids)) or any(i not in ALLOWED_PHYSICAL_GPUS for i in ids):
        raise ValueError(
            "only ascending unique physical GPUs 4,5,6,7 are allowed; 8-GPU tests canceled"
        )
    return ids


def host_mapping(csv):
    ids = physical_ids(csv)
    raw = json.loads(
        subprocess.check_output(
            ["rocm-smi", "-d", *[str(i) for i in ids], "--showbus", "--showuniqueid", "--json"],
            text=True,
        )
    )
    return {
        "allowed_physical_gpus": list(ALLOWED_PHYSICAL_GPUS),
        "physical_gpus": ids,
        "logical_gpus": list(range(len(ids))),
        "rocr_visible_devices": csv,
        "mapping": [
            {
                "logical_gpu": logical,
                "physical_gpu": physical,
                "pci_bus_id": raw[f"card{physical}"]["PCI Bus"].lower(),
                "rocm_unique_id": raw[f"card{physical}"]["Unique ID"],
            }
            for logical, physical in enumerate(ids)
        ],
    }


def validate_mapping(expected, observed, environ):
    ids = physical_ids(expected["rocr_visible_devices"])
    if environ.get("ROCR_VISIBLE_DEVICES") != expected["rocr_visible_devices"]:
        raise RuntimeError("ROCR visibility does not match physical GPU assignment")
    inherited = {key: environ[key] for key in SECONDARY_FILTERS if key in environ}
    if inherited:
        raise RuntimeError(f"secondary GPU filters must be unset, not remapped: {inherited}")
    if len(observed) != len(ids) or expected["physical_gpus"] != ids:
        raise RuntimeError("visible GPU count differs from assigned physical GPUs")
    for logical, (want, actual) in enumerate(zip(expected["mapping"], observed, strict=True)):
        if (
            actual["logical_gpu"] != logical
            or want["logical_gpu"] != logical
            or want["physical_gpu"] != ids[logical]
            or actual["pci_bus_id"].lower() != want["pci_bus_id"].lower()
        ):
            raise RuntimeError(f"physical/logical PCI mapping mismatch at logical GPU{logical}")


def verify_visible(expected, *, allocate=False):
    # Reject secondary filters before importing torch or initializing HIP.
    validate_mapping(
        expected,
        [
            {"logical_gpu": i, "pci_bus_id": d["pci_bus_id"]}
            for i, d in enumerate(expected["mapping"])
        ],
        os.environ,
    )
    import torch

    if not torch.version.hip:
        raise RuntimeError("real ROCm PyTorch is required")
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    actual = []
    for logical in range(torch.cuda.device_count()):
        bus = ctypes.create_string_buffer(32)
        status = hip.hipDeviceGetPCIBusId(bus, len(bus), logical)
        if status:
            raise RuntimeError(f"hipDeviceGetPCIBusId failed: {status}")
        props = torch.cuda.get_device_properties(logical)
        actual.append(
            {
                "logical_gpu": logical,
                "pci_bus_id": bus.value.decode().lower(),
                "torch_uuid": str(props.uuid),
                "name": props.name,
            }
        )
    validate_mapping(expected, actual, os.environ)
    if allocate:
        for logical, physical in enumerate(expected["physical_gpus"]):
            # Exact integer roundtrip establishes usable container devices after mapping validation.
            marker = torch.tensor([physical], device=f"cuda:{logical}", dtype=torch.int32)
            if marker.item() != physical:
                raise RuntimeError("GPU marker roundtrip failed")
    return {
        **expected,
        "status": "PASS",
        "observed": actual,
        "secondary_filters": {key: os.environ.get(key) for key in SECONDARY_FILTERS},
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "allocation_roundtrip": allocate,
    }


def rank_assignment():
    path = os.environ.get("DSV41_EXPECTED_GPU_MAPPING")
    if not path:
        raise RuntimeError("DSV41_EXPECTED_GPU_MAPPING required; use the guarded GPU launcher")
    expected = json.loads(Path(path).read_text())
    measured = verify_visible(expected)
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local < len(expected["mapping"]):
        raise RuntimeError("LOCAL_RANK outside verified visible devices")
    return {
        **measured,
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": local,
        "assigned_physical_gpu": expected["mapping"][local]["physical_gpu"],
        "assigned_logical_gpu": local,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("host")
    host.add_argument("--physical", required=True)
    host.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--expected", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "host":
        result = host_mapping(args.physical)
    else:
        result = verify_visible(json.loads(args.expected.read_text()), allocate=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("GPU_MAPPING", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
