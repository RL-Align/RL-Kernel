# SPDX-License-Identifier: Apache-2.0
"""Resolve physical GPU IDs through PCI; DRM minor numbers are not GPU ordinals."""

import json
import subprocess
from pathlib import Path


def resolve_allocation(actors, rollouts, *, drm=Path("/sys/class/drm"), buses=None):
    selected = actors + rollouts
    if len(set(selected)) != len(selected) or any(type(i) is not int or i < 0 for i in selected):
        raise ValueError("physical GPU allocation must contain disjoint nonnegative integers")
    if buses is None:
        data = json.loads(
            subprocess.check_output(
                ["rocm-smi", "-d", *map(str, selected), "--showbus", "--json"], text=True
            )
        )
        buses = {i: data[f"card{i}"]["PCI Bus"].lower() for i in selected}
    kfd = {}
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    for node in nodes.iterdir() if nodes.exists() else ():
        properties = dict(
            line.split(" ", 1) for line in (node / "properties").read_text().splitlines()
        )
        minor = int(properties.get("drm_render_minor", "0"))
        if minor:
            kfd[f"renderD{minor}"] = dict(
                kfd_gpu_id=int((node / "gpu_id").read_text()),
                gpu_uuid="GPU-" + f"{int(properties['unique_id']):016x}",
            )
    records = []
    for group, ids in [("actor", actors), ("rollout", rollouts)]:
        for logical, physical in enumerate(ids):
            bus = buses[physical].lower()
            matches = [
                p.name for p in drm.glob("renderD*") if (p / "device").resolve().name.lower() == bus
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"physical GPU{physical} PCI {bus} has no unique primary render node"
                )
            records.append(
                dict(
                    role=group,
                    physical_gpu=physical,
                    logical_gpu=logical,
                    pci_bus_id=bus,
                    render_node="/dev/dri/" + matches[0],
                    **kfd.get(matches[0], {}),
                )
            )
    order = {
        record["physical_gpu"]: i
        for i, record in enumerate(sorted(records, key=lambda x: x["physical_gpu"]))
    }
    for record in records:
        record["container_gpu"] = order[record["physical_gpu"]]
    return records


def check_selected_identity(identity, physical, records):
    expected = [r for r in records if r["physical_gpu"] == int(physical)]
    if len(expected) != 1 or identity["pci_bus_id"].lower() != expected[0]["pci_bus_id"].lower():
        raise RuntimeError("actual HIP PCI identity differs from the selected physical GPU")
    return expected[0]


if __name__ == "__main__":
    import sys

    print(
        json.dumps(
            resolve_allocation(*[[int(i) for i in arg.split(",")] for arg in sys.argv[1:]]),
            indent=2,
        )
    )
