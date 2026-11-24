# SPDX-License-Identifier: Apache-2.0
"""INK-42: observed device/group topology and per-visit runtime evidence."""

import torch
import torch.distributed as dist


def read_topology(device: torch.device, groups: dict, *, platform: str) -> dict:
    if platform not in ("cuda", "rocm") or device.type != "cuda":
        raise ValueError("expected CUDA or ROCm device")
    if not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm"):
        raise RuntimeError("requested accelerator unavailable")
    props = torch.cuda.get_device_properties(device)
    arch = props.gcnArchName.split(":")[0] if platform == "rocm" else f"sm{props.major}{props.minor}"
    topology = {}
    for axis, group in groups.items():
        if group is None:
            topology[axis] = {"size": 1, "rank": 0, "transport": "none"}
        else:
            if not dist.is_initialized():
                raise RuntimeError("distributed process group not initialized")
            rank = dist.get_rank(group)
            if rank < 0:
                raise RuntimeError("not a group member")
            topology[axis] = {"size": dist.get_world_size(group), "rank": rank,
                              "transport": str(dist.get_backend(group))}
    return {"platform": platform, "architecture": arch, "device": props.name,
            "torch": str(torch.__version__), "groups": topology}


class VisitEvidence:
    """Record actual adapter observations; no inferred backend or qualification."""

    def __init__(self, topology: dict):
        self.topology = dict(topology)
        self.records = {}
        self.cache_slots = set()

    def record(self, visit: int, *, backend: str, kernel: str, cache_slot: int,
               conv_slot: int, splits: int, fallback: str | None) -> None:
        if type(visit) is not int or not 0 <= visit < 48 or visit in self.records:
            raise ValueError("invalid or duplicate logical visit")
        if not backend or not kernel or type(splits) is not int or splits <= 0:
            raise ValueError("actual backend, kernel and split count required")
        if any(type(x) is not int or x < 0 for x in (cache_slot, conv_slot)):
            raise ValueError("invalid cache slot")
        if ("kv", cache_slot) in self.cache_slots or ("conv", conv_slot) in self.cache_slots:
            raise ValueError("cache slot reused across logical visits")
        self.cache_slots.update((("kv", cache_slot), ("conv", conv_slot)))
        self.records[visit] = dict(visit=visit, backend=backend, kernel=kernel,
                                   cache_slot=cache_slot, conv_slot=conv_slot,
                                   splits=splits, fallback=fallback)

    def report(self) -> dict:
        records = [self.records[j].copy() for j in sorted(self.records)]
        return {"topology": self.topology, "records": records,
                "complete": len(records) == 48,
                "fallback_free": len(records) == 48 and all(r["fallback"] is None for r in records),
                "qualified": False}
