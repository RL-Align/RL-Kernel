# SPDX-License-Identifier: Apache-2.0
"""ROCm counterpart of the Inkling 2 Mini topology gates (INK-36/37/38/39).

Everything here is ROCm-only.  It reads the HIP runtime back (device count,
``gcnArchName``, RCCL), qualifies TP2/TP4 (+EP carved out of the TP group) with
``platform="rocm"`` evidence, keeps CP/SP/PP deferred with explicit reasons, and
provides a real multi-GPU RCCL check: all-gather of per-rank FP32 partials
followed by a fixed rank-order sum that must be bitwise identical on all ranks
and across repeats.

Gating policy (fail closed): the SGLang Inkling ROCm recipe is TP4/EP4 on
MI350X/MI355X (gfx950).  MI300X (gfx942) is gated for the recipe; the RCCL
collective check below is still a real measurement on gfx942.
"""

from __future__ import annotations

import hashlib
import os
import socket
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.models.inkling2.topology_contracts import (
    FeasibilityReport,
    InklingTopology,
    TopologyEvidence,
    assess_deferred_parallelism,
    qualify_tp_ep,
)

RECIPE_ARCHES = ("gfx950",)
RECIPE_TP = 4
RECIPE_EP = 4
GATED_ARCH_REASON = (
    "SGLang Inkling2Loop recipe (TP4/EP4) lists MI350X/MI355X (gfx950) only; "
    "MI300X (gfx942) needs a separate port"
)


@dataclass(frozen=True)
class RocmReadback:
    hip_version: str | None
    device_count: int
    arch_names: tuple[str, ...]
    rccl_available: bool
    nccl_version: tuple[int, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hip_version": self.hip_version,
            "device_count": self.device_count,
            "arch_names": list(self.arch_names),
            "rccl_available": self.rccl_available,
            "nccl_version": None if self.nccl_version is None else list(self.nccl_version),
        }

    @property
    def base_arch(self) -> str | None:
        """Architecture without feature suffixes, e.g. ``gfx942:sramecc+`` -> ``gfx942``."""
        families = {a.split(":")[0] for a in self.arch_names}
        return next(iter(families)) if len(families) == 1 else None


def read_rocm_runtime() -> RocmReadback:
    hip = torch.version.hip
    count = torch.cuda.device_count() if hip is not None and torch.cuda.is_available() else 0
    arches = tuple(torch.cuda.get_device_properties(i).gcnArchName for i in range(count))
    try:
        rccl = bool(torch.distributed.is_available() and torch.distributed.is_nccl_available())
    except Exception:
        rccl = False
    version = None
    if rccl:
        try:
            version = tuple(int(v) for v in torch.cuda.nccl.version())
        except Exception:
            version = None
    return RocmReadback(hip, count, arches, rccl, version)


def build_rocm_evidence(topology: InklingTopology, readback: RocmReadback) -> TopologyEvidence:
    return TopologyEvidence(
        framework="sglang",
        platform="rocm",
        topology=topology,
        attention_backend="triton",
        router_backend="inkling2-top8-router",
        cache_isolation=True,
        reduction_order="fixed-rank-order",
        source=f"rocm-readback:{readback.hip_version}:{','.join(readback.arch_names)}",
    )


def rocm_blockers(
    topology: InklingTopology, readback: RocmReadback, *, expert_parallel: int = 1
) -> tuple[str, ...]:
    """Reasons the ROCm runtime cannot qualify this topology (empty -> qualified)."""
    blockers: list[str] = []
    if readback.hip_version is None:
        blockers.append("torch is not a HIP build")
    if readback.device_count < topology.world_size:
        blockers.append(
            f"world_size={topology.world_size} needs {topology.world_size} visible HIP devices, "
            f"found {readback.device_count}"
        )
    if not readback.rccl_available:
        blockers.append("RCCL is unavailable")
    if expert_parallel < 1 or topology.tensor_parallel % expert_parallel:
        blockers.append("expert_parallel must divide tensor_parallel (EP is carved from TP)")
    arch = readback.base_arch
    if arch is None:
        blockers.append("device architectures are missing or heterogeneous")
    elif arch not in RECIPE_ARCHES:
        blockers.append(GATED_ARCH_REASON + f" (found {arch})")
    elif topology.tensor_parallel != RECIPE_TP or expert_parallel != RECIPE_EP:
        blockers.append(f"recipe is TP{RECIPE_TP}/EP{RECIPE_EP}")
    return tuple(blockers)


def qualify_tp_ep_rocm(
    topology: InklingTopology,
    readback: RocmReadback,
    *,
    expert_parallel: int = 1,
) -> FeasibilityReport:
    """INK-36 on ROCm.  ``expert_parallel`` is the EP size carved out of the TP group."""
    base = qualify_tp_ep(topology, evidence=build_rocm_evidence(topology, readback))
    if base.status == "rejected":
        return base
    blockers = list(base.blockers) + list(
        rocm_blockers(topology, readback, expert_parallel=expert_parallel)
    )
    if blockers:
        return FeasibilityReport("INK-36", "gated", tuple(blockers), base.evidence)
    return base


def assess_deferred_parallelism_rocm(item: str, topology: InklingTopology) -> FeasibilityReport:
    """CP/SP/PP stay deferred on ROCm: the CUDA reasons plus the ROCm-specific one."""
    base = assess_deferred_parallelism(item, topology)
    if base.status == "rejected":
        return base
    extra = "no ROCm (RCCL/AITER) CP/SP/PP evidence exists; deferred until MI350X/MI355X runs"
    return FeasibilityReport(item, "gated", base.blockers + (extra,), base.evidence)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _partial(rank: int, size: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1000 + rank)
    # Wide dynamic range so that a different summation order would change bits.
    scale = torch.tensor(10.0).pow(torch.randint(-3, 4, (size,), generator=gen).float())
    return torch.randn(size, generator=gen) * scale


def fixed_rank_order_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    total = partials[0].clone()
    for part in partials[1:]:
        total = total + part
    return total


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _rank_main(rank, world_size, port, repeats, size, queue):
    import torch.distributed as dist

    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
        )
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        digests = []
        for _ in range(repeats):
            local = _partial(rank, size).cuda()
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            digests.append(_digest(fixed_rank_order_sum(gathered)))
        dist.barrier()
        dist.destroy_process_group()
        queue.put((rank, digests, None))
    except Exception as error:  # reported to the parent, never swallowed
        queue.put((rank, None, repr(error)))


def run_rccl_rank_order_check(
    world_size: int, *, repeats: int = 2, size: int = 1 << 16, timeout: float = 50.0
) -> dict[str, Any]:
    """Spawn ``world_size`` RCCL ranks; return per-rank digests plus the CPU reference."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_rank_main, args=(r, world_size, port, repeats, size, queue))
        for r in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results: dict[int, list[str]] = {}
    errors: dict[int, str] = {}
    try:
        for _ in range(world_size):
            rank, digests, error = queue.get(timeout=timeout)
            if error is not None:
                errors[rank] = error
            else:
                results[rank] = digests
    finally:
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
    reference = _digest(fixed_rank_order_sum([_partial(r, size) for r in range(world_size)]))
    return {
        "world_size": world_size,
        "port": port,
        "digests": results,
        "errors": errors,
        "cpu_reference": [reference] * repeats,
    }
