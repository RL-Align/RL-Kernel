# SPDX-License-Identifier: Apache-2.0
"""ROCm counterpart of the Inkling 2 Mini sequence contracts (INK-23, INK-34, INK-71).

Two jobs:

1. Device parity. Run the sequence-contract functions (chunked prefill ranges,
   cached decode positions, packing / unpacking, causal and local masks, the
   recompute path and schedule, unrelated-row invariance) on the HIP device and
   compare against the CPU result bit for bit. Everything compared is integer,
   boolean or exactly representable, so parity is bitwise by construction.

2. Long-context plan feasibility on the real device. The 4K / 32K / 128K / 1M
   plan entries are reported ``ready`` or ``gated`` with a reason, from the
   memory read back from the device and the estimate formula below. An estimate
   is NOT a measurement: ``ready`` means the estimated footprint fits the
   device budget and a probe allocation of that size succeeded, not that a
   full model run at that length was executed.

Estimate formula (batch 1, bf16 activations / KV, S = context length). The model
facts come from RFC #736 (Inkling 2 Mini): 24 stored decoder layers executed as 48
logical visits per token (prelude L0-5 once, core L6-17 x3, coda L18-23 once);
every visit has its own KV and short-conv state; attention is 5 local : 1 global
per 6 stored layers, i.e. 40 local-attention visits and 8 full-attention visits;
the local window is 1024 causal tokens, so a local visit caches at most
min(S, 1024) tokens; KV heads 8, head dim 128, BF16; hidden 2304; short-conv
kernel 4, i.e. 3 carried frames::

    kv_full      = FULL_VISITS  * S * 2 * kv_heads * head_dim * 2        bytes
    kv_local     = LOCAL_VISITS * min(S, 1024) * 2 * kv_heads * head_dim * 2
    conv_state   = LOGICAL_VISITS * 3 * (q_dim + 2 * kv_dim + moe_intermediate) * 2
                   (Q/K/V/MLP streams; the MLP stream width is taken as the
                   expert intermediate size, an assumption; the term is tiny)
    boundary_act = LOGICAL_VISITS * S * hidden * 2                       bytes
                   (activation checkpointing keeps one input per logical visit)
    layer_work   = S * (4 * hidden + (q_heads + 2 * kv_heads) * head_dim
                        + 3 * top_k * moe_intermediate) * 2              bytes
                   (peak transient of one visit being (re)computed; attention is
                   assumed flash-style, so no S x S score matrix)
    mask_block   = chunk * S                                             bytes
                   (bool mask for one prefill chunk; ``dense_mask`` = S * S is
                   what ``causal_position_mask`` materializes for the whole
                   sequence and is reported separately)
    required     = kv_full + kv_local + conv_state + boundary_act + layer_work
                   + mask_block
    budget       = free_device_memory * safety_fraction   (default 0.90)

A plan is ``gated`` when required > budget (the reason states the numbers, the
dominant term and the minimum number of devices under sequence-sharded context
parallelism), when the probe allocation fails, or when no ROCm device is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.utils.checkpoint  # noqa: F401  (explicit import; recompute_forward relies on it)

from .sequence_contracts import (
    LongContextPlan,
    PackedRows,
    build_recompute_schedule,
    causal_position_mask,
    chunked_prefill_ranges,
    decode_position_ids,
    pack_rows,
    recompute_forward,
    unpack_rows,
    validate_unrelated_row_invariance,
)

GIB = 1024**3

# RFC #736 (Inkling 2 Mini) model facts used by the estimate.
STORED_LAYERS = 24
LOGICAL_VISITS = 48  # prelude L0-5 x1 + core L6-17 x3 + coda L18-23 x1 = 6 + 36 + 6
FULL_ATTENTION_VISITS = 8
LOCAL_ATTENTION_VISITS = 40  # 5 local : 1 global per 6 stored layers
LOCAL_WINDOW = 1024
CONV_KERNEL = 4
CONV_STATE_FRAMES = CONV_KERNEL - 1
assert LOGICAL_VISITS == FULL_ATTENTION_VISITS + LOCAL_ATTENTION_VISITS


# --------------------------------------------------------------------------
# Runtime readback
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RocmDeviceInfo:
    hip_version: str | None
    device_count: int
    device_name: str
    gcn_arch_name: str
    total_memory_bytes: int
    free_memory_bytes: int

    @property
    def is_rocm(self) -> bool:
        return self.hip_version is not None and self.device_count > 0


def read_rocm_device(device_index: int = 0) -> RocmDeviceInfo:
    """Read HIP version, arch and device memory (``hipMemGetInfo``) from the runtime."""
    hip = getattr(torch.version, "hip", None)
    if hip is None or not torch.cuda.is_available():
        return RocmDeviceInfo(hip, 0, "", "", 0, 0)
    props = torch.cuda.get_device_properties(device_index)
    free, total = torch.cuda.mem_get_info(device_index)
    return RocmDeviceInfo(
        hip,
        torch.cuda.device_count(),
        props.name,
        getattr(props, "gcnArchName", ""),
        int(total),
        int(free),
    )


# --------------------------------------------------------------------------
# Device parity of the sequence contracts
# --------------------------------------------------------------------------
def _ranges_as_tensor(sequence_length: int, chunk_size: int) -> torch.Tensor:
    ranges = chunked_prefill_ranges(sequence_length, chunk_size)
    return torch.tensor([[r.start, r.end] for r in ranges], dtype=torch.int64)


def _recompute_exact(device: torch.device) -> dict[str, torch.Tensor]:
    """Checkpointed forward+backward on exactly representable values (device-independent)."""
    g = torch.Generator().manual_seed(2371)
    x = torch.randint(-4, 5, (6, 8), generator=g).float().to(device).requires_grad_(True)
    w = torch.randint(-4, 5, (8, 5), generator=g).float().to(device).requires_grad_(True)

    def block(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.relu(value @ matrix) * 2.0

    out = recompute_forward(block, x, w)
    grad = torch.randint(-3, 4, out.shape, generator=g).float().to(device)
    gx, gw = torch.autograd.grad(out, (x, w), grad)
    return {"out": out.detach().cpu(), "gx": gx.cpu(), "gw": gw.cpu()}


def sequence_contract_results(device: torch.device | str) -> dict[str, torch.Tensor]:
    """Run every sequence contract on ``device``; return CPU copies for comparison."""
    device = torch.device(device)
    results: dict[str, torch.Tensor] = {}

    # INK-23: chunked prefill ranges and cached decode positions
    results["ranges_131077_4096"] = _ranges_as_tensor(131077, 4096)
    results["ranges_1_4096"] = _ranges_as_tensor(1, 4096)
    cache = torch.tensor([0, 7, 12, 131071, 1048575], device=device)
    results["decode_pos_step1"] = decode_position_ids(cache).cpu()
    results["decode_pos_step4"] = decode_position_ids(cache.to(torch.int32), step=4).cpu()

    # INK-34: packing, unpacking, masks
    g = torch.Generator().manual_seed(34)
    lengths = (1, 5, 64, 65, 130)
    rows = tuple(
        torch.randint(-1000, 1000, (n, 3, 4), generator=g).to(torch.bfloat16).to(device)
        for n in lengths
    )
    packed: PackedRows = pack_rows(rows)
    results["packed_values"] = packed.values.cpu()
    results["packed_offsets"] = packed.offsets.cpu()
    results["packed_row_ids"] = packed.row_ids.cpu()
    results["packed_position_ids"] = packed.position_ids.cpu()
    for i, row in enumerate(unpack_rows(packed)):
        results[f"unpacked_{i}"] = row.cpu()
    query = torch.tensor([[0, 1, 4, 9], [3, 3, 8, 100]], device=device)
    key = torch.tensor([[0, 1, 2, 4, 9], [1, 2, 3, 50, 100]], device=device)
    results["mask_causal"] = causal_position_mask(query, key).cpu()
    for window in (1, 2, 5, 64):
        results[f"mask_window_{window}"] = causal_position_mask(query, key, window=window).cpu()
    long_pos = torch.arange(300, device=device)[None]
    results["mask_local_300"] = causal_position_mask(long_pos, long_pos, window=128).cpu()

    # INK-71: recompute path, schedule, unrelated-row invariance
    results.update({f"recompute_{k}": v for k, v in _recompute_exact(device).items()})
    schedule = build_recompute_schedule(5, repeated_passes=3)
    results["schedule"] = torch.tensor(
        [[e.physical_layer, e.pass_index, e.logical_visit] for e in schedule]
    )
    reference = torch.arange(24, device=device, dtype=torch.float32).reshape(6, 4)
    changed = reference.clone()
    changed[2] += 100.0
    validate_unrelated_row_invariance(reference, changed, row=2)
    results["invariance_ok"] = torch.tensor(1)
    return results


def device_cpu_parity(device: torch.device | str = "cuda") -> tuple[str, ...]:
    """Names of contract results whose HIP-device bytes differ from the CPU run."""
    cpu = sequence_contract_results("cpu")
    dev = sequence_contract_results(device)
    mismatches: list[str] = []
    if cpu.keys() != dev.keys():
        mismatches.append("result keys differ")
    for name in sorted(cpu.keys() & dev.keys()):
        a, b = cpu[name], dev[name]
        if a.dtype != b.dtype or a.shape != b.shape or not _bytes_equal(a, b):
            mismatches.append(name)
    return tuple(mismatches)


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a8 = a.contiguous().reshape(-1).view(torch.uint8) if a.numel() else a
    b8 = b.contiguous().reshape(-1).view(torch.uint8) if b.numel() else b
    return bool(torch.equal(a8, b8))


def recompute_matches_plain_on_device(device: torch.device | str = "cuda") -> bool:
    """Checkpointed recompute (with dropout RNG preserved) equals the plain run bitwise."""
    device = torch.device(device)

    def block(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.dropout(torch.tanh(value @ matrix), p=0.3, training=True)

    base = torch.randn(33, 64, device=device)
    weight = torch.randn(64, 17, device=device)
    seed = 20260101
    grads = []
    outs = []
    for use_recompute in (True, False):
        x = base.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        torch.manual_seed(seed)
        out = recompute_forward(block, x, w) if use_recompute else block(x, w)
        torch.manual_seed(seed + 1)  # perturb the global RNG before backward
        grad_out = torch.ones_like(out)
        grads.append(torch.autograd.grad(out, (x, w), grad_out))
        outs.append(out.detach())
    same = torch.equal(outs[0], outs[1])
    return same and all(torch.equal(a, b) for a, b in zip(grads[0], grads[1], strict=True))


# --------------------------------------------------------------------------
# Long-context plan feasibility
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InklingDims:
    hidden: int = 2304
    q_heads: int = 24
    kv_heads: int = 8
    head_dim: int = 128
    moe_top_k: int = 8
    moe_intermediate: int = 1152
    activation_bytes: int = 2  # bf16


@dataclass(frozen=True)
class MemoryEstimate:
    sequence_length: int
    kv_full: int
    kv_local: int
    conv_state: int
    boundary_activations: int
    layer_work: int
    mask_block: int
    dense_mask: int

    @property
    def kv_cache(self) -> int:
        return self.kv_full + self.kv_local

    @property
    def required(self) -> int:
        return sum(self.components().values())

    def components(self) -> dict[str, int]:
        return {
            "kv_full": self.kv_full,
            "kv_local": self.kv_local,
            "conv_state": self.conv_state,
            "boundary_activations": self.boundary_activations,
            "layer_work": self.layer_work,
            "mask_block": self.mask_block,
        }

    @property
    def dominant(self) -> str:
        parts = self.components()
        return max(parts, key=parts.__getitem__)


def estimate_long_context_memory(
    sequence_length: int, dims: InklingDims = InklingDims(), *, chunk_size: int = 4096
) -> MemoryEstimate:
    """Apply the module-docstring formula (bytes, batch 1)."""
    s, b = sequence_length, dims.activation_bytes
    kv_token = 2 * dims.kv_heads * dims.head_dim * b
    kv_full = FULL_ATTENTION_VISITS * s * kv_token
    kv_local = LOCAL_ATTENTION_VISITS * min(s, LOCAL_WINDOW) * kv_token
    streams = dims.q_heads * dims.head_dim + 2 * dims.kv_heads * dims.head_dim
    conv = LOGICAL_VISITS * CONV_STATE_FRAMES * (streams + dims.moe_intermediate) * b
    boundary = LOGICAL_VISITS * s * dims.hidden * b
    work = (
        s
        * (
            4 * dims.hidden
            + (dims.q_heads + 2 * dims.kv_heads) * dims.head_dim
            + 3 * dims.moe_top_k * dims.moe_intermediate
        )
        * b
    )
    return MemoryEstimate(s, kv_full, kv_local, conv, boundary, work, min(chunk_size, s) * s, s * s)


@dataclass(frozen=True)
class LongContextReadiness:
    sequence_length: int
    status: str  # "ready" | "gated"
    reason: str
    estimate: MemoryEstimate
    budget_bytes: int
    min_devices_context_parallel: int
    logical_visits: int
    probe_allocated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def assess_long_context_plan(
    plan: LongContextPlan | None = None,
    dims: InklingDims = InklingDims(),
    *,
    device_index: int = 0,
    safety_fraction: float = 0.90,
    chunk_size: int = 4096,
    probe: bool = True,
    info: RocmDeviceInfo | None = None,
) -> tuple[LongContextReadiness, ...]:
    """Report every qualified context length as ready or gated on this device."""
    plan = plan or LongContextPlan()
    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError("safety_fraction must be in (0, 1]")
    info = info if info is not None else read_rocm_device(device_index)
    budget = int(info.free_memory_bytes * safety_fraction)
    reports = []
    for length in plan.context_lengths:
        plan.validate(length)
        est = estimate_long_context_memory(length, dims, chunk_size=chunk_size)
        devices = max(1, -(-est.required // budget)) if budget > 0 else 0
        notes = (
            f"dense causal_position_mask would need {est.dense_mask / GIB:.3f} GiB; "
            "long contexts must build masks per chunk",
            f"estimate uses RFC #736: {LOGICAL_VISITS} logical visits, "
            f"{FULL_ATTENTION_VISITS} full + {LOCAL_ATTENTION_VISITS} local "
            f"(window {LOCAL_WINDOW}) attention visits",
            "estimate, not a measurement: no model forward was run at this length",
        )
        if not info.is_rocm:
            status, reason, allocated = "gated", "no ROCm device detected", False
        elif est.required > budget:
            status = "gated"
            reason = (
                f"estimated {est.required / GIB:.1f} GiB > budget {budget / GIB:.1f} GiB "
                f"(dominant term {est.dominant}: {est.components()[est.dominant] / GIB:.1f} GiB; "
                f"{safety_fraction:.0%} of {info.free_memory_bytes / GIB:.1f} GiB free on "
                f"{info.gcn_arch_name}); needs >= {devices} devices with sequence-sharded "
                "context parallelism"
            )
            allocated = False
        else:
            allocated = _probe_allocation(est.required, device_index) if probe else False
            if probe and not allocated:
                status, reason = "gated", "probe allocation of the estimated footprint failed"
            else:
                status = "ready"
                reason = (
                    f"estimated {est.required / GIB:.1f} GiB fits budget {budget / GIB:.1f} GiB"
                )
        reports.append(
            LongContextReadiness(
                length, status, reason, est, budget, devices, LOGICAL_VISITS, allocated, notes
            )
        )
    return tuple(reports)


def _probe_allocation(nbytes: int, device_index: int) -> bool:
    try:
        buffer = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{device_index}")
        buffer[0] = 1
        buffer[-1] = 1
        torch.cuda.synchronize(device_index)
        del buffer
        return True
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return False
    finally:
        torch.cuda.empty_cache()
