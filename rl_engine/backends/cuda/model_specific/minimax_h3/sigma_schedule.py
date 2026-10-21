# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton CUDA backend for the MiniMax-H3 sigma grid (RFC #420).

NVIDIA only: the grid-transform division is emitted as inline PTX
``div.rn.f32`` (see below), which makes the kernels NVIDIA-specific. HIP is
not supported by this backend; the registry does not offer it on rocm.

The grid is built **on the device**: no CPU construction followed by a copy,
and no torch factory functions whose GPU kernels differ bitwise from the CPU
reference. The bit-level target is the pinned CPU fixture in
``tests/reference/minimax_h3/`` (upstream diffusers ``set_timesteps`` at commit
``f53d552``), reproduced here by:

* ``linspace`` law: ``step = fl32(-1 / (n-1))``; for ``idx < n//2`` the value
  is ``fl32(1.0 + fp64(step) * fp64(idx))``, otherwise
  ``fl32(0.0 - fp64(step) * fp64(n-1-idx))`` (two-sided, single rounding);
* shift arithmetic with the scalar roundings torch applies to python-float
  operands: ``fl32(shift)`` and ``fl32(shift - 1.0)`` computed first, then
  mul / mul+add / div — each fp32 step evaluated through an exact fp64
  intermediate rounded once to fp32 (bit-identical to the correctly rounded,
  non-FTZ fp32 op that torch's CPU kernels perform; fp32 x fp32 products and
  sums are exact in fp64). Plain Triton fp32 mul/add may compile
  flush-to-zero, which destroys the subnormal numerators of subnormal shifts
  and broke parity (independent audit 2026-10-16);
* correctly-rounded, non-FTZ fp32 division emitted as inline PTX
  ``div.rn.f32``. Two measured findings forced this: (1) on Triton 3.0.0 /
  RTX 4090D ``tl.fdiv(..., ieee_rounding=True)`` and the ``/`` operator were
  bit-identical to each other and NOT correctly rounded (64707 mismatched
  elements vs the CPU division across the fingerprint grids — probe
  artifacts/.../diag/div_probe.py); (2) on Triton 3.1.0 the libdevice
  ``div_rn`` binding flushed subnormal DIVIDENDS — ``div_rn(2^-149, +0.0)``
  returned NaN instead of +inf and subnormal/normal returned +0, collapsing
  the ``1e-45`` grid (probe artifacts/.../diag/subnormal_probe.py). The bare
  PTX instruction carries no ``.ftz`` modifier, so IEEE subnormal semantics
  hold; ``enable_fp_fusion=False`` on every floating-point launch, so no
  multiply-add is contracted into an FMA (fused and separate roundings
  differ on real grids, e.g. N=50, shift=12);
* ``unique_consecutive`` semantics: keep an element when it is not IEEE-equal
  to its predecessor (NaN never equals, so NaN rows are kept, matching torch);
* ``timesteps = fl32(1.0 - sigma)`` for the first ``M - 1`` sigmas, with NaN
  sigmas passed through unchanged (torch CPU returns the quiet NaN operand
  itself from ``1.0 - NaN``, sign and payload included).

NaN and near-zero parity (independent GPU audit 2026-10-16):

* A NaN ``fl32(shift)`` propagates its own quieted bits through the whole
  grid on the CPU oracle — sign included (``+NaN`` -> ``0x7fc00000``,
  ``-NaN`` -> ``0xffc00000``; quieting = setting bit 22). ``forward``
  detects NaN/inf scalars on the host and fills the grid directly with the
  oracle's pattern; the device kernel canonicalizes NaNs to ``0x7fffffff``,
  which is why the fill exists.
* An infinite ``fl32(shift)`` (``inf``, or a finite shift above the fp32
  range such as ``1e39`` / ``3.5e38``) makes every element an invalid
  operation (``inf/inf``, or ``inf*0`` at the terminal zero), which x86/SSE
  evaluates to the default quiet NaN ``0xffc00000`` regardless of sign.
* A subnormal ``fl32(shift)`` (e.g. ``1e-45`` rounds up to the minimum
  subnormal ``2^-149``) takes the normal kernel path: element 0 is
  ``2^-149 / +0.0 = +inf`` and the remaining elements are subnormal
  quotients — upstream keeps e.g. 10 of 25 rows for ``1e-45``.
* A shift whose ``fl32`` underflows to ``+0.0`` (e.g. ``1e-46``) makes
  element 0 the invalid ``+0.0 / +0.0``: the kernel canonicalizes NaN
  division results to the CPU's ``0xffc00000`` (finite shifts CAN produce
  exactly this one NaN — the denominator ``1 + fl32(shift-1)*s`` is exactly
  ``+0.0`` at ``s = 1`` when ``fl32(shift)`` is tiny).

Dedup runs as one single-program kernel (sequential block scan), so the
compaction order is deterministic for any ``N``. The output length ``M`` is
read back with one host synchronization of a device counter — the only sync;
sigma values never leave the device.

Availability: constructing the op raises ``ImportError`` when Triton is not
importable, so the registry's first-available-candidate selection skips this
backend at construction time instead of failing later in ``forward``.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    # Correctly-rounded fp32 division (PTX div.rn.f32). The import path moved
    # across Triton versions; both known layouts are accepted.
    try:
        from triton.language.extra import libdevice as _libdevice
    except ImportError:  # pragma: no cover - alternative Triton layout
        from triton.language.extra.cuda import libdevice as _libdevice

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without a Triton wheel
    _TRITON_AVAILABLE = False

from rl_engine.reference.minimax_h3.sigma_schedule import H3SigmaSchedule

_BLOCK = 1024


def _shift_scalars(shift: float) -> tuple[float, float]:
    """Round the two python-float scalars exactly as torch would.

    ``fl32(shift)`` and ``fl32(shift - 1.0)`` with the subtraction in fp64
    first — pinned on the CPU device so an ambient default device (meta,
    CUDA) cannot move the factory or add a synchronization. These are the
    only host-side roundings; the grid itself is built by the kernels.
    """

    shift_f32 = torch.tensor(shift, dtype=torch.float32, device="cpu").item()
    shift_m1_f32 = torch.tensor(shift - 1.0, dtype=torch.float32, device="cpu").item()
    return shift_f32, shift_m1_f32


if _TRITON_AVAILABLE:

    @triton.jit
    def _shifted_grid_kernel(
        out_ptr,
        n,
        shift_f32,
        shift_m1_f32,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n

        # Base grid, pinned CPU linspace law: fp32 step, two-sided fp64
        # accumulate with a single rounding to fp32. The step division uses
        # div_rn like the main transform (tl.fdiv's ieee_rounding flag is not
        # correctly rounded on the validated Triton 3.0.0, see module docs).
        step32 = _libdevice.div_rn(-1.0, (n - 1).to(tl.float32))
        step64 = step32.to(tl.float64)
        half = n // 2
        start_side = 1.0 + step64 * offs.to(tl.float64)
        end_side = 0.0 - step64 * (n - 1 - offs).to(tl.float64)
        base = tl.where(offs < half, start_side, end_side).to(tl.float32)

        # Shift transform with pre-rounded scalars. Every fp32 step is
        # computed through an exact fp64 intermediate rounded once to fp32:
        # an fp32 x fp32 product or sum is exactly representable in fp64, so
        # this is bit-identical to the correctly rounded (round-to-nearest,
        # NON-FTZ) fp32 op that torch's CPU (SSE) kernels perform. Plain
        # Triton fp32 mul/add may compile flush-to-zero, which zeros the
        # subnormal numerators of subnormal shifts (fl32(1e-45) = 2^-149) and
        # broke upstream parity (independent audit 2026-10-16).
        base64 = base.to(tl.float64)
        numerator = (shift_f32.to(tl.float64) * base64).to(tl.float32)
        inner = (shift_m1_f32.to(tl.float64) * base64).to(tl.float32)
        denominator = (1.0 + inner.to(tl.float64)).to(tl.float32)

        # Division via inline PTX div.rn.f32: correctly rounded AND non-FTZ
        # (no .ftz modifier — IEEE subnormal handling, nonzero/0 -> inf).
        # The libdevice div_rn binding was measured on Triton 3.1.0 / RTX
        # 4090D to flush subnormal DIVIDENDS: div_rn(2^-149, +0.0) returned
        # NaN instead of +inf and subnormal/normal returned +0, collapsing
        # the 1e-45 grid (probe: remote/diag/subnormal_probe.py). This makes
        # the kernel NVIDIA-PTX-specific — see the module docstring.
        value = tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            "=r,r,r",
            [numerator, denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

        # NaN canonicalization for finite fl32(shift): the only NaN source is
        # the invalid 0/+0 at element 0 — a shift whose fl32 underflows to
        # +0.0 (e.g. 1e-46) makes (shift-1)*1 round to -1.0 and the
        # denominator exactly +0.0 there while the numerator is +0.0. torch
        # CPU evaluates 0/+0 to the x86 default quiet NaN 0xffc00000 (sign
        # bit SET); the device division yields 0x7fffffff. Rewrite NaN
        # results to the CPU bit pattern. (A subnormal fl32(shift) such as
        # 1e-45 takes numerator 2^-149 / +0.0 = +inf — a valid IEEE result —
        # and flows through untouched.)
        value_int = value.to(tl.int32, bitcast=True)
        cpu_qnan = tl.full([BLOCK], -4194304, tl.int32)  # 0xffc00000 signed
        out_int = tl.where(value != value, cpu_qnan, value_int)
        tl.store(out_ptr + offs, out_int.to(tl.float32, bitcast=True), mask=mask)

    @triton.jit
    def _unique_consecutive_kernel(
        src_ptr,
        dst_ptr,
        count_ptr,
        n,
        BLOCK: tl.constexpr,
    ):
        # Single program: sequential over blocks, so the scan order — and
        # therefore the compaction result — is deterministic for any n.
        running = tl.zeros((1,), dtype=tl.int32)
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n
            cur = tl.load(src_ptr + offs, mask=mask, other=0.0)
            # offs == 0 sees NaN as "previous" and is always kept; IEEE
            # equality keeps NaN rows (NaN != NaN), matching torch.
            prev = tl.load(src_ptr + offs - 1, mask=mask & (offs > 0), other=float("nan"))
            keep = mask & (cur != prev)
            keep_i32 = keep.to(tl.int32)
            dest = running + tl.cumsum(keep_i32, axis=0) - keep_i32
            tl.store(dst_ptr + dest, cur, mask=keep)
            running = running + tl.sum(keep_i32, axis=0)
        tl.store(count_ptr + tl.arange(0, 1), running)

    @triton.jit
    def _timesteps_kernel(sig_ptr, out_ptr, m, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < m - 1
        sigma = tl.load(sig_ptr + offs, mask=mask, other=0.0)
        # 1 - sigma, except NaN sigmas pass through unchanged: torch CPU
        # (SSE) returns the quiet NaN operand itself from 1.0 - NaN — payload
        # and sign included — while the device subtract would canonicalize to
        # 0x7fffffff. The grid's NaNs are already in CPU bit form, so copying
        # sigma reproduces the CPU result for both NaN signs.
        timesteps = tl.where(sigma != sigma, sigma, 1.0 - sigma)
        tl.store(out_ptr + offs, timesteps, mask=mask)


class H3SigmaScheduleCudaOp:
    """Device-side H3 sigma grid via Triton (NVIDIA CUDA only)."""

    def __init__(self) -> None:
        # Platform gate first: the kernels emit inline PTX (div.rn.f32) and
        # cannot run on AMD. Raising ImportError — like the Triton check
        # below — keeps the registry's candidate-skip semantics
        # (docs/architecture/runtime-dispatch.md) intact if this class ever
        # appears in a HIP device table; construction fails here, long
        # before any Triton compilation is attempted.
        if torch.version.hip is not None:
            raise ImportError(
                "H3SigmaScheduleCudaOp is NVIDIA-only: its kernels emit inline "
                "PTX div.rn.f32 for exact IEEE division and cannot run on "
                "HIP/ROCm. Use the PyTorch reference "
                "(rl_engine.reference.minimax_h3.sigma_schedule) on AMD."
            )
        if not _TRITON_AVAILABLE:
            # Construction-time failure lets the registry's candidate loop
            # skip this backend (docs/architecture/runtime-dispatch.md);
            # forward-time errors are never used for selection.
            raise ImportError("triton is required for H3SigmaScheduleCudaOp")

    def __call__(
        self,
        num_inference_steps: int,
        *,
        shift: float,
        device: str | torch.device | None = None,
    ) -> H3SigmaSchedule:
        return self.forward(num_inference_steps, shift=shift, device=device)

    def forward(
        self,
        num_inference_steps: int,
        *,
        shift: float,
        device: str | torch.device | None = None,
    ) -> H3SigmaSchedule:
        """Build the grid on the target device; see the module docstring."""
        # Same checks and messages as the CPU reference (upstream's own
        # failures; no extra policy).
        if num_inference_steps < 2:
            raise ValueError(
                "num_inference_steps must be >= 2 (terminal zero included), got "
                f"{num_inference_steps}."
            )
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")

        target = torch.device(device) if device is not None else torch.device("cuda")
        if target.type != "cuda":
            raise ValueError(
                "H3SigmaScheduleCudaOp builds the grid on an NVIDIA CUDA device "
                "(the kernels are NVIDIA-only); pass device='cuda' or use the "
                "PyTorch reference for CPU or AMD platforms."
            )

        n = int(num_inference_steps)
        shift_f32, shift_m1_f32 = _shift_scalars(shift)

        # NaN/inf scalar fast path (module docstring "NaN payloads"). A NaN
        # fl32(shift) propagates its own quieted bits through the whole grid
        # on the CPU oracle — SIGN INCLUDED, so -NaN yields 0xffc00000 while
        # +NaN yields 0x7fc00000 (quieting = setting bit 22). An infinite
        # fl32(shift) makes every element an invalid operation (inf/inf, or
        # inf*0 at the terminal zero), which x86 evaluates to the default
        # quiet NaN 0xffc00000 regardless of the shift's sign. The device
        # kernel canonicalizes NaNs to 0x7fffffff instead, so these grids are
        # filled directly with the oracle's bit patterns. All NaN rows
        # survive unique_consecutive (IEEE !=), so M = n.
        if math.isnan(shift_f32) or math.isinf(shift_f32):
            if math.isinf(shift_f32):
                payload = 0xFFC00000
            else:
                raw = int(
                    torch.tensor(shift, dtype=torch.float32, device="cpu").view(torch.int32).item()
                )
                payload = raw & 0xFFFFFFFF | 0x00400000  # quiet the NaN bits
            signed = payload if payload < 0x80000000 else payload - 0x1_0000_0000
            with torch.cuda.device(target):
                bits = torch.full((n,), signed, dtype=torch.int32, device=target)
                sigmas = bits.view(torch.float32)
                timesteps = sigmas[: n - 1]
            return H3SigmaSchedule(sigmas=sigmas, timesteps=timesteps, num_model_evals=n - 1)

        # Device guard: allocations, Triton launches (which use the current
        # device/stream) and the count read must all target `target`, also
        # when it is not the current device (cuda:k on multi-GPU hosts).
        with torch.cuda.device(target):
            grid = torch.empty(n, dtype=torch.float32, device=target)
            compacted = torch.empty(n, dtype=torch.float32, device=target)
            count = torch.empty(1, dtype=torch.int32, device=target)

            blocks = triton.cdiv(n, _BLOCK)
            _shifted_grid_kernel[(blocks,)](
                grid, n, shift_f32, shift_m1_f32, BLOCK=_BLOCK, enable_fp_fusion=False
            )
            _unique_consecutive_kernel[(1,)](grid, compacted, count, n, BLOCK=_BLOCK)
            # The single allowed host synchronization: the deduplicated length.
            m = int(count.item())
            sigmas = compacted[:m]
            timesteps = torch.empty(m - 1, dtype=torch.float32, device=target)
            if m > 1:
                _timesteps_kernel[(triton.cdiv(m - 1, _BLOCK),)](
                    compacted, timesteps, m, BLOCK=_BLOCK, enable_fp_fusion=False
                )
        return H3SigmaSchedule(sigmas=sigmas, timesteps=timesteps, num_model_evals=m - 1)
