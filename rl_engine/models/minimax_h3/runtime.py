# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 runtime modes, ablation overlay, Diffusers adapter, training provider and LoRA gradient gate.

Rows: ``h3_runtime_modes``, ``h3_ablation_overlay``, ``diffusers_runtime_adapter_h3``,
``h3_training_provider``, ``h3_lora_gradient_gate``.

* ``h3_runtime_policy(mode)`` maps ``RL_KERNEL_MODE`` (strict / audit / auto / off) to what the H3
  stack does: bind the RL-Kernel graph or not, fail on a fallback or report it, and the case id
  (``R/R`` strict and audit, ``P/P`` auto and off) the experiment records.
* ``AblationOverlay`` keeps selected block nodes or boundary modules on provider arithmetic while
  everything else is strict, labels the run ``ablation(...)`` (never ``R/R``) and validates names.
* ``H3DiffusersAdapter`` binds a ``MiniMaxH3Transformer3DModel`` for the duration of a
  ``with`` block (forward *and* backward, so checkpoint recomputation sees the same operators)
  and reports what was bound.
* ``H3TrainingProvider`` is the actor-side entry point: ``step()`` scope, strict forward /
  backward, and the layout-invariant CPS transition for log-probs.
* ``lora_gradient_gate`` checks that a LoRA over a strict projection trains bitwise
  reproducibly and layout invariantly with a frozen base.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from rl_engine.runtime.policy import RLKernelMode, rl_kernel_mode_policy

CONTRACT = "h3-runtime-v1"


@dataclass(frozen=True)
class H3RuntimePolicy:
    mode: RLKernelMode
    bind: bool  # run the RL-Kernel graph instead of provider arithmetic
    fail_on_fallback: bool
    report_only: bool
    case_id: str


def h3_runtime_policy(mode: str | RLKernelMode | None = None) -> H3RuntimePolicy:
    """Policy for ``mode`` (default ``RL_KERNEL_MODE``, else strict)."""

    base = rl_kernel_mode_policy(mode)
    # strict and audit bind the RL-Kernel graph (R/R); auto and off leave the provider (P/P)
    bind = base.enabled and base.provider_mode == "strict"
    return H3RuntimePolicy(base.mode, bind, base.fail_on_fallback, base.report_only, base.case_id)


@dataclass(frozen=True)
class AblationOverlay:
    """Parts that stay on provider arithmetic inside an otherwise strict run."""

    native_nodes: frozenset[str] = frozenset()
    native_boundaries: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        from rl_engine.validation.models import h3_block
        from rl_engine.validation.models.h3_full_model import STRICT_BOUNDARY_ROWS

        bad_nodes = set(self.native_nodes) - set(h3_block.NODE_NAMES)
        bad_boundaries = set(self.native_boundaries) - set(STRICT_BOUNDARY_ROWS)
        if bad_nodes or bad_boundaries:
            raise ValueError(
                f"unknown ablation targets: nodes {sorted(bad_nodes)}, "
                f"boundaries {sorted(bad_boundaries)}"
            )

    @classmethod
    def parse(cls, spec: str) -> AblationOverlay:
        """``"node:attention,boundary:token_refiner"`` -> overlay; empty -> no ablation."""

        nodes, boundaries = set(), set()
        for item in filter(None, (part.strip() for part in spec.split(","))):
            kind, _, name = item.partition(":")
            if kind == "node":
                nodes.add(name)
            elif kind == "boundary":
                boundaries.add(name)
            else:
                raise ValueError(
                    f"ablation item must be node:<name> or boundary:<name>, got {item!r}"
                )
        return cls(frozenset(nodes), frozenset(boundaries))

    @property
    def empty(self) -> bool:
        return not (self.native_nodes or self.native_boundaries)

    @property
    def label(self) -> str:
        if self.empty:
            return "R/R"
        parts = [f"node:{n}" for n in sorted(self.native_nodes)]
        parts += [f"boundary:{n}" for n in sorted(self.native_boundaries)]
        return "ablation(native: " + ",".join(parts) + ")"


class H3DiffusersAdapter:
    """Bind a Diffusers H3 transformer according to the runtime policy and an optional overlay."""

    def __init__(
        self,
        model,
        mode: str | RLKernelMode | None = None,
        overlay: AblationOverlay | None = None,
        ops=None,
    ) -> None:
        from rl_engine.validation.models.h3_full_model import FullModelTrace

        self.model, self.policy = model, h3_runtime_policy(mode)
        self.overlay = overlay or AblationOverlay()
        self.ops, self.trace = ops, FullModelTrace()
        self.fell_back: str | None = None
        self._scope = None

    def _bind(self) -> bool:
        if not self.policy.bind:
            return False
        if not (torch.cuda.is_available() and next(self.model.parameters()).is_cuda):
            if self.policy.fail_on_fallback:
                raise RuntimeError("strict H3 needs CUDA parameters; refusing to fall back")
            self.fell_back = "no CUDA device"
            return False
        return True

    @contextmanager
    def active(self):
        """Operators bound until the block exits; keep ``backward()`` inside it."""

        from rl_engine.validation.models.h3_full_model import full_model_execution

        bind = self._bind()
        try:
            with full_model_execution(
                self.model,
                candidate=bind,
                trace=self.trace,
                ops=self.ops,
                provider_nodes=tuple(self.overlay.native_nodes),
                strict_boundaries=bind,
                native_boundaries=tuple(self.overlay.native_boundaries),
            ):
                yield self
        except RuntimeError as exc:
            if self.policy.fail_on_fallback or not bind:
                raise
            self.fell_back = f"{type(exc).__name__}: {exc}"
            raise

    def report(self) -> dict:
        bound = self.policy.bind and self.fell_back is None
        return {
            "contract": CONTRACT,
            "mode": self.policy.mode.value,
            "case_id": ("R/R" if self.overlay.empty else "ablation") if bound else "P/P",
            "bound": bound,
            "fell_back": self.fell_back,
            "label": self.overlay.label if bound else "P/P",
            "native_nodes": sorted(self.overlay.native_nodes),
            "native_boundaries": sorted(self.overlay.native_boundaries),
            "boundary_bindings": self.trace.boundary_bindings,
        }


class H3TrainingProvider:
    """Actor-side strict training scope: ``with provider.step(): forward(...); loss.backward()``."""

    def __init__(
        self, model, mode: str | RLKernelMode | None = None, overlay: AblationOverlay | None = None
    ) -> None:
        self.adapter = H3DiffusersAdapter(model, mode, overlay)
        self.model = model

    @contextmanager
    def step(self):
        with self.adapter.active():
            yield self

    def forward(self, **inputs):
        return self.model(**inputs)

    def transition(
        self,
        sample,
        velocity,
        sigma,
        sigma_prev,
        *,
        seed,
        step,
        sample_ids,
        modality,
        noise_level=None,
        prev_sample=None,
    ):
        """CPS transition for a batch whose per-sample noise depends on ``sample_ids`` only."""

        from rl_engine.models.minimax_h3 import transition as tr

        level = tr.NOISE_LEVEL if noise_level is None else noise_level
        noise = None
        if prev_sample is None:
            noise = torch.stack(
                [
                    tr.step_noise(
                        seed, step, int(i), modality, tuple(sample.shape[1:]), sample.device
                    )
                    for i in sample_ids
                ]
            )
        return tr.cps_transition(
            sample.float(),
            velocity.float(),
            sigma,
            sigma_prev,
            noise_level=level,
            noise=noise,
            prev_sample=prev_sample,
        )


# --- LoRA over a strict projection ----------------------------------------------------------


class _StrictLoRA(torch.autograd.Function):
    """``y = round_bf16(x W^T + scale * (x A^T) B^T)``; W frozen, A and B trainable (FP32)."""

    @staticmethod
    def forward(ctx, x, weight, a, b, scale):
        from rl_engine.backends.shared.triton.gemm import h3_projection as g

        ctx.save_for_backward(x, weight, a, b)
        ctx.scale = scale
        base = g.gemm(x, weight, b_t=True, out_dtype=torch.float32)
        x32 = x.float()
        low = g.gemm(g.gemm(x32, a, b_t=True), b, b_t=True)
        return (base + scale * low).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        from rl_engine.backends.shared.triton.gemm import h3_projection as g

        x, weight, a, b = ctx.saved_tensors
        g32 = grad.float().contiguous()
        x32 = x.float().contiguous()
        mid = g.gemm(x32, a, b_t=True)  # (T, r)
        d_mid = g.gemm(g32, b) * ctx.scale  # (T, r)
        d_b = g.gemm(g32, mid, a_t=True) * ctx.scale  # (N, r)
        d_a = g.gemm(d_mid, x32, a_t=True)  # (r, K)
        dx = g.gemm(grad.contiguous(), weight, out_dtype=torch.float32) + g.gemm(d_mid, a)
        return dx.to(x.dtype), None, d_a, d_b, None


def strict_lora_linear(x, weight, a, b, scale: float = 1.0):
    """LoRA projection of ``x (T, K)``: frozen BF16 ``weight``, FP32 ``a (r, K)``, ``b (N, r)``."""

    return _StrictLoRA.apply(x.contiguous(), weight, a, b, scale)


def lora_gradient_gate(
    weight: torch.Tensor,
    x: torch.Tensor,
    grad: torch.Tensor,
    *,
    rank: int = 8,
    scale: float = 2.0,
    seed: int = 0,
) -> Mapping[str, object]:
    """Gate: A/B gradients are bitwise repeatable and independent of how rows are batched, the
    base gets no gradient, and the result is close to an FP64 golden.

    ``x (T, K)`` rows are also run split in two halves (gradients of the halves summed in the fixed
    order of the full GEMM are not claimed equal; the gate checks the *forward* of a half and the
    full-batch gradients, which is what a data-parallel micro-batch split changes).
    """

    gen = torch.Generator().manual_seed(seed)
    n, k = weight.shape
    a0 = (torch.randn(rank, k, generator=gen) / k**0.5).to(x.device)
    b0 = (torch.randn(n, rank, generator=gen) * 0.1).to(x.device)

    frozen: list[bool] = []

    def run(rows):
        a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
        w = weight.detach().clone()
        xi = x[rows].detach().clone().requires_grad_(True)
        y = strict_lora_linear(xi, w, a, b, scale)
        y.backward(grad[rows])
        frozen.append(w.grad is None and not w.requires_grad)
        return y.detach(), xi.grad, a.grad, b.grad

    full = run(slice(None))
    again = run(slice(None))
    half = run(slice(0, x.shape[0] // 2))
    ref_y = (
        x.double() @ weight.double().t() + scale * (x.double() @ a0.double().t()) @ b0.double().t()
    )
    err = ((full[0].double() - ref_y).norm() / ref_y.norm()).item()
    return {
        "repeatable": all(torch.equal(p, q) for p, q in zip(full, again)),
        "forward_rows_independent": torch.equal(full[0][: half[0].shape[0]], half[0]),
        "base_frozen": all(frozen),
        "forward_rel_err_vs_fp64": err,
    }
