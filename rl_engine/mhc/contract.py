# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Data contracts for P1: LayerContract, ResidualBatch, controller/norm params."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from rl_engine.mhc.reduction import HC_MULT

SCHEMA_VERSION = "p1-mhc-layer-v1"

# The oracle's numeric profile: FP32 math everywhere, long reductions as a
# serial ascending left fold, 4-way reductions as the pinned (a0+a1)+(a2+a3)
# tree, mul-then-add rounding (no FMA fusion). Kernel backends declare theirs.
ORACLE_PROFILE = "oracle-fp32-mhc-v1"

# --- frozen production constants (issue #2) --------------------------------
PROD_HIDDEN = 4096  # D
PROD_CONTROLLER_N = 24  # PRE 4 + POST 4 + COMB 16
SINKHORN_ITERS = 20
MHC_EPS = 1e-6  # controller / Sinkhorn epsilon
RMSNORM_EPS = 1e-6

# Controller output layout, frozen: PRE[0:4], POST[4:8], COMB[8:24].
PRE_SLICE = slice(0, 4)
POST_SLICE = slice(4, 8)
COMB_SLICE = slice(8, 24)

PLACEMENTS = ("replicated", "tp-sharded", "cp-token-sharded")
ROW_GEOMETRIES = ("one-row", "packed")

# Modes the kit defines. Anything else must fail closed (acceptance 4).
#
# ``unfused`` is CANONICAL and the default. Train/infer byte-equality is the
# goal, and the unfused decomposition is the only one where every operator
# boundary can be hashed and compared on its own, so a divergence localizes to
# one operator instead of one megakernel. ``fused-pre-norm`` exists for an
# inference engine that physically cannot expose the pre-normalization
# intermediate (TE's ``TEFusedResidualRMSNorm`` refuses to: it raises if you
# hook it). Fusion is not forbidden -- what is forbidden is a fused kernel that
# changes the reduction layout or moves a downcast point. Such a kernel must
# register its own numeric profile rather than claim to be the same operator.
FUSION_MODES = ("unfused", "fused-pre-norm")
CANONICAL_FUSION_MODE = "unfused"
TRAINABILITY_MODES = ("full", "mixer-frozen")


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a tensor, independent of layout."""
    flat = t.detach().contiguous().flatten()
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).cpu().numpy().tobytes()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(t)).hexdigest()


@dataclass(frozen=True)
class LayerContract:
    """Frozen shape/constant contract for one mHC + RMSNorm decoder boundary.

    ``hidden`` and ``controller_n`` are parameters so the start-kit fixtures
    can run a scaled-down layer on CPU, but :meth:`assert_production` pins the
    real DSv4 numbers. ``hc_mult``, ``sinkhorn_iters`` and both epsilons are
    *not* negotiable -- changing any of them is a schema bump.
    """

    hidden: int = PROD_HIDDEN
    hc_mult: int = HC_MULT
    controller_n: int = PROD_CONTROLLER_N
    sinkhorn_iters: int = SINKHORN_ITERS
    mhc_eps: float = MHC_EPS
    rmsnorm_eps: float = RMSNORM_EPS
    layer_index: int = 0
    placement: str = "replicated"
    fusion_mode: str = "unfused"
    trainability: str = "full"
    schema_version: str = SCHEMA_VERSION
    numeric_profile: str = ORACLE_PROFILE

    @property
    def flat_k(self) -> int:
        """Controller GEMM K: the flattened four-stream residual width."""
        return self.hc_mult * self.hidden

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema {self.schema_version!r} != {SCHEMA_VERSION!r}")
        if self.hc_mult != HC_MULT:
            raise ValueError(f"hc_mult is frozen at {HC_MULT}, got {self.hc_mult}")
        if self.controller_n != 2 * self.hc_mult + self.hc_mult**2:
            raise ValueError(
                f"controller_n {self.controller_n} != PRE+POST+COMB "
                f"({2 * self.hc_mult + self.hc_mult ** 2})"
            )
        if self.sinkhorn_iters != SINKHORN_ITERS:
            raise ValueError(f"sinkhorn_iters is frozen at {SINKHORN_ITERS}")
        if self.mhc_eps != MHC_EPS or self.rmsnorm_eps != RMSNORM_EPS:
            raise ValueError("eps values are frozen at 1e-6")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"unknown placement {self.placement!r}")
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion_mode {self.fusion_mode!r}; want {FUSION_MODES}")
        if self.trainability not in TRAINABILITY_MODES:
            raise ValueError(
                f"unknown trainability {self.trainability!r}; want {TRAINABILITY_MODES}"
            )
        if self.hidden <= 0:
            raise ValueError("hidden must be positive")

    def assert_production(self) -> None:
        """Fail unless this is the real DSv4 layer geometry."""
        if (self.hidden, self.controller_n, self.flat_k) != (
            PROD_HIDDEN,
            PROD_CONTROLLER_N,
            PROD_HIDDEN * HC_MULT,
        ):
            raise ValueError(
                f"not the production contract: hidden={self.hidden} "
                f"controller_n={self.controller_n} K={self.flat_k}"
            )

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for value in (
            self.hidden,
            self.hc_mult,
            self.controller_n,
            self.sinkhorn_iters,
            repr(self.mhc_eps),
            repr(self.rmsnorm_eps),
            self.fusion_mode,
            self.trainability,
            self.schema_version,
        ):
            h.update(str(value).encode())
        return h.hexdigest()


@dataclass(frozen=True)
class ControllerParams:
    """Weights of the mHC controller projection (`fp32_gemm_rms` + affine).

    All FP32: issue #2 pins the controller path to FP32 end to end, with no
    intermediate BF16 cast between the projection, the RMS scale and the
    Sinkhorn split.
    """

    weight: torch.Tensor  # FP32 [controller_n, K]
    alpha_pre: torch.Tensor  # FP32 scalar [1]
    alpha_post: torch.Tensor  # FP32 scalar [1]
    alpha_res: torch.Tensor  # FP32 scalar [1]
    bias: torch.Tensor  # FP32 [controller_n]

    @property
    def alphas(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.alpha_pre, self.alpha_post, self.alpha_res)

    def expanded_alpha(self, contract: LayerContract) -> torch.Tensor:
        """``cat([alpha_pre.expand(n), alpha_post.expand(n), alpha_res.expand(n*n)])``.

        Megatron holds three learnable *scalars* and broadcasts them across the
        PRE / POST / COMB segments (``hyper_connection.py`` ``_compute_h``); it
        does not hold 24 independent gains. Forward is identical either way,
        but backward is not: ``dAlpha`` is three numbers, each a reduction over
        its segment, so the segment reduction has to be pinned like any other.
        """
        n = contract.hc_mult
        return torch.cat(
            [
                self.alpha_pre.expand(n),
                self.alpha_post.expand(n),
                self.alpha_res.expand(n * n),
            ],
            dim=-1,
        )

    def validate(self, contract: LayerContract) -> None:
        named = (
            ("weight", self.weight),
            ("alpha_pre", self.alpha_pre),
            ("alpha_post", self.alpha_post),
            ("alpha_res", self.alpha_res),
            ("bias", self.bias),
        )
        for name, t in named:
            if t.dtype != torch.float32:
                raise TypeError(f"controller {name} must be FP32, got {t.dtype}")
        n, k = contract.controller_n, contract.flat_k
        if tuple(self.weight.shape) != (n, k):
            raise ValueError(f"controller weight {tuple(self.weight.shape)} != {(n, k)}")
        if tuple(self.bias.shape) != (n,):
            raise ValueError(f"controller bias {tuple(self.bias.shape)} != {(n,)}")
        for name, t in named[1:4]:
            if tuple(t.shape) != (1,):
                raise ValueError(f"controller {name} must be a scalar [1], got {tuple(t.shape)}")

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for t in (self.weight, self.alpha_pre, self.alpha_post, self.alpha_res, self.bias):
            h.update(tensor_bytes(t))
        return h.hexdigest()

    def to(self, device: torch.device | str) -> "ControllerParams":
        return ControllerParams(
            self.weight.to(device),
            self.alpha_pre.to(device),
            self.alpha_post.to(device),
            self.alpha_res.to(device),
            self.bias.to(device),
        )


@dataclass(frozen=True)
class NormParams:
    """RMSNorm gain. BF16 storage, promoted to FP32 inside the operator."""

    gamma: torch.Tensor  # BF16 [hidden]

    def validate(self, contract: LayerContract) -> None:
        if self.gamma.dtype != torch.bfloat16:
            raise TypeError(f"gamma must be BF16, got {self.gamma.dtype}")
        if tuple(self.gamma.shape) != (contract.hidden,):
            raise ValueError(f"gamma shape {tuple(self.gamma.shape)} != {(contract.hidden,)}")

    def fingerprint(self) -> str:
        return hashlib.sha256(tensor_bytes(self.gamma)).hexdigest()

    def to(self, device: torch.device | str) -> "NormParams":
        return NormParams(self.gamma.to(device))


@dataclass(frozen=True)
class ResidualBatch:
    """One mHC block boundary: the four-stream residual plus the sublayer edge.

    ``r_old`` is the versioned four-way residual identity entering the block.
    ``token_id`` and ``layer_index`` carry the global-token / absolute-layer
    identity required by the Foundation ``SemanticTensor``; they are passed
    through untouched and only participate in fingerprints.

    The transformer sublayer (attention / dense FFN / MoE) is *external* to
    P1: this batch carries the two tensors that cross that boundary --
    ``y_sublayer`` (its BF16 output, consumed by ``mhc_post``) and, for the
    backward direction, the incoming ``d_normalized`` / ``d_residual`` in
    :class:`GradBoundary`. That is what lets P1 be developed and accepted with
    no P2-P7 code in the loop.
    """

    r_old: torch.Tensor  # BF16 [T, 4, hidden]
    y_sublayer: torch.Tensor  # BF16 [T, hidden]
    controller: ControllerParams
    norm: NormParams
    token_id: torch.Tensor  # int64 [T] global token identity
    contract: LayerContract = field(default_factory=LayerContract)
    row_geometry: str = "packed"
    weight_fingerprint: str = ""

    @property
    def tokens(self) -> int:
        return int(self.r_old.shape[0])

    @property
    def hidden(self) -> int:
        return int(self.r_old.shape[2])

    def validate(self) -> None:
        self.contract.validate()
        if self.row_geometry not in ROW_GEOMETRIES:
            raise ValueError(f"row_geometry {self.row_geometry!r} not in {ROW_GEOMETRIES}")
        if self.r_old.dtype != torch.bfloat16:
            raise TypeError(f"r_old must be BF16, got {self.r_old.dtype}")
        if self.y_sublayer.dtype != torch.bfloat16:
            raise TypeError(f"y_sublayer must be BF16, got {self.y_sublayer.dtype}")
        if self.token_id.dtype != torch.int64:
            raise TypeError(f"token_id must be int64, got {self.token_id.dtype}")
        t, streams, hidden = self.r_old.shape
        if streams != self.contract.hc_mult:
            raise ValueError(f"r_old has {streams} streams, contract says {self.contract.hc_mult}")
        if hidden != self.contract.hidden:
            raise ValueError(f"r_old hidden {hidden} != contract {self.contract.hidden}")
        if tuple(self.y_sublayer.shape) != (t, hidden):
            raise ValueError(f"y_sublayer shape {tuple(self.y_sublayer.shape)} != {(t, hidden)}")
        if tuple(self.token_id.shape) != (t,):
            raise ValueError(f"token_id shape {tuple(self.token_id.shape)} != {(t,)}")
        if self.row_geometry == "one-row" and t != 1:
            raise ValueError("row_geometry 'one-row' requires exactly one token")
        self.controller.validate(self.contract)
        self.norm.validate(self.contract)
        expected = self.compute_weight_fingerprint()
        if self.weight_fingerprint and self.weight_fingerprint != expected:
            raise ValueError("weight_fingerprint mismatch: checkpoint bytes were modified")

    def compute_weight_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.contract.fingerprint().encode())
        h.update(self.controller.fingerprint().encode())
        h.update(self.norm.fingerprint().encode())
        return h.hexdigest()

    def sealed(self) -> "ResidualBatch":
        """Return a copy with ``weight_fingerprint`` filled in."""
        return replace(self, weight_fingerprint=self.compute_weight_fingerprint())

    def to(self, device: torch.device | str) -> "ResidualBatch":
        return ResidualBatch(
            r_old=self.r_old.to(device),
            y_sublayer=self.y_sublayer.to(device),
            controller=self.controller.to(device),
            norm=self.norm.to(device),
            token_id=self.token_id.to(device),
            contract=self.contract,
            row_geometry=self.row_geometry,
            weight_fingerprint=self.weight_fingerprint,
        )


@dataclass(frozen=True)
class GradBoundary:
    """The three incoming gradients at the P1 block's outer edges.

    - ``d_r_new``: from the next mHC block (or the loss), BF16 [T, 4, hidden].
    - ``d_normalized``: the sublayer's gradient w.r.t. the RMSNorm output.
    - ``d_residual``: the sublayer's gradient w.r.t. the *unnormalized* hidden
      forked by ``rmsnorm_residual``. Zero when nothing consumes the fork.

    Supplying the two sublayer-side gradients as data (instead of computing
    them) is what keeps P1 independent of P2-P7: ``dy`` produced by
    ``mhc_post_bwd`` is an *output* boundary that the sublayer owner consumes.
    """

    d_r_new: torch.Tensor  # BF16 [T, 4, hidden]
    d_normalized: torch.Tensor  # BF16 [T, hidden]
    d_residual: torch.Tensor  # BF16 [T, hidden]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, batch: ResidualBatch) -> None:
        t, hidden = batch.tokens, batch.hidden
        expect = {
            "d_r_new": (t, batch.contract.hc_mult, hidden),
            "d_normalized": (t, hidden),
            "d_residual": (t, hidden),
        }
        for name, shape in expect.items():
            got = getattr(self, name)
            if got.dtype != torch.bfloat16:
                raise TypeError(f"{name} must be BF16, got {got.dtype}")
            if tuple(got.shape) != shape:
                raise ValueError(f"{name} shape {tuple(got.shape)} != {shape}")

    def to(self, device: torch.device | str) -> "GradBoundary":
        return GradBoundary(
            d_r_new=self.d_r_new.to(device),
            d_normalized=self.d_normalized.to(device),
            d_residual=self.d_residual.to(device),
            metadata=dict(self.metadata),
        )


__all__ = [
    "CANONICAL_FUSION_MODE",
    "COMB_SLICE",
    "FUSION_MODES",
    "MHC_EPS",
    "ORACLE_PROFILE",
    "POST_SLICE",
    "PRE_SLICE",
    "PROD_CONTROLLER_N",
    "PROD_HIDDEN",
    "RMSNORM_EPS",
    "SCHEMA_VERSION",
    "SINKHORN_ITERS",
    "TRAINABILITY_MODES",
    "ControllerParams",
    "GradBoundary",
    "LayerContract",
    "NormParams",
    "ResidualBatch",
    "tensor_bytes",
    "tensor_sha256",
]
