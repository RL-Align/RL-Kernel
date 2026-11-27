# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Data contracts: ExpertBatch, SharedBatch, LoRA params."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from rl_engine.moe.mx_format import MX_BLOCK, MXTensor

SCHEMA_VERSION = "p5-expertbatch-v1"

GATE_CLAMP_MAX = 10.0
UP_CLAMP_MIN = -10.0
UP_CLAMP_MAX = 10.0

# The oracle's numeric profile: FP32 math, serial ascending-k reduction,
# mul-then-add rounding (no FMA fusion). Kernel backends declare their own.
ORACLE_PROFILE = "oracle-fp32-serial-v1"

ROW_GEOMETRIES = ("one-row", "packed")


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a tensor, independent of layout."""
    flat = t.detach().contiguous().flatten()
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).cpu().numpy().tobytes()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(t)).hexdigest()


def mx_fingerprint(t: MXTensor) -> str:
    h = hashlib.sha256()
    h.update(t.elem_format.encode())
    h.update(tensor_bytes(t.codes))
    h.update(tensor_bytes(t.scales))
    return h.hexdigest()


@dataclass(frozen=True)
class LoRAParams:
    """BF16 LoRA adapters shared across local experts (P5-3, P5-7).

    ``a1``/``b1`` insert after the packed gate/up projection (fc1) and
    ``a2``/``b2`` after the down projection (fc2). Base weights stay packed;
    the LoRA path never unpacks them.
    """

    a1: torch.Tensor  # BF16 [r, hidden]
    b1: torch.Tensor  # BF16 [2*ffn, r]
    a2: torch.Tensor  # BF16 [r, ffn]
    b2: torch.Tensor  # BF16 [hidden, r]
    alpha: float

    def validate(self, hidden: int, ffn: int) -> None:
        for name, t in (("a1", self.a1), ("b1", self.b1), ("a2", self.a2), ("b2", self.b2)):
            if t.dtype != torch.bfloat16:
                raise TypeError(f"LoRA {name} must be BF16, got {t.dtype}")
        rank = self.a1.shape[0]
        expect = {
            "a1": (rank, hidden),
            "b1": (2 * ffn, rank),
            "a2": (rank, ffn),
            "b2": (hidden, rank),
        }
        for name, shape in expect.items():
            got = tuple(getattr(self, name).shape)
            if got != shape:
                raise ValueError(f"LoRA {name} shape {got} != expected {shape}")

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for t in (self.a1, self.b1, self.a2, self.b2):
            h.update(tensor_bytes(t))
        h.update(repr(float(self.alpha)).encode())
        return h.hexdigest()


@dataclass(frozen=True)
class ExpertBatch:
    """Offline routed-expert input following the P5 start-kit contract.

    Rows are already EP-dispatched and sorted by local expert:
    rows ``expert_offsets[e] : expert_offsets[e + 1]`` belong to local expert
    ``e``. ``p_s`` is the route weight travelling with each row and
    ``output_slot`` is carried through untouched for the P6 combine.
    """

    x: torch.Tensor  # BF16 [M, hidden]
    expert_offsets: torch.Tensor  # int32 [n_local_experts + 1]
    p_s: torch.Tensor  # FP32 [M]
    w1: MXTensor  # e2m1 [E, 2*ffn, hidden] frozen base (gate rows then up rows)
    w2: MXTensor  # e2m1 [E, hidden, ffn] frozen base
    lora: LoRAParams | None
    output_slot: torch.Tensor  # int32 [M]
    row_geometry: str = "packed"
    schema_version: str = SCHEMA_VERSION
    numeric_profile: str = ORACLE_PROFILE
    weight_fingerprint: str = ""

    @property
    def hidden(self) -> int:
        return int(self.x.shape[1])

    @property
    def ffn(self) -> int:
        return int(self.w2.shape[2])

    @property
    def rows(self) -> int:
        return int(self.x.shape[0])

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema {self.schema_version!r} != {SCHEMA_VERSION!r}")
        if self.row_geometry not in ROW_GEOMETRIES:
            raise ValueError(f"row_geometry {self.row_geometry!r} not in {ROW_GEOMETRIES}")
        if self.x.dtype != torch.bfloat16:
            raise TypeError(f"x must be BF16, got {self.x.dtype}")
        if self.p_s.dtype != torch.float32:
            raise TypeError(f"p_s must be FP32, got {self.p_s.dtype}")
        if self.expert_offsets.dtype != torch.int32 or self.output_slot.dtype != torch.int32:
            raise TypeError("expert_offsets/output_slot must be int32")
        m, hidden = self.x.shape
        if self.p_s.shape != (m,) or self.output_slot.shape != (m,):
            raise ValueError("p_s/output_slot must have shape [M]")
        if hidden % MX_BLOCK != 0:
            raise ValueError(f"hidden {hidden} not divisible by {MX_BLOCK}")
        offsets = self.expert_offsets
        if int(offsets[0]) != 0 or int(offsets[-1]) != m:
            raise ValueError("expert_offsets must start at 0 and end at M")
        if bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError("expert_offsets must be non-decreasing")
        n_experts = offsets.numel() - 1
        ffn = self.ffn
        if tuple(self.w1.shape) != (n_experts, 2 * ffn, hidden):
            raise ValueError(f"w1 shape {self.w1.shape} != {(n_experts, 2 * ffn, hidden)}")
        if tuple(self.w2.shape) != (n_experts, hidden, ffn):
            raise ValueError(f"w2 shape {self.w2.shape} != {(n_experts, hidden, ffn)}")
        if self.w1.elem_format != "e2m1" or self.w2.elem_format != "e2m1":
            raise ValueError("base weights must be MXFP4 (e2m1)")
        if self.lora is not None:
            self.lora.validate(hidden, ffn)
        expected = self.compute_weight_fingerprint()
        if self.weight_fingerprint and self.weight_fingerprint != expected:
            raise ValueError("weight_fingerprint mismatch: packed base bytes were modified")

    def compute_weight_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(mx_fingerprint(self.w1).encode())
        h.update(mx_fingerprint(self.w2).encode())
        if self.lora is not None:
            h.update(self.lora.fingerprint().encode())
        return h.hexdigest()

    def to(self, device: torch.device | str) -> "ExpertBatch":
        lora = self.lora
        if lora is not None:
            lora = LoRAParams(
                lora.a1.to(device),
                lora.b1.to(device),
                lora.a2.to(device),
                lora.b2.to(device),
                lora.alpha,
            )
        return ExpertBatch(
            x=self.x.to(device),
            expert_offsets=self.expert_offsets.to(device),
            p_s=self.p_s.to(device),
            w1=self.w1.to(device),
            w2=self.w2.to(device),
            lora=lora,
            output_slot=self.output_slot.to(device),
            row_geometry=self.row_geometry,
            schema_version=self.schema_version,
            numeric_profile=self.numeric_profile,
            weight_fingerprint=self.weight_fingerprint,
        )


@dataclass(frozen=True)
class SharedBatch:
    """Shared-expert input: every valid token, no routing, no LoRA (P5-5, issue #64)."""

    x: torch.Tensor  # BF16 [T, hidden]
    w_fc1: torch.Tensor  # BF16 [2*ffn, hidden] frozen (gate rows then up rows)
    w_fc2: torch.Tensor  # BF16 [hidden, ffn] frozen
    placement: str = "replicated"
    schema_version: str = SCHEMA_VERSION
    numeric_profile: str = ORACLE_PROFILE
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.x.dtype != torch.bfloat16:
            raise TypeError(f"x must be BF16, got {self.x.dtype}")
        if self.w_fc1.dtype != torch.bfloat16 or self.w_fc2.dtype != torch.bfloat16:
            raise TypeError("shared weights must be BF16 in the v1 contract")
        t, hidden = self.x.shape
        two_ffn = self.w_fc1.shape[0]
        if two_ffn % 2 != 0 or self.w_fc1.shape[1] != hidden:
            raise ValueError(f"w_fc1 shape {tuple(self.w_fc1.shape)} inconsistent with x")
        if tuple(self.w_fc2.shape) != (hidden, two_ffn // 2):
            raise ValueError(f"w_fc2 shape {tuple(self.w_fc2.shape)} != {(hidden, two_ffn // 2)}")
        if self.placement not in ("replicated", "tp-sharded"):
            raise ValueError(f"unknown placement {self.placement!r}")

    def to(self, device: torch.device | str) -> "SharedBatch":
        return SharedBatch(
            x=self.x.to(device),
            w_fc1=self.w_fc1.to(device),
            w_fc2=self.w_fc2.to(device),
            placement=self.placement,
            schema_version=self.schema_version,
            numeric_profile=self.numeric_profile,
            metadata=dict(self.metadata),
        )
