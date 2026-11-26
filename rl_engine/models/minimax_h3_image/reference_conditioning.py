# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3-Image edit reference conditioning (RFC #738 ``image_reference_conditioning``).

An edit puts each encoded reference image into the packed document as fixed context. RFC #738
section 3 "Reference-image conditioning" lists eleven steps; steps 2-8 (crop, pixel
normalization, encode, posterior sample, FP16 round trip, patch crop, latent normalization)
end at the normalized reference latent and belong to ``image_vae_contract``. This module
starts from those latents and pins the rest:

* **Draw order.** One request generator feeds every random draw of an edit, in this order:
  one posterior sample per reference (VAE encode), then the generated-image noise, then one
  reference noise per reference, in reference order. :class:`DrawLedger` records each draw
  (stage, index, shape, generator state before and after, output digest) and
  :meth:`DrawLedger.check_order` refuses any other sequence. The H3 Ref2VA path draws its
  reference noise *before* the target; the image path does not (RFC #738 probe I12).
* **Noise mix.** ``x = t * z + (1 - t) * noise`` at ``t = 0.999`` with ``t`` an FP32
  tensor, so ``1 - t`` is evaluated in FP32, as the released pipeline mixes it
  (``scheduler.scale_noise(latents, torch.tensor(0.999), noise)``). The mix runs through the
  registered RFC #420 ``conditioning_noise_mix`` operator (PyTorch reference by default, or
  its CUDA kernel), not a local copy. A Python-float level keeps ``1 - t`` in FP64 and moves
  the bytes; that is a probe arm, not this contract.
* **Rows.** Each noised reference keeps its own latent geometry and is packed by the
  ``image_patch_pack_unpack`` adapter; the rows are concatenated in reference order and go
  in front of the generated rows. With a ``packed_layout`` the reference count, each
  reference's latent grid and the reserved row count must agree, or nothing is mixed.
* **Fixed rows.** The reference rows are not resampled and not written by the scheduler:
  they must be byte-identical to their initial value after every denoising step.
  :meth:`ReferenceConditioning.check_fixed` compares raw bits (``-0.0`` and NaN payloads
  included) and raises :class:`ReferenceRowsChanged` with the step and first changed row.

Upstream: ``MiniMaxH3ImageEditVaeEncoderStep`` (``encoders.py``),
``MiniMaxH3ImagePrepareLatentsStep``, ``MiniMaxH3ImageEditPrepareLatentsStep``
(``before_denoise.py``), ``MiniMaxH3ImageLoopSchedulerStep`` (``denoise.py``) and
``MiniMaxH3Scheduler.scale_noise`` of Diffusers at commit
875373ac7f5856071e345a3e394c74efccc6b9e6.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from rl_engine.models.minimax_h3.trajectory import f32_hex, tensor_digest
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import patch_pack as pp
from rl_engine.reference.conditioning.conditioning_noise_mix import NativeConditioningNoiseMixOp

CONTRACT_ID = "h3-image-reference-conditioning-v1"
SCHEMA_VERSION = "rlkernel.h3_image.reference_conditioning.v1"
REFERENCE_NOISE_LEVEL = 0.999  # MiniMaxH3ImageModularPipeline.reference_noise_aug
LATENT_CHANNELS = pp.CHANNELS
STAGES = ("posterior", "target_noise", "reference_noise")


class DrawOrderError(RuntimeError):
    """The generator was not consumed in the released order."""


class ReferenceRowsChanged(RuntimeError):
    """A reference row differs from its initial conditioned value."""


def generator_state_digest(generator: torch.Generator) -> str:
    """sha256 of the generator state bytes (same definition as ``image_vae``)."""

    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def draw_normal(
    shape: Sequence[int], generator: torch.Generator | None, device: torch.device | str
) -> torch.Tensor:
    """FP32 standard normal draw as ``diffusers.utils.torch_utils.randn_tensor`` makes it.

    A CPU generator draws on the CPU and the result moves to ``device``; a generator on
    ``device`` draws there. A CUDA generator for a CPU tensor is an error, as upstream.
    """

    device = torch.device(device)
    draw_device = device
    if generator is not None and generator.device.type != device.type:
        if generator.device.type != "cpu":
            raise ValueError(
                f"cannot draw a {device.type} tensor from a {generator.device.type} generator"
            )
        draw_device = torch.device("cpu")
    out = torch.randn(tuple(shape), generator=generator, device=draw_device, dtype=torch.float32)
    return out.to(device)


# --- draw ledger --------------------------------------------------------------------------


@dataclass(frozen=True)
class Draw:
    stage: str
    index: int
    shape: tuple[int, ...]
    before: str
    after: str
    digest: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "index": self.index,
            "shape": list(self.shape),
            "before": self.before,
            "after": self.after,
            "digest": self.digest,
        }


class DrawLedger:
    """Every draw of one edit request from its generator, in the order they happened."""

    def __init__(self, generator: torch.Generator) -> None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("DrawLedger needs the request's torch.Generator")
        self.generator = generator
        self.draws: list[Draw] = []

    def observe(self, stage: str, index: int, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run ``fn`` (which must draw from the ledger's generator) and record the draw.

        Used for draws made elsewhere, e.g. ``posterior.sample(generator=...)`` in the VAE.
        """

        _check_stage(stage, index)
        before = generator_state_digest(self.generator)
        out = fn()
        after = generator_state_digest(self.generator)
        self.draws.append(Draw(stage, index, tuple(out.shape), before, after, tensor_digest(out)))
        return out

    def draw(
        self, stage: str, index: int, shape: Sequence[int], device: torch.device | str
    ) -> torch.Tensor:
        """Draw FP32 noise (:func:`draw_normal`) and record it."""

        return self.observe(stage, index, lambda: draw_normal(shape, self.generator, device))

    def record(self, stage: str, index: int, shape: Sequence[int], before: str, after: str) -> None:
        """Record a draw whose state digests another contract already took (the image VAE
        contract's ``record["generator"]["draws"]`` entries)."""

        _check_stage(stage, index)
        self.draws.append(Draw(stage, index, tuple(int(s) for s in shape), before, after, None))

    def expected_order(self, num_references: int) -> list[tuple[str, int]]:
        refs = range(int(num_references))
        return (
            [("posterior", i) for i in refs]
            + [("target_noise", 0)]
            + [("reference_noise", i) for i in refs]
        )

    def check_order(self, num_references: int) -> None:
        """Raise :class:`DrawOrderError` unless the draws are the released sequence.

        The sequence is posterior 0..N-1, target noise, reference noise 0..N-1; each draw must
        advance the generator and start from the state the previous one left.
        """

        seen = [(d.stage, d.index) for d in self.draws]
        expected = self.expected_order(num_references)
        if seen != expected:
            raise DrawOrderError(f"generator draws {seen} != released order {expected}")
        for k, d in enumerate(self.draws):
            if d.before == d.after:
                raise DrawOrderError(
                    f"draw {k} ({d.stage} {d.index}) did not consume the generator"
                )
            if k and d.before != self.draws[k - 1].after:
                raise DrawOrderError(
                    f"draw {k} ({d.stage} {d.index}) does not start where draw {k - 1} ended: "
                    "something else consumed the generator in between"
                )

    def artifact(self) -> dict[str, Any]:
        draws = [d.to_dict() for d in self.draws]
        return {
            "generator_device": str(self.generator.device),
            "initial_seed": int(self.generator.initial_seed()),
            "draws": draws,
            "order_sha256": _sha256_json([[d["stage"], d["index"], d["shape"]] for d in draws]),
            "sha256": _sha256_json(draws),
        }


def _check_stage(stage: str, index: int) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown draw stage {stage!r}; expected one of {STAGES}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("draw index must be a non-negative int")


def draw_target_noise(
    ledger: DrawLedger, latent_hw: tuple[int, int], device: torch.device | str
) -> torch.Tensor:
    """The generated-image noise, ``(1, 24, 1, H, W)`` FP32, as ``PrepareLatentsStep`` draws it."""

    height, width = (int(v) for v in latent_hw)
    return ledger.draw("target_noise", 0, (1, LATENT_CHANNELS, 1, height, width), device)


# --- conditioning -------------------------------------------------------------------------


@dataclass
class ReferenceConditioning:
    """The conditioned reference rows of one edit request and what produced them."""

    rows: torch.Tensor  # (R, 96) FP32, references in packed order
    rows_per_reference: tuple[int, ...]
    latent_grids: tuple[tuple[int, int], ...]
    level: float
    digests: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def num_reference_rows(self) -> int:
        return int(self.rows.shape[0])

    def image_rows(self, target_rows: torch.Tensor) -> torch.Tensor:
        """``[reference rows | generated rows]``, the image rows the denoising loop starts from."""

        if target_rows.ndim != 2 or int(target_rows.shape[1]) != pp.WIDTH:
            raise ValueError(f"target rows must be (N, {pp.WIDTH}), got {list(target_rows.shape)}")
        if target_rows.dtype != self.rows.dtype or target_rows.device != self.rows.device:
            raise ValueError("target rows must share dtype and device with the reference rows")
        return torch.cat([self.rows, target_rows])

    def check_fixed(self, image_rows: torch.Tensor, *, step: int | None = None) -> str:
        """Reference prefix of ``image_rows`` must equal the initial rows bit for bit.

        ``image_rows`` is the ``(N, 96)`` image selection (references first) or a ``(1, N, 96)``
        batch of one. Returns the digest of the prefix.
        """

        rows = image_rows[0] if image_rows.ndim == 3 and image_rows.shape[0] == 1 else image_rows
        r = self.num_reference_rows
        if rows.ndim != 2 or int(rows.shape[0]) < r or int(rows.shape[1]) != pp.WIDTH:
            raise ValueError(
                f"image rows {list(image_rows.shape)} cannot hold {r} reference rows of width "
                f"{pp.WIDTH}"
            )
        prefix = rows[:r]
        if prefix.dtype != self.rows.dtype:
            raise ReferenceRowsChanged(
                f"step {step}: reference rows became {prefix.dtype}, were {self.rows.dtype}"
            )
        initial = self.rows.to(prefix.device)
        same = _bits(prefix) == _bits(initial)
        if not bool(same.all()):
            changed = (~same).any(dim=1).nonzero().flatten()
            first = int(changed[0])
            ref = sum(1 for n in _cumulative(self.rows_per_reference) if n <= first)
            raise ReferenceRowsChanged(
                f"step {step}: {int(changed.numel())} of {r} reference rows changed; first is "
                f"row {first} (reference {ref}), max |diff| "
                f"{float((prefix.float() - initial.float()).abs().max())}"
            )
        return tensor_digest(prefix)

    def artifact(self) -> dict[str, Any]:
        body = {
            "schema": SCHEMA_VERSION,
            "contract_id": CONTRACT_ID,
            "level": self.level,
            "level_f32_hex": f32_hex(self.level),
            "one_minus_level_f32_hex": f32_hex(1.0 - _f32(self.level)),
            "num_reference_rows": self.num_reference_rows,
            "rows_per_reference": list(self.rows_per_reference),
            "latent_grids": [list(g) for g in self.latent_grids],
            "digests": self.digests,
            "provenance": self.provenance,
        }
        body["artifact_sha256"] = _sha256_json(body)
        return body


def condition_references(
    reference_latents: Sequence[torch.Tensor],
    *,
    ledger: DrawLedger | None = None,
    noise: Sequence[torch.Tensor] | None = None,
    device: torch.device | str | None = None,
    layout: pl.PackedLayout | None = None,
    mix_op: Any | None = None,
    packer: pp.ImagePatchPacker | None = None,
    level: float = REFERENCE_NOISE_LEVEL,
) -> ReferenceConditioning:
    """Noise the normalized reference latents to ``t = level`` and pack them into rows.

    ``reference_latents`` are the ``image_vae_contract`` outputs, one FP32 ``(1, 24, h, w)``
    tensor per reference (whole 2x2 patches), in reference order. The noise is either drawn
    from ``ledger`` (one draw per reference, after the generated-image noise) or given as
    ``noise``, one tensor per reference with the latent's element count, as the pipeline's
    ``reference_noise`` input. ``device`` is the execution device (default: the first
    latent's); draws follow :func:`draw_normal`.
    """

    if not reference_latents:
        raise ValueError("an edit needs at least one reference latent")
    if (ledger is None) == (noise is None):
        raise ValueError("pass exactly one of ledger (draw the noise) or noise (pre-generated)")
    if noise is not None and len(noise) != len(reference_latents):
        raise ValueError(
            f"noise needs one tensor per reference, got {len(noise)} for "
            f"{len(reference_latents)} references"
        )
    grids: list[tuple[int, int]] = []
    for i, z in enumerate(reference_latents):
        if not isinstance(z, torch.Tensor) or z.ndim != 4 or tuple(z.shape[:2]) != (1, 24):
            shape = list(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            raise ValueError(f"reference {i} must be a (1, 24, h, w) latent, got {shape}")
        if z.dtype != torch.float32:
            raise TypeError(f"reference {i} must be float32 (normalized VAE latent), got {z.dtype}")
        grids.append(pp._check_hw((int(z.shape[2]), int(z.shape[3]))))
    rows_per_reference = tuple((h // pp.PATCH[1]) * (w // pp.PATCH[2]) for h, w in grids)
    if layout is not None:
        _check_layout(layout, grids, sum(rows_per_reference))
    device = torch.device(device) if device is not None else reference_latents[0].device
    op = mix_op if mix_op is not None else NativeConditioningNoiseMixOp()
    packer = packer if packer is not None else pp.ImagePatchPacker()
    t = torch.tensor(level, dtype=torch.float32, device=device)

    rows: list[torch.Tensor] = []
    noise_digests: list[str] = []
    latent_digests: list[str] = []
    row_digests: list[str] = []
    for i, z in enumerate(reference_latents):
        z = z.to(device)
        if ledger is not None:
            n = ledger.draw("reference_noise", i, tuple(z.shape), device)
        else:
            assert noise is not None
            given = noise[i]
            if given.numel() != z.numel():
                raise ValueError(
                    f"noise {i} has {given.numel()} elements, reference latent {list(z.shape)}"
                )
            n = given.to(device, torch.float32).reshape(z.shape)
        mixed = op.forward(z.contiguous(), t, n.contiguous())
        r = packer.pack_image(mixed)[0]
        rows.append(r)
        latent_digests.append(tensor_digest(z))
        noise_digests.append(tensor_digest(n))
        row_digests.append(tensor_digest(r))
    out = torch.cat(rows)
    return ReferenceConditioning(
        rows=out,
        rows_per_reference=rows_per_reference,
        latent_grids=tuple(grids),
        level=float(level),
        digests={
            "rows": tensor_digest(out),
            "per_reference_rows": row_digests,
            "latents": latent_digests,
            "noise": noise_digests,
        },
        provenance={
            "mix": {
                "semantic_op": "conditioning_noise_mix",
                "backend_id": str(getattr(op, "backend_id", type(op).__name__)),
                "level_dtype": "float32",
            },
            "pack": packer.provenance(),
            "noise_source": "ledger" if ledger is not None else "given",
            "device": str(device),
        },
    )


def _check_layout(layout: pl.PackedLayout, grids: list[tuple[int, int]], rows: int) -> None:
    if len(layout.reference_grids) != len(grids):
        raise ValueError(
            f"layout has {len(layout.reference_grids)} references, {len(grids)} latents given"
        )
    for i, (want, got) in enumerate(zip(layout.reference_grids, grids)):
        if tuple(want) != tuple(got):
            raise ValueError(f"reference {i}: layout grid {tuple(want)} != latent grid {got}")
    if layout.num_reference_rows != rows:
        raise ValueError(
            f"layout reserved {layout.num_reference_rows} reference rows, the references pack "
            f"into {rows}"
        )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32 if t.element_size() == 4 else torch.int16)


def _cumulative(values: Sequence[int]) -> list[int]:
    out, total = [], 0
    for v in values:
        total += int(v)
        out.append(total)
    return out


def _f32(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def contract_manifest() -> dict[str, Any]:
    """Static description of the contract (no tensors)."""

    return {
        "contract_id": CONTRACT_ID,
        "schema": SCHEMA_VERSION,
        "level": REFERENCE_NOISE_LEVEL,
        "level_f32_hex": f32_hex(REFERENCE_NOISE_LEVEL),
        "one_minus_level_f32_hex": f32_hex(1.0 - _f32(REFERENCE_NOISE_LEVEL)),
        "mix": "t * z + (1 - t) * noise, t an fp32 tensor (conditioning_noise_mix)",
        "draw_order": ["posterior[0..N-1]", "target_noise", "reference_noise[0..N-1]"],
        "rows": "patchify each noised reference at its own grid, concatenate in reference order",
        "fixed_rows": "reference prefix of the image rows is byte-identical after every step",
    }
