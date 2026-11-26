# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Canvas and post-decode resize of MiniMax-H3-Image (RFC #738 ``image_canvas_contract``).

A request asks for ``height x width`` pixels. The released pipeline does not generate at
that size directly:

1. **Minimum area.** An image below ``min_generation_area`` pixels is generated larger at
   the same aspect ratio: both sides are scaled by ``sqrt(min_generation_area / (h * w))``
   (double precision, ``math.sqrt``). The released default is ``2048 * 2048``; ``1024``
   disables the upscale. Values outside ``[1024, 4096 * 4096]`` are rejected.
2. **Rounding.** Each side becomes ``max(32, round(side * scale / 32) * 32)``. ``round`` is
   Python's, i.e. *half to even*: a requested 80 px side becomes a 64 px canvas, not 96.
   The canvas can therefore be smaller than the request.
3. **Latent grid.** ``canvas / 16`` per side (VAE compression 16); with the transformer
   patch 2 every canvas side is a whole number of patches.
4. **Resize back.** After decoding, the image is quantised to 8 bits by truncation (the
   VAE row's job) and, if the canvas differs from the request, the 8-bit image is resized
   to ``height x width`` with PIL ``LANCZOS`` (``VaeImageProcessor`` default, no
   ``reducing_gap``). The resize runs on uint8 pixels, so its bytes depend on the Pillow
   build; the plan records the Pillow version as part of the comparator profile.

Upstream: ``get_image_canvas_size`` and ``MiniMaxH3ImagePrepareCanvasStep`` in
``modular_pipelines/minimax_h3_image/before_denoise.py``, and the resize branch of
``MiniMaxH3ImageDecodeStep`` in ``decoders.py``, Diffusers commit
875373ac7f5856071e345a3e394c74efccc6b9e6.

``resolve_canvas(..., profile=...)`` exposes the I5 / I6 probe arms (other rounding
multiples, no minimum area). Only the ``"released"`` profile is a strict canvas.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from rl_engine.models.minimax_h3_image import fingerprint as fp

SCHEMA_VERSION = "rlkernel.h3_image.canvas.v1"
DEFAULT_SIZE = (2048, 2048)
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048
MIN_GENERATION_AREA_BOUNDS = (1024, 4096 * 4096)
RESAMPLE = "lanczos"
ROUNDING = "python round (half to even) of side * scale / multiple, at least one multiple"
PROFILES = ("released", "round16", "round64", "no_min_area")


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class CanvasPlan:
    """How one requested size is generated and returned."""

    requested: tuple[int, int]  # (height, width) the caller asked for
    min_generation_area: int
    multiple: int
    scale: float
    canvas: tuple[int, int]  # (height, width) generated on
    latent: tuple[int, int]  # canvas / VAE compression
    profile: str

    @property
    def needs_resize(self) -> bool:
        return self.canvas != self.requested

    @property
    def num_target_rows(self) -> int:
        patch = fp.transformer_config()["patch_size"]
        return (self.latent[0] // patch[1]) * (self.latent[1] // patch[2])

    def resize_rule(self) -> str:
        if not self.needs_resize:
            return "none"
        return f"pil.{RESAMPLE}(uint8, reducing_gap=None)"

    def artifact(self) -> dict[str, Any]:
        """Sealed, machine-readable plan with its own digest."""

        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            **{k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()},
            "scale_hex": float(self.scale).hex(),
            "rounding": ROUNDING,
            "needs_resize": self.needs_resize,
            "resize": self.resize_rule(),
            "pillow": _pillow_version() if self.needs_resize else None,
        }
        body["sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return body

    def evidence_block(self) -> dict[str, Any]:
        """The ``inputs.canvas`` block of an ``h3_image_evidence_schema`` trajectory."""

        return {
            "requested": list(self.requested),
            "canvas": list(self.canvas),
            "latent": list(self.latent),
            "min_generation_area": self.min_generation_area,
            "resize": self.resize_rule(),
            "plan_sha256": self.artifact()["sha256"],
        }


def _pillow_version() -> str | None:
    try:
        import PIL

        return str(PIL.__version__)
    except ImportError:  # pragma: no cover - Pillow ships with torch vision stacks
        return None


def canvas_side(side: int, scale: float, multiple: int) -> int:
    """One canvas side, exactly as ``get_image_canvas_size`` computes it."""

    return max(multiple, round(side * scale / multiple) * multiple)


def resolve_canvas(
    height: int = DEFAULT_SIZE[0],
    width: int = DEFAULT_SIZE[1],
    min_generation_area: int = DEFAULT_MIN_GENERATION_AREA,
    *,
    profile: str = "released",
) -> CanvasPlan:
    """Resolve the canvas a ``height x width`` request is generated on.

    Unsupported inputs raise before anything is computed. ``profile`` other than
    ``"released"`` builds a labelled probe plan (I5: rounding multiple, I6: no minimum
    area); such a plan is never a strict canvas.
    """

    height = _positive_int("height", height)
    width = _positive_int("width", width)
    area = _positive_int("min_generation_area", min_generation_area)
    low, high = MIN_GENERATION_AREA_BOUNDS
    if not low <= area <= high:
        raise ValueError(f"min_generation_area must be within [{low}, {high}], got {area}")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    multiple = {"round16": 16, "round64": 64}.get(profile, fp.CANVAS_MULTIPLE)
    scale = 1.0 if profile == "no_min_area" else max(1.0, math.sqrt(area / (width * height)))
    canvas = (canvas_side(height, scale, multiple), canvas_side(width, scale, multiple))
    compression = fp.VAE_SPATIAL_COMPRESSION
    latent = (canvas[0] // compression, canvas[1] // compression)
    patch = fp.transformer_config()["patch_size"]
    if profile == "released" and (
        canvas[0] % fp.CANVAS_MULTIPLE
        or canvas[1] % fp.CANVAS_MULTIPLE
        or latent[0] % patch[1]
        or latent[1] % patch[2]
    ):  # pragma: no cover - guaranteed by multiple 32, kept as the fail-closed contract
        raise ValueError(f"canvas {canvas} is not on the {fp.CANVAS_MULTIPLE}-pixel grid")
    return CanvasPlan((height, width), area, multiple, scale, canvas, latent, profile)


def check_strict(plan: CanvasPlan) -> None:
    """Fail unless ``plan`` is the released canvas for its own request."""

    if plan.profile != "released":
        raise ValueError(f"canvas profile {plan.profile!r} is a probe, not the strict canvas")
    expected = resolve_canvas(*plan.requested, plan.min_generation_area)
    if plan != expected:
        raise ValueError("canvas plan does not match the released resolution of its request")


def resize_output(images: Any, plan: CanvasPlan) -> Any:
    """Resize decoded uint8 images ``[B, 3, canvas_h, canvas_w]`` to the requested size.

    Returns the input unchanged when the canvas equals the request. Otherwise each image
    goes through PIL ``LANCZOS`` on its uint8 RGB pixels, as the released decode step does,
    and the result is uint8 ``[B, 3, height, width]``.
    """

    import numpy as np
    import torch
    from PIL import Image

    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected uint8 [B, 3, H, W], got {images.dtype} {list(images.shape)}")
    if tuple(images.shape[-2:]) != plan.canvas:
        raise ValueError(
            f"images are {list(images.shape[-2:])}, plan canvas is {list(plan.canvas)}"
        )
    if not plan.needs_resize:
        return images
    height, width = plan.requested
    out = []
    for image in images.cpu():
        pil = Image.fromarray(image.permute(1, 2, 0).contiguous().numpy())
        resized = pil.resize((width, height), resample=Image.Resampling.LANCZOS, reducing_gap=None)
        out.append(torch.from_numpy(np.array(resized)).permute(2, 0, 1))
    return torch.stack(out).contiguous()
