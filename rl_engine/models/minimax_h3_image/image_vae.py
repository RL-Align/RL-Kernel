# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3-Image VAE contract (RFC #738 ``image_vae_contract``, ``image_vae_tiling_contract``).

The released image VAE (``vae_diffusers/`` of ``MiniMaxAI/MiniMax-H3-Image``, Diffusers class
``AutoencoderKLMiniMaxH3Image``) is the MiniMax-H3 causal 3D CNN encoder run over one frame plus
a 24-layer 2D ViT decoder. It maps an RGB image to a ``(24, H/16, W/16)`` latent and back. This
module pins everything around that network that decides which latent and output bytes come out:

* **Identity.** The local ``vae_diffusers/`` folder must have the pinned ``config.json`` content,
  file sizes (and SHA-256 on request) and the pinned safetensors header table
  (:mod:`~rl_engine.models.minimax_h3_image.fingerprint`). The checkpoint is FP32 and the
  contract runs it in FP32; a VAE with any other parameter dtype is refused.
* **Pixel normalization.** ``uint8 -> float32 / 255 -> (x - mean) / std`` with ImageNet
  statistics, done by gathering from the MiniMax-H3 ``[3, 256]`` lookup table
  (:func:`rl_engine.models.minimax_h3.visual_vae.pixel_lut`): the image pipeline uses exactly
  the H3 arithmetic, and the table freezes its CUDA semantics on every device.
* **Output conversion.** ``x * std + mean``, clamp to ``[0, 1]``, ``* 255``, clamp, cast to
  ``uint8`` (truncation, not rounding). The output path stops at that canvas-size ``uint8``
  image; the post-decode resize to the requested size belongs to ``image_canvas_contract``
  (``rl_engine/models/minimax_h3_image/canvas.py``). Each stage has its own digest; none of
  them is part of latent equality.
* **Latent normalization.** ``(z - mean) / std`` and ``z * std + mean`` with the released
  24-channel statistics, on the VAE device, as separately rounded FP32 elementwise kernels.
* **Edit-reference recipe.** Crop the pixels at the top left to whole VAE cells, normalize,
  encode, draw ``posterior.sample(generator)`` from the request generator, round to FP16 and
  back, crop the latent to whole transformer patches, normalize. The generator state before
  and after every draw is recorded, so the draw order is sealed.
* **Tiling.** Encode and decode run in 256 px tiles with at least 64 px overlap, linearly
  blended. ``AutoencoderKLMiniMaxH3Image._split_tiles`` / ``_blend`` / ``_stitch_tiles`` are
  marked "Copied from" the MiniMax-H3 VAE at the pinned Diffusers commit, so the tile plan and
  its digest are the H3 ones (:func:`rl_engine.models.minimax_h3.visual_vae.plan_tiles`). The
  image contract adds fail-closed checks for geometry the Diffusers stitch handles wrongly.
  Tiled and untiled runs carry different mode identities.
* **Execution.** :class:`ImageVAEContract` runs one batch row at a time under the H3 pinned
  backend flags, with any ambient autocast disabled (the image pipeline encodes and decodes
  without autocast), and returns SHA-256 digests of every output.

The network itself is reused from Diffusers unmodified; this module has no Diffusers import and
works with any object that exposes ``encode`` / ``decode`` / ``config`` like
``AutoencoderKLMiniMaxH3Image``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl_engine.models.minimax_h3 import fingerprint as h3_fp
from rl_engine.models.minimax_h3 import visual_vae as vv
from rl_engine.models.minimax_h3_image import fingerprint as fp

CONTRACT_ID = "h3-image-vae-f16d24-v1"
TILING_CONTRACT_ID = "h3-image-vae-tiles-v1"

LATENT_CHANNELS = 24
SPATIAL_FACTOR = fp.VAE_SPATIAL_COMPRESSION  # 16
CANVAS_MULTIPLE = fp.CANVAS_MULTIPLE  # 32 = VAE factor 16 x transformer patch 2
PATCH_SIZE = CANVAS_MULTIPLE // SPATIAL_FACTOR  # 2

TILE_SIZE = 256
TILE_MIN_OVERLAP = 64
RELEASE_TILE_GEOMETRY = (TILE_SIZE, TILE_SIZE, TILE_MIN_OVERLAP, TILE_MIN_OVERLAP)

PIXEL_MEAN = fp.PIXEL_MEAN
PIXEL_STD = fp.PIXEL_STD

MODE_UNTILED = "untiled"

# The pixel table is reused from MiniMax-H3; it is only valid for the same statistics.
if (tuple(vv.PIXEL_MEAN), tuple(vv.PIXEL_STD)) != (tuple(PIXEL_MEAN), tuple(PIXEL_STD)):
    raise ImportError("MiniMax-H3 and MiniMax-H3-Image pixel statistics differ; pixel_lut reuse")

# plan_tiles reads only ``spatial_factor`` from the layout; the temporal fields are unused here.
_TILE_LAYOUT = vv.VisualVAELayout(latent_channels=LATENT_CHANNELS, spatial_factor=SPATIAL_FACTOR)

TileAxis = vv.TileAxis
TilePlan = vv.TilePlan
split_tiles = vv.split_tiles
pinned_runtime = vv.pinned_runtime
runtime_readback = vv.runtime_readback
RuntimePolicy = vv.RuntimePolicy
RELEASE_POLICY = vv.RELEASE_POLICY
VAEOutput = vv.VAEOutput
output_digests = vv.output_digests
pixel_lut_digest = vv.pixel_lut_digest
latent_stats_digest = vv.latent_stats_digest


# --- pinned identity ----------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """JSON round trip: tuples become lists, so configs built in Python compare to JSON ones."""

    return json.loads(json.dumps(value))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def release_config() -> dict[str, Any]:
    """``vae_diffusers/config.json`` of the pinned revision (a fresh copy on every call)."""

    return copy.deepcopy(dict(fp.pinned_checkpoint()["configs"][fp.DIFFUSERS_VAE_DIR]))


def release_latent_stats() -> tuple[torch.Tensor, torch.Tensor]:
    """Released ``latents_mean`` / ``latents_std`` as FP32 ``[24]`` CPU tensors.

    Built as the pipeline builds them (``torch.tensor(config.latents_mean)``); the fingerprint
    proves every entry is FP32-representable, so nothing rounds.
    """

    config = release_config()
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32)
    std = torch.tensor(config["latents_std"], dtype=torch.float32)
    return mean, std


def release_identity() -> dict[str, Any]:
    """Pinned checkpoint, reference code and VAE facts this contract is defined against.

    Raises if the fingerprint's VAE facts disagree with the constants of this module.
    """

    facts = fp.vae_facts()
    expected = {
        "latent_channels": LATENT_CHANNELS,
        "spatial_compression": SPATIAL_FACTOR,
        "tile_size": TILE_SIZE,
        "tile_overlap_min": TILE_MIN_OVERLAP,
        "encoder_tiling": True,
        "decoder_tiling": True,
        "pixel_mean": list(PIXEL_MEAN),
        "pixel_std": list(PIXEL_STD),
    }
    wrong = {key: facts[key] for key, value in expected.items() if facts[key] != value}
    if wrong:
        raise ValueError(f"fingerprint VAE facts differ from the image VAE contract: {wrong}")
    pinned = fp.pinned_checkpoint()
    mean, std = release_latent_stats()
    return {
        "repo": fp.REPO,
        "revision": fp.REVISION,
        "subfolder": fp.DIFFUSERS_VAE_DIR,
        "files": copy.deepcopy(pinned["dirs"][fp.DIFFUSERS_VAE_DIR]),
        "config_sha256": _sha256_json(release_config()),
        "tensor_table": copy.deepcopy(dict(pinned["tensor_tables"][fp.DIFFUSERS_VAE_DIR])),
        "reference": {
            "library": "diffusers",
            "commit": fp.DIFFUSERS_COMMIT,
            "class": fp.VAE_ARCHITECTURE,
        },
        "latent_stats_sha256": latent_stats_digest(mean, std),
        "fingerprint_latent_stats_sha256": facts["latent_stats_sha256"],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def check_checkpoint_dir(root: str | Path, *, verify_sha256: bool = False) -> dict[str, Any]:
    """Check a local ``vae_diffusers/`` folder against the pinned checkpoint.

    ``config.json`` must have the pinned content; every pinned file must have the pinned size
    (and SHA-256 when ``verify_sha256`` and the snapshot records one); the safetensors header
    must be the pinned tensor table (digest, tensor and parameter counts), all FP32. Returns a
    path-free summary.
    """

    root = Path(root)
    pinned = fp.pinned_checkpoint()
    config = json.loads((root / "config.json").read_text())
    if config != release_config():
        raise ValueError("config.json differs from the pinned vae_diffusers config")
    hashed = []
    for entry in pinned["dirs"][fp.DIFFUSERS_VAE_DIR]:
        path = root / Path(entry["path"]).name
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"{path.name}: size differs from the pinned checkpoint")
        if verify_sha256 and entry["sha256"] is not None:
            if _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{path.name}: sha256 differs from the pinned checkpoint")
            hashed.append(path.name)
    shards = sorted(root.glob("*.safetensors"))
    if [p.name for p in shards] != ["diffusion_pytorch_model.safetensors"]:
        raise ValueError(f"expected exactly one safetensors file, found {[p.name for p in shards]}")
    summary = check_tensor_table(h3_fp.read_safetensors_header(shards[0]))
    return {**summary, "sha256_verified": hashed}


def check_tensor_table(table: Mapping[str, tuple[str, tuple[int, ...]]]) -> dict[str, Any]:
    """Check a ``name -> (dtype, shape)`` header table against the pinned one (all FP32)."""

    expected = fp.pinned_checkpoint()["tensor_tables"][fp.DIFFUSERS_VAE_DIR]
    found = {
        "digest": h3_fp.tensor_table_digest(table),
        "tensors": len(table),
        "parameters": h3_fp.parameter_count(table),
    }
    if found != {key: expected[key] for key in found}:
        raise ValueError(f"safetensors header {found} differs from the pinned table {expected}")
    dtypes = sorted({dtype for dtype, _ in table.values()})
    if dtypes != ["F32"]:
        raise ValueError(f"the released image VAE is FP32; header has {dtypes}")
    return {**found, "dtypes": dtypes}


def load_release_vae(
    vae_cls: Any,
    root: str | Path,
    device: torch.device | str = "cpu",
    *,
    verify_sha256: bool = False,
) -> Any:
    """Build ``vae_cls`` (``AutoencoderKLMiniMaxH3Image``) from a checked ``vae_diffusers/``.

    Strict FP32 load from the single pinned safetensors file, without ``from_pretrained`` (so
    neither ``accelerate`` nor a ``torch_dtype`` conversion is involved).
    """

    from safetensors.torch import load_file

    root = Path(root)
    check_checkpoint_dir(root, verify_sha256=verify_sha256)
    vae = vae_cls.from_config(release_config())
    state = load_file(str(root / "diffusion_pytorch_model.safetensors"), device="cpu")
    vae.load_state_dict(state, strict=True)
    del state
    vae = vae.to(device).eval()
    if any(p.dtype != torch.float32 for p in vae.parameters()):
        raise TypeError("the released image VAE is FP32; a parameter loaded in another dtype")
    return vae


def parameter_dtype_digest(module: torch.nn.Module) -> str:
    """SHA-256 over the sorted ``name|dtype`` lines of every parameter and buffer."""

    lines = [f"{name}|{t.dtype}" for name, t in module.state_dict(keep_vars=True).items()]
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


# --- geometry -----------------------------------------------------------------------------


def check_canvas(height: int, width: int) -> None:
    """A generation canvas: positive, both sides multiples of 32 (VAE 16 x patch 2)."""

    if height <= 0 or width <= 0:
        raise ValueError("canvas height and width must be positive")
    if height % CANVAS_MULTIPLE or width % CANVAS_MULTIPLE:
        raise ValueError(
            f"canvas height and width must be multiples of {CANVAS_MULTIPLE}, got {height}x{width}"
        )


def check_cells(height: int, width: int, factor: int = SPATIAL_FACTOR) -> None:
    """A VAE input: positive, both sides whole VAE cells."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height % factor or width % factor:
        raise ValueError(f"height and width must be multiples of {factor}, got {height}x{width}")


def crop_to_cells(pixels: torch.Tensor, factor: int = SPATIAL_FACTOR) -> torch.Tensor:
    """Top-left crop of ``(..., H, W)`` to whole VAE cells (edit reference step 2)."""

    h, w = pixels.shape[-2] // factor * factor, pixels.shape[-1] // factor * factor
    if h == 0 or w == 0:
        raise ValueError(f"{tuple(pixels.shape[-2:])} is smaller than one {factor}px VAE cell")
    return pixels[..., :h, :w]


def crop_to_patches(latents: torch.Tensor, patch: int = PATCH_SIZE) -> torch.Tensor:
    """Top-left crop of ``(..., h, w)`` latents to whole transformer patches (step 7)."""

    h, w = latents.shape[-2] // patch * patch, latents.shape[-1] // patch * patch
    if h == 0 or w == 0:
        raise ValueError(f"{tuple(latents.shape[-2:])} latents hold no whole {patch}x{patch} patch")
    return latents[..., :h, :w]


# --- tiling (image_vae_tiling_contract) ---------------------------------------------------


def plan_tiles(
    height: int,
    width: int,
    *,
    tile_height: int = TILE_SIZE,
    tile_width: int = TILE_SIZE,
    min_overlap_height: int = TILE_MIN_OVERLAP,
    min_overlap_width: int = TILE_MIN_OVERLAP,
    spatial_factor: int = SPATIAL_FACTOR,
) -> TilePlan:
    """The tile plan of a ``height x width`` image (pixels), fail-closed.

    Same placement and digest as MiniMax-H3 (the image VAE's ``_split_tiles`` is a copy of the
    H3 one). Refused in addition to what :func:`vv.plan_tiles` refuses:

    * a non-positive minimum overlap: ``_stitch_tiles`` crops each non-last tile with
      ``tile[..., :-overlap]``, which for an overlap of 0 empties the tile, and the next blend
      then fails;
    * a plan whose last tile does not end on the image edge.
    """

    if min_overlap_height <= 0 or min_overlap_width <= 0:
        raise ValueError(
            "tile overlaps must be positive: Diffusers' _stitch_tiles crops a tile with "
            "[..., :-overlap], which empties the tile for an overlap of 0"
        )
    layout = (
        _TILE_LAYOUT
        if spatial_factor == SPATIAL_FACTOR
        else vv.VisualVAELayout(latent_channels=LATENT_CHANNELS, spatial_factor=spatial_factor)
    )
    plan = vv.plan_tiles(
        height,
        width,
        tile_height=tile_height,
        tile_width=tile_width,
        min_overlap_height=min_overlap_height,
        min_overlap_width=min_overlap_width,
        layout=layout,
    )
    if not plan.covers_exactly():
        raise ValueError(f"tile plan for {height}x{width} does not end on the image edge")
    return plan


def tiling_identity(
    plan: TilePlan | None, geometry: tuple[int, int, int, int] | None
) -> dict[str, Any]:
    """The runtime identity of the tiling mode. Tiled and untiled are never the same identity."""

    if plan is None or geometry is None:
        return {"contract_id": TILING_CONTRACT_ID, "mode": MODE_UNTILED, "tiles": 1}
    th, tw, oh, ow = geometry
    r = plan.spatial_factor
    return {
        "contract_id": TILING_CONTRACT_ID,
        "mode": f"tiled-{th}x{tw}-o{oh}x{ow}",
        "tiles": plan.tiles,
        "plan_sha256": plan.digest(),
        "rows": asdict(plan.rows),
        "cols": asdict(plan.cols),
        "latent_overlaps": [list(axis) for axis in plan.latent_overlaps()],
        "split": "fewest tiles with every overlap >= min; slack added round-robin in "
        f"{r}px steps (_split_tiles)",
        "blend": "linear, weights 1 - i/n and i/n in the tile dtype; first the tile above "
        "(dim -2), then the tile to the left (dim -1); encode blends latent pixels, decode "
        "blends output pixels",
        "boundary": "every non-last tile drops its trailing overlap before concatenation",
    }


# --- pixel normalization and output conversion --------------------------------------------


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """``uint8`` ``(B, 3, H, W)`` RGB -> FP32 ImageNet-normalized VAE input (table gather).

    Equal to the pipeline's ``(pixels.float().div(255.0) - mean) / std`` on CUDA bit for bit.
    """

    if pixels.dtype != torch.uint8:
        raise TypeError("pixels must be uint8 RGB")
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError("pixels must have shape (B, 3, H, W)")
    return vv.preprocess_pixels(pixels.unsqueeze(2)).squeeze(2)


def _pixel_stat(values: Sequence[float], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=like.device).view(1, -1, 1, 1)


def denormalize_pixels(decoded: torch.Tensor) -> torch.Tensor:
    """Decoder output -> FP32 RGB in ``[0, 1]``: ``(x.float() * std + mean).clamp(0, 1)``."""

    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError("decoded images must have shape (B, 3, H, W)")
    x = decoded.float()
    return (x * _pixel_stat(PIXEL_STD, x) + _pixel_stat(PIXEL_MEAN, x)).clamp(0, 1)


def quantize_uint8(unit: torch.Tensor) -> torch.Tensor:
    """FP32 ``[0, 1]`` -> ``uint8`` by truncation: ``(x * 255).clamp(0, 255).to(uint8)``.

    The released pipeline truncates; ``VaeImageProcessor`` alone would round (``I18``).
    """

    if unit.dtype != torch.float32:
        raise TypeError("quantize_uint8 takes the FP32 [0, 1] image")
    return (unit * 255).clamp(0, 255).to(torch.uint8)


def output_conversion_spec() -> dict[str, Any]:
    return {
        "denormalize": "x.float() * std + mean, then clamp(0, 1)",
        "pixel_mean": list(PIXEL_MEAN),
        "pixel_std": list(PIXEL_STD),
        "quantize": "(x * 255).clamp(0, 255).to(uint8), truncation",
        # TODO(image_canvas_contract): the resize to output_height x output_width happens after
        # this, in rl_engine/models/minimax_h3_image/canvas.py; it is not part of this contract.
        "resize": "not applied here (image_canvas_contract)",
    }


# --- latent normalization -----------------------------------------------------------------


def validate_latent_stats(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
    """Metadata checks only (no device synchronization)."""

    channels = mean.shape[0] if mean.ndim == 1 else -1
    if latents.ndim != 4 or latents.shape[1] != channels:
        raise ValueError(
            f"latents must have shape (B, {channels}, h, w), got {tuple(latents.shape)}"
        )
    if not latents.is_floating_point():
        raise TypeError("latents must be floating point")
    for name, stat in (("mean", mean), ("std", std)):
        if stat.shape != (channels,):
            raise ValueError(f"{name} must be a [{channels}] vector")
        if stat.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
        if stat.requires_grad:
            raise ValueError(f"{name} must be a fixed buffer (requires_grad=False)")


def _latent_stat(stat: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    return stat.to(latents.device).view(1, -1, 1, 1)


def normalize_latents(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """``(z - mean) / std`` in FP32: one subtraction kernel, then one true-division kernel."""

    validate_latent_stats(latents, mean, std)
    z = latents.float()
    return (z - _latent_stat(mean, z)) / _latent_stat(std, z)


def denormalize_latents(
    latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """``z * std + mean`` in FP32: one multiply kernel, then one add kernel (no FMA)."""

    validate_latent_stats(latents, mean, std)
    z = latents.float()
    return z * _latent_stat(std, z) + _latent_stat(mean, z)


def round_reference_latents(latents: torch.Tensor) -> torch.Tensor:
    """The release rounds the sampled reference latent to FP16 and back (``I11``)."""

    return latents.to(torch.float16).float()


def generator_state_digest(generator: torch.Generator) -> str:
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def reference_recipe() -> list[str]:
    """RFC #738 section 3 "Reference-image conditioning", steps 2-8, as this module runs them."""

    return [
        f"crop pixels at the top left to whole {SPATIAL_FACTOR}px VAE cells",
        "ImageNet-normalize by the pinned [3, 256] FP32 table",
        "encode with the FP32 VAE, autocast disabled",
        "posterior.sample(generator) with the request generator, one draw per reference",
        "round to float16, back to float32",
        f"crop latents at the top left to whole {PATCH_SIZE}x{PATCH_SIZE} transformer patches",
        "(z - latents_mean) / latents_std on the VAE device",
    ]


# --- contract driver ----------------------------------------------------------------------


@dataclass
class ImageDecodeOutput:
    """Decoded images at the two output boundaries, each with whole and per-row digests.

    ``unit`` is the FP32 ``[0, 1]`` image and ``pixels`` the truncated canvas-size ``uint8``
    image, where this contract ends (any resize is ``image_canvas_contract``'s).
    """

    unit: torch.Tensor
    pixels: torch.Tensor
    digests: dict[str, str]
    row_digests: dict[str, list[str]]
    tile_plan: TilePlan | None
    record: dict[str, Any] = field(default_factory=dict)


def _config_dict(vae: Any) -> dict[str, Any]:
    config = getattr(vae, "config", None)
    if config is None:
        raise TypeError("vae has no config")
    return dict(config)


class ImageVAEContract:
    """Drive a Diffusers-compatible MiniMax-H3-Image VAE under the pinned contract.

    ``vae`` must behave like ``AutoencoderKLMiniMaxH3Image``: ``encode(x, return_dict=False)[0]``
    returns a posterior with ``sample(generator=...)`` and ``mode()``; ``decode(z,
    return_dict=False)[0]`` returns ImageNet-normalized RGB; ``use_tiling`` and the
    ``tile_sample_*`` attributes hold the tile geometry.

    ``tiled`` declares the tiling mode the caller means to run (the release has tiling on).
    Before every call the VAE must still be in that mode, have slicing off and only FP32
    parameters. With ``require_release=True`` (the default) the config, statistics and tile
    geometry must also equal the pinned release; tests pass ``False`` to use small models.
    """

    def __init__(
        self,
        vae: Any,
        *,
        tiled: bool = True,
        policy: RuntimePolicy = RELEASE_POLICY,
        require_release: bool = True,
    ) -> None:
        self.vae = vae
        self.tiled = tiled
        self.policy = policy
        self.require_release = require_release
        config = _config_dict(vae)
        self.mean = torch.tensor(config["latents_mean"], dtype=torch.float32)
        self.std = torch.tensor(config["latents_std"], dtype=torch.float32)
        self.latent_channels = int(config["latent_channels"])
        self.spatial_factor = 1
        for factor in config["spatial_downsample_factors"]:
            self.spatial_factor *= int(factor)
        if self.mean.shape != (self.latent_channels,) or self.std.shape != self.mean.shape:
            raise ValueError("latents_mean / latents_std do not have latent_channels entries")
        self.check_state()

    # -- identity --
    def tile_geometry(self) -> tuple[int, int, int, int] | None:
        if not getattr(self.vae, "use_tiling", False):
            return None
        v = self.vae
        return (
            int(v.tile_sample_min_height),
            int(v.tile_sample_min_width),
            int(v.tile_sample_min_overlap_height),
            int(v.tile_sample_min_overlap_width),
        )

    def check_state(self) -> None:
        """Fail closed unless the VAE is in the declared, supported mode."""

        geometry = self.tile_geometry()
        if (geometry is not None) != self.tiled:
            state = "on" if geometry is not None else "off"
            raise ValueError(
                f"contract declared tiled={self.tiled} but the VAE has tiling {state}; tiled "
                "and untiled are different runtime identities"
            )
        if getattr(self.vae, "use_slicing", False):
            raise ValueError("disable VAE slicing; the contract slices batch rows itself")
        dtypes = sorted({str(p.dtype) for p in self.vae.parameters()})
        if dtypes != ["torch.float32"]:
            raise TypeError(f"the contract runs the FP32 checkpoint; VAE parameters are {dtypes}")
        if self.require_release:
            self.check_release()

    def check_release(self) -> None:
        if type(self.vae).__name__ != fp.VAE_ARCHITECTURE:
            raise ValueError(f"VAE class {type(self.vae).__name__} != {fp.VAE_ARCHITECTURE}")
        public = {k: v for k, v in _config_dict(self.vae).items() if not k.startswith("_")}
        pinned = {k: v for k, v in release_config().items() if not k.startswith("_")}
        if _canonical(public) != pinned:
            raise ValueError("VAE config differs from the pinned vae_diffusers config")
        mean, std = release_latent_stats()
        if not (torch.equal(self.mean, mean) and torch.equal(self.std, std)):
            raise ValueError("VAE latents_mean/latents_std differ from the pinned release")
        geometry = self.tile_geometry()
        if geometry is not None and geometry != RELEASE_TILE_GEOMETRY:
            raise ValueError(f"release tiling is {RELEASE_TILE_GEOMETRY}, VAE has {geometry}")

    def tile_plan(self, height: int, width: int) -> TilePlan | None:
        geometry = self.tile_geometry()
        if geometry is None:
            check_cells(height, width, self.spatial_factor)
            return None
        th, tw, oh, ow = geometry
        return plan_tiles(
            height,
            width,
            tile_height=th,
            tile_width=tw,
            min_overlap_height=oh,
            min_overlap_width=ow,
            spatial_factor=self.spatial_factor,
        )

    def _device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def _record(self, kind: str, runtime: Mapping[str, Any], plan: TilePlan | None) -> dict:
        return {
            "contract_id": CONTRACT_ID,
            "kind": kind,
            "tiling": tiling_identity(plan, self.tile_geometry()),
            "latent_channels": self.latent_channels,
            "spatial_factor": self.spatial_factor,
            "latent_stats_sha256": latent_stats_digest(self.mean, self.std),
            "pixel_lut_sha256": pixel_lut_digest(),
            "parameter_dtype_sha256": parameter_dtype_digest(self.vae),
            "release_checked": self.require_release,
            "autocast": "disabled",
            "policy": asdict(self.policy),
            "runtime": dict(runtime),
        }

    # -- encode --
    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        posterior: str = "sample",
        round_fp16: bool = True,
        crop: bool = True,
    ) -> VAEOutput:
        """Encode ``uint8`` ``(B, 3, H, W)`` images into normalized FP32 CPU latents.

        The default is the released edit-reference recipe (module docstring); ``generator`` is
        the request generator and is required for ``posterior="sample"``. Rows draw from it in
        order, as the release draws one sample per reference. ``posterior="mode"`` takes the
        posterior mean, and ``round_fp16=False`` / ``crop=False`` turn the respective steps off
        (``crop=False`` then requires whole VAE cells and patches).
        """

        if posterior not in ("sample", "mode"):
            raise ValueError("posterior must be 'sample' or 'mode'")
        if posterior == "sample" and generator is None:
            raise ValueError("posterior='sample' needs the request generator")
        if pixels.dtype != torch.uint8 or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("pixels must be uint8 with shape (B, 3, H, W)")
        self.check_state()
        batch, _, in_h, in_w = pixels.shape
        if crop:
            pixels = crop_to_cells(pixels, self.spatial_factor)
        height, width = pixels.shape[-2:]
        plan = self.tile_plan(height, width)
        latent_hw = (height // self.spatial_factor, width // self.spatial_factor)
        expected = (1, self.latent_channels, *latent_hw)
        if not crop and (latent_hw[0] % PATCH_SIZE or latent_hw[1] % PATCH_SIZE):
            raise ValueError(f"crop=False needs whole {PATCH_SIZE}x{PATCH_SIZE} latent patches")
        device = self._device()
        mean, std = self.mean.to(device), self.std.to(device)
        rows: list[torch.Tensor] = []
        draws: list[dict[str, str]] = []
        with pinned_runtime(self.policy) as runtime:
            for row in pixels.split(1):
                x = normalize_pixels(row.to(device))
                with torch.autocast(device_type=device.type, enabled=False):
                    dist = self.vae.encode(x, return_dict=False)[0]
                    if posterior == "sample":
                        assert generator is not None
                        before = generator_state_digest(generator)
                        z = dist.sample(generator=generator)
                        draws.append({"before": before, "after": generator_state_digest(generator)})
                    else:
                        z = dist.mode()
                if tuple(z.shape) != expected:
                    raise RuntimeError(f"latent shape {tuple(z.shape)} != expected {expected}")
                if round_fp16:
                    z = round_reference_latents(z)
                if crop:
                    z = crop_to_patches(z)
                rows.append(normalize_latents(z.float(), mean, std).cpu())
        latents = torch.cat(rows)
        record = self._record("encode", runtime, plan)
        record.update(
            posterior=posterior,
            round_fp16=round_fp16,
            crop=crop,
            batch=batch,
            geometry={
                "pixels_in": [in_h, in_w],
                "pixels_encoded": [height, width],
                "latent": list(latent_hw),
                "latent_out": list(latents.shape[-2:]),
            },
            generator=(
                None
                if generator is None or posterior == "mode"
                else {
                    "device": str(generator.device),
                    "initial_seed": int(generator.initial_seed()),
                    "draws": draws,
                }
            ),
            recipe=(
                reference_recipe()
                if (posterior, round_fp16, crop) == ("sample", True, True)
                else None
            ),
        )
        digest, row_digests = output_digests(rows)
        return VAEOutput(
            tensor=latents, digest=digest, row_digests=row_digests, tile_plan=plan, record=record
        )

    def encode_references(
        self, images: Sequence[torch.Tensor], generator: torch.Generator
    ) -> list[VAEOutput]:
        """The edit VAE-encode step: each reference at its own size, drawing in order."""

        outs = []
        for image in images:
            batch = image if image.ndim == 4 else image.unsqueeze(0)
            if batch.shape[0] != 1:
                raise ValueError("encode_references takes one image per entry")
            outs.append(self.encode(batch, generator=generator))
        return outs

    # -- decode --
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> ImageDecodeOutput:
        """Decode normalized FP32 ``(B, 24, h, w)`` latents into canvas-size ``uint8`` images.

        The release decode step up to its 8-bit truncation. The canvas ``(16 h, 16 w)`` must be
        a multiple of 32. Resizing to the requested output size is ``image_canvas_contract``.
        """

        self.check_state()
        validate_latent_stats(latents, self.mean, self.std)
        r = self.spatial_factor
        height, width = latents.shape[-2] * r, latents.shape[-1] * r
        check_canvas(height, width)
        plan = self.tile_plan(height, width)
        device = self._device()
        mean, std = self.mean.to(device), self.std.to(device)
        units: list[torch.Tensor] = []
        quantized: list[torch.Tensor] = []
        with pinned_runtime(self.policy) as runtime:
            for row in latents.split(1):
                z = denormalize_latents(row.to(device), mean, std)
                with torch.autocast(device_type=device.type, enabled=False):
                    image = self.vae.decode(z, return_dict=False)[0]
                if tuple(image.shape) != (1, 3, height, width):
                    raise RuntimeError(
                        f"decoded {tuple(image.shape)}, expected (1, 3, {height}, {width})"
                    )
                unit = denormalize_pixels(image)
                quantized.append(quantize_uint8(unit).cpu())
                units.append(unit.cpu())
        digests: dict[str, str] = {}
        row_digests: dict[str, list[str]] = {}
        digests["unit"], row_digests["unit"] = output_digests(units)
        digests["uint8"], row_digests["uint8"] = output_digests(quantized)
        record = self._record("decode", runtime, plan)
        record.update(
            batch=latents.shape[0],
            canvas=[height, width],
            output_conversion=output_conversion_spec(),
        )
        return ImageDecodeOutput(
            unit=torch.cat(units),
            pixels=torch.cat(quantized),
            digests=digests,
            row_digests=row_digests,
            tile_plan=plan,
            record=record,
        )


def contract_manifest(
    canvases: Sequence[tuple[int, int]] = ((1024, 1024), (2048, 2048), (1536, 2688)),
) -> dict[str, Any]:
    """Machine-readable summary of the static contract (identity, tile plans, recipes)."""

    mean, std = release_latent_stats()
    plans = {}
    for h, w in canvases:
        plan = plan_tiles(h, w)
        plans[f"{h}x{w}"] = tiling_identity(plan, RELEASE_TILE_GEOMETRY)
    return {
        "contract_id": CONTRACT_ID,
        "tiling_contract_id": TILING_CONTRACT_ID,
        "release": release_identity(),
        "latent": {
            "channels": LATENT_CHANNELS,
            "spatial_factor": SPATIAL_FACTOR,
            "canvas_multiple": CANVAS_MULTIPLE,
            "patch_size": PATCH_SIZE,
            "stats_sha256": latent_stats_digest(mean, std),
            "normalize": "(z - mean) / std",
            "denormalize": "z * std + mean",
        },
        "pixels": {
            "mean": list(PIXEL_MEAN),
            "std": list(PIXEL_STD),
            "lut_sha256": pixel_lut_digest(),
            "normalize": "uint8 -> float32 * fl32(1/255) -> (x - mean) / std (CUDA semantics)",
        },
        "output_conversion": output_conversion_spec(),
        "reference_recipe": reference_recipe(),
        "tiling": {
            "release_geometry": list(RELEASE_TILE_GEOMETRY),
            "untiled_mode": MODE_UNTILED,
            "plans": plans,
        },
    }
