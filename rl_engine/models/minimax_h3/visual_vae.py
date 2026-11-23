# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 VisualVAE contract (RFC #420 Phase B ``visual_vae_contract``).

The released VisualVAE (``vae/`` of ``MiniMaxAI/MiniMax-H3``, Diffusers class
``AutoencoderKLMiniMaxH3``) turns RGB video into the ``f16t4d24`` latent the Omni-Transformer
consumes: 24 channels, spatial factor 16, temporal factor 4. This module pins everything
around the network that decides which latent bytes come out:

* **Layout.** ``17 * n + 5`` pixel frames map to ``5 * n + 2`` latent frames (17-frame encode
  clips, 3 trailing latent frames dropped); a single frame bypasses temporal chunking.
  Height and width must be multiples of 16. :class:`VisualVAELayout` derives every count
  with integers only and mirrors the decoder's chunk/cross-fade arithmetic.
* **Tiling.** The release encodes and decodes in 256 px tiles with at least 64 px overlap,
  linearly blended. :func:`plan_tiles` reproduces Diffusers' tile placement exactly and gives
  the plan a digest, so a run can prove which tiles it used.
* **Pinned preprocessing.** Diffusers computes ``uint8.to(float32).div(255.0)`` on the VAE
  device. PyTorch performs that scalar division as a true division on CPU but as a multiply
  by the FP32 reciprocal on CUDA, so the two disagree for 126 of the 256 input values.
  :func:`pixel_lut` freezes the CUDA (release) semantics in a ``[3, 256]`` FP32 table, and
  :func:`preprocess_pixels` gathers from it: a copy with no arithmetic, identical bits on
  every device.
* **Latent normalization.** ``(z - mean) / std`` to normalize and ``z * std + mean`` to
  denormalize, each operation a separately rounded FP32 elementwise kernel, with the
  released per-channel statistics. Condition latents are sampled from the posterior under
  a fresh CPU generator (seed 42) and rounded to FP16 before normalization, as released.
* **Execution.** :class:`VisualVAEContract` drives a Diffusers-compatible VAE one batch row
  at a time (batch is a replication axis, so a row's latent never depends on its batch
  neighbours), under pinned cuDNN/TF32 flags, and returns SHA-256 digests of every latent.

The network itself is reused from Diffusers (see ``docs/operators/h3-visual-vae-contract.md``
for the batch-invariance measurements behind that decision); this module has no Diffusers
import and works with any object exposing ``encode``/``decode``/``config`` like
``AutoencoderKLMiniMaxH3``.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import struct
import threading
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from rl_engine.models.minimax_h3.trajectory import tensor_digest

CONTRACT_ID = "h3-visual-vae-f16t4d24-v1"
FINGERPRINT_PATH = Path(__file__).with_name("visual_vae_fingerprint.json")

LATENT_CHANNELS = 24
SPATIAL_FACTOR = 16
TEMPORAL_FACTOR = 4
CLIP_LENGTH = 17
TOKEN_DROP = 3

TILE_SAMPLE_SIZE = 256
PREPROCESS_FRAME_CHUNK = 8
TILE_SAMPLE_MIN_OVERLAP = 64

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
CONDITION_ENCODE_SEED = 42


# --- pinned release fingerprint -----------------------------------------------------------


@lru_cache(maxsize=1)
def _pinned_fingerprint() -> Mapping[str, Any]:
    data = json.loads(FINGERPRINT_PATH.read_text())
    if data.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"{FINGERPRINT_PATH.name} is not for {CONTRACT_ID}")
    return data


def release_fingerprint() -> Mapping[str, Any]:
    """The pinned checkpoint/config/reference identity (``visual_vae_fingerprint.json``).

    A fresh copy on every call: a caller that edits the result cannot change the pinned
    identity that later checks compare against.
    """

    return copy.deepcopy(_pinned_fingerprint())


def release_config() -> Mapping[str, Any]:
    """``vae/config.json`` of the pinned checkpoint revision."""

    return release_fingerprint()["config"]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def release_latent_stats() -> tuple[torch.Tensor, torch.Tensor]:
    """Released per-channel ``latents_mean`` / ``latents_std`` as FP32 ``[24]`` CPU tensors.

    Built the way Diffusers builds them (``torch.tensor(config.latents_mean)``): each JSON
    double rounds to the nearest FP32.
    """

    config = release_config()
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32)
    std = torch.tensor(config["latents_std"], dtype=torch.float32)
    return mean, std


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_vae(
    vae_cls: Any,
    root: str | Path,
    device: torch.device | str = "cpu",
    *,
    verify_sha256: bool = False,
) -> Any:
    """Build ``vae_cls`` (``AutoencoderKLMiniMaxH3``) from the released ``vae/`` folder.

    The folder must match the pinned checkpoint: same ``config.json`` content and shard
    sizes (and SHA-256 when ``verify_sha256``). Weights are loaded strictly in FP32, without
    ``from_pretrained`` (which needs ``accelerate`` for the FP32-pinned modules).
    """

    from safetensors.torch import load_file

    root = Path(root)
    pinned = release_fingerprint()
    config = json.loads((root / "config.json").read_text())
    if config != dict(pinned["config"]):
        raise ValueError(f"{root}/config.json differs from the pinned release config")
    for entry in pinned["checkpoint"]["files"]:
        path = root / Path(entry["path"]).name
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"{path.name}: size differs from the pinned checkpoint")
        if verify_sha256 and _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path.name}: sha256 differs from the pinned checkpoint")
    vae = vae_cls.from_config(config)
    expected = set(vae.state_dict())
    loaded: set[str] = set()
    # Only the pinned shards: another *.safetensors in the folder must not replace a
    # verified tensor. One shard at a time, so the host holds the model plus one shard,
    # not a second full copy of the weights. Missing, unexpected and repeated keys are
    # errors, as with one strict load (shape mismatches always raise).
    for entry in pinned["checkpoint"]["files"]:
        if not entry["path"].endswith(".safetensors"):
            continue
        shard = load_file(str(root / Path(entry["path"]).name), device="cpu")
        repeated = loaded & shard.keys()
        if repeated:
            raise ValueError(f"keys in more than one shard: {sorted(repeated)[:5]}")
        result = vae.load_state_dict(shard, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"unexpected keys: {sorted(result.unexpected_keys)[:5]}")
        loaded |= shard.keys()
        del shard
    if expected - loaded:
        raise ValueError(f"missing keys: {sorted(expected - loaded)[:5]}")
    vae = vae.to(device).eval()
    if any(p.dtype != torch.float32 for p in vae.parameters()):
        raise TypeError("the released VisualVAE is FP32; a parameter loaded in another dtype")
    return vae


def latent_stats_digest(mean: torch.Tensor, std: torch.Tensor) -> str:
    return hashlib.sha256(
        (tensor_digest(mean.float()) + tensor_digest(std.float())).encode()
    ).hexdigest()


# --- layout -------------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualVAELayout:
    """The ``f16t4d24`` latent geometry and its temporal chunking arithmetic."""

    latent_channels: int = LATENT_CHANNELS
    spatial_factor: int = SPATIAL_FACTOR
    temporal_factor: int = TEMPORAL_FACTOR
    clip_length: int = CLIP_LENGTH
    token_drop: int = TOKEN_DROP

    def __post_init__(self) -> None:
        for name in ("latent_channels", "spatial_factor", "temporal_factor", "clip_length"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.token_drop < 0 or self.token_drop >= self.tokens_chunk_size:
            raise ValueError("token_drop must be in [0, latent frames per clip)")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> VisualVAELayout:
        """Derive the layout from a Diffusers ``AutoencoderKLMiniMaxH3`` config."""

        return cls(
            latent_channels=int(config["latent_channels"]),
            spatial_factor=math.prod(int(f) for f in config["spatial_downsample_factors"]),
            temporal_factor=math.prod(int(f) for f in config["temporal_downsample_factors"]),
            clip_length=int(config["clip_length"]),
            token_drop=int(config["token_drop"]),
        )

    # Derived chunk geometry; names and formulas follow AutoencoderKLMiniMaxH3.__init__.
    @property
    def frame_pre_padding(self) -> int:
        return (-self.clip_length) % self.temporal_factor

    @property
    def tokens_chunk_size(self) -> int:
        return -(-self.clip_length // self.temporal_factor)

    @property
    def token_overlap(self) -> int:
        return (-self.token_drop) % self.tokens_chunk_size

    @property
    def frame_overlap(self) -> int:
        return max(self.token_overlap * self.temporal_factor - self.frame_pre_padding, 0)

    @property
    def name(self) -> str:
        return f"f{self.spatial_factor}t{self.temporal_factor}d{self.latent_channels}"

    def latent_frames(self, num_frames: int) -> int:
        """Latent frames produced by encoding ``num_frames`` pixel frames."""

        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if num_frames == 1:
            return 1
        clips = -(-num_frames // self.clip_length)
        return clips * self.tokens_chunk_size - self.token_drop

    def is_release_frame_count(self, num_frames: int) -> bool:
        """``True`` for 1 or ``clip_length * n + (tokens_chunk_size * ... )`` counts.

        These are the counts whose encode needs no trailing pad frames, i.e. the counts the
        released pipeline accepts (``17 * n + 5`` for the release layout).
        """

        if num_frames == 1:
            return True
        if num_frames < 2:
            return False
        latents = self.latent_frames(num_frames)
        return self.pixel_frames(latents) == num_frames if self.decodable(latents) else False

    def decode_chunks(self, num_latent_frames: int) -> int:
        tokens = num_latent_frames + self.token_drop
        pad = (-tokens) % self.tokens_chunk_size
        return (tokens + pad) // self.tokens_chunk_size - int(self.token_drop > 0)

    def decodable(self, num_latent_frames: int) -> bool:
        return num_latent_frames >= 1 and self.decode_chunks(num_latent_frames) > 0

    def pixel_frames(self, num_latent_frames: int) -> int:
        """Pixel frames produced by decoding ``num_latent_frames`` (mirrors ``_decode``)."""

        if not self.decodable(num_latent_frames):
            raise ValueError(
                f"{num_latent_frames} latent frames give no full decode chunk for {self.name}"
            )
        tf, tcs = self.temporal_factor, self.tokens_chunk_size
        chunk_frames = tcs * tf
        tokens = num_latent_frames + self.token_drop
        pad = (-tokens) % tcs
        padded = num_latent_frames + pad
        total = 0
        overlap: int | None = None
        for i in range(self.decode_chunks(num_latent_frames)):
            start = i * tcs
            clip_frames = (min(padded, start + tcs + self.token_overlap) - start) * tf
            for j in range(int(self.token_drop > 0) + 1):
                begin = j * chunk_frames
                frames = max(min(clip_frames, begin + chunk_frames) - begin, 0)
                frames = max(frames - self.frame_pre_padding, 0)
                if j == 0:
                    total += frames  # a cross-fade keeps the later chunk's length
                else:
                    overlap = frames
        if overlap is not None:
            total += overlap
        if pad:
            tail = self.clip_length % tf
            total -= sum(
                tail if tail and (num_latent_frames + k) % tcs == 0 else tf for k in range(pad)
            )
        return total

    def latent_shape(
        self, batch: int, num_frames: int, height: int, width: int
    ) -> tuple[int, int, int, int, int]:
        """Encoded latent shape for a ``(batch, 3, num_frames, height, width)`` video."""

        check_spatial(height, width, self)
        return (
            batch,
            self.latent_channels,
            self.latent_frames(num_frames),
            height // self.spatial_factor,
            width // self.spatial_factor,
        )


RELEASE_LAYOUT = VisualVAELayout()


def check_spatial(height: int, width: int, layout: VisualVAELayout = RELEASE_LAYOUT) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height % layout.spatial_factor or width % layout.spatial_factor:
        raise ValueError(
            f"height and width must be multiples of {layout.spatial_factor}, got {height}x{width}"
        )


# --- tiling -------------------------------------------------------------------------------


@dataclass(frozen=True)
class TileAxis:
    """Tile placement along one axis, in pixels (``overlaps[i]`` joins tile ``i`` and ``i+1``)."""

    length: int
    starts: tuple[int, ...]
    sizes: tuple[int, ...]
    overlaps: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.starts)


def split_tiles(length: int, tile_size: int, min_overlap: int, ratio: int) -> TileAxis:
    """Tile placement along one axis (``AutoencoderKLMiniMaxH3._split_tiles``).

    The fewest ``tile_size`` tiles whose overlaps are all at least ``min_overlap``; the slack
    is spread round-robin over the overlaps in whole ``ratio`` steps so that every boundary
    stays on the latent grid.
    """

    if length <= 0 or tile_size <= 0 or min_overlap < 0 or ratio <= 0:
        raise ValueError("length, tile_size and ratio must be positive; min_overlap >= 0")
    if tile_size >= length:
        return TileAxis(length, (0,), (length,), ())
    if min_overlap >= tile_size:
        raise ValueError("min_overlap must be smaller than tile_size")
    num_tiles = -(-length // tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
        num_tiles += 1
    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - length
    if remaining % ratio:
        # Diffusers drops the remainder, and the last tile then overruns the frame edge.
        raise ValueError(
            f"length, tile_size and min_overlap must sit on the {ratio}-pixel grid "
            f"(got {length}, {tile_size}, {min_overlap})"
        )
    for i in range(remaining // ratio):
        overlaps[i % (num_tiles - 1)] += ratio
    starts = [0]
    for i in range(num_tiles - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return TileAxis(length, tuple(starts), (tile_size,) * num_tiles, tuple(overlaps))


@dataclass(frozen=True)
class TilePlan:
    """Spatial tiling of one frame size, for encode (pixel tiles) and decode (latent tiles)."""

    height: int
    width: int
    rows: TileAxis
    cols: TileAxis
    tile_height: int
    tile_width: int
    min_overlap_height: int
    min_overlap_width: int
    spatial_factor: int

    @property
    def tiles(self) -> int:
        return self.rows.count * self.cols.count

    def latent_overlaps(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Encode blends tile latents over ``overlap // spatial_factor`` latent pixels."""

        r = self.spatial_factor
        return tuple(o // r for o in self.rows.overlaps), tuple(o // r for o in self.cols.overlaps)

    def covers_exactly(self) -> bool:
        """Every pixel is in some tile, and the last tile ends on the frame edge."""

        return all(
            axis.starts[-1] + axis.sizes[-1] == axis.length
            or (axis.count == 1 and axis.sizes[0] == axis.length)
            for axis in (self.rows, self.cols)
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def plan_tiles(
    height: int,
    width: int,
    *,
    tile_height: int = TILE_SAMPLE_SIZE,
    tile_width: int = TILE_SAMPLE_SIZE,
    min_overlap_height: int = TILE_SAMPLE_MIN_OVERLAP,
    min_overlap_width: int = TILE_SAMPLE_MIN_OVERLAP,
    layout: VisualVAELayout = RELEASE_LAYOUT,
) -> TilePlan:
    """The release tile plan for a ``height x width`` frame (pixels, multiples of 16)."""

    check_spatial(height, width, layout)
    for value in (tile_height, tile_width, min_overlap_height, min_overlap_width):
        if value % layout.spatial_factor:
            raise ValueError(f"tile geometry must be multiples of {layout.spatial_factor}")
    r = layout.spatial_factor
    return TilePlan(
        height=height,
        width=width,
        rows=split_tiles(height, tile_height, min_overlap_height, r),
        cols=split_tiles(width, tile_width, min_overlap_width, r),
        tile_height=tile_height,
        tile_width=tile_width,
        min_overlap_height=min_overlap_height,
        min_overlap_width=min_overlap_width,
        spatial_factor=r,
    )


def blend_weights(
    extent: int, dtype: torch.dtype, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear cross-fade weights of ``AutoencoderKLMiniMaxH3._blend``, in ``dtype``.

    ``positions / extent`` divides a tensor by a Python scalar, which CUDA evaluates as a
    multiply by the reciprocal; callers comparing devices must account for that.
    """

    positions = torch.arange(extent, device=device, dtype=dtype)
    return 1 - positions / extent, positions / extent


# --- pinned preprocessing -----------------------------------------------------------------


@lru_cache(maxsize=1)
def _pixel_lut_cpu() -> torch.Tensor:
    values = torch.arange(256, dtype=torch.float32)
    # CUDA's ``tensor.div(255.0)`` multiplies by the FP32 reciprocal of the scalar.
    inv255 = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(255.0, dtype=torch.float32)
    unit = values * inv255
    mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(3, 1)
    std = torch.tensor(PIXEL_STD, dtype=torch.float32).view(3, 1)
    lut = (unit.view(1, 256) - mean) / std
    lut.requires_grad_(False)
    return lut.contiguous()


def pixel_lut(device: torch.device | str | None = None) -> torch.Tensor:
    """``[3, 256]`` FP32 table: ``uint8`` value -> ImageNet-normalized VAE input, per channel.

    Entry ``[c, v]`` equals what the released pipeline computes on CUDA for value ``v`` in
    channel ``c``: ``((v * fl32(1/255)) - mean[c]) / std[c]``, each step rounded to FP32.
    """

    lut = _pixel_lut_cpu()
    return lut if device is None else lut.to(device)


def pixel_lut_digest() -> str:
    return tensor_digest(_pixel_lut_cpu())


def preprocess_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """``uint8`` ``(B, 3, F, H, W)`` RGB -> FP32 VAE input, by table lookup (no arithmetic)."""

    if pixels.dtype != torch.uint8:
        raise TypeError("pixels must be uint8 RGB")
    if pixels.ndim != 5 or pixels.shape[1] != 3:
        raise ValueError("pixels must have shape (B, 3, F, H, W)")
    lut = pixel_lut(pixels.device)
    channel = torch.arange(3, device=pixels.device).view(1, 3, 1, 1, 1)
    out = torch.empty(pixels.shape, dtype=torch.float32, device=pixels.device)
    # The int64 index is 8x the uint8 input; build it a few frames at a time so it never
    # spans a whole clip.
    for start in range(0, pixels.shape[2], PREPROCESS_FRAME_CHUNK):
        frames = slice(start, start + PREPROCESS_FRAME_CHUNK)
        out[:, :, frames] = lut[channel, pixels[:, :, frames].long()]
    return out


def postprocess_pixels(video: torch.Tensor) -> torch.Tensor:
    """Decoder output (ImageNet-normalized) -> FP32 RGB in ``[0, 1]`` (``decoders.py``)."""

    mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32, device=video.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, dtype=torch.float32, device=video.device).view(1, -1, 1, 1, 1)
    return (video.float() * std + mean).clamp(0, 1)


# --- latent normalization -----------------------------------------------------------------


def validate_latent_stats(
    latents: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    layout: VisualVAELayout = RELEASE_LAYOUT,
) -> None:
    """Metadata checks only (no device synchronization)."""

    if latents.ndim != 5 or latents.shape[1] != layout.latent_channels:
        raise ValueError(f"latents must have shape (B, {layout.latent_channels}, T, H, W)")
    if not latents.is_floating_point():
        raise TypeError("latents must be floating point")
    for name, stat in (("mean", mean), ("std", std)):
        if stat.shape != (layout.latent_channels,):
            raise ValueError(f"{name} must be a [{layout.latent_channels}] vector")
        if stat.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
        if stat.requires_grad:
            raise ValueError(f"{name} must be a fixed buffer (requires_grad=False)")


def _stat_view(stat: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    return stat.to(latents.device).view(1, -1, 1, 1, 1)


def normalize_latents(
    latents: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    layout: VisualVAELayout = RELEASE_LAYOUT,
) -> torch.Tensor:
    """``(z - mean) / std`` in FP32: one subtraction kernel, then one true-division kernel."""

    validate_latent_stats(latents, mean, std, layout)
    return (latents.float() - _stat_view(mean, latents)) / _stat_view(std, latents)


def denormalize_latents(
    latents: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    layout: VisualVAELayout = RELEASE_LAYOUT,
) -> torch.Tensor:
    """``z * std + mean`` in FP32: one multiply kernel, then one add kernel (no FMA)."""

    validate_latent_stats(latents, mean, std, layout)
    return latents.float() * _stat_view(std, latents) + _stat_view(mean, latents)


def round_condition_latents(latents: torch.Tensor) -> torch.Tensor:
    """The release rounds sampled condition latents to FP16 before normalizing them."""

    return latents.to(torch.float16).float()


# --- runtime policy -----------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimePolicy:
    """Backend flags pinned around every VAE call. The defaults keep PyTorch's TF32 defaults,
    so they match an unmodified Diffusers run, and pin cuDNN algorithm selection."""

    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    cudnn_allow_tf32: bool = True
    matmul_allow_tf32: bool = False


RELEASE_POLICY = RuntimePolicy()


def runtime_readback(device: torch.device | str | None = None) -> dict[str, Any]:
    """Library versions, the four backend flags, and the GPU of ``device``.

    ``device=None`` names the current CUDA device (when CUDA is available); a CPU ``device``
    names no GPU.
    """

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    target = None if device is None else torch.device(device)
    if target is None and not torch.cuda.is_available():
        return info
    if target is not None and target.type != "cuda":
        return info
    index = torch.cuda.current_device() if target is None or target.index is None else target.index
    info["device_name"] = torch.cuda.get_device_name(index)
    info["compute_capability"] = list(torch.cuda.get_device_capability(index))
    return info


# The backend flags are process-global, so one pinned scope runs at a time (re-entrant for
# nested scopes in one thread): concurrent scopes would save and restore each other's flags.
_RUNTIME_LOCK = threading.RLock()


@contextlib.contextmanager
def pinned_runtime(
    policy: RuntimePolicy = RELEASE_POLICY, *, device: torch.device | str | None = None
) -> Iterator[dict[str, Any]]:
    """Apply ``policy`` for the duration of the block and restore the previous flags.

    Yields ``runtime_readback(device)``. Scopes in different threads are serialized; a
    thread waits until the scope in another thread has restored its flags.
    """

    cudnn = torch.backends.cudnn
    matmul = torch.backends.cuda.matmul
    with _RUNTIME_LOCK:
        saved = (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32)
        try:
            cudnn.deterministic = policy.cudnn_deterministic
            cudnn.benchmark = policy.cudnn_benchmark
            cudnn.allow_tf32 = policy.cudnn_allow_tf32
            matmul.allow_tf32 = policy.matmul_allow_tf32
            yield runtime_readback(device)
        finally:
            cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32 = saved


# --- contract driver ----------------------------------------------------------------------


@dataclass
class VAEOutput:
    """A contract output tensor plus everything needed to reproduce and compare it."""

    tensor: torch.Tensor
    digest: str
    row_digests: list[str]
    tile_plan: TilePlan | None
    record: dict[str, Any] = field(default_factory=dict)


def output_digests(rows: Sequence[torch.Tensor]) -> tuple[str, list[str]]:
    """``tensor_digest`` of ``torch.cat(rows)`` and of every row, without copying the bytes.

    The values equal ``trajectory.tensor_digest``, i.e. SHA-256 over ``"{dtype}|{shape}|"``
    and the raw bytes. That helper copies each tensor twice before hashing, which costs
    hundreds of milliseconds on a decoded video. Here the rows are hashed in place, and with
    a single row the whole-tensor digest is the row digest (same dtype, shape and bytes), so
    every byte is hashed once.
    """

    if not rows:
        raise ValueError("no rows to digest")
    views = []
    row_digests = []
    for row in rows:
        cpu = row.detach().contiguous().cpu()
        view = memoryview(cpu.reshape(-1).view(torch.uint8).numpy()) if cpu.numel() else None
        digest = hashlib.sha256(f"{cpu.dtype}|{tuple(cpu.shape)}|".encode())
        if view is not None:
            digest.update(view)
        views.append(view)
        row_digests.append(digest.hexdigest())
    if len(rows) == 1:
        return row_digests[0], row_digests
    first = rows[0]
    shape = (sum(r.shape[0] for r in rows), *first.shape[1:])
    whole = hashlib.sha256(f"{first.dtype}|{shape}|".encode())
    for view in views:
        if view is not None:
            whole.update(view)
    return whole.hexdigest(), row_digests


def _config_value(vae: Any, key: str) -> Any:
    config = getattr(vae, "config", None)
    if config is None:
        raise TypeError("vae has no config")
    return config[key] if isinstance(config, Mapping) else getattr(config, key)


class VisualVAEContract:
    """Drive a Diffusers-compatible MiniMax-H3 VisualVAE under the pinned contract.

    ``vae`` must behave like ``AutoencoderKLMiniMaxH3``: ``encode(x, return_dict=False)[0]``
    returns a posterior with ``sample(generator=...)`` and ``mode()``; ``decode(z,
    return_dict=False)[0]`` returns ImageNet-normalized RGB; ``config`` carries the layout,
    ``latents_mean`` and ``latents_std``; ``use_tiling`` and the ``tile_sample_*`` attributes
    hold the tile geometry.

    With ``require_release=True`` (the default) the VAE's layout, statistics and tile
    geometry must equal the pinned release, checked again before every ``encode`` and
    ``decode`` (the VAE can be changed after construction, e.g. ``disable_tiling()``); tests
    pass ``False`` to use small models.
    """

    def __init__(
        self,
        vae: Any,
        *,
        policy: RuntimePolicy = RELEASE_POLICY,
        require_release: bool = True,
    ) -> None:
        self.vae = vae
        self.policy = policy
        config = vae.config
        self.layout = VisualVAELayout.from_config(
            config if isinstance(config, Mapping) else dict(config)
        )
        self.mean = torch.tensor(_config_value(vae, "latents_mean"), dtype=torch.float32)
        self.std = torch.tensor(_config_value(vae, "latents_std"), dtype=torch.float32)
        self.require_release = require_release
        if require_release:
            self.check_release()

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

    def check_release(self) -> None:
        if self.layout != RELEASE_LAYOUT:
            raise ValueError(f"VAE layout {self.layout} is not the release {RELEASE_LAYOUT}")
        mean, std = release_latent_stats()
        if not (torch.equal(self.mean, mean) and torch.equal(self.std, std)):
            raise ValueError("VAE latents_mean/latents_std differ from the pinned release")
        expected = (
            TILE_SAMPLE_SIZE,
            TILE_SAMPLE_SIZE,
            TILE_SAMPLE_MIN_OVERLAP,
            TILE_SAMPLE_MIN_OVERLAP,
        )
        if self.tile_geometry() != expected:
            raise ValueError(f"release tiling is {expected}, VAE has {self.tile_geometry()}")
        if getattr(self.vae, "use_slicing", False):
            raise ValueError("disable VAE slicing; the contract slices batch rows itself")

    def tile_plan(self, height: int, width: int) -> TilePlan | None:
        geometry = self.tile_geometry()
        if geometry is None:
            check_spatial(height, width, self.layout)
            return None
        th, tw, oh, ow = geometry
        return plan_tiles(
            height,
            width,
            tile_height=th,
            tile_width=tw,
            min_overlap_height=oh,
            min_overlap_width=ow,
            layout=self.layout,
        )

    def _device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def _record(self, kind: str, runtime: Mapping[str, Any], plan: TilePlan | None) -> dict:
        return {
            "contract_id": CONTRACT_ID,
            "kind": kind,
            "layout": asdict(self.layout),
            "layout_name": self.layout.name,
            "tile_plan_sha256": plan.digest() if plan is not None else None,
            "tiles": plan.tiles if plan is not None else 1,
            "latent_stats_sha256": latent_stats_digest(self.mean, self.std),
            "pixel_lut_sha256": pixel_lut_digest(),
            "policy": asdict(self.policy),
            "runtime": dict(runtime),
        }

    # -- encode --
    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        posterior: str = "sample",
        seed: int = CONDITION_ENCODE_SEED,
        round_fp16: bool = True,
    ) -> VAEOutput:
        """Encode ``uint8`` ``(B, 3, F, H, W)`` video into normalized FP32 CPU latents.

        The default is the release condition recipe: posterior sampled under a fresh CPU
        generator seeded ``seed`` for every row, rounded to FP16, then normalized.
        ``posterior="mode"`` takes the posterior mean instead.
        """

        if posterior not in ("sample", "mode"):
            raise ValueError("posterior must be 'sample' or 'mode'")
        if pixels.ndim != 5 or pixels.shape[1] != 3:
            raise ValueError("pixels must have shape (B, 3, F, H, W)")
        batch, _, frames, height, width = pixels.shape
        if self.require_release:
            self.check_release()
        plan = self.tile_plan(height, width)
        expected = self.layout.latent_shape(1, frames, height, width)
        device = self._device()
        rows = []
        with pinned_runtime(self.policy, device=device) as runtime:
            for row in pixels.split(1):
                x = preprocess_pixels(row.to(device))
                # The release encodes in FP32 with no autocast; an ambient autocast must not
                # lower the precision of the encoder.
                with torch.autocast(device_type=device.type, enabled=False):
                    dist = self.vae.encode(x, return_dict=False)[0]
                    if posterior == "sample":
                        z = dist.sample(generator=torch.Generator().manual_seed(seed))
                    else:
                        z = dist.mode()
                if tuple(z.shape) != expected:
                    raise RuntimeError(f"latent shape {tuple(z.shape)} != layout {expected}")
                if round_fp16:
                    z = round_condition_latents(z)
                rows.append(normalize_latents(z.float().cpu(), self.mean, self.std, self.layout))
        latents = torch.cat(rows)
        record = self._record("encode", runtime, plan)
        record.update(posterior=posterior, seed=seed, round_fp16=round_fp16, batch=batch)
        digest, row_digests = output_digests(rows)
        return VAEOutput(
            tensor=latents,
            digest=digest,
            row_digests=row_digests,
            tile_plan=plan,
            record=record,
        )

    # -- decode --
    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, autocast_fp16: bool = True) -> VAEOutput:
        """Decode normalized FP32 latents into FP32 RGB in ``[0, 1]`` (release decode block).

        Denormalization runs on the VAE device; the decode itself runs under FP16 autocast
        over the FP32 weights on CUDA, as released.
        """

        if self.require_release:
            self.check_release()
        validate_latent_stats(latents, self.mean, self.std, self.layout)
        if not self.layout.decodable(latents.shape[2]):
            raise ValueError(f"{latents.shape[2]} latent frames cannot be decoded")
        r = self.layout.spatial_factor
        plan = self.tile_plan(latents.shape[3] * r, latents.shape[4] * r)
        device = self._device()
        frames = self.layout.pixel_frames(latents.shape[2])
        rows = []
        with pinned_runtime(self.policy, device=device) as runtime:
            for row in latents.split(1):
                z = denormalize_latents(row.to(device), self.mean, self.std, self.layout)
                enabled = autocast_fp16 and device.type == "cuda"
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
                    video = self.vae.decode(z, return_dict=False)[0]
                if video.shape[2] != frames:
                    raise RuntimeError(f"decoded {video.shape[2]} frames, layout says {frames}")
                rows.append(postprocess_pixels(video).cpu())
        video = torch.cat(rows)
        record = self._record("decode", runtime, plan)
        record.update(autocast_fp16=autocast_fp16, batch=latents.shape[0])
        digest, row_digests = output_digests(rows)
        return VAEOutput(
            tensor=video,
            digest=digest,
            row_digests=row_digests,
            tile_plan=plan,
            record=record,
        )


def contract_manifest(
    heights_widths: Sequence[tuple[int, int]] = ((256, 256), (480, 832), (720, 1280)),
    frame_counts: Sequence[int] = (1, 22, 121, 241),
) -> dict[str, Any]:
    """Machine-readable summary of the static contract (layout, frame map, tile plans)."""

    layout = RELEASE_LAYOUT
    mean, std = release_latent_stats()
    return {
        "contract_id": CONTRACT_ID,
        "layout": asdict(layout),
        "layout_name": layout.name,
        "derived": {
            "frame_pre_padding": layout.frame_pre_padding,
            "tokens_chunk_size": layout.tokens_chunk_size,
            "token_overlap": layout.token_overlap,
            "frame_overlap": layout.frame_overlap,
        },
        "frame_map": {str(f): layout.latent_frames(f) for f in frame_counts},
        "tile_plans": {
            f"{h}x{w}": {"tiles": p.tiles, "sha256": p.digest(), **p.as_dict()}
            for h, w in heights_widths
            for p in (plan_tiles(h, w),)
        },
        "pixel_lut_sha256": pixel_lut_digest(),
        "latent_stats_sha256": latent_stats_digest(mean, std),
        "checkpoint": release_fingerprint()["checkpoint"],
        "reference": release_fingerprint()["reference"],
    }
