# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 Phase B ``phase_b_full_chain``: MiniMax-H3 from encoded inputs to decoded media.

The Phase B rows each pin one boundary of the released pipeline: the Qwen3-VL conditioner
(``qwen3vl_layer50_contract``), the two VAEs (``visual_vae_contract``, ``audio_vae_contract``)
and the two task front ends (``fl2va_pipeline_contract``, ``ref2va_pipeline_contract``). This
module chains them with the Phase A core (the packed Omni-Transformer and the dual
rectified-flow schedule) into one run and seals what every stage produced:

``request -> media_preprocess -> text_conditions -> visual_vae_encode -> audio_vae_encode ->
packed_layout -> initial_rows -> schedule -> denoise -> unpack -> video_decode -> audio_decode``

* **Per-stage artifact hashes.** Every stage records the sha256 (dtype, shape, bytes) of each
  tensor it hands on, in production order, plus its runtime provenance (contract ids, backend
  flags, device). Stage records are hash-chained and the artifact is sealed, so an edited,
  reordered or truncated artifact fails to load. The packed-layout artifact
  (``packed_layout.layout_artifact``) and the trajectory artifact (``trajectory``) are
  embedded and cross-checked.
* **First drift.** :func:`first_drift` compares two sealed runs stage by stage and names the
  first stage and component that differ. Inside ``denoise`` it goes further: the trajectory
  names the step, kind (velocity before latent) and modality, the transformer probe names the
  first module (input projections, refiner, block ``i``, output heads) whose output differs
  at that step, and with kept tensors the first differing *row*, mapped back to its packed
  identity (modality, condition / reference / target, ``(t, h, w)`` position, frame or audio
  channel).
* **Latent gate first.** :func:`gate` passes only when every stage up to ``unpack`` (layout,
  initial rows, every scheduler step, the final latents) is byte-equal. Decoded media hashes
  and metrics (``video_decode``, ``audio_decode``) are recorded and compared as secondary
  evidence; they never turn a latent failure into a pass, nor a latent pass into a failure.
* **Drift injection.** :class:`DriftInjection` perturbs one value by one ULP at a chosen
  stage, step, transformer module and row, so a test can prove the report localizes it.

The text encoder sits behind :class:`TextConditioner`. Until the layer-50 contract lands,
:class:`PinnedTextFeatures` replays features precomputed by the released conditioner,
bound to the exact conditioner input (prompt and preprocessed media) they were computed
from, and every artifact says so in its ``text_conditions`` provenance.

The transformer sits behind :class:`TransformerDenoiser`: provider arithmetic, or the strict
RL-Kernel graph bound through ``runtime.H3DiffusersAdapter`` (Hopper only today). The
sampler is the released ``MiniMaxH3Scheduler`` Euler step, written out op for op, on the
``h3_sigma_schedule`` grids.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import platform
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch

from rl_engine.models.minimax_h3 import fingerprint as fp
from rl_engine.models.minimax_h3 import packed_layout as pl
from rl_engine.models.minimax_h3 import trajectory as tj
from rl_engine.models.minimax_h3.trajectory import tensor_digest

SCHEMA_VERSION = "rlkernel.h3.full_chain.v1"
CONTRACT = "h3-phase-b-full-chain-v1"

STAGES = (
    "request",
    "media_preprocess",
    "text_conditions",
    "visual_vae_encode",
    "audio_vae_encode",
    "packed_layout",
    "initial_rows",
    "schedule",
    "denoise",
    "unpack",
    "video_decode",
    "audio_decode",
)
SECONDARY_STAGES = ("video_decode", "audio_decode")
GATED_STAGES = tuple(stage for stage in STAGES if stage not in SECONDARY_STAGES)

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
NUM_INFERENCE_STEPS = 50  # Diffusers' modular default; the released recipe

TEXT_SOURCE_PINNED = "pinned-precomputed"
TEXT_SOURCE_QWEN3VL = "qwen3vl_layer50_contract"
TEXT_SOURCES = (TEXT_SOURCE_PINNED, TEXT_SOURCE_QWEN3VL)
_PINNED_METADATA_KEY = "rlkernel.h3.text_features"
_TAG_NAMES = {pl.VIDEO_TAG: "video", pl.TEXT_TAG: "text", pl.AUDIO_TAG: "audio"}


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def media_digest(value: Any) -> str:
    """sha256 of a media object's pixels or samples: PIL image, numpy array or tensor."""

    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    if isinstance(value, np.ndarray):
        return tensor_digest(torch.from_numpy(np.array(value, copy=True, order="C")))
    if hasattr(value, "size") and hasattr(value, "mode"):  # PIL image
        return tensor_digest(torch.from_numpy(np.array(value, copy=True, order="C")))
    if isinstance(value, (list, tuple)):
        return _sha([media_digest(v) for v in value])
    raise TypeError(f"cannot digest media of type {type(value).__name__}")


def runtime_provenance(device: torch.device | str | None = None) -> dict[str, Any]:
    """Versions, backend flags and the device; no host names or paths."""

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform_machine": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": getattr(torch.version, "hip", None),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "num_threads": torch.get_num_threads(),
    }
    dev = torch.device(device) if device is not None else None
    if dev is not None:
        info["device"] = dev.type
        if dev.type == "cuda" and torch.cuda.is_available():
            index = dev.index if dev.index is not None else torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(index)
            info["compute_capability"] = list(torch.cuda.get_device_capability(index))
    return info


# --- drift injection ----------------------------------------------------------------------


@dataclass(frozen=True)
class DriftInjection:
    """Perturb one value by one ULP to prove that :func:`first_drift` localizes it.

    ``stage`` + ``component`` name a stage output (``"text_conditions"``, ``"prompt_embeds"``);
    inside ``denoise`` either ``module`` (a transformer module path such as
    ``"transformer_blocks.3"``) or ``component`` ``"velocity.video"`` / ``"velocity.audio"``
    is perturbed at ``step``. ``row`` indexes the tensor's row axis (packed rows for blocks).
    """

    stage: str
    component: str | None = None
    step: int | None = None
    module: str | None = None
    row: int = 0

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}")
        if self.stage == "denoise":
            if self.step is None or (self.module is None) == (self.component is None):
                raise ValueError("a denoise injection needs a step and one of module / component")
        elif self.module is not None or self.step is not None or self.component is None:
            raise ValueError("a stage injection needs a component and no step / module")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "component": self.component,
            "step": self.step,
            "module": self.module,
            "row": self.row,
        }


_INT_VIEW = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def bump_one_ulp(tensor: torch.Tensor, row: int = 0, row_dim: int = 0) -> torch.Tensor:
    """A copy of ``tensor`` with the first element of ``row`` (along ``row_dim``) moved one ULP."""

    out = tensor.detach().clone().contiguous()
    index = [0] * out.ndim
    index[row_dim] = row
    at = tuple(index)
    if out.dtype in _INT_VIEW:  # floating point: next representable value by bit pattern
        bits = out.view(_INT_VIEW[out.dtype])
        bits[at] += 1
    elif out.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        top = torch.iinfo(out.dtype).max  # integers (pixels): one count, down at the top
        out[at] += -1 if int(out[at]) == top else 1
    else:
        raise TypeError(f"cannot inject drift into a {tensor.dtype} tensor")
    return out


# --- text conditions ----------------------------------------------------------------------


@dataclass(frozen=True)
class TextRequest:
    """What the conditioner reads: the prompt and the preprocessed media, in request order."""

    workflow: str
    prompt: str
    media: tuple[Any, ...]
    media_kinds: tuple[str, ...]
    input_sha256: str


def conditioner_input_sha256(
    workflow: str, prompt: str, media_kinds: Sequence[str], media_digests: Sequence[str]
) -> str:
    """Digest binding text features to the exact conditioner input they were computed from."""

    return _sha(
        {
            "workflow": workflow,
            "prompt_sha256": text_sha256(prompt),
            "media": [[k, d] for k, d in zip(media_kinds, media_digests, strict=True)],
        }
    )


@dataclass(frozen=True)
class TextConditions:
    prompt_embeds: torch.Tensor  # (1, num_text_tokens, text_dim), the conditioner's dtype
    text_token_tags: torch.Tensor  # (num_text_tokens,) int64, text or video (vision block) tags
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.prompt_embeds.ndim != 3 or self.prompt_embeds.shape[0] != 1:
            raise ValueError("prompt_embeds must be (1, num_text_tokens, text_dim)")
        tags = pl.check_text_token_tags(self.text_token_tags)
        if tags.shape[0] != self.prompt_embeds.shape[1]:
            raise ValueError("one text_token_tag per prompt_embeds row is required")


class TextConditioner(Protocol):
    """The text-encoder stage. ``qwen3vl_layer50_contract`` will provide the strict one."""

    def encode(self, request: TextRequest) -> TextConditions: ...


class PinnedFeaturesMismatch(ValueError):
    """Pinned text features were computed for a different conditioner input."""


def save_pinned_text_features(
    path: str | Path,
    prompt_embeds: torch.Tensor,
    text_token_tags: torch.Tensor,
    *,
    input_sha256: str,
    producer: Mapping[str, Any],
) -> str:
    """Write features with their binding metadata; returns the file's sha256."""

    from safetensors.torch import save_file

    conditions = TextConditions(prompt_embeds.detach().cpu(), text_token_tags.detach().cpu())
    conditions.validate()
    meta = {
        "conditioner_input_sha256": input_sha256,
        "prompt_embeds_sha256": tensor_digest(conditions.prompt_embeds),
        "text_token_tags_sha256": tensor_digest(conditions.text_token_tags.long()),
        "producer": dict(producer),
    }
    save_file(
        {
            "prompt_embeds": conditions.prompt_embeds.contiguous(),
            "text_token_tags": conditions.text_token_tags.long().contiguous(),
        },
        str(path),
        metadata={_PINNED_METADATA_KEY: json.dumps(meta, sort_keys=True)},
    )
    return _file_sha256(Path(path))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PinnedTextFeatures:
    """Replay precomputed conditioner features, refusing any other conditioner input.

    The file holds ``prompt_embeds`` and ``text_token_tags`` and, in its metadata, the
    :func:`conditioner_input_sha256` they were computed for and who computed them. A request
    whose prompt or preprocessed media differ raises :class:`PinnedFeaturesMismatch`.
    """

    def __init__(self, path: str | Path, *, expected_sha256: str | None = None) -> None:
        self.path = Path(path)
        self.expected_sha256 = expected_sha256

    def encode(self, request: TextRequest) -> TextConditions:
        from safetensors.torch import load

        data = self.path.read_bytes()  # one snapshot: hashed and loaded
        file_sha256 = hashlib.sha256(data).hexdigest()
        if self.expected_sha256 is not None and file_sha256 != self.expected_sha256:
            raise PinnedFeaturesMismatch(
                f"pinned features sha256 {file_sha256} != expected {self.expected_sha256}"
            )
        tensors = load(data)  # also checks the header
        header = json.loads(data[8 : 8 + int.from_bytes(data[:8], "little")])
        raw = (header.get("__metadata__") or {}).get(_PINNED_METADATA_KEY)
        if raw is None:
            raise PinnedFeaturesMismatch("file has no pinned text-feature metadata")
        meta = json.loads(raw)
        embeds, tags = tensors["prompt_embeds"], tensors["text_token_tags"]
        if meta["conditioner_input_sha256"] != request.input_sha256:
            raise PinnedFeaturesMismatch(
                "pinned features were computed for another prompt or other media "
                f"({meta['conditioner_input_sha256'][:12]} != {request.input_sha256[:12]})"
            )
        if tensor_digest(embeds) != meta["prompt_embeds_sha256"] or (
            tensor_digest(tags) != meta["text_token_tags_sha256"]
        ):
            raise PinnedFeaturesMismatch("pinned tensors do not match their recorded digests")
        return TextConditions(
            embeds,
            tags,
            {
                "source": TEXT_SOURCE_PINNED,
                "file_sha256": file_sha256,
                "conditioner_input_sha256": meta["conditioner_input_sha256"],
                "producer": meta["producer"],
            },
        )


class TextSourceMismatch(ValueError):
    """The text conditions do not come from the source the run declared."""


class Qwen3VLLayer50Conditioner:
    """Text conditions from ``qwen3vl_layer50_contract`` (``qwen3vl_text``), computed at run time.

    It builds the released presentation exactly as Diffusers' ``MiniMaxH3FL2VATextEncoderStep`` /
    ``MiniMaxH3Ref2VATextEncoderStep`` do, with the contract's tokenization: labels and prompt
    through ``tokenize_prompt`` (verbatim, no specials, refusing a prompt that spells one),
    vision blocks sized from the processor's grids. Then it runs ``Qwen3VLLayer50Encoder``, one
    presentation per forward, stopping at ``hidden_states[50]``. The provenance names the
    contract, its fingerprint and the feature and token-id digests.
    """

    def __init__(
        self,
        encoder: Any,
        tokenizer: Any,
        processor: Any,
        *,
        load_report: Mapping[str, Any] | None = None,
        check_release: bool = True,
    ) -> None:
        from rl_engine.models.minimax_h3 import qwen3vl_text as qt

        if check_release:
            qt.verify_release_tokenizer(tokenizer)
            qt.verify_release_processor(processor)
        self.encoder, self.tokenizer, self.processor = encoder, tokenizer, processor
        self.load_report = dict(load_report or {})
        self.check_release = check_release

    @classmethod
    def from_release(
        cls, root: str | Path, *, device: str | torch.device = "cuda"
    ) -> Qwen3VLLayer50Conditioner:
        """Load ``text_encoder/`` (50 layers), ``tokenizer/`` and ``processor/`` under ``root``."""

        from transformers import AutoProcessor, AutoTokenizer

        from rl_engine.models.minimax_h3 import qwen3vl_text as qt

        root = Path(root)
        model, report = qt.load_release_text_encoder(root / "text_encoder", device=device)
        return cls(
            qt.Qwen3VLLayer50Encoder(model),
            AutoTokenizer.from_pretrained(str(root / "tokenizer")),
            AutoProcessor.from_pretrained(str(root / "processor")),
            load_report=report,
        )

    def _tokenize(self, text: str) -> list[int]:
        from rl_engine.models.minimax_h3 import qwen3vl_text as qt

        return qt.tokenize_prompt(self.tokenizer, text)

    def _token_id(self, token: str) -> int:
        from rl_engine.models.minimax_h3 import qwen3vl_text as qt

        return qt.vision_token_ids()[token]

    def presentation(self, request: TextRequest) -> tuple[list[int], list[int], dict[str, Any]]:
        """``(token ids, modality tags, vision inputs)`` of the released presentation."""

        from rl_engine.models.minimax_h3 import fl2va, ref2va

        merge = int(self.processor.image_processor.merge_size) ** 2
        vision: dict[str, Any] = {}
        if request.workflow in ("fl2va", "t2va"):
            counts: list[int] = []
            if request.media:
                features = self.processor.image_processor(
                    images=list(request.media), return_tensors="pt"
                )
                vision = {
                    "pixel_values": features["pixel_values"],
                    "image_grid_thw": features["image_grid_thw"],
                }
                counts = [int(g.prod()) // merge for g in features["image_grid_thw"]]
            ids, tags = fl2va.fl2va_presentation(
                self._tokenize, self._token_id, request.prompt, counts
            )
            return ids, tags, vision
        references = list(request.media)
        image_counts: list[int] = []
        images = [r.image for r in references if isinstance(r, ref2va.ImageReference)]
        if images:
            features = self.processor.image_processor(images=images, return_tensors="pt")
            vision["pixel_values"] = features["pixel_values"]
            vision["image_grid_thw"] = features["image_grid_thw"]
            image_counts = [int(g.prod()) // merge for g in features["image_grid_thw"]]
        block_counts: list[int] = []
        block_times: list[list[float]] = []
        videos = [r for r in references if isinstance(r, ref2va.VideoReference)]
        if videos:
            patch = int(self.processor.video_processor.temporal_patch_size)
            sampled = [
                ref2va.sample_video_condition_frames(v.frames, fps=pl.FPS, temporal_patch=patch)
                for v in videos
            ]
            block_times = [times for _, times in sampled]
            features = self.processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled],
                do_sample_frames=False,
                return_tensors="pt",
            )
            vision["pixel_values_videos"] = features["pixel_values_videos"]
            vision["video_grid_thw"] = features["video_grid_thw"]
            block_counts = [int(g[1]) * int(g[2]) // merge for g in features["video_grid_thw"]]
            for times, grid in zip(block_times, features["video_grid_thw"], strict=True):
                if int(grid[0]) != len(times):
                    raise ValueError(
                        f"processor merged a reference video into {int(grid[0])} blocks, "
                        f"the presentation labels {len(times)}"
                    )
        ids, tags = ref2va.ref2va_presentation(
            self._tokenize,
            self._token_id,
            request.prompt,
            [(r.kind, bool(r.has_audio)) for r in references],
            image_counts,
            block_counts,
            block_times,
        )
        return ids, tags, vision

    def encode(self, request: TextRequest) -> TextConditions:
        from rl_engine.models.minimax_h3 import qwen3vl_text as qt

        ids, tags, vision = self.presentation(request)
        out = self.encoder.encode(ids, vision or None)
        return TextConditions(
            out.feature,
            torch.tensor(tags, dtype=torch.long),
            {
                "source": TEXT_SOURCE_QWEN3VL,
                "contract_id": qt.CONTRACT_ID,
                "fingerprint_sha256": _file_sha256(qt.FINGERPRINT_PATH),
                "release_checked": self.check_release,
                "text_encoder_layer": self.encoder.layer,
                "feature_sha256": out.feature_digest,
                "token_ids_sha256": out.token_ids_digest,
                "num_tokens": len(ids),
                "device": str(self.encoder.device),
                "dtype": str(self.encoder.dtype),
                "runtime": out.runtime,
                "load": self.load_report,
            },
        )


class CallableTextConditioner:
    """Adapter for a conditioner function ``request -> (prompt_embeds, text_token_tags)``."""

    def __init__(self, fn: Callable[[TextRequest], Any], identity: Mapping[str, Any]) -> None:
        self.fn = fn
        self.identity = dict(identity)

    def encode(self, request: TextRequest) -> TextConditions:
        embeds, tags = self.fn(request)
        return TextConditions(
            embeds, torch.as_tensor(tags, dtype=torch.long), {"source": "callable", **self.identity}
        )


# --- dual schedule and the released Euler step --------------------------------------------


@dataclass(frozen=True)
class DualSchedule:
    """The two released grids (``h3_sigma_schedule``), FP32 on CPU."""

    num_inference_steps: int
    video_shift: float
    audio_shift: float
    video_sigmas: torch.Tensor
    audio_sigmas: torch.Tensor
    video_timesteps: torch.Tensor
    audio_timesteps: torch.Tensor

    @property
    def steps(self) -> int:
        return int(self.video_timesteps.numel())


def dual_schedule(
    num_inference_steps: int, *, video_shift: float = VIDEO_SHIFT, audio_shift: float = AUDIO_SHIFT
) -> DualSchedule:
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    op = NativeH3SigmaScheduleOp()
    video = op(num_inference_steps, shift=video_shift)
    audio = op(num_inference_steps, shift=audio_shift)
    if video.num_model_evals != audio.num_model_evals:
        # Diffusers zips the two timestep lists and indexes the audio list by the video step:
        # unequal lengths would silently truncate or fail mid-run. Refuse up front.
        raise ValueError(
            f"video ({video.num_model_evals}) and audio ({audio.num_model_evals}) schedules "
            "differ in length after deduplication"
        )
    return DualSchedule(
        int(num_inference_steps),
        float(video_shift),
        float(audio_shift),
        video.sigmas,
        audio.sigmas,
        video.timesteps,
        audio.timesteps,
    )


def euler_step(
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """``MiniMaxH3Scheduler.step`` (diffusers ``f53d552``), op for op.

    ``x0 = x + (1 - t) * v`` takes its sigma from the *timestep* the transformer saw; the blend
    ``r * x + (1 - r) * x0`` with ``r = sigma_next / sigma`` takes it from the grid. The two
    differ for sigma < 0.5 (``1 - (1 - sigma)`` is not exact in FP32), and the release keeps
    them apart, so this does too. Each operation is a separately rounded eager kernel.
    """

    sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
    while sigma_from_timestep.ndim < sample.ndim:
        sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
    denoised = sample + sigma_from_timestep * model_output
    half = (torch.float16, torch.bfloat16)
    compute_dtype = torch.float32 if sample.dtype in half else sample.dtype
    sigma = sigma.to(device=sample.device, dtype=compute_dtype)
    sigma_next = sigma_next.to(device=sample.device, dtype=compute_dtype)
    ratio = sigma_next / sigma
    prev = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
    return prev.to(dtype=sample.dtype)


# --- transformer --------------------------------------------------------------------------


_ROW_SPACES = {
    "proj_in": ("video", 1),
    "audio_proj_in": ("audio", 1),
    "context_embedder": ("text", 1),
    "token_refiner": ("text", 1),
    "time_proj": ("timestep", 0),
    "time_embedder": ("timestep", 0),
    "norm_out": ("packed", 1),
    "proj_out": ("packed", 1),
    "audio_proj_out": ("packed", 1),
}
_PROBED = (
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "token_refiner",
    "time_embedder",
    "norm_out",
    "proj_out",
    "audio_proj_out",
)


def module_row_space(path: str) -> tuple[str, int]:
    """``(row space, row dim)`` of a probed module's output."""

    if path.startswith("transformer_blocks."):
        return "packed", 1
    return _ROW_SPACES[path.split("#")[0]]


class TransformerProbe:
    """Per-step digests of every probed module's output, in execution order.

    Hooks are observers only (they never change the arithmetic) unless an injection targets
    the module at the current step. ``capture(step, path)`` decides which outputs are kept as
    CPU tensors for row localization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        capture: Callable[[int, str], bool] | None = None,
        inject: DriftInjection | None = None,
    ) -> None:
        self.model = model
        self.capture = capture
        self.inject = inject
        self.step = -1
        self.steps: list[list[list[str]]] = []
        self.tensors: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._calls: dict[str, int] = {}
        self.injected = False

    def modules(self) -> list[tuple[str, torch.nn.Module]]:
        found = [(name, getattr(self.model, name)) for name in _PROBED if hasattr(self.model, name)]
        blocks = getattr(self.model, "transformer_blocks", None)
        if blocks is not None:
            found += [(f"transformer_blocks.{i}", block) for i, block in enumerate(blocks)]
        return found

    def begin_step(self, step: int) -> None:
        self.step = step
        self.steps.append([])
        self._calls = {}

    def _hook(self, path: str) -> Callable[..., Any]:
        def hook(module: torch.nn.Module, args: Any, output: Any) -> Any:
            if self.step < 0 or not isinstance(output, torch.Tensor):
                return None
            calls = self._calls.get(path, 0)
            self._calls[path] = calls + 1
            name = path if calls == 0 else f"{path}#{calls}"
            replaced = None
            inject = self.inject
            if (
                inject is not None
                and inject.module == name
                and inject.step == self.step
                and not self.injected
            ):
                _, row_dim = module_row_space(name)
                replaced = bump_one_ulp(output, inject.row, row_dim)
                output = replaced
                self.injected = True
            self.steps[-1].append([name, tensor_digest(output)])
            if self.capture is not None and self.capture(self.step, name):
                self.tensors[f"probe/{self.step}/{name}"] = output.detach().cpu().clone()
            return replaced

        return hook

    @contextlib.contextmanager
    def attached(self) -> Iterator[TransformerProbe]:
        try:
            for path, module in self.modules():
                self._handles.append(module.register_forward_hook(self._hook(path)))
            yield self
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles = []

    def record(self) -> dict[str, Any]:
        return {"steps": self.steps}


class TransformerDenoiser:
    """One forward of a Diffusers-compatible ``MiniMaxH3Transformer3DModel`` per step.

    ``variant`` is the partition the weights came from (``"FL2VA"`` for ``transformer/``,
    ``"Ref2VA"`` for ``transformer_ref/``); the chain refuses a layout of the other variant
    before any forward (H16). ``mode=None`` runs the provider's arithmetic; ``"strict"`` /
    ``"audit"`` bind the RL-Kernel graph through ``runtime.H3DiffusersAdapter`` for the whole
    sampling loop (forward hooks of the probe still observe every module); ``ops`` replaces the
    default ``h3_block.CandidateOps`` (e.g. with query-chunked attention).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        variant: str,
        mode: str | None = None,
        identity: Mapping[str, Any] | None = None,
        ops: Any = None,
    ) -> None:
        if variant not in fp.VARIANTS:
            raise ValueError(f"unknown transformer variant {variant!r}")
        self.model = model
        self.variant = variant
        self.mode = mode
        self.extra_identity = dict(identity or {})
        self.adapter: Any = None
        if mode is not None:
            from rl_engine.models.minimax_h3.runtime import H3DiffusersAdapter

            self.adapter = H3DiffusersAdapter(model, mode, ops=ops)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def identity(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        config_dict = _jsonable(dict(config)) if isinstance(config, Mapping) else {}
        dtypes = sorted({str(p.dtype) for p in self.model.parameters()})
        return {
            "class": type(self.model).__name__,
            "variant": self.variant,
            "transformer_dir": fp.VARIANTS[self.variant],
            "config_sha256": _sha({k: v for k, v in config_dict.items() if not k.startswith("_")}),
            "num_layers": len(getattr(self.model, "transformer_blocks", ())),
            "parameter_dtypes": dtypes,
            "mode": self.mode or "provider",
            **self.extra_identity,
        }

    @contextlib.contextmanager
    def session(self) -> Iterator[None]:
        if self.adapter is None:
            yield
            return
        with self.adapter.active():
            yield

    def report(self) -> dict[str, Any]:
        if self.adapter is None:
            return {"case_id": "P/P", "bound": False, "mode": "provider"}
        report = dict(self.adapter.report())
        report["actual_backends"] = dict(self.adapter.trace.actual_backends)
        return report

    def __call__(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        timestep_indices: torch.Tensor,
        layout_inputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            hidden_states=video_rows[None],
            audio_hidden_states=audio_rows[None],
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            timestep_indices=timestep_indices,
            return_dict=False,
            **layout_inputs,
        )
        return out[0], out[1]


# --- requests -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FL2VARequest:
    """One FL2VA (or zero-keyframe t2va) request, as the released pipeline takes it."""

    prompt: str
    image: Any = None
    last_image: Any = None
    height: int | None = None
    width: int | None = None
    num_frames: int = 124
    num_inference_steps: int = NUM_INFERENCE_STEPS
    seed: int = 42
    canvas_short_edge: int = pl.CANVAS_SHORT_EDGE
    canvas_max_pixels: int = pl.CANVAS_MAX_PIXELS

    @property
    def workflow(self) -> str:
        return "fl2va" if self.image is not None or self.last_image is not None else "t2va"

    def spec(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "prompt_sha256": text_sha256(self.prompt),
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "generator": "torch.Generator('cpu').manual_seed(seed)",
            "canvas": [self.canvas_short_edge, self.canvas_max_pixels],
            "keyframes": [
                a for a, k in (("first", self.image), ("last", self.last_image)) if k is not None
            ],
        }


@dataclass(frozen=True)
class Ref2VARequest:
    """One Ref2VA request: references in request order (``ref2va.ImageReference`` ...)."""

    prompt: str
    references: tuple[Any, ...]
    height: int | None = None
    width: int | None = None
    num_frames: int = 124
    num_inference_steps: int = NUM_INFERENCE_STEPS
    seed: int = 42
    span_summation: str = pl.DEFAULT_SPAN_SUMMATION
    canvas_short_edge: int = pl.CANVAS_SHORT_EDGE
    canvas_max_pixels: int = pl.CANVAS_MAX_PIXELS
    image_short_edge: int = 2048

    workflow = "ref2va"

    def spec(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "prompt_sha256": text_sha256(self.prompt),
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "generator": "torch.Generator('cpu').manual_seed(seed)",
            "span_summation": self.span_summation,
            "canvas": [self.canvas_short_edge, self.canvas_max_pixels],
            "image_short_edge": self.image_short_edge,
            "references": [
                {
                    "kind": ref.kind,
                    "has_audio": bool(ref.has_audio),
                    "fps": getattr(ref, "fps", None),
                    "sample_rate": getattr(ref, "sample_rate", None),
                }
                for ref in self.references
            ],
        }


@dataclass
class ChainComponents:
    """What a run executes: conditioner, denoiser and the two VAE contracts."""

    text: TextConditioner
    denoiser: TransformerDenoiser
    visual_vae: Any = None  # visual_vae.VisualVAEContract (keyframes, visual refs, decode)
    audio_vae: Any = None  # audio_vae.H3AudioVAEContract (audio refs, decode)
    device: torch.device | str = "cpu"


# --- recording ----------------------------------------------------------------------------


GENESIS = "0" * 64


def _component_digest(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    if isinstance(value, str):
        return value
    return _sha(value)


class InjectionNotRun(ValueError):
    """A requested drift injection matched nothing: a run sealed with it would prove nothing."""


class ChainRecorder:
    """Stage records in :data:`STAGES` order, hash-chained; optional CPU tensor copies."""

    def __init__(
        self,
        *,
        keep_tensors: bool = False,
        inject: DriftInjection | None = None,
        text_source: str = TEXT_SOURCE_PINNED,
    ):
        self.text_source = text_source
        self.records: list[dict[str, Any]] = []
        self.keep_tensors = keep_tensors
        self.inject = inject
        self.injected = False
        self.tensors: dict[str, torch.Tensor] = {}

    def apply(self, stage: str, name: str, tensor: torch.Tensor, row_dim: int = 0) -> torch.Tensor:
        """Return ``tensor``, perturbed when the injection targets this stage output."""

        inject = self.inject
        if (
            inject is not None
            and not self.injected
            and inject.stage == stage
            and inject.component == name
            and inject.step is None
        ):
            self.injected = True
            return bump_one_ulp(tensor, inject.row, row_dim)
        return tensor

    def record(
        self,
        stage: str,
        components: Sequence[tuple[str, Any]],
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        expected = STAGES[len(self.records)] if len(self.records) < len(STAGES) else None
        if stage != expected:
            raise ValueError(f"expected stage {expected!r}, got {stage!r}")
        names = [name for name, _ in components]
        if len(set(names)) != len(names):
            raise ValueError(f"{stage}: duplicate component names")
        body = {
            "stage": stage,
            "components": [[name, _component_digest(value)] for name, value in components],
            "provenance": _jsonable(dict(provenance or {})),
        }
        previous = self.records[-1]["chain_sha256"] if self.records else GENESIS
        record = {**body, "chain_sha256": _sha({"previous": previous, "stage": body})}
        self.records.append(record)
        if self.keep_tensors:
            for name, value in components:
                if isinstance(value, torch.Tensor):
                    self.tensors[f"{stage}/{name}"] = value.detach().cpu().clone()
        return record


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class ChainRun:
    """A sealed run plus what it produced."""

    artifact: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    video: torch.Tensor | None = None
    audio: torch.Tensor | None = None

    def save_tensors(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in self.tensors.items()}, str(path))


# --- the chain ----------------------------------------------------------------------------


def _skip(reason: str) -> dict[str, Any]:
    return {"skipped": reason}


@dataclass(frozen=True)
class PreparedMedia:
    """A request's preprocessed media (the ``media_preprocess`` stage) and its text request."""

    height: int
    width: int
    num_frames: int
    media: tuple[tuple[str, torch.Tensor], ...]  # VAE inputs in request order
    text_request: TextRequest
    anchors: tuple[str, ...] = ()  # FL2VA keyframe anchors
    normalized: tuple[Any, ...] = ()  # Ref2VA normalized references


def prepare_fl2va(request: FL2VARequest) -> PreparedMedia:
    """Canvas, keyframe pixels and the conditioner input (``fl2va_pipeline_contract``)."""

    from rl_engine.models.minimax_h3 import fl2va

    keyframes = fl2va.prepare_keyframes(
        request.image,
        request.last_image,
        height=request.height,
        width=request.width,
        short_edge=request.canvas_short_edge,
        max_pixels=request.canvas_max_pixels,
    )
    pixels = [fl2va.keyframe_pixels(image) for image in keyframes.images]
    text_request = TextRequest(
        workflow=request.workflow,
        prompt=request.prompt,
        media=keyframes.images,
        media_kinds=tuple("image" for _ in keyframes.images),
        input_sha256=conditioner_input_sha256(
            request.workflow,
            request.prompt,
            ["image"] * len(pixels),
            [tensor_digest(p) for p in pixels],
        ),
    )
    return PreparedMedia(
        keyframes.height,
        keyframes.width,
        pl.resolve_num_frames(request.num_frames),
        tuple((f"keyframe_pixels.{i}", p) for i, p in enumerate(pixels)),
        text_request,
        anchors=tuple(keyframes.anchors),
    )


def prepare_ref2va(request: Ref2VARequest) -> PreparedMedia:
    """Canvas, normalized references, VAE inputs and the conditioner input."""

    from rl_engine.models.minimax_h3 import ref2va

    height, width, num_frames = ref2va.resolve_ref2va_plan(
        num_frames=request.num_frames,
        height=request.height,
        width=request.width,
        short_edge=request.canvas_short_edge,
        max_pixels=request.canvas_max_pixels,
    )
    normalized = ref2va.normalize_references(
        request.references,
        num_frames=num_frames,
        image_short_edge=request.image_short_edge,
        canvas_short_edge=request.canvas_short_edge,
        canvas_max_pixels=request.canvas_max_pixels,
    )
    media: list[tuple[str, torch.Tensor]] = []
    conditioner_media: list[tuple[str, str]] = []
    for i, ref in enumerate(normalized):
        if isinstance(ref, ref2va.ImageReference):
            pixels = torch.from_numpy(np.array(ref.image)).permute(2, 0, 1)[None, :, None]
            media.append((f"reference.{i}.pixels", pixels.contiguous()))
            conditioner_media.append((f"reference.{i}.image", media_digest(ref.image)))
        elif isinstance(ref, ref2va.VideoReference):
            count = ref2va.vae_video_frame_count(ref.frames.shape[0])
            pixels = torch.from_numpy(ref.frames[:count].copy()).permute(3, 0, 1, 2)[None]
            media.append((f"reference.{i}.pixels", pixels.contiguous()))
            conditioner_media.append((f"reference.{i}.video", media_digest(ref.frames)))
        audio = None if isinstance(ref, ref2va.ImageReference) else ref.audio
        if audio is not None:
            media.append((f"reference.{i}.waveform", audio.float()))
            conditioner_media.append((f"reference.{i}.audio", media_digest(audio)))
    text_request = TextRequest(
        workflow="ref2va",
        prompt=request.prompt,
        media=tuple(normalized),
        media_kinds=tuple(ref.kind for ref in normalized),
        input_sha256=conditioner_input_sha256(
            "ref2va",
            request.prompt,
            [name for name, _ in conditioner_media],
            [digest for _, digest in conditioner_media],
        ),
    )
    return PreparedMedia(
        height, width, num_frames, tuple(media), text_request, normalized=tuple(normalized)
    )


def prepare(request: FL2VARequest | Ref2VARequest) -> PreparedMedia:
    if isinstance(request, FL2VARequest):
        return prepare_fl2va(request)
    if isinstance(request, Ref2VARequest):
        return prepare_ref2va(request)
    raise TypeError(f"unsupported request type {type(request).__name__}")


def _record_media(
    prepared: PreparedMedia, rec: ChainRecorder, provenance: Mapping[str, Any]
) -> list[tuple[str, torch.Tensor]]:
    media = [
        (name, rec.apply("media_preprocess", name, t, 3 if t.ndim == 5 else 1))
        for name, t in prepared.media
    ]
    rec.record("media_preprocess", media, provenance)
    return media


def _fl2va_front(
    request: FL2VARequest, components: ChainComponents, rec: ChainRecorder
) -> tuple[TextConditions, pl.PackedLayout, list[tuple[str, torch.Tensor]], dict[str, Any]]:
    from rl_engine.models.minimax_h3 import fl2va

    raw = [
        (a, k) for a, k in (("first", request.image), ("last", request.last_image)) if k is not None
    ]
    rec.record(
        "request",
        [("spec", request.spec()), ("prompt", text_sha256(request.prompt))]
        + [(f"keyframe.{a}", media_digest(k)) for a, k in raw],
        {"workflow": request.workflow, "variant": "FL2VA"},
    )
    prepared = prepare_fl2va(request)
    pixels = _record_media(
        prepared,
        rec,
        {
            "contract": "fl2va_pipeline_contract",
            "canvas": [prepared.height, prepared.width],
            "anchors": list(prepared.anchors),
        },
    )
    text = _text_stage(components.text, prepared.text_request, rec)
    condition_latents: list[torch.Tensor] = []
    vae_provenance: dict[str, Any] = _skip("no keyframes")
    for i, (_, p) in enumerate(pixels):
        if components.visual_vae is None:
            raise ValueError("keyframes need a visual VAE contract")
        out = components.visual_vae.encode(p)
        condition_latents.append(
            rec.apply("visual_vae_encode", f"condition_latents.{i}", out.tensor, 1)
        )
        vae_provenance = dict(out.record)
    rec.record(
        "visual_vae_encode",
        [(f"condition_latents.{i}", z) for i, z in enumerate(condition_latents)],
        vae_provenance,
    )
    rec.record("audio_vae_encode", [], _skip("FL2VA has no audio conditions"))
    layout = fl2va.build_fl2va_layout(
        text.text_token_tags,
        height=prepared.height,
        width=prepared.width,
        num_frames=request.num_frames,
        anchors=prepared.anchors,
        workflow=request.workflow,
    )
    inputs = {f"condition_latents.{i}": tensor_digest(z) for i, z in enumerate(condition_latents)}
    return text, layout, [("video", z) for z in condition_latents], inputs


def _ref2va_front(
    request: Ref2VARequest, components: ChainComponents, rec: ChainRecorder
) -> tuple[TextConditions, pl.PackedLayout, list[tuple[str, torch.Tensor]], dict[str, Any]]:
    from rl_engine.models.minimax_h3 import ref2va

    kinds = ref2va.validate_references(request.references)
    raw: list[tuple[str, Any]] = [("spec", request.spec()), ("prompt", text_sha256(request.prompt))]
    for i, ref in enumerate(request.references):
        if ref.kind in ("image", "video"):
            raw.append((f"reference.{i}.{ref.kind}", media_digest(_ref_visual(ref))))
        if ref.has_audio:
            raw.append((f"reference.{i}.audio", media_digest(ref.audio)))
    rec.record("request", raw, {"workflow": "ref2va", "variant": "Ref2VA"})
    prepared = prepare_ref2va(request)
    media = _record_media(
        prepared,
        rec,
        {
            "contract": "ref2va_pipeline_contract",
            "canvas": [prepared.height, prepared.width],
            "num_frames": prepared.num_frames,
            "kinds": kinds,
        },
    )
    text = _text_stage(components.text, prepared.text_request, rec)

    condition_latents: list[torch.Tensor] = []
    visual_provenance: dict[str, Any] = _skip("no visual references")
    for name, pixels in media:
        if not name.endswith(".pixels"):
            continue
        if components.visual_vae is None:
            raise ValueError("visual references need a visual VAE contract")
        out = components.visual_vae.encode(pixels)
        i = len(condition_latents)
        condition_latents.append(
            rec.apply("visual_vae_encode", f"condition_latents.{i}", out.tensor, 1)
        )
        visual_provenance = dict(out.record)
    rec.record(
        "visual_vae_encode",
        [(f"condition_latents.{i}", z) for i, z in enumerate(condition_latents)],
        visual_provenance,
    )
    audio_rows: list[torch.Tensor] = []
    audio_provenance: dict[str, Any] = _skip("no soundtracks")
    for name, wave in media:
        if not name.endswith(".waveform"):
            continue
        if components.audio_vae is None:
            raise ValueError("audio references need an audio VAE contract")
        enc = components.audio_vae.encode_stereo(wave)
        i = len(audio_rows)
        audio_rows.append(rec.apply("audio_vae_encode", f"audio_rows.{i}", enc.rows, 0))
        audio_provenance = components.audio_vae.provenance()
    rec.record(
        "audio_vae_encode",
        [(f"audio_rows.{i}", r) for i, r in enumerate(audio_rows)],
        audio_provenance,
    )
    blocks = ref2va.reference_blocks(
        [(ref.kind, bool(ref.has_audio)) for ref in prepared.normalized],
        [tuple(z.shape) for z in condition_latents],
        [int(r.shape[0]) for r in audio_rows],
    )
    layout = ref2va.build_ref2va_layout(
        text.text_token_tags,
        blocks,
        height=prepared.height,
        width=prepared.width,
        num_frames=prepared.num_frames,
        span_summation=request.span_summation,
    )
    inputs = {f"condition_latents.{i}": tensor_digest(z) for i, z in enumerate(condition_latents)}
    inputs.update({f"audio_rows.{i}": tensor_digest(r) for i, r in enumerate(audio_rows)})
    conditions = [("video", z) for z in condition_latents] + [("audio", r) for r in audio_rows]
    return text, layout, conditions, inputs


def _ref_visual(ref: Any) -> Any:
    return ref.image if ref.kind == "image" else ref.frames


def _text_stage(
    conditioner: TextConditioner, request: TextRequest, rec: ChainRecorder
) -> TextConditions:
    text = conditioner.encode(request)
    text.validate()
    source = text.provenance.get("source")
    if source != rec.text_source:
        raise TextSourceMismatch(
            f"run declared text source {rec.text_source!r}, the conditioner reports {source!r}"
        )
    embeds = rec.apply("text_conditions", "prompt_embeds", text.prompt_embeds, 1)
    tags = text.text_token_tags.long().cpu()
    rec.record(
        "text_conditions",
        [("prompt_embeds", embeds), ("text_token_tags", tags)],
        {"conditioner_input_sha256": request.input_sha256, **dict(text.provenance)},
    )
    return TextConditions(embeds, tags, text.provenance)


def _unpack_video(rows: torch.Tensor, geometry: Mapping[str, Any]) -> torch.Tensor:
    """Target video rows -> ``(1, 24, T, H, W)`` (``MiniMaxH3AfterDenoiseStep``)."""

    patch_t, patch_h, patch_w = pl.PATCH_SIZE
    frames, height, width = (
        geometry["num_latent_frames"],
        geometry["latent_height"],
        geometry["latent_width"],
    )
    channels = rows.shape[-1] // (patch_t * patch_h * patch_w)
    x = rows.reshape(
        -1,
        frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(-1, channels, frames, height, width).contiguous()


def _unpack_audio(rows: torch.Tensor, num_latents: int) -> torch.Tensor:
    """Channel-major target rows -> ``(2, C, T)`` (``MiniMaxH3AfterDenoiseStep``)."""

    rows = rows.reshape(pl.AUDIO_CHANNELS, num_latents, rows.shape[-1])
    return rows.permute(0, 2, 1).contiguous()


def _audio_rows(latents: torch.Tensor) -> torch.Tensor:
    """``(2, C, T)`` -> channel-major rows ``(2 * T, C)``: the inverse of :func:`_unpack_audio`."""

    return latents.permute(0, 2, 1).reshape(-1, latents.shape[1]).contiguous()


def finite(value: Any) -> Any:
    """``value`` with every non-finite float as ``None``: strict JSON has no NaN or Infinity."""

    if isinstance(value, Mapping):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def video_metrics(video: torch.Tensor) -> dict[str, Any]:
    v = video.double()
    return finite(
        {
            "shape": list(video.shape),
            "mean_rgb": v.mean(dim=(0, 2, 3, 4)).tolist(),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
        }
    )


def audio_metrics(audio: torch.Tensor) -> dict[str, Any]:
    a = audio.double()
    return finite(
        {
            "shape": list(audio.shape),
            "rms_lr": a.pow(2).mean(dim=-1).sqrt().tolist(),
            "peak": float(a.abs().max()),
        }
    )


def run_chain(
    request: FL2VARequest | Ref2VARequest,
    components: ChainComponents,
    *,
    label: str,
    text_source: str,
    probe: bool = True,
    keep_tensors: bool = False,
    capture: Callable[[int, str], bool] | None = None,
    inject: DriftInjection | None = None,
    decode: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ChainRun:
    """Run one request through every stage and seal the result.

    ``text_source`` is the declared origin of the text conditions (``TEXT_SOURCES``, or a test
    source): the run is refused when the conditioner reports another one, so pinned features are
    never passed off as contract features or the other way round.
    ``keep_tensors`` keeps a CPU copy of every stage output, every step's velocity and latent
    and the layout index tensors (for row localization); ``capture(step, module)`` keeps
    transformer module outputs (default: all of them when ``keep_tensors``).
    """

    device = torch.device(components.device)
    rec = ChainRecorder(keep_tensors=keep_tensors, inject=inject, text_source=text_source)
    if isinstance(request, FL2VARequest):
        variant = "FL2VA"
        text, layout, conditions, layout_inputs = _fl2va_front(request, components, rec)
    elif isinstance(request, Ref2VARequest):
        variant = "Ref2VA"
        text, layout, conditions, layout_inputs = _ref2va_front(request, components, rec)
    else:
        raise TypeError(f"unsupported request type {type(request).__name__}")
    denoiser = components.denoiser
    # H16: an FL2VA layout never meets transformer_ref weights, nor Ref2VA transformer/ ones.
    pl.check_pipeline_identity(layout, transformer_variant=denoiser.variant)

    layout_inputs = dict(layout_inputs)
    layout_inputs["prompt_embeds"] = tensor_digest(text.prompt_embeds)
    layout_inputs["text_token_tags"] = tensor_digest(text.text_token_tags)
    layout_art = pl.layout_artifact(layout, inputs=layout_inputs)
    rec.record(
        "packed_layout",
        [("layout_artifact", layout_art["artifact_sha256"])]
        + [(name, t) for name, t in layout.tensors().items()],
        {
            "schema": pl.SCHEMA_VERSION,
            "workflow": layout.workflow,
            "variant": layout.variant,
            "sequence_length": layout.sequence_length,
        },
    )
    if keep_tensors:
        for name, t in layout.tensors().items():
            rec.tensors[f"layout/{name}"] = t.clone()

    generator = torch.Generator("cpu").manual_seed(int(request.seed))
    if variant == "FL2VA":
        from rl_engine.models.minimax_h3 import fl2va

        video_rows, audio_rows = fl2va.fl2va_initial_rows(
            layout, [z for _, z in conditions], generator=generator, device=device
        )
    else:
        from rl_engine.models.minimax_h3 import ref2va

        video_rows, audio_rows = ref2va.ref2va_initial_rows(
            layout,
            [z for kind, z in conditions if kind == "video"],
            [r for kind, r in conditions if kind == "audio"],
            generator=generator,
            device=device,
        )
    video_rows = rec.apply("initial_rows", "video_rows", video_rows, 0)
    audio_rows = rec.apply("initial_rows", "audio_rows", audio_rows, 0)
    rec.record(
        "initial_rows",
        [("video_rows", video_rows), ("audio_rows", audio_rows)],
        {
            "generator": "cpu",
            "seed": int(request.seed),
            "draw_device": "cpu",
            "device": str(device),
        },
    )

    schedule = dual_schedule(request.num_inference_steps)
    plan = pl.row_timestep_plan(
        layout, schedule.video_timesteps.tolist(), schedule.audio_timesteps.tolist()
    )
    plan_digest = _sha([[tensor_digest(t), tensor_digest(i)] for t, i in plan])
    rec.record(
        "schedule",
        [
            ("video_sigmas", schedule.video_sigmas),
            ("audio_sigmas", schedule.audio_sigmas),
            ("video_timesteps", schedule.video_timesteps),
            ("audio_timesteps", schedule.audio_timesteps),
            ("row_timestep_plan", plan_digest),
        ],
        {
            "operator": "h3_sigma_schedule (NativeH3SigmaScheduleOp)",
            "shifts": {"video": schedule.video_shift, "audio": schedule.audio_shift},
            "num_inference_steps": schedule.num_inference_steps,
            "model_evaluations": schedule.steps,
            "condition_noise_level": pl.CONDITION_NOISE_LEVEL,
        },
    )

    # --- denoise --------------------------------------------------------------------------
    nv, na = layout.num_condition_video_rows, layout.num_condition_audio_rows
    structural = layout.denoiser_inputs(device)
    embeds = text.prompt_embeds.to(device)
    v_sigmas, a_sigmas = schedule.video_sigmas.to(device), schedule.audio_sigmas.to(device)
    v_times, a_times = schedule.video_timesteps.to(device), schedule.audio_timesteps.to(device)
    recorder = tj.TrajectoryRecorder(
        variant=variant,
        label=label,
        seed=int(request.seed),
        sigma_grids={
            "video": schedule.video_sigmas.tolist(),
            "audio": schedule.audio_sigmas.tolist(),
        },
        initial_noise={"video": video_rows, "audio": audio_rows},
        shifts={"video": schedule.video_shift, "audio": schedule.audio_shift},
        metadata={"sampler": "MiniMaxH3Scheduler euler eta=0", "chain": CONTRACT},
        keep_tensors=keep_tensors,
    )
    if capture is None and keep_tensors:
        capture = lambda step, path: True  # noqa: E731
    probe_obj = TransformerProbe(denoiser.model, capture=capture, inject=inject) if probe else None
    probe_scope = probe_obj.attached() if probe_obj is not None else contextlib.nullcontext()
    video, audio = video_rows.clone(), audio_rows.clone()
    velocity_injected = False
    with torch.no_grad(), denoiser.session(), probe_scope:
        for i in range(schedule.steps):
            if probe_obj is not None:
                probe_obj.begin_step(i)
            unique_t, indices = (t.to(device) for t in plan[i])
            v_vel, a_vel = denoiser(
                video_rows=video,
                audio_rows=audio,
                prompt_embeds=embeds,
                timesteps=unique_t,
                timestep_indices=indices,
                layout_inputs=structural,
            )
            v_vel, a_vel, hit = _inject_velocity(inject, i, v_vel, a_vel)
            velocity_injected = velocity_injected or hit
            video[nv:] = euler_step(
                v_vel[0, nv:].float(), v_times[i], video[nv:], *v_sigmas[i : i + 2]
            )
            audio[na:] = euler_step(
                a_vel[0, na:].float(), a_times[i], audio[na:], *a_sigmas[i : i + 2]
            )
            recorder.record_step(
                i,
                velocity={"video": v_vel[0], "audio": a_vel[0]},
                latent={"video": video, "audio": audio},
            )
    trajectory = recorder.seal()
    probe_record = probe_obj.record() if probe_obj is not None else None
    rec.record(
        "denoise",
        [
            ("trajectory", trajectory_content_sha256(trajectory)),
            ("transformer_probe", _sha(probe_record) if probe_record is not None else "none"),
            ("final_video_rows", video),
            ("final_audio_rows", audio),
        ],
        {
            "transformer": denoiser.identity(),
            "execution": denoiser.report(),
            "sampler": "MiniMaxH3Scheduler.step (diffusers f53d552), op for op",
            "probe": probe_record is not None,
        },
    )
    if keep_tensors:
        for key, value in recorder.tensors.items():
            rec.tensors[f"trajectory/{key}"] = value
        if probe_obj is not None:
            rec.tensors.update(probe_obj.tensors)
    elif probe_obj is not None:
        rec.tensors.update(probe_obj.tensors)

    # --- unpack and decode ----------------------------------------------------------------
    geometry = layout.geometry
    video_latents = _unpack_video(video[nv:].detach().cpu(), geometry)
    audio_latents = _unpack_audio(audio[na:].detach().cpu(), geometry["num_audio_latents"])
    video_latents = rec.apply("unpack", "video_latents", video_latents, 2)
    audio_latents = rec.apply("unpack", "audio_latents", audio_latents, 2)
    rec.record(
        "unpack",
        [("video_latents", video_latents), ("audio_latents", audio_latents)],
        {"video": "(1, 24, T, H, W) normalized", "audio": "(2, 32, T) normalized, (left, right)"},
    )
    secondary: dict[str, Any] = {}
    decoded_video = decoded_audio = None
    if decode and components.visual_vae is not None:
        out = components.visual_vae.decode(video_latents)
        decoded_video = rec.apply("video_decode", "video", out.tensor, 2)
        rec.record("video_decode", [("video", decoded_video)], dict(out.record))
        secondary["video"] = video_metrics(decoded_video)
    else:
        rec.record("video_decode", [], _skip("decode disabled or no visual VAE"))
    if decode and components.audio_vae is not None:
        target_rows = _audio_rows(audio_latents).float()  # what ``unpack`` recorded
        dec = components.audio_vae.decode_rows(target_rows, geometry["num_audio_latents"])
        decoded_audio = rec.apply("audio_decode", "waveform", dec.waveform, 0)
        rec.record("audio_decode", [("waveform", decoded_audio)], components.audio_vae.provenance())
        secondary["audio"] = audio_metrics(decoded_audio)
    else:
        rec.record("audio_decode", [], _skip("decode disabled or no audio VAE"))

    probe_injected = probe_obj is not None and probe_obj.injected
    if inject is not None and not (rec.injected or probe_injected or velocity_injected):
        raise InjectionNotRun(f"the injection matched nothing in this run: {inject.to_dict()}")

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "architecture_fingerprint_sha256": fp.build_fingerprint()["fingerprint_sha256"],
        "checkpoint": {"repo": fp.REPO, "revision": fp.REVISION},
        "variant": variant,
        "workflow": request.workflow,
        "label": label,
        "request": _jsonable(request.spec()),
        "stages": rec.records,
        "layout": layout_art,
        "trajectory": trajectory,
        "transformer_probe": probe_record,
        "secondary_metrics": _jsonable(secondary),
        "gate_policy": {
            "hard": list(GATED_STAGES),
            "secondary": list(SECONDARY_STAGES),
            "rule": "latent trajectory equality is the gate; decoded media are recorded only",
        },
        "injection": inject.to_dict() if inject is not None else None,
        "runtime": runtime_provenance(device),
        "metadata": _jsonable(dict(metadata or {})),
    }
    artifact["artifact_sha256"] = _sha(artifact)
    verify(artifact)
    return ChainRun(
        artifact, rec.tensors, video_latents, audio_latents, decoded_video, decoded_audio
    )


def _inject_velocity(
    inject: DriftInjection | None, step: int, video: torch.Tensor, audio: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """The two velocities, and whether the injection perturbed one of them."""

    if inject is None or inject.stage != "denoise" or inject.step != step:
        return video, audio, False
    if inject.component == "velocity.video":
        return bump_one_ulp(video, inject.row, 1), audio, True
    if inject.component == "velocity.audio":
        return video, bump_one_ulp(audio, inject.row, 1), True
    return video, audio, False


# --- validation ---------------------------------------------------------------------------

# What a trajectory *computed*; its label and free-form metadata name the run and stay out.
_TRAJECTORY_CONTENT = (
    "variant",
    "seed",
    "steps",
    "modalities",
    "shifts",
    "sigma_grids",
    "initial_noise",
    "step_records",
    "final_latent_sha256",
    "architecture_fingerprint_sha256",
)


def trajectory_content_sha256(trajectory: Mapping[str, Any]) -> str:
    """Digest of a trajectory artifact's data (inputs, every step), without label or metadata."""

    return _sha({key: trajectory[key] for key in _TRAJECTORY_CONTENT})


def verify(artifact: Mapping[str, Any], *, check_fingerprint: bool = True) -> None:
    """Fail closed unless the artifact is sealed, chained and internally consistent."""

    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported full-chain schema {artifact.get('schema_version')!r}")
    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if artifact.get("artifact_sha256") != _sha(body):
        raise ValueError("artifact_sha256 does not match the content")
    stages = artifact["stages"]
    if [s["stage"] for s in stages] != list(STAGES):
        raise ValueError("stage records are missing, extra or out of order")
    previous = GENESIS
    for record in stages:
        inner = {k: v for k, v in record.items() if k != "chain_sha256"}
        if record["chain_sha256"] != _sha({"previous": previous, "stage": inner}):
            raise ValueError(f"{record['stage']}: hash chain broken")
        previous = record["chain_sha256"]
        names = [name for name, _ in record["components"]]
        if len(set(names)) != len(names):  # the recorder refuses them too
            raise ValueError(f"{record['stage']}: duplicate component names")
    tj.verify(artifact["trajectory"], check_fingerprint=check_fingerprint)
    pl.verify_artifact(artifact["layout"])
    components = {s["stage"]: dict(s["components"]) for s in stages}
    if components["denoise"]["trajectory"] != trajectory_content_sha256(artifact["trajectory"]):
        raise ValueError("denoise stage does not reference the embedded trajectory")
    probe = artifact["transformer_probe"]
    if components["denoise"]["transformer_probe"] != (_sha(probe) if probe is not None else "none"):
        raise ValueError("denoise stage does not reference the embedded transformer probe")
    if components["packed_layout"]["layout_artifact"] != artifact["layout"]["artifact_sha256"]:
        raise ValueError("packed_layout stage does not reference the embedded layout artifact")
    if check_fingerprint and (
        artifact["architecture_fingerprint_sha256"] != fp.build_fingerprint()["fingerprint_sha256"]
    ):
        raise ValueError("artifact was recorded against a different H3 architecture fingerprint")


def dumps(artifact: Mapping[str, Any]) -> str:
    verify(artifact)
    return json.dumps(artifact, indent=1, sort_keys=True, allow_nan=False) + "\n"


def loads(text: str, *, check_fingerprint: bool = True) -> dict[str, Any]:
    artifact = json.loads(text)
    verify(artifact, check_fingerprint=check_fingerprint)
    return artifact


# --- first drift --------------------------------------------------------------------------


TensorSource: TypeAlias = Mapping[str, torch.Tensor] | str | Path | None


def _load_tensors(source: TensorSource) -> Mapping[str, torch.Tensor] | None:
    if source is None or isinstance(source, Mapping):
        return source
    from safetensors.torch import load_file

    return load_file(str(source))


def _first_row_difference(a: torch.Tensor, b: torch.Tensor, row_dim: int) -> dict[str, Any]:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"shape_or_dtype_differs": True, "shapes": [list(a.shape), list(b.shape)]}
    ia, ib = a.contiguous().view(_INT_VIEW.get(a.dtype, a.dtype)), b.contiguous().view(
        _INT_VIEW.get(b.dtype, b.dtype)
    )
    differs = ia != ib
    rows = differs.movedim(row_dim, 0).reshape(differs.shape[row_dim], -1).any(dim=1)
    count = int(differs.sum())
    diff = (a.double() - b.double()).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    out: dict[str, Any] = {
        "mismatch_count": count,
        "numel": int(a.numel()),
        "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
        "differing_rows": int(rows.sum()),
    }
    if bool(rows.any()):
        out["first_row"] = int(torch.nonzero(rows)[0, 0])
        out["first_index"] = [int(i) for i in torch.nonzero(differs)[0].tolist()]
    return out


def row_identity(
    tensors: Mapping[str, torch.Tensor], layout: Mapping[str, Any], space: str, row: int
) -> dict[str, Any]:
    """Packed identity of row ``row`` of a tensor in row space ``space``.

    ``tensors`` must hold the ``layout/*`` index tensors a ``keep_tensors`` run saves;
    ``layout`` is the run's embedded packed-layout artifact.
    """

    if space == "timestep":
        return {"space": space, "local_row": row, "role": "distinct timestep"}
    tags = tensors["layout/token_tags"]
    positions = tensors["layout/position_ids"]
    packed = row if space == "packed" else int(tensors[f"layout/{space}_indices"][row])
    modality = _TAG_NAMES[int(tags[packed])]
    out: dict[str, Any] = {
        "space": space,
        "local_row": row,
        "packed_row": packed,
        "modality": modality,
        "position_thw": [float(x) for x in positions[packed].tolist()],
    }
    geometry = layout["geometry"]
    video = tensors["layout/video_indices"]
    audio = tensors["layout/audio_indices"]
    text = tensors["layout/text_indices"]
    if bool((text == packed).any()):
        out["role"] = "vision block (text side)" if modality == "video" else "text"
        out["text_row"] = packed
        return out
    if modality == "video":
        k = int(torch.nonzero(video == packed)[0, 0])
        ncv = layout["num_condition_video_rows"]
        if k < ncv:
            out.update(role="condition", condition_row=k)
        else:
            j = k - ncv
            per_frame = (geometry["latent_height"] // pl.PATCH_SIZE[1]) * (
                geometry["latent_width"] // pl.PATCH_SIZE[2]
            )
            patch_row, patch_col = divmod(
                j % per_frame, geometry["latent_width"] // pl.PATCH_SIZE[2]
            )
            out.update(
                role="target",
                latent_frame=j // per_frame,
                patch_row=patch_row,
                patch_col=patch_col,
            )
    else:
        k = int(torch.nonzero(audio == packed)[0, 0])
        nca = layout["num_condition_audio_rows"]
        if k < nca:
            out.update(role="reference", reference_row=k)
        else:
            j = k - nca
            n = geometry["num_audio_latents"]
            out.update(role="target", stereo_channel=("left", "right")[j // n], latent_index=j % n)
    return out


# Row spaces of stage components and trajectory tensors (name -> (space, row dim)).
_COMPONENT_ROWS = {
    ("text_conditions", "prompt_embeds"): ("text", 1),
    ("initial_rows", "video_rows"): ("video", 0),
    ("initial_rows", "audio_rows"): ("audio", 0),
    ("denoise", "final_video_rows"): ("video", 0),
    ("denoise", "final_audio_rows"): ("audio", 0),
}


def _recorded_digest(artifact: Mapping[str, Any], key: str) -> str | None:
    """The digest ``artifact`` records for the retained tensor ``key``, if it records one."""

    head, _, rest = key.partition("/")
    if head == "probe":
        step, _, name = rest.partition("/")
        probe = artifact.get("transformer_probe")
        steps = probe["steps"] if probe is not None else []
        return dict(steps[int(step)]).get(name) if int(step) < len(steps) else None
    if head == "trajectory":
        step, kind, modality = rest.split(".")
        records = artifact["trajectory"]["step_records"]
        if int(step) >= len(records):
            return None
        return records[int(step)][f"{kind}_sha256"].get(modality)
    for record in artifact["stages"]:
        if record["stage"] == head:
            return dict(record["components"]).get(rest)
    return None


def _locate(
    key: str,
    space_dim: tuple[str, int] | None,
    tensors: tuple[Mapping[str, torch.Tensor] | None, Mapping[str, torch.Tensor] | None],
    artifacts: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    lhs, rhs = tensors
    if lhs is None or rhs is None or key not in lhs or key not in rhs:
        return {}
    for side, kept, artifact in zip(("lhs", "rhs"), tensors, artifacts, strict=True):
        # a sidecar of another run would locate rows this pair of artifacts did not record
        if tensor_digest(kept[key]) != _recorded_digest(artifact, key):  # type: ignore[index]
            return {"row_evidence": f"unavailable: the {side} tensor is not the recorded one"}
    layout = artifacts[0]["layout"]
    space, dim = space_dim if space_dim is not None else ("flat", 0)
    found = _first_row_difference(lhs[key], rhs[key], dim)
    if "first_row" in found and space != "flat" and "layout/token_tags" in lhs:
        # the identity is read from the sidecar's layout tensors: they must be the recorded ones
        foreign = [
            name
            for name, digest in layout["layout_component_sha256"].items()
            if f"layout/{name}" not in lhs or tensor_digest(lhs[f"layout/{name}"]) != digest
        ]
        if foreign:
            found["row_identity"] = (
                f"unavailable: layout tensors {foreign} are not the recorded ones"
            )
        else:
            found["row"] = row_identity(lhs, layout, space, found["first_row"])
    return found


def compare(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    *,
    tensors_lhs: TensorSource = None,
    tensors_rhs: TensorSource = None,
    allow_text_source_change: bool = False,
) -> dict[str, Any]:
    """Stage-by-stage comparison of two sealed runs of the same request.

    Two runs whose text conditions come from different sources (``text_source``) are refused
    unless ``allow_text_source_change``: replaying a fixture with the other source is not a
    replay. Allowed, the comparison localizes the difference the source makes.

    ``first_drift`` is the first differing (stage, component), refined inside ``denoise`` to
    the step, kind, modality and transformer module (and row, with tensors). ``gate`` holds
    the hard latent verdict and, separately, the secondary decoded-media verdict.
    """

    verify(lhs)
    verify(rhs)
    if lhs["request"] != rhs["request"] or lhs["variant"] != rhs["variant"]:
        raise ValueError("the two runs are of different requests or variants")
    sources = (text_source(lhs), text_source(rhs))
    if sources[0] != sources[1] and not allow_text_source_change:
        raise TextSourceMismatch(
            f"text sources differ ({sources[0]!r} vs {sources[1]!r}); pass "
            "allow_text_source_change=True to compare them on purpose"
        )
    ta, tb = _load_tensors(tensors_lhs), _load_tensors(tensors_rhs)
    stages: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for a, b in zip(lhs["stages"], rhs["stages"], strict=True):
        ca, cb = dict(a["components"]), dict(b["components"])
        names = [n for n, _ in a["components"]] + [n for n, _ in b["components"] if n not in ca]
        differing = [n for n in names if ca.get(n) != cb.get(n)]
        stages.append(
            {
                "stage": a["stage"],
                "equal": not differing,
                "differing_components": differing,
                "provenance_equal": a["provenance"] == b["provenance"],
            }
        )
        if differing and first is None:
            first = {"stage": a["stage"], "component": differing[0]}
            if a["stage"] == "denoise":
                first.update(_denoise_drift(lhs, rhs, ta, tb))
            else:
                key = f"{a['stage']}/{differing[0]}"
                space = _COMPONENT_ROWS.get((a["stage"], differing[0]))
                first.update(_locate(key, space, (ta, tb), (lhs, rhs)))
    by_stage = {s["stage"]: s["equal"] for s in stages}
    # The gate is the latent data; transformer intermediates are localization evidence.
    gated = {
        s["stage"]: [c for c in s["differing_components"] if c != "transformer_probe"]
        for s in stages
    }
    hard = not any(gated[s] for s in GATED_STAGES)
    decoded = {s: by_stage[s] for s in SECONDARY_STAGES}
    secondary_metrics = {
        "lhs": lhs["secondary_metrics"],
        "rhs": rhs["secondary_metrics"],
    }
    if ta is not None and tb is not None:
        secondary_metrics.update(_media_closeness(ta, tb))
    return {
        "labels": [lhs["label"], rhs["label"]],
        "variant": lhs["variant"],
        "workflow": lhs["workflow"],
        "first_drift": first,
        "text_sources": list(sources),
        "stages": stages,
        "trajectory": _trajectory_report(lhs, rhs),
        "gate": {
            "hard_gate": "pass" if hard else "fail",
            "latent_trajectory_equal": not gated["denoise"] and not gated["unpack"],
            "transformer_intermediates_equal": _intermediates_equal(lhs, rhs),
            "decoded_media_equal": decoded,
            "rule": "latent trajectory equality is the gate; decoded media are secondary",
        },
        "secondary_metrics": secondary_metrics,
        "runtime_equal": lhs["runtime"] == rhs["runtime"],
    }


def _trajectory_report(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, Any]:
    problems = tj.input_mismatches(lhs["trajectory"], rhs["trajectory"])
    if problems:
        return {"comparable": False, "reason": "; ".join(problems)}
    report = tj.drift_report(lhs["trajectory"], rhs["trajectory"])
    report.pop("entries")
    return report


def text_source(artifact: Mapping[str, Any]) -> str | None:
    """Where an artifact's text conditions came from (its ``text_conditions`` provenance)."""

    for record in artifact["stages"]:
        if record["stage"] == "text_conditions":
            return record["provenance"].get("source")
    return None


def first_drift(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    *,
    tensors_lhs: TensorSource = None,
    tensors_rhs: TensorSource = None,
    allow_text_source_change: bool = False,
) -> dict[str, Any] | None:
    """The first drift of :func:`compare`, or ``None`` when every stage is byte-equal."""

    return compare(
        lhs,
        rhs,
        tensors_lhs=tensors_lhs,
        tensors_rhs=tensors_rhs,
        allow_text_source_change=allow_text_source_change,
    )["first_drift"]


def gate(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, Any]:
    """Hard latent gate plus secondary decoded-media verdict, without tensors."""

    report = compare(lhs, rhs)
    return {**report["gate"], "first_drift": report["first_drift"]}


def _probes_recorded(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    """Whether both runs recorded the transformer probe (``probe=False`` records none)."""

    return lhs.get("transformer_probe") is not None and rhs.get("transformer_probe") is not None


def _probes_observed(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    """Whether both probes hold an output for every step: only then can "equal" be claimed."""

    if not _probes_recorded(lhs, rhs):
        return False
    steps = [artifact["transformer_probe"]["steps"] for artifact in (lhs, rhs)]
    return all(bool(s) and all(s) for s in steps)


def _probe_drift(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    ta: Mapping[str, torch.Tensor] | None,
    tb: Mapping[str, torch.Tensor] | None,
) -> dict[str, Any] | None:
    """First (step, module) whose output differs, in execution order, with its first row.

    A call or a step that only one run made differs too. ``None`` means equal only when both
    runs observed every step (:func:`_probes_observed`).
    """

    if not _probes_recorded(lhs, rhs):
        return None
    pa, pb = lhs["transformer_probe"], rhs["transformer_probe"]
    for step, (sa, sb) in enumerate(itertools.zip_longest(pa["steps"], pb["steps"])):
        if sa is None or sb is None:
            return {"step": step, "module": None, "note": "only one run recorded this step"}
        for index, (ea, eb) in enumerate(itertools.zip_longest(sa, sb)):
            if ea == eb:
                continue
            if ea is None or eb is None:
                return {
                    "step": step,
                    "module": (ea or eb)[0],
                    "execution_index": index,
                    "note": f"only the {'lhs' if eb is None else 'rhs'} run made this call",
                }
            if ea[0] != eb[0]:
                return {"step": step, "module": f"{ea[0]} | {eb[0]}", "execution_index": index}
            found: dict[str, Any] = {"step": step, "module": ea[0], "execution_index": index}
            found.update(
                _locate(f"probe/{step}/{ea[0]}", module_row_space(ea[0]), (ta, tb), (lhs, rhs))
            )
            return found
    return None


def _intermediates_equal(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool | None:
    """Whether every probed output is equal; ``None`` when that was not compared."""

    if not _probes_recorded(lhs, rhs):
        return None
    if lhs["transformer_probe"] != rhs["transformer_probe"]:
        return False
    return True if _probes_observed(lhs, rhs) else None


def _denoise_drift(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    ta: Mapping[str, torch.Tensor] | None,
    tb: Mapping[str, torch.Tensor] | None,
) -> dict[str, Any]:
    """Step, transformer module and row of the first drift inside the sampling loop.

    ``trajectory`` is the first differing velocity or latent (velocity before latent, video
    before audio); ``transformer`` is the first differing probed module output. The module
    can drift at an earlier step than the trajectory, or without the trajectory drifting at
    all, when the difference is rounded away before the velocity heads. ``step`` is the
    earlier of the two.
    """

    out: dict[str, Any] = {}
    drift = tj.first_drift(lhs["trajectory"], rhs["trajectory"], require_same_inputs=False)
    if drift is not None:
        traj: dict[str, Any] = dict(drift)
        space = (drift["modality"], 0)  # velocities are recorded as v[0]: rows on dim 0
        key = f"trajectory/{tj.tensor_key(drift['step'], drift['kind'], drift['modality'])}"
        traj.update(_locate(key, space, (ta, tb), (lhs, rhs)))
        if drift["kind"] == "latent":
            traj["note"] = "velocities equal, latents differ: scheduler arithmetic"
        out["trajectory"] = traj
    else:
        out["trajectory"] = None
    probe = _probe_drift(lhs, rhs, ta, tb)
    if probe is not None and (drift is None or probe["step"] <= drift["step"]):
        out["transformer"] = {k: v for k, v in probe.items() if k != "step"}
        out["step"] = probe["step"]
        if drift is None:
            out["note"] = (
                "a transformer intermediate differs but every velocity and latent is byte-equal: "
                "the difference was rounded away before the output heads"
            )
    elif drift is not None:
        out["step"] = drift["step"]
        if not _probes_observed(lhs, rhs):
            out["transformer"] = {
                "module": None,
                "note": "not compared: at least one run observed no module output at a step",
            }
        elif drift["kind"] == "velocity":
            out["transformer"] = {
                "module": None,
                "note": "every probed module output is equal at this step: the drift is after "
                "the heads (row selection) or outside the probed modules",
            }
    if drift is not None and drift["step"] == out["step"]:  # else only under ``trajectory``
        out["kind"], out["modality"] = drift["kind"], drift["modality"]
    return out


def _media_closeness(ta: Mapping[str, torch.Tensor], tb: Mapping[str, torch.Tensor]) -> dict:
    out: dict[str, Any] = {}
    for key, name in (
        ("video_decode/video", "video_psnr_db"),
        ("audio_decode/waveform", "audio_snr_db"),
    ):
        if key in ta and key in tb and ta[key].shape == tb[key].shape:
            a, b = ta[key].double(), tb[key].double()
            mse = float((a - b).pow(2).mean())
            signal = 1.0 if name == "video_psnr_db" else float(a.pow(2).mean())
            # an undefined value is null with the reason beside it: strict JSON has no Infinity
            if mse == 0:
                out[name], out[f"{name}_note"] = None, "identical"
            elif not (math.isfinite(mse) and math.isfinite(signal)):
                out[name], out[f"{name}_note"] = None, "non-finite samples"
            elif signal == 0:
                out[name], out[f"{name}_note"] = None, "silent reference"
            else:
                out[name] = float(10 * np.log10(signal / mse))
    return out
