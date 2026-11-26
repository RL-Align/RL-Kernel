# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #738 Phase B ``phase_b_image_full_chain``: MiniMax-H3-Image from request to output bytes.

Each H3-Image row pins one boundary of the released pipeline. This module chains them into
one run, T2I or edit, and seals what every stage produced:

``request -> media_preprocess -> text_conditions -> reference_encode -> reference_conditioning
-> packed_layout -> schedule -> denoise -> unpack -> decode -> resize``

* ``media_preprocess``: the canvas plan (``image_canvas_contract``) and, for an edit, each
  reference read upright, converted to RGB and resized once for the VAE (short edge in
  ``[1024, 2048]``) and once for the vision tower (short edge 1024), both sides rounded to
  32 (``MiniMaxH3ImageEditResizeStep``; PIL LANCZOS on uint8, ``reducing_gap=None``).
* ``text_conditions``: ``image_text_presentation_contract`` (presentation, layer-50 feature).
* ``reference_encode``: ``image_vae_contract`` (posterior sample, FP16 round trip, crop,
  normalization) drawing from the request generator.
* ``reference_conditioning``: generated-image noise, then reference noise and the ``t = 0.999``
  mix (``image_reference_conditioning``); the draw ledger is checked against the released order.
* ``packed_layout``: ``image_packed_layout_schema`` over ``image_position_grid``.
* ``schedule``: the released seven-point sigma grid through its FP32 round trip, and the
  per-row timestep table of every step (references at ``0.999``), deduplicated as released.
* ``denoise``: six transformer calls, PDD head ``i`` at step ``i``, behind
  :class:`ImageDenoiser` (today the pinned Diffusers ``MiniMaxH3ImageTransformer2DModel``;
  the strict RL-Kernel transformer plugs in when the WS1 image rows land), and the released
  ``MiniMaxH3Scheduler`` step on the generated rows only. Every step goes into the
  ``h3_image_evidence_schema`` trajectory, and the reference rows are checked bit for bit.
* ``unpack``, ``decode``, ``resize``: ``image_patch_pack_unpack``, the VAE decode to the
  canvas-size uint8 image, the canvas resize back to the requested size.

Every stage records the sha256 (dtype, shape, bytes) of each tensor it hands on, and the
stage records are hash-chained into a sealed artifact. :func:`compare` checks two runs
stage by stage, names the first stage and tensor that differ and, inside ``denoise``, the
first step and kind (via the trajectory's ``first_drift``). :func:`gate` passes only when
every stage up to ``unpack`` is byte-equal; ``decode`` and ``resize`` are secondary evidence.

:func:`run_diffusers` runs the pinned Diffusers modular pipeline (``MiniMaxH3ImageBlocks`` at
875373ac) block by block, unmodified, records the same boundaries and seals them the same
way, so the two artifacts compare field for field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from rl_engine.models.minimax_h3.trajectory import _chain, tensor_digest
from rl_engine.models.minimax_h3_image import canvas as cv
from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import patch_pack as pp
from rl_engine.models.minimax_h3_image import pdd_schedule as ps
from rl_engine.models.minimax_h3_image import reference_conditioning as rc
from rl_engine.models.minimax_h3_image import trajectory as tj
from rl_engine.models.minimax_h3_image import velocity_step as vs

SCHEMA_VERSION = "rlkernel.h3_image.full_chain.v1"
CONTRACT_ID = "h3-image-phase-b-full-chain-v1"
STAGES = (
    "request",
    "media_preprocess",
    "text_conditions",
    "reference_encode",
    "reference_conditioning",
    "packed_layout",
    "schedule",
    "denoise",
    "unpack",
    "decode",
    "resize",
)
SECONDARY_STAGES = ("decode", "resize")
GATED_STAGES = tuple(s for s in STAGES if s not in SECONDARY_STAGES)
REFERENCE_SHORT_EDGE = (1024, 2048)  # reference_min_short_edge, reference_max_short_edge
VISION_SHORT_EDGE = (1024, 1024)  # vision_min_short_edge, vision_max_short_edge
GENESIS = tj.GENESIS


# --- request and preprocessing ---------------------------------------------------------------


@dataclass(frozen=True)
class ImageRequest:
    """One T2I or edit request, as the pipeline is called with it."""

    prompt: str
    height: int = cv.DEFAULT_SIZE[0]
    width: int = cv.DEFAULT_SIZE[1]
    seed: int = 0
    references: tuple[Any, ...] = ()  # PIL images, in reference order (edit)
    min_generation_area: int = cv.DEFAULT_MIN_GENERATION_AREA
    generator_device: str = "cpu"

    @property
    def task(self) -> str:
        return "edit" if self.references else "t2i"

    def generator(self) -> torch.Generator:
        return torch.Generator(self.generator_device).manual_seed(int(self.seed))


def image_digest(image: Any) -> str:
    """sha256 over the mode, size and raw bytes of a PIL image."""

    header = f"{image.mode}|{image.size[0]}x{image.size[1]}|".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def bound_short_edge(
    width: int, height: int, min_short_edge: int, max_short_edge: int, multiple: int
) -> tuple[int, int]:
    """``(width, height)`` a reference is resized to for one consumer (as released)."""

    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return width, height
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    return (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )


def _resize(image: Any, bounds: tuple[int, int]) -> Any:
    from PIL import Image

    width, height = image.size
    size = bound_short_edge(width, height, bounds[0], bounds[1], fp.CANVAS_MULTIPLE)
    if image.size == size:
        return image
    return image.resize(size, resample=Image.Resampling.LANCZOS, reducing_gap=None)


@dataclass(frozen=True)
class PreparedReference:
    source_sha256: str
    vae_image: Any  # PIL RGB
    vision_image: Any  # PIL RGB

    def summary(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "vae_image": [self.vae_image.size[1], self.vae_image.size[0]],
            "vae_image_sha256": image_digest(self.vae_image),
            "vision_image": [self.vision_image.size[1], self.vision_image.size[0]],
            "vision_image_sha256": image_digest(self.vision_image),
        }


def prepare_reference(image: Any) -> PreparedReference:
    """EXIF upright, RGB, then the VAE and vision resizes (``MiniMaxH3ImageEditResizeStep``)."""

    from PIL import ImageOps

    source = image_digest(image)
    upright = ImageOps.exif_transpose(image).convert("RGB")
    if min(upright.size) < fp.CANVAS_MULTIPLE:
        raise ValueError(
            f"a reference needs at least {fp.CANVAS_MULTIPLE} pixels on each side, got "
            f"{upright.size}"
        )
    return PreparedReference(
        source, _resize(upright, REFERENCE_SHORT_EDGE), _resize(upright, VISION_SHORT_EDGE)
    )


def pil_to_uint8(image: Any) -> torch.Tensor:
    """``(3, H, W)`` uint8, as the VAE-encode step reads a PIL image."""

    return torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()


# --- stage log -------------------------------------------------------------------------------


def _digest(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    if isinstance(value, (list, tuple)):
        return [_digest(v) for v in value]
    return value


class StageLog:
    """Hash-chained per-stage records; tensors are stored as digests (optionally kept)."""

    def __init__(self, keep_tensors: bool = False) -> None:
        self.records: list[dict[str, Any]] = []
        self.keep_tensors = keep_tensors
        self.tensors: dict[str, torch.Tensor] = {}

    def add(
        self,
        stage: str,
        tensors: Mapping[str, Any],
        meta: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        expected = STAGES[len(self.records)] if len(self.records) < len(STAGES) else None
        if stage != expected:
            raise ValueError(f"stage {stage!r} out of order; expected {expected!r}")
        body = {
            "stage": stage,
            "tensors": {k: _digest(v) for k, v in tensors.items()},
            "meta": dict(meta or {}),
            "provenance": dict(provenance or {}),
        }
        previous = self.records[-1]["chain_sha256"] if self.records else GENESIS
        body["chain_sha256"] = _chain(previous, body)
        self.records.append(body)
        if self.keep_tensors:
            for k, v in tensors.items():
                if isinstance(v, torch.Tensor):
                    self.tensors[f"{stage}.{k}"] = v.detach().cpu().clone()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seal(
    label: str,
    request: ImageRequest,
    log: StageLog,
    trajectory: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if [r["stage"] for r in log.records] != list(STAGES):
        raise ValueError("every stage must be recorded once, in order")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "label": label,
        "task": request.task,
        "architecture_fingerprint_sha256": fp.build_fingerprint()["fingerprint_sha256"],
        "stages": log.records,
        "trajectory": dict(trajectory),
        "provenance": dict(provenance),
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    return artifact


def verify(artifact: Mapping[str, Any]) -> None:
    """Raise unless the seal, the stage chain and the embedded trajectory all verify."""

    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if _sha256_json(body) != artifact.get("artifact_sha256"):
        raise ValueError("artifact seal does not verify")
    previous = GENESIS
    for record, stage in zip(artifact["stages"], STAGES, strict=True):
        if record["stage"] != stage:
            raise ValueError(f"stage {record['stage']} out of order")
        inner = {k: v for k, v in record.items() if k != "chain_sha256"}
        if _chain(previous, inner) != record["chain_sha256"]:
            raise ValueError(f"stage {stage} chain does not verify")
        previous = record["chain_sha256"]
    tj.verify(artifact["trajectory"])


# --- schedule --------------------------------------------------------------------------------


# --- denoiser --------------------------------------------------------------------------------


class ImageDenoiser(Protocol):
    """Velocity of every image row for one step (``(1, N_image, 96)``)."""

    def __call__(
        self,
        image_rows: torch.Tensor,
        prompt_embeds: torch.Tensor,
        layout: pl.PackedLayout,
        unique_timesteps: torch.Tensor,
        timestep_indices: torch.Tensor,
        step: int,
    ) -> torch.Tensor: ...


class DiffusersTransformerDenoiser:
    """The pinned ``MiniMaxH3ImageTransformer2DModel`` called as the released loop calls it."""

    def __init__(self, transformer: Any) -> None:
        self.transformer = transformer

    @property
    def backend_id(self) -> str:
        return f"diffusers:{type(self.transformer).__name__}"

    def __call__(self, image_rows, prompt_embeds, layout, unique_timesteps, timestep_indices, step):
        device = image_rows.device
        return self.transformer(
            hidden_states=image_rows[None],
            encoder_hidden_states=prompt_embeds,
            timestep=unique_timesteps.to(device),
            timestep_indices=timestep_indices.to(device),
            pdd_step_index=step,
            return_dict=False,
            position_ids=layout.position_ids.to(device),
            token_tags=layout.token_tags.to(device),
            image_indices=layout.image_indices.to(device),
            text_indices=layout.text_indices.to(device),
        )[0]


# --- text ------------------------------------------------------------------------------------


@dataclass
class TextConditions:
    """The text stage's output, from the contract or from the pinned Diffusers step."""

    source: str  # "contract" or "diffusers"
    prompt: str
    token_ids: list[int]
    prompt_embeds: torch.Tensor
    text_token_tags: torch.Tensor
    vision_inputs: Mapping[str, torch.Tensor] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def presentation_block(self) -> dict[str, Any]:
        return {
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "token_ids_sha256": tensor_digest(torch.tensor(self.token_ids, dtype=torch.long)),
            "token_tags_sha256": tensor_digest(self.text_token_tags.cpu().long()),
            "num_text_tokens": len(self.token_ids),
        }

    def stage_tensors(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "token_ids": torch.tensor(self.token_ids, dtype=torch.long),
            "text_token_tags": self.text_token_tags.cpu().long(),
            "prompt_embeds": self.prompt_embeds,
        }
        for key in sorted(self.vision_inputs):
            out[f"vision.{key}"] = self.vision_inputs[key]
        return out


def contract_text(
    request: ImageRequest,
    prepared: Sequence[PreparedReference],
    encoder: Any,
    tokenizer: Any,
    processor: Any,
) -> TextConditions:
    """``image_text_presentation_contract`` with the H3 layer-50 encoder."""

    from rl_engine.models.minimax_h3_image import text_presentation as tp

    vision: dict[str, torch.Tensor] = {}
    if request.task == "t2i":
        presentation = tp.present_text2image(tokenizer, request.prompt)
    else:
        presentation, vision = tp.present_edit_from_images(
            tokenizer, processor, request.prompt, [r.vision_image for r in prepared]
        )
    feature = tp.encode(encoder, presentation, vision or None)
    return TextConditions(
        source="contract",
        prompt=request.prompt,
        token_ids=list(presentation.token_ids),
        prompt_embeds=feature.prompt_embeds,
        text_token_tags=feature.text_token_tags,
        vision_inputs=dict(vision),
        provenance={"contract_id": tp.CONTRACT_ID, "feature": feature.output.summary()},
    )


# --- our chain -------------------------------------------------------------------------------


def _request_fields(request: ImageRequest, sources: Sequence[str]) -> dict[str, Any]:
    return {
        "task": request.task,
        "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
        "size": [int(request.height), int(request.width)],
        "seed": int(request.seed),
        "min_generation_area": int(request.min_generation_area),
        "generator_device": request.generator_device,
        "references": list(sources),
    }


def _preprocess_fields(
    canvas: Sequence[int], latent: Sequence[int], prepared: Sequence[PreparedReference]
) -> dict[str, Any]:
    return {
        "canvas": [int(v) for v in canvas],
        "latent": [int(v) for v in latent],
        "reference_vae_images": [image_digest(r.vae_image) for r in prepared],
        "reference_vision_images": [image_digest(r.vision_image) for r in prepared],
    }


def _layout_tensors(layout: pl.PackedLayout) -> dict[str, Any]:
    return {
        "position_ids_fp64": layout.position_ids.to(torch.float64).cpu(),
        "token_tags": layout.token_tags.cpu().long(),
        "image_indices": layout.image_indices.cpu().long(),
        "text_indices": layout.text_indices.cpu().long(),
        "num_reference_rows": int(layout.num_reference_rows),
    }


@dataclass
class ChainRun:
    artifact: dict[str, Any]
    output: torch.Tensor  # uint8 (1, 3, height, width)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


def _vae_block(mean: torch.Tensor, std: torch.Tensor, tiled: bool) -> dict[str, Any]:
    from rl_engine.models.minimax_h3 import visual_vae as vv
    from rl_engine.models.minimax_h3_image import image_vae as iv

    return {
        "latent_stats_sha256": vv.latent_stats_digest(mean.float().cpu(), std.float().cpu()),
        "tile_size": iv.TILE_SIZE if tiled else None,
        "tile_overlap_min": iv.TILE_MIN_OVERLAP if tiled else None,
        "encode_tiled": tiled,
        "decode_tiled": tiled,
    }


def _canvas_block(plan: cv.CanvasPlan) -> dict[str, Any]:
    return {
        "requested": list(plan.requested),
        "canvas": list(plan.canvas),
        "latent": list(plan.latent),
        "min_generation_area": plan.min_generation_area,
        "resize": plan.resize_rule(),
    }


def _layout_block(layout: pl.PackedLayout, conditioned: torch.Tensor | None) -> dict[str, Any]:
    block = {
        "position_sha256_fp64": tensor_digest(layout.position_ids.to(torch.float64).cpu()),
        "position_sha256_fp32": tensor_digest(layout.position_ids.to(torch.float32).cpu()),
        "sequence_length": layout.sequence_length,
        "image_indices_sha256": tensor_digest(layout.image_indices.cpu()),
        "text_indices_sha256": tensor_digest(layout.text_indices.cpu()),
        "num_reference_rows": layout.num_reference_rows,
    }
    if conditioned is not None:
        block["conditioned_reference_rows_sha256"] = tensor_digest(conditioned[None])
    return block


def _trajectory_inputs(
    request: ImageRequest,
    plan: cv.CanvasPlan,
    text: TextConditions,
    references: Sequence[Mapping[str, Any]],
    vae: Mapping[str, Any],
    layout: pl.PackedLayout,
    conditioned: torch.Tensor | None,
    target_noise: torch.Tensor,
    reference_noise: Sequence[str],
) -> dict[str, Any]:
    return {
        "task": request.task,
        "seed": int(request.seed),
        "canvas": _canvas_block(plan),
        "presentation": text.presentation_block(),
        "references": list(references),
        "vae": dict(vae),
        "layout": _layout_block(layout, conditioned),
        "noise": {
            "target_sha256": tensor_digest(target_noise),
            "references": list(reference_noise),
        },
        "schedule": tj.released_schedule(),
    }


@torch.no_grad()
def run_chain(
    request: ImageRequest,
    text: TextConditions,
    *,
    vae_contract: Any,
    denoiser: ImageDenoiser,
    device: torch.device | str,
    sampler: vs.ImageVelocitySampler | None = None,
    prepared: Sequence[PreparedReference] | None = None,
    label: str = "rl-kernel",
    keep_tensors: bool = False,
    mix_op: Any | None = None,
    on_step: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
) -> ChainRun:
    """Run the H3-Image chain on RL-Kernel contracts; ``text`` comes from :func:`contract_text`.

    ``sampler`` takes the six updates (``image_velocity_step`` on the ``image_sigma_pdd_schedule``
    grid); the default runs the PyTorch reference op, pass the Triton kernel on GPU. ``on_step``,
    if given, may replace the image rows after a step (drift injection in tests).
    """

    device = torch.device(device)
    log = StageLog(keep_tensors)
    prepared = (
        list(prepared)
        if prepared is not None
        else [prepare_reference(r) for r in request.references]
    )
    log.add("request", _request_fields(request, [r.source_sha256 for r in prepared]))
    plan = cv.resolve_canvas(request.height, request.width, request.min_generation_area)
    log.add(
        "media_preprocess",
        _preprocess_fields(plan.canvas, plan.latent, prepared),
        {"canvas": plan.artifact()},
    )
    if text.prompt != request.prompt:
        raise ValueError("text conditions were computed for another prompt")
    log.add("text_conditions", text.stage_tensors(), text.presentation_block(), text.provenance)

    generator = request.generator()
    ledger = rc.DrawLedger(generator)
    ref_latents: list[torch.Tensor] = []
    ref_records = []
    if prepared:
        outs = vae_contract.encode_references(
            [pil_to_uint8(r.vae_image) for r in prepared], generator
        )
        for i, out in enumerate(outs):
            (draw,) = out.record["generator"]["draws"]
            ledger.record(
                "posterior",
                i,
                (1, 24, *out.record["geometry"]["latent"]),
                draw["before"],
                draw["after"],
            )
            ref_latents.append(out.tensor)
            ref_records.append(out.record)
    log.add(
        "reference_encode",
        {"latents": ref_latents},
        {"geometry": [r["geometry"] for r in ref_records]},
        {"vae": ref_records[0]["tiling"] if ref_records else None},
    )

    target = rc.draw_target_noise(ledger, plan.latent, device)
    target_rows = pp.ImagePatchPacker().pack_image(target)[0]
    tags = text.text_token_tags.cpu().long()
    grids = [(int(z.shape[-2]), int(z.shape[-1])) for z in ref_latents]
    layout = pl.build_layout(tags, grids, plan.latent)
    cond = None
    if ref_latents:
        cond = rc.condition_references(
            ref_latents, ledger=ledger, layout=layout, device=device, mix_op=mix_op
        )
    ledger.check_order(len(ref_latents))
    image_rows = cond.image_rows(target_rows) if cond is not None else target_rows
    log.add(
        "reference_conditioning",
        {
            "target_noise": target,
            "reference_rows": cond.rows if cond is not None else [],
            "image_rows": image_rows,
        },
        {"ledger": ledger.artifact()},
        {"conditioning": cond.artifact() if cond is not None else None},
    )
    log.add("packed_layout", _layout_tensors(layout), {"layout": layout.artifact()})

    sampler = sampler if sampler is not None else vs.ImageVelocitySampler(device=device)
    if sampler.step_index != 0:
        raise ValueError("the sampler has already been stepped; use a new one per request")
    schedule = sampler.schedule
    timesteps = schedule.timesteps
    row_plan = ps.row_timestep_plan(layout, schedule)
    log.add(
        "schedule",
        {
            "sigmas": schedule.sigmas,
            "timesteps": timesteps,
            "row_timesteps": [u for u, _ in row_plan],
            "row_timestep_indices": [ix for _, ix in row_plan],
        },
        {"released_schedule": tj.released_schedule(), "pdd_schedule": ps.artifact(schedule)},
    )

    vae_mean, vae_std = vae_contract.mean, vae_contract.std
    refs_block = []
    noise_digests = []
    if cond is not None:
        for i, prep in enumerate(prepared):
            refs_block.append(
                {
                    "source_sha256": prep.source_sha256,
                    "vae_input_shape": [3, prep.vae_image.size[1], prep.vae_image.size[0]],
                    "latent": list(ref_latents[i].shape[-2:]),
                    "posterior_fp16_sha256": cond.digests["latents"][i],
                    "conditioned_rows_sha256": cond.digests["per_reference_rows"][i],
                }
            )
        noise_digests = list(cond.digests["noise"])
    recorder = tj.ImageTrajectoryRecorder(
        label=label,
        inputs=_trajectory_inputs(
            request,
            plan,
            text,
            refs_block,
            _vae_block(vae_mean, vae_std, vae_contract.tiled),
            layout,
            cond.rows if cond is not None else None,
            target,
            noise_digests,
        ),
        provenance={"denoiser": getattr(denoiser, "backend_id", type(denoiser).__name__)},
        keep_tensors=keep_tensors,
    )
    r = layout.num_reference_rows
    embeds = text.prompt_embeds.to(device)
    for i in range(schedule.num_model_evals):
        unique, indices = row_plan[i]
        velocity = denoiser(image_rows, embeds, layout, unique, indices, i)
        image_rows = sampler.step(i, velocity, image_rows, r, pdd_head=i)
        if on_step is not None:
            image_rows = on_step(i, image_rows)
        if cond is not None:
            cond.check_fixed(image_rows, step=i)
        recorder.record_step(
            i,
            pdd_head=i,
            timestep_values=unique,
            timestep_indices=indices,
            velocity=velocity,
            image_latents=image_rows[None],
        )
    log.add(
        "denoise",
        {"final_image_rows": image_rows},
        {"steps": [s.to_dict() for s in recorder.records], "updates": sampler.records},
        {
            "denoiser": getattr(denoiser, "backend_id", type(denoiser).__name__),
            "sampler": sampler.provenance(),
        },
    )
    latents = pp.ImagePatchPacker().unpack_target(image_rows[None], r, plan.latent)
    log.add("unpack", {"latents": latents})
    decoded = vae_contract.decode(latents.float())
    log.add("decode", {"canvas_uint8": decoded.pixels}, {"digests": decoded.digests})
    output = cv.resize_output(decoded.pixels, plan)
    recorder.record_decoded(output, output_size=list(plan.requested))
    log.add("resize", {"output_uint8": output}, {"resize": plan.resize_rule()})
    artifact = seal(label, request, log, recorder.seal(), {"device": str(device)})
    tensors = {**log.tensors, **{f"trajectory.{k}": v for k, v in recorder.tensors.items()}}
    return ChainRun(artifact=artifact, output=output, tensors=tensors)


# --- pinned Diffusers ------------------------------------------------------------------------


def _get(state: Any, key: str) -> Any:
    """``PipelineState.get`` for one key (typed ``Any``; its union return confuses mypy)."""

    return state.get(key)


def diffusers_text(
    request: ImageRequest,
    vision_images: Sequence[Any] | None,
    text_encoder: Any,
    tokenizer: Any,
    processor: Any,
) -> TextConditions:
    """The pinned Diffusers text-encoder step, via ``text_presentation.run_diffusers_reference``."""

    from rl_engine.models.minimax_h3_image import text_presentation as tp

    ref = tp.run_diffusers_reference(
        text_encoder,
        tokenizer,
        processor,
        request.prompt,
        list(vision_images) if vision_images else None,
    )
    return TextConditions(
        source="diffusers",
        prompt=request.prompt,
        token_ids=list(ref.token_ids),
        prompt_embeds=ref.prompt_embeds,
        text_token_tags=ref.text_token_tags,
        vision_inputs=dict(ref.vision_inputs),
        provenance={"diffusers_step": "MiniMaxH3Image(Edit)TextEncoderStep"},
    )


def diffusers_pipeline(*, vae: Any = None, transformer: Any = None, scheduler: Any = None) -> Any:
    """``MiniMaxH3ImageBlocks().init_pipeline()`` with the given components (no download)."""

    from diffusers.modular_pipelines.minimax_h3_image.modular_blocks_minimax_h3_image import (
        MiniMaxH3ImageBlocks,
    )

    pipe = MiniMaxH3ImageBlocks().init_pipeline()
    components = {
        k: v
        for k, v in (("vae", vae), ("transformer", transformer), ("scheduler", scheduler))
        if v is not None
    }
    if components:
        pipe.update_components(**components)
    return pipe


def diffusers_resize(pipe: Any, request: ImageRequest) -> tuple[list[Any], list[Any]]:
    """Run ``MiniMaxH3ImageEditResizeStep``: (reference_images, vision_images)."""

    from diffusers.modular_pipelines.minimax_h3_image.before_encoder import (
        MiniMaxH3ImageEditResizeStep,
    )
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    state = PipelineState()
    state.set("image", list(request.references))
    _, state = MiniMaxH3ImageEditResizeStep()(pipe, state)
    return list(_get(state, "reference_images")), list(_get(state, "vision_images"))


@torch.no_grad()
def run_diffusers(
    request: ImageRequest,
    text: TextConditions,
    pipe: Any,
    *,
    label: str = "diffusers",
    keep_tensors: bool = False,
) -> ChainRun:
    """The pinned modular pipeline, block by block and unmodified, sealed like :func:`run_chain`.

    ``pipe`` comes from :func:`diffusers_pipeline` with the VAE, transformer and scheduler on
    the execution device; ``text`` from :func:`diffusers_text`. The transformer call and the
    scheduler step are wrapped (not replaced) to record each step's velocity and rows.
    """

    from diffusers.modular_pipelines.minimax_h3_image import before_denoise as bd
    from diffusers.modular_pipelines.minimax_h3_image import encoders as enc
    from diffusers.modular_pipelines.minimax_h3_image import modular_blocks_minimax_h3_image as mb
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    device = pipe._execution_device
    log = StageLog(keep_tensors)
    state = PipelineState()
    state.set("prompt", request.prompt)
    state.set("generator", request.generator())
    state.set("height", request.height)
    state.set("width", request.width)
    state.set("min_generation_area", request.min_generation_area)
    state.set("output_type", "pt")
    sources = [image_digest(r) for r in request.references]
    log.add("request", _request_fields(request, sources))
    prepared: list[PreparedReference] = []
    if request.references:
        vae_images, vision_images = diffusers_resize(pipe, request)
        prepared = [
            PreparedReference(s, a, b) for s, a, b in zip(sources, vae_images, vision_images)
        ]
        state.set("reference_images", vae_images)
    core = (
        mb.MiniMaxH3ImageEditCoreDenoiseStep()
        if request.references
        else mb.MiniMaxH3ImageCoreDenoiseStep()
    )
    blocks = dict(core.sub_blocks)
    # prepare_canvas first: its outputs are the canvas the rest of the chain is built on.
    _, state = blocks.pop("prepare_canvas")(pipe, state)
    plan = cv.resolve_canvas(request.height, request.width, request.min_generation_area)
    if [_get(state, "height"), _get(state, "width")] != list(plan.canvas):
        canvas_note = {"diffusers_canvas": [_get(state, "height"), _get(state, "width")]}
    else:
        canvas_note = {}
    diffusers_canvas = (int(_get(state, "height")), int(_get(state, "width")))
    diffusers_latent = (diffusers_canvas[0] // 16, diffusers_canvas[1] // 16)
    log.add(
        "media_preprocess",
        _preprocess_fields(diffusers_canvas, diffusers_latent, prepared),
        {"canvas": plan.artifact(), **canvas_note},
    )
    log.add("text_conditions", text.stage_tensors(), text.presentation_block(), text.provenance)
    state.set("prompt_embeds", text.prompt_embeds.to(device))
    state.set("text_token_tags", text.text_token_tags)  # CPU, as the text step returns it

    ref_latents: list[torch.Tensor] = []
    if request.references:
        _, state = enc.MiniMaxH3ImageEditVaeEncoderStep()(pipe, state)
        got: list[torch.Tensor] = _get(state, "reference_latents")
        ref_latents = list(got)
    log.add("reference_encode", {"latents": [z.cpu() for z in ref_latents]})

    _, state = blocks.pop("prepare_layout")(pipe, state)
    _, state = blocks.pop("prepare_latents")(pipe, state)
    target_rows = _get(state, "latents").clone()
    num_ref = int(_get(state, "num_reference_rows") or 0)
    reference_noise: list[str] = []
    if request.references:
        original_randn = bd.randn_tensor

        def recording_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
            out = original_randn(*args, **kwargs)
            reference_noise.append(tensor_digest(out))
            return out

        bd.randn_tensor = recording_randn  # type: ignore[assignment]
        try:
            _, state = blocks.pop("prepare_condition_latents")(pipe, state)
        finally:
            bd.randn_tensor = original_randn  # type: ignore[assignment]
    image_rows = _get(state, "latents")
    target_noise = target_rows.reshape(1, plan.latent[0] // 2, plan.latent[1] // 2, 24, 2, 2)
    target_noise = target_noise.permute(0, 3, 1, 4, 2, 5).reshape(1, 24, 1, *plan.latent)
    log.add(
        "reference_conditioning",
        {
            "target_noise": target_noise,
            "reference_rows": image_rows[:num_ref] if num_ref else [],
            "image_rows": image_rows,
        },
    )
    layout = pl.PackedLayout(
        position_ids=_get(state, "position_ids").cpu(),
        token_tags=_get(state, "token_tags").cpu(),
        image_indices=_get(state, "image_indices").cpu(),
        text_indices=_get(state, "text_indices").cpu(),
        source_kind=torch.zeros(0, dtype=torch.long),
        reference_index=torch.zeros(0, dtype=torch.long),
        patch_coord=torch.zeros(0, 2, dtype=torch.long),
        num_reference_rows=num_ref,
        reference_grids=tuple((int(z.shape[-2]), int(z.shape[-1])) for z in ref_latents),
        target_grid=(int(plan.latent[0]), int(plan.latent[1])),
    )
    log.add("packed_layout", _layout_tensors(layout))
    _, state = blocks.pop("set_timesteps")(pipe, state)
    row_plan = _get(state, "row_timestep_plan")
    log.add(
        "schedule",
        {
            "sigmas": pipe.scheduler.sigmas,
            "timesteps": _get(state, "timesteps"),
            "row_timesteps": [u for u, _ in row_plan],
            "row_timestep_indices": [ix for _, ix in row_plan],
        },
        {"released_schedule": tj.released_schedule()},
    )

    refs_block = []
    for i, prep in enumerate(prepared):
        z = ref_latents[i]
        refs_block.append(
            {
                "source_sha256": prep.source_sha256,
                "vae_input_shape": [3, prep.vae_image.size[1], prep.vae_image.size[0]],
                "latent": list(z.shape[-2:]),
                "posterior_fp16_sha256": tensor_digest(z.cpu()),
                "conditioned_rows_sha256": "",
            }
        )
    start = 0
    for i, z in enumerate(ref_latents):
        n = (z.shape[-2] // 2) * (z.shape[-1] // 2)
        refs_block[i]["conditioned_rows_sha256"] = tensor_digest(image_rows[start : start + n])
        start += n
    vae = pipe.vae
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32)
    tiled = bool(getattr(vae, "use_tiling", False))
    recorder = tj.ImageTrajectoryRecorder(
        label=label,
        inputs=_trajectory_inputs(
            request,
            plan,
            text,
            refs_block,
            _vae_block(mean, std, tiled),
            layout,
            image_rows[:num_ref].clone() if num_ref else None,
            target_noise,
            reference_noise,
        ),
        provenance={"denoiser": "diffusers:MiniMaxH3ImageDenoiseStep"},
        keep_tensors=keep_tensors,
    )

    velocities: list[torch.Tensor] = []
    transformer = pipe.transformer

    def recording_hook(module: Any, args: Any, out: Any) -> None:
        # A forward hook, not a wrapper: the loop reads ``forward``'s signature to pick inputs.
        velocities.append(out[0] if isinstance(out, tuple) else out.sample)

    scheduler = pipe.scheduler
    original_step = scheduler.step

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        out = original_step(*args, **kwargs)
        i = len(recorder.records)
        unique, indices = row_plan[i]
        # ``sample`` is the generated-row view of the loop's own image-row tensor: read the
        # reference prefix from that tensor, not from the pre-loop state.
        sample = args[2] if len(args) > 2 else kwargs["sample"]
        live: torch.Tensor = sample._base if sample._base is not None else _get(state, "latents")
        rows = live.clone()
        rows[num_ref:] = out[0]
        recorder.record_step(
            i,
            pdd_head=i,
            timestep_values=unique,
            timestep_indices=indices,
            velocity=velocities[i],
            image_latents=rows[None],
        )
        return out

    handle = transformer.register_forward_hook(recording_hook)
    scheduler.step = recording_step
    try:
        _, state = blocks.pop("denoise")(pipe, state)
    finally:
        handle.remove()
        scheduler.step = original_step
    final_rows = _get(state, "latents")
    log.add(
        "denoise",
        {"final_image_rows": final_rows},
        {"steps": [s.to_dict() for s in recorder.records]},
        {"denoiser": "diffusers:MiniMaxH3ImageDenoiseStep"},
    )
    _, state = blocks.pop("after_denoise")(pipe, state)
    if blocks:
        raise RuntimeError(f"pinned core has blocks this runner does not know: {list(blocks)}")
    latents = _get(state, "latents")
    log.add("unpack", {"latents": latents})

    # The decode step, unmodified, at canvas size (no resize) and at the requested size.
    from diffusers.modular_pipelines.minimax_h3_image.decoders import MiniMaxH3ImageDecodeStep

    state.set("output_height", None)
    state.set("output_width", None)
    _, state = MiniMaxH3ImageDecodeStep()(pipe, state)
    canvas_images: torch.Tensor = _get(state, "images")
    canvas_uint8 = (canvas_images.cpu() * 255).round().to(torch.uint8)
    log.add("decode", {"canvas_uint8": canvas_uint8})
    state.set("output_height", plan.requested[0])
    state.set("output_width", plan.requested[1])
    _, state = MiniMaxH3ImageDecodeStep()(pipe, state)
    output = (_get(state, "images").cpu() * 255).round().to(torch.uint8)
    recorder.record_decoded(output, output_size=list(plan.requested))
    log.add("resize", {"output_uint8": output})
    artifact = seal(label, request, log, recorder.seal(), {"device": str(device)})
    tensors = {**log.tensors, **{f"trajectory.{k}": v for k, v in recorder.tensors.items()}}
    return ChainRun(artifact=artifact, output=output, tensors=tensors)


# --- comparison ------------------------------------------------------------------------------


def _stage_tensors(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["stage"]: r["tensors"] for r in artifact["stages"]}


def compare(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, Any]:
    """Stage-by-stage tensor digests, then the trajectory's first drift."""

    a, b = _stage_tensors(lhs), _stage_tensors(rhs)
    stages: dict[str, Any] = {}
    first: dict[str, Any] | None = None
    for stage in STAGES:
        keys = sorted(set(a[stage]) | set(b[stage]))
        differing = [k for k in keys if a[stage].get(k) != b[stage].get(k)]
        stages[stage] = {"equal": not differing, "differing": differing, "compared": keys}
        if differing and first is None:
            first = {"stage": stage, "tensor": differing[0]}
    drift = tj.first_drift(lhs["trajectory"], rhs["trajectory"])
    return {
        "stages": stages,
        "first_drift": first,
        "trajectory_first_drift": drift,
        "gate": all(stages[s]["equal"] for s in GATED_STAGES),
        "secondary_equal": all(stages[s]["equal"] for s in SECONDARY_STAGES),
    }


def gate(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    """Latent gate: every stage up to ``unpack`` byte-equal."""

    return bool(compare(lhs, rhs)["gate"])


def contract_manifest() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "schema": SCHEMA_VERSION,
        "stages": list(STAGES),
        "gated_stages": list(GATED_STAGES),
        "secondary_stages": list(SECONDARY_STAGES),
        "reference_short_edge": list(REFERENCE_SHORT_EDGE),
        "vision_short_edge": list(VISION_SHORT_EDGE),
        "pdd_schedule": ps.artifact(),
    }
