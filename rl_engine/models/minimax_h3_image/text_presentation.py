# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3-Image text presentation (RFC #738 Phase B ``image_text_presentation_contract``).

MiniMax-H3-Image conditions on the same Qwen3-VL-32B text encoder as MiniMax-H3
(``fingerprint.check_inherited_text_encoder``), read at ``hidden_states[50]`` (width 5120).
What is image-specific is the *presentation*: the token sequence the encoder sees and the
per-row modality tag every feature row carries into the transformer's AdaLN table.

* **Text-to-image** (``text2image``): the prompt verbatim. No chat template, no special
  tokens (``add_special_tokens=False``), truncated to the first 4096 tokens. Every row is
  tagged text (1).
* **Edit** (``image_conditioned``): for each reference image, in reference order, the label
  ``"Picture i: "`` (1-based, tokenized on its own) and one Qwen3-VL image vision block
  (``<|vision_start|>``, ``prod(grid_thw) / merge_size**2`` ``<|image_pad|>``,
  ``<|vision_end|>``), then the prompt. The prompt and each label are truncated to 4096
  tokens on their own; the whole presentation is not. Vision-block rows, brackets included,
  are tagged image (0); label and prompt rows are tagged text (1). The prompt names the
  references ``<Picture i>`` by this order, so the order is part of the conditioning.

Everything here is frozen against the pinned Diffusers modular pipeline
(``diffusers@875373ac``, ``modular_pipelines/minimax_h3_image/encoders.py``,
``before_encoder.py`` and ``modular_pipeline.py``, hashed in
:data:`DIFFUSERS_REFERENCE_FILES`). Where Diffusers accepts an input the release never
produces, this module fails closed instead: a prompt that spells out a special token, a
vision grid that is not a single image frame or not a whole number of merged patches, vision
inputs whose grids or order disagree with the presentation, and a tag table that disagrees
with the token ids.

The tokenizer special-token table, the processor geometry, the layer-50 forward and the
release runtime policy are reused from the MiniMax-H3 contract
(``rl_engine.models.minimax_h3.qwen3vl_text``, RFC #420 ``qwen3vl_layer50_contract``). The
H3-Image ``tokenizer/`` and ``processor/`` files are the H3 ones byte for byte
(:func:`inherited_presentation_files`, :func:`verify_presentation_files`), so the H3 pins
apply unchanged.

Prompt expansion (RFC #738 probe I24) is outside the presentation: the pinned pipeline has
no expander component, and a presentation records the SHA-256 of the exact prompt bytes it
tokenized, so any upstream rewrite shows up as a different ``prompt_sha256``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl_engine.models.minimax_h3 import qwen3vl_text as q
from rl_engine.models.minimax_h3.trajectory import tensor_digest
from rl_engine.models.minimax_h3_image import fingerprint as image_fp

CONTRACT_ID = "h3-image-text-presentation-v1"
FIXTURES_PATH = Path(__file__).with_name("text_presentation_fixtures.json")

TEXT2IMAGE = "text2image"
IMAGE_CONDITIONED = "image_conditioned"
MODES = (TEXT2IMAGE, IMAGE_CONDITIONED)

# Per-row modality tags (Diffusers ``MINIMAX_H3_IMAGE_TAG`` / ``MINIMAX_H3_TEXT_TAG``). They
# index the transformer's AdaLN table, so they are a checkpoint contract. The image tag is the
# H3 video tag: both are AdaLN row 0.
IMAGE_TAG = 0
TEXT_TAG = 1

MAX_SEQUENCE_LENGTH = image_fp.MAX_SEQUENCE_LENGTH
TEXT_ENCODER_LAYER = image_fp.TEXT_ENCODER_LAYER
FEATURE_WIDTH = q.FEATURE_WIDTH

LABEL_TEMPLATE = "Picture {index}: "
"""The label before reference ``index`` (1-based). No angle brackets, unlike the prompt's
``<Picture i>``, and a trailing space that stays a token of its own."""

DIFFUSERS_COMMIT = image_fp.DIFFUSERS_COMMIT
DIFFUSERS_REFERENCE_FILES: Mapping[str, str] = {
    "src/diffusers/modular_pipelines/minimax_h3_image/encoders.py": (
        "e4ddd0e51d9be072575f5efbb607d3d3238cb40b1274b75c487b294d0f80fe9c"
    ),
    "src/diffusers/modular_pipelines/minimax_h3_image/before_encoder.py": (
        "2e6a2012045656c1e91561b148f59fa2a027afd1da336938ab0f3a67d2224ef5"
    ),
    "src/diffusers/modular_pipelines/minimax_h3_image/modular_pipeline.py": (
        "7ecdae21bf754f6733a78bcfe6a35f29fbfe87cd39f45bf91b33b6d57e50543d"
    ),
}
"""SHA-256 of the Diffusers files the presentation is frozen against (:data:`DIFFUSERS_COMMIT`)."""

DIFFUSERS_REFERENCE_STEPS = {
    TEXT2IMAGE: "MiniMaxH3ImageTextEncoderStep",
    IMAGE_CONDITIONED: "MiniMaxH3ImageEditTextEncoderStep",
}

PRESENTATION_SUBFOLDERS = ("tokenizer", "processor")

LABEL, VISION, PROMPT = "label", "vision", "prompt"

PresentationError = q.PresentationError


# --- presentation -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One contiguous run of a presentation: a label, a vision block or the prompt."""

    kind: str
    start: int
    length: int
    reference: int | None = None  # 0-based reference index for labels and vision blocks

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start,
            "length": self.length,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class TextPresentation:
    """The token ids and per-row modality tags of one request, with how they were built."""

    mode: str
    token_ids: tuple[int, ...]
    token_tags: tuple[int, ...]
    segments: tuple[Segment, ...]
    image_grid_thw: tuple[tuple[int, int, int], ...]
    merge_size: int
    prompt_sha256: str
    prompt_num_tokens: int
    """Prompt tokens before truncation; the presentation keeps ``min(this, 4096)``."""
    max_sequence_length: int = MAX_SEQUENCE_LENGTH

    @property
    def num_rows(self) -> int:
        return len(self.token_ids)

    @property
    def prompt_truncated(self) -> bool:
        return self.prompt_num_tokens > self.max_sequence_length

    def tags_tensor(self) -> torch.Tensor:
        """The tags as Diffusers returns them (``text_token_tags``): int64, one per row."""

        return torch.tensor(self.token_tags, dtype=torch.long)

    def summary(self) -> dict[str, Any]:
        """Digests of everything that decides the presentation (no raw ids)."""

        return {
            "mode": self.mode,
            "num_rows": self.num_rows,
            "token_ids_sha256": q.token_ids_digest(self.token_ids),
            "token_tags_sha256": tensor_digest(self.tags_tensor()),
            "image_rows": sum(1 for tag in self.token_tags if tag == IMAGE_TAG),
            "text_rows": sum(1 for tag in self.token_tags if tag == TEXT_TAG),
            "image_grid_thw": [list(grid) for grid in self.image_grid_thw],
            "segments": [segment.to_dict() for segment in self.segments],
            "prompt_sha256": self.prompt_sha256,
            "prompt_num_tokens": self.prompt_num_tokens,
            "prompt_truncated": self.prompt_truncated,
        }


def _check_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise PresentationError(
            "MiniMax-H3-Image packs one request into one sequence, so the prompt must be a "
            f"single non-empty string, got {prompt!r}"
        )
    return prompt


def _plain_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(i) for i in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _check_max_length(max_sequence_length: int) -> int:
    if int(max_sequence_length) != max_sequence_length or max_sequence_length < 1:
        raise ValueError(f"max_sequence_length must be a positive int, got {max_sequence_length}")
    return int(max_sequence_length)


def _grids(image_grid_thw: Any, merge_size: int) -> tuple[tuple[int, int, int], ...]:
    """Validate the processor's ``image_grid_thw`` and return it as int triples."""

    if isinstance(image_grid_thw, torch.Tensor):
        if image_grid_thw.dtype.is_floating_point or image_grid_thw.dim() != 2:
            raise PresentationError(
                f"image_grid_thw must be an integer (n, 3) tensor, got {image_grid_thw.dtype} "
                f"{tuple(image_grid_thw.shape)}"
            )
        rows = image_grid_thw.tolist()
    else:
        rows = [list(row) for row in image_grid_thw]
    grids: list[tuple[int, int, int]] = []
    for index, row in enumerate(rows):
        if len(row) != 3 or any(isinstance(v, bool) or int(v) != v for v in row):
            raise PresentationError(f"image_grid_thw[{index}] = {row!r} is not three ints")
        t, h, w = (int(v) for v in row)
        if t != 1:
            raise PresentationError(f"reference {index + 1} has grid t={t}; an image block has t=1")
        if h <= 0 or w <= 0 or h % merge_size or w % merge_size:
            raise PresentationError(
                f"reference {index + 1} grid {h}x{w} is not a whole number of "
                f"{merge_size}x{merge_size} merged patches"
            )
        grids.append((t, h, w))
    return tuple(grids)


def present_text2image(
    tokenizer: Any,
    prompt: str,
    *,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> TextPresentation:
    """The ``text2image`` presentation: the prompt verbatim, truncated, every row text.

    Matches ``MiniMaxH3ImageTextEncoderStep`` token for token. Raises
    :class:`PresentationError` for an empty or non-string prompt (as Diffusers does) and for a
    prompt that spells out a special token (Diffusers accepts it).
    """

    limit = _check_max_length(max_sequence_length)
    prompt = _check_prompt(prompt)
    ids = q.tokenize_prompt(tokenizer, prompt)
    kept = tuple(ids[:limit])
    return TextPresentation(
        mode=TEXT2IMAGE,
        token_ids=kept,
        token_tags=(TEXT_TAG,) * len(kept),
        segments=(Segment(PROMPT, 0, len(kept)),),
        image_grid_thw=(),
        merge_size=0,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_num_tokens=len(ids),
        max_sequence_length=limit,
    )


def present_edit(
    tokenizer: Any,
    prompt: str,
    image_grid_thw: Any,
    *,
    merge_size: int = 2,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> TextPresentation:
    """The ``image_conditioned`` presentation of ``len(image_grid_thw)`` references.

    ``image_grid_thw`` is the processor output for the references in reference order (see
    :func:`edit_vision_inputs`). Matches ``MiniMaxH3ImageEditTextEncoderStep`` token for token
    and tag for tag. Fails closed on no references, a grid that is not one image frame of whole
    merged patches, and a prompt that spells out a special token.
    """

    limit = _check_max_length(max_sequence_length)
    prompt = _check_prompt(prompt)
    if int(merge_size) != merge_size or merge_size < 1:
        raise ValueError(f"merge_size must be a positive int, got {merge_size}")
    grids = _grids(image_grid_thw, int(merge_size))
    if not grids:
        raise PresentationError("an edit needs at least one reference image")
    merge = int(merge_size) ** 2
    ids: list[int] = []
    tags: list[int] = []
    segments: list[Segment] = []
    for index, (t, h, w) in enumerate(grids):
        label = _plain_ids(tokenizer, LABEL_TEMPLATE.format(index=index + 1))[:limit]
        segments.append(Segment(LABEL, len(ids), len(label), index))
        ids += label
        tags += [TEXT_TAG] * len(label)
        block = q.vision_block("image", t * h * w // merge)
        segments.append(Segment(VISION, len(ids), len(block), index))
        ids += block
        tags += [IMAGE_TAG] * len(block)
    prompt_ids = q.tokenize_prompt(tokenizer, prompt)
    kept = prompt_ids[:limit]
    segments.append(Segment(PROMPT, len(ids), len(kept)))
    ids += kept
    tags += [TEXT_TAG] * len(kept)
    return TextPresentation(
        mode=IMAGE_CONDITIONED,
        token_ids=tuple(ids),
        token_tags=tuple(tags),
        segments=tuple(segments),
        image_grid_thw=grids,
        merge_size=int(merge_size),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_num_tokens=len(prompt_ids),
        max_sequence_length=limit,
    )


def edit_vision_inputs(processor: Any, vision_images: Sequence[Any]) -> dict[str, torch.Tensor]:
    """``pixel_values`` and ``image_grid_thw`` of the references, as Diffusers computes them.

    ``vision_images`` are the references already resized for the text encoder (Diffusers
    ``MiniMaxH3ImageEditResizeStep``: short edge bounded to [1024, 1024], sides rounded to 32),
    in reference order. The whole list goes through the image processor in one call.
    """

    images = list(vision_images)
    if not images:
        raise PresentationError("an edit needs at least one reference image")
    out = processor.image_processor(images=images, return_tensors="pt")
    return {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}


def merge_size_of(processor: Any) -> int:
    return int(processor.image_processor.merge_size)


def present_edit_from_images(
    tokenizer: Any,
    processor: Any,
    prompt: str,
    vision_images: Sequence[Any],
    *,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[TextPresentation, dict[str, torch.Tensor]]:
    """Run the processor on the references and build the edit presentation over its grids."""

    vision = edit_vision_inputs(processor, vision_images)
    presentation = present_edit(
        tokenizer,
        prompt,
        vision["image_grid_thw"],
        merge_size=merge_size_of(processor),
        max_sequence_length=max_sequence_length,
    )
    return presentation, vision


# --- checks -------------------------------------------------------------------------------------


def expected_tags(token_ids: Sequence[int]) -> list[int]:
    """The tag of every row, derived from the ids alone: vision-block tokens are image rows."""

    vision = set(q.vision_token_ids().values())
    return [IMAGE_TAG if int(i) in vision else TEXT_TAG for i in token_ids]


def check_tags(presentation: TextPresentation) -> None:
    """Fail unless every row's tag is the one its token id and segment give (probe I20)."""

    tags = list(presentation.token_tags)
    if len(tags) != presentation.num_rows:
        raise PresentationError(f"{len(tags)} tags for {presentation.num_rows} rows")
    if any(tag not in (IMAGE_TAG, TEXT_TAG) for tag in tags):
        raise PresentationError(f"tags outside {{{IMAGE_TAG}, {TEXT_TAG}}}: {sorted(set(tags))}")
    from_ids = expected_tags(presentation.token_ids)
    from_segments = [TEXT_TAG] * presentation.num_rows
    covered = 0
    for segment in presentation.segments:
        if segment.start != covered:
            raise PresentationError(f"segment {segment} does not start at row {covered}")
        if segment.kind == VISION:
            from_segments[segment.start : segment.start + segment.length] = [
                IMAGE_TAG
            ] * segment.length
        covered += segment.length
    if covered != presentation.num_rows:
        raise PresentationError(f"segments cover {covered} of {presentation.num_rows} rows")
    for name, reference in (("token ids", from_ids), ("segments", from_segments)):
        if tags != reference:
            row = next(i for i, (a, b) in enumerate(zip(tags, reference)) if a != b)
            raise PresentationError(
                f"tag of row {row} is {tags[row]}, the {name} give {reference[row]}"
            )


def check_vision_inputs(
    presentation: TextPresentation, vision_inputs: Mapping[str, torch.Tensor] | None
) -> None:
    """Fail unless the vision inputs are the presentation's references, in its order.

    A ``text2image`` presentation takes none. An edit takes exactly ``pixel_values`` and
    ``image_grid_thw``, with the grids equal to the presentation's, row for row, and one
    ``pixel_values`` row per patch. Video inputs are never part of an H3-Image presentation.
    """

    inputs = dict(vision_inputs or {})
    if presentation.mode == TEXT2IMAGE:
        if inputs:
            raise PresentationError(f"text2image takes no vision inputs, got {sorted(inputs)}")
        return
    if set(inputs) != {"pixel_values", "image_grid_thw"}:
        raise PresentationError(
            f"an edit takes pixel_values and image_grid_thw, got {sorted(inputs)}"
        )
    grids = _grids(inputs["image_grid_thw"], presentation.merge_size)
    if grids != presentation.image_grid_thw:
        raise PresentationError(
            f"vision grids {list(grids)} differ from the presentation's "
            f"{list(presentation.image_grid_thw)} (reference order or geometry changed)"
        )
    patches = sum(t * h * w for t, h, w in grids)
    pixels = inputs["pixel_values"]
    if pixels.dim() != 2 or int(pixels.shape[0]) != patches:
        raise PresentationError(
            f"pixel_values {tuple(pixels.shape)} does not hold {patches} patch rows"
        )


# --- encoding -----------------------------------------------------------------------------------


@dataclass
class PresentedFeature:
    """The layer-50 feature of one presentation and the tags of its rows."""

    presentation: TextPresentation
    output: q.Layer50Output

    @property
    def prompt_embeds(self) -> torch.Tensor:
        """``(1, rows, 5120)`` in the encoder dtype, what Diffusers returns as ``prompt_embeds``."""

        return self.output.feature

    @property
    def text_token_tags(self) -> torch.Tensor:
        return self.presentation.tags_tensor()

    def summary(self) -> dict[str, Any]:
        return {**self.presentation.summary(), "feature": self.output.summary()}


def encode(
    encoder: q.Qwen3VLLayer50Encoder,
    presentation: TextPresentation,
    vision_inputs: Mapping[str, torch.Tensor] | None = None,
) -> PresentedFeature:
    """Encode one presentation with the H3 layer-50 encoder after every presentation check."""

    if encoder.layer != TEXT_ENCODER_LAYER:
        raise ValueError(f"H3-Image conditions on hidden_states[{TEXT_ENCODER_LAYER}]")
    check_tags(presentation)
    check_vision_inputs(presentation, vision_inputs)
    # The encoder checks the (1, rows, hidden_size) shape; hidden_size is 5120 for the release
    # (its config check pins it) and smaller only for the tiny test models.
    output = encoder.encode(presentation.token_ids, vision_inputs or None)
    return PresentedFeature(presentation=presentation, output=output)


# --- pins ---------------------------------------------------------------------------------------


def inherited_presentation_files() -> dict[str, tuple[int, str, str]]:
    """``{path: (size, H3-Image git oid, H3 sha256)}`` of every tokenizer/processor file.

    Offline check that the H3-Image revision ships exactly the files the H3 contract pins, at
    the same paths and sizes. The content check needs the files: :func:`verify_presentation_files`.
    """

    image_dirs = image_fp.pinned_checkpoint()["dirs"]
    h3_files = q.release_fingerprint()["files"]
    out: dict[str, tuple[int, str, str]] = {}
    for sub in PRESENTATION_SUBFOLDERS:
        for entry in image_dirs[sub]:
            path = str(entry["path"])
            pinned = h3_files.get(path)
            if pinned is None:
                raise ValueError(f"{path} is in {image_fp.REPO} but not pinned by {q.CONTRACT_ID}")
            if int(entry["size"]) != int(pinned["size_bytes"]):
                raise ValueError(f"{path}: size {entry['size']} != H3 {pinned['size_bytes']}")
            out[path] = (int(entry["size"]), str(entry["git_oid"]), str(pinned["sha256"]))
    h3_paths = {p for p in h3_files if p.split("/", 1)[0] in PRESENTATION_SUBFOLDERS}
    if set(out) != h3_paths:
        raise ValueError(f"file sets differ: {sorted(set(out) ^ h3_paths)}")
    return out


def _git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_presentation_files(root: str | Path) -> dict[str, str]:
    """Hash a local H3-Image download's tokenizer/processor files against both pins.

    Each file must have the git blob id the H3-Image snapshot pins *and* the SHA-256 the H3
    contract pins, which proves the H3-Image bytes are the H3 bytes. Returns ``{path: sha256}``.
    """

    base = Path(root)
    checked: dict[str, str] = {}
    for path, (size, oid, sha) in sorted(inherited_presentation_files().items()):
        data = (base / path).read_bytes()
        if len(data) != size:
            raise ValueError(f"{path}: size {len(data)} != pinned {size}")
        if _git_blob_oid(data) != oid:
            raise ValueError(f"{path}: not the {image_fp.REPO}@{image_fp.REVISION[:8]} file")
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise ValueError(f"{path}: sha256 {got} != H3 pin {sha}")
        checked[path] = got
    return checked


def verify_diffusers_reference(diffusers_package_dir: str | Path) -> dict[str, str]:
    """SHA-256 the installed Diffusers files the presentation is frozen against.

    ``diffusers_package_dir`` is the directory of the ``diffusers`` package
    (``Path(diffusers.__file__).parent``). Fails on any difference from
    :data:`DIFFUSERS_REFERENCE_FILES`.
    """

    base = Path(diffusers_package_dir)
    checked: dict[str, str] = {}
    for rel, sha in DIFFUSERS_REFERENCE_FILES.items():
        path = base / rel.removeprefix("src/diffusers/")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            raise ValueError(f"{rel}: sha256 {got} != pinned {sha} (diffusers@{DIFFUSERS_COMMIT})")
        checked[rel] = got
    return checked


# --- sealed fixtures ----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def sealed_fixtures() -> Mapping[str, Any]:
    """The sealed layer-50 fixtures (``text_presentation_fixtures.json``)."""

    data = json.loads(FIXTURES_PATH.read_text())
    if data.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"{FIXTURES_PATH.name} is not for {CONTRACT_ID}")
    return data


def fixture_image(recipe: Mapping[str, Any]) -> Any:
    """A deterministic RGB reference image from a fixture recipe (``width``, ``height``, ``seed``).

    A smooth two-axis gradient plus seeded uint8 noise, so every reference is distinct and the
    vision tower sees structure. Built with NumPy only; no file is read.
    """

    import numpy as np
    import PIL.Image

    width, height, seed = int(recipe["width"]), int(recipe["height"]), int(recipe["seed"])
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij")
    phase = rng.uniform(0.0, 1.0, size=3)
    base = np.stack([(xx + phase[c] + (c + 1) * yy) % 1.0 for c in range(3)], axis=-1)
    noise = rng.integers(-24, 25, size=(height, width, 3))
    pixels = np.clip(base * 230.0 + noise, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(pixels)


def fixture_prompt(case: Mapping[str, Any]) -> str:
    """The prompt of a fixture case: ``prompt`` repeated ``repeat`` times (default once)."""

    return str(case["prompt"]) * int(case.get("repeat", 1))


def build_fixture_case(
    case: Mapping[str, Any], tokenizer: Any, processor: Any | None = None
) -> tuple[TextPresentation, dict[str, torch.Tensor] | None]:
    """The presentation (and vision inputs) of one sealed fixture case."""

    prompt = fixture_prompt(case)
    if case["mode"] == TEXT2IMAGE:
        return present_text2image(tokenizer, prompt), None
    if processor is None:
        raise ValueError("an edit fixture needs the processor")
    images = [fixture_image(recipe) for recipe in case["references"]]
    return present_edit_from_images(tokenizer, processor, prompt, images)


def vision_digest(vision_inputs: Mapping[str, torch.Tensor] | None) -> dict[str, Any] | None:
    """Digests of the processor outputs of an edit (``pixel_values`` in float32)."""

    if not vision_inputs:
        return None
    return {
        "pixel_values_sha256": tensor_digest(vision_inputs["pixel_values"].float()),
        "pixel_values_shape": list(vision_inputs["pixel_values"].shape),
        "image_grid_thw": vision_inputs["image_grid_thw"].tolist(),
    }


def device_key(device: torch.device | None = None) -> str:
    """The key a sealed feature digest is stored under: GPU arch plus torch/transformers."""

    import transformers

    if device is not None and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        arch = f"sm_{major}{minor}"
    else:
        arch = "cpu"
    return f"{arch}|torch {torch.__version__}|transformers {transformers.__version__}"


# --- Diffusers reference ------------------------------------------------------------------------


@dataclass
class DiffusersReference:
    """What the pinned Diffusers text-encoder step produced for one request."""

    prompt_embeds: torch.Tensor
    text_token_tags: torch.Tensor
    token_ids: list[int]
    vision_inputs: dict[str, torch.Tensor]


@torch.no_grad()
def run_diffusers_reference(
    text_encoder: Any,
    tokenizer: Any,
    processor: Any,
    prompt: str,
    vision_images: Sequence[Any] | None = None,
    *,
    device: torch.device | str | None = None,
) -> DiffusersReference:
    """Run the pinned Diffusers step (T2I or edit) and capture the ids it handed the encoder.

    Calls ``MiniMaxH3ImageTextEncoderStep`` (no images) or ``MiniMaxH3ImageEditTextEncoderStep``
    unmodified, with the components the pipeline would hold. The step's call to
    ``get_qwen3vl_prompt_embeds`` is wrapped, not replaced, to record the token ids and vision
    tensors it builds. Needs Diffusers at :data:`DIFFUSERS_COMMIT` (checked by hash).
    """

    import types

    import diffusers
    from diffusers.modular_pipelines.minimax_h3_image import encoders
    from diffusers.modular_pipelines.minimax_h3_image import modular_pipeline as mp
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    verify_diffusers_reference(Path(diffusers.__file__).parent)
    if device is None:
        device = next(text_encoder.parameters()).device
    pipeline = mp.MiniMaxH3ImageModularPipeline
    components = types.SimpleNamespace(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        max_sequence_length=getattr(pipeline, "max_sequence_length").fget(None),
        text_encoder_layer=getattr(pipeline, "text_encoder_layer").fget(None),
        text_tag=mp.MINIMAX_H3_TEXT_TAG,
        image_tag=mp.MINIMAX_H3_IMAGE_TAG,
        _execution_device=torch.device(device),
    )
    captured: dict[str, Any] = {}
    original = encoders.get_qwen3vl_prompt_embeds

    def recording(encoder: Any, proc: Any, token_ids: list[int], vision: Any = None, **kw: Any):
        captured["token_ids"] = [int(i) for i in token_ids]
        captured["vision_inputs"] = dict(vision or {})
        return original(encoder, proc, token_ids, vision, **kw)

    state = PipelineState()
    state.set("prompt", prompt)
    if vision_images is None:
        step: Any = encoders.MiniMaxH3ImageTextEncoderStep()
    else:
        step = encoders.MiniMaxH3ImageEditTextEncoderStep()
        state.set("vision_images", list(vision_images))
    encoders.get_qwen3vl_prompt_embeds = recording  # type: ignore[assignment]
    try:
        _, state = step(components, state)
    finally:
        encoders.get_qwen3vl_prompt_embeds = original  # type: ignore[assignment]
    return DiffusersReference(
        prompt_embeds=state.get("prompt_embeds"),
        text_token_tags=state.get("text_token_tags"),
        token_ids=captured["token_ids"],
        vision_inputs=captured["vision_inputs"],
    )


def contract_manifest() -> dict[str, Any]:
    """Machine-readable description of what this contract pins (for evidence files)."""

    return {
        "contract_id": CONTRACT_ID,
        "model_id": image_fp.REPO,
        "revision": image_fp.REVISION,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "diffusers_reference_files": dict(DIFFUSERS_REFERENCE_FILES),
        "diffusers_reference_steps": dict(DIFFUSERS_REFERENCE_STEPS),
        "text_encoder": {
            "inherited_from": f"{image_fp.TEXT_ENCODER_REPO}@{image_fp.TEXT_ENCODER_REVISION}",
            "contract": q.CONTRACT_ID,
            "layer": TEXT_ENCODER_LAYER,
            "feature_width": FEATURE_WIDTH,
        },
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "label_template": LABEL_TEMPLATE,
        "tags": {"image": IMAGE_TAG, "text": TEXT_TAG},
        "special_tokens": q.vision_token_ids(),
    }
