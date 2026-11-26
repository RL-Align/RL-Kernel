# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run the public tiny Diffusers MiniMax-H3 modular pipeline and capture its preprocessing.

``hf-internal-testing/tiny-minimax-h3-modular-pipe`` has the released architecture at toy
width (patch ``(1, 2, 2)``, VAE 16x / 17-to-5 chunks, stereo audio, 40 latents/s), so every
preprocessing step of the pinned pipeline runs for real on CPU in seconds. The blocks'
``__call__`` are wrapped to snapshot the block state after selected steps; the test then
compares the RL-Kernel contract to those snapshots byte for byte.

Needs ``diffusers`` with ``modular_pipelines.minimax_h3``, ``transformers`` and
``accelerate``; tests skip when any is missing.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any, Iterator

import pytest

TINY_REPO = "hf-internal-testing/tiny-minimax-h3-modular-pipe"
TINY_CANVAS = {"canvas_short_edge": 64, "canvas_max_pixels": 64 * 128}

CAPTURE = {
    "MiniMaxH3ResizeStep": ("keyframes", "keyframe_anchors", "height", "width"),
    "MiniMaxH3Ref2VASetupStep": ("normalized_references", "height", "width", "num_frames"),
    "MiniMaxH3TextEncoderStep": ("text_token_tags",),
    "MiniMaxH3FL2VATextEncoderStep": ("text_token_tags",),
    "MiniMaxH3Ref2VATextEncoderStep": ("text_token_tags",),
    "MiniMaxH3KeyframeVaeEncoderStep": ("condition_latents",),
    "MiniMaxH3Ref2VAReferenceEncoderStep": ("condition_latents", "audio_condition_latents"),
    "MiniMaxH3PrepareLayoutStep": (
        "position_ids",
        "token_tags",
        "video_indices",
        "audio_indices",
        "text_indices",
        "num_condition_video_rows",
        "num_condition_audio_rows",
        "num_frames",
    ),
    "MiniMaxH3PrepareConditionLatentsStep": ("condition_rows",),
    "MiniMaxH3PrepareLatentsStep": ("latents", "audio_latents"),
    "MiniMaxH3FL2VAPrepareLatentsStep": ("latents",),
    "MiniMaxH3Ref2VAPrepareLatentsStep": ("latents", "audio_latents"),
    "MiniMaxH3SetTimestepsStep": ("timesteps", "audio_timesteps", "row_timestep_plan"),
}
CAPTURE["MiniMaxH3Ref2VAPrepareLayoutStep"] = CAPTURE["MiniMaxH3PrepareLayoutStep"]


def require_tiny_pipeline() -> Any:
    """Import what the tiny pipeline needs or skip the test."""

    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
    return diffusers


_PIPELINES: dict[tuple, Any] = {}


def _load(diffusers: Any, **hub: Any) -> Any:
    import torch

    pipe = diffusers.ModularPipeline.from_pretrained(TINY_REPO, **hub)
    pipe.load_components(torch_dtype=torch.float32, **hub)
    # load_components logs a component it cannot load and leaves it None.
    if pipe.null_component_names:
        raise OSError(f"{TINY_REPO}: components not loaded: {pipe.null_component_names}")
    return pipe


def load_tiny_pipeline(**config: Any) -> Any:
    """The tiny pipeline (loaded once per process, cached copy first) with a small canvas."""

    diffusers = require_tiny_pipeline()
    if "pipe" not in _PIPELINES:
        try:
            # local_files_only, not HF_HUB_OFFLINE: the Hub reads that once, at import.
            pipe = _load(diffusers, local_files_only=True)
        except Exception:  # not (completely) cached: try the Hub once
            try:
                pipe = _load(diffusers)
            except OSError as exc:  # offline without a complete cached copy
                pytest.skip(f"{TINY_REPO} unavailable: {exc}")
        _PIPELINES["pipe"] = pipe
    pipe = _PIPELINES["pipe"]
    pipe.register_to_config(**{**TINY_CANVAS, "reference_image_short_edge": 2048, **config})
    return pipe


_ABSENT = object()


@contextlib.contextmanager
def capture_steps() -> Iterator[dict[str, dict[str, Any]]]:
    """Snapshot ``CAPTURE`` fields of every listed step's block state after it runs."""

    from diffusers.modular_pipelines.minimax_h3 import before_denoise, before_encoder, encoders

    modules = (before_denoise, before_encoder, encoders)
    captured: dict[str, dict[str, Any]] = {}
    originals = []
    try:  # a step missing mid-loop must not leave the earlier ones wrapped
        for name, fields in CAPTURE.items():
            cls = next(getattr(m, name) for m in modules if hasattr(m, name))
            # the class's own ``__call__`` (or none: it inherits one), restored as it was; the
            # method wrapped is the real one, also when it is inherited from a step that is
            # already wrapped here
            own = cls.__dict__.get("__call__", _ABSENT)
            original = getattr(cls.__call__, "_capture_original", cls.__call__)

            def wrapped(self, components, state, _original=original, _name=name, _fields=fields):
                out = _original(self, components, state)
                captured[_name] = {f: copy.deepcopy(state.get(f)) for f in _fields}
                return out

            wrapped._capture_original = original  # type: ignore[attr-defined]
            originals.append((cls, own))
            cls.__call__ = wrapped
        yield captured
    finally:
        for cls, own in reversed(originals):
            if own is _ABSENT:
                del cls.__call__
            else:
                cls.__call__ = own
