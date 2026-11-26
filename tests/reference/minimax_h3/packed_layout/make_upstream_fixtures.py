# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Execute pinned original H3 preparation functions to seal CPU metadata fixtures.

No installed diffusers or RL-Kernel layout implementation is imported. See
README.md for scope, source acquisition, regeneration, and verification.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import platform
import types
from pathlib import Path

import numpy as np
import torch

UPSTREAM_COMMIT = "f53d552036a0d1bd5570782a39cd40cfabf112bc"
UPSTREAM_PATH = "src/diffusers/modular_pipelines/minimax_h3/before_denoise.py"
UPSTREAM_SHA256 = "530b007c1d689c3ee1fc1690527f5253522d2da6b44dd326bec99faaf9f72fff"
UPSTREAM_BLOB_SHA1 = "c670467a93075ae841c7b7cdd250ceb6fa85eda9"
CONSTANTS = ("_ROPE_FRAME_RESCALE", "_ROPE_FRAMES_PER_LATENT", "_ROPE_SPATIAL_SCALE")
HELPERS = (
    "_spatial_position_grid",
    "_temporal_position_grid",
    "_frame_position_grid",
    "_fill_audio_positions",
)
METHODS = {
    "MiniMaxH3PrepareLayoutStep": "build_packed_sequence",
    "MiniMaxH3Ref2VAPrepareLayoutStep": "build_ref2va_packed_sequence",
    "MiniMaxH3SetTimestepsStep": "build_row_timesteps",
}
TIME_INPUTS = {
    "video_timestep": 0.25,
    "audio_timestep": 0.5,
    "condition_video_timestep": 1.0,
    "condition_audio_timestep": 0.75,
}


def load_upstream(path: Path) -> dict:
    """Verify complete source bytes before compiling the selected original AST."""
    data = path.read_bytes()
    blob_sha1 = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256 or blob_sha1 != UPSTREAM_BLOB_SHA1:
        raise ValueError("upstream source hash mismatch; refusing an unknown revision")

    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    found = set()
    for node in ast.parse(data.decode("utf-8")).body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & set(CONSTANTS):
                nodes.append(copy.deepcopy(node))
                found.update(names & set(CONSTANTS))
        elif isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            nodes.append(copy.deepcopy(node))
            found.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name in METHODS:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == METHODS[node.name]:
                    method = copy.deepcopy(method)
                    # Bind the original static method as a plain function.
                    method.decorator_list = []
                    nodes.append(method)
                    found.add(method.name)
    expected = set(CONSTANTS) | set(HELPERS) | set(METHODS.values())
    if found != expected:
        raise ValueError(f"missing pinned source symbols: {sorted(expected - found)}")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"torch": torch, "np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def case_specs() -> list[dict]:
    """Declare inputs only; all coordinate/index/timestep construction is upstream."""
    base = {
        "text_token_tags": [1, 1, 1],
        "num_latent_frames": 2,
        "latent_height": 4,
        "latent_width": 6,
        "num_audio_latents": 3,
        "patch_size": [1, 2, 2],
        "audio_channels": 2,
        "audio_tag": 2,
        "video_tag": 0,
    }
    specs = []

    def add(name, family="FL2VA", times=None, **overrides):
        args = copy.deepcopy(base)
        args.update(overrides)
        if family == "FL2VA":
            args.setdefault("keyframe_anchors", [])
        specs.append(
            {
                "case_id": name,
                "task_family": family,
                "layout_inputs": args,
                "timestep_inputs": dict(times or TIME_INPUTS),
            }
        )

    for name, anchors in (
        ("none", []),
        ("first", ["first"]),
        ("last", ["last"]),
        ("first_last", ["first", "last"]),
    ):
        add(f"fl2va_{name}", keyframe_anchors=anchors)
    add("fl2va_text_visual_tag", text_token_tags=[1, 0, 1], keyframe_anchors=["first"])
    add("fl2va_last_16_frames", num_latent_frames=16, keyframe_anchors=["last"])
    add("fl2va_same_timestep", times={key: 0.5 for key in TIME_INPUTS})
    add(
        "fl2va_block_tiny",
        text_token_tags=[1] * 32,
        num_latent_frames=2,
        latent_height=16,
        latent_width=24,
        num_audio_latents=20,
    )

    image = {"kind": "image", "latent_shape": [1, 24, 1, 4, 6]}
    audio = {"kind": "audio", "audio_rows_shape": [4, 32]}
    silent = {"kind": "video", "has_audio": False, "latent_shape": [1, 24, 2, 2, 4]}
    sound = {
        "kind": "video",
        "has_audio": True,
        "latent_shape": [1, 24, 3, 2, 4],
        "audio_rows_shape": [6, 32],
    }
    for name, refs in (
        ("image", [image]),
        ("audio", [audio]),
        ("silent_video", [silent]),
        ("video_audio", [sound]),
        ("mixed", [image, audio, sound, silent]),
    ):
        add(
            f"ref2va_{name}",
            family="Ref2VA",
            references=copy.deepcopy(refs),
            text_token_tags=[1, 0, 1] if name in ("image", "mixed") else [1, 1, 1],
        )
    return specs


def encode_tensor(tensor: torch.Tensor) -> dict:
    """Seal exact floating bytes in canonical little-endian order, plus dtype/shape."""
    result = {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}
    if tensor.dtype == torch.int64:
        result["values"] = tensor.tolist()
    elif tensor.dtype in (torch.float32, torch.float64):
        dtype = "<f4" if tensor.dtype == torch.float32 else "<f8"
        result["data_le_hex"] = tensor.detach().cpu().numpy().astype(dtype).tobytes().hex()
    else:
        raise ValueError(f"unsupported fixture dtype: {tensor.dtype}")
    return result


def run_case(namespace: dict, spec: dict) -> dict:
    args = copy.deepcopy(spec["layout_inputs"])
    args["text_token_tags"] = torch.tensor(args["text_token_tags"], dtype=torch.int64)
    args["patch_size"] = tuple(args["patch_size"])
    if spec["task_family"] == "Ref2VA":
        refs = args.pop("references")
        # The original method reads only kind/has_audio and latent tensor SHAPES.
        args["references"] = [types.SimpleNamespace(**ref) for ref in refs]
        args["condition_latents"] = [
            torch.empty(ref["latent_shape"]) for ref in refs if "latent_shape" in ref
        ]
        args["audio_condition_latents"] = [
            torch.empty(ref["audio_rows_shape"]) for ref in refs if "audio_rows_shape" in ref
        ]
        result = namespace["build_ref2va_packed_sequence"](**args)
    else:
        args["keyframe_anchors"] = tuple(args["keyframe_anchors"])
        result = namespace["build_packed_sequence"](**args)

    positions, tags, video, audio, text, condition_video, condition_audio = result
    timestep, timestep_indices = namespace["build_row_timesteps"](
        video, audio, condition_video, condition_audio, len(text), **spec["timestep_inputs"]
    )
    fields = {
        "position_ids": positions,
        "token_tags": tags,
        "video_indices": video,
        "audio_indices": audio,
        "text_indices": text,
        "timestep": timestep,
        "timestep_indices": timestep_indices,
    }
    return {
        **spec,
        "expected": {
            "sequence_length": len(tags),
            "num_condition_video_rows": condition_video,
            "num_condition_audio_rows": condition_audio,
            "fields": {name: encode_tensor(value) for name, value in fields.items()},
        },
    }


def make_fixture(namespace: dict) -> dict:
    with torch.device("cpu"):
        cases = [run_case(namespace, spec) for spec in case_specs()]
    return {
        "format": "h3-packed-layout-upstream.v1",
        "checkpoint_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
        "upstream": {
            "repo": "huggingface/diffusers",
            "commit": UPSTREAM_COMMIT,
            "file": UPSTREAM_PATH,
            "sha256": UPSTREAM_SHA256,
            "git_blob_sha1": UPSTREAM_BLOB_SHA1,
            "url": (
                "https://github.com/huggingface/diffusers/blob/"
                f"{UPSTREAM_COMMIT}/{UPSTREAM_PATH}"
            ),
            "methods": [f"{cls}.{method}" for cls, method in METHODS.items()],
            "helpers": list(HELPERS),
            "constants": list(CONSTANTS),
            "execution": (
                "Original AST bodies; staticmethod binding removed; annotations postponed."
            ),
            "reference_inputs": (
                "kind/has_audio attributes and shape-only latent tensors; no encoder run."
            ),
        },
        "generator": "tests/reference/minimax_h3/packed_layout/make_upstream_fixtures.py",
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "scope": (
            "Low-level preparation metadata only; "
            "no complete pipeline, model, or H-C3 qualification."
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="Compare sealed data without writing it"
    )
    args = parser.parse_args()
    fixture = make_fixture(load_upstream(args.upstream_file))
    if args.check:
        sealed = json.loads(args.out.read_text(encoding="utf-8"))
        expected_keys = fixture.keys() - {"environment"}
        sealed_keys = sealed.keys() - {"environment"}
        if sealed_keys != expected_keys:
            raise SystemExit(
                "pinned upstream fixture keys mismatch "
                f"(missing: {sorted(expected_keys - sealed_keys)}, "
                f"unexpected: {sorted(sealed_keys - expected_keys)}); no file was changed"
            )
        for key in expected_keys:
            if sealed[key] != fixture[key]:
                raise SystemExit(f"pinned upstream fixture mismatch in {key}; no file was changed")
        print(
            f"PASS: {len(fixture['cases'])} source-pinned cases match; "
            "environment recorded separately"
        )
    else:
        args.out.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(fixture['cases'])} source-pinned cases to {args.out}")


if __name__ == "__main__":
    main()
