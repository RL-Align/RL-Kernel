# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Evidence runner for RFC #420 Phase B ``phase_b_full_chain`` (MiniMax-H3).

Subcommands (each writes JSON; nothing is uploaded):

``pin-text``  Run Diffusers' own ``MiniMaxH3FL2VATextEncoderStep`` /
              ``MiniMaxH3Ref2VATextEncoderStep`` (``diffusers@f53d552``) with the released
              Qwen3-VL conditioner on a fixture request and pin ``prompt_embeds`` /
              ``text_token_tags`` to a safetensors file bound to that request's conditioner
              input. ``--layers 51`` loads the first 51 decoder layers only:
              ``hidden_states[50]`` does not depend on later layers, and Diffusers refuses <= 50.
``run``       One sealed chain run of a fixture request with the released VAEs, pinned text
              features and the released transformer (``--layers K`` keeps the first K blocks;
              ``--mode strict`` binds the RL-Kernel graph, which needs all 50 and Hopper).
``compare``   Stage-by-stage comparison and first drift of two runs.
``toy``       The CPU toy matrix (repeat runs, every injection) for both workflows.

Fixture requests use procedural media (no downloads): FL2VA with first and last keyframes,
Ref2VA with an image, a video with a soundtrack and an audio reference, both on a 256x448
target canvas, 124 frames, seed 42. The reference media are normalized with the released
recipe (2048 px image short edge, 768 px video canvas).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rl_engine.models.minimax_h3 import full_chain as fc  # noqa: E402
from rl_engine.models.minimax_h3 import packed_layout as pl  # noqa: E402

SCHEMA = "rlkernel.h3.full_chain.evidence.v1"
HEIGHT, WIDTH = 256, 448
PROMPTS = {
    "fl2va": "A red paper boat drifts across a calm pond at sunset; gentle water and birdsong.",
    "ref2va": "The bird from the picture flies through the scene of the video; the melody plays.",
}


# --- fixture requests ---------------------------------------------------------------------


def _scene(height: int, width: int, phase: float) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    y, x = y / height, x / width
    sky = np.stack([0.9 - 0.5 * y, 0.5 + 0.2 * x, 0.3 + 0.6 * y], axis=-1)
    cx, cy = 0.2 + 0.6 * phase, 0.55 - 0.1 * math.sin(math.pi * phase)
    boat = ((x - cx) ** 2 / 0.01 + (y - cy) ** 2 / 0.002) < 1.0
    sky[boat] = [0.85, 0.1, 0.1]
    ripples = 0.05 * np.sin(40 * y + 6 * x + 10 * phase)[..., None]
    return (np.clip(sky + ripples, 0, 1) * 255).round().astype(np.uint8)


def _tone(seconds: float, freqs: tuple[float, ...], channels: int, rate: int = 32_000):
    t = torch.arange(int(seconds * rate), dtype=torch.float64) / rate
    wave = sum(torch.sin(2 * math.pi * f * t) for f in freqs) / (2 * len(freqs))
    wave = wave * torch.minimum(t / 0.05, torch.ones_like(t))  # 50 ms fade-in
    out = torch.stack([wave * (1.0 - 0.3 * c) for c in range(channels)])
    return out.float()


def fixture_request(workflow: str, steps: int, height: int = HEIGHT, width: int = WIDTH):
    if workflow == "fl2va":
        return fc.FL2VARequest(
            prompt=PROMPTS["fl2va"],
            image=Image.fromarray(_scene(512, 896, 0.0)),
            last_image=Image.fromarray(_scene(540, 960, 1.0)),
            height=height,
            width=width,
            num_frames=124,
            num_inference_steps=steps,
            seed=42,
        )
    from rl_engine.models.minimax_h3 import ref2va

    frames = np.stack([_scene(360, 640, i / 29) for i in range(30)])
    return fc.Ref2VARequest(
        prompt=PROMPTS["ref2va"],
        references=(
            ref2va.ImageReference(_scene(400, 600, 0.5)),
            ref2va.VideoReference(
                frames, fps=24.0, audio=_tone(1.25, (220.0, 330.0), 1), sample_rate=32_000
            ),
            ref2va.AudioReference(_tone(0.8, (440.0, 550.0, 660.0), 2), sample_rate=32_000),
        ),
        height=height,
        width=width,
        num_frames=124,
        num_inference_steps=steps,
        seed=42,
    )


# --- provenance ---------------------------------------------------------------------------


def source_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

    try:
        status = git("status", "--porcelain", "--untracked-files=no")
        return {"commit": git("rev-parse", "HEAD"), "clean": not status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "clean": None}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- pin-text -----------------------------------------------------------------------------


def pin_text(args: argparse.Namespace) -> dict[str, Any]:
    from diffusers.modular_pipelines.minimax_h3 import encoders
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from transformers import (
        AutoConfig,
        AutoProcessor,
        AutoTokenizer,
        Qwen3VLForConditionalGeneration,
    )

    request = fixture_request(args.workflow, args.steps)
    prepared = fc.prepare(request)
    root = Path(args.text_root)
    config = AutoConfig.from_pretrained(root / "text_encoder")
    full_layers = int(config.text_config.num_hidden_layers)
    config.text_config.num_hidden_layers = args.layers
    t0 = time.time()
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", config=config, torch_dtype=torch.bfloat16
    )
    encoder = encoder.to(args.device).eval()
    load_s = time.time() - t0
    components = SimpleNamespace(
        text_encoder=encoder,
        processor=AutoProcessor.from_pretrained(root / "processor"),
        tokenizer=AutoTokenizer.from_pretrained(root / "tokenizer"),
        text_tag=pl.TEXT_TAG,
        video_tag=pl.VIDEO_TAG,
        text_encoder_layer=50,
        fps=pl.FPS,
        _execution_device=torch.device(args.device),
    )
    state = PipelineState()
    state.set("prompt", request.prompt)
    if args.workflow == "fl2va":
        block = encoders.MiniMaxH3FL2VATextEncoderStep()
        state.set("keyframes", list(prepared.text_request.media))
    else:
        block = encoders.MiniMaxH3Ref2VATextEncoderStep()
        state.set("normalized_references", list(prepared.normalized))
    t0 = time.time()
    with torch.no_grad():
        block(components, state)
    encode_s = time.time() - t0
    embeds, tags = state.get("prompt_embeds"), state.get("text_token_tags")
    import diffusers
    import transformers

    producer = {
        "conditioner": "Qwen3-VL text_encoder/ of MiniMaxAI/MiniMax-H3",
        "checkpoint_revision": pl.fp.REVISION,
        "block": type(block).__name__,
        "provider": "huggingface/diffusers",
        "provider_revision": pl.PROVIDER_REVISION,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": args.device,
        "dtype": "bfloat16",
        "decoder_layers_loaded": args.layers,
        "decoder_layers_released": full_layers,
        "hidden_state": "hidden_states[50] (after decoder layer 50, before the final norm)",
        "note": "pinned until qwen3vl_layer50_contract lands",
    }
    file_sha = fc.save_pinned_text_features(
        args.out,
        embeds,
        tags,
        input_sha256=prepared.text_request.input_sha256,
        producer=producer,
    )
    report = {
        "schema": SCHEMA,
        "kind": "pin-text",
        "workflow": args.workflow,
        "file": Path(args.out).name,
        "file_sha256": file_sha,
        "conditioner_input_sha256": prepared.text_request.input_sha256,
        "prompt_embeds": {"shape": list(embeds.shape), "dtype": str(embeds.dtype)},
        "text_token_tags": {
            "rows": int(tags.numel()),
            "vision_rows": int((tags == pl.VIDEO_TAG).sum()),
        },
        "producer": producer,
        "seconds": {"load": round(load_s, 1), "encode": round(encode_s, 1)},
        "source": source_state(),
    }
    return report


# --- run ----------------------------------------------------------------------------------


def load_transformer(directory: Path, layers: int | None) -> torch.nn.Module:
    """Released transformer, optionally its first ``layers`` blocks, in the stored dtypes.

    The checkpoint stores the declared mixed precision (FP32 input/time/output modules, BF16
    elsewhere); ``assign=True`` keeps each tensor's stored dtype, which is what
    ``from_pretrained(torch_dtype=bfloat16)`` produces, without ``accelerate``.
    """

    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors import safe_open

    config = json.loads((directory / "config.json").read_text())
    if layers is not None:
        config["num_layers"] = int(layers)
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    wanted = set(model.state_dict())
    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())
    shards: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        if key in wanted:
            shards.setdefault(shard, []).append(key)
    state: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(shards.items()):
        with safe_open(str(directory / shard), framework="pt") as handle:
            for key in keys:
                state[key] = handle.get_tensor(key)
    missing = wanted - set(state)
    if missing:
        raise ValueError(f"checkpoint misses {sorted(missing)[:5]}")
    model.load_state_dict(state, strict=True, assign=True)
    # Non-persistent buffers are not in the checkpoint: rebuild the module that owns them
    # (``rope.inv_freq``) on the CPU exactly as ``__init__`` computes it.
    rope = type(model.rope)(rope_freq_dim=config["rope_freq_dim"], rope_theta=config["rope_theta"])
    model.rope = rope
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise ValueError(f"tensors left on the meta device: {left[:5]}")
    return model.eval()


class OffloadedVisualVAE:
    """``VisualVAEContract`` whose VAE lives on the CPU between calls (H100 memory budget)."""

    def __init__(self, contract: Any, device: str) -> None:
        self.contract, self.device = contract, device

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.contract.vae.to(self.device)
        try:
            return getattr(self.contract, name)(*args, **kwargs)
        finally:
            self.contract.vae.to("cpu")
            torch.cuda.empty_cache()

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("encode", *args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("decode", *args, **kwargs)


class OffloadedDenoiser(fc.TransformerDenoiser):
    """The transformer moves to the GPU for the sampling loop only."""

    def __init__(self, *args: Any, run_device: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.run_device = run_device

    def session(self):  # type: ignore[override]
        import contextlib

        @contextlib.contextmanager
        def scope():
            self.model.to(self.run_device)
            try:
                with super(OffloadedDenoiser, self).session():
                    yield
            finally:
                self.model.to("cpu")
                torch.cuda.empty_cache()

        return scope()


def build_components(args: argparse.Namespace, variant: str) -> tuple[fc.ChainComponents, dict]:
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    from rl_engine.models.minimax_h3 import audio_vae as av
    from rl_engine.models.minimax_h3 import visual_vae as vv

    device = args.device
    info: dict[str, Any] = {}
    t0 = time.time()
    resident = "cpu" if args.offload else device
    vae = vv.load_release_vae(AutoencoderKLMiniMaxH3, args.vae_dir, resident)
    vae.enable_tiling(256, 256, 64, 64)
    visual: Any = vv.VisualVAEContract(vae)
    if args.offload:
        visual = OffloadedVisualVAE(visual, device)
    checkpoint = av.AudioVAECheckpoint.read(
        args.audio_vae_dir, verify_sha256=av.pinned_audio_vae_sha256(variant)
    )
    audio = av.H3AudioVAEContract.from_checkpoint(checkpoint, device)
    info["vae_load_s"] = round(time.time() - t0, 1)
    identified = pl.identify_transformer_dir(args.transformer_dir)
    if identified != variant:
        raise pl.VariantMismatch(f"{args.transformer_dir} holds {identified}, request is {variant}")
    t0 = time.time()
    model = load_transformer(Path(args.transformer_dir), args.layers)
    if not args.offload:
        model = model.to(device)
    info["transformer_load_s"] = round(time.time() - t0, 1)
    identity = {
        "transformer_dir_identified_as": identified,
        "layers_loaded": len(model.transformer_blocks),
        "layer_truncated": len(model.transformer_blocks) != 50,
    }
    mode = None if args.mode == "provider" else args.mode
    denoiser: fc.TransformerDenoiser
    if args.offload:
        denoiser = OffloadedDenoiser(
            model, variant=variant, mode=mode, identity=identity, run_device=device
        )
    else:
        denoiser = fc.TransformerDenoiser(model, variant=variant, mode=mode, identity=identity)
    text = fc.PinnedTextFeatures(args.text, expected_sha256=args.text_sha256)
    info["identity"] = identity
    return fc.ChainComponents(text, denoiser, visual, audio, device), info


def _parse_injection(spec: str | None) -> fc.DriftInjection | None:
    return fc.DriftInjection(**json.loads(spec)) if spec else None


def _capture(spec: str | None):
    """``"1:transformer_blocks.2,1:proj_out"`` -> capture predicate; ``"all"`` keeps all."""

    if not spec:
        return None
    if spec == "all":
        return lambda step, path: True
    wanted = {(int(s), m) for s, m in (item.split(":", 1) for item in spec.split(","))}
    return lambda step, path: (step, path) in wanted


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    variant = "FL2VA" if args.workflow == "fl2va" else "Ref2VA"
    request = fixture_request(args.workflow, args.steps)
    components, info = build_components(args, variant)
    t0 = time.time()
    result = fc.run_chain(
        request,
        components,
        label=args.label,
        keep_tensors=args.keep_tensors,
        capture=_capture(args.capture),
        inject=_parse_injection(args.inject),
        metadata={"runner": "tools/validation/models/h3_full_chain.py", "source": source_state()},
    )
    info["chain_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(fc.dumps(result.artifact))
    if args.tensors:
        result.save_tensors(args.tensors)
    return {
        "schema": SCHEMA,
        "kind": "run",
        "workflow": args.workflow,
        "label": args.label,
        "artifact": Path(args.out).name,
        "artifact_sha256": result.artifact["artifact_sha256"],
        "final_latent_sha256": result.artifact["trajectory"]["final_latent_sha256"],
        "stages": {s["stage"]: dict(s["components"]) for s in result.artifact["stages"]},
        "secondary_metrics": result.artifact["secondary_metrics"],
        "sequence_length": result.artifact["layout"]["sequence_length"],
        "execution": {
            s["stage"]: s["provenance"]
            for s in result.artifact["stages"]
            if s["stage"] == "denoise"
        }["denoise"]["execution"],
        "info": info,
        "runtime": result.artifact["runtime"],
        "source": source_state(),
    }


# --- compare ------------------------------------------------------------------------------


def compare(args: argparse.Namespace) -> dict[str, Any]:
    lhs = fc.loads(Path(args.lhs).read_text())
    rhs = fc.loads(Path(args.rhs).read_text())
    report = fc.compare(lhs, rhs, tensors_lhs=args.tensors_lhs, tensors_rhs=args.tensors_rhs)
    return {"schema": SCHEMA, "kind": "compare", **report}


# --- toy ----------------------------------------------------------------------------------


TOY_STAGE_INJECTIONS = {
    "fl2va": [
        ("media_preprocess", "keyframe_pixels.1", 3),
        ("text_conditions", "prompt_embeds", 4),
        ("visual_vae_encode", "condition_latents.0", 0),
        ("initial_rows", "video_rows", 13),
        ("unpack", "video_latents", 5),
        ("video_decode", "video", 0),
    ],
    "ref2va": [
        ("media_preprocess", "reference.1.waveform", 0),
        ("text_conditions", "prompt_embeds", 9),
        ("visual_vae_encode", "condition_latents.1", 2),
        ("audio_vae_encode", "audio_rows.1", 7),
        ("initial_rows", "audio_rows", 61),
        ("audio_decode", "waveform", 1),
    ],
}
TOY_MODULE_INJECTIONS = {
    "fl2va": [
        (0, "transformer_blocks.0", 20),
        (2, "transformer_blocks.2", 300),
        (1, "proj_in", 40),
    ],
    "ref2va": [(1, "transformer_blocks.1", 30), (2, "audio_proj_in", 70), (0, "norm_out", 500)],
}


def _expectation(inject: fc.DriftInjection, drift: dict[str, Any] | None) -> dict[str, Any]:
    if drift is None:
        return {"localized": False, "reason": "no drift found"}
    if inject.stage != "denoise":
        ok = (drift["stage"], drift["component"]) == (inject.stage, inject.component)
        if (inject.stage, inject.component) in fc._COMPONENT_ROWS:  # rows mapped to the layout
            ok = ok and drift["first_row"] == inject.row
        return {"localized": ok, "first_index": drift.get("first_index")}
    found = drift.get("transformer") or {}
    ok = (
        drift["stage"] == "denoise"
        and drift.get("step") == inject.step
        and found.get("module") == inject.module
        and found.get("first_row") == inject.row
    )
    return {"localized": ok}


def toy(args: argparse.Namespace) -> dict[str, Any]:
    from tests.models.minimax_h3 import _toy_chain as toy_chain

    out: dict[str, Any] = {"schema": SCHEMA, "kind": "toy", "source": source_state(), "cases": []}
    for workflow in ("fl2va", "ref2va"):
        variant = "FL2VA" if workflow == "fl2va" else "Ref2VA"
        make = toy_chain.fl2va_request if workflow == "fl2va" else toy_chain.ref2va_request
        request = make()
        clean = fc.run_chain(
            request, toy_chain.components(variant), label="clean", keep_tensors=True
        )
        again = fc.run_chain(
            request, toy_chain.components(variant), label="again", keep_tensors=True
        )
        rep = fc.compare(
            clean.artifact, again.artifact, tensors_lhs=clean.tensors, tensors_rhs=again.tensors
        )
        out["cases"].append(
            {
                "workflow": workflow,
                "case": "repeat",
                "first_drift": rep["first_drift"],
                "gate": rep["gate"],
                "artifact_sha256": [
                    clean.artifact["artifact_sha256"],
                    again.artifact["artifact_sha256"],
                ],
            }
        )
        injections = [
            fc.DriftInjection(stage, component=component, row=row)
            for stage, component, row in TOY_STAGE_INJECTIONS[workflow]
        ] + [
            fc.DriftInjection("denoise", step=step, module=module, row=row)
            for step, module, row in TOY_MODULE_INJECTIONS[workflow]
        ]
        for inject in injections:
            run_ = fc.run_chain(
                request,
                toy_chain.components(variant),
                label="injected",
                keep_tensors=True,
                inject=inject,
            )
            rep = fc.compare(
                clean.artifact, run_.artifact, tensors_lhs=clean.tensors, tensors_rhs=run_.tensors
            )
            out["cases"].append(
                {
                    "workflow": workflow,
                    "case": "injection",
                    "injection": inject.to_dict(),
                    "first_drift": rep["first_drift"],
                    "gate": rep["gate"],
                    **_expectation(inject, rep["first_drift"]),
                }
            )
    injected = [c for c in out["cases"] if c["case"] == "injection"]
    out["summary"] = {
        "repeat_runs_equal": all(
            c["first_drift"] is None for c in out["cases"] if c["case"] == "repeat"
        ),
        "injections": len(injected),
        "localized": sum(bool(c["localized"]) for c in injected),
        "runtime": fc.runtime_provenance("cpu"),
    }
    return out


# --- main ---------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pin-text")
    p.add_argument("--workflow", choices=("fl2va", "ref2va"), required=True)
    p.add_argument(
        "--text-root", required=True, help="dir with text_encoder/ processor/ tokenizer/"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--layers", type=int, default=51)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=4)

    p = sub.add_parser("run")
    p.add_argument("--workflow", choices=("fl2va", "ref2va"), required=True)
    p.add_argument("--transformer-dir", required=True)
    p.add_argument("--vae-dir", required=True)
    p.add_argument("--audio-vae-dir", required=True)
    p.add_argument("--text", required=True, help="pinned text features (pin-text output)")
    p.add_argument("--text-sha256")
    p.add_argument("--layers", type=int, default=None, help="first K blocks (default: all 50)")
    p.add_argument("--mode", choices=("provider", "strict", "audit"), default="provider")
    p.add_argument("--steps", type=int, default=4, help="num_inference_steps (grid points)")
    p.add_argument("--label", default="run")
    p.add_argument("--device", default="cuda")
    p.add_argument("--offload", action="store_true", help="keep idle models on the CPU")
    p.add_argument("--keep-tensors", action="store_true")
    p.add_argument("--capture", help='"all" or "step:module,..." transformer outputs to keep')
    p.add_argument("--inject", help="DriftInjection as JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--tensors")
    p.add_argument("--report", required=True)

    p = sub.add_parser("compare")
    p.add_argument("lhs")
    p.add_argument("rhs")
    p.add_argument("--tensors-lhs")
    p.add_argument("--tensors-rhs")
    p.add_argument("--report", required=True)

    p = sub.add_parser("toy")
    p.add_argument("--report", required=True)

    args = parser.parse_args()
    handler = {"pin-text": pin_text, "run": run, "compare": compare, "toy": toy}[args.command]
    report = handler(args)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: report[k] for k in report if k in ("kind", "summary", "first_drift")}))


if __name__ == "__main__":
    main()
