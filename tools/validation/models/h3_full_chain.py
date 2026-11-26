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
``plot*``     Charts, each with a Markdown table of the same numbers: ``plot`` (drift map of two
              runs), ``plot-stages``, ``plot-text``, ``plot-text-hosts`` (text check on several
              hosts), ``plot-time`` and ``plot-memory`` (peak GPU memory from nvidia-smi traces).

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
from collections.abc import Mapping
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


def stream_blocks(model: torch.nn.Module, device: str) -> None:
    """Keep the blocks in pinned host memory; each moves to ``device`` for its own forward.

    Everything else of the model lives on ``device``. A block's parameters are replaced by a
    device copy right before its forward and by the host tensors right after: the same weight
    bytes and the same kernels as a resident model, at the memory cost of one block.
    """

    for name, child in model.named_children():
        if name != "transformer_blocks":
            child.to(device)
    for block in model.transformer_blocks:
        if any(True for _ in block.buffers()):
            raise ValueError("block streaming assumes blocks without buffers")
        host = [(p, p.data.pin_memory()) for p in block.parameters()]
        for p, h in host:
            p.data = h

        def before(module: torch.nn.Module, args: Any, host=host) -> None:
            for p, h in host:
                p.data = h.to(device, non_blocking=True)

        def after(module: torch.nn.Module, args: Any, output: Any, host=host) -> None:
            for p, h in host:
                p.data = h

        block.register_forward_pre_hook(before)
        block.register_forward_hook(after)


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
    if args.stream_blocks:
        stream_blocks(model, device)
    elif not args.offload:
        model = model.to(device)
    info["transformer_load_s"] = round(time.time() - t0, 1)
    identity = {
        "transformer_dir_identified_as": identified,
        "layers_loaded": len(model.transformer_blocks),
        "layer_truncated": len(model.transformer_blocks) != 50,
        "placement": (
            "stream-blocks" if args.stream_blocks else "offload" if args.offload else "resident"
        ),
    }
    mode = None if args.mode == "provider" else args.mode
    ops = None
    if mode is not None and args.attention_query_chunk:
        from rl_engine.validation.models.h3_block import CandidateOps
        from rl_engine.validation.models.h3_full_model import QueryChunkAttention

        ops = CandidateOps()
        ops.attention = QueryChunkAttention(ops.attention, args.attention_query_chunk)
        identity["attention_query_chunk"] = args.attention_query_chunk
    denoiser: fc.TransformerDenoiser
    if args.offload:
        denoiser = OffloadedDenoiser(
            model, variant=variant, mode=mode, identity=identity, ops=ops, run_device=device
        )
    else:
        denoiser = fc.TransformerDenoiser(
            model, variant=variant, mode=mode, identity=identity, ops=ops
        )
    text: Any
    if args.text_source == "pinned":
        if not args.text:
            raise SystemExit("--text-source pinned needs --text")
        text = fc.PinnedTextFeatures(args.text, expected_sha256=args.text_sha256)
    else:
        if not args.text_root:
            raise SystemExit("--text-source qwen3vl needs --text-root")
        t0 = time.time()
        text = fc.Qwen3VLLayer50Conditioner.from_release(
            args.text_root, device=args.text_device or device
        )
        info["text_encoder_load_s"] = round(time.time() - t0, 1)
    info["identity"] = identity
    return fc.ChainComponents(text, denoiser, visual, audio, device), info


TEXT_SOURCES = {"pinned": fc.TEXT_SOURCE_PINNED, "qwen3vl": fc.TEXT_SOURCE_QWEN3VL}


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
        text_source=TEXT_SOURCES[args.text_source],
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
    report = fc.compare(
        lhs,
        rhs,
        tensors_lhs=args.tensors_lhs,
        tensors_rhs=args.tensors_rhs,
        allow_text_source_change=args.allow_text_source_change,
    )
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
            request,
            toy_chain.components(variant),
            label="clean",
            keep_tensors=True,
            text_source=toy_chain.TEXT_SOURCE,
        )
        again = fc.run_chain(
            request,
            toy_chain.components(variant),
            label="again",
            keep_tensors=True,
            text_source=toy_chain.TEXT_SOURCE,
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
                text_source=toy_chain.TEXT_SOURCE,
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


# --- text-check ---------------------------------------------------------------------------


def text_check(args: argparse.Namespace) -> dict[str, Any]:
    """qwen3vl_layer50_contract features of a fixture request against a pinned features file."""

    from safetensors import safe_open

    request = fixture_request(args.workflow, args.steps)
    text_request = fc.prepare(request).text_request
    t0 = time.time()
    conditioner = fc.Qwen3VLLayer50Conditioner.from_release(args.text_root, device=args.device)
    load_s = time.time() - t0
    t0 = time.time()
    ours = conditioner.encode(text_request)
    encode_s = time.time() - t0
    with safe_open(args.pinned, framework="pt") as handle:
        meta = json.loads(handle.metadata()["rlkernel.h3.text_features"])
        pinned_embeds = handle.get_tensor("prompt_embeds")
        pinned_tags = handle.get_tensor("text_token_tags")
    a, b = ours.prompt_embeds.detach().cpu(), pinned_embeds
    same_shape = a.shape == b.shape and a.dtype == b.dtype
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "text-check",
        "workflow": args.workflow,
        "device": args.device,
        "pinned": {
            "file_sha256": _sha_file(Path(args.pinned)),
            "producer_device": meta["producer"].get("device"),
            "producer_block": meta["producer"].get("block"),
            "conditioner_input_sha256": meta["conditioner_input_sha256"],
        },
        "contract": {k: v for k, v in ours.provenance.items() if k not in ("runtime",)},
        "runtime": ours.provenance.get("runtime"),
        "same_conditioner_input": meta["conditioner_input_sha256"] == text_request.input_sha256,
        "tags_equal": torch.equal(ours.text_token_tags, pinned_tags.long()),
        "shape": list(a.shape),
        "pinned_shape": list(b.shape),
        "bitwise_equal": bool(same_shape and torch.equal(a, b)),
        "seconds": {"load": round(load_s, 1), "encode": round(encode_s, 1)},
        "source": source_state(),
    }
    if same_shape and not report["bitwise_equal"]:
        diff = (a.double() - b.double()).abs()
        rows = (a != b).reshape(a.shape[1], -1).any(dim=1)
        first = int(torch.nonzero(rows)[0, 0])
        tags = ours.text_token_tags
        report.update(
            max_abs_diff=float(diff.max()),
            mean_abs_diff=float(diff.mean()),
            differing_elements=int((a != b).sum()),
            numel=int(a.numel()),
            differing_tokens=int(rows.sum()),
            first_differing_token=first,
            first_differing_token_tag=int(tags[first]),
            differing_text_tokens=int(rows[tags == pl.TEXT_TAG].sum()),
            differing_vision_tokens=int(rows[tags == pl.VIDEO_TAG].sum()),
            relative_l2=float((a.double() - b.double()).norm() / b.double().norm()),
        )
    return report


# --- plot ---------------------------------------------------------------------------------

EQUAL, DIFFERS, INK, INK2, SURFACE = "#d9d8d3", "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"


def plot(args: argparse.Namespace) -> dict[str, Any]:
    """Drift map of two runs: one row per step, one cell per probed module output (execution
    order) and per trajectory tensor; blue where the two runs' digests differ."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    lhs = fc.loads(Path(args.lhs).read_text())
    rhs = fc.loads(Path(args.rhs).read_text())
    steps_a, steps_b = lhs["transformer_probe"]["steps"], rhs["transformer_probe"]["steps"]
    modules = [name for name, _ in steps_a[0]]
    traj = [("velocity", "video"), ("velocity", "audio"), ("latent", "video"), ("latent", "audio")]
    gap = 1.0
    width = len(modules) + gap + len(traj)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.16 * width + 2.5), 0.55 * len(steps_a) + 2.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    first = None
    for step, (sa, sb) in enumerate(zip(steps_a, steps_b, strict=True)):
        y = len(steps_a) - 1 - step
        for i, (ea, eb) in enumerate(zip(sa, sb, strict=True)):
            differs = ea != eb
            if differs and first is None:
                first = (i, y, ea[0], step)
            ax.add_patch(
                Rectangle((i + 0.06, y + 0.08), 0.88, 0.84, color=DIFFERS if differs else EQUAL)
            )
        ra, rb = lhs["trajectory"]["step_records"][step], rhs["trajectory"]["step_records"][step]
        for j, (kind, modality) in enumerate(traj):
            differs = ra[f"{kind}_sha256"][modality] != rb[f"{kind}_sha256"][modality]
            x = len(modules) + gap + j
            if differs and first is None:  # only the trajectory differs at this step
                first = (x, y, f"{kind} {modality}", step)
            ax.add_patch(
                Rectangle((x + 0.06, y + 0.08), 0.88, 0.84, color=DIFFERS if differs else EQUAL)
            )
    device = lhs["runtime"].get("device_name", lhs["runtime"].get("device", "?"))
    case = _case_id(lhs)
    subtitle = f"{device}, transformer {case}; no difference"
    if first is not None:
        i, y, name, step = first
        ax.add_patch(Rectangle((i, y), 1, 1, fill=False, edgecolor=INK, linewidth=1.5))
        subtitle = f"{device}, transformer {case}; first drift (outlined): step {step}, {name}"
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.set_xlim(0, width)
    ax.set_ylim(0, len(steps_a))
    ax.set_yticks([len(steps_a) - 0.5 - s for s in range(len(steps_a))])
    ax.set_yticklabels([f"step {s}" for s in range(len(steps_a))], color=INK2, fontsize=9)
    blocks = [n for n in modules if n.startswith("transformer_blocks.")]

    def labelled(name: str) -> bool:
        if not name.startswith("transformer_blocks.") or len(blocks) <= 10:
            return True
        index = int(name.split(".")[1].split("#")[0])
        return index % 5 == 0 or index == len(blocks) - 1 or (first and first[2] == name)

    marks = {
        i: n.replace("transformer_blocks.", "block ") for i, n in enumerate(modules) if labelled(n)
    }
    ticks = [i + 0.5 for i in marks] + [len(modules) + gap + j + 0.5 for j in range(len(traj))]
    labels = list(marks.values()) + [f"{k} {m}" for k, m in traj]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=7, color=INK2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(args.title, loc="left", fontsize=11, color=INK, pad=22)
    ax.legend(
        handles=[Patch(color=EQUAL, label="byte-equal"), Patch(color=DIFFERS, label="differs")],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.2,
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    _save(fig, args.png)
    report = (
        load_report(args.compare)
        if args.compare
        else fc.compare(lhs, rhs, allow_text_source_change=True)
    )
    step_rows = []
    for step, (sa, sb) in enumerate(zip(steps_a, steps_b, strict=True)):
        differing = [ea[0] for ea, eb in zip(sa, sb, strict=True) if ea != eb]
        ra, rb = lhs["trajectory"]["step_records"][step], rhs["trajectory"]["step_records"][step]
        step_rows.append(
            [step, f"{len(differing)} / {len(sa)}", differing[0] if differing else "none"]
            + [
                _eq(ra[f"{kind}_sha256"][modality] == rb[f"{kind}_sha256"][modality])
                for kind, modality in traj
            ]
        )
    step_headers = ["Step", "Module outputs that differ", "First that differs"] + [
        f"{k} {m}" for k, m in traj
    ]
    return {
        "schema": SCHEMA,
        "kind": "plot",
        "png": Path(args.png).name,
        "device": device,
        "transformer": case,
        "first_drift_cell": None if first is None else {"step": first[3], "module": first[2]},
        "tables": {
            "first_drift": _first_drift_table(report),
            "steps": {"headers": step_headers, "rows": step_rows},
            "stages": _stage_table([("A vs B", report)]),
        },
        "markdown": "\n\n".join(
            [
                _md(*_first_drift_table(report).values()),
                _md(step_headers, step_rows),
                _md(*_stage_table([(args.pair_label, report)]).values()),
            ]
        ),
        "source": source_state(),
    }


# --- chart helpers ------------------------------------------------------------------------

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # palette slots 1-3 (validated adjacent/all-pairs)


def _save(fig: Any, path: str) -> None:
    fig.tight_layout()
    # no timestamp or software string beyond the library name
    fig.savefig(path, dpi=160, facecolor=SURFACE, metadata={"Software": "matplotlib"})


def _eq(equal: bool) -> str:
    return "equal" if equal else "differs"


def _md(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    lines.append("|" + "---|" * len(headers))
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def load_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _case_id(artifact: Mapping[str, Any]) -> str:
    for record in artifact["stages"]:
        if record["stage"] == "denoise":
            return str(record["provenance"]["execution"].get("case_id"))
    return "?"


def _row_text(row: Mapping[str, Any] | None, found: Mapping[str, Any] | None = None) -> str:
    """The packed identity of ``row``; without one, what is known about the location."""

    if not row:
        found = found or {}
        if "row_evidence" in found:
            return f"n/a ({found['row_evidence']})"
        if "first_row" in found:
            why = found.get("row_identity", "no packed identity")
            return f"row {found['first_row']} of the tensor ({why})"
        if "first_index" in found:
            return f"index {found['first_index']} (no packed rows)"
        return "n/a (no tensors kept)"
    role = row.get("role", "")
    if row.get("modality") == "video" and role == "target":
        return (
            f"target video, latent frame {row['latent_frame']}, "
            f"patch ({row['patch_row']}, {row['patch_col']})"
        )
    if role == "condition":
        return f"{row['modality']} condition row {row['condition_row']}"
    if role == "reference":
        return f"audio reference row {row['reference_row']}"
    if role == "target":
        return f"target audio, {row['stereo_channel']}, latent {row['latent_index']}"
    return str(role)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e7:
            return str(int(value))
        return f"{value:.{digits}g}"
    return str(value)


def _trajectory_text(report: Mapping[str, Any]) -> str:
    """The trajectory's own first drift: nested under a ``denoise`` first drift (with rows when
    tensors are kept), else from the report's independent trajectory comparison."""

    fd = report["first_drift"]
    if fd["stage"] == "denoise" and "trajectory" in fd:
        traj = fd["trajectory"]
        if traj is None:
            return "none (velocities and latents equal)"
        return (
            f"{traj['kind']} {traj['modality']}, step {traj['step']}, "
            f"{_fmt(traj.get('differing_rows'))} rows, max |Δ| {_fmt(traj.get('max_abs_diff'))}"
        )
    whole = report.get("trajectory") or {}
    if "comparable" in whole and not whole["comparable"]:
        return f"not comparable: {whole['reason']}"
    if whole.get("bitwise_equal"):
        return "none (velocities and latents equal)"
    drift = whole.get("first_drift")
    if drift is None:
        return "n/a"
    return f"{drift['kind']} {drift['modality']}, step {drift['step']}"


def _first_drift_table(report: Mapping[str, Any]) -> dict[str, Any]:
    fd = report["first_drift"]
    headers = [
        "First drift: stage",
        "Step",
        "Module",
        "Packed row",
        "Row",
        "Module max |Δ|",
        "Trajectory first drift",
    ]
    if fd is None:
        return {"headers": headers, "rows": [["none"] + ["n/a"] * 6]}
    # inside ``denoise`` the location is the module's, elsewhere the stage component's
    found = (fd.get("transformer") or {}) if fd["stage"] == "denoise" else fd
    row = found.get("row") or {}
    return {
        "headers": headers,
        "rows": [
            [
                fd["stage"],
                _fmt(fd.get("step")),
                found.get("module") or "n/a",
                _fmt(
                    row.get(
                        "packed_row", None if "row_identity" in found else found.get("first_row")
                    )
                ),
                _row_text(row, found),
                _fmt(found.get("max_abs_diff")),
                _trajectory_text(report),
            ]
        ],
    }


PROBE_ONLY = "intermediates only"


def _stage_state(stage: Mapping[str, Any]) -> str:
    """equal / differs / intermediates only (only the transformer probe digests differ)."""

    if stage["equal"]:
        return "equal"
    if stage["differing_components"] == ["transformer_probe"]:
        return PROBE_ONLY
    return "differs"


def _stage_table(pairs: list[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    headers = ["Pair"] + list(fc.STAGES) + ["Hard gate"]
    rows = []
    for label, report in pairs:
        state = {s["stage"]: _stage_state(s) for s in report["stages"]}
        rows.append([label] + [state[st] for st in fc.STAGES] + [report["gate"]["hard_gate"]])
    return {"headers": headers, "rows": rows}


def plot_stages(args: argparse.Namespace) -> dict[str, Any]:
    """Stage-by-stage digest equality of several run pairs (one row per compare report)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    pairs = []
    for item in args.pair:
        label, _, path = item.partition("=")
        pairs.append((label, load_report(path)))
    stages = list(fc.STAGES)
    gap = 0.6
    fig, ax = plt.subplots(figsize=(10.5, 0.5 * len(pairs) + 3.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for r, (_label, report) in enumerate(pairs):
        y = len(pairs) - 1 - r
        state = {st["stage"]: _stage_state(st) for st in report["stages"]}
        for c, stage in enumerate(stages):
            x = c + (gap if stage in fc.SECONDARY_STAGES else 0.0)
            cell = (x + 0.05, y + 0.08)
            if state[stage] == PROBE_ONLY:
                hatched = dict(facecolor=EQUAL, edgecolor=DIFFERS, hatch="///", linewidth=0)
                ax.add_patch(Rectangle(cell, 0.9, 0.84, **hatched))
            else:
                fill = EQUAL if state[stage] == "equal" else DIFFERS
                ax.add_patch(Rectangle(cell, 0.9, 0.84, color=fill))
    ax.set_xlim(0, len(stages) + gap)
    ax.set_ylim(0, len(pairs))
    ax.set_yticks([len(pairs) - 0.5 - r for r in range(len(pairs))])
    ax.set_yticklabels([label for label, _ in pairs], fontsize=8, color=INK2)
    ax.set_xticks(
        [c + 0.5 + (gap if st in fc.SECONDARY_STAGES else 0.0) for c, st in enumerate(stages)]
    )
    ax.set_xticklabels(stages, rotation=40, ha="right", fontsize=8, color=INK2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(args.title, loc="left", fontsize=11, color=INK, pad=22)
    ax.text(
        0,
        1.02,
        "hard gate: every stage up to unpack, transformer intermediates excluded; "
        "decodes (right) are secondary",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK2,
        va="bottom",
    )
    ax.legend(
        handles=[
            Patch(color=EQUAL, label="byte-equal"),
            Patch(facecolor=EQUAL, edgecolor=DIFFERS, hatch="///", label=PROBE_ONLY),
            Patch(color=DIFFERS, label="differs"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.36),
        borderaxespad=0.2,
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    _save(fig, args.png)
    table = _stage_table(pairs)
    return {
        "schema": SCHEMA,
        "kind": "plot-stages",
        "png": Path(args.png).name,
        "pairs": [label for label, _ in pairs],
        "tables": {"stages": table},
        "markdown": _md(table["headers"], table["rows"]),
        "source": source_state(),
    }


def plot_text(args: argparse.Namespace) -> dict[str, Any]:
    """Contract text features against the pinned file: relative errors, CPU and CUDA."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = load_report(args.analysis)
    metrics = [
        ("relative L2, all tokens", "relative_l2"),
        ("median per-token rel. error, text", "text"),
        ("median per-token rel. error, vision", "vision"),
    ]
    rows, values = [], {}
    for wf in ("fl2va", "ref2va"):
        for dev in ("cpu", "cuda"):
            check = load_report(str(Path(args.dir) / f"{wf}-{dev}.json"))
            if check["bitwise_equal"]:
                vals = {"relative_l2": 0.0, "text": 0.0, "vision": 0.0}
            else:
                # ``--analysis`` is the per-token analysis of the CUDA runs only
                a = analysis[wf] if dev == "cuda" else {}
                vals = {
                    "relative_l2": check["relative_l2"],
                    "text": a.get("per_token_relative_l2_text"),
                    "vision": a.get("per_token_relative_l2_vision"),
                }
            values[wf, dev] = vals
            rows.append(
                [
                    wf.upper().replace("REF2VA", "Ref2VA").replace("FL2VA", "FL2VA"),
                    dev.upper(),
                    check["shape"][1],
                    "yes" if check["bitwise_equal"] else "no",
                ]
                + [_fmt(vals[key]) for _, key in metrics]
                + [_fmt(check.get("max_abs_diff")), _fmt(check.get("first_differing_token"))]
            )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    floor = 1e-4
    for ax, wf in zip(axes, ("fl2va", "ref2va"), strict=True):
        ax.set_facecolor(SURFACE)
        for m, (_label, key) in enumerate(metrics):
            for d, dev in enumerate(("cuda", "cpu")):
                y = m * 3 + d
                v = values[wf, dev][key]
                if v is None:
                    ax.text(floor * 1.4, y, "n/a", va="center", fontsize=8, color=INK2)
                elif v == 0.0:
                    ax.plot([floor], [y], marker="o", ms=6, color=SERIES[1])
                    ax.text(floor * 1.4, y, "= 0, bitwise", va="center", fontsize=8, color=INK)
                else:
                    ax.barh(y, v - floor, height=0.8, color=SERIES[0], left=floor)  # ends at v
                    ax.text(v * 1.08, y, f"{v:.3g}", va="center", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(floor, 3.0)
        ax.set_yticks([m * 3 + d for m in range(len(metrics)) for d in range(2)])
        ax.set_yticklabels(
            [f"{label} ({dev})" for label, _ in metrics for dev in ("CUDA", "CPU")],
            fontsize=7.5,
            color=INK2,
        )
        ax.invert_yaxis()
        ax.set_title(
            {"fl2va": "FL2VA (259 tokens)", "ref2va": "Ref2VA (8,217 tokens)"}[wf],
            loc="left",
            fontsize=9.5,
            color=INK,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color=EQUAL, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=7.5)
    axes[0].set_xlabel("relative error vs the pinned CPU features (log)", fontsize=8, color=INK2)
    axes[1].set_xlabel("relative error vs the pinned CPU features (log)", fontsize=8, color=INK2)
    fig.suptitle(args.title, x=0.01, ha="left", fontsize=11, color=INK)
    _save(fig, args.png)
    headers = [
        "Workflow",
        "Contract on",
        "Tokens",
        "Bitwise equal",
        *[label for label, _ in metrics],
        "max |Δ|",
        "First differing token",
    ]
    return {
        "schema": SCHEMA,
        "kind": "plot-text",
        "png": Path(args.png).name,
        "tables": {"text_check": {"headers": headers, "rows": rows}},
        "markdown": _md(headers, rows),
        "source": source_state(),
    }


def plot_time(args: argparse.Namespace) -> dict[str, Any]:
    """Load and chain wall time per run (run reports), grouped bars."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for item in args.run:
        label, _, path = item.partition("=")
        runs.append((label, load_report(path)))
    series = [
        ("VAE load", "vae_load_s"),
        ("transformer load", "transformer_load_s"),
        ("chain", "chain_s"),
    ]
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(runs) + 1.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    height = 0.26
    rows = []
    for r, (label, rep) in enumerate(runs):
        info = rep["info"]
        rows.append([label] + [f"{float(info[key]):.1f}" for _, key in series])
        for k, (name, key) in enumerate(series):
            y = r + (k - 1) * height
            value = float(info[key])
            ax.barh(y, value, height=height * 0.9, color=SERIES[k], label=name if r == 0 else None)
            ax.text(value + 1.5, y, f"{value:.1f} s", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([label for label, _ in runs], fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("seconds (shared GPU, indicative)", fontsize=8, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=EQUAL, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlim(0, max(float(rep["info"][k]) for _, rep in runs for _, k in series) * 1.15)
    ax.set_title(args.title, loc="left", fontsize=11, color=INK, pad=22)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=8)
    _save(fig, args.png)
    headers = ["Run"] + [f"{name} (s)" for name, _ in series]
    return {
        "schema": SCHEMA,
        "kind": "plot-time",
        "png": Path(args.png).name,
        "tables": {"time": {"headers": headers, "rows": rows}},
        "markdown": _md(headers, rows),
        "source": source_state(),
    }


def _label_path(item: str) -> tuple[str, str]:
    label, sep, path = item.partition("=")
    if not sep or not label or not path:
        raise SystemExit(f"expected LABEL=PATH, got {item!r}")
    return label, path


def plot_text_hosts(args: argparse.Namespace) -> dict[str, Any]:
    """Contract text features against the pinned file on several hosts: CPU and CUDA.

    Only what each ``text-check`` report holds (no per-token analysis, which needs the
    feature tensors): relative L2 over all tokens, max |Δ| and the differing tokens.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hosts = [_label_path(item) for item in args.host]
    checks = {
        (host, wf, dev): load_report(str(Path(d) / f"{wf}-{dev}.json"))
        for host, d in hosts
        for wf in ("fl2va", "ref2va")
        for dev in ("cpu", "cuda")
    }
    rows = []
    for host, _ in hosts:
        for wf in ("fl2va", "ref2va"):
            for dev in ("cpu", "cuda"):
                c = checks[host, wf, dev]
                if c["workflow"] != wf or c["device"] != dev:
                    raise SystemExit(f"{host} {wf}-{dev}.json holds {c['workflow']}/{c['device']}")
                rows.append(
                    [
                        host,
                        "FL2VA" if wf == "fl2va" else "Ref2VA",
                        dev.upper(),
                        c["shape"][1],
                        "yes" if c["bitwise_equal"] else "no",
                        _fmt(0.0 if c["bitwise_equal"] else c["relative_l2"]),
                        _fmt(c.get("max_abs_diff", 0.0)),
                        f"{c.get('differing_text_tokens', 0)} / "
                        f"{c.get('differing_vision_tokens', 0)} (of {c['shape'][1]})",
                        _fmt(c.get("first_differing_token", "none")),
                    ]
                )
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * 2 * len(hosts) + 1.9), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    floor = 1e-4
    colors = {"cuda": SERIES[0], "cpu": SERIES[1]}
    for ax, wf in zip(axes, ("fl2va", "ref2va"), strict=True):
        ax.set_facecolor(SURFACE)
        labels = []
        for h, (host, _) in enumerate(hosts):
            for d, dev in enumerate(("cuda", "cpu")):
                y = h * 2.6 + d
                c = checks[host, wf, dev]
                v = 0.0 if c["bitwise_equal"] else float(c["relative_l2"])
                if v == 0.0:
                    ax.plot([floor], [y], marker="o", ms=6, color=colors[dev])
                    ax.text(floor * 1.4, y, "= 0, bitwise", va="center", fontsize=8, color=INK)
                else:
                    ax.barh(y, v - floor, height=0.8, color=colors[dev], left=floor)  # ends at v
                    ax.text(v * 1.08, y, f"{v:.3g}", va="center", fontsize=8, color=INK)
                labels.append((y, f"{host}, contract on {dev.upper()}"))
        ax.set_xscale("log")
        ax.set_xlim(floor, 3.0)
        ax.set_yticks([y for y, _ in labels])
        ax.set_yticklabels([text for _, text in labels], fontsize=7.5, color=INK2)
        ax.invert_yaxis()
        tokens = checks[hosts[0][0], wf, "cpu"]["shape"][1]
        name = "FL2VA" if wf == "fl2va" else "Ref2VA"
        ax.set_title(f"{name} ({tokens:,} tokens)", loc="left", fontsize=9.5, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color=EQUAL, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.set_xlabel(
            "relative L2 vs the pinned features, all tokens (log)", fontsize=8, color=INK2
        )
    fig.suptitle(args.title, x=0.01, ha="left", fontsize=11, color=INK)
    _save(fig, args.png)
    headers = [
        "Host",
        "Workflow",
        "Contract on",
        "Tokens",
        "Bitwise equal",
        "relative L2, all tokens",
        "max |Δ|",
        "Differing tokens (text / vision)",
        "First differing token",
    ]
    return {
        "schema": SCHEMA,
        "kind": "plot-text-hosts",
        "png": Path(args.png).name,
        "tables": {"text_check_hosts": {"headers": headers, "rows": rows}},
        "markdown": _md(headers, rows),
        "source": source_state(),
    }


def plot_memory(args: argparse.Namespace) -> dict[str, Any]:
    """Peak GPU memory per run from ``nvidia-smi`` traces (``unix_time,MiB`` per line)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = []
    for item in args.trace:
        label, path = _label_path(item)
        used = [
            int(line.split(",")[1])
            for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        traces.append((label, max(used), min(used), len(used)))
    capacity = args.capacity_mib
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(traces) + 1.7))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    rows = []
    for r, (label, peak, low, samples) in enumerate(traces):
        ax.barh(r, peak / 1024, height=0.6, color=SERIES[0])
        share = f", {100 * peak / capacity:.1f} % of the card" if capacity else ""
        ax.text(
            peak / 1024 - 1.0,
            r,
            f"{peak / 1024:.1f} GiB{share}",
            va="center",
            ha="right",
            fontsize=7.5,
            color=SURFACE,
        )  # inside the bar: a bar near the card line leaves no room after it
        rows.append(
            [
                label,
                peak,
                f"{peak / 1024:.1f}",
                f"{100 * peak / capacity:.1f}" if capacity else "n/a",
                low,
                samples,
            ]
        )
    if capacity:
        ax.axvline(capacity / 1024, color=INK2, linewidth=1, linestyle="--")
        ax.text(
            capacity / 1024 + 1.0,
            0,
            f"card:\n{capacity / 1024:.1f} GiB",
            fontsize=7.5,
            color=INK2,
            ha="left",
            va="center",
        )
    ax.set_yticks(range(len(traces)))
    ax.set_yticklabels([label for label, *_ in traces], fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, max([peak for _, peak, *_ in traces] + [capacity]) / 1024 * 1.18)
    ax.set_xlabel("peak memory used on the card (GiB, nvidia-smi samples)", fontsize=8, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=EQUAL, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(args.title, loc="left", fontsize=11, color=INK, pad=14)
    _save(fig, args.png)
    headers = ["Trace", "Peak (MiB)", "Peak (GiB)", "% of card", "Lowest sample (MiB)", "Samples"]
    return {
        "schema": SCHEMA,
        "kind": "plot-memory",
        "png": Path(args.png).name,
        "capacity_mib": capacity,
        "tables": {"memory": {"headers": headers, "rows": rows}},
        "markdown": _md(headers, rows),
        "source": source_state(),
    }


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
    p.add_argument(
        "--text-source",
        choices=sorted(TEXT_SOURCES),
        required=True,
        help="pinned: replay --text; qwen3vl: run qwen3vl_layer50_contract from --text-root",
    )
    p.add_argument("--text", help="pinned text features (pin-text output)")
    p.add_argument("--text-sha256")
    p.add_argument("--text-root", help="dir with text_encoder/ tokenizer/ processor/")
    p.add_argument("--text-device", help="device of the text encoder (default: --device)")
    p.add_argument("--layers", type=int, default=None, help="first K blocks (default: all 50)")
    p.add_argument("--mode", choices=("provider", "strict", "audit"), default="provider")
    p.add_argument("--steps", type=int, default=4, help="num_inference_steps (grid points)")
    p.add_argument("--label", default="run")
    p.add_argument("--device", default="cuda")
    p.add_argument("--offload", action="store_true", help="keep idle models on the CPU")
    p.add_argument(
        "--attention-query-chunk",
        type=int,
        default=0,
        help="strict/audit: run the interim attention in query chunks (bounded workspace)",
    )
    p.add_argument(
        "--stream-blocks",
        action="store_true",
        help="blocks in pinned host memory, each moved to the GPU for its forward only",
    )
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
    p.add_argument("--allow-text-source-change", action="store_true")
    p.add_argument("--report", required=True)

    p = sub.add_parser("text-check")
    p.add_argument("--workflow", choices=("fl2va", "ref2va"), required=True)
    p.add_argument(
        "--text-root", required=True, help="dir with text_encoder/ tokenizer/ processor/"
    )
    p.add_argument("--pinned", required=True, help="pinned features (pin-text output)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--report", required=True)

    p = sub.add_parser("plot")
    p.add_argument("lhs")
    p.add_argument("rhs")
    p.add_argument("--png", required=True)
    p.add_argument("--title", default="Per-step digest equality of two runs")
    p.add_argument("--compare", help="compare report of the same pair (row identity)")
    p.add_argument("--pair-label", default="A vs B")
    p.add_argument("--report", required=True)

    p = sub.add_parser("plot-stages")
    p.add_argument("--pair", action="append", required=True, help="LABEL=compare-report.json")
    p.add_argument("--png", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--report", required=True)

    p = sub.add_parser("plot-text")
    p.add_argument("--dir", required=True, help="dir with {fl2va,ref2va}-{cpu,cuda}.json")
    p.add_argument("--analysis", required=True, help="per-token analysis of the CUDA runs")
    p.add_argument("--png", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--report", required=True)

    p = sub.add_parser("plot-time")
    p.add_argument("--run", action="append", required=True, help="LABEL=run-report.json")
    p.add_argument("--png", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--report", required=True)

    p = sub.add_parser("plot-text-hosts")
    p.add_argument(
        "--host",
        action="append",
        required=True,
        help="LABEL=dir with {fl2va,ref2va}-{cpu,cuda}.json",
    )
    p.add_argument("--png", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--report", required=True)

    p = sub.add_parser("plot-memory")
    p.add_argument("--trace", action="append", required=True, help="LABEL=nvidia-smi trace CSV")
    p.add_argument("--capacity-mib", type=int, default=0, help="card memory, for the % column")
    p.add_argument("--png", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--report", required=True)

    p = sub.add_parser("toy")
    p.add_argument("--report", required=True)

    args = parser.parse_args()
    if getattr(args, "offload", False) and getattr(args, "stream_blocks", False):
        parser.error("--offload and --stream-blocks are alternatives")
    handler = {
        "pin-text": pin_text,
        "run": run,
        "compare": compare,
        "toy": toy,
        "plot": plot,
        "plot-stages": plot_stages,
        "plot-text": plot_text,
        "plot-time": plot_time,
        "plot-text-hosts": plot_text_hosts,
        "plot-memory": plot_memory,
        "text-check": text_check,
    }[args.command]
    report = handler(args)
    text = json.dumps(fc.finite(report), indent=1, sort_keys=True, default=str, allow_nan=False)
    Path(args.report).write_text(text + "\n")
    print(json.dumps({k: report[k] for k in report if k in ("kind", "summary", "first_drift")}))


if __name__ == "__main__":
    main()
