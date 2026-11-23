# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Released-weight evidence for RFC #420 ``qwen3vl_layer50_contract`` (one CUDA GPU).

Loads the released Qwen3-VL-32B ``text_encoder/`` twice: the full 64-layer model (the Diffusers
reference, ``get_qwen3vl_prompt_embeds``) and the 50-layer stack the contract runs. The two are
never resident together: :func:`release_after` checks the full model is gone (no reference left,
at most ``--max-retained-gib`` still allocated) before the 50-layer stack loads, so the peak is
one model plus activations and the run fits a 95 GB card. Then it records:

* pinned files (SHA-256 of tokenizer/processor/text_encoder) and the tokenizer pin;
* the contract feature against Diffusers ``hidden_states[50]`` (bitwise), plus the
  ``hidden_states[49]``/``[51]``/last digests that an off-by-one would produce;
* repeatability of the contract feature over several runs;
* batch composition: HF padded batching (right and left padding), the prior-art path for
  encoding several prompts in one forward, against the per-row feature;
* SDPA backend sensitivity, prefix invariance, vision-tower batch composition;
* latency and weight memory of the Diffusers call versus the contract.

Usage::

    PYTHONPATH=<repo>:<diffusers f53d5520>/src \
        python tools/validation/models/h3_qwen3vl_evidence.py --root <download> --out evidence.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from rl_engine.models.minimax_h3 import qwen3vl_text as q
from rl_engine.models.minimax_h3.trajectory import tensor_digest

PROMPTS = {
    "short_en": "A red fox runs through fresh snow at dawn.",
    "zh": "镜头缓慢推进，一位老人在雨中撑伞走过石桥，河面泛起涟漪。",
    "mid_en": (
        "Close-up of a barista pouring latte art in a sunlit cafe. Warm light, "
        "shallow depth of field, steam rising from the cup, soft jazz in the background, "
        "24 fps, handheld camera."
    ),
    "long_en": " ".join(
        [
            "A drone shot glides over a coastal village at golden hour,",
            "waves breaking on black rocks, gulls circling above fishing boats",
            "while a market fills the harbour square with colour.",
        ]
        * 8
    ),
    "audio_cue": (
        "A street drummer plays a fast rhythm; the crowd claps along. "
        "Sound: drums, claps, city noise."
    ),
    "one_token": "Rain",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(fn: Any, repeats: int, device: torch.device) -> tuple[Any, list[float]]:
    out = fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        _sync(device)
        times.append((time.perf_counter() - start) * 1e3)
    return out, times


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a32, b32 = a.float(), b.float()
    mismatched = (a != b).any(dim=-1)
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs": float((a32 - b32).abs().max()),
        "mean_abs": float((a32 - b32).abs().mean()),
        "rows_mismatched": int(mismatched.sum()),
        "rows": int(mismatched.numel()),
        "elements_mismatched_frac": float((a != b).float().mean()),
    }


def sdpa_backend_cell(
    backend: str, run: Callable[[], torch.Tensor], reference: torch.Tensor
) -> dict[str, Any]:
    """``run()`` with SDPA forced to ``backend``, diffed against ``reference``.

    A backend with no kernel for the shape is recorded as an error cell (the chart's
    "unsupported"); any other failure raises, so a broken call cannot pass as unsupported.
    """

    from torch.nn.attention import SDPBackend, sdpa_kernel

    try:
        with sdpa_kernel([getattr(SDPBackend, backend)]), torch.no_grad():
            out = run()
    except RuntimeError as exc:
        # CUDA: "No available kernel. Aborting execution."; CPU: "No viable backend ...".
        if not str(exc).startswith(("No available kernel", "No viable backend")):
            raise
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return _diff(out, reference)


def _images(count: int) -> list[Any]:
    import numpy as np
    from PIL import Image

    out = []
    for index in range(count):
        y, x = np.mgrid[0:480, 0:832]
        rgb = np.stack(
            [(x * (index + 1)) % 256, (y * 3 + 40 * index) % 256, ((x + y) // 2) % 256], -1
        )
        out.append(Image.fromarray(rgb.astype(np.uint8)))
    return out


def _vision_presentation(
    tokenizer: Any, processor: Any, count: int, prompt: str
) -> tuple[list[int], dict[str, Any]]:
    """FL2VA-style presentation: ``"<Picture i>: "`` + a vision block per keyframe, then prompt."""

    feats = processor.image_processor(images=_images(count), return_tensors="pt")
    ids: list[int] = []
    merge = processor.image_processor.merge_size**2
    for index, grid in enumerate(feats["image_grid_thw"].tolist()):
        ids += tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        ids += q.vision_block("image", grid[0] * grid[1] * grid[2] // merge)
    ids += q.tokenize_prompt(tokenizer, prompt)
    return ids, {"pixel_values": feats["pixel_values"], "image_grid_thw": feats["image_grid_thw"]}


@torch.no_grad()
def _hf_padded_batch(
    model: Any, rows: list[list[int]], side: str, pad_id: int
) -> list[torch.Tensor]:
    """Prior-art batched call: one padded forward, ``hidden_states[50]`` sliced per row."""

    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        span = slice(0, len(row)) if side == "right" else slice(width - len(row), width)
        ids[index, span] = torch.tensor(row)
        mask[index, span] = 1
    device = model.model.language_model.embed_tokens.weight.device
    out = model.model(
        input_ids=ids.to(device),
        attention_mask=mask.to(device),
        mm_token_type_ids=torch.zeros_like(ids).to(device),
        use_cache=False,
        output_hidden_states=True,
    )
    hidden = out.hidden_states[q.TEXT_ENCODER_LAYER]
    result = []
    for index, row in enumerate(rows):
        span = slice(0, len(row)) if side == "right" else slice(width - len(row), width)
        result.append(hidden[index : index + 1, span])
    return result


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - evidence metadata only
        return "unknown"


def _full_model_stages(
    full: Any,
    *,
    rows: dict[str, list[int]],
    cases: dict[str, tuple[list[int], dict[str, Any]]],
    tokenizer: Any,
    processor: Any,
    device: torch.device,
    repeats: int,
) -> dict[str, Any]:
    """Stages 3-6 on the full 64-layer model.

    Returns JSON-able reports only (digests, diffs, timings), so nothing that references the full
    model or its activations outlives this frame; :func:`release_after` checks that.
    """

    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

    class _Proc:
        @staticmethod
        def create_mm_token_type_ids(batch: list[list[int]]) -> list[list[int]]:
            return processor.create_mm_token_type_ids(batch)

    # 3. Full model: Diffusers reference and off-by-one neighbours.
    full_bytes = sum(p.numel() * p.element_size() for p in full.parameters())
    reference: dict[str, Any] = {}
    report: dict[str, Any] = {"reference": reference, "full_model_param_bytes": full_bytes}
    reference_tensors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, (ids, vision) in cases.items():
            fn = lambda ids=ids, vision=vision, full=full: get_qwen3vl_prompt_embeds(  # noqa: E731
                full, _Proc(), ids, vision, text_encoder_layer=50, device=device, dtype=full.dtype
            )
            ref, times = _timed(fn, repeats, device)
            input_ids = torch.tensor([ids], device=device)
            kwargs = {
                k: (v.to(device, full.dtype) if k.startswith("pixel_") else v.to(device))
                for k, v in vision.items()
            }
            hs = full.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=torch.tensor(
                    processor.create_mm_token_type_ids([ids]), device=device
                ),
                use_cache=False,
                output_hidden_states=True,
                **kwargs,
            ).hidden_states
            hook_full = q.Qwen3VLLayer50Encoder(full).encode(ids, vision or None)
            reference_tensors[name] = ref
            reference[name] = {
                "num_tokens": len(ids),
                "diffusers_sha256": tensor_digest(ref),
                "hidden_states_50_sha256": tensor_digest(hs[50]),
                "hidden_states_49_sha256": tensor_digest(hs[49]),
                "hidden_states_51_sha256": tensor_digest(hs[51]),
                "hidden_states_last_sha256": tensor_digest(hs[-1]),
                "num_hidden_states": len(hs),
                "contract_on_full_model_sha256": hook_full.feature_digest,
                "diffusers_ms": times,
            }

    # 4. SDPA backends and prefix invariance on the full model (Diffusers call).
    backends: dict[str, Any] = {}
    for backend in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        entry: dict[str, Any] = {}
        for name in ("short_en", "long_en", "fl2va_1kf"):
            ids, vision = cases[name]
            entry[name] = sdpa_backend_cell(
                backend,
                lambda ids=ids, vision=vision: get_qwen3vl_prompt_embeds(
                    full, _Proc(), ids, vision, 50, device=device, dtype=full.dtype
                ),
                reference_tensors[name],
            )
        backends[backend] = entry
    report["sdpa_backends_vs_default"] = backends

    prefix: dict[str, Any] = {}
    with torch.no_grad():
        for name in ("mid_en", "long_en"):
            ids = rows[name]
            for k in (len(ids) // 4, len(ids) // 2):
                part = get_qwen3vl_prompt_embeds(
                    full, _Proc(), ids[:k], {}, 50, device=device, dtype=full.dtype
                )
                prefix[f"{name}[:{k}]"] = _diff(part, reference_tensors[name][:, :k])
    report["prefix_invariance"] = prefix

    # 5. Prior-art batching: HF padded batch versus per-row.
    text_names = list(rows)
    batching: dict[str, Any] = {}
    pad_id = int(tokenizer.pad_token_id)
    for side in ("right", "left"):
        outs = _hf_padded_batch(full, [rows[n] for n in text_names], side, pad_id)
        batching[f"hf_padded_{side}_all6"] = {
            n: _diff(o, reference_tensors[n]) for n, o in zip(text_names, outs, strict=True)
        }
    pairs = {}
    for other in text_names:
        if other == "short_en":
            continue
        outs = _hf_padded_batch(full, [rows["short_en"], rows[other]], "right", pad_id)
        pairs[f"short_en+{other}"] = _diff(outs[0], reference_tensors["short_en"])
    batching["hf_padded_right_pairs"] = pairs
    report["prior_art_batching"] = batching

    # 6. Vision tower batch composition (all three libraries batch a request's images).
    feats = processor.image_processor(images=_images(2), return_tensors="pt")
    with torch.no_grad():
        both = full.model.get_image_features(
            feats["pixel_values"].to(device, full.dtype), feats["image_grid_thw"].to(device)
        )
        one = processor.image_processor(images=_images(1), return_tensors="pt")
        alone = full.model.get_image_features(
            one["pixel_values"].to(device, full.dtype), one["image_grid_thw"].to(device)
        )
    pooled_both = both.pooler_output if hasattr(both, "pooler_output") else both[0]
    pooled_alone = alone.pooler_output if hasattr(alone, "pooler_output") else alone[0]
    first_both = pooled_both[0] if isinstance(pooled_both, (list, tuple)) else pooled_both
    first_alone = pooled_alone[0] if isinstance(pooled_alone, (list, tuple)) else pooled_alone
    report["vision_tower_batch_composition"] = {
        "image0_alone_vs_with_image1": _diff(first_alone, first_both[: first_alone.shape[0]]),
    }
    return report


def release_after(
    load: Callable[[], Any],
    stages: Callable[[Any], dict[str, Any]],
    *,
    device: torch.device,
    max_retained_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a model, run ``stages`` on it, and prove the model is gone before returning.

    Fails if the model is still referenced after the stages (a closure, default argument or
    module attribute holding it), if the report holds anything but plain JSON data, or if more
    than ``max_retained_bytes`` stay allocated on ``device``.
    """

    model = load()
    alive = weakref.ref(model)
    report = stages(model)
    del model
    gc.collect()
    if alive() is not None:
        holders = sorted({type(r).__name__ for r in gc.get_referrers(alive())})
        raise RuntimeError(f"full model still referenced after stages 3-6 (by {holders})")
    json.dumps(report)  # plain data only: no tensor or module survives in the report
    freed: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        retained = torch.cuda.memory_allocated(device)
        freed = {
            "allocated_after_full_model_free_gib": retained / 2**30,
            "peak_allocated_full_model_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        if retained > max_retained_bytes:
            raise RuntimeError(
                f"{retained / 2**30:.2f} GiB still allocated on {device} after freeing the full "
                f"model (limit {max_retained_bytes / 2**30:.2f} GiB)"
            )
    return report, freed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path, help="download of MiniMaxAI/MiniMax-H3")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--max-retained-gib",
        type=float,
        default=1.0,
        help="fail if more than this stays allocated after the full model is freed",
    )
    parser.add_argument(
        "--skip-hash", action="store_true", help="do not re-hash the 67 GB of shards"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1 (the latency medians need a timed run)")

    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    # "contract" is filled with the 50-layer results in stage 7; the manifest keeps its own key.
    evidence: dict[str, Any] = {"manifest": q.contract_manifest()}
    props = torch.cuda.get_device_properties(0)
    import transformers

    evidence["environment"] = {
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty_files": len(
            [x for x in _git("status", "--porcelain", "--untracked-files=no").splitlines() if x]
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "python": platform.python_version(),
        "runtime_readback": q.runtime_readback(),
    }

    # 1. Pinned files.
    start = time.perf_counter()
    skip = (".safetensors",) if args.skip_hash else ()
    checked = q.verify_release_files(
        args.root, subfolders=("tokenizer", "processor", "text_encoder"), skip_suffixes=skip
    )
    evidence["files"] = {
        "verified": len(checked),
        "sha256": checked,
        "shards_hashed": not args.skip_hash,
        "seconds": time.perf_counter() - start,
    }

    # 2. Tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(str(args.root / "tokenizer"))
    processor = AutoProcessor.from_pretrained(str(args.root / "processor"))
    q.verify_release_tokenizer(tokenizer)
    q.verify_release_tokenizer(processor.tokenizer)
    q.verify_release_processor(processor)
    rows = {name: q.tokenize_prompt(tokenizer, text) for name, text in PROMPTS.items()}
    repeat_ok = all(q.tokenize_prompt(processor.tokenizer, PROMPTS[n]) == rows[n] for n in rows)
    rejected = {}
    for token in sorted(q.special_token_ids()):
        try:
            q.tokenize_prompt(tokenizer, f"a {token} b")
            rejected[token] = False
        except q.PresentationError:
            rejected[token] = True
    evidence["tokenizer"] = {
        "prompts": {
            n: {"num_tokens": len(r), "token_ids_sha256": q.token_ids_digest(r)}
            for n, r in rows.items()
        },
        "tokenizer_and_processor_agree": repeat_ok,
        "special_token_prompts_rejected": rejected,
    }
    vision_cases = {
        "fl2va_1kf": _vision_presentation(tokenizer, processor, 1, PROMPTS["short_en"]),
        "fl2va_2kf": _vision_presentation(tokenizer, processor, 2, PROMPTS["mid_en"]),
    }
    for ids, vision in vision_cases.values():
        q.check_presentation(ids, vision)
    cases: dict[str, tuple[list[int], dict[str, Any]]] = {n: (r, {}) for n, r in rows.items()}
    cases.update(vision_cases)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    def load_full() -> Any:
        full: Any = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.root / "text_encoder"), dtype=torch.bfloat16
        )
        return full.to(device).eval()

    full_report, freed = release_after(
        load_full,
        lambda full: _full_model_stages(
            full,
            rows=rows,
            cases=cases,
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            repeats=args.repeats,
        ),
        device=device,
        max_retained_bytes=int(args.max_retained_gib * 2**30),
    )
    reference = full_report["reference"]
    full_bytes = full_report["full_model_param_bytes"]
    batching = full_report["prior_art_batching"]
    for key in (
        "sdpa_backends_vs_default",
        "prefix_invariance",
        "prior_art_batching",
        "vision_tower_batch_composition",
    ):
        evidence[key] = full_report[key]

    # 7. Contract on the 50-layer stack.
    model, report = q.load_release_text_encoder(args.root / "text_encoder", device=device)
    trunc_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    encoder = q.Qwen3VLLayer50Encoder(model)
    contract: dict[str, Any] = {"load": report}
    for name, (ids, vision) in cases.items():
        out, times = _timed(
            lambda ids=ids, vision=vision: encoder.encode(ids, vision or None),
            args.repeats,
            device,
        )
        digests = {encoder.encode(ids, vision or None).feature_digest for _ in range(3)}
        contract[name] = {
            "feature_sha256": out.feature_digest,
            "token_ids_sha256": out.token_ids_digest,
            "equals_diffusers": out.feature_digest == reference[name]["diffusers_sha256"],
            "repeat_digests": len(digests | {out.feature_digest}),
            "contract_ms": times,
        }
    names = list(cases)
    compositions = []
    for shift in range(1, len(names)):
        order = names[shift:] + names[:shift]
        got = encoder.encode_many([(cases[n][0], cases[n][1] or None) for n in order])
        compositions.append(
            all(
                o.feature_digest == contract[n]["feature_sha256"]
                for o, n in zip(got, order, strict=True)
            )
        )
    contract["encode_many_rotations_bitwise"] = compositions
    evidence["contract"] = contract
    evidence["reference"] = reference
    evidence["memory"] = {
        "full_model_param_gib": full_bytes / 2**30,
        "layer50_param_gib": trunc_bytes / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        **freed,
    }
    latency = {}
    for name in cases:
        latency[name] = {
            "num_tokens": reference[name]["num_tokens"],
            "diffusers_median_ms": statistics.median(reference[name]["diffusers_ms"]),
            "contract_median_ms": statistics.median(contract[name]["contract_ms"]),
        }
    evidence["latency"] = latency
    evidence["verdict"] = {
        "contract_equals_diffusers_all": all(contract[n]["equals_diffusers"] for n in cases),
        "contract_on_full_equals_diffusers_all": all(
            reference[n]["contract_on_full_model_sha256"] == reference[n]["diffusers_sha256"]
            for n in cases
        ),
        "hidden_states_50_is_diffusers_all": all(
            reference[n]["hidden_states_50_sha256"] == reference[n]["diffusers_sha256"]
            for n in cases
        ),
        "off_by_one_distinguishable_all": all(
            len({reference[n][f"hidden_states_{k}_sha256"] for k in ("49", "50", "51")}) == 3
            for n in cases
        ),
        "contract_repeatable_all": all(contract[n]["repeat_digests"] == 1 for n in cases),
        "contract_batch_composition_invariant": all(compositions),
        "hf_padded_batch_bitwise_all": all(
            d["bitwise_equal"]
            for key in ("hf_padded_right_all6", "hf_padded_left_all6")
            for d in batching[key].values()
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(evidence["verdict"], indent=1))


if __name__ == "__main__":
    main()
