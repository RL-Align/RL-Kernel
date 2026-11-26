# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Seal and check the RFC #738 ``image_text_presentation_contract`` layer-50 fixtures.

Three subcommands:

``seal-cpu``
    Build every case of ``text_presentation_fixtures.json`` with the released tokenizer and
    processor and write the token-id, tag, grid and pixel digests into the file (CPU only).
``evidence``
    One CUDA GPU. Loads the released Qwen3-VL ``text_encoder/`` (64 layers), runs the pinned
    Diffusers step and the contract on every case and records both feature digests, then
    reloads the 50-layer stack the contract uses in production and records it again. Also
    runs the I9/I19 probes on the released weights. Writes one JSON file.
``seal-gpu``
    Copy the feature digests of an ``evidence`` JSON into the fixtures file, keyed by
    :func:`text_presentation.device_key`. Refuses unless every case matched Diffusers.

Usage::

    python tools/validation/models/h3_image_text_presentation.py seal-cpu --root <H3-Image download>
    python tools/validation/models/h3_image_text_presentation.py evidence \
        --root <H3-Image download> \
        --text-encoder-root <download holding text_encoder/> --out evidence.json
    python tools/validation/models/h3_image_text_presentation.py seal-gpu --evidence evidence.json
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from rl_engine.models.minimax_h3 import qwen3vl_text as q
from rl_engine.models.minimax_h3.trajectory import tensor_digest
from rl_engine.models.minimax_h3_image import text_presentation as tp


def _load_presentation_parts(root: Path) -> tuple[Any, Any]:
    from transformers import AutoProcessor, AutoTokenizer

    tp.verify_presentation_files(root)
    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    processor = AutoProcessor.from_pretrained(str(root / "processor"))
    q.verify_release_tokenizer(tokenizer)
    q.verify_release_processor(processor)
    return tokenizer, processor


def _write_fixtures(data: dict[str, Any]) -> None:
    tp.FIXTURES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tp.sealed_fixtures.cache_clear()


def cpu_expectation(case: dict[str, Any], tokenizer: Any, processor: Any) -> dict[str, Any]:
    presentation, vision = tp.build_fixture_case(case, tokenizer, processor)
    tp.check_tags(presentation)
    tp.check_vision_inputs(presentation, vision)
    return {**presentation.summary(), "vision": tp.vision_digest(vision)}


def seal_cpu(args: argparse.Namespace) -> None:
    tokenizer, processor = _load_presentation_parts(Path(args.root))
    data = json.loads(tp.FIXTURES_PATH.read_text())
    for case in data["cases"]:
        case["expected"] = cpu_expectation(case, tokenizer, processor)
        print(case["name"], case["expected"]["num_rows"], case["expected"]["token_ids_sha256"])
    _write_fixtures(data)


def _digest(t: torch.Tensor) -> str:
    return tensor_digest(t.detach().cpu())


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"same_shape": False, "shapes": [list(a.shape), list(b.shape)]}
    d = (a.float() - b.float()).abs()
    return {
        "same_shape": True,
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs": float(d.max()),
        "rows_differing": int((d.amax(dim=-1) > 0).sum()),
    }


def _timed(fn: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1e3


def evidence(args: argparse.Namespace) -> None:
    import diffusers
    import transformers

    root, te_root = Path(args.root), Path(args.text_encoder_root)
    device = torch.device("cuda")
    tokenizer, processor = _load_presentation_parts(root)
    record: dict[str, Any] = {
        "manifest": tp.contract_manifest(),
        "inherited_text_encoder_sha256": tp.image_fp.check_inherited_text_encoder(),
        "presentation_files": tp.verify_presentation_files(root),
        "diffusers_files": tp.verify_diffusers_reference(Path(diffusers.__file__).parent),
        "env": {
            "device_key": tp.device_key(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
            "python": platform.python_version(),
            "runtime": q.runtime_readback(),
        },
    }
    if args.verify_weights:
        start = time.perf_counter()
        checked = q.verify_release_files(te_root, subfolders=("text_encoder",))
        record["text_encoder_files"] = {
            "count": len(checked),
            "seconds": round(time.perf_counter() - start, 1),
            "pin": f"{q.CONTRACT_ID} ({tp.image_fp.TEXT_ENCODER_REPO})",
        }
    fixtures = tp.sealed_fixtures()
    cases = {case["name"]: case for case in fixtures["cases"]}
    built = {}
    for name, case in cases.items():
        presentation, vision = tp.build_fixture_case(case, tokenizer, processor)
        cpu = {**presentation.summary(), "vision": tp.vision_digest(vision)}
        built[name] = (presentation, vision, cpu == case.get("expected"))

    # Stage 1: the full 64-layer model. Diffusers and the contract run on the same weights.
    full, report = q.load_release_text_encoder(
        te_root / "text_encoder", device=device, num_layers=q.RELEASE_NUM_LAYERS
    )
    record["full_model_load"] = report
    encoder = q.Qwen3VLLayer50Encoder(full)
    results: dict[str, Any] = {}
    references: dict[str, torch.Tensor] = {}
    for name, case in cases.items():
        presentation, vision, cpu_ok = built[name]
        images = None
        if case["mode"] == tp.IMAGE_CONDITIONED:
            images = [tp.fixture_image(r) for r in case["references"]]
        # functools.partial binds the model now; the model is deleted before stage 2.
        ref, ref_ms = _timed(
            functools.partial(
                tp.run_diffusers_reference,
                full,
                tokenizer,
                processor,
                tp.fixture_prompt(case),
                images,
                device=device,
            )
        )
        ours, ours_ms = _timed(functools.partial(tp.encode, encoder, presentation, vision))
        given = vision or {}
        same_vision = set(ref.vision_inputs) == set(given) and all(
            torch.equal(ref.vision_inputs[k].cpu(), v.cpu()) for k, v in given.items()
        )
        references[name] = ref.prompt_embeds.detach().cpu()
        results[name] = {
            "cpu_digests_match_sealed": cpu_ok,
            "token_ids_equal_diffusers": ref.token_ids == list(presentation.token_ids),
            "tags_equal_diffusers": torch.equal(
                ref.text_token_tags.cpu(), presentation.tags_tensor()
            ),
            "vision_inputs_equal_diffusers": same_vision,
            "num_rows": presentation.num_rows,
            "diffusers": {
                "shape": list(ref.prompt_embeds.shape),
                "dtype": str(ref.prompt_embeds.dtype).removeprefix("torch."),
                "sha256": _digest(ref.prompt_embeds),
                "ms": round(ref_ms, 1),
            },
            "contract_full_model": {
                "sha256": ours.output.feature_digest,
                "ms": round(ours_ms, 1),
                **_diff(ours.prompt_embeds.cpu(), references[name]),
            },
        }
        print(name, results[name]["contract_full_model"], flush=True)

    # Probes on the released weights (RFC #738 I9 / I19): each changes one factor.
    probes: dict[str, Any] = {}
    p_edit, v_edit, _ = built["edit_swapped"]
    probes["I9_swap_reference_order"] = {
        "token_ids_differ": built["edit"][0].token_ids != p_edit.token_ids,
        **_diff(references["edit"], references["edit_swapped"]),
    }
    t2i_prompt = tp.fixture_prompt(cases["t2i"])
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": t2i_prompt}], tokenize=True, add_generation_prompt=False
    )
    chat = chat["input_ids"] if hasattr(chat, "keys") else chat
    templated = [int(i) for i in chat]
    feat = encoder.encode(templated).feature.cpu()
    probes["I19_chat_template"] = {"rows": len(templated), **_diff(feat, references["t2i"])}
    with_specials = [int(i) for i in tokenizer(t2i_prompt, add_special_tokens=True)["input_ids"]]
    probes["I19_add_special_tokens"] = {
        "ids_equal_release": with_specials == list(built["t2i"][0].token_ids),
        "note": "the released tokenizer adds no BOS/EOS, so this policy change is a no-op",
    }
    long_ids = q.tokenize_prompt(tokenizer, tp.fixture_prompt(cases["t2i_truncated"]))
    for label, cut in (("I19_truncate_4095", 4095), ("I19_truncate_4097", 4097)):
        probe = encoder.encode(long_ids[:cut]).feature.cpu()
        shared = min(cut, tp.MAX_SEQUENCE_LENGTH)
        probes[label] = {
            "rows": cut,
            **_diff(probe, references["t2i_truncated"]),
            "shared_prefix": _diff(probe[:, :shared], references["t2i_truncated"][:, :shared]),
        }
    p, v, _ = built["edit"]
    prompt_first = list(p.token_ids[p.segments[-1].start :]) + list(
        p.token_ids[: p.segments[-1].start]
    )
    feat = encoder.encode(prompt_first, v).feature.cpu()
    probes["I19_prompt_before_references"] = {
        "rows": len(prompt_first),
        **_diff(feat, references["edit"]),
    }
    record["probes"] = probes
    del encoder, full
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2: the 50-layer stack the contract loads in production.
    stack, report = q.load_release_text_encoder(te_root / "text_encoder", device=device)
    record["stack_load"] = report
    torch.cuda.reset_peak_memory_stats(device)
    encoder = q.Qwen3VLLayer50Encoder(stack)
    for name in cases:
        presentation, vision, _ = built[name]
        ours, ms = _timed(functools.partial(tp.encode, encoder, presentation, vision))
        repeat = tp.encode(encoder, presentation, vision)
        results[name]["contract_50_layer_stack"] = {
            "sha256": ours.output.feature_digest,
            "ms": round(ms, 1),
            "repeat_bitwise_equal": repeat.output.feature_digest == ours.output.feature_digest,
            **_diff(ours.prompt_embeds.cpu(), references[name]),
        }
        print(name, results[name]["contract_50_layer_stack"], flush=True)
    record["stack_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    record["cases"] = results
    record["all_equal_diffusers"] = all(
        r["token_ids_equal_diffusers"]
        and r["tags_equal_diffusers"]
        and r["vision_inputs_equal_diffusers"]
        and r["contract_full_model"].get("bitwise_equal") is True
        and r["contract_50_layer_stack"].get("bitwise_equal") is True
        for r in results.values()
    )
    record["all_cpu_digests_match_sealed"] = all(
        r["cpu_digests_match_sealed"] for r in results.values()
    )
    Path(args.out).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print("all_equal_diffusers", record["all_equal_diffusers"])
    print("all_cpu_digests_match_sealed", record["all_cpu_digests_match_sealed"])


def seal_gpu(args: argparse.Namespace) -> None:
    record = json.loads(Path(args.evidence).read_text())
    if not (record["all_equal_diffusers"] and record["all_cpu_digests_match_sealed"]):
        raise SystemExit("evidence does not match Diffusers or the sealed CPU digests; not sealing")
    key = record["env"]["device_key"]
    data = json.loads(tp.FIXTURES_PATH.read_text())
    for case in data["cases"]:
        result = record["cases"][case["name"]]
        case.setdefault("features", {})[key] = {
            "shape": result["diffusers"]["shape"],
            "dtype": result["diffusers"]["dtype"],
            "sha256": result["diffusers"]["sha256"],
        }
    _write_fixtures(data)
    print("sealed", key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seal-cpu")
    p.add_argument("--root", required=True, help="H3-Image download with tokenizer/, processor/")
    p.set_defaults(func=seal_cpu)
    p = sub.add_parser("evidence")
    p.add_argument("--root", required=True, help="H3-Image download with tokenizer/, processor/")
    p.add_argument(
        "--text-encoder-root",
        required=True,
        help="download holding the inherited text_encoder/ (H3-Image or MiniMax-H3)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--verify-weights", action="store_true", help="SHA-256 every weight shard")
    p.set_defaults(func=evidence)
    p = sub.add_parser("seal-gpu")
    p.add_argument("--evidence", required=True)
    p.set_defaults(func=seal_gpu)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
