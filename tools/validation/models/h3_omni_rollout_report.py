# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""G3 of ``final_distributed_h3_run``: the vLLM-Omni TP rollout runs the actor's H3, bit for bit.

Three stages, each writing ``<out>/<stage>.json``:

``engine`` (``--tp`` GPUs, vLLM-Omni installed)
    Starts vLLM-Omni's H3 pipeline on the released checkpoint at TP ``--tp`` with
    ``H3OmniWorkerExtension``, binds the DiT on every worker (``rlk_h3_install``), optionally puts
    the seeded LoRA of the parallel harness on it, and runs request A alone, request B alone, then
    A and B together. Every DiT call is recorded by every rank as digests of its Diffusers inputs
    and outputs; rank 0 also saves the first ``--record-calls`` calls in full (``calls/``).
    Checks: every rank returns the same bytes for every call; a request run with another
    (``num_requests > 1``) gives the same outputs as the same inputs run alone. With a LoRA, the
    LoRA reaches the workers as the actor's per-rank shards (``rlk_h3_load_lora``): a sync with one
    changed element is refused on every rank, the true one installs the sender's digests.

``canonical`` (1 GPU)
    Loads the whole model on one GPU (``Grid(1, 1)``, the canonical run of #701 / #732 / #733)
    with the same LoRA and replays every saved call: the outputs must equal the rollout's.

``native`` (``--tp`` GPUs, reported only)
    vLLM-Omni's own DiT, unbound, on request A: its outputs against a B = 0 engine run's saved
    calls (``--compare-calls``) on the same inputs.

``actor`` (CPU, verl-omni installed)
    Feeds each saved call's vLLM-Omni kwargs through verl-omni's rollout storage
    (``MiniMaxH3PipelineWithLogProb._layout_outputs``) and actor input builder
    (``MiniMaxH3FlowGRPO.prepare_model_inputs``): its model inputs must equal the ones the binding
    passed to the model.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

EXTENSION = "rl_engine.integrations.engines.rollout.vllm.h3_omni_worker.H3OmniWorkerExtension"
PROMPTS = {
    "A": "A paper boat crosses a rain-filled street while distant traffic and water sound in sync.",
    "B": "At night, three cats march into a bedroom playing tiny brass instruments, then file out.",
}
MODEL_KEYS = (
    "hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "timestep",
    "timestep_indices",
    "token_tags",
    "position_ids",
    "video_indices",
    "audio_indices",
    "text_indices",
)


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in data.items() if not isinstance(v, (list, dict))}))


# --- engine --------------------------------------------------------------------------------------


def _make_engine(args):
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    return AsyncOmni(
        model=str(args.engine_model or args.checkpoint),
        trust_remote_code=True,
        num_gpus=args.tp,
        tensor_parallel_size=args.tp,
        data_parallel_size=1,
        text_encoder_tp_size=args.tp,
        vae_patch_parallel_size=1,
        enforce_eager=True,
        step_execution=True,
        max_num_seqs=args.max_num_seqs,  # step mode runs this many requests in one DiT call
        worker_extension_cls=EXTENSION,
        stage_init_timeout=args.init_timeout,
        init_timeout=args.init_timeout,
    )


async def _run_request(engine, args, rid: str, key: str, seed: int):
    """One T2VA request of ``--steps`` steps at the stage's size; returns its last output."""

    import copy

    p = copy.deepcopy(engine.default_sampling_params_list)
    d = p[0]
    d.width, d.height, d.fps = args.width, args.height, 24
    d.num_inference_steps, d.seed = args.steps, seed
    d.extra_args = {
        "task": "t2va",
        "duration": args.duration,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    last = None
    async for out in engine.generate(prompt=PROMPTS[key], request_id=rid, sampling_params_list=p):
        if out.finished:
            last = out
    if last is None:
        raise RuntimeError(f"{rid} finished without an output")
    return last


def batch_twins(calls: list[dict[str, Any]]) -> tuple[int, int]:
    """``(batched, with_twin)``: request calls made in a batch, and those of them whose input
    digest also ran alone (the batch-invariance pairs). Every batched call needs a twin."""

    alone = {c["inputs"] for c in calls if c["num_requests"] == 1}
    batched = [c for c in calls if c["num_requests"] > 1]
    return len(batched), sum(c["inputs"] in alone for c in batched)


def _fresh_dir(path: Path) -> Path:
    """An empty capture directory for this run (captures of an earlier run are removed)."""

    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("call*.pt"):
        old.unlink()
    return path


STAGE_REPORTS = ("engine.json", "canonical.json", "actor.json")


def _invalidate_reports(out: Path) -> None:
    """An engine run starts by removing the earlier run's reports: a run that fails before it
    writes ``engine.json`` leaves no manifest for the replays to read."""

    for name in STAGE_REPORTS:
        (out / name).unlink(missing_ok=True)


def check_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse, before any engine starts, what a stage could only fail on later."""

    if args.lora_rank < 0 or 0 < args.lora_rank < 4:  # the control changes attn.to_v.A[3, 5]
        ap.error(f"--lora-rank must be 0 or at least 4, got {args.lora_rank}")
    if args.stage == "native" and (args.compare_calls is None or not args.compare_calls.is_dir()):
        ap.error("--stage native needs --compare-calls DIR, the calls/ of a B = 0 engine run")


def _same_tensor_inputs(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """The same DiT inputs: the same tensor-valued kwargs (latents, positions, timesteps and the
    prompt conditioning alike), each with equal bytes; a kwarg only one side holds is a difference.
    """

    keys_a = {k for k, v in a.items() if isinstance(v, torch.Tensor)}
    keys_b = {k for k, v in b.items() if isinstance(v, torch.Tensor)}
    return keys_a == keys_b and all(_same_bytes(a[k], b[k]) for k in keys_a)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_capture(path: Path) -> tuple[dict[str, Any], str]:
    """A capture and the sha256 of the bytes it was loaded from (one read: no file swap between
    the two)."""

    data = path.read_bytes()
    return torch.load(io.BytesIO(data), weights_only=True), hashlib.sha256(data).hexdigest()


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise: dtype, shape and bytes (``torch.equal`` compares values, so ``-0.0 == 0.0``)."""

    from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om

    return om.tensor_digest(a) == om.tensor_digest(b)


def _listed_captures(args, engine: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    """The captures a passing engine run listed in ``engine.json``, loaded, and only those.

    Each file's sha256 must equal the manifest's, so a replay never pairs a capture with a report
    of another run, nor reads a capture whose inputs, kwargs or audio were replaced (the video
    digest alone would not see those). The video digest is checked too.
    """

    from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om

    if engine.get("pass") is not True:
        raise RuntimeError("engine.json is not a passing engine run: rerun the engine stage")
    listed = engine.get("captures") or []
    if not listed:
        raise RuntimeError("engine.json lists no captures of this run: rerun the engine stage")
    paths = [args.out / "calls" / f"{c['call']}.pt" for c in listed]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"captures listed in engine.json are missing: {missing}")
    loaded = []
    for c, path in zip(listed, paths):
        rec, sha256 = _load_capture(path)
        if c.get("sha256") is None or sha256 != c["sha256"]:
            raise RuntimeError(f"{path.name} does not match engine.json: rerun the engine stage")
        if om.tensor_digest(rec["video"]) != c.get("video_digest"):
            raise RuntimeError(f"{path.name} does not match engine.json: rerun the engine stage")
        loaded.append((path, rec))
    return loaded


async def _engine(args) -> dict[str, Any]:
    from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om

    _invalidate_reports(args.out)
    calls_dir = _fresh_dir(args.out / "calls")
    engine = _make_engine(args)
    report: dict[str, Any] = {
        "tp": args.tp,
        "checkpoint": "MiniMax-H3 (released)",
        "steps": args.steps,
        "max_num_seqs": args.max_num_seqs,
    }
    try:
        bound = await _rpc(
            engine,
            "rlk_h3_install",
            kwargs=dict(
                transformer_dir=str(args.checkpoint / "transformer"),
                strict_boundaries=args.strict_boundaries,
                release_weights=args.release_weights,
                record_dir=str(calls_dir),
                record_calls=tuple(range(args.record_calls)),
                record_digests=True,  # the report compares every call's digests across ranks
            ),
        )
        report["install"] = bound
        if len(bound) != args.tp or not all(
            isinstance(r, dict) and r.get("contract") for r in bound
        ):
            raise RuntimeError(f"rlk_h3_install did not succeed on every rank: {bound}")
        if args.lora_rank:
            report["lora"] = await _lora_sync(engine, args)

        def one(rid: str, key: str, seed: int):
            return _run_request(engine, args, rid, key, seed)

        timings = {}
        for rid, key, seed in (("A-alone", "A", 11), ("B-alone", "B", 22)):
            t0 = time.perf_counter()
            await one(rid, key, seed)
            timings[rid] = time.perf_counter() - t0
        t0 = time.perf_counter()
        await asyncio.gather(one("A-batched", "A", 11), one("B-batched", "B", 22))
        timings["A+B together"] = time.perf_counter() - t0
        report["seconds"] = timings
        ranks = await _rpc(engine, "rlk_h3_report")
        report["ranks"] = ranks
    finally:
        engine.shutdown()
    calls = [r["digests"] for r in ranks]
    report["calls_per_rank"] = [len(c) for c in calls]
    report["ranks_agree"] = all(c == calls[0] for c in calls)
    by_input: dict[str, set] = {}
    for c in calls[0]:
        by_input.setdefault(c["inputs"], set()).add((c["video"], c["audio"]))
    batched, pairs = batch_twins(calls[0])
    report["request_calls"] = len(calls[0])
    report["request_calls_in_batches"] = batched
    report["distinct_inputs"] = len(by_input)
    report["same_inputs_same_outputs"] = all(len(v) == 1 for v in by_input.values())
    report["batched_calls_with_an_alone_twin"] = pairs
    report["batched_calls_without_an_alone_twin"] = batched - pairs
    # this run's captures (rank 0's), each one of the outputs every rank reported
    run_videos = {c["video"] for c in calls[0]}
    captures = []
    for path in sorted(calls_dir.glob("call*.pt"), key=lambda p: int(p.stem[4:])):
        rec, sha256 = _load_capture(path)
        digest = om.tensor_digest(rec["video"])
        captures.append(
            {
                "call": path.stem,
                "video_digest": digest,
                "sha256": sha256,  # the whole capture (inputs, kwargs, both outputs), as loaded
                "in_run": digest in run_videos,
            }
        )
    report["captures"] = captures
    report["record_errors"] = [e for r in ranks for e in r.get("record_errors", [])]
    report["pass"] = bool(
        report["ranks_agree"]
        and report["same_inputs_same_outputs"]
        and batched > 0
        and pairs == batched  # every batched call has its alone twin
        and len(calls) == args.tp
        and len(captures) == args.record_calls
        and all(c["in_run"] for c in captures)
        and not report["record_errors"]
        and (not args.lora_rank or report["lora"]["pass"])
    )
    return report


def _save_actor_shards(args, seed: int, out: Path, tamper_rank: int | None = None) -> str:
    """What the actor's TP rank ``r`` sends rollout rank ``r``: its shards of the seeded LoRA and
    their digests, one file per rank. ``tamper_rank`` changes one element of that rank's shards
    after the digests are taken (the negative control)."""

    from rl_engine.integrations.engines.rollout.vllm.h3_omni_model import tensor_digest
    from rl_engine.validation.models import h3_lora

    out.mkdir(parents=True, exist_ok=True)
    full = [
        h3_lora.make_block_lora(b, rank=args.lora_rank, seed=seed, b_zero=args.lora_b_zero)
        for b in range(50)
    ]
    for r in range(args.tp):
        shards = [  # own storage per rank: shard_lora may return the same A or a view
            {
                t: {p: v.clone() for p, v in parts.items()}
                for t, parts in h3_lora.shard_block_lora(block, args.tp, r).items()
            }
            for block in full
        ]
        digests = {
            f"{b}.{t}.{p}": tensor_digest(v)
            for b, parts_by_target in enumerate(shards)
            for t, parts in parts_by_target.items()
            for p, v in parts.items()
        }
        if r == tamper_rank:
            a = shards[17]["attn.to_v"]["A"]
            a[3, 5] = a[3, 5] + 1  # one element: BF16 + 1 always changes it
        scale = args.lora_alpha / args.lora_rank  # the worker refuses a file of another scale
        torch.save({"shards": shards, "digests": digests, "scale": scale}, out / f"rank{r}.pt")
    return str(out / "rank{rank}.pt")


async def _lora_sync(engine, args) -> dict[str, Any]:
    """Acceptance 4 on the engine: a decoy LoRA (seed + 1) is attached; a sync whose rank-2 (or
    last) shards have one changed element must be refused with every rank keeping the decoy; the
    true sync (seed) must install on every rank with the sender's digests. The requests then run
    on the synced LoRA, and the canonical stage attaches the seed's LoRA directly."""

    import shutil

    scale = args.lora_alpha / args.lora_rank
    decoy = await _rpc(
        engine,
        "rlk_h3_attach_seeded_lora",
        kwargs=dict(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            seed=args.lora_seed + 1,
            b_zero=args.lora_b_zero,
        ),
    )
    tamper_rank = min(2, args.tp - 1)
    try:  # the shard files are removed whatever happens
        bad = _save_actor_shards(args, args.lora_seed, args.out / "lora-bad", tamper_rank)
        try:
            await _rpc(engine, "rlk_h3_load_lora", args=(bad,), kwargs=dict(scale=scale))
            refused = None
        except RpcError as e:  # expected
            refused = str(e)[:600]
        after_bad = await _rpc(engine, "rlk_h3_lora_digests")
        good = _save_actor_shards(args, args.lora_seed, args.out / "lora-good")
        installed = await _rpc(engine, "rlk_h3_load_lora", args=(good,), kwargs=dict(scale=scale))
        sent = [
            torch.load(good.format(rank=r), weights_only=True)["digests"] for r in range(args.tp)
        ]
    finally:
        for d in ("lora-bad", "lora-good"):
            shutil.rmtree(args.out / d, ignore_errors=True)
    out = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "seed": args.lora_seed,
        "b_zero": args.lora_b_zero,
        "shards_per_rank": [len(d) for d in installed],
        "tampered": {"rank": tamper_rank, "block": 17, "target": "attn.to_v.A", "at": [3, 5]},
        "tampered_sync_refused": refused,
        # a readback from every rank, each non-empty: two empty replies would also be equal
        "every_rank_kept_the_decoy": len(decoy) == len(after_bad) == args.tp
        and all(decoy)
        and after_bad == decoy,
        "every_rank_has_the_senders_digests": installed == sent,
    }
    out["pass"] = bool(
        refused
        and "digest mismatch" in refused
        and out["every_rank_kept_the_decoy"]
        and out["every_rank_has_the_senders_digests"]
        and len(installed) == args.tp
    )
    return out


class RpcError(RuntimeError):
    """A worker method failed. AsyncOmni 6039984e returns the failure as the stage's result
    instead of raising: ``{"supported": False, "error": "<message>"}`` (measured), or
    ``{"error": True, "reason": "<message>"}`` from the stage client."""


def _rpc_error(r: Any) -> str | None:
    if not isinstance(r, dict) or "error" not in r:
        return None
    if r["error"] is True:
        return str(r.get("reason"))
    if r.get("supported") is False:
        return str(r["error"])
    return None


async def _rpc(engine, method: str, **kw: Any) -> list:
    """``engine.collective_rpc`` that raises on a failed stage and returns the ranks' results."""

    results = _flat(await engine.collective_rpc(method, **kw))
    errors = [e for e in map(_rpc_error, results) if e is not None]
    if errors:
        raise RpcError(f"{method}: {errors}")
    return results


def _flat(results: Any) -> list:
    """collective_rpc returns per stage, per rank; keep the DiT stage's rank results."""

    out = []
    for r in results if isinstance(results, list) else [results]:
        if isinstance(r, list):
            out.extend(_flat(r))
        elif isinstance(r, dict) and r.get("todo"):
            continue
        else:
            out.append(r)
    return out


# --- native (acceptance 6: reported, not a gate) ---------------------------------------------


async def _native(args) -> dict[str, Any]:
    """vLLM-Omni's own DiT (the P/P path) on request A's first calls, against this binding's
    outputs for the same inputs from ``--compare-calls`` (a B = 0 engine run: B = 0 gives the base
    projections exactly). Measured, not a gate."""

    from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om

    engine = _make_engine(args)
    try:
        await _rpc(
            engine,
            "rlk_h3_record_native",
            kwargs=dict(
                record_dir=str(args.out / "native-calls"),
                record_calls=tuple(range(args.record_calls)),
            ),
        )
        await _run_request(engine, args, "A-alone", "A", 11)
    finally:
        engine.shutdown()
    rows = []
    for path in sorted((args.out / "native-calls").glob("call*.pt"), key=lambda p: int(p.stem[4:])):
        nat = torch.load(path, weights_only=True)
        ours = torch.load(args.compare_calls / path.name, weights_only=True)
        same_inputs = _same_tensor_inputs(nat["omni_kwargs"], ours["omni_kwargs"])
        v, a = om.diffusers_to_omni(ours["video"], ours["audio"], ours["omni_kwargs"])
        row: dict[str, Any] = {"call": path.stem, "same_inputs": same_inputs}
        for name, mine, theirs in (("video", v, nat["video"]), ("audio", a, nat["audio"])):
            d = (mine.double() - theirs.double()).abs()
            row[name] = {
                "bitwise": _same_bytes(mine, theirs),
                "elements_differing": int((mine != theirs).sum()),
                "elements": int(mine.numel()),
                "max_abs": float(d.max()),
                "rel_l2": float(d.norm() / theirs.double().norm()),
            }
        rows.append(row)
        print(json.dumps(row))
    return {
        "tp": args.tp,
        "engine": "vLLM-Omni native DiT (enforce_eager, step mode, as the engine stage)",
        "compared_with": "this binding, LoRA B = 0",
        "calls": rows,
    }


# --- canonical -----------------------------------------------------------------------------------


def _canonical(args) -> dict[str, Any]:
    from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om
    from rl_engine.validation.models import h3_lora
    from rl_engine.validation.models import h3_parallel_model as pm
    from rl_engine.validation.models import h3_sharded_load as sl
    from rl_engine.validation.models import h3_tp
    from rl_engine.validation.models.h3_full_model import FullModelTrace

    engine = json.loads((args.out / "engine.json").read_text())
    model = sl.load_sharded_model(args.checkpoint / "transformer", 1, 0, "cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lora = engine.get("lora")
    if lora:
        h3_lora.attach_lora(
            model,
            1,
            0,
            rank=lora["rank"],
            alpha=lora["alpha"],
            seed=lora["seed"],
            b_zero=lora["b_zero"],
        )
    grid = h3_tp.Grid(1, 1, 0, 0, None, None, "h3_full", "ring")
    installed = _flat(engine.get("install") or [])
    strict = bool(installed[0]["strict_boundaries"]) if installed else args.strict_boundaries
    rows = []
    for path, rec in _listed_captures(args, engine):
        inputs = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in rec["inputs"].items()
        }
        with (
            torch.no_grad(),
            om.torch_default_sdpa(),  # the binding's backends, whatever this process imported
            pm.parallel_model_execution(
                model, grid, sp=False, trace=FullModelTrace(), strict_boundaries=strict
            ),
        ):
            video, audio = model(**inputs)
        rows.append(
            {
                "call": path.stem,
                "rows": int(inputs["token_tags"].numel()),
                "video_bitwise": _same_bytes(video, rec["video"]),
                "audio_bitwise": _same_bytes(audio, rec["audio"]),
                "video_digest": om.tensor_digest(video),
                "in_rollout_digests": om.tensor_digest(video)
                in {c["video"] for c in engine["ranks"][0]["digests"]},
                "max_abs_video": float((video.float().cpu() - rec["video"].float()).abs().max()),
            }
        )
        print(json.dumps(rows[-1]))
    return {
        "grid": "Grid(1, 1), one GPU",
        "strict_boundaries": strict,
        "lora": lora,
        "calls": rows,
        "pass": bool(rows)
        and all(
            r["video_bitwise"] and r["audio_bitwise"] and r["in_rollout_digests"] for r in rows
        ),
    }


# --- actor ---------------------------------------------------------------------------------------


def _actor(args) -> dict[str, Any]:
    from tensordict import TensorDict
    from verl_omni.pipelines.minimax_h3_diffusion_nft.common import pack_video_audio_rows
    from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import (
        MiniMaxH3FlowGRPO,
    )
    from verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter import (
        MiniMaxH3PipelineWithLogProb,
    )

    engine = json.loads((args.out / "engine.json").read_text())
    rows = []
    for path, rec in _listed_captures(args, engine):
        k, mine = rec["omni_kwargs"], rec["inputs"]
        img = k["img_pos_info"]["position_ids"].long()
        aud = k["audio_pos_info"]["position_ids"].long()
        txt = k["text_pos_info"]["position_ids"].long()
        update = k["update_mask"].bool()
        row_t = k["unique_timesteps"][k["inverse_indices"].long()]
        video_t = float(row_t[img[update][0]])
        audio_t = float(row_t[aud[0]])
        branch = SimpleNamespace(
            img_pos=img,
            audio_pos=aud,
            static_kwargs={"token_tags": k["token_tags"]},
            update_mask_dev=update,
        )
        packed = {
            "cu_seqlens": k["packed_seq_params"]["cu_seqlens_q"],
            "img_position_ids": k["img_position_ids"][0],
            "text_pos": txt,
        }
        stub = SimpleNamespace(_h3_max_text_len=args.max_text_len)
        stored = MiniMaxH3PipelineWithLogProb._layout_outputs(
            stub, branch, packed, k["prompt_embeds"]
        )
        video_rows = k["x"][0].index_select(0, img)
        audio_rows = k["audio_x"][0].index_select(0, aud)
        latents = pack_video_audio_rows(video_rows[None], audio_rows[None])[:, None]
        micro = TensorDict(
            {
                **stored,
                "all_next_latents": latents,
                "h3_step_indices": torch.zeros(1, 1, dtype=torch.long),
                "h3_audio_timesteps": torch.tensor([[audio_t]]),
            },
            batch_size=[1],
        )
        actor, _ = MiniMaxH3FlowGRPO.prepare_model_inputs(
            None,
            None,
            latents,
            torch.tensor([[video_t]]),
            stored["prompt_embeds"],
            stored["prompt_embeds_mask"],
            None,
            None,
            micro,
            0,
        )
        diffs = []
        for key in MODEL_KEYS:
            a, b = actor[key].cpu(), mine[key]
            if not _same_bytes(a, b):
                diffs.append(
                    f"{key}: actor {a.dtype} {tuple(a.shape)} vs rollout {b.dtype} {tuple(b.shape)}"
                )
        rows.append({"call": path.stem, "rows": int(mine["token_tags"].numel()), "differ": diffs})
        print(json.dumps(rows[-1]))
    return {"calls": rows, "pass": bool(rows) and all(not r["differ"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=("engine", "canonical", "actor", "native"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--engine-model",
        type=Path,
        help="what vLLM-Omni loads (the release's FL2VA partition, as in verl-omni's recipe); "
        "the bound model always reads --checkpoint/transformer",
    )
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--duration", type=float, default=4.0)  # H3 takes 4-15 s
    ap.add_argument("--max-num-seqs", type=int, default=2)  # 1 would never batch A with B
    ap.add_argument("--record-calls", type=int, default=2)
    ap.add_argument("--strict-boundaries", action="store_true")
    ap.add_argument("--release-weights", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-alpha", type=float, default=128.0)
    ap.add_argument("--lora-seed", type=int, default=7)
    ap.add_argument("--lora-b-zero", action="store_true")
    ap.add_argument("--max-text-len", type=int, default=1024)
    ap.add_argument("--init-timeout", type=float, default=3600.0)
    ap.add_argument(
        "--compare-calls",
        type=Path,
        help="native: the calls/ of a B = 0 engine run to compare with",
    )
    args = ap.parse_args()
    check_args(ap, args)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "engine":
        report = asyncio.run(_engine(args))
    elif args.stage == "native":
        report = asyncio.run(_native(args))
    elif args.stage == "canonical":
        report = _canonical(args)
    else:
        report = _actor(args)
    report.update(stage=args.stage, torch=torch.__version__)
    if torch.cuda.is_available():
        report["device"] = torch.cuda.get_device_name(0)
    _write(args.out / f"{args.stage}.json", report)


if __name__ == "__main__":
    main()
