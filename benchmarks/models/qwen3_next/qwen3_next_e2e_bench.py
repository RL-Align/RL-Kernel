#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""End-to-end speed of the strict Qwen3-Next path against stock vLLM and stock VIME.

Run it after a passing certified chain at the same commit, on the same four GPUs:

    python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 ...
    python benchmarks/models/qwen3_next/qwen3_next_e2e_bench.py --run <l2 run dir> --out <dir>
    python benchmarks/models/qwen3_next/plot_qwen3_next_e2e_bench.py <dir>/report.json

Phases and arms (the same four GPUs and synthetic inputs; rollout at the l2 run's TP4/EP):
- rollout: vLLM generate. ``strict`` is the certified RL-Kernel engine (CUSTOM attention,
  shared model, batch-invariant mode, eager) with vLLM's own scheduler and no tracing.
  ``native_graph`` / ``native_eager`` / ``native_bi`` are stock vLLM with CUDA graphs,
  without them, and with ``VLLM_BATCH_INVARIANT=1``.
- replay: one VIME/Megatron forward over packed prompt+response plus the selected-token
  logprob. ``strict`` uses the RL-Kernel spec and logprob provider, ``native`` the stock
  VIME Qwen3-Next spec and its logprob.
- train: VIME ``train_one_step`` (4 samples per step, one-sample micro-batches on each
  data-parallel rank, CPU-offloaded Adam), with the optimizer step timed separately.

Actor layouts (``layouts`` in the report). The strict replay/train arm runs the l2 run's
certified layout: TP4 with its EP (training only on an EP1 run). The stock VIME arm cannot
run TP4 at all: Qwen3-Next has two KV heads, and when TP exceeds them Megatron gives each
rank the query heads of a whole KV group, keeps its share for attention, but does not
narrow the attention output gate, so the forward fails on the gate's shape. Its expert
weights also only load with expert tensor parallel 1 (Megatron's grouped expert layers
carry no tensor-parallel flag, so VIME's loader copies the whole expert tensor). So stock
VIME runs its own recipe's layout on the same four GPUs: TP2 with sequence parallel (without
it Megatron's MoE layer refuses to train at TP > 1), expert tensor parallel 1, experts over all
four ranks, data parallel 2. Each step processes the same samples; the
two data-parallel replicas split them (a single sample leaves one replica a filler copy)
and the step time is the slowest rank's.

Every arm × phase × workload runs in its own process; rounds alternate the arm order.
Timing only; the l2 run is the consistency evidence for the strict arm.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/validation/models/qwen3_next"))

from benchmarks.models.qwen3_next.e2e_bench_cases import (  # noqa: E402
    PHASES,
    STOCK_VIME_LAYOUT,
    STRICT_ARM,
    WORKLOADS,
    actor_layout,
    actor_unsupported,
    arm_environment,
    arm_order,
    data_parallel_share,
    merge_rounds,
    packed_lengths,
    prompts,
    ratios,
    response_positions,
    sequences,
    shape_id,
    summarize,
)

VERSION_PACKAGES = (
    "torch",
    "vllm",
    "flashinfer-python",
    "triton",
    "transformer-engine",
    "megatron-core",
    "vime",
)


def _git(source, *args):
    return subprocess.check_output(["git", *args], cwd=source, text=True).strip()


def load_run(run_dir):
    """A passed certified l2 stage: its manifest, identity and clean source clone."""
    from qwen3_next_l2 import model_identity

    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "run.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    if manifest.get("stage") != "l2" or result.get("full_model_l2") != "passed":
        raise SystemExit("--run must be a passed l2 stage run")
    source = Path(manifest["source"])
    if _git(source, "rev-parse", "HEAD^{tree}") != manifest["source_tree"] or _git(
        source, "status", "--porcelain", "--untracked-files=no"
    ):
        raise SystemExit("The l2 run's source clone changed since the run")
    return manifest, model_identity(manifest)


# --------------------------------------------------------------------------- rollout


def native_engine_arguments(manifest, arm, workload):
    spec = WORKLOADS[workload]
    checkpoint = manifest["checkpoint"]
    return dict(
        model=checkpoint,
        tokenizer=checkpoint,
        skip_tokenizer_init=True,
        trust_remote_code=False,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        enable_expert_parallel=manifest["topology"]["ep"] == 4,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        seed=1234,
        enforce_eager=arm == "native_eager",
        enable_prefix_caching=False,
        max_model_len=spec["max_model_len"],
        max_num_seqs=64,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85,
        kv_cache_memory_bytes=spec["kv_cache_bytes"],
        generation_config="vllm",
    )


def strict_engine_arguments(identity, workload):
    """The certified engine without the L2 trace scheduler and teacher forcing."""
    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_native_rollout import (
        engine_arguments,
    )
    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_model import (
        register_shared_model,
    )
    from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next.profile import (
        activate_for_identity,
    )

    if activate_for_identity(identity) != "certified":
        raise RuntimeError("The strict arm must run the certified provider profile")
    arguments = engine_arguments(identity, register_shared_model())
    for name in ("scheduler_cls", "long_prefill_token_threshold", "logits_processors"):
        arguments.pop(name)
    spec = WORKLOADS[workload]
    arguments.update(
        max_model_len=spec["max_model_len"], kv_cache_memory_bytes=spec["kv_cache_bytes"]
    )
    return arguments


def rollout_worker(manifest, identity, arm, workload, repeats):
    from vllm import LLM, SamplingParams

    if arm == STRICT_ARM:
        arguments = strict_engine_arguments(identity, workload)
    else:
        arguments = native_engine_arguments(manifest, arm, workload)
    engine = LLM(**arguments)
    records, first_rows = {}, {}
    try:
        for shape in WORKLOADS[workload]["rollout"]:
            rows = prompts(shape)

            def generate(max_tokens, rows=rows):
                params = SamplingParams(
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    seed=1234,
                    ignore_eos=True,
                    max_tokens=max_tokens,
                    detokenize=False,
                )
                start = time.perf_counter()
                output = engine.generate(
                    [{"prompt_token_ids": row} for row in rows], params, use_tqdm=False
                )
                seconds = time.perf_counter() - start
                tokens = [list(item.outputs[0].token_ids) for item in output]
                if any(len(row) != max_tokens for row in tokens):
                    raise RuntimeError("The engine stopped before max_tokens with ignore_eos")
                return seconds, tokens

            generate(shape[2])
            prefill, total, outputs = [], [], []
            for _ in range(repeats):
                prefill.append(generate(1)[0])
                seconds, tokens = generate(shape[2])
                total.append(seconds)
                outputs.append(tokens)
            prompt, batch, response = shape
            first_rows[shape] = outputs[0][0]
            single = first_rows.get((prompt, 1, response))
            records[shape_id(shape)] = {
                "prefill_values_s": prefill,
                "total_values_s": total,
                "same_tokens_across_repeats": all(item == outputs[0] for item in outputs),
                "row0_equals_batch1": None if batch == 1 else single == outputs[0][0],
                "tokens_sha256": hashlib.sha256(json.dumps(outputs[0]).encode()).hexdigest(),
            }
    finally:
        engine.llm_engine.engine_core.shutdown()
    return {"engine_arguments": _public_arguments(arguments), "shapes": records}


def _public_arguments(arguments):
    """Engine arguments without checkpoint paths or the identity record."""
    hidden = ("model", "tokenizer", "additional_config", "hf_overrides")
    return {key: value for key, value in arguments.items() if key not in hidden}


# --------------------------------------------------------------------------- VIME actor


def _synchronized(function):
    """Wall time of ``function`` on the slowest of the four ranks."""
    import torch
    import torch.distributed as dist

    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    function()
    torch.cuda.synchronize()
    value = torch.tensor([time.perf_counter() - start], device="cuda", dtype=torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def _rank_max(seconds):
    import torch
    import torch.distributed as dist

    value = torch.tensor([seconds], device="cuda", dtype=torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def replay_shapes(manifest, arm, workload, repeats):
    import torch
    from megatron.core import mpu
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.utils import get_model_config
    from vime.backends.megatron_utils import loss as vime_loss

    from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime_runtime import (
        create_actor,
    )

    spec = WORKLOADS[workload]
    layout = actor_layout(manifest["topology"]["ep"], arm)
    _, model, _, _ = create_actor(
        manifest["checkpoint"],
        training=False,
        ep=layout["ep"],
        providers=arm == STRICT_ARM,
        seq_length=spec["max_model_len"],
        tp=layout["tp"],
    )
    dp_rank, dp_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    if dp_size != layout["dp"]:
        raise RuntimeError(f"Expected data parallel {layout['dp']}, Megatron built {dp_size}")
    model[0].eval()
    raw = model[0]
    while hasattr(raw, "module"):
        raw = raw.module
    group = mpu.get_tensor_model_parallel_group()
    if get_model_config(model[0]).sequence_parallel != layout["sp"]:
        raise RuntimeError(f"Expected sequence parallel {layout['sp']}")
    multiple = layout["tp"] if layout["sp"] else 1
    captured = []

    def keep_logits(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    raw.output_layer.register_forward_hook(keep_logits)
    records = {}
    for shape in spec["replay"]:
        rows, _filler = data_parallel_share(sequences(shape), dp_rank, dp_size)
        lengths = packed_lengths([len(row) for row in rows], multiple)
        flat = [t for row in rows for t in row]
        flat += [0] * (sum(lengths) - len(flat))
        tokens = torch.tensor(flat, device="cuda").reshape(1, -1)
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
        )
        packed = PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
            qkv_format="thd",
        )
        positions, targets = (
            torch.tensor(value, device="cuda") for value in response_positions(shape, rows)
        )

        def step(tokens=tokens, packed=packed, positions=positions, targets=targets):
            captured.clear()
            with torch.no_grad():
                model[0](
                    input_ids=tokens,
                    position_ids=None,
                    attention_mask=None,
                    labels=None,
                    packed_seq_params=packed,
                )
                if len(captured) != 1:
                    raise RuntimeError("Expected exactly one LM-head output per forward")
                logits = captured[0].reshape(-1, captured[0].shape[-1])[positions]
                # Looked up at call time: the RL-Kernel provider when installed, else VIME's.
                vime_loss.calculate_log_probs_and_entropy(logits, targets, group, with_entropy=True)

        step()
        records[shape_id(shape)] = {"values_s": [_synchronized(step) for _ in range(repeats)]}
    return records


def train_batch(shape, device, dp_rank=0, dp_size=1):
    """VIME policy-loss fields for a data-parallel rank's synthetic samples.

    Same keys as the acceptance fixture. The four samples of a step are split round-robin
    over the data-parallel ranks, as VIME gives each rank its own share of the rollout.
    """
    import torch

    prompt, batch, response = shape
    if batch != 4 or batch % dp_size:
        raise ValueError("The pinned policy settings train four samples per step")
    advantages = [1.0, -0.5, 0.75, -0.25]
    rows = sequences(shape)
    local = list(range(batch))[dp_rank::dp_size]
    return {
        "tokens": [torch.tensor(rows[i], device=device) for i in local],
        "rollout_log_probs": [
            torch.full((response,), -2.0, dtype=torch.float32, device=device) for _ in local
        ],
        "advantages": [
            torch.full((response,), advantages[i], dtype=torch.float32, device=device)
            for i in local
        ],
        "loss_masks": [torch.ones(response, dtype=torch.int32, device=device) for _ in local],
        "rollout_mask_sums": torch.full((len(local),), float(response), device=device),
        "total_lengths": [prompt + response] * len(local),
        "response_lengths": [response] * len(local),
        "partition": local,
    }


def train_shapes(manifest, arm, workload, repeats):
    import torch
    from megatron.core import mpu
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.utils import get_model_config
    from vime.backends.megatron_utils.data import DataIterator
    from vime.backends.megatron_utils.model import train_one_step

    from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime_runtime import (
        create_actor,
    )

    spec = WORKLOADS[workload]
    layout = actor_layout(manifest["topology"]["ep"], arm)
    args, model, optimizer, scheduler = create_actor(
        manifest["checkpoint"],
        training=True,
        ep=layout["ep"],
        providers=arm == STRICT_ARM,
        seq_length=spec["max_model_len"],
        tp=layout["tp"],
    )
    dp_rank, dp_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    if dp_size != layout["dp"]:
        raise RuntimeError(f"Expected data parallel {layout['dp']}, Megatron built {dp_size}")
    for chunk in model:
        chunk.train()
        config = get_model_config(chunk)
        config.grad_scale_func = optimizer.scale_loss
        config.finalize_model_grads_func = finalize_model_grads
        config.timers = None
    shape = spec["train"]
    batch = train_batch(shape, torch.cuda.current_device(), dp_rank, dp_size)
    local = len(batch["tokens"])
    original_step = optimizer.step
    optimizer_seconds = []

    def timed_optimizer_step(*step_args, **step_kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_step(*step_args, **step_kwargs)
        torch.cuda.synchronize()
        optimizer_seconds.append(time.perf_counter() - start)
        return result

    optimizer.step = timed_optimizer_step
    steps, optimizer_steps = [], []
    for index in range(1 + repeats):
        iterator = DataIterator(dict(batch), [[i] for i in range(local)])
        seconds = _synchronized(
            lambda iterator=iterator, index=index: train_one_step(
                args,
                rollout_id=0,
                step_id=index,
                data_iterator=[iterator],
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=scheduler,
                num_microbatches=local,
                step_global_batch_size=shape[1],
            )
        )
        if len(optimizer_seconds) != index + 1:
            raise RuntimeError("train_one_step must call the optimizer exactly once")
        optimizer_time = _rank_max(optimizer_seconds[-1])
        if index:  # the first step warms up allocators and kernels
            steps.append(seconds)
            optimizer_steps.append(optimizer_time)
    return {shape_id(shape): {"step_values_s": steps, "optimizer_values_s": optimizer_steps}}


def _actor_rank(rank, phase, manifest, arm, workload, repeats, staging):
    import torch
    import torch.distributed as dist

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{staging}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=60),
        device_id=torch.device("cuda", rank),
    )
    try:
        run = replay_shapes if phase == "replay" else train_shapes
        records = run(manifest, arm, workload, repeats)
        if rank == 0:
            Path(staging, "rank-0.json").write_text(json.dumps(records))
        dist.barrier()
    except BaseException:
        # Exit instead of tearing down a group whose peers may sit in a collective.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(3)
    finally:
        dist.destroy_process_group()


def actor_worker(phase, manifest, arm, workload, repeats, result_path):
    import torch.multiprocessing as mp

    staging = result_path.with_name(result_path.stem + "-ranks")
    staging.mkdir(parents=True, exist_ok=False)
    mp.spawn(
        _actor_rank,
        args=(phase, manifest, arm, workload, repeats, str(staging)),
        nprocs=4,
        join=True,
    )
    return {"shapes": json.loads((staging / "rank-0.json").read_text())}


# --------------------------------------------------------------------------- orchestration


def provider_build(manifest):
    """The RL-Kernel extension must be the one the l2 run built in its source clone."""
    import rl_engine._C as extension

    path = Path(extension.__file__).resolve()
    if not path.is_relative_to(Path(manifest["source"]).resolve()):
        raise RuntimeError("rl_engine._C does not come from the l2 run's source clone")
    return path.name


def worker_main(args):
    manifest, identity = load_run(args.run)
    record = {"phase": args.worker, "arm": args.arm, "workload": args.workload}
    if args.arm == STRICT_ARM:
        record["provider_build"] = provider_build(manifest)
    if args.worker == "rollout":
        record.update(rollout_worker(manifest, identity, args.arm, args.workload, args.repeats))
    else:
        record.update(
            actor_worker(args.worker, manifest, args.arm, args.workload, args.repeats, args.result)
        )
    args.result.write_text(json.dumps(record, indent=2))
    return 0


def versions():
    from importlib import metadata

    found = {}
    for name in VERSION_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    try:
        import megatron.core

        found["megatron-core"] = found["megatron-core"] or megatron.core.__version__
    except Exception as exc:  # a broken optional import must not hide the other versions
        found["megatron-core-import-error"] = type(exc).__name__
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
    )
    found["gpus"] = sorted(set(line.strip() for line in gpus.splitlines() if line.strip()))
    print(json.dumps(found))
    return 0


def wait_gpus_idle(timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        busy = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if not busy:
            return True
        time.sleep(1)
    return False


# Refusals by the engine itself: the arm does not exist for this model, it did not crash.
UNSUPPORTED_MESSAGES = ("VLLM batch_invariant mode is not supported for",)


def unsupported_reason(log_text):
    for marker in UNSUPPORTED_MESSAGES:
        index = log_text.find(marker)
        if index >= 0:
            return log_text[index:].splitlines()[0].strip()
    return None


def run_worker(manifest, base_env, script, out, phase, arm, workload, round_index, args):
    name = f"{phase}-{arm}-{workload}-round{round_index}"
    result = out / "workers" / f"{name}.json"
    log = out / "logs" / f"{name}.log"
    command = [
        manifest["python"],
        str(script),
        "--worker",
        phase,
        "--arm",
        arm,
        "--workload",
        workload,
        "--repeats",
        str(args.repeats),
        "--run",
        str(args.run),
        "--result",
        str(result),
    ]
    with log.open("w") as handle:
        code = subprocess.run(
            command,
            cwd=manifest["source"],
            env=arm_environment(base_env, arm),
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        ).returncode
    idle = wait_gpus_idle()
    if code or not result.is_file():
        refusal = unsupported_reason(log.read_text(errors="replace"))
        if refusal is not None:
            return {"unsupported": refusal, "gpus_idle_after": idle}
        return {"error": f"worker exited with {code}; see logs/{log.name}", "gpus_idle_after": idle}
    record = json.loads(result.read_text())
    record["gpus_idle_after"] = idle
    return record


def orchestrate(args):
    from qwen3_next_run import stage_environment

    manifest, identity = load_run(args.run)
    out = args.out.resolve()
    for directory in ("workers", "logs"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    base_env = stage_environment(manifest, out, "bench")
    script = Path(manifest["source"]) / "benchmarks/models/qwen3_next/qwen3_next_e2e_bench.py"
    probe = subprocess.run(
        [manifest["python"], str(script), "--versions"],
        cwd=manifest["source"],
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    raw = {}
    for phase in args.phases:
        rounds = args.train_rounds if phase == "train" else args.rounds
        for round_index in range(rounds):
            for arm in arm_order(phase, round_index):
                if phase == "rollout" and arm not in args.rollout_arms:
                    continue
                for workload in args.workloads:
                    refusal = (
                        None
                        if phase == "rollout"
                        else actor_unsupported(manifest["topology"]["ep"], phase, arm)
                    )
                    record = (
                        {"unsupported": refusal}
                        if refusal
                        else run_worker(
                            manifest, base_env, script, out, phase, arm, workload, round_index, args
                        )
                    )
                    raw.setdefault(phase, {}).setdefault(arm, {}).setdefault(workload, []).append(
                        record
                    )
                    print(
                        f"{phase} {arm} {workload} round {round_index}: "
                        f"{'failed' if 'error' in record else 'done'}",
                        flush=True,
                    )
    report = build_report(manifest, identity, raw, json.loads(probe.stdout), args)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


def _all_known(values):
    """True/False over the recorded values; None when the check does not apply."""
    known = [value for value in values if value is not None]
    return all(known) if known else None


def build_report(manifest, identity, raw, found_versions, args):
    shapes_by_id = {
        (workload, phase, shape_id(shape)): shape
        for workload, spec in WORKLOADS.items()
        for phase in PHASES
        for shape in (spec[phase] if phase != "train" else [spec[phase]])
    }
    phases = {}
    for phase, arms in raw.items():
        results, extras = {}, {}
        for arm, workloads in arms.items():
            for workload, records in workloads.items():
                failed = [record["error"] for record in records if "error" in record]
                refused = sorted({r["unsupported"] for r in records if "unsupported" in r})
                if refused:
                    extras.setdefault("unsupported", []).append(
                        {"arm": arm, "workload": workload, "reasons": refused}
                    )
                good = [r for r in records if "error" not in r and "unsupported" not in r]
                target = results.setdefault(arm, {}).setdefault(workload, {})
                if failed:
                    extras.setdefault("failures", []).append(
                        {"arm": arm, "workload": workload, "errors": failed}
                    )
                if not good:
                    continue
                if phase == "rollout":
                    extras.setdefault("engine_arguments", {})[arm] = good[0]["engine_arguments"]
                for key, pooled in merge_rounds([record["shapes"] for record in good]).items():
                    summary = summarize(phase, shapes_by_id[(workload, phase, key)], pooled)
                    summary["checks"] = {
                        name: _all_known(values) for name, values in pooled["checks"].items()
                    }
                    summary["rounds"] = len(good)
                    target[key] = summary
        if phase != "rollout":
            extras["layouts"] = {arm: actor_layout(manifest["topology"]["ep"], arm) for arm in arms}
        phases[phase] = {"results": results, "strict_over_native": ratios(phase, results), **extras}
    return {
        "schema_version": 1,
        "scope": "qwen3_next_e2e_speed",
        "rl_kernel_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree"],
        "tracked_tree_dirty": False,
        "l2_run_id": manifest["run_id"],
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "topology": manifest["topology"],
        "versions": found_versions,
        "settings": {
            "repeats": args.repeats,
            "rounds": args.rounds,
            "train_rounds": args.train_rounds,
            "workloads": {name: WORKLOADS[name] for name in args.workloads},
            "statistic": "median over rounds x repeats; actor phases take the slowest rank",
            "strict_over_native": "strict seconds / native seconds; above 1 means strict is slower",
            "stock_vime_layout": STOCK_VIME_LAYOUT,
            "stock_vime_layout_reason": (
                "stock VIME cannot run Qwen3-Next at TP4 (2 KV heads; Megatron's attention "
                "output gate is not narrowed below one KV group per rank), so it runs its "
                "recipe's TP2 with sequence parallel and expert tensor parallel 1 on the same "
                "four GPUs"
            ),
        },
        "phases": phases,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, help="passed certified l2 stage run directory")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phases", type=lambda v: v.split(","), default=list(PHASES))
    parser.add_argument("--workloads", type=lambda v: v.split(","), default=list(WORKLOADS))
    parser.add_argument(
        "--rollout-arms",
        type=lambda v: v.split(","),
        default=["strict", "native_graph", "native_eager", "native_bi"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--train-rounds", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3 * 3600)
    parser.add_argument("--worker", choices=PHASES)
    parser.add_argument("--arm")
    parser.add_argument("--workload", choices=tuple(WORKLOADS))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--versions", action="store_true")
    args = parser.parse_args(argv)
    if args.versions or args.worker:
        if args.worker and None in (args.arm, args.workload, args.result, args.run):
            parser.error("--worker needs --arm, --workload, --result and --run")
        return args
    if args.run is None or args.out is None:
        parser.error("--run and --out are required")
    unknown = [p for p in args.phases if p not in PHASES]
    unknown += [w for w in args.workloads if w not in WORKLOADS]
    if unknown:
        parser.error(f"unknown phases/workloads: {unknown}")
    if args.repeats < 1 or args.rounds < 1 or args.train_rounds < 1:
        parser.error("--repeats, --rounds and --train-rounds must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.versions:
        return versions()
    if args.worker:
        return worker_main(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
