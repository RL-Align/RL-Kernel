# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Native four-rank RL updates and restart controls, separated from rollout GPUs."""

import json
import os
import re
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import (
    load_tensor_artifact,
    validate_fixture,
)
from rl_engine.validation.models.qwen3_next.qwen3_next_rl_fixture import PolicyFixture


def policy_fixture(path):
    frozen = load_tensor_artifact(path)
    validate_fixture(frozen)
    return PolicyFixture.freeze(
        torch.cat((frozen["prompts"]["64"], frozen["responses"]["64"]), dim=1),
        frozen["old_logprobs"]["64"],
        baseline_identity=frozen["baseline_identity"],
    )


def active_parameter_groups(model):
    """Every parameter tensor in the full actor must participate in the update.

    Routed experts use one fused parameter per layer: unselected expert rows
    may be zero, while at least one selected row must carry a real gradient.
    """
    from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_training import (
        named_trainable_parameters,
    )

    parameters = named_trainable_parameters(model)
    validate_active_parameter_names(parameters)
    return {name: (name,) for name in parameters}


def validate_active_parameter_names(parameters):
    layers = {
        int(match.group(1))
        for name in parameters
        for match in [re.search(r"(?:^|\.)decoder\.layers\.(\d+)\.", name)]
        if match
    }
    if layers != set(range(48)):
        raise ValueError("Active parameters must cover exactly the full 48 decoder layers")
    for layer in range(48):
        for component in ("self_attention.", "mlp.", "pre_mlp_layernorm."):
            prefix = f"decoder.layers.{layer}.{component}"
            if not any(prefix in name for name in parameters):
                raise ValueError(f"Full actor omits an active parameter path: {prefix}")
    for component in ("embedding.", "output_layer.", "final_layernorm."):
        if not any(component in name for name in parameters):
            raise ValueError(f"Full actor omits {component}")


def policy_step(args, model, optimizer, scheduler, fixture, *, step, required_groups):
    """Track consumed samples only after native loss/backward/update/scheduler succeed."""
    from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_training import (
        native_policy_update,
    )

    before, after = (step - 1) * 4, step * 4
    if args.consumed_train_samples != before or scheduler.num_steps != before:
        raise ValueError("Native update starts from inconsistent sample/scheduler progress")
    audit = native_policy_update(
        args, model, optimizer, scheduler, fixture, step=step, required_groups=required_groups
    )
    if scheduler.num_steps != after or args.consumed_train_samples != before:
        raise ValueError("Native VIME scheduler/sample-counter behavior differs from pinned API")
    # Pinned VIME train_one_step advances the scheduler but owns no consumed
    # sample counter. The controller advances that counter after its success.
    args.consumed_train_samples = after
    audit["progress"] = {
        "consumed_before": before,
        "consumed_after": after,
        "scheduler_before": before,
        "scheduler_after": after,
        "sample_counter_owner": "qwen3_next_native_training.policy_step",
    }
    return audit


def retire_prior_checkpoints(checkpoint, root, identity):
    """Called only after all-rank publication/load verifies the successor bytes."""
    from qwen3_next_runtime import retire_directory

    checkpoint, root = Path(checkpoint), Path(root)
    if (
        root.is_symlink()
        or checkpoint.is_symlink()
        or checkpoint.resolve().parent != root.resolve()
    ):
        raise ValueError("Checkpoint retention is restricted to this run's direct directory")
    checkpoint, root = checkpoint.resolve(), root.resolve()
    owner = root / "RUN_OWNER.json"
    if owner.is_symlink() or json.loads(owner.read_text()) != identity:
        raise ValueError("Checkpoint retention lacks a matching run owner")
    successor = json.loads((checkpoint / "COMPLETE.json").read_text())
    if successor.get("identity") != identity or successor.get("status") != "complete":
        raise ValueError("Checkpoint successor is not a published state for this run")
    retired = []
    for path in root.iterdir():
        if (
            not path.is_dir()
            or path == checkpoint
            or path.suffix == ".partial"
            or path.name.startswith(".")
        ):
            continue
        marker = path / "COMPLETE.json"
        if path.is_symlink() or marker.is_symlink():
            raise ValueError("Checkpoint retention cannot follow symlinks")
        previous = json.loads(marker.read_text())
        if (
            previous.get("identity") != identity
            or previous.get("status") != "complete"
            or previous.get("step", -1) >= successor["step"]
        ):
            raise ValueError("Checkpoint retention found an unrelated or ambiguous successor")
        retirement = retire_directory(
            path,
            root,
            identity,
            proof={
                "complete": previous,
                "successor": str(checkpoint),
                "successor_complete": successor,
            },
        )
        retired.append({"path": str(path), "complete": previous, "retirement": retirement})
    return retired


def _rank_main(rank, request, staging):
    from qwen3_next_contract import atomic_json

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
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from qwen3_next_training_checkpoint import (
            _rank_zero_call,
            load_training_checkpoint,
            save_training_checkpoint,
            training_state_identity,
        )
        from qwen3_next_training_export import export_training_weights

        from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime_runtime import (
            create_actor,
        )
        from rl_engine.validation.models.qwen3_next.qwen3_next_binary import (
            verify_imported_provider,
        )

        fixture = policy_fixture(request["fixture"])
        manifest = request["manifest"]
        identity = request["identity"]
        provider_binary = verify_imported_provider(
            request["provider_binary"], source_tree=identity["source_tree"]
        )
        if identity["fixture_sha256"] != fixture.content_sha256:
            raise ValueError("Native RL worker fixture identity changed")
        args, model, optimizer, scheduler = create_actor(manifest["checkpoint"], training=True)
        if optimizer is None or scheduler is None or len(model) != 1:
            raise ValueError("Native RL worker requires the actual optimizer and scheduler")
        tracker = get_cuda_rng_tracker()
        groups = active_parameter_groups(model)
        phase = request["phase"]
        record = {
            "rank": rank,
            "phase": phase,
            "identity": identity,
            "pid": os.getpid(),
            "provider_binary": provider_binary,
            "native_entrypoints": [
                "vime.backends.megatron_utils.model.setup_model_and_optimizer",
                "vime.backends.megatron_utils.model.train_one_step",
                "vime.backends.megatron_utils.loss.policy_loss_function",
            ],
        }
        restored = None
        if request.get("checkpoint"):
            restored = load_training_checkpoint(
                args,
                model,
                optimizer,
                scheduler,
                fixture,
                request["checkpoint"],
                identity=identity,
                tracker=tracker,
            )
            record["restored"] = restored
        expected_previous = {
            "update1": 0,
            "update2": 1,
            "reload": 2,
            "continuous": 0,
            "resumed": 2,
        }[phase]
        actual_previous = restored["step"] if restored else 0
        recover_published_update = phase in ("update1", "update2") and actual_previous == (
            1 if phase == "update1" else 2
        )
        if actual_previous != expected_previous and not recover_published_update:
            raise ValueError("Native RL update resumed from the wrong complete step")
        if phase in ("continuous", "resumed"):
            record["controls"] = {}
            for step in (1, 2, 3) if phase == "continuous" else (3,):
                audit = policy_step(
                    args, model, optimizer, scheduler, fixture, step=step, required_groups=groups
                )
                state = training_state_identity(
                    args,
                    model,
                    optimizer,
                    scheduler,
                    fixture,
                    step=step,
                    sample_position=step * 4,
                    identity=identity,
                    tracker=tracker,
                )
                expected = request["expected_states"][str(step)][rank] if step < 3 else None
                if expected is not None and state != expected:
                    differing = sorted(
                        key
                        for key in expected.keys() | state.keys()
                        if expected.get(key) != state.get(key)
                    )
                    atomic_json(
                        Path(staging) / f"divergence-rank-{rank}.json",
                        {
                            "rank": rank,
                            "step": step,
                            "differing": differing,
                            "expected": expected,
                            "actual": state,
                            "update": audit,
                        },
                    )
                    raise ValueError(
                        f"Continuous control diverged from original update {step} "
                        f"on rank {rank}: {', '.join(differing)}"
                    )
                record["controls"][str(step)] = {"update": audit, "state": state}
            verify_imported_provider(
                request["provider_binary"], source_tree=identity["source_tree"]
            )
            atomic_json(Path(staging) / f"rank-{rank}.json", record)
            dist.barrier()
            return
        step = 1 if phase == "update1" else 2
        if recover_published_update:
            record["update"] = restored["rank_evidence"]["update_evidence"]
            if record["update"].get("step") != step:
                raise ValueError("Published checkpoint omits the committed update evidence")
            record["recovered_committed_update"] = True
            checkpoint = request["checkpoint"]
        elif phase != "reload":
            record["update"] = policy_step(
                args, model, optimizer, scheduler, fixture, step=step, required_groups=groups
            )
            saved = save_training_checkpoint(
                args,
                model,
                optimizer,
                scheduler,
                fixture,
                request["checkpoint_root"],
                step=step,
                sample_position=step * 4,
                identity=identity,
                tracker=tracker,
                export_bytes=request["export_bytes"],
                update_evidence=record["update"],
            )
            record["saved"] = saved
            checkpoint = saved["checkpoint"]
        else:
            checkpoint = request["checkpoint"]
        record["retired_checkpoints"] = _rank_zero_call(
            lambda: retire_prior_checkpoints(checkpoint, request["checkpoint_root"], identity),
            "retire previous complete checkpoints after verified successor",
        )
        checkpoint_bytes = sum(
            path.stat().st_size for path in Path(checkpoint).rglob("*") if path.is_file()
        )
        # Export must reflect the reloaded BF16 actor in the reload phase. Never
        # reuse the update2 export as purported evidence that loading worked.
        record["export"] = export_training_weights(
            args,
            model,
            request["export_root"],
            step=step,
            weight_version={"update1": "update-1", "update2": "update-2", "reload": "reload-2"}[
                phase
            ],
            identity=identity,
            base_checkpoint=manifest["checkpoint"],
            base_checkpoint_identity=manifest["checkpoint_identity"],
            checkpoint_bytes=checkpoint_bytes,
            retained_complete_bytes=checkpoint_bytes,
        )
        record["state"] = training_state_identity(
            args,
            model,
            optimizer,
            scheduler,
            fixture,
            step=step,
            sample_position=step * 4,
            identity=identity,
            tracker=tracker,
        )
        verify_imported_provider(request["provider_binary"], source_tree=identity["source_tree"])
        atomic_json(Path(staging) / f"rank-{rank}.json", record)
        dist.barrier()
    except BaseException:
        # One rank must not take the others down by deadlock: tearing down the
        # process group while its peers sit in a collective can block forever.
        # Report, then exit so the spawning parent terminates
        # the sibling ranks and the stage fails instead of running out its budget.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(3)
    finally:
        dist.destroy_process_group()


def run_training_phase(request, output):
    """One fresh native actor lifetime, then all CUDA processes exit."""
    from qwen3_next_contract import atomic_json

    if request["phase"] not in ("update1", "update2", "reload", "continuous", "resumed"):
        raise ValueError("Unknown native RL phase")
    if torch.cuda.device_count() != 4:
        raise ValueError("Native training requires exactly four visible GPUs")
    identity, manifest = request["identity"], request["manifest"]
    if (
        identity.get("topology") != {"tp": 4, "pp": 1, "cp": 1, "ep": 1}
        or identity.get("architecture_layers") != 48
        or identity.get("checkpoint") != manifest.get("checkpoint")
        or identity.get("checkpoint_identity") != manifest.get("checkpoint_identity")
    ):
        raise ValueError("Native training request changed base checkpoint/depth/topology")
    staging = Path(output).parent / f"native-training-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    mp.spawn(_rank_main, args=(request, str(staging)), nprocs=4, join=True)
    records = [json.loads((staging / f"rank-{rank}.json").read_text()) for rank in range(4)]
    for rank, record in enumerate(records):
        if (record["rank"], record["phase"], record["identity"]) != (
            rank,
            request["phase"],
            request["identity"],
        ):
            raise ValueError("Native RL rank record identity mismatch")
        if record.get("export") != records[0].get("export"):
            raise ValueError("Native RL ranks disagree on the published HF export")
        if record.get("provider_binary") != records[0].get("provider_binary"):
            raise ValueError("Native RL ranks imported different CUDA provider binaries")
    result = {
        "schema_version": 1,
        "status": "passed",
        "phase": request["phase"],
        "identity": request["identity"],
        "ranks": records,
        "export": records[0].get("export"),
        "checkpoint": records[0].get("saved", {}).get("checkpoint", request.get("checkpoint")),
        "all_rank_processes_exited": True,
        "provider_binary": request["provider_binary"],
    }
    atomic_json(output, result)
    return result
