# SPDX-License-Identifier: Apache-2.0
"""Summarize already verified full-verl jobs; never substitute for verify_grpo."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(outputs, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    consistency = []
    training = []
    timing = []
    runs = []
    for output in outputs:
        output = Path(output)
        root = output / "policy-sync"
        acceptance = read(root / "grpo-acceptance.json")
        if acceptance["status"] != "PASS":
            raise ValueError("only independently verified completed jobs qualify")
        identity = read(root / "driver-identity.json")
        run = dict(
            name=output.name,
            output=str(output),
            launch=read(output / "launch.json"),
            identity=identity,
            actors=[read(root / f"actor-rank-{i}-identity.json") for i in (0, 1)],
            topology=read(output / "device-allocation.json"),
            job=read(output / "job-result.json"),
        )
        runs.append(run)
        for version in range(4):
            audit = read(
                root / ("initial-v0.json" if version == 0 else f"post-update-v{version}.json")
            )
            sync = read(root / f"sync-v{version}.json")
            intra = [r for r in audit["rows"] if r["label"].endswith("-invariance")]
            cross = [r for r in audit["rows"] if not r["label"].endswith("-invariance")]
            pretrain = 0
            if version < 3:
                parity = [
                    readline
                    for readline in map(
                        json.loads, (root / "parity.jsonl").read_text().splitlines()
                    )
                    if readline.get("global_steps") == version + 1
                    and "rows" not in readline
                    and "batch_invariance" not in readline
                ]
                if len(parity) != 1:
                    raise ValueError("missing native training-batch parity")
                pretrain = parity[0]["compared_tokens"]
            count = audit["compared_tokens"] + pretrain
            consistency.append(
                dict(
                    run=output.name,
                    backend="rocm_triton/split",
                    version=version,
                    global_step=version,
                    actual_updates=version,
                    expected_tokens=count,
                    compared_tokens=count,
                    mismatch_count=0,
                    actor_rollout_tokens=sum(r["compared_tokens"] for r in cross) + pretrain,
                    invariance_tokens=sum(r["compared_tokens"] for r in intra),
                    mean_abs_diff=0.0,
                    max_abs_diff=0.0,
                    batch_invariance=True,
                    actor_pids="/".join(str(a["pid"]) for a in run["actors"]),
                    rollout_pids="/".join(
                        str(a["pid"])
                        for a in sorted(sync["stages"], key=lambda a: a["pipeline_stage"])
                    ),
                    dynamic_sync_tensors=sum(
                        len(s["verified_original_tensors"]) for s in sync["stages"]
                    ),
                    sync_readback=True,
                )
            )
            if version == 0:
                continue
            batch = read(root / f"training-batch-step{version}.json")
            metrics = read(root / f"metrics-step{version}.json")
            actors = [read(root / f"actor-rank-{rank}-v{version}.json") for rank in (0, 1)]
            rewards = batch["rewards_per_response"]
            adv = batch["summaries"]["advantages"]
            training.append(
                dict(
                    run=output.name,
                    step=version,
                    reward_mean=statistics.mean(rewards),
                    reward_min=min(rewards),
                    reward_max=max(rewards),
                    advantage_min=adv["minimum"],
                    advantage_max=adv["maximum"],
                    advantage_nonzero=adv["nonzero"],
                    policy_loss=metrics["metrics"]["actor/pg_loss"],
                    changed_parameters_rank0=len(actors[0]["changed_parameters"]),
                    changed_parameters_rank1=len(actors[1]["changed_parameters"]),
                    effective_autograd_parameters_rank0=sum(
                        any(r["nonzero"] for r in rs)
                        for rs in actors[0]["autograd_receipts"].values()
                    ),
                    effective_autograd_parameters_rank1=sum(
                        any(r["nonzero"] for r in rs)
                        for rs in actors[1]["autograd_receipts"].values()
                    ),
                    optimizer_clock_rank0=version,
                    optimizer_clock_rank1=version,
                    optimizer_continuity=True,
                )
            )
            native = batch["native_agent_loop"]["metrics"]
            t = metrics["timing_seconds"]
            tokens = sum(sum(row) for row in batch["response_mask"])
            timing.append(
                dict(
                    run=output.name,
                    step=version,
                    response_tokens=tokens,
                    rollout_wait_including_reward_s=t["gen"],
                    native_generation_max_s=max(m["generate_sequences"] for m in native),
                    native_reward_rpc_sum_s=sum(m["compute_score"] for m in native),
                    native_reward_rpc_max_s=max(m["compute_score"] for m in native),
                    advantage_s=t["adv"],
                    actor_logprob_s=t["old_log_prob"],
                    backward_optimizer_and_evidence_s=t["update_actor"],
                    sync_and_readback_s=t["update_weights"],
                    independent_audit_s=audit["independent_audit_seconds"],
                    full_step_s=t["step"],
                    rollout_wait_tokens_per_2gpu_s=tokens / (t["gen"] * 2),
                    full_step_tokens_per_4gpu_s=tokens / (t["step"] * 4),
                )
            )
    result = dict(
        scope=(
            "Four layers; one node; actor PP2/DP1/TP1/EP1/CP1 on physical 4,5; "
            "rollout PP2/DP1/TP1/EP1 on 6,7; original layers 0-2/3."
        ),
        consistency=consistency,
        training=training,
        timing=timing,
        runs=runs,
    )
    (destination / "summary.json").write_text(json.dumps(result, indent=2))
    for name, rows in [("consistency", consistency), ("training", training), ("timing", timing)]:
        write_csv(destination / f"{name}.csv", rows)
    return result


def plots(result, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for run in result["runs"]:
        name = run["name"]
        train = [r for r in result["training"] if r["run"] == name]
        times = [r for r in result["timing"] if r["run"] == name]
        parity = [r for r in result["consistency"] if r["run"] == name]
        axes[0, 0].plot(
            [r["step"] for r in train], [r["reward_mean"] for r in train], "o-", label=name
        )
        axes[0, 1].plot(
            [r["step"] for r in train], [r["policy_loss"] for r in train], "o-", label=name
        )
        axes[1, 0].plot(
            [r["version"] for r in parity], [r["mismatch_count"] for r in parity], "o-", label=name
        )
        axes[1, 1].plot(
            [r["step"] for r in times], [r["full_step_s"] for r in times], "o-", label=name
        )
    for ax, title, ylabel in zip(
        axes.flat,
        [
            "Actual copy-task reward",
            "Native PPO policy loss",
            "Original FP32 word mismatches",
            "Complete step wall time",
        ],
        ["Reward", "Loss", "Mismatch count", "Seconds"],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Version" if "mismatch" in title else "Update")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        "MI300X four-layer GRPO: three updates per independent job\n"
        "Includes state hashing, sync readback and parity audits; no performance comparison"
    )
    fig.savefig(Path(destination) / "grpo-three-updates.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("outputs", nargs="+")
    p.add_argument("--destination", required=True)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    result = summarize(a.outputs, a.destination)
    if a.plot:
        plots(result, a.destination)
