# SPDX-License-Identifier: Apache-2.0
"""Aggregate actual native/RLK jobs; BLOCKED is never zero mismatches/throughput."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

KINDS = ("actor_vs_rollout", "actor_repeat_or_batch_position", "rollout_repeat_or_batch_position")


def read(path):
    return json.loads(Path(path).read_text())


def summarize_scores(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["comparison_kind"] not in KINDS or row["weight_version"] not in range(4):
            raise ValueError("unexpected comparison/version")
        n = row["expected_tokens"]
        if not isinstance(n, int) or n <= 0 or row["compared_tokens"] != n:
            raise ValueError("empty/incomplete comparison")
        if row["dtype"] != "torch.float32" or row["all_finite"] is not True:
            raise ValueError("cast or nonfinite comparison")
        for k in ["mismatch_rate", "max_abs_diff", "mean_abs_diff"]:
            if not math.isfinite(row[k]) or row[k] < 0:
                raise ValueError("invalid diagnostic metric")
        if row["mismatch_count"] not in range(n + 1):
            raise ValueError("invalid mismatch count")
        if row["mismatch_rate"] != row["mismatch_count"] / n:
            raise ValueError("mismatch rate does not match original byte count")
        if row["mismatch_count"] and not row.get("first_mismatch"):
            raise ValueError("missing original first mismatch evidence")
        grouped[(row["weight_version"], row["comparison_kind"])].append(row)
    if set(grouped) != {(v, k) for v in range(4) for k in KINDS}:
        raise ValueError("missing required version/comparison type")
    result = []
    for (version, kind), values in sorted(grouped.items()):
        count = sum(x["compared_tokens"] for x in values)
        mismatch = sum(x["mismatch_count"] for x in values)
        result.append(
            dict(
                version=version,
                comparison_kind=kind,
                expected_tokens=sum(x["expected_tokens"] for x in values),
                compared_tokens=count,
                mismatch_count=mismatch,
                mismatch_rate=mismatch / count,
                max_abs_diff=max(x["max_abs_diff"] for x in values),
                mean_abs_diff=sum(x["mean_abs_diff"] * x["compared_tokens"] for x in values)
                / count,
                all_finite=True,
                batch_invariance=all(
                    x["mismatch_count"] == 0
                    for x in rows
                    if x["weight_version"] == version and x["comparison_kind"] != "actor_vs_rollout"
                ),
                first_mismatch=next(
                    (x["first_mismatch"] for x in values if x.get("first_mismatch")), None
                ),
            )
        )
    return result


def version_of(event):
    if "version" in event:
        return event["version"]
    text = str(event.get("command", "")) + str(event.get("configuration", ""))
    text += str(event.get("request_id", "")) + str(event.get("path", ""))
    match = re.search(r"(?:^v|[-/]v)([0-3])(?:[-/]|$)", text)
    return int(match[1]) if match else None


def performance(root, job):
    events = [
        json.loads(line)
        for p in sorted(root.glob("timings-*.jsonl"))
        for line in p.read_text().splitlines()
    ]
    if not events:
        raise ValueError("missing actual timing observations")
    for event in events:
        if "seconds" in event and (not math.isfinite(event["seconds"]) or event["seconds"] < 0):
            raise ValueError("nonfinite/negative timing")
        if event.get("success") is False:
            raise ValueError("failed measured worker stage")
    rows = []
    for v in range(4):
        subset = [x for x in events if version_of(x) == v]

        def phase(name, _subset=subset):
            return [x for x in _subset if x["phase"] == name]

        def total(name, _phase=phase):
            return sum(x["seconds"] for x in _phase(name))

        actor_by_rank = defaultdict(list)
        for e in phase("actor_logprob"):
            actor_by_rank[e["rank"]].append(e)
        if set(actor_by_rank) != {"0", "1"}:
            raise ValueError("missing real PP-rank actor timers")
        actor_time = max(sum(e["seconds"] for e in es) for es in actor_by_rank.values())
        actor_inputs = sum(e["input_tokens"] for e in actor_by_rank["1"])
        actor_padding = sum(e["padded_input_tokens"] for e in actor_by_rank["1"])
        generates = phase("rollout_generate")
        workloads = phase("generation_workload")
        if not generates or len(generates) != len(workloads):
            raise ValueError("incomplete real generated token workload/timers")
        response = sum(e["response_tokens"] for e in workloads)
        prompts = sum(e["prompt_tokens"] for e in workloads)
        generation_s = total("rollout_generate")
        train = phase("forward_backward_optimizer")
        opt = phase("optimizer_step")
        if v and (len(train) != 2 or len(opt) != 2 or len(phase("weight_sync")) != 1):
            raise ValueError("missing native update/optimizer/sync stage timers")
        full = phase("full_acceptance_cycle")
        if len(full) != 1:
            raise ValueError("missing full diagnostic acceptance cycle")
        rows.append(
            dict(
                version=v,
                rollout_generate_seconds=generation_s,
                generated_prompt_tokens=prompts,
                generated_response_tokens=response,
                rollout_response_tokens_per_second=response / generation_s,
                rollout_response_tokens_per_gpu_second=response / (generation_s * 2),
                rollout_teacher_seconds=total("rollout_teacher"),
                actor_logprob_pp_max_rank_seconds=actor_time,
                actor_logprob_response_scored_tokens=sum(
                    e["response_tokens"] for e in actor_by_rank["1"]
                ),
                rollout_teacher_input_tokens=sum(
                    e["input_tokens"] for e in phase("rollout_teacher")
                ),
                teacher_aux_emitted_tokens=sum(e["samples"] for e in phase("rollout_teacher")),
                actor_logprob_input_tokens=actor_inputs,
                actor_logprob_padded_input_tokens=actor_padding,
                actor_logprob_input_tokens_per_second=actor_inputs / actor_time,
                actor_logprob_input_tokens_per_gpu_second=actor_inputs / (actor_time * 2),
                forward_backward_optimizer_pp_max_rank_seconds=max(
                    (e["seconds"] for e in train), default=None
                ),
                optimizer_pp_max_rank_seconds=max((e["seconds"] for e in opt), default=None),
                train_input_tokens=max((e["input_tokens"] for e in train), default=None),
                train_response_loss_tokens=max((e["response_tokens"] for e in train), default=None),
                weight_sync_seconds=total("weight_sync") if v else None,
                full_diagnostic_acceptance_cycle_seconds=full[0]["seconds"],
                comparison_tool_seconds=total("comparison_tool"),
                timing_scope="diagnostic acceptance, not GRPO/PPO end-to-end throughput",
            )
        )
    samples = [json.loads(line) for line in Path(job["memory_samples"]).read_text().splitlines()]
    if not samples:
        raise ValueError("missing actual VRAM samples")
    peaks = {
        g: max(int(e["devices"][f"card{g}"]["VRAM Total Used Memory (B)"]) for e in samples)
        for g in (4, 5, 6, 7)
    }
    init = [
        e
        for e in events
        if e["phase"] in ("actor_init_load_compile_warmup", "rollout_init_load_compile_warmup")
    ]
    if len(init) != 3:
        raise ValueError("missing initialization timers for actor PP0/1 and rollout driver")
    return dict(
        versions=rows,
        initialization=init,
        events=events,
        sampled_peak_vram_bytes=peaks,
        memory_sample_count=len(samples),
        memory_sampling=(
            "roughly 1-second polling; observed resident VRAM peak, " "not hardware-exact peak"
        ),
        allocator_peak_events=[x for x in events if "peak_allocated_bytes" in x],
        job_elapsed_seconds=job["elapsed_seconds"],
        trace_overhead_separately_measurable=False,
        timing_limit=(
            "Module captures, hashing, state audits, IPC and I/O overlap compute; "
            "no isolated trace-free performance or speedup claim"
        ),
    )


def aggregate(native_root, rlk_root, native_job, rlk_job, output):
    native_root, rlk_root, output = map(Path, (native_root, rlk_root, output))
    output.mkdir(parents=True, exist_ok=True)
    pjob, rjob = read(native_job), read(rlk_job)
    if (
        pjob["image"]["id"] != rjob["image"]["id"]
        or pjob["fixed_cases_sha256"] != rjob["fixed_cases_sha256"]
    ):
        raise ValueError("unmatched image or predeclared fixed inputs")
    if (native_root / "predeclared-cases.json").read_bytes() != (
        rlk_root / "predeclared-cases.json"
    ).read_bytes():
        raise ValueError("fixed input bytes differ")
    if (
        hashlib.sha256((rlk_root / "predeclared-cases.json").read_bytes()).hexdigest()
        != rjob["fixed_cases_sha256"]
    ):
        raise ValueError("predeclared input SHA mismatch")
    if read(rlk_root / "cases-v0-fixed.json") != read(rlk_root / "predeclared-cases.json"):
        raise ValueError("actual fixed scoring inputs differ from predeclared samples")
    checkpoint_hashes = []
    for root in (native_root, rlk_root):
        config = read(root / "container-inspect.json")["Config"]
        env = dict(x.split("=", 1) for x in config["Env"] if "=" in x)
        checkpoint_hashes.append(env["DSV4_CHECKPOINT_SHA256"])
    if checkpoint_hashes[0] != checkpoint_hashes[1]:
        raise ValueError("actual container checkpoint SHA differs")
    pstatus = read(native_root / "native-status.json")
    if pstatus["status"] != "BLOCKED" or pstatus["completed_steps"] != 0:
        raise ValueError(
            "native arm requires completed native training aggregation, not startup-only data"
        )
    errors = [read(p) for p in sorted(native_root.glob("native-*-error.json"))]
    if not errors or any(e["rlk_modules_loaded"] for e in errors):
        raise ValueError("native probe lacks pristine per-process failure evidence")
    if __package__:
        from .verify_continuous import verify_continuous_run
    else:
        from verify_continuous import verify_continuous_run
    accepted = verify_continuous_run(rlk_root, dp_size=1)
    assert accepted["accepted"]
    state_audit = read(rlk_root / "actual-update-and-dtype-evidence.json")
    if state_audit["accepted"] is not True:
        raise ValueError("saved state and checkpoint dtype audit failed")
    result = read(rlk_root / "results.json")
    summary = summarize_scores(result["rows"])
    if any(x["mismatch_count"] for x in summary):
        raise ValueError("R/R is not bitwise consistent")
    for x in summary:
        x.update(
            group="R/R",
            status="PASSED",
            synchronization="initial checkpoint" if x["version"] == 0 else "verified",
        )
    missing = [
        dict(
            group="P/P",
            version=v,
            comparison_kind=k,
            status="BLOCKED",
            expected_tokens=None,
            compared_tokens=None,
            mismatch_count=None,
            mismatch_rate=None,
            max_abs_diff=None,
            mean_abs_diff=None,
            all_finite=None,
            batch_invariance=None,
            synchronization="not reached",
            first_mismatch=None,
        )
        for v in range(4)
        for k in KINDS
    ]
    perf = performance(rlk_root, rjob)
    provenance = dict(
        native=read(native_root / "native-provenance.json"),
        rlk=read(rlk_root / "runtime-versions.json"),
        native_plugin_policy=read(native_root / "native-plugin-policy.json"),
    )
    if provenance["native"]["versions"] != provenance["rlk"]:
        raise ValueError("actual source/dependency provenance differs")
    hashes = {}
    for root in (native_root, rlk_root):
        for name in [
            "results.json",
            "native-status.json",
            "runtime-versions.json",
            "native-provenance.json",
            "predeclared-cases.json",
            "continuous-acceptance.json",
        ]:
            p = root / name
            if p.exists():
                hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    data = dict(
        groups={
            "P/P": dict(
                status="BLOCKED", completed_steps=0, output=str(native_root), errors=errors
            ),
            "R/R": dict(status="PASSED", completed_steps=3, output=str(rlk_root)),
        },
        consistency=missing + summary,
        performance={"P/P": None, "R/R": perf},
        provenance=provenance,
        actual_state_and_dtype_audit=state_audit,
        actual_container_checkpoint_sha256=checkpoint_hashes[0],
        jobs={"P/P": pjob, "R/R": rjob},
        input_file_sha256=hashes,
        reward="not measured",
        KL="not measured",
        task_benefit="not measured",
        full_grpo_ppo_reward_advantage_pipeline=False,
        inference=(
            "One attempted native and one RLK job, three updates; "
            "no bootstrap intervals or stable performance conclusions"
        ),
    )
    (output / "summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    fields = [
        "group",
        "version",
        "comparison_kind",
        "status",
        "expected_tokens",
        "compared_tokens",
        "mismatch_count",
        "mismatch_rate",
        "max_abs_diff",
        "mean_abs_diff",
        "all_finite",
        "batch_invariance",
        "synchronization",
    ]
    with (output / "consistency.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["consistency"])
    with (output / "performance.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(perf["versions"][0]))
        writer.writeheader()
        writer.writerows(perf["versions"])
    (output / "raw-log-index.json").write_text(
        json.dumps(
            dict(
                native=str(native_root),
                rlk=str(rlk_root),
                native_job=str(native_job),
                rlk_job=str(rlk_job),
            ),
            indent=2,
        )
        + "\n"
    )
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["native-job", "rlk-job", "output-dir"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--native-output")
    p.add_argument("--rlk-output")
    a = p.parse_args()
    aggregate(
        a.native_output or read(a.native_job)["output"],
        a.rlk_output or read(a.rlk_job)["output"],
        a.native_job,
        a.rlk_job,
        a.output_dir,
    )


if __name__ == "__main__":
    main()
