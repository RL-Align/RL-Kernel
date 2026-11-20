# SPDX-License-Identifier: Apache-2.0
"""Three consecutive updates in the same real verl/vLLM worker processes."""

import json
import time

from examples.operators.dsv4_4l.cycle import command, save_json, wait_file
from examples.operators.dsv4_4l.measurement import emit
from examples.operators.dsv4_4l.verify_pipeline import verify_pipeline_step, verify_rollout_shards


async def run_continuous(
    root,
    server,
    process,
    cases,
    results,
    first_ack,
    *,
    score_fixed,
    generate_case,
    matrix,
    dp_size,
):
    original_pids = sorted((x["pipeline_stage"], x["pid"]) for x in first_ack)
    summaries = []
    for version in range(4):
        version_start = time.perf_counter()
        step_path = root / "steps" / f"v{version}"
        if version:
            name = f"step-v{version}"
            metadata = dict(expected_version=version - 1, step_version=version)
            if dp_size == 2:
                metadata["cases_by_dp"] = {"0": [cases[0]], "1": [cases[1]]}
            update = await command(process, name, "step", [cases[0]], **metadata)
            if update["weight_version"] != version or update["before"] == update["after"]:
                raise RuntimeError("actual consecutive actor parameter update failed")
            evidence = json.loads((step_path / "actor-distributed-step.json").read_text())
            transfers = verify_pipeline_step(evidence, dp_size=dp_size, version=version)
            save_json(step_path / "pipeline-transfer-verification.json", transfers)
            path = f"/outputs/steps/v{version}/actor-update.pt"
            ack = await server.collective_rpc("load_actor_update", args=(path,))
            save_json(step_path / "rollout-sync-ack.json", ack)
            manifest = json.loads((step_path / "actor-update-manifest.json").read_text())
            verified = verify_rollout_shards(ack, manifest, version=version)
            if sorted((x["pipeline_stage"], x["pid"]) for x in ack) != original_pids:
                raise RuntimeError("rollout workers restarted during consecutive synchronization")
            save_json(
                step_path / "rollout-shards-verified.json",
                dict(weight_version=version, actual_verified_tensors=sorted(verified)),
            )
            await wait_file(step_path / "actor-step-status.json", process)
            if not await matrix(version, cases):
                raise RuntimeError(f"fixed token matrix mismatched at v{version}")
        save_json(root / f"cases-v{version}-fixed.json", cases)
        new_cases = []
        for old in cases:
            new_cases.append(
                await generate_case(
                    old["id"] + "-new",
                    old["prompt_token_ids"],
                    len(old["response_token_ids"]),
                    version,
                )
            )
        save_json(root / f"cases-v{version}-generated.json", new_cases)
        if not await matrix(version, new_cases, prefix="generated-"):
            raise RuntimeError(f"newly generated token matrix mismatched at v{version}")
        after = await server.collective_rpc("set_case", args=(f"v{version}-verified",))
        save_json(root / f"rollout-v{version}-ack.json", after)
        if sorted((x["pipeline_stage"], x["pid"]) for x in after) != original_pids or any(
            x["weight_version"] != version for x in after
        ):
            raise RuntimeError("actual rollout PID/version changed")
        selected = [x for x in results if x["weight_version"] == version]
        if not selected or any(x["mismatch_count"] for x in selected):
            raise RuntimeError(f"incomplete/mismatched strict comparisons at v{version}")
        summaries.append(
            dict(
                version=version,
                accepted=True,
                actor_vs_rollout_compared_tokens=sum(
                    x["compared_tokens"]
                    for x in selected
                    if x["comparison_kind"] == "actor_vs_rollout"
                ),
                mismatch_count=sum(x["mismatch_count"] for x in selected),
                real_update_executed=version > 0,
                same_rollout_pids=True,
            )
        )
        emit(
            "full_acceptance_cycle",
            version=version,
            seconds=time.perf_counter() - version_start,
            diagnostic_overhead_included=True,
            includes="update/export/sync/fixed and generated matrices/compare/trace",
        )
        save_json(root / "continuous-progress.json", summaries)
        save_json(
            root / "results.json",
            dict(
                before_passed=True,
                after_passed=version == 3,
                pure_rlk_passed=version == 3,
                update_executed=version == 3,
                same_rollout_process=True,
                expanded_coverage_passed=version == 3,
                continuous_steps=3,
                completed_version=version,
                versions=summaries,
                rows=results,
                actor_parallelism=dict(TP=1, PP=2, EP=1, CP=1, DP=dp_size),
                rollout_parallelism=dict(TP=1, PP=2, EP=1, CP=1, DP=1),
                multi_gpu_verified=dp_size == 2 and version == 3,
                cross_replica_dp_reduction_executed=dp_size > 1,
                scope=f"actor PP2/DP{dp_size} + rollout PP2/DP1, three continuous real updates",
            ),
        )
    from examples.operators.dsv4_4l.verify_continuous import verify_continuous_run

    verified = verify_continuous_run(root, dp_size=dp_size)
    save_json(root / "continuous-acceptance.json", verified)
