# SPDX-License-Identifier: Apache-2.0
"""Fail closed on missing version/matrix/state/communication evidence."""

import json
from pathlib import Path

if __package__:
    from .optimizer_evidence import verify_optimizer_transition
    from .verify_pipeline import verify_pipeline_step, verify_rollout_shards, verify_worker_topology
    from .verify_result import verify_result
else:
    from optimizer_evidence import verify_optimizer_transition
    from verify_pipeline import verify_pipeline_step, verify_rollout_shards, verify_worker_topology
    from verify_result import verify_result

MATRIX = [
    ("A", ["A"]),
    ("B", ["B"]),
    ("C", ["C"]),
    ("AB", ["A", "B"]),
    ("BA", ["B", "A"]),
    ("A-repeat", ["A"]),
    ("B-repeat", ["B"]),
    ("C-repeat", ["C"]),
    ("AB-repeat", ["A", "B"]),
    ("BA-repeat", ["B", "A"]),
]


def check_matrix(rows, version, cases, *, prefix=""):
    by_id = {c["id"]: c for c in cases}
    suffix = "-new" if prefix else ""
    for label, ids in MATRIX:
        ids = [i + suffix for i in ids]
        name = f"v{version}-{prefix}{label}-matrix"
        group = [by_id[i] for i in ids]
        actual = [
            x
            for x in rows
            if x["comparison_kind"] == "actor_vs_rollout"
            and x["configuration"] == name
            and x["weight_version"] == version
        ]
        if (
            [x["case_id"] for x in actual] != ids
            or [x["batch_position"] for x in actual] != list(range(len(ids)))
            or any(
                x["batch_size"] != len(ids) or x["expected_tokens"] != len(c["response_token_ids"])
                for x, c in zip(actual, group, strict=True)
            )
        ):
            raise ValueError(f"missing/incorrect actual matrix cell: {name}")
        # Every repeat or batch-position cell requires independent per-engine byte invariance.
        if len(ids) > 1 or label.endswith("repeat"):
            for kind in ["actor_repeat_or_batch_position", "rollout_repeat_or_batch_position"]:
                for pos, c in enumerate(group):
                    matches = [
                        x
                        for x in rows
                        if x["comparison_kind"] == kind
                        and x["configuration"] == name + "-" + c["id"] + "-" + kind
                        and x["weight_version"] == version
                    ]
                    if (
                        len(matches) != 1
                        or matches[0]["batch_size"] != len(ids)
                        or matches[0]["batch_position"] != pos
                    ):
                        raise ValueError(
                            f"missing per-engine batch/repeat invariance: {name}/{kind}"
                        )


def verify_continuous_run(root, *, dp_size):
    root = Path(root)

    def read(name):
        return json.loads((root / name).read_text())

    result = read("results.json")
    checked_rows = verify_result(result)
    if result.get("continuous_steps") != 3 or result.get("completed_version") != 3:
        raise ValueError("three consecutive versions were not completed")
    if dp_size not in (1, 2):
        raise ValueError("unsupported DP topology")
    for key, dp in [("actor_parallelism", dp_size), ("rollout_parallelism", 1)]:
        if result.get(key) != dict(TP=1, PP=2, EP=1, CP=1, DP=dp):
            raise ValueError("incorrect selected continuous-run topology")
    if result.get("cross_replica_dp_reduction_executed") != (dp_size > 1):
        raise ValueError("DP1 must not claim cross-replica gradient reduction")
    if dp_size == 2 and result.get("multi_gpu_verified") is not True:
        raise ValueError("DP2 strict acceptance is incomplete")
    topology, start = read("actor-topology.json"), read("rollout-start-ack.json")
    buses = verify_worker_topology(topology, start, dp_size=dp_size)
    rollout_pids = sorted((x["pipeline_stage"], x["pid"]) for x in start)
    if any(x["weight_version"] != 0 for x in start):
        raise ValueError("rollout did not start at initial version")
    allocation_path = root / "device-allocation.json"
    if allocation_path.exists():
        allocation = json.loads(allocation_path.read_text())
        expected_buses = {r["pci_bus_id"] for r in allocation}
        if set(buses) != expected_buses:
            raise ValueError(
                "actual worker PCI identities differ from selected physical allocation"
            )
        for worker in topology:
            expected = next(
                r
                for r in allocation
                if r["role"] == "actor" and r["logical_gpu"] == worker["logical_gpu"]
            )
            if worker["device_identity"]["pci_bus_id"] != expected["pci_bus_id"]:
                raise ValueError("actual actor rank remapped to the wrong physical GPU")
    actor_pids = {x["rank"]: x["pid"] for x in topology}
    previous = {rank: read(f"actor-rank-{rank}/optimizer-initial.json") for rank in actor_pids}
    fixed = read("cases-v0.json")
    if [x["id"] for x in fixed] != ["A", "B", "C"]:
        raise ValueError("missing original A/B/C fixed cases")
    if len({len(c["prompt_token_ids"]) + len(c["response_token_ids"]) for c in fixed}) != 3:
        raise ValueError("A/B/C must have three different actual full-sequence lengths")
    if {x["weight_version"] for x in result["rows"]} != set(range(4)):
        raise ValueError("missing/unexpected raw-score version")
    table = []
    for version in range(4):
        cases = read(f"cases-v{version}-fixed.json")
        if cases != fixed:
            raise ValueError("fixed sequences changed between versions")
        generated = read(f"cases-v{version}-generated.json")
        if [x["id"] for x in generated] != ["A-new", "B-new", "C-new"]:
            raise ValueError("missing current-version generated cases")
        for original, new in zip(fixed, generated, strict=True):
            if original["prompt_token_ids"] != new["prompt_token_ids"] or len(
                original["response_token_ids"]
            ) != len(new["response_token_ids"]):
                raise ValueError("generated response boundaries do not match the intended case")
            name = f"generate-{new['id']}-v{version}"
            rows = [
                x
                for x in result["rows"]
                if x["weight_version"] == version
                and x["configuration"] == name
                and x["comparison_kind"] == "actor_vs_rollout"
            ]
            if len(rows) != 1 or rows[0]["expected_tokens"] != len(new["response_token_ids"]):
                raise ValueError("missing actual generated raw-score comparison")
        for prefix, group in [("", fixed), ("generated-", generated)]:
            check_matrix(result["rows"], version, group, prefix=prefix)
            by_id = {x["id"]: x for x in group}
            for label, ids in MATRIX:
                if prefix:
                    ids = [i + "-new" for i in ids]
                name = f"v{version}-{prefix}{label}-matrix"
                actor_ack = read(f"actor-{name}.json")
                if actor_ack["weight_version"] != version or actor_ack["pid"] != actor_pids[0]:
                    raise ValueError("actor version/PID differs from continuous run")
                metadata = read(f"rollout-batches-{name}.json")
                expected_inputs = [
                    by_id[i]["prompt_token_ids"] + by_id[i]["response_token_ids"] for i in ids
                ]
                if (
                    len(metadata) != 2
                    or any(x["weight_version"] != version or x["case_id"] != name for x in metadata)
                    or not any(
                        b["num_requests"] == len(ids) and b["input_token_rows"] == expected_inputs
                        for x in metadata
                        for b in x["batches"]
                    )
                ):
                    raise ValueError("actual rollout batch/position evidence is missing")
        acks = read(f"rollout-v{version}-ack.json")
        if sorted((x["pipeline_stage"], x["pid"]) for x in acks) != rollout_pids or any(
            x["weight_version"] != version for x in acks
        ):
            raise ValueError("rollout workers/version changed during continuous run")
        synced = 0
        if version:
            directory = f"steps/v{version}"
            records = read(directory + "/actor-distributed-step.json")
            transfers = verify_pipeline_step(records, dp_size=dp_size, version=version)
            manifest = read(directory + "/actor-update-manifest.json")
            owned = {}
            for record in records:
                rank = record["rank"]
                if (
                    record["pid"] != actor_pids[rank]
                    or record["previous_weight_version"] != version - 1
                ):
                    raise ValueError("actor rank restarted or skipped weight version")
                opt = record["optimizer_evidence"]
                verify_optimizer_transition(
                    opt["before"], opt["after"], version=version, previous=previous[rank]
                )
                previous[rank] = opt["after"]
                gradients = read(record["gradient_evidence"])
                expected = set(gradients["actual_trainable_parameters"])
                if (
                    not expected
                    or set(gradients["local_gradients"]) != expected
                    or set(gradients["reduced_gradients"]) != expected
                    or not all(x["finite"] for x in gradients["local_gradients"].values())
                    or not all(x["finite"] for x in gradients["reduced_gradients"].values())
                    or not any(x["nonzero"] for x in gradients["reduced_gradients"].values())
                    or not record["changed_parameters"]
                ):
                    raise ValueError("actual backward/parameter-change coverage failed")
                if record["cross_replica_reduction_executed"] != (dp_size > 1):
                    raise ValueError("incorrect cross-replica reduction claim")
                changes = read(record["state_change_evidence"])
                if not changes["changed_parameters"] or not changes["synchronization_inventory"]:
                    raise ValueError("actual parameter/state inventory missing")
                if record["pp_rank"] == 1 and not changes["router_bias_exports"]:
                    raise ValueError("router bias absent from synchronization inventory")
                # DP copies were already byte-checked by native export; one owner per PP stage.
                if record["dp_rank"] == 0:
                    for key, value in record["local_sync_state"].items():
                        if key in owned:
                            raise ValueError("synchronizable tensor has duplicate PP ownership")
                        owned[key] = {k: value[k] for k in ["shape", "dtype", "sha256"]}
            if owned != manifest:
                raise ValueError(
                    "export manifest does not equal complete actual stage state inventory"
                )
            sync = read(directory + "/rollout-sync-ack.json")
            if sorted((x["pipeline_stage"], x["pid"]) for x in sync) != rollout_pids or any(
                x.get("cache_generation_after") != x.get("cache_generation_before", -2) + 1
                or x.get("prefix_caching_enabled") is not False
                or x.get("prior_weight_version") != version - 1
                or x.get("input_update_path") != f"/outputs/{directory}/actor-update.pt"
                for x in sync
            ):
                raise ValueError("stale/replaced rollout worker update acknowledgement")
            synced = len(verify_rollout_shards(sync, manifest, version=version))
            status = read(directory + "/actor-step-status.json")
            if (
                status["weight_version"] != version
                or status["optimizer_returned"] is not True
                or status["parameter_changed"] is not True
                or status["all_stage_updates_verified"] is not True
            ):
                raise ValueError("actual optimizer step did not complete on all PP stages")
            if len(transfers) < 2 * dp_size:
                raise ValueError("missing real PP forward/backward transfers")
        rows = [x for x in result["rows"] if x["weight_version"] == version]
        selected = [x for x in rows if x["comparison_kind"] == "actor_vs_rollout"]
        table.append(
            dict(
                version=version,
                expected_tokens=sum(x["expected_tokens"] for x in selected),
                compared_tokens=sum(x["compared_tokens"] for x in selected),
                mismatch_count=sum(x["mismatch_count"] for x in selected),
                batch_invariance=True,
                original_fp32=True,
                finite=True,
                real_update=version > 0,
                synchronized_tensors=synced,
                same_actor_and_rollout_pids=True,
                optimizer_state_continuous=version > 0,
            )
        )
    return dict(
        scope=f"actor PP2/DP{dp_size} + rollout PP2/DP1",
        continuous_updates=3,
        accepted=True,
        comparison_rows=checked_rows,
        actual_device_pci_ids=buses,
        cross_replica_dp_reduction_executed=dp_size > 1,
        versions=table,
    )
