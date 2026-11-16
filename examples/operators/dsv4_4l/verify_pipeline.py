# SPDX-License-Identifier: Apache-2.0
"""Reject missing communication, reduction or per-shard weight synchronization."""


def verify_pipeline_step(records, *, dp_size, residual_width=16384):
    if len(records) != 2 * dp_size or {r["rank"] for r in records} != set(range(2 * dp_size)):
        raise ValueError("missing actual actor ranks")
    if {r["weight_version"] for r in records} != {1}:
        raise ValueError("actor ranks do not share the updated version")
    for r in records:
        if (
            not r["dp_gradient_reduction_verified"]
            or r["dp_replicas"] != dp_size
            or (dp_size > 1 and not r["different_local_gradients"])
            or r["before"] == r["after"]
        ):
            raise ValueError("actual per-stage update/DDP reduction evidence failed")
        witness = r["gradient_witness"]
        if not all(
            witness.get(k) is True
            for k in ["backward_executed", "gradient_finite", "gradient_nonzero"]
        ):
            raise ValueError("actual backward witness failed")
    transfers = []
    for dp in range(dp_size):
        first = next(r for r in records if r["dp_rank"] == dp and r["pp_rank"] == 0)
        last = next(r for r in records if r["dp_rank"] == dp and r["pp_rank"] == 1)
        for src, dst, send, recv in [
            (first, last, "forward_send", "forward_recv"),
            (last, first, "backward_send", "backward_recv"),
        ]:
            sent = {
                r["microbatch"]: r for r in src["pipeline_boundaries"] if r["direction"] == send
            }
            received = {
                r["microbatch"]: r for r in dst["pipeline_boundaries"] if r["direction"] == recv
            }
            if not sent or set(sent) != set(received):
                raise ValueError(f"missing native pipeline communication: {send}/{recv}")
            for i, a in sent.items():
                b = received[i]
                if (
                    not a["finite"]
                    or not b["finite"]
                    or a["dtype"] != "torch.bfloat16"
                    or len(a["shape"]) != 3
                    or a["shape"][-1] != residual_width
                    or any(a[k] != b[k] for k in ["shape", "dtype", "sha256"])
                ):
                    raise ValueError(f"native pipeline bytes differ: {send}/{recv}")
                transfers.append(
                    dict(
                        dp_rank=dp,
                        direction=send,
                        microbatch=i,
                        shape=a["shape"],
                        dtype=a["dtype"],
                        sha256=a["sha256"],
                    )
                )
    return transfers


def verify_rollout_shards(acks, manifest, *, version=1):
    if len(acks) != 2 or {r["pipeline_stage"] for r in acks} != {0, 1}:
        raise ValueError("missing actual rollout PP workers")
    checked = {}
    for r in acks:
        if (
            r["weight_version"] != version
            or not r["parameter_load_completed"]
            or not r["weight_changed"]
        ):
            raise ValueError("rollout stage did not load actual updated state")
        expected_layers = [0, 1, 2] if r["pipeline_stage"] == 0 else [3]
        if r["original_layer_ids"] != expected_layers:
            raise ValueError("actual rollout layer partition differs from selected PP2 layout")
        for name, value in r["verified_original_tensors"].items():
            if (
                name in checked
                or name not in manifest
                or value["sha256"] != manifest[name]["sha256"]
            ):
                raise ValueError(f"rollout shard ownership/content mismatch: {name}")
            checked[name] = value
    if set(checked) != set(manifest):
        raise ValueError("updated actor tensors missing from rollout shards")
    return checked


def verify_worker_topology(actor, rollout, *, dp_size):
    """Require real stage ownership, native process groups and distinct PCI devices."""
    if len(actor) != 2 * dp_size or {r["rank"] for r in actor} != set(range(2 * dp_size)):
        raise ValueError("missing actual actor topology")
    if {(r["pp_rank"], r["dp_rank"]) for r in actor} != {
        (pp, dp) for pp in range(2) for dp in range(dp_size)
    }:
        raise ValueError("actor topology does not contain the requested PP/DP mesh")
    for r in actor:
        pp_members = {x["rank"] for x in actor if x["dp_rank"] == r["dp_rank"]}
        dp_members = {x["rank"] for x in actor if x["pp_rank"] == r["pp_rank"]}
        if set(r["pp_ranks"]) != pp_members or set(r["dp_ranks"]) != dp_members:
            raise ValueError("actual native PP/DP groups differ from the selected mesh")
        if r["original_layer_ids"] != ([0, 1, 2] if r["pp_rank"] == 0 else [3]):
            raise ValueError("actor does not own the requested original checkpoint layers")
    if len(rollout) != 2 or {r["pipeline_stage"] for r in rollout} != {0, 1}:
        raise ValueError("missing actual rollout topology")
    for r in rollout:
        if r["original_layer_ids"] != ([0, 1, 2] if r["pipeline_stage"] == 0 else [3]):
            raise ValueError("rollout does not own the requested original checkpoint layers")
    identities = [r["device_identity"] for r in actor + rollout]
    buses = [r["pci_bus_id"] for r in identities]
    if any(not bus for bus in buses) or len(set(buses)) != len(buses):
        raise ValueError("actual actor and rollout GPU PCI identities are not disjoint")
    if any(r["architecture"].split(":")[0] != "gfx942" for r in identities):
        raise ValueError("multi-GPU acceptance requires actual gfx942 devices")
    return buses


def verify_pipeline_run(root, *, dp_size):
    import json
    from pathlib import Path

    if __package__:
        from .verify_result import verify_result
    else:
        from verify_result import verify_result

    root = Path(root)

    def read(name):
        return json.loads((root / name).read_text())

    result = read("results.json")
    count = verify_result(result)
    for key, dp in [("actor_parallelism", dp_size), ("rollout_parallelism", 1)]:
        if result.get(key) != dict(TP=1, PP=2, EP=1, CP=1, DP=dp):
            raise ValueError(f"missing actual selected parallel configuration: {key}")
    if dp_size == 2 and result.get("multi_gpu_verified") is not True:
        raise ValueError("final PP2/DP2 model-parallel acceptance is incomplete")
    # Every matrix cell must exist; a nonempty subset cannot pass final acceptance.
    for version in (0, 1):
        for label, ids in [
            ("A", ["A"]),
            ("B", ["B"]),
            ("C", ["C"]),
            ("AB", ["A", "B"]),
            ("BA", ["B", "A"]),
            ("A-repeat", ["A"]),
            ("AB-repeat", ["A", "B"]),
        ]:
            rows = [
                r
                for r in result["rows"]
                if r["comparison_kind"] == "actor_vs_rollout"
                and r["configuration"] == f"v{version}-{label}-matrix"
                and r["weight_version"] == version
            ]
            if [r["case_id"] for r in rows] != ids:
                raise ValueError(f"missing actual batch/order matrix cell: v{version}-{label}")
    generated = read("cases-v1-generated.json")
    if {r["id"] for r in generated} != {"A-new", "B-new", "C-new"}:
        raise ValueError("missing newly generated updated sequences")
    for case in generated:
        rows = [
            r
            for r in result["rows"]
            if r["case_id"] == case["id"]
            and r["weight_version"] == 1
            and r["comparison_kind"] == "actor_vs_rollout"
        ]
        if len(rows) != 1 or rows[0]["expected_tokens"] != len(case["response_token_ids"]):
            raise ValueError("missing updated generated token comparison")
    start, end, sync = (read(f"rollout-{part}-ack.json") for part in ["start", "after", "sync"])
    for acks, version in [(start, 0), (end, 1), (sync, 1)]:
        if len(acks) != 2 or {r["pipeline_stage"] for r in acks} != {0, 1}:
            raise ValueError("missing live rollout stage acknowledgement")
        if any(r["weight_version"] != version for r in acks):
            raise ValueError("live rollout stage has the wrong weight version")
    if (
        len(
            {
                tuple(sorted((r["pipeline_stage"], r["pid"]) for r in acks))
                for acks in [start, end, sync]
            }
        )
        != 1
    ):
        raise ValueError("rollout processes were replaced during synchronization")
    buses = verify_worker_topology(read("actor-topology.json"), start, dp_size=dp_size)
    transfers = verify_pipeline_step(read("actor-distributed-step.json"), dp_size=dp_size)
    tensors = verify_rollout_shards(sync, read("actor-update-manifest.json"))
    return dict(
        comparison_rows=count,
        actual_device_pci_ids=buses,
        checked_pipeline_transfers=len(transfers),
        checked_updated_tensors=len(tensors),
        actor_dp=dp_size,
        actor_pp=2,
        rollout_pp=2,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--dp-size", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    print(
        "PIPELINE_CYCLE_ACCEPTANCE_PASS",
        json.dumps(verify_pipeline_run(args.output, dp_size=args.dp_size)),
    )
