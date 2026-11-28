# SPDX-License-Identifier: Apache-2.0
"""Fail-closed audit of the real verl GRPO recipe, never a diagnostic-loss run."""

import argparse
import json
import os
from pathlib import Path

import torch


def _split(name):
    """Original layers per pipeline stage, e.g. "3,1" (the four-layer default)."""
    value = os.environ.get(name) or os.environ.get("RLK_DSV4_PP_SPLIT", "3,1")
    return [int(x) for x in value.split(",")]


def actor_split():
    return _split("RLK_DSV4_PP_SPLIT")


def rollout_split():
    """The rollout may use its own split (RLK_ROLLOUT_PP_SPLIT); default: the actor's."""
    return _split("RLK_ROLLOUT_PP_SPLIT")


def scope_text(total_steps):
    actor, rollout = actor_split(), rollout_split()
    if (actor, rollout, total_steps) == ([3, 1], [3, 1], 3):
        return (
            "four-GPU, four-layer actor PP2/DP1 and rollout PP2/DP1; "
            "three continuous native verl GRPO/PPO updates."
        )
    return (
        f"{len(actor) + len(rollout)}-GPU, {sum(actor)}-layer actor PP{len(actor)}/DP1 "
        f"(layers {actor}) and rollout PP{len(rollout)}/DP1 (layers {rollout}); "
        f"{total_steps} continuous native verl GRPO/PPO updates."
    )


def worker_key(record):
    return record.get("worker_identity", record["pid"])


def verify_stress(root, repeats, pids):
    from verl.experimental.rl_kernel.trainer_contract import strict_response_parity

    summary = json.loads((root / "stress-v3.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["weight_version"] == 3
    assert summary["repeats"] == summary["completed_repeats"] == repeats > 0
    assert summary["additional_optimizer_updates"] == 0
    assert summary["actor_state_unchanged"] and summary["rollout_state_unchanged"]
    assert summary["finite"]
    assert summary["mismatch_count"] == summary["mean_abs_diff"] == summary["max_abs_diff"] == 0
    audit = json.loads((root / "post-update-v3.json").read_text())
    groups = dict(
        fixed=json.loads((root / "fixed-sequences.json").read_text()),
        fresh=audit["fresh_sequences"],
    )
    orders = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("A",), ("B",), ("C",)]
    output = root / "stress-v3"
    records = [json.loads(s) for s in (output / "comparisons.jsonl").read_text().splitlines()]
    assert len(records) == repeats
    count = 0
    for iteration, record in enumerate(records):
        assert record["iteration"] == iteration and record["file"] == f"{iteration:05d}.pt"
        raw = torch.load(output / record["file"], map_location="cpu", weights_only=True)
        group = ("fixed", "fresh")[(iteration // len(orders)) % 2]
        order = orders[iteration % len(orders)]
        selected = [{c["id"]: c for c in groups[group]}[name] for name in order]
        assert raw["iteration"] == iteration and raw["weight_version"] == 3
        assert raw["group"] == group and tuple(raw["order"]) == order
        assert raw["cases"] == selected
        assert len(raw["actor"]) == len(raw["rollout"]) == len(record["rows"]) == len(selected)
        for i, (case, left, right, row) in enumerate(
            zip(selected, raw["actor"], raw["rollout"], record["rows"], strict=True)
        ):
            index = "ABC".index(case["id"])
            name = case["id"]
            baseline = f"audit-v3-{group}-{index:02d}-{name}-{name}.pt"
            assert raw["baseline_files"][i] == baseline
            original = torch.load(root / baseline, map_location="cpu", weights_only=True)
            assert original["weight_version"] == 3 and original["case"] == case
            assert torch.equal(left.view(torch.int32), original["actor"].view(torch.int32))
            checked = strict_response_parity(left, right, torch.ones_like(left))
            assert row == checked and row["expected_tokens"] == len(case["response_ids"])
            count += row["compared_tokens"]
        assert len(raw["stages"]) == len(rollout_split())
        for stage in raw["stages"]:
            assert (
                stage["weight_version"] == 3 and worker_key(stage) == pids[stage["pipeline_stage"]]
            )
            assert len(stage["batches"]) == 1
            batch = stage["batches"][0]
            assert batch["num_requests"] == len(selected)
            starts = batch["query_start_loc"]
            assert [b - a for a, b in zip(starts[:-1], starts[1:], strict=True)] == [
                len(c["prompt_ids"]) + len(c["response_ids"]) for c in selected
            ]
            if batch["input_token_rows"] is not None:
                assert batch["input_token_rows"] == [
                    c["prompt_ids"] + c["response_ids"] for c in selected
                ]
            if len(order) == 2:
                assert stage["order"]
    assert count == summary["expected_tokens"] == summary["compared_tokens"] > 0
    expected = json.loads((root / "sync-v3-raw.json").read_text())["manifest"]
    actors = json.loads((root / "stress-v3-actor-readback.json").read_text())
    assert len(actors) == len(actor_split()) and all(
        actor["weight_version"] == 3 and actor["manifest"] == expected for actor in actors
    )
    acks = json.loads((root / "stress-v3-rollout-readback.json").read_text())
    assert len(acks) == len(rollout_split())
    covered = set()
    for ack in acks:
        assert ack["weight_version"] == 3 and worker_key(ack) == pids[ack["pipeline_stage"]]
        assert ack["readback_only"]
        current = ack["verified_original_tensors"]
        assert not covered.intersection(current)
        assert all(
            item[key] == expected[name][key]
            for name, item in current.items()
            for key in ("sha256", "dtype", "shape")
        )
        covered.update(current)
    assert covered == set(expected)
    return summary


def verify(root, inventory):
    from verl.experimental.rl_kernel.trainer_contract import (
        strict_response_parity,
        validate_stage_acks,
    )
    from examples.operators.dsv4_4l.optimizer_evidence import verify_optimizer_transition

    if not __debug__:
        raise RuntimeError("run this strict evidence verifier without -O")
    original = json.loads(Path(inventory).read_text())
    original_state = original["sync_eligible_original_tensors"]
    root = Path(root)
    job = root.parent
    if json.loads((job / "exit-code.json").read_text())["exit_code"] != 0:
        raise ValueError("training process did not exit successfully")
    fixed = json.loads((root / "fixed-sequences.json").read_text())
    if len(fixed) != 3 or {c["id"] for c in fixed} != set("ABC"):
        raise ValueError("all three fixed cases required")
    versions = []
    pids = None
    previous_actor = {}
    identities = {
        i: json.loads((root / f"actor-rank-{i}-identity.json").read_text())
        for i in range(len(actor_split()))
    }
    orders = [
        ("A",),
        ("B",),
        ("C",),
        ("A",),
        ("B",),
        ("C",),
        ("A", "B"),
        ("B", "A"),
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("A", "C"),
        ("C", "A"),
    ]
    total_steps = int(os.environ.get("RLK_TOTAL_STEPS", "3"))
    for version in range(total_steps + 1):
        sync = json.loads((root / f"sync-v{version}.json").read_text())
        rawsync = json.loads((root / f"sync-v{version}-raw.json").read_text())
        assert sync["weight_version"] == rawsync["stages"][0]["weight_version"] == version
        pids = validate_stage_acks(sync["stages"], rawsync["manifest"], version, pids)
        split = rollout_split()
        assert {s["pipeline_stage"]: s["original_layer_ids"] for s in sync["stages"]} == {
            i: list(range(sum(split[:i]), sum(split[: i + 1]))) for i in range(len(split))
        }
        router_bias = [key for key in rawsync["manifest"] if key.endswith(".ffn.gate.bias")]
        assert set(router_bias) == {key for key in original_state if key.endswith(".ffn.gate.bias")}
        assert set(rawsync["manifest"]) == set(original_state)
        for key, item in original_state.items():
            item = item.get("runtime_abi", item)
            assert all(
                rawsync["manifest"][key][field] == item[field] for field in ("shape", "dtype")
            )
            if version == 0:
                assert rawsync["manifest"][key]["sha256"] == item["sha256"]
        path = root / ("initial-v0.json" if version == 0 else f"post-update-v{version}.json")
        audit = json.loads(path.read_text())
        assert audit["weight_version"] == version and audit["global_steps"] == version
        assert audit["batch_invariance"] and audit["finite"]
        cases_by_group = {
            "fixed": {c["id"]: c for c in fixed},
            "fresh": {c["id"]: c for c in audit["fresh_sequences"]},
        }
        expected_labels = set()
        for group in cases_by_group:
            seen = set()
            for ordinal, order in enumerate(orders):
                label = f"{group}-{ordinal:02d}-" + "".join(order)
                for name in order:
                    expected_labels.add(label + "-" + name)
                    if name in seen:
                        expected_labels.update(
                            (
                                label + "-" + name + "-actor-invariance",
                                label + "-" + name + "-rollout-invariance",
                            )
                        )
                    else:
                        seen.add(name)
                    if group == "fresh" and ordinal < 3:
                        expected_labels.add(label + "-" + name + "-decode")
        assert {r["label"] for r in audit["rows"]} == expected_labels
        assert len(audit["rows"]) == len(expected_labels)
        expected = compared = 0
        for row in audit["rows"]:
            raw = torch.load(root / row["evidence_file"], map_location="cpu", weights_only=True)
            checked = strict_response_parity(
                raw["actor"], raw["rollout"], torch.ones_like(raw["actor"], dtype=torch.bool)
            )
            assert all(row[k] == checked[k] for k in checked)
            assert raw["weight_version"] == version
            expected += row["expected_tokens"]
            compared += row["compared_tokens"]
            group, ordinal, order, name, *_ = row["label"].split("-")
            cases = [cases_by_group[group][n] for n in order]
            assert raw["case"] == cases_by_group[group][name]
            assert raw["actor"].ndim == raw["rollout"].ndim == 1
            assert (
                raw["actor"].numel() == len(raw["case"]["response_ids"]) == row["expected_tokens"]
            )
            assert raw["actual_actor_micro_batch_size"] == len(order)
            assert len(raw["actor_forward_batches"]) == 1
            actor = raw["actor_forward_batches"][0]
            assert actor["shape"][0] == len(order)
            for tokens, case in zip(actor["tokens"], cases, strict=True):
                seq = case["prompt_ids"] + case["response_ids"]
                assert tokens[: len(seq)] == seq and all(t == 0 for t in tokens[len(seq) :])
            assert len(raw["stages"]) == len(rollout_split())
            for stage in raw["stages"]:
                assert (
                    stage["weight_version"] == version
                    and worker_key(stage) == pids[stage["pipeline_stage"]]
                )
                assert len(stage["batches"]) == 1
                batch = stage["batches"][0]
                assert batch["num_requests"] == len(order)
                starts = batch["query_start_loc"]
                assert [b - a for a, b in zip(starts[:-1], starts[1:], strict=True)] == [
                    len(c["prompt_ids"]) + len(c["response_ids"]) for c in cases
                ]
                if batch["input_token_rows"] is not None:
                    assert batch["input_token_rows"] == [
                        c["prompt_ids"] + c["response_ids"] for c in cases
                    ]
                if len(order) == 2:
                    assert stage["order"]
        assert (
            expected == compared == audit["expected_tokens"] == audit["compared_tokens"]
            and compared > 0
        )
        assert audit["mismatch_count"] == audit["mean_abs_diff"] == audit["max_abs_diff"] == 0
        versions.append(
            dict(
                version=version,
                global_step=version,
                actual_updates=version,
                expected_tokens=expected,
                compared_tokens=compared,
                mismatch_count=0,
                mean_abs_diff=0.0,
                max_abs_diff=0.0,
                batch_invariance=True,
                sync=True,
                pids=pids,
            )
        )
        if version:
            batch = json.loads((root / f"training-batch-step{version}.json").read_text())
            assert (
                batch["algorithm"] == "grpo"
                and batch["native_policy_loss"] == "verl.workers.utils.losses.ppo_loss"
            )
            assert batch["weight_version"] == version - 1 and batch["global_steps"] == version
            native = batch["native_agent_loop"]
            assert len(native["metrics"]) == len(native["extra_fields"]) == 4
            assert all(
                m["generate_sequences"] > 0 and m["compute_score"] > 0 for m in native["metrics"]
            )
            assert all(
                e["min_global_steps"] == e["max_global_steps"] == e["global_steps"] == version - 1
                for e in native["extra_fields"]
            )
            assert batch["summaries"]["advantages"]["nonzero"] > 0
            assert all(item["finite"] for item in batch["summaries"].values())
            scores = torch.load(
                root / f"parity-step{version}-raw.pt", map_location="cpu", weights_only=True
            )
            strict_response_parity(
                scores["old_log_probs"], scores["rollout_log_probs"], scores["response_mask"]
            )
            ranks = []
            actor_stages = len(actor_split())
            for rank in range(actor_stages):
                actor = json.loads((root / f"actor-rank-{rank}-v{version}.json").read_text())
                assert actor["rank"] == rank and actor["weight_version"] == version
                assert (
                    actor["gradients_finite"]
                    and actor["gradients_nonzero"]
                    and actor["changed_parameters"]
                )
                assert actor["optimizer_before"]["pid"] == identities[rank]["pid"]
                assert set(actor["parameter_before"]) == set(
                    identities[rank]["trainable_inventory"]
                )
                receipts = actor["autograd_receipts"]
                assert receipts and all(row["finite"] for rows in receipts.values() for row in rows)
                assert all(
                    name in receipts
                    for name, gradient in actor["gradient_records"].items()
                    if gradient["nonzero"]
                )
                assert actor["native_scheduler"]["moments_unchanged"]
                verify_optimizer_transition(
                    actor["optimizer_before"],
                    actor["optimizer_after"],
                    version=version,
                    previous=previous_actor.get(rank),
                )
                loss_calls = json.loads(
                    (root / f"native-loss-rank{rank}-v{version}.json").read_text()
                )
                if rank == actor_stages - 1:
                    assert len(loss_calls) == 4 and all(
                        c["finite"]
                        and c["requires_grad"]
                        and c["callable"] == "verl.workers.utils.losses.ppo_loss"
                        for c in loss_calls
                    )
                else:
                    assert loss_calls == []
                previous_actor[rank] = actor["optimizer_after"]
                ranks.append(actor)
            for boundary, (send, recv) in [
                (b, pair)
                for b in range(actor_stages - 1)
                for pair in (("forward_send", "forward_recv"), ("backward_recv", "backward_send"))
            ]:
                left = [
                    r for r in ranks[boundary]["pipeline_transfers"] if r["direction"] == send
                ]
                right = [
                    r for r in ranks[boundary + 1]["pipeline_transfers"] if r["direction"] == recv
                ]
                assert left and right and len(left) == len(right) == 4
                assert all(r["finite"] for r in left + right)
                assert [r["sha256"] for r in sorted(left, key=lambda r: r["microbatch"])] == [
                    r["sha256"] for r in sorted(right, key=lambda r: r["microbatch"])
                ]
                if send.startswith("backward"):
                    assert any(r["nonzero"] for r in left)
            metrics = json.loads((root / f"metrics-step{version}.json").read_text())
            assert metrics["global_steps"] == version and metrics["includes_audit"]
    parity = [json.loads(line) for line in (root / "parity.jsonl").read_text().splitlines()]
    assert len(parity) == 2 * total_steps + 1 and all(
        r["expected_tokens"] == r["compared_tokens"] > 0 and r["mismatch_count"] == 0
        for r in parity
    )
    stress = None
    config = job / "hydra/.hydra/config.yaml"
    if config.exists():
        from omegaconf import OmegaConf

        repeats = int(OmegaConf.load(config).rlk.get("stress_final_repeats", 0))
        if repeats:
            stress = verify_stress(root, repeats, pids)
    placement = None
    if config.exists():
        from omegaconf import OmegaConf

        declared = OmegaConf.load(config)
        if declared.rlk.get("multinode", False):
            from verl.experimental.rl_kernel.cluster_contract import topology, verify_placement

            pre = json.loads((root / "cluster-preflight.json").read_text())
            assert pre["plan"] == topology(declared)
            observed = json.loads((root / "cluster-placement.json").read_text())
            placement = verify_placement(
                pre["plan"], observed["actor"], observed["rollout"], pre["nodes"]
            )
            # Verify placement against actual execution evidence, not just a sidecar.
            for rank in range(len(actor_split())):
                identity = identities[rank]
                placed = next(r for r in placement["actor"] if r["rank"] == rank)
                assert identity["source_identity"]["node_id"] == placed["node_id"]
                assert identity["pid"] == placed["pid"]
                assert identity["device_identity"]["pci_bus_id"] == placed["pci_bus_id"]
            initial = json.loads((root / "sync-v0-raw.json").read_text())["stages"]
            for stage in initial:
                placed = next(
                    r for r in placement["rollout"] if r["rank"] == stage["pipeline_stage"]
                )
                assert worker_key(stage) == placed["worker_identity"]
                assert stage["node_id"] == placed["node_id"]
                assert stage["device_identity"]["pci_bus_id"] == placed["pci_bus_id"]
    result = dict(
        status="PASS",
        placement=placement,
        scope=(
            ("Explicit multi-node " if placement else "Single-node ") + scope_text(total_steps)
        ),
        versions=versions,
        dp_gradient_reduction=False,
        full_quantized_base_training=False,
        additional_v3_stress=stress,
    )
    (root / "grpo-acceptance.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--inventory", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.inventory), indent=2))
