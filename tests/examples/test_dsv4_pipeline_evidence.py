# SPDX-License-Identifier: Apache-2.0
"""CPU evidence guards and observers only, not multi-GPU model acceptance."""

import copy

import pytest
import torch

from examples.operators.dsv4_4l.distributed_evidence import PipelineBoundaryEvidence, tensor_record
from examples.operators.dsv4_4l.verify_pipeline import verify_pipeline_step, verify_rollout_shards


def records():
    result = []
    value = torch.ones(3, 2, 16, dtype=torch.bfloat16)
    for rank in range(4):
        pp, dp = rank // 2, rank % 2
        events = [
            dict(direction=direction, microbatch=0, **tensor_record(value))
            for direction in (
                ["forward_send", "backward_recv"] if pp == 0 else ["forward_recv", "backward_send"]
            )
        ]
        result.append(
            dict(
                rank=rank,
                pp_rank=pp,
                dp_rank=dp,
                weight_version=1,
                before="old",
                after="new",
                dp_gradient_reduction_verified=True,
                dp_replicas=2,
                different_local_gradients=True,
                pipeline_boundaries=events,
                gradient_witness=dict(
                    backward_executed=True, gradient_finite=True, gradient_nonzero=True
                ),
            )
        )
    return result


def test_complete_transfer_records_are_checked_in_both_directions():
    transfers = verify_pipeline_step(records(), dp_size=2, residual_width=16)
    assert len(transfers) == 4 and {x["direction"] for x in transfers} == {
        "forward_send",
        "backward_send",
    }


@pytest.mark.parametrize(
    "change",
    [
        "missing_rank",
        "missing_transfer",
        "bad_bits",
        "nonfinite",
        "no_dp",
        "same_grad",
        "old_version",
    ],
)
def test_missing_or_corrupt_distributed_evidence_fails_closed(change):
    value = records()
    if change == "missing_rank":
        value.pop()
    elif change == "missing_transfer":
        value[2]["pipeline_boundaries"] = []
    elif change == "bad_bits":
        value[2]["pipeline_boundaries"][0]["sha256"] = "wrong"
    elif change == "nonfinite":
        value[2]["pipeline_boundaries"][0]["finite"] = False
    elif change == "no_dp":
        value[0]["dp_replicas"] = 1
    elif change == "same_grad":
        value[0]["different_local_gradients"] = False
    elif change == "old_version":
        value[2]["weight_version"] = 0
    with pytest.raises(ValueError):
        verify_pipeline_step(value, dp_size=2, residual_width=16)


def test_rollout_shard_union_requires_every_original_tensor_and_changed_live_workers():
    acks = [
        dict(
            pipeline_stage=i,
            original_layer_ids=ids,
            weight_version=1,
            parameter_load_completed=True,
            weight_changed=True,
            verified_original_tensors={name: dict(sha256=name)},
        )
        for i, ids, name in [(0, [0, 1, 2], "embed.weight"), (1, [3], "norm.weight")]
    ]
    manifest = {name: dict(sha256=name) for name in ["embed.weight", "norm.weight"]}
    assert set(verify_rollout_shards(acks, manifest)) == set(manifest)
    for change in ["missing", "duplicate", "old", "unchanged", "wrong_layers"]:
        broken = copy.deepcopy(acks)
        if change == "missing":
            broken[1]["verified_original_tensors"] = {}
        elif change == "duplicate":
            broken[1]["verified_original_tensors"].update(broken[0]["verified_original_tensors"])
        elif change == "old":
            broken[1]["weight_version"] = 0
        elif change == "unchanged":
            broken[1]["weight_changed"] = False
        elif change == "wrong_layers":
            broken[0]["original_layer_ids"] = [0, 1]
        with pytest.raises(ValueError):
            verify_rollout_shards(broken, manifest)


def test_observer_preserves_the_original_gradient_through_both_pipeline_boundaries():
    class Stage(torch.nn.Module):
        def __init__(self, first):
            super().__init__()
            self.pre_process = first
            self.post_process = not first

        def set_input_tensor(self, value):
            self.input = value

        def forward(self, x=None):
            return x * 2 if self.pre_process else self.input * 3

    first, last = Stage(True), Stage(False)
    a, b = PipelineBoundaryEvidence(first, 0), PipelineBoundaryEvidence(last, 0)
    a.begin("step")
    b.begin("step")
    x = torch.ones(3, 2, 16, dtype=torch.bfloat16, requires_grad=True)
    sent = first(x)
    wire = sent.detach().clone().requires_grad_()
    last.set_input_tensor(wire)
    last().sum().backward()
    sent.backward(wire.grad)
    assert torch.equal(x.grad, torch.full_like(x, 6))
    by_direction = {r["direction"]: r for r in a.records + b.records}
    assert by_direction["forward_send"]["sha256"] == by_direction["forward_recv"]["sha256"]
    assert by_direction["backward_send"]["sha256"] == by_direction["backward_recv"]["sha256"]


def test_topology_requires_actual_native_groups_and_disjoint_worker_pci_devices():
    from examples.operators.dsv4_4l.verify_pipeline import verify_worker_topology

    actor = [
        dict(
            rank=i,
            pp_rank=i // 2,
            dp_rank=i % 2,
            pp_ranks=[i % 2, i % 2 + 2],
            dp_ranks=[2 * (i // 2), 2 * (i // 2) + 1],
            original_layer_ids=[0, 1, 2] if i < 2 else [3],
            device_identity=dict(pci_bus_id=str(i), architecture="gfx942:sramecc+:xnack-"),
        )
        for i in range(4)
    ]
    rollout = [
        dict(
            pipeline_stage=i,
            original_layer_ids=[0, 1, 2] if i == 0 else [3],
            device_identity=dict(pci_bus_id=str(i + 4), architecture="gfx942"),
        )
        for i in range(2)
    ]
    assert len(verify_worker_topology(actor, rollout, dp_size=2)) == 6
    for change in ["shared_gpu", "dp_group", "pp_group", "layers", "replicas", "wrong_arch"]:
        a, r = copy.deepcopy((actor, rollout))
        if change == "shared_gpu":
            r[0]["device_identity"]["pci_bus_id"] = a[0]["device_identity"]["pci_bus_id"]
        elif change == "dp_group":
            a[0]["dp_ranks"] = [0]
        elif change == "pp_group":
            a[0]["pp_ranks"] = [0]
        elif change == "layers":
            r[0]["original_layer_ids"] = [0, 1]
        elif change == "replicas":
            a[2]["pp_rank"] = 0
        elif change == "wrong_arch":
            a[0]["device_identity"]["architecture"] = "gfx950"
        with pytest.raises(ValueError):
            verify_worker_topology(a, r, dp_size=2)


def test_single_card_score_success_cannot_pass_pipeline_acceptance(tmp_path):
    import json

    from examples.operators.dsv4_4l.verify_pipeline import verify_pipeline_run

    result = {
        name: True
        for name in [
            "before_passed",
            "after_passed",
            "pure_rlk_passed",
            "update_executed",
            "same_rollout_process",
            "expanded_coverage_passed",
        ]
    }
    result["rows"] = [
        dict(
            expected_tokens=1,
            compared_tokens=1,
            mismatch_count=0,
            all_finite=True,
            dtype="torch.float32",
        )
    ]
    result["actor_parallelism"] = dict(TP=1, PP=1, EP=1, CP=1, DP=1)
    (tmp_path / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="selected parallel configuration"):
        verify_pipeline_run(tmp_path, dp_size=2)


def test_actual_verl_global_token_loss_and_backward_match_the_global_token_mean():
    tu = pytest.importorskip("verl.utils.tensordict_utils")
    from tensordict import TensorDict

    from examples.operators.dsv4_4l.distributed_evidence import distributed_token_loss

    values = [
        torch.tensor([-1.0, -2.0], requires_grad=True),
        torch.tensor([-3.0, -4.0, -5.0], requires_grad=True),
    ]
    local_losses = []
    for raw in values:
        data = TensorDict(
            {
                "loss_mask": torch.nested.as_nested_tensor(
                    [torch.ones_like(raw)], layout=torch.jagged
                )
            },
            batch_size=[1],
        )
        tu.assign_non_tensor(data, batch_num_tokens=5)
        output = {"raw_log_probs": torch.nested.as_nested_tensor([raw], layout=torch.jagged)}
        loss, metrics = distributed_token_loss(output, data, dp_size=2)
        local_losses.append(loss)
        assert metrics == {}
    # Native DDP averages local parameter gradients; the global token mean is
    # correct even though these replicas have different valid response counts.
    reduced = sum(local_losses) / 2
    assert reduced.item() == 3.0
    reduced.backward()
    for raw in values:
        assert torch.equal(raw.grad, torch.full_like(raw, -0.2))


@pytest.mark.parametrize("total", [None, 0, -1, float("nan"), float("inf")])
def test_missing_or_invalid_actual_global_token_counts_do_not_get_a_fallback(total):
    tu = pytest.importorskip("verl.utils.tensordict_utils")
    from tensordict import TensorDict

    from examples.operators.dsv4_4l.distributed_evidence import distributed_token_loss

    data = TensorDict({}, batch_size=[])
    if total is not None:
        tu.assign_non_tensor(data, batch_num_tokens=total)
    with pytest.raises(ValueError, match="actual global valid-token count"):
        distributed_token_loss({}, data, dp_size=2)
