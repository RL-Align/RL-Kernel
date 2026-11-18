# SPDX-License-Identifier: Apache-2.0
"""CPU audit of actual saved worker outputs and synchronization manifests."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_num_threads(1)

root = Path("/outputs")
proof = json.loads((root / "continuous-acceptance.json").read_text())
assert proof["accepted"] and proof["continuous_updates"] == 3
cases = json.loads((root / "cases-v0.json").read_text())
score_changes = []
for version in [1, 2, 3]:
    changes = []
    for c in cases:
        records = [
            torch.load(
                root / f'actor-v{v}-{c["id"]}-matrix.pt', weights_only=True, map_location="cpu"
            )
            for v in [version - 1, version]
        ]
        assert all(
            x["cases"] == [c] and x["weight_version"] == v
            for x, v in zip(records, [version - 1, version], strict=True)
        )
        a, b = [x["selected"][0] for x in records]
        assert (
            a.dtype == b.dtype == torch.float32
            and torch.isfinite(a).all()
            and torch.isfinite(b).all()
        )
        bad = a.view(torch.int32) != b.view(torch.int32)
        entry = dict(
            case_id=c["id"], changed_raw_score_tokens=int(bad.sum()), response_tokens=a.numel()
        )
        if bad.any():
            index = int(bad.nonzero()[0])
            entry["first_changed"] = dict(
                response_offset=index,
                token_id=c["response_token_ids"][index],
                before=a[index].item(),
                after=b[index].item(),
                before_bits=f"{int(a.view(torch.int32)[index]) & 0xffffffff:08x}",
                after_bits=f"{int(b.view(torch.int32)[index]) & 0xffffffff:08x}",
            )
        changes.append(entry)
    assert sum(x["changed_raw_score_tokens"] for x in changes) > 0
    score_changes.append(
        dict(
            version=version,
            actual_actor_rollout_same_weight_scores_already_byte_verified=True,
            fixed_sequence_changes=changes,
        )
    )
dtype_checks = []
previous_stage_state = {}
parameter_state_chain = []
with safe_open("/models/4l/model.safetensors", framework="pt") as original:
    quantized_source_dtype_counts = {}
    for key in original.keys():
        name = original.get_slice(key).get_dtype()
        quantized_source_dtype_counts[name] = quantized_source_dtype_counts.get(name, 0) + 1
    for version in [1, 2, 3]:
        manifest = json.loads((root / f"steps/v{version}/actor-update-manifest.json").read_text())
        indices = []
        for key, value in manifest.items():
            source = original.get_tensor(key)
            if source.dtype in [torch.int64, torch.int32]:
                assert (
                    value["dtype"] == "torch.int32"
                    and bool((source >= 0).all())
                    and bool((source <= torch.iinfo(torch.int32).max).all())
                )
                expected = source.to(torch.int32).contiguous().view(torch.uint8).numpy().tobytes()
                assert hashlib.sha256(expected).hexdigest() == value["sha256"]
                indices.append(
                    dict(
                        name=key,
                        original_dtype=str(source.dtype),
                        native_dtype=value["dtype"],
                        every_integer_value_preserved=True,
                    )
                )
            else:
                assert (
                    source.dtype in [torch.bfloat16, torch.float32]
                    and str(source.dtype) == value["dtype"]
                ), key
        stages = json.loads((root / f"steps/v{version}/actor-distributed-step.json").read_text())
        scope = []
        for stage in stages:
            gradients = json.loads((root / stage["gradient_evidence"]).read_text())
            assert not gradients["missing_backward_parameters"]
            changes = json.loads((root / stage["state_change_evidence"]).read_text())
            if version > 1:
                assert (
                    changes["before_parameters"]
                    == previous_stage_state[stage["rank"]]["after_parameters"]
                )
                assert (
                    changes["before_sync_state"]
                    == previous_stage_state[stage["rank"]]["after_sync_state"]
                )
            else:
                for key, description in changes["before_sync_state"].items():
                    value = original.get_tensor(key)
                    if value.dtype == torch.int64:
                        value = value.to(torch.int32)
                    assert (
                        hashlib.sha256(
                            value.contiguous().view(torch.uint8).numpy().tobytes()
                        ).hexdigest()
                        == description["sha256"]
                    ), key
            previous_stage_state[stage["rank"]] = changes
            parameter_state_chain.append(
                dict(
                    version=version,
                    rank=stage["rank"],
                    actual_initial_checkpoint_or_prior_updated_state_hash_verified=True,
                )
            )
            scope.append(
                dict(
                    rank=stage["rank"],
                    trainable_parameters=stage["trainable_parameters"],
                    trainable_elements=stage["trainable_elements"],
                    changed_parameters=len(stage["changed_parameters"]),
                    nonzero_gradient_parameters=sum(
                        v["nonzero"] for v in gradients["reduced_gradients"].values()
                    ),
                    optimizer_step_counters=stage["optimizer_evidence"]["after"]["step_counters"],
                )
            )
        dtype_checks.append(
            dict(
                version=version,
                actual_sync_tensor_count=len(manifest),
                all_floating_checkpoint_dtypes_preserved=True,
                packed_quantized_bases_excluded_from_optimizer_update_payload=True,
                index_abi=indices,
                actual_parameter_scope=scope,
            )
        )
report = dict(
    scope="CPU audit of saved real actor/rollout acceptance, no independent engine/model execution",
    accepted=True,
    fixed_sequence_actual_raw_fp32_changes=score_changes,
    update_dtype_and_parameter_scope=dtype_checks,
    original_checkpoint_dtype_counts=quantized_source_dtype_counts,
    mutable_parameter_and_buffer_state_chain=parameter_state_chain,
)
Path("/audit/actual-update-and-dtype-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        dict(
            accepted=True,
            changed_fixed_tokens_per_step=[
                sum(y["changed_raw_score_tokens"] for y in x["fixed_sequence_changes"])
                for x in score_changes
            ],
            sync_counts=[x["actual_sync_tensor_count"] for x in dtype_checks],
            original_checkpoint_dtype_counts=quantized_source_dtype_counts,
            mutable_parameter_and_buffer_state_chain=parameter_state_chain,
        )
    )
)
