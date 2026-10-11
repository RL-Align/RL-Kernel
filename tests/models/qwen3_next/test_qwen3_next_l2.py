# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU artifact and matrix contracts, never real-model acceptance evidence."""

import copy
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.validation.models.qwen3_next_l2 import (
    PROMPT_LENGTHS,
    RESPONSE_LENGTH,
    VOCAB_SIZE,
    L2Case,
    case_matrix,
    compare_artifacts,
    compare_batch_prefill,
    embedding_digest,
    first_divergent_layer,
    freeze_responses,
    layer_digest,
    layer_state_names,
    load_tensor_artifact,
    prompt_fixture,
    save_tensor_artifact,
    validate_fixture,
    validate_model_artifact,
    worker_request,
)


class UnitTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in text]


@pytest.fixture(scope="module")
def frozen():
    prompts = prompt_fixture(UnitTokenizer(), {"tokenizer.json": "test-tokenizer-hash"})
    baseline = {
        str(length): {
            "response_tokens": torch.arange(64, dtype=torch.int64).view(4, 16),
            "logprobs": torch.full((4, 16), -1.0),
        }
        for length in PROMPT_LENGTHS
    }
    return freeze_responses(
        prompts,
        baseline,
        baseline_identity="unit-test-only",
        baseline_records={
            str(length): {"path": f"/unit-test-only/{length}.pt", "sha256": "1" * 64}
            for length in PROMPT_LENGTHS
        },
    )


@pytest.fixture(scope="module")
def small_artifact(frozen):
    request = worker_request(
        L2Case(8, 1, "full"), frozen, role="rollout", identity={"test_only": True}
    )
    return request, make_artifact(request)


def subnode_values(batch):
    from rl_engine.models.qwen3_next.qwen3_next_subnode import SUBNODES, subnode_layout

    return {
        name: torch.zeros((batch, 16, subnode_layout(name)[0]), dtype=subnode_layout(name)[1])
        for name in SUBNODES
    }


def make_artifact(request):
    batch = len(request["prompt_tokens"])
    output = torch.zeros((batch, 16, 2048), dtype=torch.bfloat16)

    def states(layer):
        return {
            name: torch.ones(
                (batch, 1, 2, 2),
                dtype=torch.float32 if "recurrent_state" in name else torch.bfloat16,
            )
            for name in layer_state_names(layer)
        }

    return {
        "metadata": {
            key: request[key]
            for key in ("schema_version", "case_id", "role", "identity", "request_sha256")
        },
        "execution": request["execution"],
        "prompt_tokens": torch.tensor(request["prompt_tokens"], dtype=torch.int64),
        "response_tokens": torch.tensor(request["response_tokens"], dtype=torch.int64),
        "prediction_positions": torch.tensor(request["prediction_positions"], dtype=torch.int64),
        "logits": torch.zeros((batch, RESPONSE_LENGTH, VOCAB_SIZE), dtype=torch.bfloat16),
        "logprobs": torch.full((batch, RESPONSE_LENGTH), -1.0, dtype=torch.float32),
        "layer_digests": [
            layer_digest(layer, output, states(layer), subnodes=subnode_values(batch))
            for layer in range(48)
        ],
        "embedding_digest": embedding_digest(torch.zeros((batch, 16, 2048), dtype=torch.bfloat16)),
    }


def test_exact_sixteen_case_matrix():
    cases = case_matrix()
    assert len(cases) == 16 and len({case.case_id for case in cases}) == 16
    assert {case.prompt_length for case in cases} == {8, 64, 256, 1024}
    assert {case.batch_size for case in cases} == {1, 4}
    assert {case.prefill for case in cases} == {"full", "chunked"}
    assert all(
        sum(case.chunks) == case.prompt_length and all(size > 0 for size in case.chunks)
        for case in cases
    )
    assert L2Case(1024, 4, "chunked").chunks[:2] == [1, 63]


def test_offsets_and_batch_one_reuse_frozen_first_sample(frozen):
    validate_fixture(frozen)
    one = worker_request(L2Case(64, 1, "chunked"), frozen, role="recompute", identity={})
    four = worker_request(L2Case(64, 4, "full"), frozen, role="rollout", identity={})
    assert one["prompt_tokens"][0] == four["prompt_tokens"][0]
    assert one["response_tokens"][0] == four["response_tokens"][0]
    assert one["prediction_positions"] == list(range(63, 79))
    assert one["consumed_response_tokens"] == 15
    assert len(one["consumed_input_tokens"][0]) == 79
    assert one["state_position"] == 78
    assert one["execution"] == {"recompute_mode": "full_from_tokens", "input_sequence_length": 79}
    assert one["prefill_chunks"] is None
    assert one["rollout_prefill_chunks"] == [1, 63]
    assert not any(key in one for key in ("hidden_states", "recurrent_state", "rollout_state"))


def test_frozen_tokens_and_old_logprobs_cannot_change(frozen):
    for field in ("prompts", "responses", "old_logprobs"):
        changed = copy.deepcopy(frozen)
        changed[field]["64"][0, 0] += 1
        with pytest.raises(ValueError):
            validate_fixture(changed)


def test_weights_only_tensor_serialization_roundtrip(tmp_path, frozen):
    path = tmp_path / "input.pt"
    save_tensor_artifact(path, frozen)
    restored = load_tensor_artifact(path)
    validate_fixture(restored)
    assert restored["content_sha256"] == frozen["content_sha256"]
    assert not path.with_suffix(".pt.partial").exists()


def test_valid_original_bf16_artifact(small_artifact):
    request, artifact = small_artifact
    validate_model_artifact(artifact, request)
    report = compare_artifacts(artifact, artifact)
    assert report["passed"] and report["first_divergent_layer"] is None


@pytest.mark.parametrize("field", ["logits", "logprobs"])
def test_signed_zero_is_a_real_full_model_difference(small_artifact, field):
    _, artifact = small_artifact
    candidate = {**artifact, field: artifact[field].clone()}
    reference = {**artifact, field: artifact[field].clone()}
    reference[field].reshape(-1)[0] = 0.0
    candidate[field].reshape(-1)[0] = -0.0
    report = compare_artifacts(reference, candidate)
    assert not report["passed"]
    assert report["first_difference"]["tensor"] == field
    assert report["first_difference"]["flat_element"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_logits_never_pass(small_artifact, value):
    request, artifact = small_artifact
    changed = {**artifact, "logits": artifact["logits"].clone()}
    changed["logits"][0, 0, 0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        validate_model_artifact(changed, request)


def test_widening_logits_or_truncating_vocab_is_rejected(small_artifact):
    request, artifact = small_artifact
    for logits in (artifact["logits"].float(), artifact["logits"][..., :1024]):
        with pytest.raises(ValueError, match="original storage"):
            validate_model_artifact({**artifact, "logits": logits}, request)


def test_wrong_response_offset_and_changed_tokens_are_rejected(small_artifact):
    request, artifact = small_artifact
    with pytest.raises(ValueError, match="offsets"):
        validate_model_artifact(
            {**artifact, "prediction_positions": artifact["prediction_positions"] + 1}, request
        )
    with pytest.raises(ValueError, match="frozen response"):
        validate_model_artifact(
            {**artifact, "response_tokens": artifact["response_tokens"] + 1}, request
        )


def test_missing_layers_and_placeholder_hashes_are_rejected(small_artifact):
    request, artifact = small_artifact
    with pytest.raises(ValueError, match="48-layer"):
        validate_model_artifact(
            {**artifact, "layer_digests": artifact["layer_digests"][:-1]}, request
        )
    changed = copy.deepcopy(artifact["layer_digests"])
    changed[0]["sample_output_sha256"] = []
    with pytest.raises(ValueError, match="Every layer"):
        validate_model_artifact({**artifact, "layer_digests": changed}, request)
    changed = copy.deepcopy(artifact["layer_digests"])
    changed[0]["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound"):
        validate_model_artifact({**artifact, "layer_digests": changed}, request)


def test_first_divergence_tracks_state_even_when_final_logits_match(small_artifact):
    _, artifact = small_artifact
    changed = copy.deepcopy(artifact["layer_digests"])
    changed[11]["sample_state_sha256"] = ["e" * 64]
    candidate = {**artifact, "layer_digests": changed}
    result = compare_artifacts(artifact, candidate)
    assert not result["passed"] and result["first_divergent_layer"] == 11


@pytest.mark.parametrize(
    "node", ["input_layernorm", "mixer", "post_attention_layernorm", "router", "mlp"]
)
def test_first_divergent_node_names_the_sub_node(small_artifact, node):
    from rl_engine.validation.models.qwen3_next_l2 import first_divergent_node

    _, artifact = small_artifact
    changed = copy.deepcopy(artifact["layer_digests"])
    changed[23]["sample_subnode_sha256"][node] = ["e" * 64]
    changed[30]["sample_output_sha256"] = ["e" * 64]
    candidate = {**artifact, "layer_digests": changed}
    assert first_divergent_node(artifact, candidate) == {"layer": 23, "node": node}
    assert compare_artifacts(artifact, candidate)["first_divergent_node"]["node"] == node


def test_first_divergent_node_starts_at_the_embedding(small_artifact):
    from rl_engine.validation.models.qwen3_next_l2 import first_divergent_node

    _, artifact = small_artifact
    changed = copy.deepcopy(artifact["layer_digests"])
    changed[0]["sample_subnode_sha256"]["input_layernorm"] = ["e" * 64]
    candidate = {
        **artifact,
        "layer_digests": changed,
        "embedding_digest": {"sample_sha256": ["e" * 64], "sha256": "x"},
    }
    assert first_divergent_node(artifact, candidate) == {"layer": 0, "node": "embedding"}


def test_first_divergent_node_falls_back_to_layer_output_and_state(small_artifact):
    from rl_engine.validation.models.qwen3_next_l2 import first_divergent_node

    _, artifact = small_artifact
    assert first_divergent_node(artifact, artifact) is None
    changed = copy.deepcopy(artifact["layer_digests"])
    changed[5]["sample_state_sha256"] = ["e" * 64]
    assert first_divergent_node(artifact, {**artifact, "layer_digests": changed}) == {
        "layer": 5,
        "node": "state",
    }


def test_layer_digest_requires_every_sub_node_in_order(small_artifact):
    request, artifact = small_artifact
    output = torch.zeros((1, 16, 2048), dtype=torch.bfloat16)
    states = {
        name: torch.ones(
            (1, 1, 2, 2), dtype=torch.float32 if "recurrent_state" in name else torch.bfloat16
        )
        for name in layer_state_names(0)
    }
    nodes = subnode_values(1)
    with pytest.raises(ValueError, match="sub-nodes"):
        layer_digest(0, output, states, subnodes=dict(reversed(list(nodes.items()))))
    nodes["router"] = nodes["router"].int()
    with pytest.raises(ValueError, match="router"):
        layer_digest(0, output, states, subnodes=nodes)


def test_batch_invariance_selects_first_samples(frozen, small_artifact):
    _, candidate = small_artifact
    request = worker_request(
        L2Case(8, 4, "full"), frozen, role="rollout", identity={"test_only": True}
    )
    reference = make_artifact(request)
    assert all(compare_batch_prefill(reference, candidate).values())
    reference["logits"][3, 0, 0] = 3.0
    assert all(compare_batch_prefill(reference, candidate).values())
    reference["logits"][0, 0, 0] = 3.0
    assert not compare_batch_prefill(reference, candidate)["logits"]
    reference["layer_digests"][7]["sample_state_sha256"][0] = "f" * 64
    assert first_divergent_layer(reference, candidate) == 7


def test_only_initial_rollout_can_generate_fixture(frozen):
    for role, case in (
        ("recompute", L2Case(64, 4, "full")),
        ("rollout", L2Case(64, 1, "full")),
        ("rollout", L2Case(64, 4, "chunked")),
    ):
        with pytest.raises(ValueError, match="Only initial"):
            worker_request(case, frozen, role=role, identity={}, generate=True)


def test_native_worker_absence_fails_readiness(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))
    import qwen3_next_l2 as runner

    def absent(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(runner.importlib, "import_module", absent)
    with pytest.raises(RuntimeError, match="native_workers_not_ready"):
        runner.check_ready()


def test_gpu_release_never_cancels_an_unrelated_process(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))
    import qwen3_next_l2 as runner

    commands = []

    def output(command, **kwargs):
        commands.append(command)
        return "99, GPU-owned, 10\\n123, GPU-other, 20\\n".replace("\\n", "\n")

    monkeypatch.setattr(runner.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="gpu_memory_not_released"):
        runner.wait_gpu_release(["GPU-owned"], timeout=0)
    runner.wait_gpu_release(["GPU-free"], timeout=0)
    assert all(command[0] == "nvidia-smi" for command in commands)


def test_batch_callback_loads_once_and_freezes_before_recompute_cases(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))
    import qwen3_next_l2 as runner

    calls = []

    def native_batch(cases):
        calls.append([])
        for request, path in cases:
            calls[-1].append(request)
            responses = request["response_tokens"]
            batch = len(request["prompt_tokens"])
            if responses is None:
                responses = torch.arange(batch * 16).view(batch, 16).tolist()
            save_tensor_artifact(
                path,
                {
                    "response_tokens": torch.tensor(responses),
                    "logprobs": torch.full((batch, 16), -1.0),
                },
            )

    def eager_recompute(cases):
        # The actual recompute worker must collect requests before spawning its
        # single actor lifetime; none of these outputs exist during collection.
        requests = list(cases)
        assert len(requests) == 16
        assert all(not path.exists() for _, path in requests)
        native_batch(requests)

    module = SimpleNamespace(run_rollout_cases=native_batch, run_recompute_cases=eager_recompute)
    monkeypatch.setattr(runner, "check_ready", lambda: (module, {"batched_cases": True}))
    monkeypatch.setattr(runner, "verify_environment", lambda *a, **k: None)
    monkeypatch.setattr(runner, "verify_before_model_load", lambda *a, **k: None)
    monkeypatch.setattr(runner, "validate_model_artifact", lambda *a: None)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: UnitTokenizer())
        ),
    )
    checkpoint = {"metadata": {"tokenizer": "pinned"}}
    manifest = {
        "source_commit": "unit",
        "source_tree": "unit",
        "environment_sha256": "unit",
        "checkpoint": "/unit",
        "checkpoint_identity": checkpoint,
        "provider_config": {},
        "topology": {"tp": 4, "pp": 1, "cp": 1, "ep": 1},
        "environment": {},
    }
    monkeypatch.setattr(runner, "checkpoint_identity", lambda path: checkpoint)
    fixture = tmp_path / "inputs/frozen.pt"
    runner.worker_batch(manifest, tmp_path / "rollout", "rollout", fixture)
    assert len(calls) == 1 and len(calls[0]) == 20
    assert all(row["response_tokens"] is None for row in calls[0][:4])
    assert all(row["response_tokens"] is not None for row in calls[0][4:])
    frozen_inputs = load_tensor_artifact(fixture)
    validate_fixture(frozen_inputs)
    runner.worker_batch(manifest, tmp_path / "recompute", "recompute", fixture)
    assert len(calls) == 2 and len(calls[1]) == 16
    assert all(row["response_tokens"] is not None for row in calls[1])
    result = json.loads((tmp_path / "recompute/worker-result.json").read_text())
    assert len(result["cases"]) == 16
    updated = {
        **manifest,
        "weight_version": "update-1",
        "base_checkpoint": "/unit",
        "base_checkpoint_identity": checkpoint,
        "checkpoint": "/updated-unit",
        "checkpoint_identity": {"metadata": {"tokenizer": "pinned", "index": "new"}},
    }
    monkeypatch.setattr(runner, "verify_updated_weights", lambda manifest: {})
    runner.worker_batch(updated, tmp_path / "updated", "rollout", fixture)
    assert len(calls) == 3 and len(calls[2]) == 16
    assert all(row["response_tokens"] is not None for row in calls[2])


def test_passed_flag_cannot_replace_full_matrix():
    from rl_engine.validation.models.qwen3_next_l2 import validate_matrix_report

    with pytest.raises(ValueError, match="sixteen-case"):
        validate_matrix_report({"status": "passed", "raw_bit_comparison": True})


def test_updated_weights_require_existing_baseline_fixture_before_worker(tmp_path, monkeypatch):
    import qwen3_next_l2 as runner

    def unexpected():
        raise AssertionError("A missing frozen fixture must fail before loading native workers")

    monkeypatch.setattr(runner, "check_ready", unexpected)
    with pytest.raises(RuntimeError, match="already frozen baseline"):
        runner.worker_batch(
            {"weight_version": "update-1"}, tmp_path / "out", "rollout", tmp_path / "missing.pt"
        )


def test_updated_preload_binds_actual_export_to_baseline_source_and_version(tmp_path, monkeypatch):
    import qwen3_next_l2 as runner
    import qwen3_next_training_export as export

    identity = {
        "source_commit": "source",
        "source_tree": "tree",
        "environment_sha256": "env",
        "provider_config": {"provider": "strict"},
        "topology": {"tp": 4},
    }
    record = {
        "weight_version": "update-1",
        "base_checkpoint": "/official",
        "base_checkpoint_identity": {"revision": "pinned"},
        "training_identity": identity,
    }
    manifest = {
        **identity,
        **{
            key: record[key]
            for key in ("weight_version", "base_checkpoint", "base_checkpoint_identity")
        },
        "checkpoint": "/updated",
        "checkpoint_identity": record,
    }
    calls = []

    def deep_verify(path, expected):
        calls.append((path, expected))
        return record

    monkeypatch.setattr(export, "validate_training_export", deep_verify)
    assert runner.verify_before_model_load(manifest, tmp_path) == record
    assert calls == [("/updated", record)]
    manifest["source_commit"] = "different"
    with pytest.raises(RuntimeError, match="changed source_commit"):
        runner.verify_updated_weights(manifest)


def test_debug_capture_must_contain_matching_layer_state(small_artifact):
    from rl_engine.validation.models.qwen3_next_l2 import validate_debug_capture

    request, artifact = small_artifact
    request = {**request, "trace_layers": [2]}
    with pytest.raises(ValueError, match="missing"):
        validate_debug_capture(artifact, request)
    capture = {
        "layer.2.output": torch.zeros((1, 16, 2048), dtype=torch.bfloat16),
        **{
            f"layer.2.{name}": torch.ones(
                (1, 1, 2, 2), dtype=torch.float32 if "recurrent_state" in name else torch.bfloat16
            )
            for name in layer_state_names(2)
        },
        **{f"layer.2.subnode.{name}": value for name, value in subnode_values(1).items()},
    }
    validate_debug_capture({**artifact, "debug_tensors": capture}, request)
    capture["layer.2.tp0.recurrent_state"].zero_()
    with pytest.raises(ValueError, match="differ"):
        validate_debug_capture({**artifact, "debug_tensors": capture}, request)


def test_model_artifact_hash_is_rechecked_before_deserialization(tmp_path):
    from qwen3_next_runtime import file_digest

    from rl_engine.validation.models.qwen3_next_l2 import checked_artifact_path

    artifact = tmp_path / "output.pt"
    save_tensor_artifact(artifact, {"logits": torch.zeros(2, dtype=torch.bfloat16)})
    record = {"path": str(artifact), "sha256": file_digest(artifact)}
    checked_artifact_path(record, artifact_root=tmp_path)
    save_tensor_artifact(artifact, {"logits": torch.ones(2, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="digest changed"):
        checked_artifact_path(record, artifact_root=tmp_path)


def scheduler_request_and_audit(frozen):
    identity = {
        "weight_version": "baseline",
        "provider_config": {
            "profile": "qwen3_next_tp4_shared_eager_v1",
            "native_rollout_kv_cache_memory_bytes": 4 << 30,
        },
    }
    request = worker_request(L2Case(64, 4, "chunked"), frozen, role="rollout", identity=identity)
    computed = 0
    steps = []
    for count in [*request["prefill_chunks"], *([1] * 15)]:
        steps.append(
            {
                "samples": [3, 1, 0, 2],
                "computed_before": [computed] * 4,
                "token_counts": [count] * 4,
            }
        )
        computed += count
    return request, {
        "provider_execution_mode": "qualification",
        "native_runtime_profile": {
            "kv_cache_memory_bytes_per_gpu": 4 << 30,
            "max_model_len": 1040,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 16384,
            "scheduler": (
                "rl_engine.integrations.engines.rollout.vllm.qwen3_next_native_rollout"
                ".ExactPrefillScheduler"
            ),
            "async_scheduling": False,
            "prefix_caching": False,
            "weights_dtype": "bfloat16",
            "recurrent_state_dtype": "float32",
            "model_runner": "v1",
        },
        "native_scheduler_steps": steps,
    }


def test_actual_scheduler_must_follow_requested_chunks_and_offsets(frozen):
    from rl_engine.validation.models.qwen3_next_l2 import validate_native_scheduler_execution

    request, artifact = scheduler_request_and_audit(frozen)
    validate_native_scheduler_execution(artifact, request)
    for key in ("token_counts", "computed_before", "samples"):
        changed = copy.deepcopy(artifact)
        changed["native_scheduler_steps"][0][key][0] += 1
        with pytest.raises(ValueError, match="sample/chunk/offset"):
            validate_native_scheduler_execution(changed, request)
    changed = {**artifact, "native_scheduler_steps": artifact["native_scheduler_steps"][:-1]}
    with pytest.raises(ValueError, match="boundary"):
        validate_native_scheduler_execution(changed, request)


def test_scheduler_cannot_shrink_cache_or_claim_certified_execution(frozen):
    from rl_engine.validation.models.qwen3_next_l2 import validate_native_scheduler_execution

    request, artifact = scheduler_request_and_audit(frozen)
    artifact["native_runtime_profile"]["kv_cache_memory_bytes_per_gpu"] //= 2
    with pytest.raises(ValueError, match="runtime"):
        validate_native_scheduler_execution(artifact, request)
    request, artifact = scheduler_request_and_audit(frozen)
    artifact["provider_execution_mode"] = "certified"
    with pytest.raises(ValueError, match="certificate identity"):
        validate_native_scheduler_execution(artifact, request)


def test_layer_hashes_cannot_replace_four_original_worker_traces(frozen):
    from rl_engine.validation.models.qwen3_next_l2 import validate_native_worker_traces

    request, artifact = scheduler_request_and_audit(frozen)
    with pytest.raises(ValueError, match="each TP rank"):
        validate_native_worker_traces(artifact, request)


def test_matrix_recomputes_tensor_comparisons_even_after_reports_are_rehashed(tmp_path, frozen):
    import json

    from qwen3_next_runtime import file_digest

    from rl_engine.validation.models.qwen3_next_binary import (
        archive_provider_binary,
        validate_binary_record,
    )
    from rl_engine.validation.models.qwen3_next_l2 import json_digest, validate_matrix_report

    source = tmp_path / "source"
    (source / "rl_engine").mkdir(parents=True)
    (source / "rl_engine/_C.cpython-unit.so").write_bytes(b"unit native binary")
    provider_binary = archive_provider_binary(source, tmp_path, source_tree="unit")
    actual_binary = validate_binary_record(provider_binary)

    identity = {
        "architecture_layers": 48,
        "weight_version": "baseline",
        "topology": {"tp": 4, "pp": 1, "cp": 1, "ep": 1},
        "test_only": True,
        "source_tree": "unit",
    }

    def record(path):
        return {"path": str(path), "sha256": file_digest(path)}

    def compact_artifact(request):
        result = make_artifact(request)
        result["provider_binary"] = actual_binary
        result["logits"] = torch.zeros((), dtype=torch.bfloat16).expand(result["logits"].shape)
        return result

    baselines, baseline_records = {}, {}
    for length in PROMPT_LENGTHS:
        request = worker_request(
            L2Case(length, 4, "full"), frozen, role="rollout", identity=identity, generate=True
        )
        value = compact_artifact(
            {**request, "response_tokens": frozen["responses"][str(length)].tolist()}
        )
        path = tmp_path / f"baseline-{length}.pt"
        save_tensor_artifact(path, value)
        baselines[str(length)] = value
        baseline_records[str(length)] = record(path)
    fixture = freeze_responses(
        frozen,
        baselines,
        baseline_identity=json_digest(identity),
        baseline_records=baseline_records,
    )
    fixture_path = tmp_path / "fixture.pt"
    save_tensor_artifact(fixture_path, fixture)
    checks = {name: True for name in ("response_tokens", "logits", "logprobs", "layer_states")}
    processes = {
        role: {"exit_code": 0, "pid": index + 1, "gpu_release_verified": True}
        for index, role in enumerate(("rollout", "recompute"))
    }
    workers = {
        role: {
            "status": "passed",
            "role": role,
            "identity": identity,
            "fixture_sha256": fixture["content_sha256"],
            "pid": processes[role]["pid"],
            "cases": {},
            "provider_binary": actual_binary,
        }
        for role in processes
    }
    cases = {}
    for case in case_matrix():
        records = {}
        for role in processes:
            request = worker_request(case, fixture, role=role, identity=identity)
            path = tmp_path / f"{role}-{case.case_id}.pt"
            save_tensor_artifact(path, compact_artifact(request))
            records[role] = record(path)
            workers[role]["cases"][case.case_id] = {
                **records[role],
                "request_sha256": request["request_sha256"],
            }
        cases[case.case_id] = {
            "passed": True,
            "comparisons": checks,
            "batch_prefill": {role: checks for role in processes},
            "baseline_generation": {role: checks for role in processes},
            "artifacts": records,
        }
    worker_records = {}
    for role in processes:
        path = tmp_path / f"{role}-worker.json"
        path.write_text(json.dumps(workers[role]))
        worker_records[role] = record(path)
    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "qwen3_next_full_model_l2",
        "raw_bit_comparison": True,
        "logits_dtype": "bfloat16",
        "logprobs_dtype": "float32",
        "response_length": 16,
        "vocab_size": 151936,
        "identity": identity,
        "case_ids": [case.case_id for case in case_matrix()],
        "cases": cases,
        "processes": processes,
        "worker_results": worker_records,
        "fixture": {**record(fixture_path), "content_sha256": fixture["content_sha256"]},
        "provider_binary": provider_binary,
    }
    validate_matrix_report(report, expected_identity=identity, artifact_root=tmp_path)
    case_id = L2Case(8, 1, "full").case_id
    path = Path(cases[case_id]["artifacts"]["recompute"]["path"])
    changed = load_tensor_artifact(path)
    changed["logits"] = changed["logits"].clone()
    changed["logits"][0, 0, 0] = -0.0
    save_tensor_artifact(path, changed)
    cases[case_id]["artifacts"]["recompute"] = record(path)
    workers["recompute"]["cases"][case_id].update(record(path))
    worker_path = Path(worker_records["recompute"]["path"])
    worker_path.write_text(json.dumps(workers["recompute"]))
    report["worker_results"]["recompute"] = record(worker_path)
    with pytest.raises(ValueError, match="tensor bytes contradict"):
        validate_matrix_report(report, expected_identity=identity, artifact_root=tmp_path)
