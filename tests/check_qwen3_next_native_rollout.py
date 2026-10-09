# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Native scheduler/sampler plumbing tests; these are not GPU model evidence."""

from types import SimpleNamespace as Namespace

import pytest
import torch

from rl_engine.integrations import qwen3_next_native_rollout as rollout
from rl_engine.testing.qwen3_next_l2 import L2Case, json_digest


def request(batch=4, length=8, chunked=False, frozen=False):
    chunks = L2Case(length, batch, "chunked" if chunked else "full").chunks
    prompts = [[row + 1] * length for row in range(batch)]
    responses = [list(range(16)) for _ in range(batch)] if frozen else None
    result = {
        "schema_version": 1,
        "role": "rollout",
        "case_id": "cpu-plumbing",
        "identity": {
            "architecture_layers": 48,
            "topology": {"tp": 4, "pp": 1, "cp": 1, "ep": 1},
            "checkpoint": "/explicit/test-only-checkpoint",
            "provider_config": {"native_rollout_kv_cache_memory_bytes": 4 << 30},
        },
        "prompt_tokens": prompts,
        "response_tokens": responses,
        "sample_ids": list(range(batch)),
        "prefill_chunks": chunks,
        "execution": {
            "rollout_mode": "autoregressive",
            "prefill_chunks": chunks,
            "response_mode": "teacher_forced" if frozen else "generate",
        },
        "response_length": 16,
        "vocab_size": 151936,
        "logits_dtype": "bfloat16",
        "logprobs_dtype": "float32",
        "prediction_positions": list(range(length - 1, length + 15)),
        "state_position": length + 14,
        "consumed_response_tokens": 15,
        "required_layer_digests": list(range(48)),
        "trace_layers": [],
        "consumed_input_tokens": (
            None
            if responses is None
            else [left + right[:-1] for left, right in zip(prompts, responses)]
        ),
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 1234,
            "ignore_eos": True,
        },
    }
    result["request_sha256"] = json_digest(result)
    return result


def seal(value):
    value["request_sha256"] = json_digest(
        {key: item for key, item in value.items() if key != "request_sha256"}
    )
    return value


@pytest.mark.parametrize("length", [8, 64, 256, 1024])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("chunked", [False, True])
def test_exact_boundaries_cover_every_prefill_and_fifteen_decode_inputs(length, batch, chunked):
    value = request(batch, length, chunked, frozen=True)
    assert rollout.validate_request(value) == (batch, length)
    computed = 0
    for count in value["prefill_chunks"] + [1] * 15:
        assert rollout.next_scheduled_tokens(computed, length, value["prefill_chunks"]) == count
        computed += count
    assert computed == length + 15
    with pytest.raises(ValueError, match="exact requested"):
        rollout.next_scheduled_tokens(computed, length, value["prefill_chunks"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction_positions", list(range(8, 24))),
        ("consumed_response_tokens", 16),
        ("prefill_chunks", [7]),
        ("logits_dtype", "float32"),
        ("trace_layers", [48]),
    ],
)
def test_worker_rejects_even_resigned_requests_that_change_the_contract(field, value):
    item = request()
    item[field] = value
    with pytest.raises(ValueError):
        rollout.validate_request(seal(item))


def scheduled_request(sample, computed=0, chunks=(1, 63), batch=4):
    descriptor = {
        "request_sha256": "same-case",
        "sample_id": sample,
        "batch_size": batch,
        "prompt_length": sum(chunks),
        "prefill_chunks": list(chunks),
    }
    return Namespace(
        request_id=str(sample),
        sampling_params=Namespace(extra_args={"shared_l2": descriptor}),
        num_computed_tokens=computed,
        is_finished=lambda: False,
    )


def scheduler(requests):
    value = object.__new__(rollout.ExactPrefillScheduler)
    value.requests = {row.request_id: row for row in requests}
    value.scheduler_config = Namespace(long_prefill_token_threshold=0)
    value._pause_state = rollout.PauseState.UNPAUSED
    return value


def test_native_scheduler_barrier_waits_for_all_four_requests(monkeypatch):
    item = scheduler([scheduled_request(0)])
    calls = []

    def native(self, throttle):
        calls.append(self._pause_state)
        return Namespace(total_num_scheduled_tokens=0)

    monkeypatch.setattr(rollout.Scheduler, "schedule", native)
    item.schedule()
    assert calls == [rollout.PauseState.PAUSED_ALL]
    assert item._pause_state == rollout.PauseState.UNPAUSED


def test_native_scheduler_caps_real_decisions_and_rejects_partial_batches(monkeypatch):
    item = scheduler([scheduled_request(rank, computed=1) for rank in range(4)])

    def native(self, throttle):
        assert self.scheduler_config.long_prefill_token_threshold == 63
        return Namespace(
            num_scheduled_tokens={str(rank): 63 for rank in range(4)}, preempted_req_ids=set()
        )

    monkeypatch.setattr(rollout.Scheduler, "schedule", native)
    assert item.schedule().num_scheduled_tokens == {str(rank): 63 for rank in range(4)}
    monkeypatch.setattr(
        rollout.Scheduler,
        "schedule",
        lambda self, throttle: Namespace(num_scheduled_tokens={"0": 63}, preempted_req_ids=set()),
    )
    with pytest.raises(RuntimeError, match="exact_scheduler_mismatch"):
        item.schedule()


def test_native_scheduler_rejects_desynchronized_request_offsets():
    item = scheduler([scheduled_request(rank, computed=rank) for rank in range(4)])
    with pytest.raises(RuntimeError, match="synchronized"):
        item.schedule()


def test_native_logits_processor_forces_frozen_tokens_without_mutating_raw_logits():
    processor = rollout.ForceFrozenTokens(None, torch.device("cpu"), False)
    assert processor.is_argmax_invariant() is False
    original = torch.arange(151936, dtype=torch.float32)
    frozen = list(range(16))
    apply = processor.new_req_logits_processor(
        Namespace(extra_args={"shared_l2": {"response_tokens": frozen}})
    )
    forced = apply(frozen[:7], original)
    assert int(forced.argmax()) == 7
    assert int(original.argmax()) == 151935
    assert forced.data_ptr() != original.data_ptr()
    assert (
        processor.new_req_logits_processor(
            Namespace(extra_args={"shared_l2": {"response_tokens": None}})
        )
        is None
    )


def test_runtime_profile_reads_actual_config_and_rejects_changed_cache_budget():
    expected = rollout.NATIVE_ROLLOUT_PROFILE
    config = Namespace(
        scheduler_config=Namespace(
            scheduler_cls=rollout.ExactPrefillScheduler,
            max_num_seqs=64,
            max_num_batched_tokens=16384,
            async_scheduling=False,
        ),
        model_config=Namespace(max_model_len=1040, dtype=torch.bfloat16),
        cache_config=Namespace(
            kv_cache_memory_bytes=4 << 30,
            enable_prefix_caching=False,
            mamba_ssm_cache_dtype="float32",
        ),
        use_v2_model_runner=False,
    )
    assert rollout.runtime_profile(config) == expected
    config.cache_config.kv_cache_memory_bytes *= 2
    with pytest.raises(ValueError, match="changed the explicit"):
        rollout.runtime_profile(config)


def test_implementation_readiness_does_not_claim_scheduler_or_model_acceptance():
    capabilities = rollout.capabilities()
    assert capabilities["rollout"] and capabilities["native_scheduler"]
    assert not capabilities["scheduler_batch_invariance_verified"]
    assert not capabilities["full_model_l2_verified"]


def test_engine_arguments_are_accepted_by_the_installed_native_api(monkeypatch):
    import huggingface_hub.constants
    from vllm.engine.arg_utils import EngineArgs

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    value = EngineArgs(**rollout.engine_arguments(request()["identity"], "ExplicitTestModel"))
    assert value.tensor_parallel_size == 4
    assert value.kv_cache_memory_bytes == 4 << 30
    assert value.scheduler_cls.endswith(".ExactPrefillScheduler")
    assert value.attention_config.backend.name == "CUSTOM"
    assert value.compilation_config.mode == 0


def trace_for_cpu_bookkeeping():
    value = object.__new__(rollout.RolloutTrace)
    value.request = request(frozen=True)
    value.worker = Namespace(model_runner=Namespace(input_batch=Namespace(req_ids=["r2", "r0"])))
    value.step = {"r0": (0, 7, 1), "r2": (2, 7, 1)}
    value.responses = torch.tensor(value.request["response_tokens"], dtype=torch.int64)
    value.progress = {"r0": 7, "r2": 7}
    value.scheduled = Namespace(total_num_scheduled_tokens=2)
    value.schedule = []
    return value


def test_trace_uses_actual_native_request_order_and_absolute_positions():
    value = trace_for_cpu_bookkeeping()
    value.begin_forward(torch.tensor([3, 1]), torch.tensor([7, 7]))
    assert value.current == [(2, 0, 1, 7), (0, 1, 2, 7)]
    assert value.progress == {"r0": 8, "r2": 8}
    assert value.schedule == [
        {"samples": [2, 0], "computed_before": [7, 7], "token_counts": [1, 1]}
    ]
    value.step = {"r0": (0, 8, 1), "r2": (2, 8, 1)}
    value.begin_forward(torch.tensor([0, 0]), torch.tensor([8, 8]))
    assert value.progress == {"r0": 9, "r2": 9}


@pytest.mark.parametrize("tokens,positions", [([2, 1], [7, 7]), ([3, 1], [8, 7])])
def test_trace_rejects_wrong_consumed_tokens_or_prediction_offsets(tokens, positions):
    value = trace_for_cpu_bookkeeping()
    with pytest.raises(ValueError):
        value.begin_forward(torch.tensor(tokens), torch.tensor(positions))


@pytest.mark.parametrize("field,replacement", [("sha256", "b" * 64), ("path", "/other/_C.so")])
def test_rollout_rejects_provider_binary_changed_during_case(monkeypatch, field, replacement):
    value = object.__new__(rollout.RolloutTrace)
    value.provider_binary = {"module": "rl_engine._C", "path": "/actual/_C.so", "sha256": "a" * 64}
    changed = {**value.provider_binary, field: replacement}
    monkeypatch.setattr(rollout, "imported_provider_identity", lambda: changed)
    # This must fail before examining numerical observations or publishing a trace.
    with pytest.raises(RuntimeError, match="binary changed during"):
        value.finish()


def test_rollout_rejects_provider_replaced_between_worker_initialization_and_case(monkeypatch):
    provider = {"module": "rl_engine._C", "path": "/actual/_C.so", "sha256": "a" * 64}
    worker = Namespace(shared_provider_binary=provider)
    monkeypatch.setattr(
        rollout, "imported_provider_identity", lambda: {**provider, "sha256": "b" * 64}
    )
    with pytest.raises(RuntimeError, match="binary changed after"):
        rollout.RolloutTrace(worker, request(frozen=True), "/unused-cpu-test")


@pytest.mark.parametrize("tamper", ["missing", "module", "path", "sha256"])
def test_assemble_rejects_different_binary_in_one_rehashed_rank_trace(tmp_path, tamper):
    value = request(frozen=True)
    provider = {"module": "rl_engine._C", "path": "/actual/_C.so", "sha256": "a" * 64}
    records = []
    for rank in range(4):
        observed = dict(provider)
        if rank == 3:
            if tamper == "missing":
                observed = None
            else:
                observed[tamper] = {
                    "module": "fallback",
                    "path": "/other/_C.so",
                    "sha256": "b" * 64,
                }[tamper]
        path = tmp_path / f"rank-{rank}.pt"
        rollout.save_tensor_artifact(
            path,
            {
                "rank": rank,
                "request_sha256": value["request_sha256"],
                "provider_binary": observed,
            },
        )
        records.append(
            {
                "rank": rank,
                "path": str(path),
                "sha256": rollout.file_digest(path),
                "request_sha256": value["request_sha256"],
            }
        )
    # File SHA is valid: per-rank provenance must still reject the rewritten identity.
    with pytest.raises(ValueError, match="binary evidence|different provider binaries"):
        rollout.assemble_trace(value, records, value["response_tokens"])
