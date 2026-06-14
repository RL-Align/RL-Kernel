# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json

import pytest

from rl_engine.benchmarks.logprob_cross_engine import (
    LogprobBenchmarkFixture,
    LogprobCrossBenchmarkConfig,
    build_synthetic_rollout_fixture,
    load_rollout_fixture,
    main,
    run_cross_engine_benchmark,
    score_training_logprobs,
)


def test_synthetic_cross_engine_smoke_passes_and_writes_outputs(tmp_path):
    config = LogprobCrossBenchmarkConfig(
        rollout_engine="synthetic",
        training_engine="torch",
        model="synthetic",
        output_dir=tmp_path / "smoke",
        device="cpu",
        dtype="float32",
        seed=123,
        num_prompts=2,
        prompt_len=4,
        max_new_tokens=5,
        vocab_size=32,
        hidden_size=16,
        training_micro_batch_size=1,
    )

    report = run_cross_engine_benchmark(config)

    assert report["status"] == "pass"
    assert report["summary"]["active_tokens"] == 10
    assert report["summary"]["max_abs_error"] == pytest.approx(0.0, abs=1e-7)
    assert report["worst_drift"]["sequence_id"] == "synthetic-0-0"
    assert (tmp_path / "smoke" / "report.json").exists()
    assert (tmp_path / "smoke" / "rollout_fixture.json").exists()
    assert (tmp_path / "smoke" / "token_drifts.jsonl").exists()
    assert "max_abs_error" in (tmp_path / "smoke" / "summary.md").read_text(encoding="utf-8")


def test_fixture_jsonl_ingest_replays_same_tokens(tmp_path):
    base_config = LogprobCrossBenchmarkConfig(
        rollout_engine="synthetic",
        training_engine="torch",
        model="synthetic",
        output_dir=tmp_path / "unused",
        device="cpu",
        dtype="float32",
        seed=9,
        num_prompts=1,
        prompt_len=3,
        max_new_tokens=4,
        vocab_size=24,
        hidden_size=12,
    )
    fixture = build_synthetic_rollout_fixture(base_config)
    jsonl_path = tmp_path / "rollout.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sequence in fixture.sequences:
            handle.write(json.dumps(sequence.to_dict()) + "\n")

    loaded = load_rollout_fixture(jsonl_path)

    assert isinstance(loaded, LogprobBenchmarkFixture)
    assert loaded.sequences[0].prompt_token_ids == fixture.sequences[0].prompt_token_ids
    assert loaded.sequences[0].completion_token_ids == fixture.sequences[0].completion_token_ids

    config = LogprobCrossBenchmarkConfig(
        rollout_engine="fixture",
        training_engine="torch",
        model="synthetic",
        rollout_fixture=jsonl_path,
        output_dir=tmp_path / "ingest",
        device="cpu",
        dtype="float32",
        seed=9,
        vocab_size=24,
        hidden_size=12,
    )
    report = run_cross_engine_benchmark(config)

    assert report["status"] == "pass"
    assert report["summary"]["max_abs_error"] == pytest.approx(0.0, abs=1e-7)


def test_training_replay_batches_ragged_completion_lengths(tmp_path):
    config = LogprobCrossBenchmarkConfig(
        rollout_engine="synthetic",
        training_engine="torch",
        model="synthetic",
        output_dir=tmp_path,
        device="cpu",
        dtype="float32",
        seed=55,
        num_prompts=2,
        prompt_len=3,
        max_new_tokens=5,
        vocab_size=32,
        hidden_size=16,
        training_micro_batch_size=2,
    )
    fixture = build_synthetic_rollout_fixture(config)
    short = fixture.sequences[1]
    fixture = LogprobBenchmarkFixture(
        schema_version=fixture.schema_version,
        created_at=fixture.created_at,
        rollout_engine=fixture.rollout_engine,
        model=fixture.model,
        tokenizer=fixture.tokenizer,
        dtype=fixture.dtype,
        metadata=fixture.metadata,
        sequences=[
            fixture.sequences[0],
            type(short)(
                sequence_id=short.sequence_id,
                prompt_token_ids=short.prompt_token_ids,
                completion_token_ids=short.completion_token_ids[:3],
                rollout_logprobs={"policy": short.rollout_logprobs["policy"][:3]},
                completion_mask=[True, True, True],
                prompt_text=short.prompt_text,
                completion_text=short.completion_text,
                request_id=short.request_id,
                metadata=short.metadata,
            ),
        ],
    )

    scores = score_training_logprobs(fixture, config)

    assert set(scores["policy"]) == {sequence.sequence_id for sequence in fixture.sequences}
    assert len(scores["policy"][fixture.sequences[0].sequence_id]) == 5
    assert len(scores["policy"][fixture.sequences[1].sequence_id]) == 3


def test_reference_channel_is_compared_when_reference_model_is_configured(tmp_path):
    base_config = LogprobCrossBenchmarkConfig(
        rollout_engine="synthetic",
        training_engine="torch",
        model="synthetic",
        output_dir=tmp_path / "unused",
        device="cpu",
        dtype="float32",
        seed=31,
        num_prompts=1,
        prompt_len=3,
        max_new_tokens=4,
        vocab_size=24,
        hidden_size=12,
    )
    fixture = build_synthetic_rollout_fixture(base_config)
    sequence = fixture.sequences[0]
    fixture = LogprobBenchmarkFixture(
        schema_version=fixture.schema_version,
        created_at=fixture.created_at,
        rollout_engine=fixture.rollout_engine,
        model=fixture.model,
        tokenizer=fixture.tokenizer,
        dtype=fixture.dtype,
        metadata=fixture.metadata,
        sequences=[
            type(sequence)(
                sequence_id=sequence.sequence_id,
                prompt_token_ids=sequence.prompt_token_ids,
                completion_token_ids=sequence.completion_token_ids,
                rollout_logprobs={
                    "policy": sequence.rollout_logprobs["policy"],
                    "ref": sequence.rollout_logprobs["policy"],
                },
                completion_mask=sequence.completion_mask,
                prompt_text=sequence.prompt_text,
                completion_text=sequence.completion_text,
                request_id=sequence.request_id,
                metadata=sequence.metadata,
            )
        ],
    )

    skipped_scores = score_training_logprobs(fixture, base_config)
    compared_scores = score_training_logprobs(
        fixture,
        LogprobCrossBenchmarkConfig(
            rollout_engine="synthetic",
            training_engine="torch",
            model="synthetic",
            reference_model="synthetic",
            output_dir=tmp_path / "unused2",
            device="cpu",
            dtype="float32",
            seed=31,
            vocab_size=24,
            hidden_size=12,
        ),
    )

    assert set(skipped_scores) == {"policy"}
    assert set(compared_scores) == {"policy", "ref"}


def test_injected_token_shift_fails_with_actionable_worst_token(tmp_path):
    config = LogprobCrossBenchmarkConfig(
        rollout_engine="synthetic",
        training_engine="torch",
        model="synthetic",
        output_dir=tmp_path / "shifted",
        device="cpu",
        dtype="float32",
        seed=7,
        num_prompts=1,
        prompt_len=4,
        max_new_tokens=5,
        vocab_size=32,
        hidden_size=16,
        inject_mismatch="token_shift",
        max_abs_error=1e-8,
        mean_abs_error=1e-8,
        max_relative_error=1e-8,
        fail_on_drift=False,
    )

    report = run_cross_engine_benchmark(config)

    assert report["status"] == "fail"
    assert report["worst_drift"]["sequence_id"] == "synthetic-0-0"
    assert report["worst_drift"]["region"] == "completion"
    assert "completion_index" in report["worst_drift"]
    assert report["summary"]["max_abs_error"] > 0.0


def test_cli_smoke_returns_success_and_writes_report(tmp_path):
    exit_code = main(
        [
            "--smoke",
            "--output-dir",
            str(tmp_path / "cli"),
            "--no-summary",
        ]
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "cli" / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"


def test_cli_returns_failure_when_drift_exceeds_threshold(tmp_path):
    exit_code = main(
        [
            "--smoke",
            "--output-dir",
            str(tmp_path / "cli-fail"),
            "--inject-mismatch",
            "token_shift",
            "--max-abs-error",
            "1e-8",
            "--mean-abs-error",
            "1e-8",
            "--max-relative-error",
            "1e-8",
            "--no-summary",
        ]
    )

    assert exit_code == 1
    payload = json.loads((tmp_path / "cli-fail" / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
