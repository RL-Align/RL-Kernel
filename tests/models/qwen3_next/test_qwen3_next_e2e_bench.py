# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks of the end-to-end speed test: inputs, arm environments and the report."""

import json
from types import SimpleNamespace

import pytest

from benchmarks.models.qwen3_next import e2e_bench_cases as cases
from benchmarks.models.qwen3_next import qwen3_next_e2e_bench as bench
from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime_runtime import (
    SHARED_SPEC,
    STOCK_SPEC,
    runtime_argv,
)


def _option(argv, name):
    index = argv.index(f"--{name}")
    return argv[index + 1]


def test_workloads_fit_their_engine_length():
    for spec in cases.WORKLOADS.values():
        for phase in ("rollout", "replay"):
            for prompt, _, response in spec[phase]:
                assert prompt + response <= spec["max_model_len"]
        prompt, batch, response = spec["train"]
        assert batch == 4 and prompt + response <= spec["max_model_len"]
    assert cases.WORKLOADS["certified"]["max_model_len"] == 1040


def test_first_prompt_row_is_the_same_for_every_batch_size():
    single = cases.prompts((64, 1, 16))
    batch = cases.prompts((64, 4, 16))
    assert single[0] == batch[0] and len(batch) == 4 and len(set(map(tuple, batch))) == 4
    assert all(1000 <= token < 150000 for row in batch for token in row)


def test_response_positions_predict_each_response_token():
    shape = (8, 3, 4)
    rows, targets = cases.response_positions(shape)
    flat = [token for row in cases.sequences(shape) for token in row]
    assert len(rows) == len(targets) == 12
    # Logits at packed row i predict token i + 1 of the same sequence.
    assert [flat[row + 1] for row in rows] == targets
    assert rows[:4] == [7, 8, 9, 10] and rows[4] == 12 + 7


def test_strict_arm_keeps_the_certified_environment():
    base = {
        "VLLM_BATCH_INVARIANT": "1",
        "QWEN_SHARED_PROFILE_RESULT": "/run/result.json",
        "TRITON_F32_DEFAULT": "ieee",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }
    assert cases.arm_environment(base, "strict") == base
    with pytest.raises(ValueError, match="certified"):
        cases.arm_environment({**base, "VLLM_BATCH_INVARIANT": "0"}, "strict")


@pytest.mark.parametrize("arm, invariant", [("native_graph", "0"), ("native_bi", "1")])
def test_native_arms_drop_the_strict_runtime(arm, invariant):
    base = {
        "VLLM_BATCH_INVARIANT": "1",
        "QWEN_SHARED_PROFILE_RESULT": "/run/result.json",
        "TRITON_F32_DEFAULT": "ieee",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PATH": "/bin",
    }
    env = cases.arm_environment(base, arm)
    assert env == {"VLLM_BATCH_INVARIANT": invariant, "PATH": "/bin"}
    assert base["QWEN_SHARED_PROFILE_RESULT"] == "/run/result.json"


def test_stock_vime_keeps_megatron_connection_limit():
    """Stock Megatron refuses TP > 1 without CUDA_DEVICE_MAX_CONNECTIONS=1; stock vLLM runs
    at the CUDA default."""
    base = {"VLLM_BATCH_INVARIANT": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "PATH": "/bin"}
    assert cases.arm_environment(base, "native")["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in cases.arm_environment(base, "native_eager")


def test_arm_order_alternates_between_rounds():
    assert cases.arm_order("replay", 0) == ["strict", "native"]
    assert cases.arm_order("replay", 1) == ["native", "strict"]
    assert cases.arm_order("rollout", 1)[0] == "native_bi"


def test_stock_actor_uses_vime_spec_and_drops_rl_kernel_provider():
    stock = runtime_argv("/checkpoint", training=True, providers=False, seq_length=2048, tp=2, ep=4)
    shared = runtime_argv("/checkpoint", training=True)
    assert stock[stock.index("--spec") + 1 :][:2] == list(STOCK_SPEC)
    assert shared[shared.index("--spec") + 1 :][:2] == list(SHARED_SPEC)
    assert "--custom-model-provider-path" not in stock
    assert "--custom-model-provider-path" in shared
    assert "--moe-permute-fusion" in stock and "--moe-permute-fusion" not in shared
    assert _option(stock, "seq-length") == "2048" and _option(shared, "seq-length") == "1040"
    with pytest.raises(ValueError, match="1040"):
        runtime_argv("/checkpoint", training=False, seq_length=512)


def test_rollout_metrics_split_prefill_and_decode():
    metrics = cases.rollout_metrics((100, 2, 11), [0.5, 0.4, 0.6], [2.5, 2.5, 2.4])
    assert metrics["prefill_s"] == 0.5 and metrics["total_s"] == 2.5
    assert metrics["prefill_tok_per_s"] == pytest.approx(400)
    assert metrics["decode_tok_per_s"] == pytest.approx(2 * 10 / 2.0)


def _rollout_record(prefill, total, tokens="a", deterministic=True):
    return {
        "p64-b1-r16": {
            "prefill_values_s": prefill,
            "total_values_s": total,
            "same_tokens_across_repeats": deterministic,
            "row0_equals_batch1": None,
            "tokens_sha256": tokens,
        }
    }


def test_merge_rounds_pools_values_and_compares_tokens_across_processes():
    pooled = cases.merge_rounds([_rollout_record([1.0], [2.0]), _rollout_record([3.0], [4.0])])
    entry = pooled["p64-b1-r16"]
    assert entry["prefill_values_s"] == [1.0, 3.0] and entry["total_values_s"] == [2.0, 4.0]
    assert entry["checks"]["same_tokens_across_rounds"] == [True]
    changed = cases.merge_rounds(
        [_rollout_record([1.0], [2.0]), _rollout_record([1.0], [2.0], "b")]
    )
    assert changed["p64-b1-r16"]["checks"]["same_tokens_across_rounds"] == [False]
    single = cases.merge_rounds([_rollout_record([1.0], [2.0])])
    assert single["p64-b1-r16"]["checks"]["same_tokens_across_rounds"] == [None]


def test_report_summarizes_ratios_and_failures():
    raw = {
        "rollout": {
            "strict": {
                "certified": [{"engine_arguments": {}, "shapes": _rollout_record([1], [4])}]
            },
            "native_graph": {
                "certified": [
                    {"engine_arguments": {}, "shapes": _rollout_record([1], [2], "x", False)}
                ]
            },
            "native_bi": {"certified": [{"error": "worker exited with 1; see logs/x.log"}]},
            "native_eager": {
                "certified": [
                    {"unsupported": "VLLM batch_invariant mode is not supported for GDN_ATTN."}
                ]
            },
        },
        "train": {
            arm: {
                "certified": [
                    {
                        "shapes": {
                            "p64-b4-r16": {
                                "step_values_s": [seconds, seconds],
                                "optimizer_values_s": [1.0, 1.0],
                            }
                        }
                    }
                ]
            }
            for arm, seconds in (("strict", 6.0), ("native", 3.0))
        },
    }
    manifest = {
        "source_commit": "c" * 40,
        "source_tree": "t" * 40,
        "run_id": "run",
        "topology": {"tp": 4, "pp": 1, "cp": 1, "ep": 1},
    }
    args = SimpleNamespace(repeats=1, rounds=1, train_rounds=1, workloads=["certified"])
    report = bench.build_report(manifest, {"id": 1}, raw, {"gpus": ["B200"]}, args)
    json.dumps(report)
    rollout = report["phases"]["rollout"]
    assert rollout["strict_over_native"] == {"native_graph": {"certified": {"p64-b1-r16": 2.0}}}
    assert rollout["failures"][0]["arm"] == "native_bi"
    assert rollout["unsupported"] == [
        {
            "arm": "native_eager",
            "workload": "certified",
            "reasons": ["VLLM batch_invariant mode is not supported for GDN_ATTN."],
        }
    ]
    assert "native_eager" not in rollout["strict_over_native"]
    native = rollout["results"]["native_graph"]["certified"]["p64-b1-r16"]
    assert native["checks"]["same_tokens_across_repeats"] is False
    assert native["checks"]["row0_equals_batch1"] is None
    train = report["phases"]["train"]
    assert train["strict_over_native"]["native"]["certified"]["p64-b4-r16"] == 2.0
    assert train["results"]["strict"]["certified"]["p64-b4-r16"][
        "forward_backward_median_s"
    ] == pytest.approx(5.0)


def test_cli_requires_run_and_output():
    with pytest.raises(SystemExit):
        bench.parse_args([])
    with pytest.raises(SystemExit):
        bench.parse_args(["--worker", "rollout", "--arm", "strict", "--workload", "certified"])
    args = bench.parse_args(["--run", "r", "--out", "o", "--phases", "rollout,train"])
    assert args.phases == ["rollout", "train"] and args.workloads == ["certified", "long"]
    with pytest.raises(SystemExit):
        bench.parse_args(["--run", "r", "--out", "o", "--phases", "decode"])


def test_worker_gets_the_run_directory_as_an_absolute_path(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, cwd, **_):
        commands.append((command, cwd))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench, "wait_gpus_idle", lambda: True)
    monkeypatch.setattr(bench, "arm_environment", lambda base, arm: base)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "logs").mkdir(parents=True)
    source = tmp_path / "source"
    args = SimpleNamespace(run=bench.Path("runs/l2"), repeats=1, timeout=1)
    manifest = {"python": "python", "source": str(source)}
    bench.run_worker(
        manifest, {}, "bench.py", tmp_path / "out", "rollout", "strict", "long", 0, args
    )
    ((command, cwd),) = commands
    assert cwd == str(source)
    assert _option(command, "run") == str(tmp_path / "runs" / "l2")


def test_busy_gpus_before_a_worker_mean_not_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "wait_gpus_idle", lambda: False)
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: pytest.fail("worker started"))
    args = SimpleNamespace(run=tmp_path, repeats=1, timeout=1)
    manifest = {"python": "python", "source": str(tmp_path)}
    record = bench.run_worker(manifest, {}, "b.py", tmp_path, "rollout", "strict", "long", 0, args)
    assert record == {"error": "GPUs still busy before the worker; not measured"}


@pytest.mark.parametrize("leftover", ["report.json", "workers/rollout-strict-long-round0.json"])
def test_output_directory_of_an_earlier_run_is_refused(tmp_path, monkeypatch, leftover):
    (tmp_path / leftover).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / leftover).write_text("{}")
    monkeypatch.setattr(bench, "load_run", lambda run: pytest.fail("run loaded"))
    args = SimpleNamespace(run=tmp_path, out=tmp_path)
    with pytest.raises(SystemExit, match="already holds"):
        bench.orchestrate(args)


def test_worker_timeout_is_recorded_not_raised(tmp_path, monkeypatch):
    def timed_out(command, **kwargs):
        raise bench.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(bench.subprocess, "run", timed_out)
    monkeypatch.setattr(bench, "wait_gpus_idle", lambda: True)
    monkeypatch.setattr(bench, "arm_environment", lambda base, arm: base)
    (tmp_path / "logs").mkdir()
    args = SimpleNamespace(run=tmp_path, repeats=1, timeout=5)
    manifest = {"python": "python", "source": str(tmp_path)}
    record = bench.run_worker(manifest, {}, "b.py", tmp_path, "train", "strict", "long", 0, args)
    assert record == {
        "error": "worker timed out after 5 s; see logs/train-strict-long-round0.log",
        "gpus_idle_after": True,
    }


def test_engine_refusal_is_unsupported_not_a_crash():
    log = (
        "x\nERROR [multiproc_executor.py:944] RuntimeError: "
        "VLLM batch_invariant mode is not supported for GDN_ATTN.\nmore"
    )
    assert bench.unsupported_reason(log).endswith("not supported for GDN_ATTN.")
    assert bench.unsupported_reason("CUDA out of memory") is None


def test_stock_vime_runs_its_recipe_layout_and_strict_stays_certified():
    stock = runtime_argv("/checkpoint", training=True, providers=False, tp=2, ep=4)
    assert _option(stock, "tensor-model-parallel-size") == "2"
    assert _option(stock, "expert-model-parallel-size") == "4"
    assert _option(stock, "expert-tensor-parallel-size") == "1"
    assert "--sequence-parallel" in stock
    shared = runtime_argv("/checkpoint", training=False, ep=4)
    assert "--sequence-parallel" not in shared
    assert _option(shared, "tensor-model-parallel-size") == "4"
    assert _option(shared, "expert-tensor-parallel-size") == "1"
    with pytest.raises(ValueError, match="TP2 EP4"):
        runtime_argv("/checkpoint", training=False, providers=False)
    with pytest.raises(ValueError, match="training stays EP1"):
        runtime_argv("/checkpoint", training=True, ep=4)
    with pytest.raises(ValueError, match="training stays EP1"):
        runtime_argv("/checkpoint", training=False, tp=2)


def test_actor_layouts_and_unsupported_strict_training():
    assert cases.actor_layout(1, "strict") == {"tp": 4, "sp": False, "ep": 1, "etp": 4, "dp": 1}
    assert cases.actor_layout(4, "strict") == {"tp": 4, "sp": False, "ep": 4, "etp": 1, "dp": 1}
    for ep in (1, 4):
        stock = {"tp": 2, "sp": True, "ep": 4, "etp": 1, "dp": 2}
        assert cases.actor_layout(ep, "native") == stock
        assert cases.actor_unsupported(ep, "replay", "strict") is None
        assert cases.actor_unsupported(ep, "train", "native") is None
    assert cases.actor_unsupported(1, "train", "strict") is None
    assert cases.actor_unsupported(4, "train", "strict") == cases.STRICT_TRAINING_NEEDS_EP1


def test_data_parallel_share_splits_rows_and_fills_an_empty_rank():
    rows = [[1], [2], [3], [4]]
    assert cases.data_parallel_share(rows, 0, 2) == ([[1], [3]], False)
    assert cases.data_parallel_share(rows, 1, 2) == ([[2], [4]], False)
    assert cases.data_parallel_share(rows[:1], 1, 2) == ([[1]], True)
    assert cases.data_parallel_share(rows, 0, 1) == (rows, False)


def test_response_positions_of_a_rank_share_match_its_packed_rows():
    shape = (8, 4, 3)
    full = cases.sequences(shape)
    share, _ = cases.data_parallel_share(full, 1, 2)
    positions, targets = cases.response_positions(shape, share)
    packed = [token for row in share for token in row]
    assert targets == [token for row in share for token in row[8:]]
    assert [packed[p + 1] for p in positions] == targets
    assert cases.response_positions(shape) == cases.response_positions(shape, full)


def test_packed_lengths_pad_to_the_sequence_parallel_split():
    assert cases.packed_lengths([80, 80], 2) == [80, 80]
    assert cases.packed_lengths([80, 81], 2) == [80, 81, 1]
    assert cases.packed_lengths([5], 4) == [5, 3]
    assert cases.packed_lengths([7, 9], 1) == [7, 9]
    for workload in cases.WORKLOADS.values():
        for shape in workload["replay"]:
            for rank in (0, 1):
                rows, _ = cases.data_parallel_share(cases.sequences(shape), rank, 2)
                lengths = [len(row) for row in rows]
                assert cases.packed_lengths(lengths, 2) == lengths  # no shape needs padding


def _plot_report(commit, phases, gpus=("NVIDIA B200",) * 4):
    return {
        "rl_kernel_commit": commit,
        "topology": {"tp": 4, "ep": 1},
        "versions": {"gpus": list(gpus)},
        "phases": {phase: {"results": {}} for phase in phases},
    }


def test_chart_takes_a_rerun_phase_from_a_later_report():
    first = _plot_report("37e04258" + "0" * 32, ("rollout", "replay", "train"))
    rerun = _plot_report("3e06906f" + "0" * 32, ("rollout",))
    rerun["phases"]["rollout"]["results"] = {"native_graph": {}}
    report, commits = cases.combine_reports([first, rerun])
    assert report["phases"]["rollout"]["results"] == {"native_graph": {}}
    assert first["phases"]["rollout"]["results"] == {}
    assert cases.commit_line(commits) == "rollout at 3e06906f; replay and training at 37e04258"
    assert cases.commit_line(cases.combine_reports([first])[1]) == "at 37e04258"
    with pytest.raises(ValueError, match="gpus"):
        cases.combine_reports(
            [first, _plot_report("3e06906f", ("rollout",), gpus=("NVIDIA H100",) * 4)]
        )
    other_vllm = _plot_report("3e06906f", ("rollout",))
    other_vllm["versions"]["vllm"] = "0.31.0"
    with pytest.raises(ValueError, match="vllm"):
        cases.combine_reports([first, other_vllm])
