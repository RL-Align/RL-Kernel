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
        "PATH": "/bin",
    }
    env = cases.arm_environment(base, arm)
    assert env == {"VLLM_BATCH_INVARIANT": invariant, "PATH": "/bin"}
    assert base["QWEN_SHARED_PROFILE_RESULT"] == "/run/result.json"


def test_arm_order_alternates_between_rounds():
    assert cases.arm_order("replay", 0) == ["strict", "native"]
    assert cases.arm_order("replay", 1) == ["native", "strict"]
    assert cases.arm_order("rollout", 1)[0] == "native_bi"


def test_stock_actor_uses_vime_spec_and_drops_rl_kernel_provider():
    stock = runtime_argv("/checkpoint", training=True, providers=False, seq_length=2048)
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


def test_engine_refusal_is_unsupported_not_a_crash():
    log = (
        "x\nERROR [multiproc_executor.py:944] RuntimeError: "
        "VLLM batch_invariant mode is not supported for GDN_ATTN.\nmore"
    )
    assert bench.unsupported_reason(log).endswith("not supported for GDN_ATTN.")
    assert bench.unsupported_reason("CUDA out of memory") is None
