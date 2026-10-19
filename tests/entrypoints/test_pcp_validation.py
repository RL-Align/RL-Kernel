from copy import deepcopy

from rl_engine.integrations.orchestrators.vime.experiments.dense.run_arm import _rollout_topology
from rl_engine.integrations.orchestrators.vime.experiments.dense.validate_run import (
    _validate_pcp_readbacks,
    _validate_topology,
)


def _case():
    topology = {"rollout_tp": 2, "rollout_cp": 4, "rollout_engines": 1}
    arm = {"attention_case": "R/R"}
    readbacks = []
    for cp_rank in range(4):
        for _ in range(2):
            readbacks.append(
                {
                    "framework": "vllm",
                    "target": "rollout",
                    "operators": {
                        "attention": {
                            "call_count": 12,
                            "provenance": {
                                "execution": {
                                    "cp_world_size": 4,
                                    "cp_rank": cp_rank,
                                    "tp_world_size": 2,
                                    "kv_storage": "token_sharded",
                                    "attention_queries": "disjoint_cp_partitions",
                                    "attention_merge": "rank_ordered_output_gather_no_reduction",
                                    "fallback": False,
                                }
                            },
                        }
                    },
                }
            )
    return readbacks, topology, arm


def test_actual_pcp_worker_coverage_is_required():
    records, topology, arm = _case()
    assert _validate_pcp_readbacks(records, topology, arm)["passed"]
    assert not _validate_pcp_readbacks(records[:-1], topology, arm)["passed"]
    assert not _validate_pcp_readbacks([], topology, arm)["passed"]


def test_replicated_cache_or_wrong_topology_is_rejected():
    records, topology, arm = _case()
    for key, value in [("kv_storage", "replicated"), ("cp_world_size", 1), ("fallback", True)]:
        changed = deepcopy(records)
        changed[0]["operators"]["attention"]["provenance"]["execution"][key] = value
        assert not _validate_pcp_readbacks(changed, topology, arm)["passed"]


def test_cp1_needs_no_pcp_records():
    assert _validate_pcp_readbacks([], {"rollout_cp": 1}, {})["passed"]


def test_tp1_offload_contract_holds_for_every_rollout_cp():
    for cp in (1, 2, 4, 8):
        topology = _rollout_topology(1, cp, tensor_parallel_size=1, context_parallel_size=8)
        assert topology["offload_train"]
        assert _validate_topology(topology) == []
