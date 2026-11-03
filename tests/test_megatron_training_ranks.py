# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P3 (WS2): rank-set integrity over real gate evidence (CPU).

Per-rank evidence comes from the real operator/provider and the real gate, as in
the P1 gate tests. The strict-kernel double is wrapped so that every TP rank
returns the merged full-vocab result, as the strict TP route does. Coordinates
are written as a rank would observe them; one test exchanges them over a real
4-process CPU gloo group.
"""

import copy
import itertools
import queue
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
from p7_doubles.fixtures import make_fixture, vime_request
from p7_doubles.harness import TP_GROUP, make_context, run, strict_observer
from p7_doubles.strict_linear_logp import CpuStrictLinearLogpDouble

import rl_engine.integrations.megatron_training_readback as readback
from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate
from rl_engine.alignment.cross_config.adapters import megatron_training_ranks as ranks
from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError

pytestmark = pytest.mark.unit

TP, CP = 2, 2
FIXTURE = make_fixture(
    "tp2_cp2_ranks",
    sample_lengths=(8, 4),
    real_vocab=60,
    padded_vocab=64,
    tp_world=TP,
    cp_world=CP,
)
TOPOLOGY = ranks.ExpectedRankTopology(tp=TP, cp=CP)


class MergedTpDouble(CpuStrictLinearLogpDouble):
    """Every TP rank returns the merged full-vocab logprob, like the strict TP route."""

    def __init__(self, full_weight: torch.Tensor, real_vocab: int) -> None:
        super().__init__()
        self._full = full_weight
        self._real = real_vocab

    def __call__(self, hidden, lm_head_weight, target_ids, bias=None, **kwargs):
        super().__call__(hidden, lm_head_weight, target_ids, bias, **kwargs)
        logits = hidden.float() @ self._full.float().t()
        logits[:, self._real :] = float("-inf")
        return logits.log_softmax(-1).gather(1, target_ids.view(-1, 1).long()).view(-1)


def _global_rank(tp_rank: int, cp_rank: int) -> int:
    return cp_rank * TP + tp_rank  # Megatron order: TP fastest


def _coordinates(tp_rank: int, cp_rank: int) -> dict[str, Any]:
    rank = _global_rank(tp_rank, cp_rank)
    return {
        "source": "observed",
        "global_rank": rank,
        "world_size": TP * CP,
        "hostname": "node0",
        "device": {"type": "cuda", "index": rank, "uuid": f"GPU-{rank}"},
        "tp": {
            "rank": tp_rank,
            "world_size": TP,
            "ranks": [_global_rank(r, cp_rank) for r in range(TP)],
        },
        "cp": {
            "rank": cp_rank,
            "world_size": CP,
            "ranks": [_global_rank(tp_rank, r) for r in range(CP)],
        },
        "dp": {"rank": 0, "world_size": 1, "ranks": [rank]},
        "pp": {"rank": 0, "world_size": 1, "ranks": [rank]},
    }


def _evidence(monkeypatch, tp_rank: int, cp_rank: int) -> gate.TrainingM10Evidence:
    def coordinates(group: Any) -> dict[str, Any]:
        if group is TP_GROUP:
            return {"group": "present", "rank": tp_rank, "world_size": TP}
        return {"group": None}

    with monkeypatch.context() as patch, strict_observer(patch) as observer:
        patch.setattr(readback, "_group_coordinates", coordinates)
        request = vime_request(FIXTURE, cp_rank=cp_rank, tp_rank=tp_rank)
        request.tensor_parallel_group = TP_GROUP
        wrapper = MergedTpDouble(FIXTURE.lm_head_weight, FIXTURE.real_vocab)
        snapshot = run(observer, request, wrapper)
        return gate.validate_training_call(snapshot, make_context(FIXTURE, cp_rank=cp_rank))


@pytest.fixture(scope="module")
def readbacks() -> list[dict[str, Any]]:
    patcher = pytest.MonkeyPatch()
    try:
        return [
            ranks.RankReadback.from_evidence(
                _evidence(patcher, tp_rank, cp_rank), _coordinates(tp_rank, cp_rank)
            ).to_dict()
            for cp_rank, tp_rank in itertools.product(range(CP), range(TP))
        ]
    finally:
        patcher.undo()


def _load(payloads) -> list[ranks.RankReadback]:
    return [ranks.RankReadback.from_dict(item) for item in payloads]


def _expect(status: str, payloads, field: str, **kwargs) -> T05ContractError:
    with pytest.raises(T05ContractError) as info:
        ranks.validate_rank_set(_load(payloads), TOPOLOGY, **kwargs)
    assert info.value.status == status, str(info.value)
    assert info.value.field_name.startswith(field), str(info.value)
    return info.value


def _expected_rows() -> set[tuple[str, int]]:
    return {(f"s{sample}", position) for sample, position in FIXTURE.row_identity()}


# --------------------------------------------------------------------------- positive paths


def test_complete_rank_set_passes_but_is_not_certified(readbacks):
    report = ranks.validate_rank_set(_load(readbacks), TOPOLOGY, expected_row_keys=_expected_rows())
    assert report.complete and not report.certified
    assert report.vocab_ownership == ((0, 32), (32, 64))
    assert [item["global_rank"] for item in report.ranks] == [0, 2, 1, 3]  # (tp, cp, dp) order
    assert report.active_tokens == int(FIXTURE.active_mask.sum())
    missing = {item["field"] for item in report.missing_evidence}
    assert {"reduction.lse", "reduction.canonical_split_actual"} <= missing
    assert report.fingerprint == ranks.validate_rank_set(_load(readbacks), TOPOLOGY).fingerprint


def test_order_of_arrival_does_not_matter(readbacks):
    forward = ranks.validate_rank_set(_load(readbacks), TOPOLOGY)
    backward = ranks.validate_rank_set(_load(readbacks[::-1]), TOPOLOGY)
    assert forward.fingerprint == backward.fingerprint


def test_tp_peers_carry_the_same_merged_bytes(readbacks):
    by_cp: dict[int, set[str]] = {}
    for item in readbacks:
        cp_rank = item["coordinates"]["cp"]["rank"]
        by_cp.setdefault(cp_rank, set()).add(item["evidence"]["active_bytes_sha256"])
    assert all(len(digests) == 1 for digests in by_cp.values())
    assert len({next(iter(digests)) for digests in by_cp.values()}) == CP


def test_pipeline_last_stage_only():
    topology = ranks.ExpectedRankTopology(tp=1, pp=4)
    coords = _coordinates(0, 0)
    coords["tp"] = {"rank": 0, "world_size": 1, "ranks": [0]}
    coords["cp"] = {"rank": 0, "world_size": 1, "ranks": [0]}
    coords["pp"] = {"rank": 1, "world_size": 4, "ranks": [0, 1, 2, 3]}
    evidence = {"schema_version": gate.TRAINING_M10_EVIDENCE_SCHEMA_VERSION}
    payload = ranks.RankReadback(coordinates=coords, evidence=evidence)
    with pytest.raises(T05ContractError) as info:
        ranks.validate_rank_set([payload], topology)
    assert info.value.status == "IDENTITY_DRIFT"
    assert info.value.field_name == "coordinates.pp"


# --------------------------------------------------------------------------- rank set


def test_missing_rank(readbacks):
    error = _expect("MISSING_RANK", readbacks[:3], "coordinates")
    assert "(1, 1, 0)" in str(error)


def test_empty_rank_set():
    with pytest.raises(T05ContractError) as info:
        ranks.validate_rank_set([], TOPOLOGY)
    assert info.value.status == "MISSING_RANK"


def test_duplicate_global_rank(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[3]["coordinates"]["global_rank"] = 0
    _expect("IDENTITY_DRIFT", payloads, "coordinates")


def test_duplicate_coordinate_from_another_rank(readbacks):
    payloads = copy.deepcopy(readbacks) + [copy.deepcopy(readbacks[0])]
    payloads[-1]["coordinates"]["global_rank"] = 9
    payloads[-1]["coordinates"]["device"]["uuid"] = "GPU-9"
    for dim in ("tp", "cp", "dp", "pp"):
        ranks_list = payloads[-1]["coordinates"][dim]["ranks"]
        position = payloads[-1]["coordinates"][dim]["rank"]
        ranks_list[position] = 9
    _expect("IDENTITY_DRIFT", payloads, "coordinates")


def test_shared_device(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[1]["coordinates"]["device"]["uuid"] = "GPU-0"
    _expect("IDENTITY_DRIFT", payloads, "coordinates.device")


def test_unexpected_world_size(readbacks):
    with pytest.raises(T05ContractError) as info:
        ranks.validate_rank_set(_load(readbacks), ranks.ExpectedRankTopology(tp=4))
    assert info.value.status == "IDENTITY_DRIFT"
    assert info.value.field_name == "coordinates.tp"


def test_group_membership_mismatch(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[0]["coordinates"]["cp"]["ranks"] = [0, 3]  # rank 0 thinks its CP peer is 3
    _expect("IDENTITY_DRIFT", payloads, "coordinates.cp.ranks")


def test_configured_coordinates_never_certify(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[2]["coordinates"]["source"] = "configured"
    _expect("MISSING_PROVENANCE", payloads, "coordinates.source")


def test_evidence_tp_rank_disagrees_with_coordinates(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[1]["evidence"]["tp_rank"] = 0
    _expect("IDENTITY_DRIFT", payloads, "evidence.tp_rank")


@pytest.mark.parametrize(
    ("target", "field"),
    [("rank_readback", "rank_readback.schema_version"), ("evidence", "evidence.schema_version")],
)
def test_schema_mismatch(readbacks, target, field):
    payloads = copy.deepcopy(readbacks)
    if target == "rank_readback":
        payloads[0]["schema_version"] = "other"
        with pytest.raises(T05ContractError) as info:
            _load(payloads)
        assert info.value.status == "SCHEMA_MISMATCH"
        return
    payloads[0]["evidence"]["schema_version"] = "other"
    _expect("SCHEMA_MISMATCH", payloads, field)


# --------------------------------------------------------------------------- vocab ownership


def test_vocab_gap(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[1]["evidence"]["vocab_ownership"] = [34, 64]
    error = _expect("IDENTITY_DRIFT", payloads, "evidence.vocab_ownership")
    assert "gap" in str(error)


def test_vocab_overlap(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[1]["evidence"]["vocab_ownership"] = [30, 64]
    error = _expect("IDENTITY_DRIFT", payloads, "evidence.vocab_ownership")
    assert "overlap" in str(error)


def test_vocab_short_of_padded(readbacks):
    payloads = copy.deepcopy(readbacks)
    for index in (1, 3):
        payloads[index]["evidence"]["vocab_ownership"] = [32, 60]
    _expect("IDENTITY_DRIFT", payloads, "evidence.vocab_ownership")


def test_target_owner_disagreement(readbacks):
    payloads = copy.deepcopy(readbacks)
    owners = payloads[1]["evidence"]["target_owner_ranks"]
    owners[0] = 1 - owners[0]
    _expect("IDENTITY_DRIFT", payloads, "evidence.target_owner_ranks")


def test_merged_output_not_replicated(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[1]["evidence"]["active_bytes_sha256"] = "0" * 64
    _expect("IDENTITY_DRIFT", payloads, "evidence.active_bytes_sha256")


def test_weight_replica_drift(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[2]["evidence"]["weight_shard"]["sha256"] = "f" * 64  # TP0 on CP1
    _expect("PRE_UPDATE_WEIGHT_DRIFT", payloads, "evidence.weight_shard")


@pytest.mark.parametrize("name", ["entrypoint", "build", "canonical_configured", "path"])
def test_rank_invariant_drift(readbacks, name):
    payloads = copy.deepcopy(readbacks)
    payloads[3]["evidence"][name] = "drifted"
    _expect("IDENTITY_DRIFT", payloads, f"evidence.{name}")


# --------------------------------------------------------------------------- global tokens


def test_tp_peers_disagree_on_rows(readbacks):
    payloads = copy.deepcopy(readbacks)
    payloads[1]["evidence"]["row_keys"][0] = ["s9", 0]
    _expect("AMBIGUOUS_GLOBAL_TOKEN_MAPPING", payloads, "evidence.row_keys")


def test_row_held_by_two_cp_ranks(readbacks):
    payloads = copy.deepcopy(readbacks)
    stolen = payloads[0]["evidence"]["row_keys"][0]
    for index in (2, 3):  # both TP peers of CP rank 1
        payloads[index]["evidence"]["row_keys"][0] = list(stolen)
    _expect("AMBIGUOUS_GLOBAL_TOKEN_MAPPING", payloads, "evidence.row_keys")


def test_expected_rows_missing(readbacks):
    expected = _expected_rows() | {("s9", 0)}
    _expect("INCOMPLETE_ARTIFACT", readbacks, "evidence.row_keys", expected_row_keys=expected)


def test_rows_outside_the_failure_batch(readbacks):
    expected = _expected_rows() - {("s0", 0)}
    _expect(
        "AMBIGUOUS_GLOBAL_TOKEN_MAPPING", readbacks, "evidence.row_keys", expected_row_keys=expected
    )


# --------------------------------------------------------------------------- real gloo exchange


def _gloo_worker(rank, world, init_method, payloads, result_queue):
    import traceback

    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
        tp_groups = [dist.new_group([cp * TP + tp for tp in range(TP)]) for cp in range(CP)]
        cp_groups = [dist.new_group([cp * TP + tp for cp in range(CP)]) for tp in range(TP)]
        tp_rank, cp_rank = rank % TP, rank // TP
        coords = ranks.observe_rank_coordinates(
            tp_group=tp_groups[cp_rank], cp_group=cp_groups[tp_rank]
        )
        coords["device"] = {"type": "cuda", "index": rank, "uuid": f"GPU-{rank}"}
        local = ranks.RankReadback(coordinates=coords, evidence=payloads[rank]["evidence"])
        gathered = ranks.gather_rank_readbacks(local)
        report = ranks.validate_rank_set(gathered, TOPOLOGY)
        result_queue.put(
            {"rank": rank, "ok": True, "fingerprint": report.fingerprint, "coords": coords}
        )
    except Exception:  # noqa: BLE001 - report to the parent
        result_queue.put({"rank": rank, "ok": False, "traceback": traceback.format_exc()})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_real_gloo_exchange_matches_local_validation(readbacks):
    ctx = mp.get_context("spawn")
    world = TP * CP
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "gloo_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_gloo_worker, args=(rank, world, init_method, readbacks, result_queue)
            )
            for rank in range(world)
        ]
        for process in processes:
            process.start()
        results = []
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=60))
        except queue.Empty:
            pytest.fail("timed out waiting for gloo workers")
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
    results.sort(key=lambda item: item["rank"])
    for result in results:
        assert result["ok"], result.get("traceback")
    assert len({result["fingerprint"] for result in results}) == 1
    for result in results:
        expected = _coordinates(result["rank"] % TP, result["rank"] // TP)
        observed = result["coords"]
        for dim in ("tp", "cp", "dp", "pp"):
            assert observed[dim] == expected[dim], dim


def test_gather_refuses_a_numeric_backend(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_backend", lambda group=None: "nccl")
    local = ranks.RankReadback(coordinates={}, evidence={})
    with pytest.raises(T05ContractError) as info:
        ranks.gather_rank_readbacks(local)
    assert info.value.status == "UNSUPPORTED_CAPABILITY"
