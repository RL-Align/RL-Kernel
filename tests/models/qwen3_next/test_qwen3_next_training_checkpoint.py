# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU serialization/protocol regressions; these do not certify native GPU training."""

import json
import random
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))
import qwen3_next_optimizer_checkpoint as optimizer_checkpoint  # noqa: E402
import qwen3_next_training_checkpoint as checkpoint  # noqa: E402
from qwen3_next_checkpoint import publish_checkpoint  # noqa: E402

from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_training import (  # noqa: E402
    capture_rng_state,
    state_digest,
)
from rl_engine.validation.models.qwen3_next_rl_fixture import PolicyFixture  # noqa: E402

IDENTITY = {"source": "pinned", "provider": "strict", "checkpoint": "official", "tp": 4}


class ProtocolModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.bfloat16).reshape(2, 4))
        self.register_buffer("counter", torch.tensor(12, dtype=torch.int64))

    def state_dict_for_save_checkpoint(self):
        return self.state_dict()


class ProtocolOptimizer:
    """Tiny CPU stand-in only for exercising serialization and corruption paths."""

    def __init__(self, model):
        self.model = model
        self.calls = []
        self.model_param_group_index_map = {model.weight: (0, 0)}
        self.gbuf_ranges = [
            {
                (torch.bfloat16, torch.float32): [
                    {
                        "param_map": {
                            model.weight: {
                                "param": SimpleNamespace(start=0, end=8),
                                "gbuf_local": SimpleNamespace(start=0, end=8),
                            }
                        }
                    }
                ]
            }
        ]
        self.buffers = [SimpleNamespace(buckets=[SimpleNamespace(grad_data=torch.empty(8))])]
        self.optimizer = SimpleNamespace(
            cpu_optimizer_cls=torch.optim.AdamW, gpu_optimizer_cls=None
        )
        self.states = {
            "param": model.weight.detach().float().clone(),
            "exp_avg": torch.full_like(model.weight, 0.125, dtype=torch.float32),
            "exp_avg_sq": torch.full_like(model.weight, 0.0625, dtype=torch.float32),
            "step": torch.tensor(2.0),
        }
        self.step = 2
        self.corrupt_load = False

    def _get_main_param_and_optimizer_states(self, parameter):
        assert parameter is self.model.weight
        return self.states

    def state_dict(self):
        return {"step": self.step, "cpu_optimizer_cls": torch.optim.AdamW}

    def load_state_dict(self, header):
        self.calls.append("header")
        assert header["cpu_optimizer_cls"] is torch.optim.AdamW
        self.step = header["step"]
        # Emulate native HybridDeviceOptimizer's RNG-consuming dummy initialization.
        with torch.no_grad():
            self.model.weight.copy_(torch.randn_like(self.model.weight))

    def save_parameter_state(self, path):
        raise AssertionError("Whole-bucket packing must never be called")

    def load_parameter_state(self, path):
        raise AssertionError("Whole-bucket unpacking must never be called")

    def _set_main_param_and_optimizer_states(self, parameter, states):
        assert parameter is self.model.weight
        self.calls.append("native_set")
        assert states is self.states
        if self.corrupt_load:
            self.states["exp_avg"][0, 0] = -0.125


class ProtocolScheduler:
    def __init__(self):
        self.num_steps = 8

    def state_dict(self):
        return {"num_steps": self.num_steps, "lr": 1e-6}

    def load_state_dict(self, state):
        self.num_steps = state["num_steps"]


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    model = ProtocolModel()
    optimizer = ProtocolOptimizer(model)
    scheduler = ProtocolScheduler()
    fixture = PolicyFixture.freeze(
        torch.arange(320, dtype=torch.int64).reshape(4, 80),
        torch.full((4, 16), -2.0, dtype=torch.float32),
        baseline_identity="real-worker-baseline-required-in-production",
    )
    return [model], optimizer, scheduler, fixture


def save_rank(tmp_path, runtime):
    model, optimizer, scheduler, fixture = runtime
    staging = tmp_path / "step-2.partial"
    staging.mkdir()
    record = checkpoint._save_rank(
        model,
        optimizer,
        scheduler,
        fixture,
        staging,
        leaves=[optimizer],
        identity=IDENTITY,
        step=2,
        sample_position=8,
        tracker=None,
        native_args=SimpleNamespace(consumed_train_samples=8),
        update_evidence={"native_step": 2, "gradient_norm": 1.0},
    )
    return staging, record


def load_rank(staging, runtime):
    model, optimizer, scheduler, fixture = runtime
    return checkpoint._load_rank(
        model,
        optimizer,
        scheduler,
        fixture,
        staging,
        leaves=[optimizer],
        identity=IDENTITY,
        complete={"step": 2, "sample_position": 8},
        tracker=None,
        native_args=SimpleNamespace(consumed_train_samples=0),
    )


def test_bounded_load_calls_hdo_sync(tmp_path, runtime, monkeypatch):
    pytest.importorskip("megatron")
    staging, _ = save_rank(tmp_path, runtime)
    model, optimizer, _, _ = runtime
    calls = []

    class FakeHDO:
        def _sync_hdo_state_to_sub_optimizers(self):
            calls.append("sync")

    optimizer.optimizer = FakeHDO()
    monkeypatch.setattr(
        "megatron.core.optimizer.cpu_offloading.hybrid_optimizer.HybridDeviceOptimizer",
        FakeHDO,
    )
    optimizer_checkpoint.load_bounded_parameter_state(model, [optimizer], staging / "rank-0")
    assert calls == ["sync"]


def test_load_rank_fails_when_hdo_sync_skipped(tmp_path, runtime, monkeypatch):
    staging, _ = save_rank(tmp_path, runtime)
    model, optimizer, scheduler, fixture = runtime

    def load_without_sync(model, leaves, rank_dir):
        rank_dir = Path(rank_dir)
        touched = []
        index = json.loads((rank_dir / "optimizer-index.json").read_text())
        current = optimizer_checkpoint.native_parameter_layout(model, leaves)
        extras = torch.load(rank_dir / "optimizer-extra.pt", map_location="cpu", weights_only=True)
        with torch.no_grad():
            for name, record in current.items():
                entry = index["parameters"][name]
                states = optimizer_checkpoint.native_parameter_states(record)
                for block in entry["blocks"]:
                    start, end = block["start"], block["end"]
                    payload = torch.load(
                        rank_dir / block["path"], map_location="cpu", weights_only=True
                    )
                    for key in optimizer_checkpoint.PARAMETER_KEYS:
                        states[key].reshape(-1)[start:end].copy_(payload[key])
                for key, value in extras[name].items():
                    if isinstance(states[key], torch.Tensor):
                        states[key].copy_(value)
                    else:
                        states[key] = value
                record["leaf"]._set_main_param_and_optimizer_states(record["parameter"], states)
                if record["leaf"] not in touched:
                    touched.append(record["leaf"])

    monkeypatch.setattr(checkpoint, "load_bounded_parameter_state", load_without_sync)
    monkeypatch.setattr(
        checkpoint,
        "verify_hdo_optimizer_master_aliases",
        lambda model, leaves: (_ for _ in ()).throw(
            ValueError("HDO FP32 master copy diverged from checkpoint-visible master weights")
        ),
    )
    with pytest.raises(ValueError, match="FP32 master copy diverged"):
        checkpoint._load_rank(
            model,
            optimizer,
            scheduler,
            fixture,
            staging,
            leaves=[optimizer],
            identity=IDENTITY,
            complete={"step": 2, "sample_position": 8},
            tracker=None,
            native_args=SimpleNamespace(consumed_train_samples=0),
        )


def test_rank_roundtrip_uses_parameter_state_and_restores_rng_last(tmp_path, runtime):
    model, optimizer, scheduler, _ = runtime
    original_rng = state_digest(capture_rng_state())
    original_model = state_digest(model[0].state_dict())
    staging, record = save_rank(tmp_path, runtime)
    with torch.no_grad():
        model[0].weight.fill_(-3)
        model[0].counter.fill_(42)
    optimizer.states["param"].fill_(5)
    optimizer.states["exp_avg"].fill_(0)
    optimizer.step = 100
    scheduler.num_steps = 400
    random.random()
    np.random.rand()
    torch.rand(3)
    assert load_rank(staging, runtime) == record
    assert optimizer.calls == ["header", "native_set"]
    assert optimizer.step == 2 and scheduler.num_steps == 8
    assert state_digest(model[0].state_dict()) == original_model
    assert state_digest(capture_rng_state()) == original_rng


def test_all_small_state_files_are_weights_only_safe(tmp_path, runtime):
    staging, _ = save_rank(tmp_path, runtime)
    for path in (staging / "rank-0").glob("*.pt"):
        assert torch.load(path, map_location="cpu", weights_only=True) is not None


def test_native_parameter_load_corruption_is_rejected(tmp_path, runtime):
    staging, _ = save_rank(tmp_path, runtime)
    runtime[1].corrupt_load = True
    with pytest.raises(ValueError, match="master weights or Adam moments"):
        load_rank(staging, runtime)


def test_header_only_optimizer_is_not_a_complete_checkpoint(runtime):
    del runtime[1].states["exp_avg_sq"]
    with pytest.raises(ValueError, match="Adam moments are missing"):
        checkpoint._optimizer_parameter_records(runtime[0], [runtime[1]])


def test_optimizer_must_cover_each_model_parameter_exactly_once(runtime):
    model, optimizer, _, _ = runtime
    with pytest.raises(ValueError, match="partition model parameters"):
        checkpoint._optimizer_parameter_records(model, [optimizer, optimizer])
    optimizer.model_param_group_index_map = {}
    with pytest.raises(ValueError, match="absent from optimizer groups"):
        checkpoint._optimizer_parameter_records(model, [optimizer])


def test_optimizer_master_dtype_is_never_silently_cast(runtime):
    runtime[1].states["param"] = runtime[1].states["param"].bfloat16()
    with pytest.raises(ValueError, match="Invalid fully offloaded native param"):
        checkpoint._optimizer_parameter_records(runtime[0], [runtime[1]])


def test_native_class_header_uses_only_running_optimizer_allowlist(runtime):
    allowed = checkpoint._native_class_allowlist([runtime[1]])
    header = {"group": {"cls": torch.optim.AdamW, "options": (1, True)}}
    encoded = checkpoint._encode_header(header, allowed)
    assert checkpoint._decode_header(encoded, allowed) == header
    with pytest.raises(TypeError, match="Unpinned class"):
        checkpoint._encode_header({"cls": ProtocolOptimizer}, allowed)
    with pytest.raises(ValueError, match="absent from the running"):
        checkpoint._decode_header({checkpoint.CLASS_TAG: "untrusted.arbitrary.Class"}, allowed)


def test_saved_fixture_identity_cannot_change(tmp_path, runtime):
    staging, _ = save_rank(tmp_path, runtime)
    runtime[3].tokens[0, 0] += 1
    with pytest.raises(ValueError, match="Frozen policy fixture"):
        checkpoint.load_training_checkpoint(
            None, *runtime, staging, identity=IDENTITY, tracker=None
        )


def test_model_schema_rejects_dtype_change(tmp_path, runtime):
    staging, record = save_rank(tmp_path, runtime)
    path = staging / "rank-0/model-0.pt"
    value = torch.load(path, weights_only=True)
    value["weight"] = value["weight"].float()
    torch.save(value, path)
    record["model_sha256"][0] = state_digest(value)
    (staging / "rank-0/progress.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="dtypes, shapes or state keys"):
        load_rank(staging, runtime)


def test_numpy_rng_roundtrip_preserves_all_bits(runtime):
    state = capture_rng_state()
    assert state_digest(checkpoint._decode_rng(checkpoint._encode_rng(state))) == state_digest(
        state
    )


def test_shared_storage_disk_estimate_counts_padding_once():
    buffer = torch.arange(1024, dtype=torch.bfloat16)
    assert checkpoint._tensor_bytes({"one": buffer[:8], "two": buffer[8:16]}) == 2048


def test_bounded_blocks_do_not_serialize_underlying_bucket(tmp_path, runtime, monkeypatch):
    import qwen3_next_optimizer_checkpoint as bounded

    monkeypatch.setattr(bounded, "BLOCK_BYTES", 48)
    optimizer = runtime[1]
    for key in bounded.PARAMETER_KEYS:
        backing = torch.arange(1024 * 1024, dtype=torch.float32)
        optimizer.states[key] = backing[100:108].view(2, 4)
    staging, _ = save_rank(tmp_path, runtime)
    blocks = sorted((staging / "rank-0/optimizer-state").glob("*.pt"))
    assert len(blocks) == 2
    for path in blocks:
        assert path.stat().st_size < 4096
        payload = torch.load(path, weights_only=True)
        assert sum(value.numel() * value.element_size() for value in payload.values()) == 48
    load_rank(staging, runtime)


@pytest.mark.parametrize("change", ["gap", "overlap", "truncated"])
def test_bounded_block_ranges_must_cover_parameter_once(tmp_path, runtime, monkeypatch, change):
    import qwen3_next_optimizer_checkpoint as bounded

    monkeypatch.setattr(bounded, "BLOCK_BYTES", 48)
    staging, _ = save_rank(tmp_path, runtime)
    path = staging / "rank-0/optimizer-index.json"
    index = json.loads(path.read_text())
    blocks = index["parameters"]["chunk0.weight"]["blocks"]
    if change == "gap":
        blocks[1]["start"] += 1
    elif change == "overlap":
        blocks[1]["start"] -= 1
    else:
        blocks.pop()
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="cover each parameter exactly once|complete parameter"):
        load_rank(staging, runtime)


def test_native_dp_range_cannot_describe_partial_parameter(runtime):
    optimizer = runtime[1]
    ranges = next(iter(optimizer.gbuf_ranges[0].values()))[0]["param_map"][runtime[0][0].weight]
    ranges["param"].start = 1
    with pytest.raises(ValueError, match="cover each complete parameter"):
        checkpoint._optimizer_parameter_records(runtime[0], [optimizer])


def test_state_identity_includes_native_consumed_samples_and_rng(runtime, monkeypatch):
    monkeypatch.setattr(checkpoint, "_native_leaves", lambda *args: [runtime[1]])
    args = SimpleNamespace(consumed_train_samples=8)
    first = checkpoint.training_state_identity(
        args, *runtime, step=2, sample_position=8, identity=IDENTITY, tracker=None
    )
    args.consumed_train_samples = 12
    second = checkpoint.training_state_identity(
        args, *runtime, step=2, sample_position=8, identity=IDENTITY, tracker=None
    )
    assert first["state_sha256"] != second["state_sha256"]
    args.consumed_train_samples = 8
    torch.rand(1)
    third = checkpoint.training_state_identity(
        args, *runtime, step=2, sample_position=8, identity=IDENTITY, tracker=None
    )
    assert first["rng_sha256"] != third["rng_sha256"]


def test_native_required_files_are_bound_to_checked_manifest(tmp_path, runtime, monkeypatch):
    model, optimizer, scheduler, fixture = runtime
    staging = tmp_path / "all.partial"
    staging.mkdir()
    for rank in range(4):
        monkeypatch.setattr(checkpoint.dist, "get_rank", lambda rank=rank: rank)
        checkpoint._save_rank(
            model,
            optimizer,
            scheduler,
            fixture,
            staging,
            leaves=[optimizer],
            identity=IDENTITY,
            step=2,
            sample_position=8,
            tracker=None,
            native_args=SimpleNamespace(consumed_train_samples=8),
            update_evidence={"native_step": 2, "gradient_norm": 1.0},
        )
    complete = tmp_path / "all"
    publish_checkpoint(staging, complete, world_size=4, identity=IDENTITY)
    assert checkpoint._validate_native_checkpoint(complete, IDENTITY, 1)["step"] == 2
    with pytest.raises(ValueError, match="omits or substitutes"):
        checkpoint._validate_native_checkpoint(complete, IDENTITY, 2)
    (complete / "rank-3/optimizer-state/p00000-b00000.pt").write_bytes(b"corrupt")
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint, "_all_rank_call", lambda action, label: action())
    monkeypatch.setattr(checkpoint, "_rank_zero_call", lambda action, label: action())
    monkeypatch.setattr(checkpoint, "_native_leaves", lambda *args: [optimizer])
    optimizer.calls.clear()
    with pytest.raises(ValueError, match="checksum|size mismatch"):
        checkpoint.load_training_checkpoint(
            None, *runtime, complete, identity=IDENTITY, tracker=None
        )
    assert not optimizer.calls


def _serial_worker(rank, root, fail_owner):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=f"file://{root}/init-{fail_owner}",
        timeout=timedelta(seconds=30),
    )
    try:

        def action():
            lock = Path(root) / f"lock-{fail_owner}"
            with lock.open("x"):
                path = Path(root) / f"visited-{fail_owner}"
                with path.open("a") as stream:
                    stream.write(f"{rank}\n")
                if rank == fail_owner:
                    raise ValueError("simulated native serialization failure")
            lock.unlink()
            return rank

        try:
            result = checkpoint._serial_rank_call(action, "protocol test")
        except RuntimeError as exc:
            result = str(exc)
        (Path(root) / f"result-{fail_owner}-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_owner", [-1, 2])
def test_real_four_process_serial_protocol_propagates_failures(tmp_path, fail_owner):
    mp.spawn(_serial_worker, args=(str(tmp_path), fail_owner), nprocs=4, join=True)
    visited = (tmp_path / f"visited-{fail_owner}").read_text().splitlines()
    assert visited == (["0", "1", "2", "3"] if fail_owner == -1 else ["0", "1", "2"])
    for rank in range(4):
        result = json.loads((tmp_path / f"result-{fail_owner}-{rank}.json").read_text())
        if fail_owner == -1:
            assert result == rank
        else:
            assert "simulated native serialization failure" in result


def _wire_sharded_hdo(optimizer, monkeypatch, *, break_alias=False, diverge=False):
    """Emulate Megatron: the HDO is keyed by the DP shard, the getter returns the shard's master."""
    pytest.importorskip("megatron")
    model_weight = optimizer.model.weight
    shard = torch.nn.Parameter(model_weight.detach().clone().reshape(-1))
    master = model_weight.detach().float().clone().reshape(-1)
    fp32_inner = master.clone()
    if diverge:
        fp32_inner[0] += 1.0
    hdo_state = {"master_param": master, "exp_avg": torch.zeros(8), "exp_avg_sq": torch.zeros(8)}

    class FakeHDO:
        pass

    hdo = FakeHDO()
    hdo.param_groups = [{"params": [shard]}]
    hdo.state = {shard: hdo_state}
    hdo.param_to_fp32_param = {shard: fp32_inner}
    hdo.param_to_inner_param = {shard: fp32_inner}
    cpu_state = {fp32_inner: {} if break_alias else hdo_state}
    hdo.cpu_optimizers = [SimpleNamespace(param_groups=[{"params": [fp32_inner]}], state=cpu_state)]
    monkeypatch.setattr(
        "megatron.core.optimizer.cpu_offloading.hybrid_optimizer.HybridDeviceOptimizer", FakeHDO
    )
    monkeypatch.setattr(
        optimizer_checkpoint,
        "_hdo_param_generator",
        lambda cpu_optimizer: [p for g in cpu_optimizer.param_groups for p in g["params"]],
    )
    optimizer.optimizer = hdo
    optimizer.states["param"] = master  # the real getter returns state["master_param"]
    return hdo


def test_hdo_alias_check_uses_the_sharded_parameter_not_the_model_parameter(runtime, monkeypatch):
    model, optimizer, _, _ = runtime
    hdo = _wire_sharded_hdo(optimizer, monkeypatch)
    assert model[0].weight not in hdo.param_to_fp32_param  # keyed by the shard, as in Megatron
    optimizer_checkpoint.verify_hdo_optimizer_master_aliases(model, [optimizer])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"break_alias": True}, "not aliased"),
        ({"diverge": True}, "diverged"),
    ],
)
def test_hdo_alias_check_rejects_broken_wiring(runtime, monkeypatch, kwargs, message):
    model, optimizer, _, _ = runtime
    _wire_sharded_hdo(optimizer, monkeypatch, **kwargs)
    with pytest.raises(ValueError, match=message):
        optimizer_checkpoint.verify_hdo_optimizer_master_aliases(model, [optimizer])
