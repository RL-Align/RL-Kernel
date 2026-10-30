# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P2c: per-step gradient write evidence on a CPU double of Megatron DDP."""

import pytest
import torch
from p7_doubles.megatron_ddp import FakeMegatronDDP, TinyModel

from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError
from rl_engine.integrations.megatron_training_gradients import (
    GradientStepObserver,
    check_base_grad_absent,
    check_emitted_against_producer,
)

pytestmark = pytest.mark.unit

X = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8


def _step(ddp, observer, *, micro_batches=(X,), sync=True, zero=True):
    if zero:
        ddp.zero_grad_buffer()
    for batch in micro_batches:
        ddp.module(batch).sum().backward()
    if sync:
        ddp.finish_grad_sync()
    return observer.end_step()


@pytest.fixture
def setup():
    ddp = FakeMegatronDDP(TinyModel())
    observer = GradientStepObserver(ddp, digest_buckets=True)
    observer.install()
    yield ddp, observer
    observer.uninstall()


def test_write_paths_are_classified(setup):
    ddp, observer = setup
    evidence = _step(ddp, observer)
    assert evidence.record("adapter").write_path == "autograd_main_grad"
    assert evidence.record("head").write_path == "fused_main_grad"
    assert evidence.record("unused").wrote is False
    assert evidence.record("base").write_path == "none" and evidence.record("base").wrote is False
    assert evidence.record("base").basis == "requires_grad=False"
    assert set(evidence.emitted) == {"adapter", "head"}
    assert evidence.record("adapter").grad_present_at_end is False, "DDP moved .grad to main_grad"


def test_cancelling_micro_batches_still_count_as_writes(setup):
    ddp, observer = setup
    evidence = _step(ddp, observer, micro_batches=(X, -X))
    assert torch.count_nonzero(ddp.module.adapter.main_grad) == 0, "contributions cancelled"
    assert evidence.record("adapter").wrote is True
    assert evidence.record("adapter").accumulate_hook_calls == 2


def test_each_step_window_starts_fresh(setup):
    ddp, observer = setup
    _step(ddp, observer)
    second = _step(ddp, observer, micro_batches=())  # zeroed + synced, no backward
    assert second.step_index == 2
    assert second.emitted == ()
    assert second.record("head").flag_at_begin is False


def test_flag_left_set_by_cuda_graph_mode_is_unknown():
    ddp = FakeMegatronDDP(TinyModel(), cuda_graph_impl="transformer_engine")
    observer = GradientStepObserver(ddp)
    observer.install()
    try:
        _step(ddp, observer)
        residual = _step(ddp, observer, micro_batches=())  # flag never reset by zero_grad_buffer
    finally:
        observer.uninstall()
    record = residual.record("head")
    assert record.wrote is None and record.write_path == "unknown"
    with pytest.raises(T05ContractError) as info:
        check_emitted_against_producer(residual, {"adapter", "head"})
    assert info.value.status == "MISSING_PROVENANCE"


@pytest.mark.parametrize("zero, sync", [(False, True), (True, False)])
def test_incomplete_window_is_missing_provenance(setup, zero, sync):
    ddp, observer = setup
    with pytest.raises(T05ContractError) as info:
        _step(ddp, observer, zero=zero, sync=sync)
    assert (info.value.status, info.value.field_name) == ("MISSING_PROVENANCE", "gradient_window")


def test_base_grad_absent(setup):
    ddp, observer = setup
    evidence = _step(ddp, observer)
    check_base_grad_absent(evidence, ["base"])
    with pytest.raises(T05ContractError) as info:
        check_base_grad_absent(evidence, ["adapter"])
    assert info.value.status == "IDENTITY_DRIFT"
    with pytest.raises(T05ContractError) as info:
        check_base_grad_absent(evidence, ["not_a_param"])
    assert info.value.status == "MISSING_PROVENANCE"


def test_producer_gradient_set_cross_check(setup):
    ddp, observer = setup
    evidence = _step(ddp, observer)
    check_emitted_against_producer(evidence, {"adapter", "head"})
    for declared, status in (
        (None, "UPSTREAM_EVIDENCE_MISSING"),
        ({"adapter"}, "UPSTREAM_CONTRACT_MISMATCH"),
    ):
        with pytest.raises(T05ContractError) as info:
            check_emitted_against_producer(evidence, declared)
        assert info.value.status == status


def test_bucket_mapping_and_digests_are_recorded_but_do_not_decide(setup):
    ddp, observer = setup
    evidence = _step(ddp, observer, micro_batches=(X, -X))
    adapter = evidence.record("adapter")
    assert (adapter.buffer_index, adapter.bucket_id) == (0, 0)
    assert adapter.buffer_range == (0, 16)
    assert evidence.bucket_digests is not None and len(evidence.bucket_digests) == 1
    assert adapter.wrote is True  # despite an all-zero adapter slice in the bucket


def test_observer_is_read_only_and_uninstall_is_clean():
    def run(observe):
        torch.manual_seed(0)
        ddp = FakeMegatronDDP(TinyModel())
        observer = GradientStepObserver(ddp)
        if observe:
            observer.install()
        ddp.zero_grad_buffer()
        ddp.module(X).sum().backward()
        ddp.finish_grad_sync()
        if observe:
            observer.end_step()
            observer.uninstall()
        return {
            name: p.main_grad.clone()
            for name, p in ddp.module.named_parameters()
            if p.requires_grad
        }, ddp

    plain, _ = run(False)
    observed, ddp = run(True)
    for name in plain:
        assert torch.equal(plain[name].view(torch.int32), observed[name].view(torch.int32))
    assert "zero_grad_buffer" not in vars(ddp) and "finish_grad_sync" not in vars(ddp)


def test_hooks_are_removed_on_uninstall():
    ddp = FakeMegatronDDP(TinyModel())
    observer = GradientStepObserver(ddp)
    observer.install()
    observer.uninstall()
    observer.install()
    ddp.zero_grad_buffer()
    ddp.module(X).sum().backward()
    ddp.finish_grad_sync()
    assert (
        observer.end_step().record("adapter").accumulate_hook_calls == 1
    ), "no stale duplicate hook"
    observer.uninstall()


def test_non_ddp_module_is_rejected():
    with pytest.raises(T05ContractError) as info:
        GradientStepObserver(torch.nn.Linear(2, 2))
    assert info.value.status == "UNSUPPORTED_CAPABILITY"
