# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P0: per-call evidence snapshots with source tags (CPU, operator double)."""

from types import SimpleNamespace

import pytest
import torch
from p7_doubles.fixtures import build_fixtures, vime_request

from rl_engine.integrations.megatron_training_readback import (
    EvidenceSource,
    TrainingLogpObserver,
    json_safe,
    tensor_identity,
)

pytestmark = pytest.mark.unit

FIXTURES = build_fixtures()


class FakeLinearLogp:
    backend_id = "rlkernel.linear_logp.bitwise.v1"

    def __init__(self):
        self.provenance = {}


class FakeMegatronLogpOperator:
    """Interface double of ``MegatronLogpOperator``: provenance is rewritten every call."""

    def __init__(self):
        self._linear_logp = FakeLinearLogp()
        self._last_provenance = {}
        self.calls = 0

    @property
    def provenance(self):
        return dict(self._last_provenance)

    def __call__(self, request):
        self.calls += 1
        local = request.target_ids.reshape(-1, 1) - request.context.vocab_partition.local_start
        owned = (local >= 0) & (local < request.logits.size(1))
        gathered = (
            request.logits.float()
            .log_softmax(-1)
            .gather(1, local.clamp(0, request.logits.size(1) - 1))
        )
        logp = torch.where(owned, gathered, torch.zeros_like(gathered))
        self._linear_logp.provenance = {"strict_entrypoint": f"call-{self.calls}"}
        self._last_provenance = {
            "actual_backend": "rlkernel.linear_logp.bitwise.v1",
            "call": self.calls,
            "fallback": False,
        }
        return SimpleNamespace(
            logp=logp,
            entropy=None,
            backend_id="rlkernel.linear_logp.bitwise.v1",
            contract_id="linear_logp:tp=1:cp=1:vocab=64",
            provenance={"call": self.calls},
        )


@pytest.fixture
def observed():
    observer = TrainingLogpObserver()
    observer.install(FakeMegatronLogpOperator)
    yield observer, FakeMegatronLogpOperator()
    observer.uninstall()


def test_fields_carry_their_source(observed):
    observer, operator = observed
    operator(vime_request(FIXTURES["normal"]))
    snap = observer.snapshots[0]
    assert snap.field("logits").source is EvidenceSource.OBSERVED
    assert snap.field("context.hidden").source is EvidenceSource.OBSERVED
    assert snap.field("metadata").source is EvidenceSource.CONFIGURED
    assert snap.field("backend_id").source is EvidenceSource.DECLARED
    assert snap.field("provenance").source is EvidenceSource.DECLARED
    assert snap.field("linear_logp.provenance").source is EvidenceSource.DECLARED
    assert snap.capture_error is None and snap.operator_error is None


def test_tensors_are_recorded_as_identity_not_values(observed):
    observer, operator = observed
    request = vime_request(FIXTURES["normal"])
    operator(request)
    hidden = observer.snapshots[0].field("context.hidden").value
    assert hidden == tensor_identity(request.context.hidden)
    assert hidden["dtype"] == "bfloat16" and "values" not in hidden


def test_each_snapshot_binds_its_own_call(observed):
    observer, operator = observed
    operator(vime_request(FIXTURES["normal"]))
    operator(vime_request(FIXTURES["padded_vocab_tp2"], tp_rank=1))
    first, second = observer.snapshots
    assert first.field("linear_logp.provenance").value == {"strict_entrypoint": "call-1"}
    assert second.field("linear_logp.provenance").value == {"strict_entrypoint": "call-2"}
    # Same hidden rows by construction; the LM-head shard and vocab offset differ per call.
    assert first.field("context.hidden").value == second.field("context.hidden").value
    assert (
        first.field("context.projection.weight").value
        != second.field("context.projection.weight").value
    )
    assert first.field("context.vocab_partition.local_start").value == 0
    assert second.field("context.vocab_partition.local_start").value == 32
    assert (first.call_index, second.call_index) == (0, 1)


def test_snapshot_is_immune_to_later_mutation(observed):
    observer, operator = observed
    request = vime_request(FIXTURES["normal"])
    operator(request)
    before = observer.snapshots[0].to_dict()
    request.metadata["tp_world_size"] = 99
    operator._linear_logp.provenance["strict_entrypoint"] = "tampered"
    request.context.hidden.add_(1.0)
    assert observer.snapshots[0].to_dict() == before


def test_identical_calls_share_fingerprint_but_not_index(observed):
    observer, _ = observed
    a, b = FakeMegatronLogpOperator(), FakeMegatronLogpOperator()
    a(vime_request(FIXTURES["normal"]))
    b(vime_request(FIXTURES["normal"]))
    first, second = observer.snapshots
    assert first.fingerprint == second.fingerprint
    assert first.call_index != second.call_index


def test_missing_fields_are_marked_absent_not_defaulted(observed):
    observer, operator = observed
    request = vime_request(FIXTURES["normal"])
    del request.chunk_size
    operator(request)
    chunk = observer.snapshots[0].field("chunk_size")
    assert chunk.present is False and chunk.value is None


def test_capture_failure_never_reaches_the_training_call(observed):
    observer, operator = observed

    class Hostile(SimpleNamespace):
        @property
        def metadata(self):
            raise RuntimeError("boom")

    request = Hostile(
        **{k: v for k, v in vars(vime_request(FIXTURES["normal"])).items() if k != "metadata"}
    )
    result = operator(request)
    assert result.logp.shape == (8, 1)
    snap = observer.snapshots[0]
    assert snap.capture_error is not None and "boom" in snap.capture_error


def test_operator_failure_is_recorded_and_reraised(observed):
    observer, operator = observed
    request = vime_request(FIXTURES["normal"])
    request.target_ids = request.target_ids.float()  # gather rejects float indices
    with pytest.raises(RuntimeError):
        operator(request)
    assert observer.snapshots[0].operator_error == "RuntimeError"


def test_broken_sink_is_counted_not_raised():
    def sink(_snapshot):
        raise ValueError("sink down")

    observer = TrainingLogpObserver(sink=sink)
    observer.install(FakeMegatronLogpOperator)
    try:
        FakeMegatronLogpOperator()(vime_request(FIXTURES["normal"]))
    finally:
        observer.uninstall()
    assert len(observer.snapshots) == 1
    assert observer.sink_errors and "sink down" in observer.sink_errors[0]


def test_json_safe_distinguishes_signed_zero_and_freezes_mappings():
    plus = torch.zeros(2)
    minus = torch.tensor([0.0, -0.0])
    assert json_safe(plus)["sha256"] != json_safe(minus)["sha256"]
    source = {"a": [1, {"b": torch.tensor([1.0])}]}
    frozen = json_safe(source)
    source["a"][1]["b"].add_(1.0)
    assert frozen["a"][1]["b"] == tensor_identity(torch.tensor([1.0]))


def test_max_snapshots_bounds_memory():
    observer = TrainingLogpObserver(max_snapshots=2)
    observer.install(FakeMegatronLogpOperator)
    try:
        operator = FakeMegatronLogpOperator()
        for _ in range(3):
            operator(vime_request(FIXTURES["normal"]))
    finally:
        observer.uninstall()
    assert [snap.call_index for snap in observer.snapshots] == [1, 2]
