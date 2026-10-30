# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 G1 (GPU, SM90): the real strict route through the recorded provider and gate.

Runs the real ``LinearLogpWrapper`` (CUDA SM90 kernels), the real
``MegatronLogpOperator`` and VIME's own request types. Skipped unless a Hopper
GPU, the SM90 extension and VIME are all available.
"""

import pytest
import torch
from p7_doubles import t02_sampling
from p7_doubles.fixtures import make_fixture
from p7_doubles.strict_linear_logp import CpuStrictLinearLogpDouble

from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate
from rl_engine.alignment.cross_config.adapters.megatron_training import (
    PreUpdateWeightLock,
    ShardDigest,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_provider import (
    M10ProviderInput,
    RecordedTrainingProvider,
)


def _sm90_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    except Exception:
        return False
    return bool(_EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90"))


pytestmark = [
    pytest.mark.skipif(not _sm90_ready(), reason="needs a Hopper GPU and the SM90 extension"),
]

FIXTURE = make_fixture(
    "gpu_g1", sample_lengths=(64, 32), width=128, real_vocab=500, padded_vocab=512
)


@pytest.fixture(autouse=True)
def _strict_env(monkeypatch):
    pytest.importorskip("vime.backends.megatron_utils.linear_logp_provider")
    monkeypatch.setenv("VIME_RL_KERNEL_STRICT", "1")
    monkeypatch.setenv("RL_KERNEL_LOGP_CASE", "R/R")


def _m10(rows, sample_ids, sample_lengths):
    weight = FIXTURE.lm_head_weight.cuda()
    shard = ShardDigest.from_tensor(weight, tp_rank=0, vocab_start=0, vocab_end=512)
    return M10ProviderInput(
        final_decoder_rows=FIXTURE.hidden[rows].cuda(),
        lm_head_weight_local=weight,
        target_token_ids=FIXTURE.target_ids[rows].cuda(),
        active_token_mask=FIXTURE.active_mask[rows].clone(),
        vocab_start=0,
        real_vocab_size=500,
        padded_vocab_size=512,
        token_layout_world_size=1,
        token_layout_rank=0,
        token_layout="single",
        row_layout=gate.RowLayout(sample_ids, sample_lengths),
        checkpoint_id="ckpt-0",
        weight_id="w-0",
        weight_lock=PreUpdateWeightLock("ckpt-0", "w-0", (shard,)),
        lm_head_lora=False,
        sampling_record=t02_sampling.record(t02_sampling.IDENTITY_REQUEST),
        metadata={"tp_rank": 0, "tp_world_size": 1},
    )


def _full():
    return _m10(list(range(96)), ("s0", "s1"), (64, 32))


def _per_token_bytes(evidence):
    raw = evidence.selected_logprob.contiguous().view(torch.int32)
    return {key: int(raw[i]) for i, key in enumerate(evidence.row_keys) if key is not None}


def test_real_strict_route_passes_the_gate():
    evidence = RecordedTrainingProvider().score(_full())
    assert evidence.path == "recompute"
    assert evidence.entrypoint == "sm90_deterministic_linear_logp"
    assert evidence.runtime_platform == "cuda"
    assert evidence.selected_logprob.dtype == torch.float32


def test_cpu_double_provenance_keys_match_the_real_wrapper():
    from rl_engine.integrations.linear_logp import LinearLogpWrapper

    m10 = _full()
    real = LinearLogpWrapper()
    double = CpuStrictLinearLogpDouble()
    args = (m10.final_decoder_rows, m10.lm_head_weight_local, m10.target_token_ids)
    kwargs = dict(vocab_start_index=0, global_vocab_size=512, real_vocab_size=500)
    real(*args, **kwargs)
    double(*args, **kwargs)
    assert set(real.provenance) == set(double.provenance)
    for key in ("strict_entrypoint", "actual_backend", "contract_version", "arithmetic_contract"):
        assert real.provenance[key] == double.provenance[key], key


def test_l1_repeat_is_bitwise_identical():
    first = RecordedTrainingProvider().score(_full())
    for _ in range(2):
        again = RecordedTrainingProvider().score(_full())
        assert again.active_raw_bytes == first.active_raw_bytes
        assert again.row_keys == first.row_keys


def test_l2_batch_composition_is_bitwise_invariant():
    full = _per_token_bytes(RecordedTrainingProvider().score(_full()))
    only_second = _per_token_bytes(
        RecordedTrainingProvider().score(_m10(list(range(64, 96)), ("s1",), (32,)))
    )
    assert only_second == {key: full[key] for key in only_second}
    reordered = _per_token_bytes(
        RecordedTrainingProvider().score(
            _m10(list(range(64, 96)) + list(range(64)), ("s1", "s0"), (32, 64))
        )
    )
    assert reordered == full
