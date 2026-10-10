# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rl_engine.integrations.engines.rollout.vllm.operators as operators


@pytest.mark.parametrize("top_p", [0.95, 1.0])
@pytest.mark.parametrize("worker", [True, False])
@pytest.mark.parametrize("temperature", [0.3, 0.7, 1.6])
@pytest.mark.parametrize("inference", [False, True])
def test_sampler_replays_active_request_support_and_preserves_raw_logits(
    monkeypatch, top_p, worker, temperature, inference
):
    monkeypatch.setattr(operators, "_require_nvidia_cuda", lambda *_: None)
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setenv("RL_KERNEL_VLLM_TEMPERATURE", str(temperature))
    context = SimpleNamespace(
        hidden=torch.zeros(1, 2),
        lm_head_weight=torch.zeros(4, 2),
        tp_group=None,
        vocab_start_index=0,
        global_vocab_size=8,
        real_vocab_size=8,
    )
    monkeypatch.setattr(operators, "take_rollout_linear_logp_context", lambda: context)
    tensors = namedtuple("LogprobsTensors", "logprob_token_ids logprobs")

    @dataclass
    class Result:
        sampled_token_ids: torch.Tensor
        logprobs_tensors: object

    result = Result(torch.tensor([[1]]), tensors(torch.tensor([[1, 1, 2]]), torch.zeros(1, 3)))

    def native(sampler, logits, metadata, **kwargs):
        logits.fill_(-100)  # A one-row narrow is contiguous but still aliases its source.
        return result

    calls = []

    class Wrapper:
        backend_id = "test"
        provenance = {
            "deterministic_linear_logp": True,
            "actual_backend": "test",
            "strict_entrypoint": "rocm_vocab_parallel_logp_from_local_logits_tp",
        }

        def from_local_logits(self, logits, ids, **kwargs):
            calls.append("dense")
            assert torch.equal(logits, torch.arange(4).reshape(1, 4).float())
            assert kwargs["temperature"] == temperature
            return torch.tensor([-2.5])

        def from_local_logits_sparse_nucleus(self, logits, ids, replay_ids, **kwargs):
            self.from_local_logits(logits, ids, **kwargs)
            self.provenance = {
                **self.provenance,
                "strict_entrypoint": "sparse_nucleus_logp_from_local_logits_tp",
            }
            calls.append("top_p")
            assert torch.equal(replay_ids, result.logprobs_tensors.logprob_token_ids)
            return torch.tensor([-2.5])

        def from_replicated_logits_sparse_nucleus(self, logits, ids, replay_ids, **kwargs):
            assert torch.equal(logits, torch.arange(8).reshape(1, 8).float())
            assert kwargs == {"real_vocab_size": 8, "temperature": temperature, "tp_group": None}
            self.provenance = {
                **self.provenance,
                "strict_entrypoint": "sparse_nucleus_logp_from_replicated_logits",
            }
            calls.append("replicated")
            return torch.tensor([-2.5])

    monkeypatch.setattr(operators, "LinearLogpWrapper", Wrapper)
    sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(top_p=SimpleNamespace(np=np.array([0.5, top_p, 0.8])))
    )
    metadata = (
        SimpleNamespace(idx_mapping_np=np.array([1]))
        if worker
        else SimpleNamespace(top_p=torch.tensor([top_p]))
    )

    class RocmTemperature:
        def __lt__(self, other):
            raise AssertionError("ROCm must not execute unused CUDA greedy preprocessing")

    metadata.temperature = RocmTemperature()
    op = operators.VllmLogpOperator(native, worker_sampler=worker, strict_linear_logp=True)
    with torch.no_grad() if inference else torch.enable_grad():
        actual = op(sampler, torch.arange(8).reshape(1, 8).float(), metadata)
    expected = (
        ["replicated"]
        if inference and worker and top_p < 1
        else (["dense", "top_p"] if top_p < 1 else ["dense"])
    )
    assert calls == expected
    assert torch.equal(actual.logprobs_tensors.logprobs, torch.tensor([[-2.5, -2.5, 0.0]]))
