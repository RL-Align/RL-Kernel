# SPDX-License-Identifier: Apache-2.0
"""Exercise the applied companion's request construction without Megatron startup."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_companion_temperature_and_top_p_contract(monkeypatch, sparse, temperature):
    root = os.environ.get("RLK_TEST_VIME_ROOT")
    if not root:
        pytest.skip("set RLK_TEST_VIME_ROOT to an applied ROCm companion checkout")
    source = Path(root, "vime/backends/megatron_utils/loss.py").read_text()
    function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_log_probs_and_entropy"
    )
    # Compile the real adapter function, stubbing only distributed/framework APIs.
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    captured = {}

    def compute(**kwargs):
        captured["request"] = kwargs["request"]
        return torch.zeros(2, 1), None

    support = torch.tensor([[0, 1], [1, -1]])
    mask = torch.tensor([[True, True, False], [False, True, False]])
    scope = dict(
        torch=torch,
        os=os,
        mpu=SimpleNamespace(
            get_tensor_model_parallel_group=lambda: None,
            get_context_parallel_world_size=lambda: 1,
            get_context_parallel_rank=lambda: 0,
            get_tensor_model_parallel_rank=lambda: 0,
            get_tensor_model_parallel_world_size=lambda: 1,
        ),
        _build_shifted_tokens=lambda *a: torch.tensor([0, 1]),
        _build_topp_sparse_ids=lambda *a: support,
        _build_topp_keep_mask=lambda *a: mask,
        LinearLogpRequest=SimpleNamespace,
        TokenLayout=SimpleNamespace,
        compute_linear_logp=compute,
        calculate_log_probs_and_entropy=object(),
        _extract_per_sample=lambda *a: ([a[0]], None),
    )
    exec(compile(module, "companion-loss-contract", "exec"), scope)
    monkeypatch.setenv("RL_KERNEL_SPARSE_TOP_P_REPLAY", "1" if sparse else "0")
    logits = torch.tensor([[[40.0, 39.0, 35.0], [32.0, 33.0, 34.0]]])
    scope["get_log_probs_and_entropy"](
        logits,
        args=SimpleNamespace(
            rollout_temperature=temperature, log_probs_chunk_size=2, allgather_cp=False
        ),
        unconcat_tokens=[],
        total_lengths=[2],
        response_lengths=[2],
        top_p_token_ids=[[0, 1, 1]],
        top_p_token_offsets=[[0, 2, 3]],
    )
    request = captured["request"]
    assert request.temperature == temperature
    assert request.metadata["logits_are_temperature_scaled"] is (not sparse)
    assert torch.equal(
        request.logits, logits.squeeze(0) if sparse else logits.squeeze(0) / temperature
    )
    if sparse:
        assert request.metadata["top_p_sparse_token_ids"] is support
        assert request.log_prob_keep_mask is None
    else:
        assert request.metadata["top_p_sparse_token_ids"] is None
        assert request.log_prob_keep_mask is mask
