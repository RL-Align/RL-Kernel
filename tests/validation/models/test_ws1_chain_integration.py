# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-safe C10 report / schema / bitwise-rule tests. Full-model execute is H20."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rl_engine.config.workload import load_manifest
from rl_engine.contracts.numerical import load_contract
from rl_engine.models.qwen3.reference import Qwen3DenseSpec
from rl_engine.ops.autograd.vjp_fp32 import (
    reduce_keyed_outers_fp32,
    reduce_keyed_rows_fp32,
    row_local_linear_dw_fp32,
    row_local_linear_dx_fp32,
)
from rl_engine.validation.models.chain_gate import (
    GRADIENT_SCOPE,
    LAYOUT_CELLS,
    PRIMARY_CELLS,
    REQUIRED_GRAD_NAMES,
    ChainGateReport,
    _c8_evidence_path,
    _c8_source_commit,
    _collect_parameter_grads,
    _compare_logp_maps,
    _configure_required_gradients,
    _logp_aggregate_verdict,
    _node_token_fingerprints,
    _release_parameter_grads,
    _representative_case_ids,
)


def test_c10_primary_cells_match_c2_matrix():
    assert PRIMARY_CELLS == (
        "B1-singleton_aggregate/full",
        "BN/full",
        "B1-singleton_aggregate/chunked",
        "BN/chunked",
    )


def test_c10_packing_is_declared_supported_required_axis():
    manifest = load_manifest()
    assert manifest.fixtures["packing"]["status"] == "supported"


def test_c10_spec_fingerprint_is_full_qwen3():
    spec = Qwen3DenseSpec.from_manifest(load_manifest())
    assert spec.num_hidden_layers == 36
    assert spec.hidden_size == 4096
    assert spec.vocab_size == 151936


def test_c10_bitwise_invariance_rule_is_zero_tol():
    contract = load_contract()
    lhs = {("s0", 1): torch.tensor(0.25), ("s1", 2): torch.tensor(-0.5)}
    rhs = {("s0", 1): torch.tensor(0.25), ("s1", 2): torch.tensor(-0.5)}
    detail = _compare_logp_maps(
        lhs,
        rhs,
        contract=contract,
        judgment="forward_invariance",
        dtype="bfloat16",
        backend_profile="cuda_bf16",
        config_pair=("BN/full", "B1-singleton_aggregate/full"),
    )
    assert detail.atol == 0.0
    assert detail.rtol == 0.0
    assert detail.passed


def test_c10_bitwise_invariance_fails_on_drift():
    contract = load_contract()
    lhs = {("s0", 1): torch.tensor(0.25)}
    rhs = {("s0", 1): torch.tensor(0.26)}
    detail = _compare_logp_maps(
        lhs,
        rhs,
        contract=contract,
        judgment="forward_invariance",
        dtype="bfloat16",
        backend_profile="cuda_bf16",
        config_pair=("BN/full", "BN/chunked"),
    )
    assert detail.passed is False
    assert detail.max_abs_error > 0.0


def test_c10_report_schema_fields():
    fields = set(ChainGateReport.__dataclass_fields__)
    for name in (
        "backend_profile",
        "workload_id",
        "fixture_hash",
        "config_fingerprint",
        "weight_hash",
        "backend_provenance",
        "runtime_backend_observations",
        "backward_runtime_observations",
        "invariance",
        "gradient_invariance",
        "train_infer",
        "first_drift",
        "aggregates",
        "accuracy_aggregates",
        "decode_prefill",
        "gpu_name",
        "representative_case_ids",
        "workflow_url",
        "c8_evidence_path",
        "c8_source_commit",
        "passed",
        "backward_executed",
        "train_infer_executed",
        "accuracy_executed",
        "gradient_accuracy_executed",
        "accuracy",
        "gradient_accuracy",
        "train_infer_bn",
        "gradient_scope",
        "required_grad_names",
        "all_parameter_gradients",
        "disclaimer",
    ):
        assert name in fields


def test_c10_required_gradients_are_enabled_before_forward():
    names = (
        "norm.weight",
        "lm_head.weight",
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
    )
    tensors = {name: torch.tensor(2.0) for name in REQUIRED_GRAD_NAMES}
    tensors["unused.weight"] = torch.tensor(3.0)
    model = SimpleNamespace(weights=SimpleNamespace(tensors=tensors))
    _configure_required_gradients(model, enabled=True)
    loss = tensors["norm.weight"] * tensors["lm_head.weight"]
    loss.backward()
    assert tensors["norm.weight"].grad is not None
    assert tensors["lm_head.weight"].grad is not None
    assert tensors["unused.weight"].requires_grad is False
    for name in names:
        assert tensors[name].requires_grad is True


def test_c10_gradient_contract_covers_required_trainable_weights():
    assert GRADIENT_SCOPE == "all_required_trainable_parameters"
    assert "embed_tokens.weight" in REQUIRED_GRAD_NAMES
    assert "lm_head.weight" in REQUIRED_GRAD_NAMES
    assert "norm.weight" in REQUIRED_GRAD_NAMES
    assert "layers.0.self_attn.k_proj.weight" in REQUIRED_GRAD_NAMES
    assert "layers.0.self_attn.v_proj.weight" in REQUIRED_GRAD_NAMES
    assert "layers.0.self_attn.o_proj.weight" in REQUIRED_GRAD_NAMES
    assert "layers.0.mlp.gate_proj.weight" in REQUIRED_GRAD_NAMES
    assert "layers.0.mlp.up_proj.weight" in REQUIRED_GRAD_NAMES
    assert "layers.0.mlp.down_proj.weight" in REQUIRED_GRAD_NAMES
    assert "layers.35.mlp.down_proj.weight" in REQUIRED_GRAD_NAMES
    assert len(REQUIRED_GRAD_NAMES) == 3 + 36 * 11


def test_c10_layout_cells_include_padding_permutation_and_packing():
    assert "BN/padded_right" in LAYOUT_CELLS
    assert "BN/padded_left" in LAYOUT_CELLS
    assert "BN/permuted" in LAYOUT_CELLS
    assert "BN/packed" in LAYOUT_CELLS


def test_c10_fp32_accuracy_uses_three_aggregates():
    contract = load_contract()
    lhs = {("s0", 1): torch.tensor(0.25), ("s1", 2): torch.tensor(-0.5)}
    rhs = {("s0", 1): torch.tensor(0.25), ("s1", 2): torch.tensor(-0.5)}
    verdict = _logp_aggregate_verdict(
        lhs,
        rhs,
        contract=contract,
        report_kind="forward_accuracy",
    )
    payload = verdict.to_dict()
    assert payload["aggregates"]["max_abs_dlogp"] == 0.0
    assert "approx_kl0" in payload["aggregates"]
    assert "clipfrac0" in payload["aggregates"]
    assert verdict.passed


def test_c10_representative_case_ids_are_profile_scoped():
    manifest = load_manifest()
    cuda_ids = _representative_case_ids(manifest, "cuda_bf16")
    triton_ids = _representative_case_ids(manifest, "triton_cuda_bf16")
    assert cuda_ids
    assert triton_ids
    assert any("cuda" in case_id for case_id in cuda_ids)
    assert any("triton" in case_id for case_id in triton_ids)
    assert set(cuda_ids).isdisjoint(triton_ids)


def test_logical_row_reduction_is_independent_of_insertion_order():
    rows = {
        ("s1", 2): torch.tensor([1.0, 2.0]),
        ("s0", 0): torch.tensor([3.0, 4.0]),
        ("s0", 1): torch.tensor([5.0, 6.0]),
    }
    shuffled = {
        ("s0", 1): rows[("s0", 1)],
        ("s1", 2): rows[("s1", 2)],
        ("s0", 0): rows[("s0", 0)],
    }
    assert torch.equal(reduce_keyed_rows_fp32(rows), reduce_keyed_rows_fp32(shuffled))


def test_row_local_linear_vjp_matches_per_row_outer_and_gemv():
    # Exact dyadic inputs keep this algebra check independent of BLAS layout
    # and fused multiply-add choices, without relaxing bitwise assertions.
    hidden = (torch.arange(12, dtype=torch.float32).reshape(4, 3) - 6) / 8
    weight = (torch.arange(15, dtype=torch.float32).reshape(5, 3) - 7) / 8
    grad = (torch.arange(20, dtype=torch.float32).reshape(4, 5) - 10) / 8
    dx = row_local_linear_dx_fp32(grad, weight)
    dw = row_local_linear_dw_fp32(grad, hidden)
    expected_dx = torch.stack([torch.mv(weight.t(), row) for row in grad], dim=0)
    expected_dw = reduce_keyed_outers_fp32(
        {("s", i): grad[i] for i in range(4)},
        {("s", i): hidden[i] for i in range(4)},
    )
    assert torch.equal(dx, expected_dx)
    assert torch.equal(dw, expected_dw)


def test_declared_lm_head_backward_uses_det_gemm():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "rl_engine" / "backends"
    cuda_lm = (root / "cuda" / "embedding" / "lm_head.py").read_text(encoding="utf-8")
    triton_lm = (root / "shared" / "triton" / "embedding" / "lm_head.py").read_text(
        encoding="utf-8"
    )
    cuda_bwd = cuda_lm.split("def backward")[1]
    triton_bwd = triton_lm.split("def backward")[1]
    assert "det_gemm_fwd" in cuda_bwd
    assert "_triton_gemm" in triton_bwd
    assert ".matmul(" not in cuda_bwd
    assert ".matmul(" not in triton_bwd
    assert "torch.mv" not in cuda_bwd
    assert "addmm_" not in cuda_bwd
    assert "torch.mv" not in triton_bwd
    assert "addmm_" not in triton_bwd


def test_c10_node_fingerprints_follow_logical_tokens():
    class FakeModel:
        def __init__(self, output):
            self.output = output

        def captured_node_outputs(self):
            return {"embedding": self.output}

    canonical = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    canonical_restore = (("s0", 0), ("s0", 1)), (("s1", 0), ("s1", 1))
    permuted = canonical.index_select(0, torch.tensor([1, 0]))
    permuted_restore = (canonical_restore[1], canonical_restore[0])
    lhs = _node_token_fingerprints(FakeModel(canonical), canonical_restore)
    rhs = _node_token_fingerprints(FakeModel(permuted), permuted_restore)
    assert lhs == rhs

    changed = canonical.clone()
    changed[1, 1, 0] += 1.0
    drifted = _node_token_fingerprints(FakeModel(changed), canonical_restore)
    assert lhs != drifted


def test_c10_gradient_snapshots_are_cpu_native_dtype_and_releasable():
    tensors = {
        name: torch.tensor(1.0, dtype=torch.bfloat16, requires_grad=True)
        for name in REQUIRED_GRAD_NAMES
    }
    for tensor in tensors.values():
        tensor.grad = torch.tensor(2.0, dtype=torch.bfloat16)
    model = SimpleNamespace(weights=SimpleNamespace(tensors=tensors))
    grads = _collect_parameter_grads(model, cell_id="test")
    assert set(grads) == set(REQUIRED_GRAD_NAMES)
    assert all(value.device.type == "cpu" for value in grads.values())
    assert all(value.dtype == torch.bfloat16 for value in grads.values())

    cell = SimpleNamespace(grads=grads)
    _release_parameter_grads(cell)
    assert set(cell.grads) == set(REQUIRED_GRAD_NAMES)
    assert all(value.numel() == 0 for value in cell.grads.values())


def test_backward_runtime_records_constituent_kernel_ids():
    from rl_engine.ops.autograd.backward_runtime import (
        record_backward,
        reset_backward_runtime,
        snapshot_backward_runtime,
    )

    reset_backward_runtime()
    record_backward(
        "det_gemm",
        kernel_id="cuda.da+cuda.db",
        impl="cuda_det_gemm",
        family="cuda",
    )
    event = snapshot_backward_runtime()["det_gemm"]
    assert event["kernel_ids"] == ["cuda.da", "cuda.db"]
    assert event["implementation_ids"] == ["cuda.da", "cuda.db"]
    assert event["execution_count"] == 1


def test_c8_runtime_evidence_path_binds_external_artifact(monkeypatch, tmp_path):
    import json

    evidence = tmp_path / "ws1-c8-ci.json"
    evidence.write_text(
        json.dumps({"git": {"commit": "abc123", "dirty": False}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("WS1_C8_EVIDENCE_PATH", str(evidence))
    assert _c8_evidence_path() == str(evidence)
    assert _c8_source_commit() == "abc123"
