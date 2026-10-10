# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Unit tests for WS1 C3 forward config-invariance harness."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from rl_engine.config.workload import LogicalBatch, LogicalSample, PaddedBatch, load_manifest
from rl_engine.contracts.numerical import (
    BackendProvenance,
    load_contract,
    normalize_dtype_name,
    resolve_tolerance,
)
from rl_engine.validation.operators.forward_invariance import (
    ConfigSpec,
    ForwardInvarianceReport,
    RuntimeObservation,
    TensorComparisonDetail,
    _validate_provenance,
)
from rl_engine.validation.operators.forward_invariance import (
    assert_forward_batch_invariant as _assert_forward_batch_invariant,
)
from rl_engine.validation.operators.forward_invariance import build_config_matrix
from rl_engine.validation.operators.gradient_adapters import (
    get_adapter,
    load_adapter_gold,
    load_adapter_operator,
    make_forward_runner,
    required_forward_adapters,
)


def assert_forward_batch_invariant(*args: Any, **kwargs: Any) -> ForwardInvarianceReport:
    """Supply explicit synthetic runtime metadata for CPU-safe harness tests."""

    kwargs.setdefault("candidate_id", "synthetic-test-candidate")
    kwargs.setdefault("device", "cpu:test-double")
    kwargs.setdefault("compute_capability", "synthetic")
    kwargs.setdefault("observed_actual_backend", kwargs["provenance"].actual_backend)
    kwargs.setdefault("observed_kernel_id", "synthetic-test-candidate")
    kwargs.setdefault("observed_output_dtype", kwargs["provenance"].output_dtype)
    return _assert_forward_batch_invariant(*args, **kwargs)


@pytest.fixture()
def contract() -> dict[str, Any]:
    return load_contract()


@pytest.fixture()
def manifest():
    return load_manifest()


@pytest.fixture()
def simple_batch() -> LogicalBatch:
    samples = (
        LogicalSample(sample_id="s0", token_ids=(1, 2, 3, 4), prompt_len=2, seq_len=4),
        LogicalSample(sample_id="s1", token_ids=(5, 6, 7, 8), prompt_len=1, seq_len=4),
    )
    return LogicalBatch(workload_id="test", seed=42, samples=samples)


def _make_identity_op(value: float = 1.0):
    """Op that returns identical outputs regardless of config (batch-invariant)."""

    def op(config: ConfigSpec, **kwargs: Any) -> dict[tuple[str, int], torch.Tensor]:
        result: dict[tuple[str, int], torch.Tensor] = {}
        for sample in config.logical_batch.samples:
            for tok in sample.active_tokens():
                result[(tok.sample_id, tok.token_position)] = torch.tensor(
                    value, dtype=torch.bfloat16
                )
        return result

    return op


def _make_drifting_op(drift: float = 0.1):
    """Op that adds drift per sample to break invariance."""

    def op(config: ConfigSpec, **kwargs: Any) -> dict[tuple[str, int], torch.Tensor]:
        result: dict[tuple[str, int], torch.Tensor] = {}
        for idx, sample in enumerate(config.logical_batch.samples):
            for tok in sample.active_tokens():
                result[(tok.sample_id, tok.token_position)] = torch.tensor(
                    1.0 + idx * drift, dtype=torch.bfloat16
                )
        return result

    return op


def _make_provenance(
    backend_profile: str = "cuda_bf16",
    requested: str = "cuda",
    actual: str = "cuda",
) -> BackendProvenance:
    return BackendProvenance(
        backend_profile=backend_profile,
        requested_backend=requested,
        actual_backend=actual,
        execution_dtype="bfloat16",
        accumulation_dtype="float32",
        output_dtype="bfloat16",
        reference_dtype="float32",
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )


class TestReportStructure:
    def test_accuracy_and_invariance_reported_separately(self, contract, manifest):
        op = _make_identity_op()
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_make_identity_op(1.0),
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=False,
        )
        assert isinstance(report, ForwardInvarianceReport)
        assert hasattr(report, "accuracy_reports")
        assert hasattr(report, "invariance_reports")
        assert isinstance(report.accuracy_reports, tuple)
        assert isinstance(report.invariance_reports, tuple)
        assert len(report.invariance_reports) > 0
        assert len(report.accuracy_reports) == len(build_config_matrix(manifest))

    def test_report_contains_required_runtime_metadata(self, contract, manifest):
        report = assert_forward_batch_invariant(
            _make_identity_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_make_identity_op(),
            op_class="logprob",
            include_logprob_smoke=False,
            candidate_id="cuda-test-kernel",
            device="cuda:0:test-device",
            compute_capability="sm90",
        )
        payload = report.to_dict()
        assert payload["candidate_id"] == "cuda-test-kernel"
        assert payload["device"] == "cuda:0:test-device"
        assert payload["compute_capability"] == "sm90"
        assert payload["seed"] == manifest.seed
        assert payload["fallback_reason"] is None

    def test_missing_runtime_metadata_fails_closed(self, contract, manifest):
        report = _assert_forward_batch_invariant(
            _make_identity_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_make_identity_op(),
            include_logprob_smoke=False,
        )
        assert report.provenance_valid
        assert not report.metadata_valid
        assert not report.passed

    def test_report_contains_max_abs_rel_tensor_name(self, contract, manifest):
        op = _make_identity_op()
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_make_identity_op(),
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=False,
        )
        for inv in report.invariance_reports:
            for detail in inv.details:
                assert isinstance(detail, TensorComparisonDetail)
                assert detail.tensor_name is not None
                assert detail.max_abs_error is not None
                assert detail.max_rel_error is not None
                assert detail.config_pair is not None
                assert len(detail.config_pair) == 2


class TestInvariance:
    def test_invariance_bitwise_zero_tolerance(self, contract, manifest):
        op = _make_identity_op()
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_make_identity_op(),
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=False,
        )
        for inv in report.invariance_reports:
            for detail in inv.details:
                assert detail.judgment == "forward_invariance"
                assert detail.atol == 0.0
                assert detail.rtol == 0.0

    def test_identity_op_passes_invariance(self, contract, manifest):
        op = _make_identity_op()
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_make_identity_op(),
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=False,
        )
        for inv in report.invariance_reports:
            assert inv.passed, f"invariance failed for {inv.transformed_config_id}"
        assert report.passed

    def test_logical_unpadding_before_compare(self, contract, manifest):
        op = _make_identity_op()
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_make_identity_op(),
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=False,
            active_only=True,
        )
        for inv in report.invariance_reports:
            assert inv.passed

    def test_padding_configs_use_c2_padded_layout(self, manifest):
        padded = [c for c in build_config_matrix(manifest) if c.transform_kind == "padding"]
        assert {c.physical_layout.pad_side for c in padded} == {"left", "right"}
        assert all(isinstance(c.physical_layout, PaddedBatch) for c in padded)

    def test_missing_active_token_hard_fails(self, contract, manifest):
        def incomplete(config: ConfigSpec, **kwargs: Any):
            result = _make_identity_op()(config, **kwargs)
            result.pop(next(iter(result)))
            return result

        with pytest.raises(ValueError, match="C2 logical identity"):
            assert_forward_batch_invariant(
                incomplete,
                contract=contract,
                manifest=manifest,
                backend_profile="cuda_bf16",
                provenance=_make_provenance(),
                gold_fn=_make_identity_op(),
                include_logprob_smoke=False,
            )

    def test_padded_tensor_is_logically_unpadded(self, contract, manifest):
        def physical_identity(config: ConfigSpec, **kwargs: Any):
            layout = config.physical_layout
            if isinstance(layout, PaddedBatch):
                return torch.ones(
                    (len(layout.restore_map), layout.padded_len), dtype=torch.bfloat16
                )
            return torch.ones(len(layout.restore_map), dtype=torch.bfloat16)

        report = assert_forward_batch_invariant(
            physical_identity,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=physical_identity,
            include_logprob_smoke=False,
        )
        padding_reports = [r for r in report.invariance_reports if r.transform_kind == "padding"]
        assert len(padding_reports) == 2
        assert all(r.passed for r in padding_reports)


class TestAccuracy:
    def test_missing_reference_is_rejected(self, contract, manifest):
        with pytest.raises(ValueError, match="gold_fn is required"):
            assert_forward_batch_invariant(
                _make_identity_op(),
                contract=contract,
                manifest=manifest,
                backend_profile="cuda_bf16",
                provenance=_make_provenance(),
                gold_fn=None,
                include_logprob_smoke=False,
            )

    def test_accuracy_uses_c1_tolerances(self, contract, manifest):
        op = _make_identity_op(1.0)
        gold = _make_identity_op(1.0)
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=gold,
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=False,
        )
        for acc in report.accuracy_reports:
            for detail in acc.details:
                assert detail.judgment == "forward_accuracy"
                spec = resolve_tolerance(
                    contract,
                    judgment="forward_accuracy",
                    op_class="logprob",
                    dtype=torch.bfloat16,
                    backend_profile="cuda_bf16",
                )
                assert detail.atol == spec.atol
                assert detail.rtol == spec.rtol

    def test_no_private_thresholds(self, contract, manifest):
        op = _make_identity_op(1.0)
        gold = _make_identity_op(1.0)
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=gold,
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=False,
        )
        for acc in report.accuracy_reports:
            for detail in acc.details:
                spec = resolve_tolerance(
                    contract,
                    judgment=detail.judgment,
                    op_class=acc.op_class,
                    dtype=torch.bfloat16,
                    backend_profile=acc.backend_profile,
                )
                assert detail.atol == spec.atol
                assert detail.rtol == spec.rtol


class TestBackendProvenance:
    def test_valid_provenance_passes(self, contract):
        provenance = _make_provenance("cuda_bf16", "cuda", "cuda")
        assert _validate_provenance(contract, provenance, "cuda_bf16") is True

    def test_silent_fallback_rejected(self, contract):
        provenance = _make_provenance("cuda_bf16", "cuda", "triton")
        assert _validate_provenance(contract, provenance, "cuda_bf16") is False

    def test_cross_profile_fallback_rejected(self, contract):
        provenance = _make_provenance("triton_cuda_bf16", "triton", "triton")
        assert _validate_provenance(contract, provenance, "cuda_bf16") is False

    def test_none_provenance_fails_closed(self, contract):
        assert _validate_provenance(contract, None, "cuda_bf16") is False

    @pytest.mark.parametrize(
        ("profile", "family"),
        [("cuda_bf16", "cuda"), ("triton_cuda_bf16", "triton")],
    )
    def test_required_profiles_share_report_schema(self, contract, manifest, profile, family):
        provenance = _make_provenance(profile, family, family)
        report = assert_forward_batch_invariant(
            _make_identity_op(),
            contract=contract,
            manifest=manifest,
            backend_profile=profile,
            provenance=provenance,
            gold_fn=_make_identity_op(),
            include_logprob_smoke=False,
        )
        assert report.passed
        assert set(report.to_dict()) == set(
            ForwardInvarianceReport(
                op_name="x",
                backend_profile=profile,
                accuracy_reports=(),
                invariance_reports=(),
                logprob_smoke=None,
                backend_provenance=provenance,
                candidate_id="x",
                device="x",
                compute_capability=None,
                seed=manifest.seed,
                fallback_reason=None,
                passed=True,
                provenance_valid=True,
                metadata_valid=True,
            ).to_dict()
        )

    def test_provenance_failure_fails_report(self, contract, manifest):
        op = _make_identity_op()
        bad_provenance = _make_provenance("cuda_bf16", "cuda", "triton")
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=bad_provenance,
            gold_fn=_make_identity_op(),
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=False,
        )
        assert report.provenance_valid is False
        assert report.passed is False

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("observed_actual_backend", "triton"),
            ("observed_kernel_id", "other-kernel"),
            ("observed_output_dtype", "float32"),
        ],
    )
    def test_runtime_observation_mismatch_fails_closed(self, contract, manifest, field, value):
        kwargs = {
            "observed_actual_backend": "cuda",
            "observed_kernel_id": "synthetic-test-candidate",
            "observed_output_dtype": "bfloat16",
        }
        kwargs[field] = value

        def observed_op(config: ConfigSpec, **kwargs: Any):
            return RuntimeObservation(
                output=_make_identity_op()(config, **kwargs),
                actual_backend="cuda",
                kernel_id="synthetic-test-candidate",
                output_dtype="bfloat16",
                device="cpu:test-double",
            )

        report = _assert_forward_batch_invariant(
            observed_op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_make_identity_op(),
            include_logprob_smoke=False,
            candidate_id="synthetic-test-candidate",
            device="cpu:test-double",
            compute_capability="synthetic",
            **kwargs,
        )
        assert report.metadata_valid is False
        assert report.passed is False


class TestConfigMatrix:
    def test_config_matrix_covers_c2_cells(self, manifest):
        configs = build_config_matrix(manifest)
        config_ids = [c.config_id for c in configs]
        assert any("BN/full" in cid for cid in config_ids)
        assert any("BN/chunked" in cid for cid in config_ids)
        assert any("B1-singleton_aggregate/full" in cid for cid in config_ids)
        assert any("B1-singleton_aggregate/chunked" in cid for cid in config_ids)
        assert any("permuted" in cid for cid in config_ids)
        assert any("padded_right" in cid for cid in config_ids)
        assert any("padded_left" in cid for cid in config_ids)

    def test_canonical_config_exists(self, manifest):
        configs = build_config_matrix(manifest)
        canonical = [c for c in configs if c.is_canonical]
        assert len(canonical) == 1
        assert canonical[0].config_id == "BN/full"


class TestLogprobSmoke:
    def test_logprob_smoke_passes_for_identical(self, contract, manifest):
        op = _make_identity_op(0.0)
        gold = _make_identity_op(0.0)
        report = assert_forward_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=gold,
            op_class="logprob",
            dtype=torch.bfloat16,
            op_name="test_op",
            include_logprob_smoke=True,
        )
        assert report.logprob_smoke is not None
        assert report.logprob_smoke.passed


_REQUIRED_FORWARD_OPS = (
    "embedding",
    "rms_norm",
    "qk_norm",
    "det_gemm",
    "rope",
    "attention",
    "silu",
    "swiglu",
    "lm_head",
    "logp",
    "batch_invariant_logp",
    "pack",
)
_OPTIONAL_FORWARD_OPS = ("linear_logp",)


class TestForwardAdapters:
    """C3 must run every C2 required chain op (plus pack) through one runner."""

    def test_required_forward_ops_are_enumerable(self):
        names = {spec.op_name for spec in required_forward_adapters()}
        assert set(_REQUIRED_FORWARD_OPS) <= names
        for op_name in _REQUIRED_FORWARD_OPS:
            adapter = get_adapter(op_name)
            assert adapter.requirement != "absent_not_required"

    def test_native_rms_norm_forward_passes_c3(self, contract, manifest):
        report = _run_native_forward("rms_norm", contract, manifest)
        assert report.passed
        assert all(item.passed for item in report.accuracy_reports)
        assert all(item.passed for item in report.invariance_reports)

    @pytest.mark.parametrize("op_name", _REQUIRED_FORWARD_OPS + _OPTIONAL_FORWARD_OPS)
    def test_native_forward_adapter_is_batch_invariant(self, op_name, contract, manifest):
        report = _run_native_forward(op_name, contract, manifest)
        assert all(item.passed for item in report.invariance_reports), report.to_dict()
        assert all(item.passed for item in report.accuracy_reports), report.to_dict()
        assert report.passed


def _run_native_forward(op_name: str, contract, manifest) -> ForwardInvarianceReport:
    adapter = get_adapter(op_name)
    gold = load_adapter_gold(op_name)
    candidate = load_adapter_operator(op_name, "pytorch")
    shape = {
        "hidden": 8,
        "vocab_size": 256,
        "n_heads": 4,
        "n_kv_heads": 1,
        "head_dim": 16,
    }
    runner = make_forward_runner(
        op_name,
        candidate,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        reference=False,
        backend_family="cuda",
        kernel_id=f"pytorch-{op_name}",
        **shape,
    )
    probe = runner(next(config for config in build_config_matrix(manifest) if config.is_canonical))
    if isinstance(probe, RuntimeObservation):
        observed_dtype = probe.output_dtype
    else:
        observed_dtype = normalize_dtype_name(next(iter(probe.values())).dtype)
    return assert_forward_batch_invariant(
        runner,
        contract=contract,
        manifest=manifest,
        backend_profile="cuda_bf16",
        provenance=_make_provenance(),
        gold_fn=make_forward_runner(
            op_name,
            gold,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            reference=True,
            **shape,
        ),
        op_class=adapter.op_class,
        op_name=op_name,
        include_logprob_smoke=adapter.op_class == "logprob",
        candidate_id=f"pytorch-{op_name}",
        device="cpu:test-double",
        compute_capability="synthetic",
        observed_actual_backend="cuda",
        observed_kernel_id=f"pytorch-{op_name}",
        observed_output_dtype=observed_dtype,
    )
