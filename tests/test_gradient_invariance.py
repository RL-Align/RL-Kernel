# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Unit tests for WS1 C4 gradient config-invariance harness."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from rl_engine.kernels.gtest.forward_invariance import ConfigSpec
from rl_engine.kernels.gtest.gradient_adapters import (
    GRADIENT_ADAPTERS,
    adapter_names,
    get_adapter,
    gradient_adapter_status_matrix,
    listed_source_paths,
    load_adapter_gold,
    load_adapter_operator,
    make_gradient_runner,
    required_gradient_adapters,
)
from rl_engine.kernels.gtest.gradient_invariance import (
    GradientInvarianceReport,
    GradientObservation,
    GradientTensorSpec,
    MissingBackwardError,
)
from rl_engine.kernels.gtest.gradient_invariance import (
    assert_gradient_batch_invariant as _assert_gradient_batch_invariant,
)
from rl_engine.kernels.gtest.tolerance import BackendProvenance, load_contract, resolve_tolerance
from rl_engine.testing.ws1_workload import PaddedBatch, load_manifest

_RMS_TENSORS = (
    GradientTensorSpec("dx", "token", "x"),
    GradientTensorSpec("dweight", "parameter", "weight"),
)


def assert_gradient_batch_invariant(*args: Any, **kwargs: Any) -> GradientInvarianceReport:
    kwargs.setdefault("candidate_id", "synthetic-test-candidate")
    kwargs.setdefault("device", "cpu:test-double")
    kwargs.setdefault("compute_capability", "synthetic")
    kwargs.setdefault("observed_actual_backend", kwargs["provenance"].actual_backend)
    kwargs.setdefault("observed_kernel_id", "synthetic-test-candidate")
    kwargs.setdefault("observed_output_dtype", kwargs["provenance"].output_dtype)
    kwargs.setdefault("grad_tensors", _RMS_TENSORS)
    kwargs.setdefault("op_class", "reduction")
    return _assert_gradient_batch_invariant(*args, **kwargs)


@pytest.fixture()
def contract() -> dict[str, Any]:
    return load_contract()


@pytest.fixture()
def manifest():
    return load_manifest()


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


def _identity_grad_op(scale: float = 1.0):
    def op(config: ConfigSpec, **kwargs: Any) -> dict[str, Any]:
        denom = float(kwargs["active_token_denominator"])
        order = tuple(kwargs["aggregation_order"])
        samples = {sample.sample_id: sample for sample in config.logical_batch.samples}
        dx: dict[tuple[str, int], torch.Tensor] = {}
        dweight: torch.Tensor | None = None
        for sample_id in order:
            sample = samples.get(sample_id)
            if sample is None:
                continue
            sample_weight = torch.zeros(2, dtype=torch.float32)
            for tok in sample.active_tokens():
                dx[(tok.sample_id, tok.token_position)] = torch.tensor(
                    scale * float(tok.token_position + 1), dtype=torch.bfloat16
                )
                sample_weight = (
                    sample_weight
                    + torch.tensor([float((tok.token_id % 7) + 1), 1.0], dtype=torch.float32)
                    * scale
                    / denom
                )
            dweight = sample_weight if dweight is None else dweight + sample_weight
        if dweight is None:
            dweight = torch.zeros(2, dtype=torch.float32)
        return {"dx": dx, "dweight": dweight}

    return op


def _physical_tensor_op(*, layout_sensitive: bool):
    """Return token grads as a physical tensor, so C2's restore map is exercised.

    With ``layout_sensitive`` the row value depends on the physical index rather
    than the logical identity, which is exactly the class of defect an adapter
    that ignores ``config.physical_layout`` can never surface.
    """

    def _row_value(key: tuple[str, int] | None, physical_index: int) -> float:
        if key is None:
            return 0.0
        if layout_sensitive:
            return float(physical_index)
        return float(sum(ord(ch) for ch in key[0]) + key[1])

    def op(config: ConfigSpec, **kwargs: Any) -> dict[str, Any]:
        layout = config.physical_layout
        if isinstance(layout, PaddedBatch):
            grid = [
                [_row_value(key, index) for index, key in enumerate(row)]
                for row in layout.restore_map
            ]
            dx = torch.tensor(grid, dtype=torch.bfloat16).unsqueeze(-1)
        else:
            flat = [_row_value(key, index) for index, key in enumerate(layout.restore_map)]
            dx = torch.tensor(flat, dtype=torch.bfloat16).unsqueeze(-1)
        # Integer-valued per-sample contributions: the N x B=1 aggregate matches
        # the B=N sum exactly, so any failure comes from the token grads.
        dweight = torch.zeros(2, dtype=torch.float32)
        for sample in config.logical_batch.samples:
            dweight = dweight + torch.tensor(
                [float(len(list(sample.tokens()))), 1.0], dtype=torch.float32
            )
        return {"dx": dx, "dweight": dweight}

    return op


def _drifting_grad_op():
    def op(config: ConfigSpec, **kwargs: Any) -> dict[str, Any]:
        result = _identity_grad_op()(config, **kwargs)
        # Extra B=1-only term: N independent B=1 grads no longer reconstruct BN.
        if len(config.logical_batch.samples) == 1:
            result["dweight"] = result["dweight"] + 1.0
        return result

    return op


class TestReportStructure:
    def test_accuracy_and_invariance_reported_separately(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _identity_grad_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
            op_name="test_op",
        )
        assert isinstance(report, GradientInvarianceReport)
        assert report.accuracy_reports
        assert report.invariance_reports
        assert report.singleton_aggregate_reports
        assert report.grad_tensor_names == ("dx", "dweight")
        assert report.loss_reduction == (
            "sum_over_active_tokens_then_optional_mean_by_active_count"
        )
        assert report.active_token_denominator > 0

    def test_report_contains_diagnostics(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _identity_grad_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
        )
        payload = report.to_dict()
        assert "first_failing_op" in payload
        assert "first_failing_tensor" in payload
        assert "singleton_aggregate_reports" in payload
        for inv in (*report.invariance_reports, *report.singleton_aggregate_reports):
            for detail in inv.details:
                assert detail.tensor_name
                assert detail.max_abs_error is not None
                assert detail.max_rel_error is not None
                assert detail.config_pair


class TestInvariance:
    def test_invariance_bitwise_zero_tolerance(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _identity_grad_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
        )
        for inv in (*report.invariance_reports, *report.singleton_aggregate_reports):
            for detail in inv.details:
                assert detail.judgment == "gradient_invariance"
                assert detail.atol == 0.0
                assert detail.rtol == 0.0
                assert detail.comparison_lhs_role == "transformed_config"
                assert detail.comparison_rhs_role == "canonical_config"
                assert detail.comparison_lhs_role != "singleton_aggregate"

    def test_identity_op_passes(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _identity_grad_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
        )
        assert report.passed
        assert report.first_failing_tensor is None

    def test_parameter_drift_fails_singleton_aggregate(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _drifting_grad_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
        )
        assert not report.passed
        assert report.first_failing_tensor == "dweight"
        assert any(not item.passed for item in report.singleton_aggregate_reports)

    def test_b1_bn_share_denominator_and_order(self, contract, manifest):
        seen: list[tuple[int, tuple[str, ...], str]] = []

        def op(config: ConfigSpec, **kwargs: Any) -> dict[str, Any]:
            seen.append(
                (
                    int(kwargs["active_token_denominator"]),
                    tuple(kwargs["aggregation_order"]),
                    str(kwargs["loss_reduction"]),
                )
            )
            return _identity_grad_op()(config, **kwargs)

        report = assert_gradient_batch_invariant(
            op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
        )
        assert report.passed
        assert len({item[0] for item in seen}) == 1
        assert len({item[1] for item in seen}) == 1
        assert all(item[0] == report.active_token_denominator for item in seen)
        assert all(
            item[2] == "sum_over_active_tokens_then_optional_mean_by_active_count" for item in seen
        )

    def test_missing_active_token_hard_fails(self, contract, manifest):
        def incomplete(config: ConfigSpec, **kwargs: Any) -> dict[str, Any]:
            result = _identity_grad_op()(config, **kwargs)
            result["dx"].pop(next(iter(result["dx"])))
            return result

        with pytest.raises(ValueError, match="C2 logical identity"):
            assert_gradient_batch_invariant(
                incomplete,
                contract=contract,
                manifest=manifest,
                backend_profile="cuda_bf16",
                provenance=_make_provenance(),
                gold_fn=_identity_grad_op(),
            )


class TestPhysicalLayout:
    """The C2 matrix must actually change what the operator sees.

    Before these guards every config fed the operator identical inputs, so the
    bitwise verdicts were tautologies rather than assertions.
    """

    def test_physical_tensor_is_restored_through_c2_map(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _physical_tensor_op(layout_sensitive=False),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_physical_tensor_op(layout_sensitive=False),
            op_name="physical_tensor_op",
        )
        assert report.passed, report.to_dict()
        covered = {inv.transformed_config_id for inv in report.invariance_reports}
        # packed, chunked and both pad sides all round-trip through restore.
        assert {"BN/chunked", "BN/permuted", "BN/padded_left", "BN/padded_right"} <= covered

    def test_layout_sensitive_op_is_detected(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _physical_tensor_op(layout_sensitive=True),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_physical_tensor_op(layout_sensitive=True),
            op_name="layout_sensitive_op",
        )
        assert not report.passed
        assert report.first_failing_tensor == "dx"
        failing = {inv.transformed_config_id for inv in report.invariance_reports if not inv.passed}
        assert "BN/padded_right" in failing
        assert "BN/permuted" in failing

    def test_bn_is_one_call_and_chunking_splits(self):
        from rl_engine.kernels.gtest.forward_invariance import build_config_matrix
        from rl_engine.kernels.gtest.gradient_adapters import make_gradient_runner
        from rl_engine.testing.ws1_workload import load_manifest as _load
        from rl_engine.testing.ws1_workload import singleton_aggregate_plan

        m = _load()
        configs = {config.config_id: config for config in build_config_matrix(m)}
        canonical = configs["BN/full"]
        plan = singleton_aggregate_plan(canonical.logical_batch)
        operator = load_adapter_operator("rms_norm", "pytorch")
        seen: list[tuple[int, ...]] = []
        original = operator.forward

        def spy(**kwargs: Any) -> Any:
            seen.append(tuple(kwargs["x"].shape))
            return original(**kwargs)

        operator.forward = spy
        run = make_gradient_runner(
            "rms_norm",
            operator,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            reference=False,
            hidden=8,
            backend_family="cuda",
            kernel_id="spy",
        )
        kwargs = {
            "active_token_denominator": canonical.logical_batch.active_token_count(),
            "loss_reduction": "sum_over_active_tokens_then_optional_mean_by_active_count",
            "aggregation_order": plan.aggregation_order,
        }

        seen.clear()
        run(canonical, **kwargs)
        total_tokens = len(canonical.logical_batch.logical_keys(active_only=False))
        assert seen == [(total_tokens, 8)], "B=N must be one batched call, not N x B=1"

        seen.clear()
        run(configs["BN/chunked"], **kwargs)
        assert len(seen) > 1, "chunked-prefill must split the call"
        assert sum(shape[0] for shape in seen) == total_tokens

        seen.clear()
        run(configs["B1-singleton_aggregate/full/s0"], **kwargs)
        assert len(seen) == 1
        assert seen[0][0] < total_tokens

        seen.clear()
        run(configs["BN/padded_right"], **kwargs)
        assert seen[0][0] > total_tokens, "padding must reach the operator"

    def test_non_differentiable_candidate_raises_missing_backward(self):
        from rl_engine.kernels.gtest.forward_invariance import build_config_matrix
        from rl_engine.kernels.gtest.gradient_adapters import make_gradient_runner
        from rl_engine.testing.ws1_workload import load_manifest as _load
        from rl_engine.testing.ws1_workload import singleton_aggregate_plan

        class _DetachedRMSNorm:
            """Stands in for a candidate wired straight to a C++ entry point."""

            def forward(self, **kwargs: Any) -> torch.Tensor:
                x = kwargs["x"]
                return torch.empty_like(x).copy_(x).detach()

        m = _load()
        canonical = next(c for c in build_config_matrix(m) if c.is_canonical)
        plan = singleton_aggregate_plan(canonical.logical_batch)
        run = make_gradient_runner(
            "rms_norm",
            _DetachedRMSNorm(),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            reference=True,
            hidden=8,
        )
        with pytest.raises(MissingBackwardError, match="missing backward is red"):
            run(
                canonical,
                active_token_denominator=canonical.logical_batch.active_token_count(),
                loss_reduction="sum_over_active_tokens_then_optional_mean_by_active_count",
                aggregation_order=plan.aggregation_order,
            )

    def test_pack_inactive_tokens_contribute_zero(self):
        from rl_engine.kernels.gtest.forward_invariance import build_config_matrix
        from rl_engine.kernels.gtest.gradient_adapters import make_gradient_runner
        from rl_engine.testing.ws1_workload import load_manifest as _load
        from rl_engine.testing.ws1_workload import singleton_aggregate_plan

        m = _load()
        canonical = next(c for c in build_config_matrix(m) if c.is_canonical)
        plan = singleton_aggregate_plan(canonical.logical_batch)
        run = make_gradient_runner(
            "pack",
            load_adapter_operator("pack", "pytorch"),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            reference=True,
            hidden=8,
        )
        grads = run(
            canonical,
            active_token_denominator=canonical.logical_batch.active_token_count(),
            loss_reduction="sum_over_active_tokens_then_optional_mean_by_active_count",
            aggregation_order=plan.aggregation_order,
        )
        inactive = {
            (token.sample_id, token.token_position)
            for sample in canonical.logical_batch.samples
            for token in sample.tokens()
            if not token.is_active
        }
        assert inactive, "fixture must contain inactive tokens for this guard to mean anything"
        for key in inactive:
            assert torch.count_nonzero(grads["dx"][key]) == 0


class TestAccuracy:
    def test_missing_reference_is_rejected(self, contract, manifest):
        with pytest.raises(ValueError, match="gold_fn is required"):
            assert_gradient_batch_invariant(
                _identity_grad_op(),
                contract=contract,
                manifest=manifest,
                backend_profile="cuda_bf16",
                provenance=_make_provenance(),
                gold_fn=None,
            )

    def test_accuracy_uses_c1_gradient_rows(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _identity_grad_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
        )
        spec = resolve_tolerance(
            contract,
            judgment="gradient_accuracy",
            op_class="reduction",
            dtype=torch.bfloat16,
            backend_profile="cuda_bf16",
        )
        for acc in report.accuracy_reports:
            for detail in acc.details:
                assert detail.judgment == "gradient_accuracy"
                assert detail.atol == spec.atol
                assert detail.rtol == spec.rtol

    def test_no_private_thresholds(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _identity_grad_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
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
    def test_required_profiles_share_report_schema(self, contract, manifest):
        keys = None
        for profile, family in (("cuda_bf16", "cuda"), ("triton_cuda_bf16", "triton")):
            report = assert_gradient_batch_invariant(
                _identity_grad_op(),
                contract=contract,
                manifest=manifest,
                backend_profile=profile,
                provenance=_make_provenance(profile, family, family),
                gold_fn=_identity_grad_op(),
            )
            assert report.passed
            payload_keys = set(report.to_dict())
            keys = payload_keys if keys is None else keys
            assert payload_keys == keys

    def test_cross_profile_fallback_fails(self, contract, manifest):
        report = assert_gradient_batch_invariant(
            _identity_grad_op(),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance("cuda_bf16", "cuda", "triton"),
            gold_fn=_identity_grad_op(),
        )
        assert report.provenance_valid is False
        assert report.passed is False

    def test_runtime_observation_mismatch_fails_closed(self, contract, manifest):
        def observed_op(config: ConfigSpec, **kwargs: Any) -> GradientObservation:
            return GradientObservation(
                grads=_identity_grad_op()(config, **kwargs),
                actual_backend="cuda",
                kernel_id="synthetic-test-candidate",
                output_dtype="bfloat16",
                device="cpu:test-double",
            )

        report = _assert_gradient_batch_invariant(
            observed_op,
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=_identity_grad_op(),
            grad_tensors=_RMS_TENSORS,
            op_class="reduction",
            candidate_id="synthetic-test-candidate",
            device="cpu:test-double",
            compute_capability="synthetic",
            observed_actual_backend="triton",
            observed_kernel_id="synthetic-test-candidate",
            observed_output_dtype="bfloat16",
        )
        assert report.metadata_valid is False
        assert report.passed is False


class TestAdapters:
    def test_required_ops_are_enumerable(self):
        names = set(adapter_names())
        for required in (
            "rms_norm",
            "qk_norm",
            "det_gemm",
            "attention",
            "embedding",
            "lm_head",
            "logp",
            "batch_invariant_logp",
            "rope",
            "silu",
            "swiglu",
            "pack",
        ):
            assert required in names
            adapter = get_adapter(required)
            assert adapter.tensors
            assert adapter.atomic_add == "forbidden"
            assert adapter.shape_dependent_bwd_accum == "forbidden"

    def test_stable_grad_names(self):
        assert tuple(tensor.name for tensor in get_adapter("mhc_pre_h_aggregate").tensors) == (
            "dresidual",
            "dpre",
        )
        assert tuple(t.name for t in get_adapter("rms_norm").tensors) == ("dx", "dweight")
        assert tuple(t.name for t in get_adapter("det_gemm").tensors) == ("dX", "dW")
        assert tuple(t.name for t in get_adapter("attention").tensors) == ("dQ", "dK", "dV")
        assert tuple(t.name for t in get_adapter("lm_head").tensors) == ("dhidden", "dweight")
        assert tuple(t.name for t in get_adapter("logp").tensors) == ("dlogits",)
        assert tuple(t.name for t in get_adapter("swiglu").tensors) == ("dgate", "dup")

    def test_kv_is_absent_not_required(self):
        adapter = get_adapter("kv_cache_attention")
        assert adapter.requirement == "absent_not_required"
        assert adapter.tensors == ()

    def test_status_matrix_has_no_untracked_red(self, manifest):
        rows = gradient_adapter_status_matrix(manifest)
        assert rows
        untracked = [row for row in rows if row.untracked_red]
        assert untracked == []
        tracked = [row for row in rows if row.tracked_red]
        assert tracked == []
        kv_rows = [row for row in rows if row.op_name == "kv_cache_attention"]
        assert kv_rows
        assert all(row.candidate_status == "absent_not_required" for row in kv_rows)
        pack_rows = [row for row in rows if row.op_name == "pack"]
        assert pack_rows
        assert all(row.adapter_registered for row in pack_rows)

    def test_profiles_do_not_borrow_candidates(self, manifest):
        rows = gradient_adapter_status_matrix(manifest)
        by_key = {(row.backend_profile, row.op_name): row for row in rows}
        for adapter in required_gradient_adapters():
            if adapter.requirement != "required":
                continue
            cuda = by_key[("cuda_bf16", adapter.op_name)]
            triton = by_key[("triton_cuda_bf16", adapter.op_name)]
            if cuda.candidate_status != "declared" or triton.candidate_status != "declared":
                continue
            assert cuda.candidate_path != triton.candidate_path
            assert cuda.expected_backend_id != triton.expected_backend_id

    def test_no_atomic_add_in_bi_sources(self):
        for adapter in GRADIENT_ADAPTERS.values():
            for path in listed_source_paths(adapter):
                assert path.is_file(), path
                text = path.read_text(encoding="utf-8")
                assert "atomicAdd" not in text, path


class TestRealAdapter:
    def _run_native_rms_norm(self, contract, manifest):
        gold = load_adapter_gold("rms_norm")
        candidate = load_adapter_operator("rms_norm", "pytorch")
        return assert_gradient_batch_invariant(
            make_gradient_runner(
                "rms_norm",
                candidate,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
                reference=False,
                hidden=8,
                backend_family="cuda",
                kernel_id="pytorch-rms-norm",
            ),
            contract=contract,
            manifest=manifest,
            backend_profile="cuda_bf16",
            provenance=_make_provenance(),
            gold_fn=make_gradient_runner(
                "rms_norm",
                gold,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
                reference=True,
                hidden=8,
            ),
            grad_tensors=get_adapter("rms_norm").tensors,
            op_class="reduction",
            op_name="rms_norm",
            candidate_id="pytorch-rms-norm",
            device="cpu:test-double",
            compute_capability="synthetic",
            observed_actual_backend="cuda",
            observed_kernel_id="pytorch-rms-norm",
            observed_output_dtype="bfloat16",
        )

    def test_native_rms_norm_gradient_accuracy_passes(self, contract, manifest):
        report = self._run_native_rms_norm(contract, manifest)
        assert report.accuracy_reports
        assert all(item.passed for item in report.accuracy_reports), report.to_dict()
        assert report.provenance_valid
        assert report.metadata_valid

    def test_native_rms_norm_padding_and_permutation_are_bitwise(self, contract, manifest):
        report = self._run_native_rms_norm(contract, manifest)
        by_config = {inv.transformed_config_id: inv for inv in report.invariance_reports}
        for config_id in ("BN/permuted", "BN/padded_left", "BN/padded_right"):
            assert by_config[config_id].passed, config_id
            for detail in by_config[config_id].details:
                assert detail.max_abs_error == 0.0

    def test_native_rms_norm_chunk_non_invariance_is_detected(self, contract, manifest):
        # NativeRMSNormOp is the FP32 reference, not a batch-invariant kernel:
        # its dweight reduction re-associates when the token stream is chunked
        # or split into N x B=1 runs. The harness must surface that, and this is
        # the assertion that fails if adapters stop honouring the layout.
        report = self._run_native_rms_norm(contract, manifest)
        assert not report.passed
        assert report.first_failing_tensor == "dweight"
        chunked = next(
            inv for inv in report.invariance_reports if inv.transformed_config_id == "BN/chunked"
        )
        assert not chunked.passed
        assert report.singleton_aggregate_reports
        assert any(not item.passed for item in report.singleton_aggregate_reports)
