# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse

import pytest
import torch

from rl_engine.kernels.gtest.op_checks import CandidateSpec, OperatorCase, run_operator_suite
from rl_engine.kernels.gtest.operator_specs import (
    make_candidate,
    make_operator_case,
    operator_names,
)
from rl_engine.kernels.gtest.tolerance import BackendProvenance, ContractResolveError
from rl_engine.kernels.ops.pytorch.linear.embedding import NativeEmbeddingOp
from rl_engine.kernels.ops.pytorch.linear.lm_head import NativeLMHeadOp
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp


def _logp_case(name: str, dtype: torch.dtype, *, seed: int = 0) -> OperatorCase:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 8, 257, dtype=dtype, generator=generator)
    token_ids = torch.randint(0, logits.size(-1), (2, 8), generator=generator)
    return OperatorCase(
        name=name,
        op_class="logprob",
        dtype=dtype,
        inputs={"logits": logits, "token_ids": token_ids},
        gold_fn=NativeLogpOp().forward_fp32,
    )


def _logp_backward_case(name: str, *, seed: int = 0) -> OperatorCase:
    case = _logp_case(name, torch.float32, seed=seed)
    return OperatorCase(
        name=case.name,
        op_class=case.op_class,
        dtype=case.dtype,
        inputs=case.inputs,
        gold_fn=case.gold_fn,
        grad_input_names=("logits",),
    )


def _embedding_case(name: str, *, seed: int = 0) -> OperatorCase:
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.tensor([[1, 7, 3], [7, 0, 5]], dtype=torch.long)
    weight = torch.randn(11, 5, generator=generator)
    return OperatorCase(
        name=name,
        op_class="elementwise",
        dtype=torch.float32,
        inputs={"token_ids": token_ids, "weight": weight},
        gold_fn=NativeEmbeddingOp().forward_fp32,
        grad_input_names=("weight",),
    )


def _lm_head_case(name: str, *, seed: int = 0) -> OperatorCase:
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(2, 3, 5, generator=generator)
    weight = torch.randn(13, 5, generator=generator)
    return OperatorCase(
        name=name,
        op_class="reduction",
        dtype=torch.float32,
        inputs={"hidden": hidden, "weight": weight, "bias": None},
        gold_fn=NativeLMHeadOp().forward_fp32,
        grad_input_names=("hidden", "weight"),
    )


def _spec_args(op: str) -> argparse.Namespace:
    return argparse.Namespace(
        op=op,
        candidate="native",
        arch_key=None,
        batch=1,
        seq=2,
        vocab=17,
        seed=123,
        input_mode="constant",
        constant_value=0.5,
        token_value=3,
        normalized_dim=8,
        k_dim=8,
        n_dim=8,
        n_heads=2,
        head_dim=8,
        theta=1.0e6,
        eps=1.0e-6,
    )


def test_logp_native_candidate_suite_passes():
    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="native-logp", backend="pytorch", fn=NativeLogpOp())],
        cases=[
            _logp_case("fp32", torch.float32, seed=1),
            _logp_case("bf16", torch.bfloat16, seed=2),
            _logp_case("fp16", torch.float16, seed=3),
        ],
    )

    assert report.passed
    assert report.pass_rate == 1.0
    assert report.candidates[0].passed_outputs == 3
    assert all(case.passed for case in report.candidates[0].cases)


def test_embedding_native_candidate_suite_passes_issue_108_helper():
    report = run_operator_suite(
        "embedding",
        candidates=[
            CandidateSpec(name="native-embedding", backend="pytorch", fn=NativeEmbeddingOp())
        ],
        cases=[_embedding_case("fp32", seed=10)],
        check_grad=True,
    )

    assert report.passed
    gradient = report.candidates[0].cases[0].outputs[1]
    assert gradient.message == "gradient:weight"
    assert gradient.judgment == "gradient_accuracy"
    assert gradient.comparison_lhs_role == "bf16_candidate"
    assert gradient.comparison_rhs_role == "fp32_reference"


def test_lm_head_native_candidate_suite_passes_issue_108_helper():
    report = run_operator_suite(
        "lm_head",
        candidates=[CandidateSpec(name="native-lm-head", backend="pytorch", fn=NativeLMHeadOp())],
        cases=[_lm_head_case("fp32", seed=11)],
        check_grad=True,
    )

    assert report.passed
    messages = [output.message for output in report.candidates[0].cases[0].outputs]
    assert "gradient:hidden" in messages
    assert "gradient:weight" in messages


def test_issue151_ops_pass_shared_issue_108_spec_path():
    assert {"embedding", "lm_head", "qk_norm", "pack"}.issubset(operator_names())

    for op_name in ("embedding", "lm_head", "qk_norm", "pack"):
        args = _spec_args(op_name)
        report = run_operator_suite(
            op_name,
            candidates=[make_candidate(args)],
            cases=[make_operator_case(args, torch.float32, torch.device("cpu"))],
            check_grad=True,
        )
        assert report.passed


def test_suite_reports_failure_for_bad_candidate():
    def bad_logp(logits, token_ids):
        del token_ids
        return torch.zeros(logits.shape[:-1], dtype=logits.dtype)

    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="bad-logp", backend="test", fn=bad_logp)],
        cases=[_logp_case("fp32", torch.float32, seed=5)],
    )

    output = report.candidates[0].cases[0].outputs[0]
    assert not report.passed
    assert report.pass_rate == 0.0
    assert output.max_abs_error > 0.0


def test_suite_report_to_dict_contains_error_metrics():
    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="native-logp", backend="pytorch", fn=NativeLogpOp())],
        cases=[_logp_case("fp32", torch.float32, seed=6)],
    )

    data = report.to_dict()
    output = data["candidates"][0]["cases"][0]["outputs"][0]
    assert data["suite_name"] == "logp"
    assert "max_abs_error" in output
    assert "atol" in output
    assert "passed" in output


def test_ws1_report_persists_roles_and_backend_provenance():
    provenance = BackendProvenance(
        backend_profile="cuda_bf16",
        requested_backend="cuda",
        actual_backend="cuda",
        execution_dtype="bfloat16",
        accumulation_dtype="float32",
        output_dtype="bfloat16",
        reference_dtype="float32",
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )
    report = run_operator_suite(
        "logp",
        candidates=[
            CandidateSpec(
                name="cuda-logp",
                backend="cuda",
                fn=NativeLogpOp(),
                provenance=provenance,
            )
        ],
        cases=[_logp_case("bf16", torch.bfloat16, seed=12)],
    )
    output = report.candidates[0].cases[0].outputs[0]
    assert output.judgment == "forward_accuracy"
    assert output.comparison_lhs_role == "bf16_candidate"
    assert output.comparison_rhs_role == "fp32_reference"
    data = report.to_dict()["candidates"][0]
    assert data["backend_provenance"]["actual_backend"] == "cuda"
    assert "baseline" not in data["cases"][0]["outputs"][0]


def test_ws1_report_accepts_triton_backend_provenance():
    provenance = BackendProvenance(
        backend_profile="triton_cuda_bf16",
        requested_backend="triton",
        actual_backend="triton",
        execution_dtype="bfloat16",
        accumulation_dtype="float32",
        output_dtype="bfloat16",
        reference_dtype="float32",
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )
    report = run_operator_suite(
        "logp",
        candidates=[
            CandidateSpec(
                name="triton-logp",
                backend="triton",
                fn=NativeLogpOp(),
                provenance=provenance,
            )
        ],
        cases=[_logp_case("bf16", torch.bfloat16, seed=15)],
    )
    output = report.candidates[0].cases[0].outputs[0]
    assert output.judgment == "forward_accuracy"
    assert output.comparison_lhs_role == "bf16_candidate"
    assert output.comparison_rhs_role == "fp32_reference"
    data = report.to_dict()["candidates"][0]
    assert data["backend_provenance"]["actual_backend"] == "triton"
    assert "baseline" not in data["cases"][0]["outputs"][0]


def test_ws1_report_rejects_backend_provenance_mismatch():
    provenance = BackendProvenance(
        backend_profile="cuda_bf16",
        requested_backend="cuda",
        actual_backend="cuda",
        execution_dtype="bfloat16",
        accumulation_dtype="float32",
        output_dtype="bfloat16",
        reference_dtype="float32",
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )
    with pytest.raises(ContractResolveError, match="actual_backend"):
        run_operator_suite(
            "logp",
            candidates=[
                CandidateSpec(
                    name="bad",
                    backend="triton",
                    fn=NativeLogpOp(),
                    provenance=provenance,
                )
            ],
            cases=[_logp_case("bf16", torch.bfloat16, seed=13)],
        )


def test_ws1_report_checks_observed_output_dtype_against_provenance():
    provenance = BackendProvenance(
        backend_profile="cuda_bf16",
        requested_backend="cuda",
        actual_backend="cuda",
        execution_dtype="bfloat16",
        accumulation_dtype="float32",
        output_dtype="bfloat16",
        reference_dtype="float32",
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )

    def wrong_output_dtype(logits, token_ids):
        return NativeLogpOp().forward(logits, token_ids).float()

    with pytest.raises(ContractResolveError, match="candidate output dtype"):
        run_operator_suite(
            "logp",
            candidates=[
                CandidateSpec(
                    name="wrong-output",
                    backend="cuda",
                    fn=wrong_output_dtype,
                    provenance=provenance,
                )
            ],
            cases=[_logp_case("bf16", torch.bfloat16, seed=14)],
        )

    wrong_gold_case = _logp_case("bf16", torch.bfloat16, seed=14)
    wrong_gold_case = OperatorCase(
        name=wrong_gold_case.name,
        op_class=wrong_gold_case.op_class,
        dtype=wrong_gold_case.dtype,
        inputs=wrong_gold_case.inputs,
        gold_fn=lambda **inputs: NativeLogpOp().forward(**inputs),
        grad_input_names=wrong_gold_case.grad_input_names,
    )
    with pytest.raises(ContractResolveError, match="gold output dtype"):
        run_operator_suite(
            "logp",
            candidates=[
                CandidateSpec(
                    name="wrong-gold-output",
                    backend="cuda",
                    fn=NativeLogpOp(),
                    provenance=provenance,
                )
            ],
            cases=[wrong_gold_case],
        )


def test_candidate_arch_key_uses_tolerance_override():
    def slightly_shifted_logp(logits, token_ids):
        return NativeLogpOp().forward_fp32(logits, token_ids) + 0.02

    contract = {
        "accuracy": {
            "default": {
                "logprob": {
                    "float32": {"atol": 1.0e-5, "rtol": 0.0},
                }
            },
            "arch_overrides": {
                "testarch": {
                    "logprob": {
                        "float32": {"atol": 5.0e-2, "rtol": 0.0},
                    }
                }
            },
        }
    }
    report = run_operator_suite(
        "logp",
        candidates=[
            CandidateSpec(
                name="shifted-logp",
                backend="test",
                fn=slightly_shifted_logp,
                arch_key="testarch",
            )
        ],
        cases=[_logp_case("fp32", torch.float32, seed=7)],
        contract=contract,
    )

    output = report.candidates[0].cases[0].outputs[0]
    assert report.passed
    assert output.atol == 5.0e-2


def test_legacy_contract_rejects_non_forward_judgment():
    """Legacy accuracy mirrors must not be reused as gradient thresholds."""
    from rl_engine.kernels.gtest.op_checks import _resolve_tolerance

    contract = {
        "accuracy": {
            "default": {
                "logprob": {
                    "float32": {"atol": 1.0e-5, "rtol": 0.0},
                }
            }
        }
    }
    with pytest.raises(ContractResolveError, match="legacy accuracy contracts"):
        _resolve_tolerance(
            contract,
            op_class="logprob",
            dtype=torch.float32,
            judgment="gradient_accuracy",
        )


def test_logp_native_candidate_backward_suite_passes():
    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="native-logp", backend="pytorch", fn=NativeLogpOp())],
        cases=[_logp_backward_case("fp32", seed=8)],
        check_grad=True,
    )

    assert report.passed
    assert report.candidates[0].passed_outputs == 2
    assert report.candidates[0].cases[0].outputs[1].message == "gradient:logits"


def test_backward_suite_reports_failure_for_bad_gradient():
    def bad_grad_logp(logits, token_ids):
        values = NativeLogpOp().forward_fp32(logits, token_ids)
        return values.detach() + logits.sum(dim=-1) * 0.0

    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="bad-grad-logp", backend="test", fn=bad_grad_logp)],
        cases=[_logp_backward_case("fp32", seed=9)],
        check_grad=True,
    )

    gradient_output = report.candidates[0].cases[0].outputs[1]
    assert not report.passed
    assert gradient_output.message == "gradient:logits"
    assert gradient_output.max_abs_error > 0.0


def test_random_grad_mode_catches_nonuniform_upstream_gradient_bug():
    # Forward is identity, so only a non-uniform upstream gradient can expose
    # the intentionally wrong backward below.
    class MeanUpstreamIdentity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            # Wrong for random upstream gradients, but correct when all values are 1.
            return grad_output.mean().expand_as(grad_output)

    def bad_identity(x):
        return MeanUpstreamIdentity.apply(x)

    case = OperatorCase(
        name="identity",
        op_class="elementwise",
        dtype=torch.float32,
        inputs={"x": torch.randn(8, dtype=torch.float32)},
        gold_fn=lambda x: x,
        grad_input_names=("x",),
    )

    ones_report = run_operator_suite(
        "identity",
        candidates=[CandidateSpec(name="bad-identity", backend="test", fn=bad_identity)],
        cases=[case],
        check_grad=True,
        grad_mode="ones",
    )
    # ones passes by design; random must fail and prove the stricter path works.
    random_report = run_operator_suite(
        "identity",
        candidates=[CandidateSpec(name="bad-identity", backend="test", fn=bad_identity)],
        cases=[case],
        check_grad=True,
        grad_mode="random",
        grad_seed=7,
    )

    assert ones_report.passed
    gradient_output = random_report.candidates[0].cases[0].outputs[1]
    assert not random_report.passed
    assert gradient_output.message == "gradient:x"
    assert gradient_output.max_abs_error > 0.0


class _FrozenScaledIdentity(torch.autograd.Function):
    """Identity forward with a deliberately NON-derived backward (3x).

    autograd through any plain forward of this op yields the TRUE 1x
    gradient; the frozen contract pins 3x. This difference makes the gold
    gradient SOURCE observable: differentiating gold_fn must fail the
    bitwise bar, while gold_backward_fn (the frozen chain) must pass it.
    """

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 3.0


def _frozen_scaled_identity(x):
    return _FrozenScaledIdentity.apply(x)


def _bump_one_ulp_bf16(t: torch.Tensor) -> torch.Tensor:
    out = t.detach().clone()
    view = out.view(torch.int16)
    view[0, 0] += 1
    return out


def _strict_identity_case() -> OperatorCase:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(4, 5, generator=generator).to(torch.bfloat16)
    return OperatorCase(
        name="strict-identity-bf16",
        op_class="elementwise",
        dtype=torch.bfloat16,
        inputs={"x": x},
        gold_fn=lambda x: x.float(),
        grad_input_names=("x",),
        bitwise_strict=True,
        gold_backward_fn=_frozen_scaled_identity,
    )


def test_bitwise_strict_forward_rejects_one_ulp_despite_tolerance():
    # one bf16 ulp is far inside the generic tolerance, but the strict bar
    # compares bit patterns: the bumped candidate must FAIL.
    report = run_operator_suite(
        "identity",
        candidates=[
            CandidateSpec(name="bumped", backend="test", fn=lambda x: _bump_one_ulp_bf16(x))
        ],
        cases=[_strict_identity_case()],
        check_grad=False,
    )
    forward = report.candidates[0].cases[0].outputs[0]
    assert not forward.passed
    assert "bitwise: bit-pattern mismatch" in forward.message
    # the tolerance judgment alone WOULD have passed it -- stats stay diagnostic
    assert forward.max_abs_error > 0.0


def test_bitwise_strict_uses_frozen_gold_backward_not_autograd_graph():
    # candidate implements the frozen 3x backward: passes against
    # gold_backward_fn but would differ from autograd-through-gold_fn (1x).
    report = run_operator_suite(
        "identity",
        candidates=[CandidateSpec(name="frozen", backend="test", fn=_frozen_scaled_identity)],
        cases=[_strict_identity_case()],
        check_grad=True,
    )
    assert report.passed
    for output in report.candidates[0].cases[0].outputs:
        assert output.passed
        assert "bitwise" in output.message


def test_bitwise_strict_gradient_rejects_one_ulp():
    class _FrozenScaledIdentityOffByOne(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            return _bump_one_ulp_bf16(grad_output * 3.0)

    report = run_operator_suite(
        "identity",
        candidates=[
            CandidateSpec(
                name="grad-off-one-ulp", backend="test", fn=_FrozenScaledIdentityOffByOne.apply
            )
        ],
        cases=[_strict_identity_case()],
        check_grad=True,
    )
    gradient = report.candidates[0].cases[0].outputs[1]
    assert not report.passed
    assert not gradient.passed
    assert "bitwise: bit-pattern mismatch" in gradient.message
