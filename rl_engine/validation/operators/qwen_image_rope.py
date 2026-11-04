# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image operator-only four-judgment gate using the standard C3/C4 APIs.

Reuses C2's logical sample/config transforms and C1's elementwise tolerances.
It does not certify the Dense Qwen3 chain, distributed execution or Qwen-Image
model assembly. The operator has its own OP_SPECS entry and gradient adapter.
"""

import torch

from rl_engine.contracts.numerical import BackendProvenance
from rl_engine.ops.rope.multi_axis_rope import get_multi_axis_rope_op
from rl_engine.reference.rope.multi_axis_rope import NativeMultiAxisRoPEOp
from rl_engine.validation.operators.forward_invariance import assert_forward_batch_invariant
from rl_engine.validation.operators.gradient_adapters import (
    get_adapter,
    make_forward_runner,
    make_gradient_runner,
)
from rl_engine.validation.operators.gradient_invariance import assert_gradient_batch_invariant


def run_multi_axis_rope_ws1(backend):
    if backend not in ("cuda", "triton"):
        raise ValueError("WS1 GPU evidence requires an explicit cuda or triton backend")
    device = torch.device("cuda")
    profile = "cuda_bf16" if backend == "cuda" else "triton_cuda_bf16"
    op = get_multi_axis_rope_op(backend, device=device)
    gold = NativeMultiAxisRoPEOp().forward_fp32
    trace = op.trace()
    dtype = torch.bfloat16
    provenance = BackendProvenance(
        backend_profile=profile,
        requested_backend=backend,
        actual_backend=backend,
        execution_dtype="bfloat16",
        accumulation_dtype="float32",
        output_dtype="bfloat16",
        reference_dtype="float32",
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )
    metadata = {
        "backend_profile": profile,
        "provenance": provenance,
        "op_class": "elementwise",
        "dtype": dtype,
        "op_name": "multi_axis_rope",
        "candidate_id": trace["actual_backend"],
        "device": str(device),
        "compute_capability": "sm%d%d" % torch.cuda.get_device_capability(),
        "observed_actual_backend": backend,
        "observed_kernel_id": trace["kernel_id"],
        "observed_output_dtype": "bfloat16",
    }
    runner_options = {"device": device, "dtype": dtype, "n_heads": 3, "head_dim": 128}
    candidate_options = {"backend_family": backend, "kernel_id": trace["kernel_id"]}
    forward = assert_forward_batch_invariant(
        make_forward_runner(
            "multi_axis_rope", op, reference=False, **runner_options, **candidate_options
        ),
        gold_fn=make_forward_runner("multi_axis_rope", gold, reference=True, **runner_options),
        include_logprob_smoke=False,
        **metadata,
    )
    gradient = assert_gradient_batch_invariant(
        make_gradient_runner(
            "multi_axis_rope", op, reference=False, **runner_options, **candidate_options
        ),
        gold_fn=make_gradient_runner("multi_axis_rope", gold, reference=True, **runner_options),
        grad_tensors=get_adapter("multi_axis_rope").tensors,
        **metadata,
    )
    return {
        "scope": "Qwen-Image multi_axis_rope operator; C1 elementwise and C2 config transforms",
        "trace": trace,
        "passed": forward.passed and gradient.passed,
        "forward": forward.to_dict(),
        "gradient": gradient.to_dict(),
    }
