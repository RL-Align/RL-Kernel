# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.kernels.gtest.operator_inputs import make_operator_inputs, operator_shape_name
from rl_engine.kernels.gtest.op_checks import CandidateSpec, OperatorCase


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    op_class: str
    gold_path: str
    registry_name: str
    candidate_paths: dict[str, str]


def _load_object(path: str) -> Any:
    module_path, object_name = path.rsplit(".", 1)
    # dynamic loading ops
    module = importlib.import_module(module_path)
    return getattr(module, object_name)


OP_SPECS = {
    "logp": OperatorSpec(
        name="logp",
        op_class="logprob",
        gold_path="rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp",
        registry_name="logp",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp",
            "cuda": "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpGenericOp",
            "cuda-generic": "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpGenericOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpSM90Op",
        },
    ),
}


class _LogpSM90CandidateAdapter:
    def __init__(self, candidate: Any) -> None:
        self._candidate = candidate

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        orig_shape = logits.shape[:-1]
        logits_2d = logits.contiguous().view(-1, logits.size(-1))
        labels_1d = token_ids.contiguous().view(-1)
        return self._candidate(logits_2d, labels_1d).view(orig_shape)


def operator_names() -> tuple[str, ...]:
    return tuple(OP_SPECS)


def make_operator_case(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> OperatorCase:
    spec = OP_SPECS[args.op]
    gold_op = _load_object(spec.gold_path)()
    return OperatorCase(
        name=f"{args.op}-{dtype}-{operator_shape_name(args.op, args)}",
        op_class=spec.op_class,
        dtype=dtype,
        inputs=make_operator_inputs(args.op, args, dtype, device),
        gold_fn=gold_op.forward_fp32,
    )


def make_candidate(args: argparse.Namespace) -> CandidateSpec:
    spec = OP_SPECS[args.op]
    candidate_name = "pytorch" if args.candidate == "native" else args.candidate

    if candidate_name in spec.candidate_paths:
        candidate_op = _load_object(spec.candidate_paths[candidate_name])()
        if args.op == "logp" and candidate_name == "cuda-sm90":
            candidate_op = _LogpSM90CandidateAdapter(candidate_op)
        return CandidateSpec(
            name=f"{candidate_name}-{args.op}",
            backend=candidate_name,
            arch_key=args.arch_key,
            fn=candidate_op,
        )

    if candidate_name == "registry":
        from rl_engine.kernels.registry import kernel_registry

        return CandidateSpec(
            name=f"registry-{args.op}",
            backend="registry",
            arch_key=args.arch_key,
            fn=kernel_registry.get_op(spec.registry_name),
        )

    supported = sorted([*spec.candidate_paths, "native", "registry"])
    raise ValueError(
        f"unsupported candidate {args.candidate!r} for op {args.op!r}; "
        f"supported candidates: {', '.join(supported)}"
    )
