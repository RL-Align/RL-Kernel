# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Instrument the full pinned H3 provider with the actual RL-Kernel block graph.

Input/output projections and both text-refiner layers retain provider arithmetic.
Each of the 50 distinct blocks executes the existing candidate graph with its own
parameters. No weights are copied from block zero. This is integration evidence,
not qualification of the reference/interim bindings as final RFC kernels.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from types import MethodType

import torch

from rl_engine.validation.models import h3_block

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PROVIDER_REVISION = "f53d552036a0d1bd5570782a39cd40cfabf112bc"


def validate_full_structure(model):
    expected = dict(
        num_layers=50,
        num_refiner_layers=2,
        hidden_size=5376,
        num_attention_heads=56,
        attention_head_dim=128,
    )
    for key, value in expected.items():
        if getattr(model.config, key) != value:
            raise ValueError(f"full H3 requires {key}={value}")
    blocks = model.transformer_blocks
    if len(blocks) != 50 or len({id(b) for b in blocks}) != 50:
        raise ValueError("50 distinct transformer blocks are required")
    owned = set()
    for i, block in enumerate(blocks):
        params = list(block.parameters())
        if not params or any(id(p) in owned for p in params):
            raise ValueError(f"block {i} has missing or shared parameters")
        owned.update(id(p) for p in params)
    required = (
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "time_proj",
        "time_embedder",
        "token_refiner",
        "norm_out",
        "proj_out",
        "audio_proj_out",
    )
    if any(not hasattr(model, name) for name in required):
        raise ValueError("full-model boundary components are missing")
    refiners = model.token_refiner.refiner_blocks
    if len(refiners) != 2 or len({id(b) for b in refiners}) != 2:
        raise ValueError("two distinct real refiner blocks are required")
    for block in refiners:
        params = list(block.parameters())
        if not params or any(id(p) in owned for p in params):
            raise ValueError("refiner parameters are missing or shared")
        owned.update(id(p) for p in params)
    return required


class FullModelTrace:
    def __init__(self):
        self.calls = Counter()
        self.backward_calls = Counter()
        self.snapshots = {}
        self.capture = False
        self.actual_backends = {}
        self.candidate = False
        self.candidate_calls = Counter()
        self.forward_order = []
        self.backward_order = []
        self.execution_order = []
        self._events = set()
        self.boundary_bindings = {}

    def record_event(self, name):
        if name not in self._events:
            self._events.add(name)
            self.execution_order.append(name)

    def observe(self, name, value):
        self.calls[name] += 1
        self.record_event("nodes." + name)
        if name not in self.forward_order:
            self.forward_order.append(name)
        if not isinstance(value, torch.Tensor):
            return
        if self.capture:
            self.snapshots[name] = value.detach().cpu()
        if value.requires_grad:

            def record(grad):
                self.backward_calls[name] += 1
                self.record_event("nodes." + name + ".grad")
                if name not in self.backward_order:
                    self.backward_order.append(name)
                if self.capture:
                    self.snapshots[name + ".grad"] = grad.detach().cpu()

            value.register_hook(record)

    def validate(self, *, backward=False):
        required = (
            [f"block.{i}" for i in range(50)]
            + ["refiner.0", "refiner.1"]
            + [
                "proj_in",
                "audio_proj_in",
                "context_embedder",
                "token_refiner",
                "time_embedder",
                "norm_out",
                "proj_out",
                "audio_proj_out",
            ]
        )
        missing = [name for name in required if not self.calls[name]]
        if backward:
            missing += [name + ".backward" for name in required if not self.backward_calls[name]]
        if missing:
            raise ValueError(f"incomplete full-model execution: {missing}")
        if self.candidate:
            missing_nodes = [
                f"block.{i}.{node.name}"
                for i in range(50)
                for node in h3_block.NODES
                if not self.candidate_calls[f"block.{i}.{node.name}"]
            ]
            if missing_nodes:
                raise ValueError(f"candidate nodes did not execute: {missing_nodes}")
        return dict(
            candidate_calls=dict(self.candidate_calls),
            execution_order=self.execution_order,
            forward_order=self.forward_order,
            backward_order=self.backward_order,
            actual_backends=self.actual_backends,
            calls=dict(self.calls),
            backward_calls=dict(self.backward_calls),
            bindings={
                n.name: dict(
                    rfc_row=n.binding.rfc_row, status=n.binding.status, backend=n.binding.backend
                )
                for n in h3_block.NODES
            },
            boundary_binding=(
                "strict rows + provider" if self.boundary_bindings else "provider/reference"
            ),
            boundary_bindings=self.boundary_bindings,
            full_transformer_blocks=50,
            refiner_layers=2,
        )


def require_candidate_cuda(ops):
    backends = ops.backends()
    for name in ("projection", "norm", "gate"):
        if ".cuda." not in backends[name]:
            raise ValueError(f"candidate {name} fell back from its CUDA implementation")
    return backends


# Boundary modules bound to RL-Kernel rows by ``strict_boundaries``; the others keep provider
# arithmetic and are reported as such (``proj_in``/``proj_out`` belong to the video-projection
# rows claimed by another contributor; ``time_embedder``/``norm_out`` have their own rows
# but are not part of this binding yet).
STRICT_BOUNDARY_ROWS = {
    "audio_proj_in": "audio_input_projection_fp32",
    "audio_proj_out": "audio_output_projection_fp32",
    "context_embedder": "text_context_projection",
    "token_refiner": "token_refiner_block",
}
PROVIDER_BOUNDARIES = ("proj_in", "proj_out", "time_embedder", "norm_out")


@contextmanager
def _strict_boundaries(model, trace, ops, native=frozenset()):
    """Rebind the audio projections, the text projection and the whole token refiner to rows."""

    from rl_engine.runtime.registry import KernelRegistry
    from rl_engine.validation.models import h3_refiner

    registry = KernelRegistry()
    row_ops = {
        name: registry.get_op(row, device="cuda")
        for name, row in STRICT_BOUNDARY_ROWS.items()
        if name != "token_refiner" and name not in native
    }
    originals = []

    def project(name):
        module, op = getattr(model, name), row_ops[name]

        def forward(x):
            flat = x.reshape(-1, x.shape[-1])
            out = op.forward(flat, module.weight, module.bias)
            return out.view(*x.shape[:-1], out.shape[-1])

        originals.append((module, module.forward))
        module.forward = forward

    for name in row_ops:
        project(name)

    refiner = model.token_refiner
    params = dict(model.named_parameters())

    def refine(hidden):
        if hidden.dim() != 3:
            raise ValueError("token refiner expects (batch, tokens, hidden)")
        return h3_refiner.run_refiner(
            "candidate",
            params,
            hidden,
            ops,
            num_blocks=len(refiner.refiner_blocks),
            on_block=lambda b, h: trace.observe(f"refiner.{b}", h),
            observer_factory=lambda b: (
                lambda node, _b=b: trace.candidate_calls.update([f"refiner.{_b}.{node}"])
            ),
        )

    if "token_refiner" not in native:
        originals.append((refiner, refiner.forward))
        refiner.forward = refine
    trace.boundary_bindings = {
        **{
            n: dict(row=r, status="provider" if n in native else "row")
            for n, r in STRICT_BOUNDARY_ROWS.items()
        },
        **{n: dict(row=None, status="provider") for n in PROVIDER_BOUNDARIES},
    }
    try:
        yield
    finally:
        for module, _original in originals:
            module.__dict__.pop("forward", None)  # back to the class method


@contextmanager
def full_model_execution(
    model,
    *,
    candidate: bool,
    trace: FullModelTrace,
    ops=None,
    provider_nodes=(),
    strict_boundaries: bool = False,
    native_boundaries=(),
):
    """Temporarily bind each real block, preserving module parameters and restoring on error.

    ``provider_nodes`` (diagnostic hybrids only) names candidate graph nodes that run
    their provider replay in every block; such a run localises drift and is never a
    candidate result.
    """
    unknown = set(provider_nodes) - {n.name for n in h3_block.NODES}
    if unknown:
        raise ValueError(f"unknown block nodes: {sorted(unknown)}")
    boundaries = validate_full_structure(model)
    originals, handles = [], []
    trace.candidate = candidate
    if strict_boundaries and not candidate:
        raise ValueError("strict_boundaries applies to candidate execution only")
    unknown_boundaries = set(native_boundaries) - set(STRICT_BOUNDARY_ROWS)
    if unknown_boundaries:
        raise ValueError(f"unknown boundary modules: {sorted(unknown_boundaries)}")
    if candidate:
        ops = ops or h3_block.CandidateOps()
        if hasattr(ops, "backends"):
            trace.actual_backends = require_candidate_cuda(ops)
    boundary_context = (
        _strict_boundaries(model, trace, ops, frozenset(native_boundaries))
        if strict_boundaries
        else None
    )
    try:
        if boundary_context is not None:
            boundary_context.__enter__()
        for i, block in enumerate(model.transformer_blocks):
            if candidate:
                original = block.forward
                originals.append((block, original))

                def forward(
                    self,
                    hidden_states,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    attention_mask=None,
                    _block_index=i,
                ):
                    if attention_mask is not None:
                        raise ValueError(
                            "the qualified H3 candidate requires an unmasked packed sequence"
                        )
                    weights = {h3_block.BLOCK + "." + n: p for n, p in self.named_parameters()}
                    ctx = dict(
                        weights=weights,
                        hidden=hidden_states,
                        temb=temb,
                        index=adaln_indices,
                        rope=rotary_emb,
                        provider_nodes=frozenset(provider_nodes),
                        node_observer=lambda name: trace.candidate_calls.update(
                            [f"block.{_block_index}.{name}"]
                        ),
                    )
                    return h3_block.run_graph("candidate", ctx, ops)[h3_block.OUTPUT]

                block.forward = MethodType(forward, block)

            def hook(module, args, output, name=f"block.{i}"):
                trace.observe(name, output)

            handles.append(block.register_forward_hook(hook))
        for i, block in enumerate(model.token_refiner.refiner_blocks):

            def hook(module, args, output, name=f"refiner.{i}"):
                trace.observe(name, output)

            handles.append(block.register_forward_hook(hook))
        for name in boundaries:

            def hook(module, args, output, name=name):
                trace.observe(name, output)

            handles.append(getattr(model, name).register_forward_hook(hook))
        yield trace
    finally:
        for handle in handles:
            handle.remove()
        for block, original in originals:
            block.forward = original
        if boundary_context is not None:
            boundary_context.__exit__(None, None, None)
