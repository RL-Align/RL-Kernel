# SPDX-License-Identifier: Apache-2.0
"""Local frozen-Gate bridge: route gradients propagate into upstream BF16 LoRA."""

import torch

from rl_engine.backends.rocm.moe.router_training import implementation
from rl_engine.ops.moe.hash_route import HashRouteTable
from rl_engine.ops.moe.learned_route import (
    LearnedRoutePlan,
    _version,
    prepare_learned_route_plan,
    tensor_sha256,
)
from rl_engine.ops.moe.router_gate import router_gate_projection
from rl_engine.ops.moe.router_training import normal_hash_route, normal_learned_route


class DeepSeekV4RouterTraining(torch.nn.Module):
    """First-phase scope: base Gate/bias frozen; no implicit Gate fine-tuning.

    The caller owns upstream LoRA modules, loss normalization and synchronized
    model versions. This module is a local adapter, not a framework patch.
    """

    def __init__(
        self,
        gate_weight,
        *,
        absolute_layer,
        model_version,
        bias=None,
        bias_version=None,
        hash_table=None,
    ):
        super().__init__()
        _version(model_version, "model_version")
        if not isinstance(gate_weight, torch.nn.Parameter) or gate_weight.requires_grad:
            raise ValueError("first-phase router requires a frozen loaded Gate Parameter")
        if (
            gate_weight.dtype != torch.bfloat16
            or gate_weight.ndim != 2
            or gate_weight.shape[0] != 256
        ):
            raise ValueError("router Gate must be unquantized BF16 [256,K]")
        if not gate_weight.is_cuda or torch.version.hip is None:
            raise RuntimeError("router training requires ROCm")
        with torch.cuda.device(gate_weight.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("router training does not support Graph capture")
        if type(absolute_layer) is not int or not 0 <= absolute_layer < 43:
            raise ValueError("invalid Flash-0731 absolute layer")
        if absolute_layer < 3:
            if not isinstance(hash_table, HashRouteTable) or bias is not None:
                raise ValueError("Hash layers require a prepared table and no selection bias")
            if (hash_table.absolute_layer, hash_table.model_version) != (
                absolute_layer,
                model_version,
            ):
                raise ValueError("Hash layer/model version mismatch")
        else:
            _version(bias_version, "bias_version")
            if hash_table is not None or not isinstance(bias, torch.Tensor):
                raise ValueError("learned layers require selection bias and no Hash table")
            if (
                bias.dtype != torch.float32
                or bias.shape != (256,)
                or bias.device != gate_weight.device
            ):
                raise ValueError("selection bias must be FP32 [256] on the Gate device")
            if bias.requires_grad:
                raise ValueError("selection-only bias must be frozen in first-phase training")
        if not torch.isfinite(gate_weight).all().item() or (
            bias is not None and not torch.isfinite(bias).all().item()
        ):
            raise ValueError("finite frozen router parameters required")
        self.gate_weight = gate_weight
        self.register_buffer("selection_bias", None if bias is None else bias.detach().clone())
        self.absolute_layer = absolute_layer
        self.model_version = model_version
        self.bias_version = bias_version
        self.hash_table = hash_table
        self._gate_sha256 = tensor_sha256(gate_weight)
        self._bias_sha256 = None if bias is None else tensor_sha256(self.selection_bias)

    def _logits(self, hidden):
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("router hidden states must be BF16 tensors")
        if hidden.is_cuda:
            with torch.cuda.device(hidden.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("router training does not support Graph capture")
        implementation()
        if self.gate_weight.requires_grad or tensor_sha256(self.gate_weight) != self._gate_sha256:
            raise ValueError("frozen Gate changed; rebind a synchronized model version")
        if (
            self.selection_bias is not None
            and tensor_sha256(self.selection_bias) != self._bias_sha256
        ):
            raise ValueError("frozen selection bias changed; rebind its version")
        return router_gate_projection(hidden, self.gate_weight, model_version=self.model_version)

    def prepare_plan(self, hidden, logical_ids, domain, *, plan_version, routed_scaling_factor=1.5):
        if self.selection_bias is None:
            raise ValueError("Hash routes use their immutable table, not learned capacity plans")
        if (domain.absolute_layer, domain.model_version) != (
            self.absolute_layer,
            self.model_version,
        ):
            raise ValueError("capacity domain layer/model barrier mismatch")
        with torch.no_grad():
            gate = self._logits(hidden)
            return prepare_learned_route_plan(
                gate.logits,
                logical_ids,
                self.selection_bias,
                domain,
                bias_version=self.bias_version,
                plan_version=plan_version,
                routed_scaling_factor=routed_scaling_factor,
            )

    def forward(
        self,
        hidden,
        logical_ids,
        *,
        record_version,
        plan=None,
        vocabulary_ids=None,
        renormalize=True,
        routed_scaling_factor=1.5,
        rank=0,
        world_size=1,
    ):
        gate = self._logits(hidden)
        common = dict(
            record_version=record_version,
            rank=rank,
            world_size=world_size,
            protected=(self.gate_weight, hidden),
            projection_execution=gate.execution,
            absolute_layer=self.absolute_layer,
            model_version=self.model_version,
        )
        if self.selection_bias is not None:
            if not isinstance(plan, LearnedRoutePlan):
                raise ValueError("learned training requires a versioned complete-domain plan")
            if vocabulary_ids is not None or not renormalize or routed_scaling_factor != 1.5:
                raise ValueError("learned-route scaling/capacity belong to the versioned plan")
            result = normal_learned_route(
                gate.logits,
                logical_ids,
                self.selection_bias,
                plan,
                bias_version=self.bias_version,
                domain_version=plan.domain.domain_version,
                plan_version=plan.plan_version,
                **common,
            )
        else:
            if plan is not None:
                raise ValueError("Hash training cannot silently apply learned-route capacity")
            result = normal_hash_route(
                gate.logits,
                logical_ids,
                vocabulary_ids,
                self.hash_table,
                table_version=self.hash_table.table_version,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                **common,
            )
        return result
