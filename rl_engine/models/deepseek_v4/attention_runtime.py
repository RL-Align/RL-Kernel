# SPDX-License-Identifier: Apache-2.0
"""Request-level mode admission around the shared DSV4 joint attention plan."""

from collections import defaultdict
from dataclasses import dataclass
from threading import RLock

import torch

from rl_engine.contracts.operators.joint_attention import require
from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata


def _same_bytes(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@dataclass(frozen=True)
class AttentionResult:
    output: torch.Tensor
    lse: torch.Tensor
    mode: str
    version: str
    implementation: str


class AttentionRuntime:
    """Conservative request gate; numerical work remains in the existing plan.

    KV rows and metadata come from the caller's versioned cache/Indexer adapters.
    This class does not implement a fallback cache, Top-K, or quantization.
    """

    def __init__(self, plan: JointAttentionPlan, *, version: str, autograd=None):
        require(isinstance(plan, JointAttentionPlan), "joint attention plan required")
        require(isinstance(version, str) and bool(version), "model/cache version required")
        require(autograd is None or autograd.plan is plan, "autograd must share the forward plan")
        self.plan, self.autograd, self.version = plan, autograd, version
        self._lock = RLock()
        self._positions = {}
        self._training = {}

    def capture(self):
        with self._lock:
            self.plan.capture()

    def release_request(self, request_id: int):
        with self._lock:
            require(type(request_id) is int and request_id >= 0, "invalid request ID")
            require(request_id in self._positions, "unknown request")
            del self._positions[request_id]

    def replace_version(self, version: str):
        with self._lock:
            require(
                not self._positions and not self._training, "release requests and training records"
            )
            require(
                isinstance(version, str) and bool(version) and version != self.version,
                "new model/cache version required",
            )
            self.version = version

    def clear_training(self):
        with self._lock:
            self._training.clear()

    def _admit_generation(self, mode, keys):
        grouped = defaultdict(list)
        for request, position in keys:
            grouped[request].append(position)
        updates = {}
        for request, positions in grouped.items():
            require(len(positions) == len(set(positions)), "duplicate logical token")
            require(mode != "decode" or len(positions) == 1, "decode is one token per request")
            ordered = sorted(positions)
            previous = self._positions.get(request, -1)
            require(
                ordered == list(range(previous + 1, previous + 1 + len(ordered))),
                "missing, repeated, or out-of-order request position",
            )
            require(mode != "decode" or previous >= 0, "decode requires a prefill")
            updates[request] = ordered[-1]
        return updates

    def forward(self, q, kv, sink, *, metadata: JointMetadata, mode: str, use_graph=False):
        require(mode in ("train", "recompute", "prefill", "decode"), "unknown runtime mode")
        require(isinstance(metadata, JointMetadata), "bound logical metadata required")
        with self._lock:
            require(metadata.version == self.version, "stale model/cache version")
            metadata.validate(
                self.plan.contract, q.shape[0], kv.shape[0], self.plan.device, self.version
            )
            rows = [
                (i, tuple(key))
                for i, key in enumerate(metadata.query_keys.cpu().tolist())
                if key[0] >= 0
            ]
            keys = [key for _, key in rows]
            updates = self._admit_generation(mode, keys) if mode in ("prefill", "decode") else {}
            if mode == "train":
                require(not any(key in self._training for key in keys), "duplicate training token")
            if mode == "recompute":
                require(all(key in self._training for key in keys), "missing training forward")
                canonical_kv = kv.index_select(0, metadata.permutation)
                for row, key in rows:
                    prior = self._training[key]
                    require(
                        torch.equal(prior[2], metadata.kv_keys)
                        and torch.equal(prior[3], metadata.indices[row]),
                        f"recompute KV identity/selection mismatch at logical token {key}",
                    )
                    require(
                        _same_bytes(prior[4], q[row])
                        and _same_bytes(prior[5], canonical_kv)
                        and _same_bytes(prior[6], sink),
                        f"recompute Q/KV/sink input mismatch at logical token {key}",
                    )
            needs_grad = torch.is_grad_enabled() and any(
                tensor.requires_grad for tensor in (q, kv, sink)
            )
            require(
                not needs_grad or mode in ("train", "recompute"),
                "generation does not record gradients",
            )
            require(
                not needs_grad or self.autograd is not None,
                "training gradients require the native joint autograd adapter",
            )
            if needs_grad:
                result = self.autograd(
                    q,
                    kv,
                    sink,
                    metadata=metadata,
                    version=self.version,
                    mode=mode,
                    use_graph=use_graph,
                )
                output, lse = result.output, result.lse
            else:
                saved = self.plan.forward_record(
                    q,
                    kv,
                    sink,
                    metadata=metadata,
                    version=self.version,
                    mode=mode,
                    use_graph=use_graph,
                )
                output, lse = saved.output, saved.lse
            if mode == "train":
                canonical_kv = kv.index_select(0, metadata.permutation).detach().clone()
                saved_sink = sink.detach().clone()
                saved_keys = metadata.kv_keys.clone()
                for row, key in rows:
                    self._training[key] = (
                        output[row].detach().clone(),
                        lse[row].detach().clone(),
                        saved_keys,
                        metadata.indices[row].clone(),
                        q[row].detach().clone(),
                        canonical_kv,
                        saved_sink,
                    )
            elif mode == "recompute":
                for row, key in rows:
                    expected_o, expected_lse = self._training[key][:2]
                    require(
                        _same_bytes(expected_o, output[row].detach())
                        and _same_bytes(expected_lse, lse[row].detach()),
                        f"train/recompute byte mismatch at logical token {key}",
                    )
                for key in keys:
                    del self._training[key]
            else:
                self._positions.update(updates)
            return AttentionResult(output, lse, mode, self.version, self.plan.implementation)
