# SPDX-License-Identifier: Apache-2.0
"""Indexer score from raw ICV: scale, ReLU, head weights and head sum (items 10/14)."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.indexer_score import IndexerScoreContract, require


def stamp(t):
    return (t.data_ptr(), t._version, tuple(t.shape), tuple(t.stride()))


@dataclass(frozen=True)
class IndexerScoreRecord:
    """Forward result plus the intermediates the VJP and item-14 comparisons need.

    Under TP>1 ``scores`` is this rank's partial sum over its local heads.
    """

    scores: torch.Tensor
    relu: torch.Tensor = field(repr=False)
    scaled_weights: torch.Tensor = field(repr=False)
    implementation_fingerprint: str
    _stamps: tuple = field(repr=False)

    def validate(self):
        require(
            (stamp(self.relu), stamp(self.scaled_weights)) == self._stamps,
            "saved Indexer score inputs changed after forward",
        )


class IndexerScore:
    def __init__(self, contract, *, backend=None, expected_implementation=None):
        require(isinstance(contract, IndexerScoreContract), "typed Indexer score contract required")
        if backend is None:
            from rl_engine.backends.rocm.attention.indexer_score import AiterIndexerScoreBackend

            backend = AiterIndexerScoreBackend()
        require(backend.backend_id == contract.backend_id, "backend identity mismatch")
        self.contract, self.backend = contract, backend
        identity = {
            "contract": contract.fingerprint,
            "backend": getattr(backend, "provenance", {"test_backend": type(backend).__name__}),
            "model_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "plan": backend.plan(contract),
        }
        self.implementation_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        if expected_implementation is not None:
            require(
                expected_implementation == self.implementation_fingerprint,
                "train/rollout implementation fingerprint mismatch",
            )

    def _check(self, *tensors):
        for t in tensors:
            require(
                isinstance(t, torch.Tensor) and t.dtype == torch.float32, "FP32 tensors required"
            )
            require(t.device.type == self.backend.device_type, "tensor/backend device mismatch")
            if hasattr(self.backend, "device"):
                require(t.device == self.backend.device, "bind a backend on this device")
            if t.device.type == "cuda":
                with torch.cuda.device(t.device):
                    require(
                        not torch.cuda.is_current_stream_capturing(),
                        "Graph capture is not validated",
                    )
            require(bool(torch.isfinite(t).all()), "non-finite Indexer score input")

    @torch.no_grad()
    def forward(self, icv, weights):
        """icv: raw FP32 [M, local_heads, N] from item 16; weights: FP32 [M, local_heads]."""
        c = self.contract
        self._check(icv, weights)
        require(
            icv.ndim == 3 and icv.shape[1] == c.local_heads,
            "ICV must be [queries, local_heads, keys]",
        )
        require(
            tuple(weights.shape) == tuple(icv.shape[:2]), "weights must be [queries, local_heads]"
        )
        relu = torch.relu(icv.detach() * c.softmax_scale)
        weighted = weights.detach() * c.head_scale
        scores = self.backend.row_dot(
            relu.transpose(1, 2).contiguous(),
            weighted[:, None, :].expand(-1, icv.shape[2], -1),
        )
        return IndexerScoreRecord(
            scores,
            relu,
            weighted,
            self.implementation_fingerprint,
            (stamp(relu), stamp(weighted)),
        )

    @torch.no_grad()
    def backward(self, record, grad_scores):
        """VJP w.r.t. raw ICV and unscaled weights; ReLU has derivative 0 at 0."""
        c = self.contract
        require(
            record.implementation_fingerprint == self.implementation_fingerprint,
            "record produced by a different implementation",
        )
        record.validate()
        self._check(grad_scores)
        require(
            tuple(grad_scores.shape) == tuple(record.scores.shape),
            "score gradient must be [queries, keys]",
        )
        require(
            grad_scores.shape[1] <= c.max_backward_keys, "backward key rows exceed the contract"
        )
        grad = grad_scores.detach()[:, None, :]
        d_weights = self.backend.row_dot(record.relu, grad.expand_as(record.relu)) * c.head_scale
        d_icv = grad * record.scaled_weights[:, :, None]
        d_icv = torch.where(record.relu > 0, d_icv * c.softmax_scale, 0.0)
        return d_icv, d_weights
