# SPDX-License-Identifier: Apache-2.0
"""Item 17: stable Top-K over Indexer scores, returned in the attention read order."""

import hashlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.indexer_topk import (
    MAX_KEYS,
    IndexerTopKContract,
    require,
)


class IndexerTopK:
    """Score descending, ties to the smaller column; -inf and columns >= length are never chosen.

    Output is column descending with -1 padding last, the compressed-row read order.
    """

    def __init__(self, contract=None, *, backend=None, expected_implementation=None):
        contract = IndexerTopKContract() if contract is None else contract
        require(isinstance(contract, IndexerTopKContract), "typed Indexer Top-K contract required")
        if backend is None:
            from rl_engine.backends.rocm.attention.indexer_topk import TritonIndexerTopKBackend

            backend = TritonIndexerTopKBackend()
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
        require(
            expected_implementation in (None, self.implementation_fingerprint),
            "train/rollout implementation fingerprint mismatch",
        )

    @torch.no_grad()
    def __call__(self, scores, lengths=None):
        """scores FP32 [M,N]; lengths int [M] visible columns. Returns int32 [M,K]."""
        require(
            isinstance(scores, torch.Tensor)
            and scores.ndim == 2
            and scores.dtype == torch.float32
            and scores.stride(1) == 1,
            "scores must be FP32 [rows, keys] with unit column stride",
        )
        require(scores.shape[1] <= MAX_KEYS, "key count exceeds int32 column IDs")
        require(scores.device.type == self.backend.device_type, "tensor/backend device mismatch")
        if hasattr(self.backend, "device"):
            require(scores.device == self.backend.device, "bind a backend on this device")
        if scores.device.type == "cuda":
            with torch.cuda.device(scores.device):
                require(
                    not torch.cuda.is_current_stream_capturing(), "Graph capture is not validated"
                )
        rows, keys = scores.shape
        if lengths is None:
            lengths = torch.full((rows,), keys, dtype=torch.int32, device=scores.device)
        require(
            isinstance(lengths, torch.Tensor)
            and lengths.shape == (rows,)
            and lengths.dtype in (torch.int32, torch.int64)
            and lengths.device == scores.device,
            "lengths must be int [rows] on the score device",
        )
        lengths = lengths.to(torch.int32).contiguous()
        require(bool(((lengths >= 0) & (lengths <= keys)).all()), "length outside [0, keys]")
        visible = torch.arange(keys, device=scores.device) < lengths[:, None]
        require(
            not bool((visible & (torch.isnan(scores) | (scores == float("inf")))).any()),
            "visible scores must be finite or -inf",
        )
        return self.backend.select(scores, lengths, self.contract.topk)
