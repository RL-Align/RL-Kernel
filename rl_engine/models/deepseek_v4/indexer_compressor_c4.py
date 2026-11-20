# SPDX-License-Identifier: Apache-2.0
"""Small Indexer-specific layer over the shared C4 admission/autograd machinery."""

from dataclasses import dataclass
import hashlib
import json

import torch

from rl_engine.contracts.operators.indexer_compressor_c4 import IndexerC4Contract
from rl_engine.contracts.operators.kv_compressor_c4 import require, source_hash
from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor


@dataclass(frozen=True)
class IndexerC4Output:
    values: torch.Tensor
    packed: torch.Tensor
    scales: torch.Tensor
    group_ids: torch.Tensor
    owner_rank: int
    version: str
    contract_fingerprint: str
    implementation_fingerprint: str
    layout: str
    precision: str


class IndexerC4Compressor(C4Compressor):
    contract_type = IndexerC4Contract

    @staticmethod
    def _make_backend():
        from rl_engine.backends.rocm.attention.indexer_compressor_c4 import AiterIndexerC4Backend

        return AiterIndexerC4Backend()

    def __init__(self, contract, *, backend=None, expected_implementation=None):
        super().__init__(contract, backend=backend)
        self.implementation_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "shared": self.implementation_fingerprint,
                    "indexer_model": source_hash(__file__),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if expected_implementation is not None:
            require(
                expected_implementation == self.implementation_fingerprint,
                "Indexer C4 implementation fingerprint mismatch",
            )

    def __call__(self, bound, ape, norm, cos, sin, *, version, parameter_id=""):
        out = super().__call__(
            bound, ape, norm, cos, sin, version=version, parameter_id=parameter_id
        )
        # Both training and rollout expose identical quantized bytes/scales.
        # Quantization is explicitly detached, NOT a hidden straight-through VJP.
        with torch.no_grad():
            packed, scales = self.backend.quantize(out.values.detach(), self.contract)
        fp4 = self.contract.precision == "mxfp4"
        rows = len(bound.logical_keys)
        require(
            packed.shape == (rows, 64 if fp4 else 128)
            and scales.shape == (rows, 4 if fp4 else 1)
            and packed.dtype == (torch.uint8 if fp4 else torch.float8_e4m3fnuz)
            and scales.dtype == (torch.uint8 if fp4 else torch.float32)
            and packed.device == out.values.device
            and scales.device == out.values.device
            and packed.is_contiguous()
            and scales.is_contiguous(),
            "Indexer C4 provider returned an invalid quantized layout",
        )
        return IndexerC4Output(
            out.values,
            packed,
            scales,
            out.group_ids,
            out.owner_rank,
            out.version,
            out.contract_fingerprint,
            self.implementation_fingerprint,
            self.contract.layout,
            self.contract.precision,
        )
