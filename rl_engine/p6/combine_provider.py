# SPDX-License-Identifier: Apache-2.0
"""Frozen-case T05 candidate for T01's registry; not a production adapter."""

import platform
from dataclasses import asdict
from pathlib import Path

import torch

from .combine import prepare_combine
from .contract import PROFILE, TRACE_SCHEMA, digest, exact_keys, manifest, require
from .provider import input_plan, validate_envelope
from .recordings import boundary_recordings


class TritonCombineProvider:
    """Actual GPU conformance readback, restricted to nine input-bound fixtures.

    This intentionally runs debug/readback and checks frozen bytes. Production
    callers use prepare_combine; no oracle or host comparison belongs in timing.
    'live' denotes actual execution in the registry, not Foundation certification.
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self._records = {
            (r["operator"], r["case_id"]): r
            for r in boundary_recordings()
            if r["operator"] == "fused_moe_combine_fwd"
        }

    def describe(self):
        return {
            "contract": manifest(),
            "profile": PROFILE,
            "kind": "live",
            "capabilities": [list(k) for k in sorted(self._records)],
            "production_certified": False,
        }

    def run(self, operator, case_id, inputs):
        key = operator, case_id
        require(key in self._records, "UNSUPPORTED_CAPABILITY", "T05 frozen forward case only")
        record = self._records[key]
        exact_keys(inputs, record["inputs"], "T05 inputs")
        plan = input_plan(inputs)
        require(digest(inputs) == digest(record["inputs"]), "IDENTITY_DRIFT", "frozen input")
        prepared = prepare_combine(plan, plan.context, self.device)
        n, h, p = len(plan.token_ids), plan.hidden_size, len(plan.inverse_map)
        tensors = [
            torch.tensor(inputs[name], dtype=torch.bfloat16, device=prepared.device).reshape(shape)
            for name, shape in (("rows", (p, h)), ("shared", (n, h)), ("residual", (n, h)))
        ]
        result = prepared.forward(*tensors, debug=True)
        stages = result.stage_bytes()
        require(stages == record["expected"], "BYTE_MISMATCH", "T05 frozen stage bytes")
        import triton

        root = Path(__file__).resolve().parents[2]
        files = (
            "rl_engine/p6/combine.py",
            "rl_engine/p6/combine_provider.py",
            "rl_engine/kernels/ops/triton/moe/combine.py",
        )
        envelope = {
            "schema": TRACE_SCHEMA,
            "profile": PROFILE,
            "operator": operator,
            "case_id": case_id,
            "input_sha256": digest(inputs),
            "plan_fingerprint": plan.fingerprint,
            "order_hash": plan.order_hash,
            "kind": "live",
            "fallback": False,
            "provenance": {
                "readback_kind": "actual",
                "backend": "triton-cuda",
                "device": str(prepared.device),
                "gpu": torch.cuda.get_device_name(prepared.device),
                "compute_capability": list(torch.cuda.get_device_capability(prepared.device)),
                "implementation_sha256": digest({f: (root / f).read_text() for f in files}),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "scope": "synthetic-local-WS1; not Foundation/WS2 certification",
            },
            "stages": stages,
            "boundary_hashes": {k: digest(v) for k, v in stages.items()},
            "producer_verdict": "REFERENCE_BYTES_PASS",
            "context": asdict(plan.context),
            "boundary": record["boundary"],
            "phase": record["phase"],
        }
        validate_envelope(envelope)
        return envelope
