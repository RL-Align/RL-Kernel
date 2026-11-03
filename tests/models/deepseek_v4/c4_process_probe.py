# SPDX-License-Identifier: Apache-2.0
"""Fresh-process C4 output/gradient fingerprint."""

import hashlib
import json
from pathlib import Path
import sys

import torch

from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract
from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.c4_helpers import fixture


def digest(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


if __name__ == "__main__":
    values, ids, ape, norm, cos, sin = fixture(8, device="cuda", requires_grad=True)
    operator = C4Compressor(C4Contract())
    with canonical_backward_session() as session:
        output = operator(
            operator.bind(values, ids, version="v1"),
            ape,
            norm,
            cos,
            sin,
            version="v1",
            parameter_id="probe",
        )
        output.values.sum().backward()
        session.validate_complete()
    Path(sys.argv[1]).write_text(
        json.dumps(
            {
                "output": digest(output.values),
                "dvalues": digest(values.grad),
                "dape": digest(ape.grad),
                "dnorm": digest(norm.grad),
                "implementation": operator.implementation_fingerprint,
            },
            indent=2,
        )
    )
