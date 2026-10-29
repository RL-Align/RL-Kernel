# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_engine.contracts.operators.indexer_compressor_c4 import IndexerC4Contract
from rl_engine.models.deepseek_v4.indexer_compressor_c4 import IndexerC4Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.indexer_c4_helpers import fixture


def digest(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


records = {}
for precision in ("mxfp4", "fp8"):
    op = IndexerC4Compressor(IndexerC4Contract(precision=precision))
    x, ids, a, n, co, si = fixture(8, device="cuda", requires_grad=True)
    with canonical_backward_session() as session:
        out = op(op.bind(x, ids, version="v1"), a, n, co, si, version="v1", parameter_id="probe")
        out.values.sum().backward()
        session.validate_complete()
    records[precision] = {
        "implementation": op.implementation_fingerprint,
        "values": digest(out.values),
        "packed": digest(out.packed),
        "scales": digest(out.scales),
        "dx": digest(x.grad),
        "da": digest(a.grad),
        "dn": digest(n.grad),
    }
Path(sys.argv[1]).write_text(json.dumps(records, indent=2))
