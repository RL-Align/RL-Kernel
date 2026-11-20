"""Fresh-process real native auxiliary-loss/gradient digest."""

import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_engine.models.deepseek_v4.indexer_training import canonical_indexer_auxiliary_loss
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.test_indexer_auxiliary_rocm import make_case


def digest(tensor):
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    op, q, k, w, tq, tk, meta = make_case(sys.argv[1])
    for value in (q.master, k.master, w):
        value.requires_grad_()
    with canonical_backward_session() as session:
        result = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        loss = canonical_indexer_auxiliary_loss([result])
        loss.backward()
        session.validate_complete()
    Path(sys.argv[2]).write_text(
        json.dumps(
            {
                "loss": digest(loss),
                "q_grad": digest(q.master.grad),
                "k_grad": digest(k.master.grad),
                "weights_grad": digest(w.grad),
                "q_bytes": digest(q.quantized.data),
                "k_bytes": digest(k.quantized.data),
                "implementation": op.implementation,
            },
            sort_keys=True,
        )
    )
