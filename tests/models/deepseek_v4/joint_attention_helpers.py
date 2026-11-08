"""Independent real-valued oracle, explicitly injected only by CPU tests."""

import torch

from rl_engine.contracts.operators.joint_attention import JOINT_PROFILE, JointAttentionContract
from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata


def reference(q, kv, sink, indices, scale):
    outputs, lses = [], []
    for row in range(q.shape[0]):
        values = kv[indices[row][indices[row] >= 0].long()].double()
        logits = q[row].double() @ values.t() * scale
        logits = torch.cat((logits, sink.double()[:, None]), 1)
        lse = logits.logsumexp(1)
        probs = (logits[:, :-1] - lse[:, None]).exp()
        out = probs @ values
        outputs.append(out)
        lses.append(lse)
    if not outputs:
        return q.double().clone(), torch.empty(q.shape[:2], dtype=torch.float64, device=q.device)
    return torch.stack(outputs), torch.stack(lses)


def reference_from_keys(q, kv, sink, query_keys, kv_keys, compressed, window_size, scale):
    """Select from caller-order keys, without reading bound metadata or its indices."""
    queries = query_keys.cpu().tolist()
    keys = kv_keys.cpu().tolist()
    choices = compressed.cpu().tolist()
    windows = {
        (request, position): row for row, (request, kind, position) in enumerate(keys) if kind == 1
    }
    rows = []
    for (request, position), topk in zip(queries, choices, strict=True):
        if request < 0:
            rows.append([])
            continue
        selected = sorted((row for row in topk if row >= 0), key=lambda row: tuple(keys[row]))
        recent = [
            windows[request, p] for p in range(max(0, position - window_size + 1), position + 1)
        ]
        rows.append(selected + recent)
    indices = torch.full(
        (len(rows), max(map(len, rows), default=0)), -1, dtype=torch.int64, device=q.device
    )
    for row, selected in enumerate(rows):
        indices[row, : len(selected)] = torch.tensor(selected, device=q.device)
    return reference(q, kv, sink, indices, scale)


class ReferenceBackend:
    profile = JOINT_PROFILE
    device_type = "cpu"
    device = torch.device("cpu")
    provenance = {"test_only": "independent FP64 softmax"}

    def forward_into(self, q, kv, sink, indices, out, lse, contract):
        want, want_lse = reference(q, kv, sink, indices, contract.softmax_scale)
        out.copy_(want)
        lse.copy_(want_lse)


def fixture_keys(m, ratio, topk, padding, device):
    kh = []
    for request in (0, 1):
        kh.extend((request, 1, p) for p in range(16))
        kh.extend((request, 0, p) for p in range(ratio - 1, 16, ratio))
    kvkeys = torch.tensor(kh, dtype=torch.int64, device=device)
    qh = [[i % 2, 5 + i // 2] for i in range(m)]
    if padding and m:
        qh[-1] = [-1, -1]
    qkeys = torch.tensor(qh, dtype=torch.int64, device=device).reshape(m, 2)
    chosen = []
    for req, pos in qh:
        candidates = [
            i for i, (r, kind, end) in enumerate(kh) if r == req and kind == 0 and end <= pos
        ]
        candidates = candidates[-topk:] if topk else []
        chosen.append(candidates + [-1] * (topk - len(candidates)))
    comp = torch.tensor(chosen, dtype=torch.int32, device=device).reshape(m, topk)
    return qkeys, kvkeys, comp


def case_reference(plan, q, kv, sink, *, padding=False):
    c = plan.contract
    qkeys, kvkeys, comp = fixture_keys(
        q.shape[0], c.compression_ratio, c.compressed_topk, padding, q.device
    )
    return reference_from_keys(q, kv, sink, qkeys, kvkeys, comp, c.window_size, c.softmax_scale)


def case(*, backend=None, m=4, heads=8, dim=128, window=4, topk=2, ratio=4, padding=False):
    backend = backend or ReferenceBackend()
    device = backend.device
    torch.manual_seed(24)
    c = JointAttentionContract(heads, dim, max(m, 8), 64, topk, window, compression_ratio=ratio)
    qkeys, kvkeys, comp = fixture_keys(m, ratio, topk, padding, device)
    meta = JointMetadata.bind(c, qkeys, kvkeys, comp, version="v1")
    q = torch.randn(m, heads, dim, device=device).bfloat16()
    kv = torch.randn(kvkeys.shape[0], dim, device=device).bfloat16()
    sink = torch.randn(heads, device=device)
    plan = JointAttentionPlan(c, backend=backend)
    return plan, q, kv, sink, meta, comp


def record(plan, q, kv, sink, meta, **kwargs):
    return plan.forward_record(
        q, kv, sink, metadata=meta, version="v1", mode=kwargs.pop("mode", "train"), **kwargs
    )


def same_bytes(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
