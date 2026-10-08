"""Independent original Torch semantics. Diagnostic only; never a byte oracle."""

import torch
import torch.nn.functional as F


def forward(z, round_policy, *, token_ids=None, table=None, bias=None):
    z = torch.as_tensor(z, dtype=torch.float32)
    zp = z if round_policy == "fp32_direct" else z.to(torch.bfloat16).float()
    s = F.softplus(zp, beta=1, threshold=20).sqrt()
    if table is not None:
        ids = torch.as_tensor(table, dtype=torch.int64)[torch.as_tensor(token_ids)]
        q = s
    else:
        q = s + torch.as_tensor(bias, dtype=torch.float32)
        ids = torch.argsort(q, dim=-1, descending=True, stable=True)[:, :6]
    a = s.gather(1, ids)
    Z = ((a[:, 0] + a[:, 1]) + (a[:, 2] + a[:, 3])) + (a[:, 4] + a[:, 5]) + 1e-20
    return {"z_prime": zp, "s": s, "q": q, "ids": ids, "weights": a / Z[:, None] * 1.5}
