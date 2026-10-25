"""Warm up before capture; NVTX identifies each F2 call path."""

import argparse
import torch
from rl_engine.ops.rope.qknorm_scaled_rope import qknorm_scaled_rope_forward, _dispatch_backward
from rl_engine.reference.rope.qknorm_scaled_rope import build_rope_table

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["cuda", "triton"])
p.add_argument("--phase", choices=["forward", "backward"])
p.add_argument("--tokens", type=int)
p.add_argument("--local", choices=["0", "1"])
p.add_argument("--repeats", type=int, default=10)
a = p.parse_args()
c, s = build_rope_table(131072, device="cuda")
cases = []
for tokens in ([a.tokens] if a.tokens else [1, 2048]):
    gen = torch.Generator(device="cuda").manual_seed(501)
    q = torch.randn(tokens, 32, 128, device="cuda", generator=gen).bfloat16()
    k = torch.randn(tokens, 2, 128, device="cuda", generator=gen).bfloat16()
    gq, gk = torch.ones_like(q), torch.ones_like(k)
    positions = torch.arange(tokens, dtype=torch.int64, device="cuda")
    for local in ([bool(int(a.local))] if a.local is not None else [False, True]):
        for backend in ([a.backend] if a.backend else ["cuda", "triton"]):
            saved = qknorm_scaled_rope_forward(
                q, k, positions, c, s, is_local=local, backend=backend
            )
            for phase in ([a.phase] if a.phase else ["forward", "backward"]):
                if phase == "forward":

                    def fn(q=q, k=k, positions=positions, local=local, backend=backend):
                        return qknorm_scaled_rope_forward(
                            q, k, positions, c, s, is_local=local, backend=backend
                        )

                else:

                    def fn(
                        q=q,
                        k=k,
                        gq=gq,
                        gk=gk,
                        saved=saved,
                        positions=positions,
                        local=local,
                        backend=backend,
                    ):
                        return _dispatch_backward(
                            q,
                            k,
                            gq,
                            gk,
                            *saved[2:],
                            positions,
                            c,
                            s,
                            is_local=local,
                            backend=backend,
                        )

                for _ in range(5):
                    fn()
                cases.append((f'T{tokens}/{"local" if local else "global"}/{backend}/{phase}', fn))
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for label, fn in cases:
    with torch.cuda.nvtx.range(label):
        for _ in range(a.repeats):
            fn()
        torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("Profiled", len(cases), "cases;", a.repeats, "calls per case")
