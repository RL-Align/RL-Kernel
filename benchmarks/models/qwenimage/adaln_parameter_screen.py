#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the fixed parameter reducer on cancellation cases that defeat naive sums."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.models.qwenimage.adaln_evidence import compare, source_state
from rl_engine.backends.extension import _C

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
report = dict(
    source=source_state(),
    torch=torch.__version__,
    gpu=torch.cuda.get_device_name(),
    construction="grad[B,2]=1; grad[0]=2**26; grad[-1]=-2**26; act[B,7]=1; act[:,1::2]=1.5",
    cases=[],
)
for dtype in [torch.float32, torch.bfloat16]:
    for batch in [255, 256, 257, 2048]:
        g = torch.ones(batch, 2, device="cuda", dtype=torch.float32)
        g[0] = 2**26
        g[-1] = -(2**26)
        x = torch.ones(batch, 7, device="cuda", dtype=dtype)
        x[:, 1::2] = 1.5
        golden = g.double().cpu().T @ x.double().cpu()
        baseline = _C.qwenimage_det_linear_backward_weight(g, x, dtype, True)[0]
        report["cases"].append(
            dict(
                dtype=str(dtype),
                batch=batch,
                fp64=golden.tolist(),
                fixed_actual=baseline.float().cpu().tolist(),
                fixed_contract=compare(baseline, golden, dtype, "gradient_accuracy"),
            )
        )
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print("fixed failures", sum(not c["fixed_contract"]["passed"] for c in report["cases"]))
