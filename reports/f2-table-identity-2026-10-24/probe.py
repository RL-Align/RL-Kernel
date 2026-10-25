"""Record canonical A100 table identity and initialization/validation costs."""
from dataclasses import asdict
import json
import time

import torch
import triton

from rl_engine.ops.rope.rope_table import build_shared_rope_table, table_identity

torch.empty(1, device="cuda")
torch.cuda.synchronize()
start = time.perf_counter()
c, s = build_shared_rope_table(131072, device="cuda")
build_seconds = time.perf_counter() - start
identity = table_identity(c, s)
start = time.perf_counter()
for _ in range(10000):
    assert table_identity(c, s) is identity
validation_us = (time.perf_counter() - start) * 1e6 / 10000
print(json.dumps(dict(identity=asdict(identity), gpu=torch.cuda.get_device_name(),
                     capability=torch.cuda.get_device_capability(), torch=torch.__version__,
                     cuda=torch.version.cuda, triton=triton.__version__,
                     build_seconds=build_seconds, validation_us=validation_us), indent=2))
