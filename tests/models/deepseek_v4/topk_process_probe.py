# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import sys

import torch

from rl_engine.models.deepseek_v4.indexer_topk import IndexerTopK

g = torch.Generator(device="cuda").manual_seed(17)
s = torch.round(torch.randn(32, 6000, generator=g, device="cuda") * 3) / 3
out = IndexerTopK()(s)
Path(sys.argv[1]).write_text(
    json.dumps({"sha256": hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()})
)
