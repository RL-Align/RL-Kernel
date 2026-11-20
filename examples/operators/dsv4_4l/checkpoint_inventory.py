# SPDX-License-Identifier: Apache-2.0
"""Hash original sync-eligible tensors without loading the frozen packed base."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def inventory(checkpoint, output):
    checkpoint = Path(checkpoint)
    count = Counter()
    parameters = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        for name in f.keys():
            kind = f.get_slice(name).get_dtype()
            count[kind] += 1
            if kind not in ("BF16", "F32", "I64"):
                continue
            tensor = f.get_tensor(name)
            raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            parameters[name] = dict(
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
            if name.endswith(".tid2eid") and tensor.dtype == torch.int64:
                # Pinned native MCore and AMD vLLM consume int32 routing indices.
                # This is an exact integer ABI conversion, not a weight/score cast.
                native = tensor.to(torch.int32)
                if not torch.equal(native.to(torch.int64), tensor):
                    raise ValueError("routing indices do not fit the native int32 ABI")
                parameters[name]["runtime_abi"] = dict(
                    shape=list(native.shape),
                    dtype=str(native.dtype),
                    sha256=hashlib.sha256(
                        native.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    roundtrip_exact=True,
                )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    result = dict(
        checkpoint_sha256=digest.hexdigest(),
        checkpoint_bytes=checkpoint.stat().st_size,
        dtype_counts=dict(count),
        sync_eligible_original_tensors=parameters,
    )
    Path(output).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("output")
    a = p.parse_args()
    result = inventory(a.checkpoint, a.output)
    print(result["checkpoint_sha256"], len(result["sync_eligible_original_tensors"]))
