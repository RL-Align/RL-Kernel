# SPDX-License-Identifier: Apache-2.0
"""Export actual saved FP32 token logprobs and storage bits, without a GPU allocation."""

import argparse
import json
import os
from pathlib import Path

from .gpu_visibility import SECONDARY_FILTERS, physical_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    args = parser.parse_args()
    physical_ids(os.environ["ROCR_VISIBLE_DEVICES"])
    if any(key in os.environ for key in SECONDARY_FILTERS):
        raise RuntimeError("unset secondary GPU visibility filters before running")
    import torch

    files = list(args.runs.rglob("outputs.pt")) + list(args.runs.rglob("logprobs-rank*.pt"))
    if not files:
        raise ValueError("no measured model outputs to export")
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=True)
        raw = data["raw_fp32_token_logprobs"]
        if raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()):
            raise ValueError("model raw logprobs must be finite FP32")
        name = (
            "summary.json"
            if path.name == "outputs.pt"
            else "summary-" + path.stem.split("-", 1)[1] + ".json"
        )
        summary = json.loads(path.with_name(name).read_text())
        assignment = summary["gpu_assignment"]
        record = {
            "source_file": path.name,
            "dtype": str(raw.dtype),
            "shape": list(raw.shape),
            "physical_gpu_ids": assignment["physical_gpus"],
            "logical_gpu_ids": assignment["logical_gpus"],
            "assigned_physical_gpu": assignment["assigned_physical_gpu"],
            "assigned_logical_gpu": assignment["assigned_logical_gpu"],
            "values": raw.tolist(),
            "uint32_hex": [
                [f"0x{int(v) & 0xFFFFFFFF:08x}" for v in row]
                for row in raw.view(torch.int32).tolist()
            ],
        }
        path.with_name(path.stem + "-raw-logprobs.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
    print("MEASURED_FP32_LOGPROBS_EXPORTED", len(files))


if __name__ == "__main__":
    main()
