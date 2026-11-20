# SPDX-License-Identifier: Apache-2.0
"""Audit baked sources/import paths and execute a real gfx942 calculation."""

import importlib
import importlib.metadata as metadata
import json

import torch

from examples.operators.dsv4_4l.audit_dependencies import audit


def main():
    assert torch.version.hip and torch.cuda.is_available()
    device = torch.cuda.get_device_properties(0)
    assert device.gcnArchName.split(":")[0] == "gfx942"
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    result = torch.dot(x, x).item()
    assert result == 10416.0
    report = dict(
        audit=audit(),
        imports={
            name: importlib.import_module(name).__file__
            for name in ("rl_engine", "verl", "megatron.core", "vllm")
        },
        versions={
            name: metadata.version(name)
            for name in ("torch", "triton", "amd-aiter", "megatron-bridge")
        },
        hip=torch.version.hip,
        architecture=device.gcnArchName,
        logical_gpu=0,
        real_gpu_dot=result,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
