# SPDX-License-Identifier: Apache-2.0
"""Run from outside the checkout after installing a RoPE-enabled wheel.

Does not change sys.path or compile/load a JIT extension. --package-root makes
the expected installation location an explicit assertion in package validation.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

import rl_engine
from rl_engine.backends import extension
from rl_engine.ops.rope.multi_axis_rope import (
    build_multi_axis_cos_sin,
    get_multi_axis_rope_op,
    qwen_image_positions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = Path(rl_engine.__file__).resolve().parent
    assert package.is_relative_to(args.package_root.resolve()), package
    license_path = package / "backends/shared/triton/rope/FLASH_ATTENTION_LICENSE"
    assert "Redistribution and use" in license_path.read_text(encoding="utf-8")
    positions, _ = qwen_image_positions(3, 5, 0)
    cos, sin = build_multi_axis_cos_sin(positions, device="cuda")
    packed = torch.randn(2, 15, 3, 3, 128, device="cuda", dtype=torch.bfloat16)
    x = packed[:, :, 0].detach().requires_grad_()
    grad = torch.randn_like(x)
    gold = get_multi_axis_rope_op("reference")(x, cos, sin)
    gold_dx = torch.autograd.grad(gold, x, grad)[0]
    outcomes = {}
    for backend in ("cuda", "triton"):
        op = get_multi_axis_rope_op(backend, device=x.device)
        for mode, fn in (("eager", op), ("compiled", torch.compile(op, fullgraph=True))):
            result = fn(x, cos, sin)
            dx = torch.autograd.grad(result, x, grad)[0]
            assert torch.equal(result.view(torch.uint8), gold.contiguous().view(torch.uint8))
            assert torch.equal(dx.view(torch.uint8), gold_dx.contiguous().view(torch.uint8))
            outcomes[f"{backend}_{mode}"] = "passed"
        outcomes[f"{backend}_trace"] = op.trace()
    native = importlib.util.find_spec("rl_engine._rope_cuda")
    report = {
        "package": str(package),
        "companion_library": native.origin if native else None,
        "full_extension_library": getattr(extension._C, "__file__", None),
        "native_dispatcher_registered": hasattr(torch.ops.rl_kernel_native, "multi_axis_rope"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "license_present": True,
        "packed_qkv_stride": list(x.stride()),
        "checks": outcomes,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
