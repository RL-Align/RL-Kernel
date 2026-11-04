# SPDX-License-Identifier: Apache-2.0
"""Run from outside the checkout after installing a RoPE-enabled wheel.

Does not change sys.path or compile/load a JIT extension. --package-root makes
the expected installation location an explicit assertion in package validation.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

import rl_engine
from rl_engine.backends import extension
from rl_engine.ops.rope.multi_axis_rope import (
    build_multi_axis_cos_sin,
    get_multi_axis_rope_op,
    qwen_image_positions,
)

_PROVIDER_PROBE = """
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass

full, companion, mode = sys.argv[1:]
paths = [full, companion] if mode != 'companion-first' else [companion, full]
if mode == 'concurrent':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(torch.ops.load_library, paths))
elif mode in ('full-import-first', 'companion-import-first'):
    import rl_engine
    companion_name = Path(companion).name.split('.')[0]
    modules = [('_C', full), (companion_name, companion)]
    if mode == 'companion-import-first':
        modules.reverse()
    for name, path in modules:
        spec = importlib.util.spec_from_file_location('rl_engine.' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
else:
    for path in paths:
        torch.ops.load_library(path)

from rl_engine.ops.rope.multi_axis_rope import get_multi_axis_rope_op

torch.set_flush_denormal(False)
reference = get_multi_axis_rope_op('reference')
checks = 0
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    tiny = torch.finfo(dtype).tiny
    x = torch.tensor([tiny, -tiny, tiny/2, -tiny/2, 1., -1., 0., -0.], dtype=dtype)
    x = x.repeat(16).reshape(1, 1, 1, 128)
    for c, s in ((.5, .25), (1., 0.), (1., -0.)):
        cos, sin = torch.full((1, 64), c), torch.full((1, 64), s)
        for transpose in (False, True):
            expected = reference(x, cos, -sin if transpose else sin)
            actual = torch.ops.rl_kernel_native.multi_axis_rope(
                x.cuda(), cos.cuda(), sin.cuda(), transpose
            ).cpu()
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            checks += 1
print(json.dumps({'mode': mode, 'checks': checks, 'passed': True}))
"""


def check_native_provider_coexistence(full: Path, companion: Path) -> list[dict]:
    """Fresh processes isolate dispatcher lifetime and both native load orders."""
    full, companion = full.resolve(strict=True), companion.resolve(strict=True)
    results = []
    for mode in (
        "full-first",
        "companion-first",
        "concurrent",
        "full-import-first",
        "companion-import-first",
    ):
        run = subprocess.run(
            [sys.executable, "-c", _PROVIDER_PROBE, str(full), str(companion), mode],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if run.returncode:
            raise RuntimeError(f"native provider check {mode} failed: {run.stderr[-4000:]}")
        results.append(json.loads(run.stdout.splitlines()[-1]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--companion-library",
        type=Path,
        help="Also test a same-revision companion against the installed full extension",
    )
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
    gold_dx = get_multi_axis_rope_op("reference")(grad, cos, -sin)
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
    coexistence = []
    provider_libraries = {}
    if args.companion_library is not None:
        full = getattr(extension._C, "__file__", None)
        if full is None:
            raise RuntimeError("provider coexistence requires an installed full rl_engine._C")
        coexistence = check_native_provider_coexistence(Path(full), args.companion_library)
        provider_libraries = {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in (("full", Path(full)), ("companion", args.companion_library))
        }
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
        "provider_coexistence_libraries": provider_libraries,
        "provider_coexistence": coexistence,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
