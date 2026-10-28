# SPDX-License-Identifier: Apache-2.0
"""Compile only the production RoPE CUDA source for focused GPU validation.

Loads the production TORCH_LIBRARY registration directly. Normal package builds
include the same source in rl_engine._C. No generated binding or monkeypatch is
used. This validates the operator library, not every unrelated native operator.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_focused_extension():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    from rl_engine.backends.cuda.rope.multi_axis_rope import native_available

    if native_available():
        print("Using the installed production RoPE dispatcher library")
        return

    build = ROOT / "artifacts" / "multi_axis_rope_native"
    build.mkdir(parents=True, exist_ok=True)
    # Conda CUDA toolkits use lib/, whereas the Windows SDK convention is lib/x64/.
    link_flags = []
    if os.name == "nt" and CUDA_HOME and (Path(CUDA_HOME) / "lib/cudart.lib").is_file():
        link_flags.append(f'/LIBPATH:"{Path(CUDA_HOME) / "lib"}"')
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    load(
        name="rlk_multi_axis_rope_native",
        sources=[str(ROOT / "csrc/cuda/rope/multi_axis_rope.cu")],
        build_directory=str(build),
        extra_cflags=(["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]),
        extra_cuda_cflags=["-O3", "-std=c++17", "--fmad=false"],
        extra_ldflags=link_flags,
        is_python_module=False,
        verbose=True,
    )
    assert hasattr(torch.ops.rl_kernel_native, "multi_axis_rope")
    print(
        f"Focused CUDA source build; torch={torch.__version__}, GPU={torch.cuda.get_device_name()}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--ws1", choices=("cuda", "triton"))
    parser.add_argument("--output", type=Path)
    args, remaining = parser.parse_known_args()
    load_focused_extension()
    if args.ws1:
        from rl_engine.validation.operators.qwen_image_rope import run_multi_axis_rope_ws1

        result = run_multi_axis_rope_ws1(args.ws1)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"WS1 four judgments backend={args.ws1}: passed={result['passed']}")
        raise SystemExit(0 if result["passed"] else 1)
    elif args.benchmark:
        sys.argv = ["benchmark_multi_axis_rope", *remaining]
        if args.output:
            sys.argv.extend(["--output", str(args.output)])
        runpy.run_module("benchmarks.operators.rope.benchmark_multi_axis_rope", run_name="__main__")
    else:
        import pytest

        raise SystemExit(
            pytest.main([str(ROOT / "tests/ops/rope/test_multi_axis_rope.py"), *remaining])
        )


if __name__ == "__main__":
    main()
