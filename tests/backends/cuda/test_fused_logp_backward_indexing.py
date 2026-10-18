# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise actual CUDA loop arithmetic near INT_MAX without allocating a huge tensor."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


def _loop_arithmetic_harness(source: str) -> str:
    kernel = source.split("__global__ void fused_logp_backward_kernel(", 1)[1].split(
        "template <typename scalar_t, typename output_t, int BlockSize>", 1
    )[0]
    # Reuse the real control-variable types, increments, and unrolled offsets.
    # Only the initialization changes: start at the last two tiles, so these
    # INT_MAX boundary cases take milliseconds and allocate no logits storage.
    declarations = re.findall(r"const (?:int|int64_t) (?:stride|tile) = [^;]+;", kernel)
    loops = re.findall(
        r"for \(((?:int|int64_t) (i|base) = )threadIdx.x; "
        r"\2 < vocab_size; \2 \+= (stride|tile)\)",
        kernel,
    )
    indices = re.findall(r"(?:int|int64_t) i = base \+ u \* stride;", kernel)
    assert len(declarations) == 2 and len(loops) == 4 and len(indices) == 4
    unroll = int(re.search(r"kFusedLogpBackwardUnroll = (\d+);", source)[1])

    functions = []
    checks = []
    for number, (initializer, variable, step) in enumerate(loops):
        if variable == "base":
            offsets = "\n".join(
                "{ " + index + " if (i < base || i != int64_t(base) + u * int64_t(stride)) "
                "return 2; }"
                for index in sorted(set(indices))
            )
            body = f"for (int u = 0; u < kUnroll; ++u) {{ {offsets} }}"
        else:
            body = ""
        functions.append(
            f"""
int check_{number}(int vocab_size, int thread) {{
    constexpr int kUnroll = {unroll};
    struct {{ int x; }} blockDim{{256}};
    {' '.join(declarations)}
    const int64_t last = thread + ((int64_t(vocab_size) - 1 - thread) / {step}) * {step};
    const int64_t start = last - {step};
    int iterations = 0;
    for ({initializer}start; {variable} < vocab_size; {variable} += {step}) {{
        if ({variable} < 0 || ++iterations > 2) return 1;
        {body}
    }}
    return iterations == 2 ? 0 : 3;
}}
"""
        )
        checks.append(f"if (check_{number}(vocab_size, thread)) return {number + 1};")
    return "\n".join(
        [
            "#include <cstdint>",
            "#include <climits>",
            "#include <initializer_list>",
            *functions,
            f"""
int main() {{
    for (int delta : {{0, 1, 255, 256, 1023, 1024}}) {{
        const int vocab_size = INT_MAX - delta;
        for (int thread : {{0, 1, 255}}) {{ {' '.join(checks)} }}
    }}
    return 0;
}}
""",
        ]
    )


@pytest.mark.unit
def test_backward_vocab_loop_arithmetic_does_not_overflow_int_max(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the CUDA indexing arithmetic regression")
    source_path = Path(__file__).resolve().parents[3] / "csrc/cuda/logprob/fused_logp_kernel.cu"
    harness = tmp_path / "backward_indexing.cpp"
    executable = tmp_path / "backward_indexing"
    harness.write_text(_loop_arithmetic_harness(source_path.read_text()), encoding="utf-8")
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            str(harness),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr or "invalid or nonterminating vocabulary walk"
