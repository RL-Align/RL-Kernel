# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Regenerate the pinned-upstream fixture for ``test_h3_sigma_schedule``.

Executes the ORIGINAL ``MiniMaxH3Scheduler.set_timesteps`` extracted from the
upstream diffusers source file — no formula is transcribed by hand. The
method's AST node is compiled and run against a namespace shim that supplies
the scheduler attributes it touches, so the construction arithmetic (linspace,
exponential shift, ``unique_consecutive``, timestep conversion) is upstream's
own code.

The upstream file's git blob sha must equal ``UPSTREAM_BLOB_SHA`` below, so
this script cannot silently regenerate the fixture from a different revision.

Usage::

    python make_upstream_fixture.py \
        --upstream-file <scheduling_minimax_h3.py> \
        --out h3_sigma_schedule_upstream_fixture.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import types

import torch

UPSTREAM_URL = (
    "https://github.com/huggingface/diffusers/blob/f53d552/src/diffusers/schedulers/"
    "scheduling_minimax_h3.py"
)
UPSTREAM_COMMIT = "f53d552036a0d1bd5570782a39cd40cfabf112bc"
UPSTREAM_BLOB_SHA = "ca69ae6a2f6aae5bb68498216316a71eb07c1f7b"
UPSTREAM_FETCH_NOTE = (
    "fetched 2026-10-12 via the api.github.com contents endpoint; the blob "
    "matched at commit f53d552 and at main at fetch time"
)

# RFC #420 fingerprint matrix: video shift 12.0, audio shift 3.0.
CASES = [
    (2, 3.0),
    (2, 12.0),
    (4, 3.0),
    (4, 12.0),
    (25, 3.0),
    (25, 12.0),
]


def _git_blob_sha(data: bytes) -> str:
    header = b"blob %d\x00" % len(data)
    return hashlib.sha1(header + data).hexdigest()


def _extract_set_timesteps(source: str) -> ast.FunctionDef:
    """Return the ``set_timesteps`` FunctionDef of ``MiniMaxH3Scheduler``."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Scheduler":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "set_timesteps":
                    return item
    raise SystemExit("MiniMaxH3Scheduler.set_timesteps not found in upstream source")


def _run_upstream_set_timesteps(
    func: ast.FunctionDef, num_inference_steps: int, shift: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the extracted upstream method once against a shim instance."""
    module = ast.Module(body=[copy.deepcopy(func)], type_ignores=[])
    code = compile(module, "<upstream set_timesteps>", "exec")
    namespace: dict = {"torch": torch}
    exec(code, namespace)  # the point is to run upstream's own code

    scheduler = types.SimpleNamespace(
        num_inference_steps=None,
        sigmas=None,
        timesteps=None,
        _shift=float(shift),
        _step_index=None,
        _begin_index=None,
    )
    namespace["set_timesteps"](scheduler, num_inference_steps=num_inference_steps, device=None)
    return scheduler.sigmas, scheduler.timesteps


def _bits(tensor: torch.Tensor) -> list[str]:
    values = tensor.detach().cpu().contiguous().view(torch.int32).tolist()
    return [f"{value & 0xFFFFFFFF:08x}" for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.upstream_file, "rb") as handle:
        data = handle.read()
    blob_sha = _git_blob_sha(data)
    if blob_sha != UPSTREAM_BLOB_SHA:
        raise SystemExit(
            f"upstream file blob sha {blob_sha} != pinned {UPSTREAM_BLOB_SHA}; "
            "refusing to regenerate the fixture from an unknown revision"
        )

    func = _extract_set_timesteps(data.decode("utf-8"))
    cases = []
    for num_inference_steps, shift in CASES:
        sigmas, timesteps = _run_upstream_set_timesteps(func, num_inference_steps, shift)
        cases.append(
            {
                "num_inference_steps": num_inference_steps,
                "shift": shift,
                "sigmas_count": int(sigmas.numel()),
                "sigmas_bits": _bits(sigmas),
                "timesteps_count": int(timesteps.numel()),
                "timesteps_bits": _bits(timesteps),
                "num_model_evals": int(timesteps.numel()),
            }
        )

    fixture = {
        "upstream": {
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "blob_sha1": UPSTREAM_BLOB_SHA,
            "fetch_note": UPSTREAM_FETCH_NOTE,
            "class": "MiniMaxH3Scheduler",
            "method": "set_timesteps",
            "execution": (
                "the method body is AST-extracted and executed as-is against a "
                "namespace shim; no formula is transcribed"
            ),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
        "generator": "make_upstream_fixture.py --upstream-file <src> --out <this file>",
        "cases": cases,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out} with {len(cases)} cases")


if __name__ == "__main__":
    main()
