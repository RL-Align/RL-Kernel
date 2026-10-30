# SPDX-License-Identifier: Apache-2.0
"""Fail closed on missing native H3 kernels and pin the candidate/reference identities."""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rl_engine.backends.extension import _C
from rl_engine.validation.models.h3_block import CandidateOps
from rl_engine.validation.models.h3_full_model import require_candidate_cuda
from rl_engine.validation.models.h3_weights import load_h3_manifest, sha256_tensor
from tools.validation.models.h3_evidence_integrity import verify_inventory
from tools.validation.models.h3_full_model import digest, verify_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True)  # still refuses an output that exists
    manifest = Path("examples/models/minimax_h3/reference-manifest.json")
    verify_inventory(a.reference_root, manifest)
    verify_checkpoint(
        a.checkpoint, Path("rl_engine/validation/models/h3_full_manifest.json"), all_files=True
    )
    if _C is None or not hasattr(_C, "h3_timestep_sinusoid_forward"):
        raise ValueError("candidate extension lacks H3 kernels")
    ops = CandidateOps()
    backends = require_candidate_cuda(ops)
    # Inject a real eager CUDA fallback and require the same guard used by the
    # full-model path to reject it before it can count as candidate execution.
    original = ops.norm
    try:
        ops.norm = torch.nn.RMSNorm(128).cuda()
        try:
            require_candidate_cuda(ops)
        except ValueError as exc:
            if "candidate norm fell back" not in str(exc):
                raise
        else:
            raise AssertionError("eager CUDA norm fallback was accepted")
    finally:
        ops.norm = original
    require_candidate_cuda(ops)
    extension = Path(_C.__file__).resolve()
    if not extension.is_relative_to(Path.cwd().resolve()):
        raise ValueError("candidate loaded an extension outside its clean source snapshot")
    spec = load_h3_manifest()
    index = json.loads(
        (a.checkpoint / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    weights = a.output / "weights"
    weights.mkdir()
    for section, filename in [
        ("tensors", "h3_conditioning.safetensors"),
        ("block_tensors", "h3_block0.safetensors"),
    ]:
        values = {}
        for shard in sorted({index[name] for name in spec[section]}):
            with safe_open(str(a.checkpoint / "transformer" / shard), framework="pt") as handle:
                for name, row in spec[section].items():
                    if index[name] == shard:
                        value = handle.get_tensor(name).contiguous()
                        if sha256_tensor(value) != row["sha256"]:
                            raise ValueError(f"block weight identity changed: {name}")
                        values[name] = value
        save_file(values, str(weights / filename))
    report = dict(
        extension=str(extension),
        extension_sha256=digest(extension),
        torch=torch.__version__,
        backends=backends,
        negative_controls={"eager_cuda_norm_fallback_rejected": True},
        reference_manifest_sha256=digest(manifest),
        build={
            key: os.environ.get(key)
            for key in ("CUDA_HOME", "MAX_JOBS", "KERNEL_ALIGN_USE_FAST_MATH")
        },
    )
    (a.output / "identity.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
