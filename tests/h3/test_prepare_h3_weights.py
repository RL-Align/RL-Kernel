# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Weight extraction rejects off-manifest tensors even under Python optimization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.testing.h3_weights import EXTRACTED_FILE, sha256_file, sha256_tensor


@pytest.mark.parametrize("optimize", [0, 1])
@pytest.mark.parametrize("mismatch", [None, "dtype", "shape", "sha256"])
def test_manifest_validation_before_write(tmp_path, monkeypatch, optimize, mismatch):
    """Enforce tensor contracts before extraction writes, including under Python optimization."""

    import huggingface_hub
    from safetensors.torch import load_file, save_file

    tensor = torch.tensor([0.0, 1.0])
    config = tmp_path / "config.json"
    config.write_text("{}")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
    shard = tmp_path / "shard.safetensors"
    save_file({"weight": tensor}, str(shard))
    manifest = {
        "model_identity": {
            "hf_repo": "test/h3",
            "subfolder": "dit",
            "revision": "pinned",
            "config_sha256": sha256_file(config),
            "index_file": index.name,
            "index_sha256": sha256_file(index),
        },
        "weight_shards": {shard.name: {"sha256": sha256_file(shard)}},
        "tensors": {
            "weight": {
                "dtype": "bfloat16" if mismatch == "dtype" else "float32",
                "shape": [3] if mismatch == "shape" else [2],
                "sha256": "0" * 64 if mismatch == "sha256" else sha256_tensor(tensor),
            }
        },
    }
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, filename, **kwargs: str(tmp_path / Path(filename).name),
    )
    out_dir = tmp_path / "extracted"
    monkeypatch.setattr(sys, "argv", ["prepare_h3_weights.py", "--out", str(out_dir)])
    script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_h3_weights.py"
    namespace = {"__name__": "prepare_h3_weights_test", "__file__": str(script)}
    # Compiling the actual script at optimize=1 reproduces python -O's assert removal.
    exec(compile(script.read_text(), str(script), "exec", optimize=optimize), namespace)
    namespace["load_h3_manifest"] = lambda: manifest
    target = out_dir / EXTRACTED_FILE
    if mismatch:
        with pytest.raises(SystemExit, match=f"weight: {mismatch} .* != manifest"):
            namespace["main"]()
        assert not target.exists()
    else:
        namespace["main"]()
        assert torch.equal(load_file(str(target))["weight"], tensor)
