# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

from rl_engine.models.deepseek_v41.checkpoint import LayerCheckpoint, PinnedCheckpointReader


def test_original_layer_names_and_packed_scale_validation(tmp_path):
    tensors = {
        "layers.14.ffn.experts.0.w1.weight": torch.zeros(32, 16, dtype=torch.int8),
        "layers.14.ffn.experts.0.w1.scale": torch.full((32, 1), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
    }
    save_file(tensors, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "part.safetensors" for k in tensors}})
    )
    ckpt = LayerCheckpoint(tmp_path)
    assert len(ckpt.keys_for_layers((14,))) == 2
    assert ckpt.keys_for_layers((1,)) == ()
    w, s = ckpt.projection("layers.14.ffn.experts.0.w1", device="cpu")
    assert w.dtype == s.dtype == torch.uint8
    assert w.shape == (32, 16)
    with pytest.raises(KeyError):
        ckpt.tensor("layers.0.ffn.experts.0.w1.weight")


def test_index_cannot_escape_root(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside"}})
    )
    with pytest.raises(ValueError):
        LayerCheckpoint(tmp_path).tensor("x")


def test_mutable_revision_rejected_before_network():
    with pytest.raises(ValueError):
        PinnedCheckpointReader(revision="main")


def test_http_range_must_be_honored(monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, *_):
            raise AssertionError("must reject before reading body")

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())
    reader = object.__new__(PinnedCheckpointReader)
    reader.base = "https://huggingface.co/example/"
    reader.evidence = []
    with pytest.raises(RuntimeError, match="exact range"):
        reader._range("model-00001-of-00048.safetensors", 0, 7)
