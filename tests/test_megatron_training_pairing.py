# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for the pairing handoff: exact inputs and rejection of identity drift."""

import hashlib
import json

import pytest
import torch
from megatron_training_pairing import load_inputs, prepare_inputs

pytestmark = pytest.mark.unit


def test_pairing_inputs_reproduce_same_identity_and_preserve_rng(tmp_path):
    rng = torch.get_rng_state().clone()
    left, right = tmp_path / "a", tmp_path / "b"
    a, b = prepare_inputs(left), prepare_inputs(right)
    fixture, manifest = load_inputs(left)
    assert a["fixture_checksum"] == b["fixture_checksum"]
    assert a["weight_lock_fingerprint"] == b["weight_lock_fingerprint"]
    assert fixture.hidden.dtype == fixture.lm_head_weight.dtype == torch.bfloat16
    assert fixture.target_ids.dtype == torch.int64
    assert fixture.active_mask.dtype == torch.bool
    assert len(manifest["row_keys"]) == 96
    assert int(fixture.active_mask.sum()) == 77
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(FileExistsError):
        prepare_inputs(left)


def test_pairing_refuses_changed_input_even_with_updated_file_checksum(tmp_path):
    directory = tmp_path / "input"
    prepare_inputs(directory)
    path = directory / "inputs.pt"
    tensors = torch.load(path, weights_only=True)
    tensors["lm_head_weight_local"].view(torch.int16)[0, 0] ^= 1
    torch.save(tensors, path)
    with pytest.raises(ValueError, match="inputs.pt checksum mismatch"):
        load_inputs(directory)
    meta = directory / "fixture.json"
    manifest = json.loads(meta.read_text())
    manifest["inputs_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lm_head_weight_local"):
        load_inputs(directory)


@pytest.mark.parametrize("field", ["row_keys", "environment", "weight_lock"])
def test_pairing_refuses_changed_identity(tmp_path, field):
    directory = tmp_path / "input"
    prepare_inputs(directory)
    meta = directory / "fixture.json"
    manifest = json.loads(meta.read_text())
    manifest[field] = None
    meta.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity/configuration"):
        load_inputs(directory)
