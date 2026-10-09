# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tiny CPU export-publication regressions, not full-model numerical acceptance."""

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen3_next_training_export as export  # noqa: E402
from qwen3_next_runtime import file_digest  # noqa: E402


def write_weights(root, values, filename):
    save_file(values, root / filename)
    index = {
        "metadata": {
            "total_size": sum(value.numel() * value.element_size() for value in values.values())
        },
        "weight_map": {name: filename for name in values},
    }
    (root / export.INDEX_FILE).write_text(json.dumps(index))


@pytest.fixture
def staged(tmp_path, monkeypatch):
    base = tmp_path / "unit-official"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps({"model_type": "qwen3_next", "num_hidden_layers": 48})
    )
    (base / "tokenizer.json").write_text(json.dumps({"unit_fixture": "not a real tokenizer"}))
    values = {
        f"model.layers.{layer}.input_layernorm.weight": torch.ones(2, dtype=torch.bfloat16)
        for layer in range(48)
    }
    values.update(
        {
            "model.embed_tokens.weight": torch.ones((4, 2), dtype=torch.bfloat16),
            "model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
            "lm_head.weight": torch.ones((4, 2), dtype=torch.bfloat16),
        }
    )
    write_weights(base, values, "base.safetensors")
    base_identity = {
        "revision": base.name,
        "metadata": {
            path.name: file_digest(path) for path in base.iterdir() if path.suffix != ".safetensors"
        },
        "shards": {"base.safetensors": {"size": (base / "base.safetensors").stat().st_size}},
    }
    monkeypatch.setattr(export, "_reference", lambda: {"revision": base.name})
    schema_reference = {
        "tensor_count": len(values),
        "weight_schema_sha256": export._json_digest(
            export._weight_schema(base, check_finite=False)
        ),
    }
    monkeypatch.setattr(export, "_schema_reference", lambda: schema_reference)

    def identity(path):
        assert Path(path) == base
        return copy.deepcopy(base_identity)

    monkeypatch.setattr(export, "checkpoint_identity", identity)
    staging, destination = tmp_path / "export.partial", tmp_path / "export-complete"
    staging.mkdir()
    for name in ("config.json", "tokenizer.json"):
        shutil.copy2(base / name, staging / name)
    write_weights(
        staging, {name: value + 1 for name, value in values.items()}, "updated.safetensors"
    )
    (staging / "STAGING.json").write_text(json.dumps({"scope": export.KIND, "status": "staging"}))
    training = {
        "source_commit": "unit-source",
        "source_tree": "unit-tree",
        "environment_sha256": "unit-env",
        "provider_config": {"provider": "strict"},
        "topology": {"tp": 4, "pp": 1, "cp": 1, "ep": 1},
        "checkpoint": str(base),
        "checkpoint_identity": base_identity,
    }
    return {
        "staging": staging,
        "destination": destination,
        "step": 1,
        "weight_version": "update-1",
        "identity": training,
        "base_checkpoint": base,
        "base_checkpoint_identity": base_identity,
        "model_hashes": [str(rank) * 64 for rank in range(4)],
    }


def publish(staged):
    return export._publish_export(**staged)


def test_atomic_native_export_identity_and_all_file_verification(staged):
    record = publish(staged)
    assert not staged["staging"].exists()
    assert (
        record["metadata"][export.INDEX_FILE]
        != staged["base_checkpoint_identity"]["metadata"][export.INDEX_FILE]
    )
    assert export.validate_training_export(staged["destination"], record) == record
    assert record["tensor_count"] == 51
    assert record["model_identity_method"] == "native_parameter_hashes_after"


def test_history_record_survives_weight_retirement_but_cannot_load(staged):
    record = publish(staged)
    (staged["destination"] / "updated.safetensors").unlink()
    assert export.validate_training_export_record(record, record) == record
    with pytest.raises(ValueError, match="file inventory changed"):
        export.validate_training_export(staged["destination"], record)


def test_same_size_weight_corruption_is_rejected(staged):
    record = publish(staged)
    path = staged["destination"] / "updated.safetensors"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        export.validate_training_export(staged["destination"], record)


def test_export_requires_completion_publication(staged):
    with pytest.raises(ValueError, match="unpublished"):
        export.validate_training_export(staged["staging"], {})


@pytest.mark.parametrize("missing", ["model.layers.47.input_layernorm.weight", "lm_head.weight"])
def test_reduced_model_or_missing_head_cannot_publish(staged, missing):
    from safetensors.torch import load_file

    values = load_file(staged["staging"] / "updated.safetensors")
    del values[missing]
    write_weights(staged["staging"], values, "updated.safetensors")
    with pytest.raises(ValueError, match="omitted trained official tensors"):
        publish(staged)
    assert not (staged["staging"] / export.COMPLETE_FILE).exists()


@pytest.mark.parametrize("mutation", ["dtype", "nonfinite", "shape"])
def test_export_preserves_finite_original_weight_schema(staged, mutation):
    from safetensors.torch import load_file

    values = load_file(staged["staging"] / "updated.safetensors")
    key = "lm_head.weight"
    if mutation == "dtype":
        values[key] = values[key].float()
    elif mutation == "nonfinite":
        values[key][0, 0] = float("nan")
    else:
        values[key] = values[key][:2].contiguous()
    write_weights(staged["staging"], values, "updated.safetensors")
    with pytest.raises(ValueError, match="BF16|Nonfinite|complete official tensor schema"):
        publish(staged)


def test_modified_tokenizer_cannot_publish(staged):
    (staged["staging"] / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="tokenizer/configuration assets"):
        publish(staged)


def test_different_requested_export_identity_is_rejected(staged):
    record = publish(staged)
    expected = {**record, "weight_version": "reload-2", "step": 2}
    with pytest.raises(ValueError, match="requested weights"):
        export.validate_training_export(staged["destination"], expected)


@pytest.mark.parametrize("field", ["weights_content_sha256", "metadata", "model_identity_method"])
def test_self_consistent_expected_record_cannot_bypass_required_proofs(staged, field):
    record = publish(staged)
    record[field] = {} if field == "metadata" else "wrong"
    with pytest.raises(ValueError, match="proof|parameter audit"):
        export.validate_training_export_record(record, record)


def test_export_base_and_native_training_provenance_must_match(staged):
    record = publish(staged)
    record["training_identity"]["checkpoint"] = "/different-base"
    with pytest.raises(ValueError, match="baseline/source/environment provenance"):
        export.validate_training_export_record(record, record)


def test_no_unmanifested_file_or_symlink_is_accepted(staged):
    record = publish(staged)
    destination = staged["destination"]
    (destination / "extra").write_text("unmanifested")
    with pytest.raises(ValueError, match="file inventory changed"):
        export.validate_training_export(destination, record)
    (destination / "extra").unlink()
    (destination / "tokenizer.json").unlink()
    (destination / "tokenizer.json").symlink_to(staged["base_checkpoint"] / "tokenizer.json")
    with pytest.raises(ValueError, match="external HF export"):
        export.validate_training_export(destination, record)


def _add_mtp_head_to_base(staged, monkeypatch, n_tensors=3):
    """Give the official base an untrained MTP head the native saver never writes."""
    from safetensors.torch import load_file

    base = staged["base_checkpoint"]
    values = load_file(base / "base.safetensors")
    mtp = {
        f"mtp.layers.0.mlp.experts.{i}.down_proj.weight": torch.full(
            (2, 2), float(i + 1), dtype=torch.bfloat16
        )
        for i in range(n_tensors)
    }
    values.update(mtp)
    write_weights(base, values, "base.safetensors")
    staged["base_checkpoint_identity"]["metadata"][export.INDEX_FILE] = file_digest(
        base / export.INDEX_FILE
    )
    staged["identity"]["checkpoint_identity"] = staged["base_checkpoint_identity"]
    schema_reference = {
        "tensor_count": len(values),
        "weight_schema_sha256": export._json_digest(
            export._weight_schema(base, check_finite=False)
        ),
    }
    monkeypatch.setattr(export, "_schema_reference", lambda: schema_reference)
    return mtp


def test_untrained_mtp_head_is_carried_from_the_base(staged, monkeypatch):
    from safetensors.torch import load_file

    mtp = _add_mtp_head_to_base(staged, monkeypatch)
    record = publish(staged)
    passthrough = record["base_passthrough"]
    assert passthrough["tensor_count"] == len(mtp) and passthrough["prefixes"] == ["mtp."]
    (shard,) = passthrough["shards"]
    assert passthrough["shards"][shard] == {"source": "base.safetensors", "tensor_count": len(mtp)}
    index = json.loads((staged["destination"] / export.INDEX_FILE).read_text())
    assert {name: index["weight_map"][name] for name in mtp} == {name: shard for name in mtp}
    carried = load_file(staged["destination"] / shard)
    assert all(torch.equal(carried[name], value) for name, value in mtp.items())
    assert record["tensor_count"] == 51 + len(mtp)
    assert export.validate_training_export(staged["destination"], record) == record


def test_passthrough_is_limited_to_the_mtp_head(staged, monkeypatch):
    from safetensors.torch import load_file

    _add_mtp_head_to_base(staged, monkeypatch)
    values = load_file(staged["staging"] / "updated.safetensors")
    del values["model.norm.weight"]
    write_weights(staged["staging"], values, "updated.safetensors")
    with pytest.raises(
        ValueError, match="omitted trained official tensors: \\['model.norm.weight'\\]"
    ):
        publish(staged)
    assert not (staged["staging"] / export.COMPLETE_FILE).exists()


def test_saver_must_not_write_part_of_the_passthrough_head(staged, monkeypatch):
    from safetensors.torch import load_file

    mtp = _add_mtp_head_to_base(staged, monkeypatch)
    values = load_file(staged["staging"] / "updated.safetensors")
    name = next(iter(mtp))
    values[name] = mtp[name]
    write_weights(staged["staging"], values, "updated.safetensors")
    with pytest.raises(ValueError, match="part of the base passthrough head"):
        publish(staged)


def test_record_must_declare_the_base_passthrough(staged):
    record = publish(staged)
    assert record["base_passthrough"] == {"prefixes": ["mtp."], "tensor_count": 0, "shards": {}}
    broken = copy.deepcopy(record)
    del broken["base_passthrough"]
    with pytest.raises(ValueError, match="came from the base"):
        export.validate_training_export_record(broken, broken)
