# SPDX-License-Identifier: Apache-2.0
"""Real file-content and load-boundary evidence tests, requiring no GPU."""

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from types import ModuleType

import pytest

from rl_engine.contracts.strict_provider import StrictProviderError
from rl_engine.runtime.provenance.provider_identity import (
    checkpoint_identity,
    create_worker_load_receipt,
    environment_identity,
    file_inventory,
    observe_imported_extension,
)

pytestmark = pytest.mark.unit


def test_checkpoint_content_drift_even_with_same_size_and_mtime(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "part-1.bin"}}), encoding="utf-8"
    )
    shard = tmp_path / "part-1.bin"
    shard.write_bytes(b"weights1")
    before = checkpoint_identity(tmp_path, model_id="test/model", revision="v1")
    stat = shard.stat()
    shard.write_bytes(b"weights2")
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = checkpoint_identity(tmp_path, model_id="test/model", revision="v1")
    assert before["shard_inventory_sha256"] != after["shard_inventory_sha256"]
    assert before["evidence_kind"] == "disk_checkpoint"


def test_inventory_is_relocatable_but_rejects_escape_and_duplicates(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for root in (a, b):
        (root / "provider.py").write_bytes(b"same code")
    assert file_inventory(a, ["provider.py"]) == file_inventory(b, ["provider.py"])
    with pytest.raises(StrictProviderError, match="inside root"):
        file_inventory(a, ["../b/provider.py"])
    with pytest.raises(StrictProviderError, match="Duplicate"):
        file_inventory(a, ["provider.py", "provider.py"])


def test_imported_extension_observation_does_not_claim_loaded_bytes(tmp_path, monkeypatch):
    artifact = tmp_path / "native.pyd"
    artifact.write_bytes(b"binary A")
    name = "_strict_manifest_fake_extension"
    module = ModuleType(name)
    loader = importlib.machinery.ExtensionFileLoader(name, str(artifact))
    module.__spec__ = importlib.util.spec_from_file_location(name, artifact, loader=loader)
    module.__file__ = str(artifact)
    monkeypatch.setitem(sys.modules, name, module)
    before = observe_imported_extension(module)
    artifact.write_bytes(b"binary B")
    after = observe_imported_extension(module)
    assert before["current_disk_sha256"] != after["current_disk_sha256"]
    assert before["loaded_binary_verified"] is False
    assert after["loaded_binary_verified"] is False
    assert before["evidence_kind"] == "imported_extension_current_disk"
    monkeypatch.delitem(sys.modules, name)
    with pytest.raises(StrictProviderError, match="sys.modules"):
        observe_imported_extension(module)


def test_python_module_is_not_native_evidence():
    with pytest.raises(StrictProviderError, match="ExtensionFileLoader"):
        observe_imported_extension(json)


def test_worker_load_receipt_checks_content_and_refresh():
    loaded = {"part-1": b"loaded shard 1", "part-2": b"loaded shard 2"}
    expected = {name: hashlib.sha256(content).hexdigest() for name, content in loaded.items()}
    kwargs = dict(
        snapshot_id="snapshot-1",
        generation=1,
        worker_id="worker-0",
        expected_shard_digests=expected,
        read_loaded_shard=loaded.__getitem__,
        required_refresh=["packed_weights"],
        read_refreshed_component=lambda name: b"packed",
    )
    receipt = create_worker_load_receipt(**kwargs)
    assert receipt.validate() is receipt
    assert receipt.load_complete and receipt.cache_refresh_complete
    assert receipt.evidence_kind == "worker_load"
    loaded["part-1"] = b"wrong weights"
    with pytest.raises(StrictProviderError) as error:
        create_worker_load_receipt(**kwargs)
    assert error.value.code == "loaded_content_mismatch"
    loaded["part-1"] = b"loaded shard 1"
    kwargs["read_refreshed_component"] = None
    with pytest.raises(StrictProviderError) as error:
        create_worker_load_receipt(**kwargs)
    assert error.value.code == "incomplete_refresh"


def test_load_complete_boolean_is_not_content_evidence():
    with pytest.raises(StrictProviderError, match="byte view"):
        create_worker_load_receipt(
            snapshot_id="s",
            generation=0,
            worker_id="w",
            expected_shard_digests={"part": "a" * 64},
            read_loaded_shard=lambda name: True,
        )


def test_environment_collection_uses_only_explicit_names(monkeypatch):
    monkeypatch.setenv("STRICT_TEST_FLAG", "1")
    monkeypatch.setenv("SECRET_DO_NOT_COLLECT", "secret")
    result = environment_identity(environment_names=["STRICT_TEST_FLAG"])
    assert result["environment"] == {"STRICT_TEST_FLAG": {"value": "1"}}
    assert "secret" not in repr(result)
