# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_arch_fingerprint`` (RFC #420): the pinned MiniMax-H3 architecture facts.

Everything here is stdlib/CPU. The tests pin three things:

* the analytic tensor table (derived from ``config.json``) equals the digest of the
  real checkpoint headers, so the fingerprint cannot drift from the release;
* the graph vocabulary equals the operator-validation graph in ``h3_block`` and
  every node is owned by exactly one RFC row;
* corrupted inputs (wrong shape, wrong dtype, wrong config, edited fixture) fail closed.
"""

from __future__ import annotations

import copy
import json
import os
import struct
from pathlib import Path

import pytest

from rl_engine.models.minimax_h3 import fingerprint as fp
from rl_engine.validation.models import h3_block

SNAPSHOT = fp.pinned_checkpoint()


def test_fixture_is_pinned_to_the_documented_revision() -> None:
    assert SNAPSHOT["repo"] == fp.REPO
    assert SNAPSHOT["revision"] == fp.REVISION
    assert set(SNAPSHOT["tensor_tables"]) == set(fp.VARIANTS.values())


@pytest.mark.parametrize("variant", sorted(fp.VARIANTS))
def test_analytic_table_equals_checkpoint_headers(variant: str) -> None:
    table = fp.expected_tensor_table(fp.transformer_config(variant))
    pinned = SNAPSHOT["tensor_tables"][fp.VARIANTS[variant]]
    assert fp.tensor_table_digest(table) == pinned["digest"]
    assert fp.parameter_count(table) == pinned["parameters"] == 33_122_992_896
    assert len(table) == pinned["tensors"] == 638


def test_variants_share_one_architecture() -> None:
    assert fp.transformer_config("FL2VA") == fp.transformer_config("Ref2VA")
    digests = {info["digest"] for info in SNAPSHOT["tensor_tables"].values()}
    assert len(digests) == 1


def test_derived_quantities() -> None:
    derived = fp.derived_quantities(fp.transformer_config())
    assert derived["attention_inner_dim"] == 7168
    assert derived["attention_inner_dim_differs_from_hidden"] is True
    assert derived["rotary_channels"] == 96
    assert derived["rotary_pass_through_channels"] == 32
    assert derived["adaln_projection_out"] == 96768
    assert derived["ffn_gate_up_out"] == 28672


def test_mixed_precision_matches_the_declared_modules() -> None:
    table = fp.expected_tensor_table(fp.transformer_config())
    for name, (dtype, _shape) in table.items():
        expect_fp32 = any(name.startswith(module + ".") for module in fp.FP32_MODULES)
        assert dtype == ("F32" if expect_fp32 else "BF16"), name


def test_fingerprint_is_stable_and_self_consistent() -> None:
    first = fp.build_fingerprint()
    second = fp.build_fingerprint()
    assert first == second
    assert first["fingerprint_sha256"] == fp.fingerprint_digest(first)
    assert first["graph"]["blocks"] == 50
    assert first["graph"]["refiner_blocks"] == 2
    assert first["graph"]["block_nodes_per_block"] == 17
    assert first["graph"]["refiner_nodes_per_block"] == len(fp.REFINER_NODES)
    expected_nodes = (
        len(fp.STEM_NODES)
        + 1  # packed_index_copy
        + 2 * len(fp.REFINER_NODES)
        + 1  # refiner final norm
        + 50 * len(fp.BLOCK_NODES)
        + len(fp.TAIL_NODES)
    )
    assert first["graph"]["nodes"] == expected_nodes


def test_committed_golden_matches() -> None:
    golden = (
        Path(__file__).resolve().parents[3] / "rl_engine/models/minimax_h3/arch_fingerprint.json"
    )
    if not golden.exists():
        pytest.skip("no committed golden fingerprint")
    assert json.loads(golden.read_text()) == fp.build_fingerprint()


# --- graph ------------------------------------------------------------------------------------


def test_block_vocabulary_equals_the_operator_validation_graph() -> None:
    assert fp.BLOCK_NODE_NAMES == h3_block.NODE_NAMES


def test_block_node_owners_equal_h3_block_bindings() -> None:
    owners = dict(fp.BLOCK_NODES)
    bindings = {node.name: node.binding.rfc_row for node in h3_block.NODES}
    assert owners == bindings


def test_every_graph_node_has_exactly_one_owner_and_a_precision() -> None:
    graph = fp.build_graph()
    paths = [node.path for node in graph]
    assert len(paths) == len(set(paths))
    assert all(node.row for node in graph)
    assert {node.precision for node in graph} <= {"fp32", "bf16"}


def test_graph_covers_both_precisions_in_the_right_places() -> None:
    by_path = {node.path: node for node in fp.build_graph()}
    for path in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
        assert by_path[path].precision == "fp32", path
    assert by_path["transformer_blocks.0.q_proj"].precision == "bf16"
    assert by_path["transformer_blocks.49.ffn_down"].precision == "bf16"


def test_graph_order_is_execution_order() -> None:
    paths = [node.path for node in fp.build_graph()]
    assert paths.index("transformer_blocks.0.adaln_projection") < paths.index(
        "transformer_blocks.1.adaln_projection"
    )
    assert paths.index("token_refiner.refiner_blocks.1.ffn_down") < paths.index(
        "transformer_blocks.0.adaln_projection"
    )
    assert paths[-1] == "packed_index_select"


def test_unclaimed_rows_are_visible_in_the_graph() -> None:
    rows = set(fp.rows_in_graph())
    for row in (
        "audio_input_projection_fp32",
        "audio_output_projection_fp32",
        "text_context_projection",
        "h3_qkv_gemm",
        "token_refiner_block",
    ):
        assert row in rows


# --- scheduler --------------------------------------------------------------------------------


def test_scheduler_shifts_match_the_released_configs() -> None:
    assert float(SNAPSHOT["configs"]["scheduler"]["shift"]) == fp.SCHEDULER["video_shift"] == 12.0
    assert (
        float(SNAPSHOT["configs"]["audio_scheduler"]["shift"]) == fp.SCHEDULER["audio_shift"] == 3.0
    )
    assert fp.SCHEDULER["eta"] == 0.0


def test_component_hashes_cover_every_pinned_directory() -> None:
    hashes = fp.component_hashes()
    assert set(hashes) == set(SNAPSHOT["dirs"])
    assert all(len(value) == 64 for value in hashes.values())


# --- fail-closed ------------------------------------------------------------------------------


def _write_shard(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    header = {
        name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}
        for name, (dtype, shape) in tensors.items()
    }
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_checkpoint_headers_accept_the_expected_subset(tmp_path: Path) -> None:
    table = fp.expected_tensor_table(fp.transformer_config())
    subset = dict(list(table.items())[:40])
    _write_shard(tmp_path / "a.safetensors", subset)
    assert fp.validate_checkpoint_headers(tmp_path) == 40


def test_checkpoint_headers_reject_wrong_dtype_shape_and_unknown_names(tmp_path: Path) -> None:
    table = fp.expected_tensor_table(fp.transformer_config())
    name = "transformer_blocks.0.attn.to_q.weight"
    dtype, shape = table[name]
    bad_dtype = "F32" if dtype == "BF16" else "BF16"
    for tensors, needle in (
        ({name: (bad_dtype, shape)}, "dtype"),
        ({name: (dtype, shape[::-1])}, "shape"),
        ({"transformer_blocks.0.surprise.weight": (dtype, shape)}, "surprise"),
    ):
        directory = tmp_path / needle
        directory.mkdir()
        _write_shard(directory / "a.safetensors", tensors)
        with pytest.raises(ValueError, match=needle):
            fp.validate_checkpoint_headers(directory)


def test_checkpoint_headers_require_shards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no safetensors"):
        fp.validate_checkpoint_headers(tmp_path)


def test_edited_fixture_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = copy.deepcopy(dict(SNAPSHOT))
    broken["tensor_tables"]["transformer"] = {"digest": "0" * 64, "parameters": 1, "tensors": 1}
    monkeypatch.setattr(fp, "pinned_checkpoint", lambda: broken)
    with pytest.raises(ValueError, match="does not match"):
        fp.build_fingerprint()


def test_config_change_changes_the_analytic_digest() -> None:
    config = dict(fp.transformer_config())
    base = fp.tensor_table_digest(fp.expected_tensor_table(config))
    config["num_layers"] = config["num_layers"] - 1
    assert fp.tensor_table_digest(fp.expected_tensor_table(config)) != base


# --- live artifacts (skip when unavailable) ---------------------------------------------------


def test_local_shard_headers_match_the_fingerprint() -> None:
    root = os.environ.get("RL_KERNEL_H3_WEIGHTS")
    directory = Path(root) / "hf" / "transformer" if root else None
    if directory is None or not list(directory.glob("*.safetensors")):
        pytest.skip("RL_KERNEL_H3_WEIGHTS/hf/transformer has no shards")
    assert fp.validate_checkpoint_headers(directory) > 0


def test_diffusers_model_on_meta_device_matches_the_fingerprint() -> None:
    diffusers = pytest.importorskip("diffusers")
    torch = pytest.importorskip("torch")
    cls = getattr(diffusers, fp.ARCHITECTURE, None)
    if cls is None:
        pytest.skip(f"diffusers has no {fp.ARCHITECTURE}")
    with torch.device("meta"):
        model = cls.from_config(fp.transformer_config())
    fp.validate_diffusers_model(model)
    # A model with one fewer block must be rejected.
    smaller = dict(fp.transformer_config())
    smaller["num_layers"] -= 1
    with torch.device("meta"):
        bad = cls.from_config(smaller)
    with pytest.raises(ValueError):
        fp.validate_diffusers_model(bad)
