# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_image_arch_fingerprint`` (RFC #738): the pinned MiniMax-H3-Image architecture facts.

Everything here is stdlib/CPU except two torch / Diffusers cross-checks, which skip when the
package is missing. The tests pin:

* the analytic tensor table (H3 minus audio minus ``proj_out`` plus six PDD heads) equals
  the digest of the real Diffusers-layout headers, and the fused PDD file has the stacked
  ``[6, 96, 5376]`` / ``[6, 96]`` FP32 table;
* the six PDD heads are byte-identical across the two layouts, pairwise distinct, and the
  upstream layout's ``final_layer.video_out`` is none of them;
* the sigma grid, its float32 round trip and the velocity-form scheduler settings;
* the text encoder is the MiniMax-H3 one, file by file;
* the graph reuses the H3 block vocabulary and every node has one owning RFC row;
* corrupted inputs (wrong shape, wrong dtype, wrong head count, edited snapshot) fail closed.
"""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from rl_engine.models.minimax_h3 import fingerprint as h3_fp
from rl_engine.models.minimax_h3_image import fingerprint as fp

SNAPSHOT = fp.pinned_checkpoint()
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def edited(monkeypatch: pytest.MonkeyPatch):
    """Yield a deep copy of the snapshot that ``fp.pinned_checkpoint()`` then returns."""

    data: dict[str, Any] = copy.deepcopy(dict(SNAPSHOT))
    monkeypatch.setattr(fp, "pinned_checkpoint", lambda: data)
    return data


def test_fixture_is_pinned_to_the_documented_revisions() -> None:
    assert SNAPSHOT["repo"] == fp.REPO == "MiniMaxAI/MiniMax-H3-Image"
    assert SNAPSHOT["revision"] == fp.REVISION
    assert SNAPSHOT["inherited_text_encoder"]["revision"] == h3_fp.REVISION
    assert set(SNAPSHOT["tensor_tables"]) == {
        "transformer",
        "transformer_diffusers",
        "vae",
        "vae_diffusers",
        "pdd",
    }


def test_analytic_table_equals_diffusers_checkpoint_headers() -> None:
    table = fp.expected_tensor_table(fp.transformer_config())
    pinned = SNAPSHOT["tensor_tables"]["transformer_diffusers"]
    assert fp.tensor_table_digest(table) == pinned["digest"]
    assert fp.parameter_count(table) == pinned["parameters"] == 33_125_224_384
    assert len(table) == pinned["tensors"] == 644


def test_table_is_h3_without_audio_or_proj_out_plus_pdd_heads() -> None:
    h3 = h3_fp.expected_tensor_table(h3_fp.transformer_config())
    image = fp.expected_tensor_table(fp.transformer_config())
    removed = set(h3) - set(image)
    added = set(image) - set(h3)
    assert removed == {
        "audio_proj_in.weight",
        "audio_proj_in.bias",
        "audio_proj_out.weight",
        "audio_proj_out.bias",
        "proj_out.weight",
        "proj_out.bias",
    }
    assert added == {f"pdd_proj_out.{i}.{k}" for i in range(6) for k in ("weight", "bias")}
    for name in set(h3) & set(image):
        assert h3[name] == image[name], name
    for index in range(6):
        assert image[f"pdd_proj_out.{index}.weight"] == h3["proj_out.weight"]


def test_pdd_file_header_is_the_stacked_fp32_table() -> None:
    pdd = SNAPSHOT["tensor_tables"]["pdd"]
    assert pdd["table"] == {
        "video_out.bias": ["F32", [6, 96]],
        "video_out.weight": ["F32", [6, 96, 5376]],
    }
    assert pdd["digest"] == fp.tensor_table_digest(fp.expected_pdd_table(fp.transformer_config()))


def test_pdd_heads_are_identical_across_layouts_and_distinct() -> None:
    heads = fp.pdd_head_digests()
    assert len(heads) == fp.NUM_PDD_HEADS == 6
    assert heads == SNAPSHOT["head_sha256"]["diffusers_pdd_proj_out"]
    assert len({(h["weight"], h["bias"]) for h in heads}) == 6


def test_upstream_shared_head_is_not_a_pdd_head() -> None:
    shared = SNAPSHOT["head_sha256"]["upstream_video_out"]
    assert all(shared["weight"] != h["weight"] for h in fp.pdd_head_digests())


def test_pdd_head_mismatch_fails_closed(edited: dict) -> None:
    edited["head_sha256"]["diffusers_pdd_proj_out"][3]["weight"] = "0" * 64
    with pytest.raises(ValueError, match="PDD head 3"):
        fp.pdd_head_digests()


def test_reordered_pdd_heads_fail_closed(edited: dict) -> None:
    heads = edited["head_sha256"]["diffusers_pdd_proj_out"]
    heads[0], heads[1] = heads[1], heads[0]
    with pytest.raises(ValueError, match="PDD head 0"):
        fp.pdd_head_digests()


def test_shared_head_equal_to_a_pdd_head_fails_closed(edited: dict) -> None:
    edited["head_sha256"]["upstream_video_out"] = dict(edited["head_sha256"]["pdd_fused"][2])
    with pytest.raises(ValueError, match="coincides"):
        fp.pdd_head_digests()


def test_derived_quantities() -> None:
    derived = fp.derived_quantities(fp.transformer_config())
    assert derived["attention_inner_dim"] == 7168
    assert derived["rotary_channels"] == 96
    assert derived["rotary_pass_through_channels"] == 32
    assert derived["image_patch_width"] == 96
    assert derived["adaln_projection_out"] == 96768
    assert derived["final_adaln_out"] == 10752
    assert derived["canvas_multiple"] == 32


def test_upstream_layout_agrees_with_diffusers_layout(edited: dict) -> None:
    edited["configs"]["transformer"]["ffn_hidden_size"] = 14000
    with pytest.raises(ValueError, match="ffn_dim"):
        fp.build_fingerprint()


def test_upstream_audio_must_be_disabled(edited: dict) -> None:
    edited["configs"]["transformer"]["audio_enabled"] = True
    with pytest.raises(ValueError, match="audio_enabled"):
        fp.build_fingerprint()


def test_mixed_precision_matches_the_declared_modules() -> None:
    table = fp.expected_tensor_table(fp.transformer_config())
    for name, (dtype, _shape) in table.items():
        expect_fp32 = any(name.startswith(module + ".") for module in fp.FP32_MODULES)
        assert dtype == ("F32" if expect_fp32 else "BF16"), name


# --- sigma grid / scheduler ------------------------------------------------------------------


def test_sigma_grid_is_the_released_seven_points() -> None:
    grid = fp.sigma_grid()
    assert grid["config_sigmas"] == [
        1.0,
        0.9675236940383911,
        0.8875485062599182,
        0.75,
        0.5327739119529724,
        0.23200887441635132,
        0.0,
    ]
    assert len(grid["timesteps"]) == 6
    assert grid["timesteps"][0] == 0.0
    # The released grid survives the float32 round trip unchanged.
    assert grid["step_sigmas"] == grid["config_sigmas"]
    assert grid["round_trip_moved"] == []


def test_sigma_round_trip_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    sigmas = SNAPSHOT["configs"]["pdd_config"]["sigmas"]
    # MiniMaxH3ImageSetTimestepsStep, then MiniMaxH3Scheduler.set_timesteps(sigmas=...).
    timesteps = torch.tensor([1.0 - s for s in sigmas[:-1]], dtype=torch.float32)
    step = torch.cat([1.0 - timesteps, torch.zeros(1)])
    step = torch.as_tensor(step, dtype=torch.float64).to(torch.float32)
    grid = fp.sigma_grid()
    assert step.tolist() == grid["step_sigmas"]
    assert (1.0 - step[:-1]).tolist() == grid["timesteps"] == timesteps.tolist()


def _f32_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_round_trip_reports_a_sigma_it_moves(edited: dict) -> None:
    # Below 0.5, float32(1 - float32(1 - s)) can differ from s by one ulp. Walk float32 values
    # up from just above 0.2 until one moves, and put it in place of sigma 5 (0.232...).
    bits = struct.unpack("<I", struct.pack("<f", 0.2))[0]
    for offset in range(1, 64):
        value = _f32_bits(bits + offset)
        if (
            struct.unpack(
                "<f",
                struct.pack("<f", 1.0 - struct.unpack("<f", struct.pack("<f", 1.0 - value))[0]),
            )[0]
            != value
        ):
            break
    else:
        pytest.fail("no float32 value above 0.2 moved under the round trip")
    edited["configs"]["pdd_config"]["sigmas"][5] = value
    edited["configs"]["modular_model_index"]["sample_sigmas"][5] = value
    assert fp.sigma_grid()["round_trip_moved"] == [5]


@pytest.mark.parametrize(
    ("key", "index", "value", "needle"),
    [
        ("pdd_config", 6, 0.01, "1.0 to 0.0"),
        ("pdd_config", 2, 0.96875, "strictly decreasing"),
        ("pdd_config", 2, 0.1, "float32-representable"),
        ("modular_model_index", 3, 0.7, "differ"),
    ],
)
def test_bad_sigma_grids_fail_closed(
    edited: dict, key: str, index: int, value: float, needle: str
) -> None:
    field = "sigmas" if key == "pdd_config" else "sample_sigmas"
    if key == "pdd_config":
        edited["configs"]["modular_model_index"]["sample_sigmas"][index] = value
    edited["configs"][key][field][index] = value
    with pytest.raises(ValueError, match=needle):
        fp.sigma_grid()


def test_scheduler_settings_match_the_released_config(edited: dict) -> None:
    assert SNAPSHOT["configs"]["scheduler"]["euler_update_form"] == "velocity"
    assert SNAPSHOT["configs"]["scheduler"]["shift_provided_sigmas"] is False
    edited["configs"]["scheduler"]["euler_update_form"] = "blend"
    with pytest.raises(ValueError, match="euler_update_form"):
        fp.build_fingerprint()


# --- text encoder / VAE -----------------------------------------------------------------------


def test_text_encoder_is_inherited_from_minimax_h3() -> None:
    assert len(fp.check_inherited_text_encoder()) == 64
    assert len(SNAPSHOT["dirs"]["text_encoder"]) == len(SNAPSHOT["inherited_text_encoder"]["files"])


def test_changed_text_encoder_file_fails_closed(edited: dict) -> None:
    edited["dirs"]["text_encoder"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="text_encoder differs"):
        fp.check_inherited_text_encoder()


def test_vae_facts() -> None:
    vae = fp.vae_facts()
    assert vae["latent_channels"] == 24
    assert vae["spatial_compression"] == 16
    assert (vae["tile_size"], vae["tile_overlap_min"]) == (256, 64)
    assert vae["encoder_tiling"] and vae["decoder_tiling"]
    assert vae["decoder"]["layers"] == 24 and vae["decoder"]["register_tokens"] == 4
    assert vae["pixel_mean"] == [0.485, 0.456, 0.406]


def test_vae_layouts_must_share_latent_statistics(edited: dict) -> None:
    edited["configs"]["vae"]["latents_std"][7] = 1.0
    with pytest.raises(ValueError, match="latents_std"):
        fp.vae_facts()


# --- fingerprint ------------------------------------------------------------------------------


def test_fingerprint_is_stable_and_self_consistent() -> None:
    first = fp.build_fingerprint()
    assert first == fp.build_fingerprint()
    assert first["fingerprint_sha256"] == fp.fingerprint_digest(first)
    assert first["reference_implementation"]["commit"] == fp.DIFFUSERS_COMMIT
    assert first["pdd"]["upstream_shared_head_is_a_pdd_head"] is False
    expected_nodes = (
        len(fp.STEM_NODES)
        + 1  # packed_index_copy
        + 2 * len(h3_fp.REFINER_NODES)
        + 1  # refiner final norm
        + 50 * len(h3_fp.BLOCK_NODES)
        + len(fp.TAIL_NODES)
    )
    assert first["graph"]["nodes"] == expected_nodes == 890


def test_committed_golden_matches() -> None:
    golden = REPO_ROOT / "rl_engine/models/minimax_h3_image/arch_fingerprint.json"
    assert json.loads(golden.read_text()) == fp.build_fingerprint()


def test_edited_tensor_table_digest_fails_closed(edited: dict) -> None:
    edited["tensor_tables"]["transformer_diffusers"]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="analytic tensor table"):
        fp.build_fingerprint()


def test_wrong_head_count_fails_closed(edited: dict) -> None:
    edited["configs"]["transformer_diffusers"]["num_pdd_heads"] = 5
    with pytest.raises(ValueError, match="num_pdd_heads"):
        fp.build_fingerprint()


def test_wrong_revision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fp.pinned_checkpoint.cache_clear()
    monkeypatch.setattr(fp, "REVISION", "0" * 40)
    try:
        with pytest.raises(ValueError, match="repo/revision"):
            fp.pinned_checkpoint()
    finally:
        monkeypatch.undo()
        fp.pinned_checkpoint.cache_clear()


# --- graph ------------------------------------------------------------------------------------


def test_graph_reuses_the_h3_block_and_refiner_vocabulary() -> None:
    graph = fp.build_graph()
    block = [n for n in graph if n.stage == "transformer_blocks" and n.layer == 0]
    assert [(n.node, n.row) for n in block] == list(h3_fp.BLOCK_NODES)
    assert all(
        n.rfc == fp.RFC_H3 for n in graph if n.stage in ("transformer_blocks", "token_refiner")
    )


def test_every_node_has_one_owner_and_a_precision() -> None:
    graph = fp.build_graph()
    assert len({n.path for n in graph}) == len(graph)
    assert all(n.row and n.rfc in (fp.RFC_H3, fp.RFC_IMAGE) for n in graph)
    assert {n.precision for n in graph} <= set(fp.PRECISIONS)


def test_image_rows_are_the_rfc738_rows_and_no_audio_remains() -> None:
    rows = fp.rows_in_graph()
    assert rows["738"] == [
        "h3_image_pdd_head_contract",
        "image_packed_layout_schema",
        "image_patch_pack_unpack",
        "image_position_grid",
        "image_transformer_output_gather",
    ]
    assert not any("audio" in row for row in rows["420"])
    assert "video_output_projection_fp32" not in rows["420"]


def test_graph_order_is_execution_order() -> None:
    paths = [n.path for n in fp.build_graph()]
    assert paths.index("position_grid") < paths.index("rope")
    assert paths.index("token_refiner.final_norm") < paths.index("packed_index_copy")
    assert paths[-3:] == ["norm_out", "pdd_proj_out", "packed_index_select"]


# --- checkpoint headers -----------------------------------------------------------------------


def _write_shard(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    header = {
        name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}
        for name, (dtype, shape) in tensors.items()
    }
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_checkpoint_headers_accept_the_expected_subset_and_pdd_file(tmp_path: Path) -> None:
    config = fp.transformer_config()
    table = fp.expected_tensor_table(config)
    _write_shard(tmp_path / "a.safetensors", dict(list(table.items())[:40]))
    _write_shard(tmp_path / fp.PDD_FILE, fp.expected_pdd_table(config))
    assert fp.validate_checkpoint_headers(tmp_path) == 42


@pytest.mark.parametrize(
    ("tensors", "needle"),
    [
        ({"video_out.weight": ("F32", (5, 96, 5376)), "video_out.bias": ("F32", (6, 96))}, "shape"),
        (
            {"video_out.weight": ("BF16", (6, 96, 5376)), "video_out.bias": ("F32", (6, 96))},
            "dtype",
        ),
        ({"video_out.weight": ("F32", (6, 96, 5376))}, "missing tensor video_out.bias"),
    ],
)
def test_bad_pdd_file_fails_closed(tmp_path: Path, tensors: dict, needle: str) -> None:
    _write_shard(tmp_path / fp.PDD_FILE, tensors)
    with pytest.raises(ValueError, match=needle):
        fp.validate_checkpoint_headers(tmp_path)


def test_checkpoint_headers_reject_audio_and_single_proj_out(tmp_path: Path) -> None:
    for name in ("audio_proj_in.weight", "proj_out.weight"):
        directory = tmp_path / name
        directory.mkdir()
        _write_shard(directory / "a.safetensors", {name: ("F32", (96, 5376))})
        with pytest.raises(ValueError, match=f"unexpected tensor {name}"):
            fp.validate_checkpoint_headers(directory)


def test_live_diffusers_model_matches() -> None:
    transformers_mod = pytest.importorskip("diffusers.models.transformers")
    cls = getattr(transformers_mod, fp.ARCHITECTURE, None)
    if cls is None:
        pytest.skip(f"installed diffusers has no {fp.ARCHITECTURE}")
    torch = pytest.importorskip("torch")
    with torch.device("meta"):
        model = cls(**fp.transformer_config())
    fp.validate_diffusers_model(model)
