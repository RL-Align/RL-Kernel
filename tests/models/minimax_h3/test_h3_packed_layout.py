# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU schema acceptance against independent hand and pinned-upstream fixtures."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_engine.models.minimax_h3.packed_layout import H3PackedLayout, H3PackedLayoutError
from rl_engine.validation.models.h3_provider import provider_adaln_row_gather

REFERENCES = Path(__file__).resolve().parents[2] / "reference/minimax_h3/packed_layout"
HAND = json.loads((REFERENCES / "hand_written_case.json").read_text(encoding="utf-8"))
UPSTREAM = json.loads((REFERENCES / "upstream_cases.json").read_text(encoding="utf-8"))
CASES = {case["case_id"]: case for case in UPSTREAM["cases"]}
FIELDS = tuple(HAND["dtypes"])
PRIMARY = tuple(name for name in FIELDS if name != "adaln_indices")


def hand_inputs(*, phase="before", **overrides):
    result = {
        name: torch.tensor(
            HAND["timestep"] if name == "timestep" else HAND[phase][name],
            dtype=getattr(torch, HAND["dtypes"][name]),
            device="cpu",
        )
        for name in PRIMARY
    }
    result.update(
        batch_size=1,
        num_condition_video_rows=1,
        num_condition_audio_rows=0,
        provenance={
            "checkpoint_revision": UPSTREAM["checkpoint_revision"],
            "runtime_revision": UPSTREAM["upstream"]["commit"],
            "task_family": "FL2VA",
            "preparation": {"kind": "hand_written", "case": HAND["case_id"]},
        },
    )
    result.update(overrides)
    return result


def upstream_inputs(case_id, batch_size=1):
    case = CASES[case_id]
    result = {}
    for name, field in case["expected"]["fields"].items():
        dtype = getattr(torch, field["dtype"])
        if "data_le_hex" in field:
            np_dtype = "<f8" if field["dtype"] == "float64" else "<f4"
            values = np.frombuffer(bytes.fromhex(field["data_le_hex"]), dtype=np_dtype).copy()
            result[name] = torch.from_numpy(values).reshape(field["shape"])
        else:
            result[name] = torch.tensor(field["values"], dtype=dtype, device="cpu")
    result.update(
        batch_size=batch_size,
        num_condition_video_rows=case["expected"]["num_condition_video_rows"],
        num_condition_audio_rows=case["expected"]["num_condition_audio_rows"],
        provenance={
            "checkpoint_revision": UPSTREAM["checkpoint_revision"],
            "runtime_revision": UPSTREAM["upstream"]["commit"],
            "task_family": case["task_family"],
            "preparation": {"layout": case["layout_inputs"], "time": case["timestep_inputs"]},
        },
    )
    return result


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.contiguous().numpy().tobytes() == expected.contiguous().numpy().tobytes()


def scalar_scatter(inputs, sources, batch_size):
    """Independent row-by-row consumer, not a schema implementation helper."""
    packed = torch.empty(batch_size, len(inputs["token_tags"]), 2, dtype=torch.int64)
    for stream in ("text", "video", "audio"):
        for source_row, packed_row in enumerate(inputs[f"{stream}_indices"].tolist()):
            packed[:, packed_row] = sources[stream][:, source_row]
    return packed


def hand_row_differences(layout, baseline):
    """Locate differences from sealed hand facts, without candidate permutations."""
    expected_rows = baseline["after"]
    source_by_id = {
        row_id: (stream, source_row)
        for stream, row_ids in baseline["source_row_ids"].items()
        for source_row, row_id in enumerate(row_ids)
    }
    differences = []
    for packed_row, row_id in enumerate(expected_rows["row_ids"]):
        stream, source_row = source_by_id[row_id]
        time_row = expected_rows["timestep_indices"][packed_row]
        expected = {
            "source_stream": stream,
            "source_index": source_row,
            "modality": expected_rows["token_tags"][packed_row],
            "timestep_table_row": time_row,
            "timestep": baseline["timestep"][time_row],
            "position": tuple(expected_rows["position_ids"][packed_row]),
            "output_stream": None if stream == "text" else stream,
            "output_selection_index": None if stream == "text" else source_row,
            "role": baseline["source_roles"][stream][source_row],
        }
        actual = layout.row_identity(packed_row)
        for field, value in expected.items():
            if actual[field] != value:
                differences.append((packed_row, row_id, field, value, actual[field]))
    return differences


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_upstream_fields_are_preserved_and_each_row_is_addressable(case_id):
    inputs = upstream_inputs(case_id)
    layout = H3PackedLayout(**inputs)
    assert layout.sequence_length == CASES[case_id]["expected"]["sequence_length"]
    for name in PRIMARY:
        assert_bits(getattr(layout, name), inputs[name])
    for stream in ("text", "video", "audio"):
        for source_row, packed_row in enumerate(inputs[f"{stream}_indices"].tolist()):
            identity = layout.row_identity(packed_row)
            assert identity["source_stream"] == stream
            assert identity["source_index"] == source_row
            assert identity["modality"] == int(inputs["token_tags"][packed_row])
            assert identity["position"] == tuple(inputs["position_ids"][packed_row].tolist())
            assert identity["output_selection_index"] == (None if stream == "text" else source_row)


def test_hand_written_permutation_matches_every_independent_expected_field():
    layout = H3PackedLayout(**hand_inputs())
    before = layout.to_dict()
    moved = layout.permute_rows(torch.tensor(HAND["perm_new_to_old"]))
    assert isinstance(moved, H3PackedLayout) and moved is not layout
    for name in FIELDS:
        values = HAND["timestep"] if name == "timestep" else HAND["after"][name]
        expected = torch.tensor(values, dtype=getattr(torch, HAND["dtypes"][name]))
        assert_bits(getattr(moved, name), expected)
    assert layout.to_dict() == before
    assert moved.row_identity(0) == {
        "batch_index": 0,
        "packed_row": 0,
        "source_stream": "video",
        "source_index": 1,
        "modality": 0,
        "timestep_table_row": 0,
        "timestep": 0.25,
        "position": (1.0, 0.0, 0.5),
        "output_stream": "video",
        "output_selection_index": 1,
        "role": "target",
    }
    assert moved.row_identity(2)["role"] == "condition"


@pytest.mark.parametrize("case_id", sorted(CASES))
@pytest.mark.parametrize("batch_size", [1, 3])
def test_random_permutations_preserve_real_cpu_consumers_and_source_output_order(
    case_id, batch_size
):
    inputs = upstream_inputs(case_id, batch_size)
    layout = H3PackedLayout(**inputs)
    size = layout.sequence_length
    generator = torch.Generator().manual_seed(420)
    perm = torch.randperm(size, generator=generator)
    moved = layout.permute_rows(perm)
    for name in ("position_ids", "token_tags", "timestep_indices"):
        assert_bits(getattr(moved, name), inputs[name].index_select(0, perm))
    assert_bits(moved.timestep, inputs["timestep"])
    sources = {}
    for stream_id, stream in enumerate(("text", "video", "audio")):
        count = len(inputs[f"{stream}_indices"])
        values = torch.arange(batch_size * count * 2).reshape(batch_size, count, 2)
        sources[stream] = values + stream_id * 10000
    original = scalar_scatter(inputs, sources, batch_size)
    moved_fields = {name: getattr(moved, name) for name in PRIMARY}
    actual = scalar_scatter(moved_fields, sources, batch_size)
    assert_bits(actual, original.index_select(1, perm))
    for stream in ("video", "audio"):
        assert_bits(actual.index_select(1, getattr(moved, f"{stream}_indices")), sources[stream])
    tables = tuple(
        torch.arange(inputs["timestep"].numel() * 3 * 4).reshape(-1, 4).float() + part * 100
        for part in range(6)
    )
    consumed = provider_adaln_row_gather(tables, moved.timestep_indices, moved.token_tags)
    for part, actual_rows in enumerate(consumed):
        expected = torch.stack(
            [
                tables[part][
                    int(inputs["timestep_indices"][int(old)]) * 3
                    + int(inputs["token_tags"][int(old)])
                ]
                for old in perm
            ]
        )
        assert_bits(actual_rows, expected)
    inverse = torch.argsort(perm)
    assert moved.permute_rows(inverse).to_dict() == layout.to_dict()
    second = torch.randperm(size, generator=generator)
    assert moved.permute_rows(second).to_dict() == layout.permute_rows(perm[second]).to_dict()


def test_batch_placement_and_unrelated_sample_changes():
    layout = H3PackedLayout(**hand_inputs(batch_size=3))
    baseline = layout.row_identity(2, batch_index=0)
    placed = layout.row_identity(2, batch_index=2)
    assert placed == {**baseline, "batch_index": 2}
    sources = {
        name: torch.tensor(rows).reshape(1, -1, 1).repeat(3, 1, 2)
        for name, rows in HAND["source_payloads"].items()
    }
    before = scalar_scatter(hand_inputs(), sources, 3)
    sources["video"][1].fill_(-999)
    after = scalar_scatter(hand_inputs(), sources, 3)
    assert_bits(before[0], after[0])
    assert_bits(before[2], after[2])
    assert not torch.equal(before[1], after[1])


def test_input_getter_snapshot_and_provenance_mutations_do_not_change_layout():
    inputs = hand_inputs()
    inputs["provenance"]["preparation"]["steps"] = ([1],)
    layout = H3PackedLayout(**inputs)
    baseline = layout.to_dict()
    assert baseline["provenance"]["preparation"]["steps"] == [[1]]
    for name in PRIMARY:
        inputs[name].zero_()
        getattr(layout, name).zero_()
    inputs["provenance"]["preparation"]["kind"] = "changed"
    inputs["provenance"]["preparation"]["steps"][0].append(2)
    layout.provenance["preparation"]["kind"] = "changed"
    snapshot = layout.to_dict()
    snapshot["provenance"]["preparation"]["kind"] = "changed"
    snapshot["provenance"]["preparation"]["steps"][0].append(3)
    snapshot["tensors"]["token_tags"]["shape"][0] = 100
    assert layout.to_dict() == baseline


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_exact_json_round_trip(case_id):
    layout = H3PackedLayout(**upstream_inputs(case_id))
    restored = H3PackedLayout.from_dict(json.loads(json.dumps(layout.to_dict())))
    assert restored.to_dict() == layout.to_dict()
    for name in PRIMARY:
        assert_bits(getattr(restored, name), getattr(layout, name))


def test_signed_zero_subnormal_and_noncontiguous_metadata_are_preserved():
    inputs = hand_inputs()
    inputs["position_ids"][0] = torch.tensor([-0.0, 5e-324, -5e-324], dtype=torch.float64)
    for name in PRIMARY:
        value = inputs[name]
        buffer = torch.empty(*value.shape[:-1], value.shape[-1] * 2, dtype=value.dtype)
        view = buffer[..., ::2]
        view.copy_(value)
        if view.numel() > 1:
            assert not view.is_contiguous()
        inputs[name] = view
    layout = H3PackedLayout(**inputs)
    restored = H3PackedLayout.from_dict(layout.to_dict())
    for name in PRIMARY:
        assert_bits(getattr(restored, name), inputs[name])


def test_unsorted_distinct_timestep_table_and_empty_source_stream():
    inputs = hand_inputs(
        timestep=torch.tensor([0.5, 0.25, 1.0]), timestep_indices=torch.tensor([1, 0, 2, 1])
    )
    layout = H3PackedLayout(**inputs)
    assert layout.row_identity(0)["timestep"] == 0.25
    assert layout.timestep.tolist() == [0.5, 0.25, 1.0]
    inputs = hand_inputs(
        text_indices=torch.empty(0, dtype=torch.int64),
        video_indices=torch.tensor([0, 2, 3]),
        token_tags=torch.tensor([0, 2, 0, 0]),
    )
    layout = H3PackedLayout(**inputs)
    assert layout.source_lengths == {"text": 0, "video": 3, "audio": 1}
    assert H3PackedLayout.from_dict(layout.to_dict()).source_lengths == layout.source_lengths


def test_snapshot_preserves_raw_subnormal_bits_with_flush_to_zero_enabled():
    script = """
import json, sys, torch
from rl_engine.models.minimax_h3.packed_layout import H3PackedLayout
hand = json.loads(sys.argv[1])
fields = {
    name: torch.tensor(hand['timestep'] if name == 'timestep' else hand['before'][name],
                       dtype=getattr(torch, dtype))
    for name, dtype in hand['dtypes'].items() if name != 'adaln_indices'
}
fields['position_ids'].view(torch.int64)[0] = torch.tensor([-2**63, 1, -2**63 + 1])
before = fields['position_ids'].view(torch.int64).clone()
torch.set_flush_denormal(True)
layout = H3PackedLayout(**fields, provenance={
    'checkpoint_revision': '0' * 40, 'runtime_revision': '1' * 40, 'task_family': 'FL2VA'})
restored = H3PackedLayout.from_dict(layout.to_dict())
assert torch.equal(restored.position_ids.view(torch.int64), before)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(HAND)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_layout_preserves_provenance_with_flush_to_zero_enabled():
    snapshot = H3PackedLayout(**hand_inputs()).to_dict()
    script = """
import json
import struct
import sys

import torch

from rl_engine.models.minimax_h3.packed_layout import H3PackedLayout

torch.set_flush_denormal(False)
snapshot = json.loads(sys.argv[1])
scale_bits = struct.pack("<Q", 1)
snapshot["provenance"]["preparation"]["scale"] = struct.unpack("<d", scale_bits)[0]
layout = H3PackedLayout.from_dict(snapshot)
if not torch.set_flush_denormal(True):
    sys.exit(77)
moved = layout.permute_rows(torch.arange(layout.sequence_length))
exported = layout.to_dict()
restored = H3PackedLayout.from_dict(exported)
for path, context in (
    ("getter", layout.provenance),
    ("to_dict", exported["provenance"]),
    ("from_dict", restored.provenance),
    ("permute_rows", moved.provenance),
):
    actual_bits = struct.pack("<d", context["preparation"]["scale"])
    assert actual_bits == scale_bits, (path, actual_bits.hex())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(snapshot)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if result.returncode == 77:
        pytest.skip("CPU flush-to-zero mode unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


def test_timestep_validation_and_identity_are_independent_of_ftz():
    script = """
import math, random, struct, torch
from rl_engine.models.minimax_h3.packed_layout import H3PackedLayout, H3PackedLayoutError

def inputs(bits):
    size = len(bits)
    return dict(
        text_indices=torch.empty(0, dtype=torch.int64),
        audio_indices=torch.empty(0, dtype=torch.int64),
        video_indices=torch.arange(size), token_tags=torch.zeros(size, dtype=torch.int64),
        timestep=torch.tensor(bits, dtype=torch.int32).view(torch.float32),
        timestep_indices=torch.arange(size), position_ids=torch.zeros(size, 3, dtype=torch.float64),
        provenance={'checkpoint_revision': '0' * 40, 'runtime_revision': '1' * 40,
                    'task_family': 'FL2VA'},
    )

torch.set_flush_denormal(False)
bits = list(dict.fromkeys([0, 1, 0x007fffff, 0x00800000, 0x3f800000]
                         + random.Random(420).sample(range(0x3f800001), 256)))
positive = inputs(bits)
positive['position_ids'].view(torch.int64)[0] = torch.tensor([-2**63, 1, -2**63 + 1])
expected_position = struct.pack('<qqq', -2**63, 1, -2**63 + 1)
expected_times = [struct.unpack('<f', struct.pack('<I', value))[0] for value in bits]
assert expected_times[:5] == [0.0, math.ldexp(1.0, -149), math.ldexp(2**23 - 1, -149),
                            math.ldexp(1.0, -126), 1.0]
existing = H3PackedLayout(**positive)
snapshot = existing.to_dict()
for flush in (False, True):
    torch.set_flush_denormal(flush)
    for layout in (existing, H3PackedLayout(**positive), H3PackedLayout.from_dict(snapshot)):
        assert layout.to_dict() == snapshot
        assert struct.pack('<ddd', *layout.row_identity(0)['position']) == expected_position
        for row, expected in enumerate(expected_times):
            actual = layout.row_identity(row)['timestep']
            assert struct.pack('<d', actual) == struct.pack('<d', expected), (flush, row)
    for bits in ([-2147483647], [0, -2147483648], [0, 0], [1, 1],
                 [0x3f800001], [0x7f800000], [0x7fc00000]):
        try:
            H3PackedLayout(**inputs(bits))
        except H3PackedLayoutError:
            pass
        else:
            raise AssertionError(('invalid time table accepted', flush, bits))
    negative_zero = H3PackedLayout(**inputs([-2147483648]))
    assert struct.pack('<d', negative_zero.row_identity(0)['timestep']) == struct.pack('<d', -0.0)
    for duplicate_bits, value in (([1, 0, 1], '1.401298464324817e-45'),
                                  ([0, -2147483648], '-0.0')):
        try:
            H3PackedLayout(**inputs(duplicate_bits))
        except H3PackedLayoutError as error:
            row = len(duplicate_bits) - 1
            assert str(error) == (
                f'timestep: expected distinct values; timestep[{row}] duplicates '
                f'timestep[0] (value {value})'
            ), (flush, str(error))
        else:
            raise AssertionError(('duplicate accepted', flush, duplicate_bits))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("bits", "duplicate_row", "first_row", "value"),
    [
        ([0x3E800000, 0x3F000000, 0x3E800000], 2, 0, "0.25"),
        ([0x3E800000, 0x3F000000, 0x3F000000], 2, 1, "0.5"),
        ([0, -(1 << 31), 0x3F800000], 1, 0, "-0.0"),
        ([-(1 << 31), 0, 0x3F800000], 1, 0, "0.0"),
        ([1, 0, 1], 2, 0, "1.401298464324817e-45"),
    ],
)
def test_duplicate_timestep_reports_value_and_both_table_rows(
    bits, duplicate_row, first_row, value
):
    times = torch.tensor(bits, dtype=torch.int32).view(torch.float32)
    with pytest.raises(H3PackedLayoutError) as caught:
        H3PackedLayout(**hand_inputs(timestep=times))
    assert str(caught.value) == (
        f"timestep: expected distinct values; timestep[{duplicate_row}] duplicates "
        f"timestep[{first_row}] (value {value})"
    )


def test_text_stream_visual_tag_is_not_guessed_from_its_source():
    layout = H3PackedLayout(**upstream_inputs("fl2va_text_visual_tag"))
    identity = layout.row_identity(1)
    assert identity["source_stream"] == "text" and identity["modality"] == 0
    assert identity["output_stream"] is None and identity["output_selection_index"] is None


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize(
    ("stream", "source_row", "tag", "allowed"),
    [
        ("text", 0, 2, "0, 1"),
        ("audio", 0, 0, "2"),
        ("audio", 0, 1, "2"),
        ("video", 0, 1, "0"),
        ("video", 0, 2, "0"),
        ("video", 1, 2, "0"),
    ],
)
def test_source_modality_mismatch_identifies_packed_and_source_rows(
    phase, stream, source_row, tag, allowed
):
    inputs = hand_inputs(phase=phase)
    packed_row = int(inputs[f"{stream}_indices"][source_row])
    inputs["token_tags"][packed_row] = tag
    with pytest.raises(H3PackedLayoutError) as caught:
        H3PackedLayout(**inputs)
    assert str(caught.value) == (
        f"token_tags[{packed_row}]: expected one of {{{allowed}}} for "
        f"{stream}_indices[{source_row}], got {tag}"
    )


@pytest.mark.parametrize(("stream", "tag"), [("text", 2), ("video", 1), ("audio", 0)])
def test_snapshot_restore_rejects_source_modality_mismatch(stream, tag):
    inputs = hand_inputs()
    snapshot = H3PackedLayout(**inputs).to_dict()
    packed_row = int(inputs[f"{stream}_indices"][0])
    inputs["token_tags"][packed_row] = tag
    snapshot["tensors"]["token_tags"]["data_le_hex"] = (
        inputs["token_tags"].view(torch.uint8).numpy().tobytes().hex()
    )
    with pytest.raises(
        H3PackedLayoutError,
        match=rf"token_tags\[{packed_row}\].*{stream}_indices\[0\]",
    ):
        H3PackedLayout.from_dict(snapshot)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unmoved_tags", r"token_tags\[1\].*text_indices\[0\]"),
        ("stale_cross_stream_selectors", r"token_tags\[3\].*video_indices\[1\]"),
    ],
)
def test_cross_stream_stale_metadata_is_rejected(change, message):
    inputs = hand_inputs(phase="after")
    if change == "unmoved_tags":
        inputs["token_tags"] = torch.tensor(HAND["before"]["token_tags"])
    else:
        for name in ("text_indices", "video_indices", "audio_indices"):
            inputs[name] = torch.tensor(HAND["before"][name])
    with pytest.raises(H3PackedLayoutError, match=message):
        H3PackedLayout(**inputs)


@pytest.mark.parametrize(
    ("change", "first_difference", "affected_rows"),
    [
        ("unmoved_timestep_indices", (1, "T0", "timestep_table_row"), {"T0", "A0"}),
        ("unmoved_position_ids", (0, "V1", "position"), {"V1", "T0", "A0"}),
        ("wrong_legal_timestep", (0, "V1", "timestep_table_row"), {"V1"}),
        ("wrong_legal_tag", (1, "T0", "modality"), {"T0"}),
        ("stale_selectors", (0, "V1", "source_index"), {"V1", "V0"}),
    ],
)
def test_structurally_valid_semantic_mismatch_is_located_against_sealed_baseline(
    change, first_difference, affected_rows
):
    # The fixture's hand-calculated AFTER state is the oracle, not permute_rows().
    sealed = json.dumps(HAND, sort_keys=True)
    baseline = json.loads(sealed)
    inputs = hand_inputs(phase="after")
    assert hand_row_differences(H3PackedLayout(**inputs), baseline) == []
    if change.startswith("unmoved_"):
        field = change.removeprefix("unmoved_")
        inputs[field] = torch.tensor(baseline["before"][field], dtype=inputs[field].dtype)
    elif change == "wrong_legal_timestep":
        inputs["timestep_indices"][0] = 1
    elif change == "wrong_legal_tag":
        # Both text and visual tags are legal in the text source stream.
        inputs["token_tags"][1] = 0
    elif change == "stale_selectors":
        # Same video destinations and tags, but the source slots are swapped.
        inputs["video_indices"] = inputs["video_indices"].flip(0)
    else:
        raise AssertionError(f"unknown probe {change}")

    # Each probe is still structurally valid. Only the sealed baseline reveals intent.
    changed = H3PackedLayout(**inputs)
    differences = hand_row_differences(changed, baseline)
    assert differences[0][:3] == first_difference
    assert {row_id for _, row_id, _, _, _ in differences} == affected_rows
    assert json.dumps(baseline, sort_keys=True) == sealed

    if change in (
        "unmoved_timestep_indices",
        "wrong_legal_tag",
        "wrong_legal_timestep",
    ):
        tables = tuple(torch.arange(9).reshape(-1, 1).float() + part * 100 for part in range(6))
        actual = provider_adaln_row_gather(tables, changed.timestep_indices, changed.token_tags)
        expected_indices = torch.tensor(baseline["after"]["adaln_indices"])
        expected = [table.index_select(0, expected_indices) for table in tables]
        row = first_difference[0]
        for actual_rows, expected_rows in zip(actual, expected, strict=True):
            assert not torch.equal(actual_rows[row], expected_rows[row])


@pytest.mark.parametrize("stream", ["video", "audio"])
def test_stale_output_selectors_change_the_expected_source_slots(stream):
    # Use independently sealed fields and unique payloads in each source stream.
    inputs = upstream_inputs("fl2va_first")
    sources = {
        name: torch.arange(inputs[f"{name}_indices"].numel() * 2).reshape(1, -1, 2)
        + stream_id * 1000
        for stream_id, name in enumerate(("text", "video", "audio"))
    }
    packed = scalar_scatter(inputs, sources, batch_size=1)
    indices = inputs[f"{stream}_indices"].clone()
    indices[[0, -1]] = indices[[-1, 0]]
    inputs[f"{stream}_indices"] = indices
    changed = H3PackedLayout(**inputs)
    selected = packed.index_select(1, getattr(changed, f"{stream}_indices"))[0]
    expected = sources[stream][0]
    assert (selected != expected).any(dim=1).nonzero().flatten().tolist() == [0, len(expected) - 1]
    assert torch.equal(selected[0], expected[-1]) and torch.equal(selected[-1], expected[0])


def test_mixed_variant_is_an_identity_mismatch_even_when_tensor_fields_match():
    inputs = upstream_inputs("fl2va_first")
    identity_keys = ("checkpoint_revision", "runtime_revision", "task_family")
    sealed = json.dumps({key: inputs["provenance"][key] for key in identity_keys}, sort_keys=True)
    baseline_identity = json.loads(sealed)
    # Combine FL2VA tensors with the source declaration from a Ref2VA case.
    inputs["provenance"] = upstream_inputs("ref2va_image")["provenance"]
    mixed = H3PackedLayout(**inputs)
    differences = {
        f"provenance.{key}": (expected, mixed.provenance[key])
        for key, expected in baseline_identity.items()
        if mixed.provenance[key] != expected
    }
    assert differences == {"provenance.task_family": ("FL2VA", "Ref2VA")}
    for field in PRIMARY:
        assert_bits(getattr(mixed, field), inputs[field])


def test_cpu_metadata_is_independent_of_ambient_default_device():
    inputs = hand_inputs()
    perm = torch.tensor(HAND["perm_new_to_old"])
    with torch.device("meta"):
        layout = H3PackedLayout(**inputs).permute_rows(perm)
        restored = H3PackedLayout.from_dict(layout.to_dict())
    assert restored.video_indices.device.type == "cpu"
    assert restored.video_indices.tolist() == [2, 0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("token_tags", [1, 2, 3, 0], r"token_tags\[2\]"),
        ("token_tags", [1, 2, -1, 0], r"token_tags\[2\]"),
        ("token_tags", [1, 2], r"token_tags: expected shape"),
        ("timestep_indices", [0, 1, 3, 0], r"timestep_indices\[2\]"),
        ("video_indices", [2, 4], r"video_indices\[1\]"),
        ("video_indices", [2, -1], r"video_indices\[1\]"),
        ("video_indices", [2, 2], r"video_indices\[1\].*already owned"),
        ("video_indices", [0, 3], r"video_indices\[0\].*already owned by text_indices"),
        ("video_indices", [2], r"source_indices: packed row 3 has no source"),
        ("timestep", [0.25, 0.25, 1.0], r"timestep: expected distinct"),
        ("timestep", [0.25, 0.5, 1.5], r"timestep\[2\]"),
        ("timestep", [0.25, float("nan"), 1.0], r"timestep\[1\]"),
    ],
)
def test_invalid_values_identify_the_field_and_position(field, value, message):
    inputs = hand_inputs()
    inputs[field] = torch.tensor(value, dtype=inputs[field].dtype)
    with pytest.raises(H3PackedLayoutError, match=message):
        H3PackedLayout(**inputs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 0, "batch_size"),
        ("batch_size", True, "batch_size"),
        ("num_condition_video_rows", 3, "num_condition_video_rows"),
        ("num_condition_audio_rows", -1, "num_condition_audio_rows"),
        (
            "token_tags",
            torch.tensor([1, 2, 0, 0], dtype=torch.int32),
            "token_tags: expected torch.int64",
        ),
        ("token_tags", [1, 2, 0, 0], "token_tags: expected a torch.Tensor"),
        ("token_tags", torch.tensor([[1, 2, 0, 0]]), "token_tags: expected a 1-D"),
        ("token_tags", torch.empty(4, dtype=torch.int64, device="meta"), "expected CPU metadata"),
        ("position_ids", torch.zeros(4, 3), "position_ids: expected torch.float64"),
        ("position_ids", torch.zeros(12, dtype=torch.float64), "position_ids: expected shape"),
        ("position_ids", torch.zeros(1, 4, 3, dtype=torch.float64), "position_ids: expected shape"),
        ("position_ids", torch.zeros(4, 2, dtype=torch.float64), "position_ids: expected shape"),
        ("position_ids", torch.zeros(0, 3, dtype=torch.float64), "position_ids: expected shape"),
        (
            "position_ids",
            torch.full((4, 3), float("inf"), dtype=torch.float64),
            r"position_ids\[0,0\]",
        ),
        (
            "position_ids",
            torch.zeros(4, 3, dtype=torch.float64, requires_grad=True),
            "without gradients",
        ),
        ("timestep", torch.empty(0), "timestep: expected at least one"),
    ],
)
def test_unsupported_shapes_types_and_context_are_explicit_errors(field, value, message):
    with pytest.raises(H3PackedLayoutError, match=message):
        H3PackedLayout(**hand_inputs(**{field: value}))


@pytest.mark.skipif(not hasattr(torch.Tensor, "refine_names"), reason="named tensors unavailable")
@pytest.mark.filterwarnings("ignore:Named tensors and all their associated APIs:UserWarning")
@pytest.mark.parametrize("field", PRIMARY)
def test_named_metadata_is_rejected_with_field_error(field):
    inputs = hand_inputs()
    names = ("row", "axis") if field == "position_ids" else ("row",)
    inputs[field] = inputs[field].refine_names(*names)
    with pytest.raises(H3PackedLayoutError, match=rf"^{field}: expected unnamed metadata"):
        H3PackedLayout(**inputs)
    assert inputs[field].names == names


@pytest.mark.skipif(not hasattr(torch.Tensor, "refine_names"), reason="named tensors unavailable")
@pytest.mark.filterwarnings("ignore:Named tensors and all their associated APIs:UserWarning")
def test_partially_named_metadata_is_rejected():
    inputs = hand_inputs()
    inputs["position_ids"] = inputs["position_ids"].refine_names(None, "axis")
    with pytest.raises(H3PackedLayoutError, match=r"^position_ids: expected unnamed metadata"):
        H3PackedLayout(**inputs)
    assert inputs["position_ids"].names == (None, "axis")


@pytest.mark.parametrize(
    "change",
    [
        {"task_family": "unknown"},
        {"runtime_revision": "main"},
        {"checkpoint_revision": None},
        {"extra": float("nan")},
        {"extra": {1: "non-string JSON key"}},
    ],
)
def test_provenance_requires_explicit_supported_identity(change):
    inputs = hand_inputs()
    inputs["provenance"].update(change)
    with pytest.raises(H3PackedLayoutError, match="provenance"):
        H3PackedLayout(**inputs)


def test_provenance_rejects_a_circular_reference():
    inputs = hand_inputs()
    preparation = inputs["provenance"]["preparation"]
    preparation["nested"] = [preparation]
    with pytest.raises(H3PackedLayoutError, match="provenance: circular reference"):
        H3PackedLayout(**inputs)


def test_provenance_validates_the_text_of_a_task_family():
    class TaskAlias(str):
        def __eq__(self, other):
            return True

    inputs = hand_inputs()
    inputs["provenance"]["task_family"] = TaskAlias("unsupported")
    with pytest.raises(H3PackedLayoutError, match=r"^provenance\.task_family:"):
        H3PackedLayout(**inputs)


@pytest.mark.parametrize(
    "perm",
    [
        torch.tensor([0, 0, 2, 3]),
        torch.tensor([0, 1, 2, 4]),
        torch.tensor([0, 1, 2]),
        torch.arange(4, dtype=torch.int32),
    ],
)
def test_illegal_permutation_is_rejected_without_changing_original(perm):
    layout = H3PackedLayout(**hand_inputs())
    original = layout.to_dict()
    with pytest.raises(H3PackedLayoutError, match="perm"):
        layout.permute_rows(perm)
    assert layout.to_dict() == original


@pytest.mark.skipif(not hasattr(torch.Tensor, "refine_names"), reason="named tensors unavailable")
@pytest.mark.filterwarnings("ignore:Named tensors and all their associated APIs:UserWarning")
def test_named_permutation_is_rejected_without_changing_original():
    layout = H3PackedLayout(**hand_inputs())
    original = layout.to_dict()
    perm = torch.tensor(HAND["perm_new_to_old"]).refine_names("row")
    with pytest.raises(H3PackedLayoutError, match=r"^perm: expected unnamed metadata"):
        layout.permute_rows(perm)
    assert perm.names == ("row",)
    assert layout.to_dict() == original


@pytest.mark.parametrize(("row", "batch"), [(-1, 0), (4, 0), (0, 1), (True, 0)])
def test_identity_address_errors(row, batch):
    with pytest.raises(H3PackedLayoutError, match="packed_row|batch_index"):
        H3PackedLayout(**hand_inputs()).row_identity(row, batch_index=batch)


def test_snapshot_rejects_bytes_for_hex_data():
    snapshot = H3PackedLayout(**hand_inputs()).to_dict()
    field = snapshot["tensors"]["position_ids"]
    field["data_le_hex"] = field["data_le_hex"].encode("ascii")
    with pytest.raises(
        H3PackedLayoutError,
        match=r"^snapshot\.tensors\.position_ids\.data_le_hex: expected a hex string$",
    ):
        H3PackedLayout.from_dict(snapshot)


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "profile",
        "extra",
        "missing",
        "dtype",
        "hex",
        "length",
        "shape",
        "huge_shape",
        "extra_tensor",
    ],
)
def test_invalid_snapshot_is_rejected(change):
    snapshot = copy.deepcopy(H3PackedLayout(**hand_inputs()).to_dict())
    field = snapshot["tensors"]["position_ids"]
    if change in ("version", "profile"):
        snapshot["schema_version" if change == "version" else "profile"] = "unsupported"
    elif change == "extra":
        snapshot["extra"] = 1
    elif change == "missing":
        del snapshot["tensors"]["token_tags"]
    elif change == "extra_tensor":
        field["extra"] = 1
    elif change == "dtype":
        field["dtype"] = "float32"
    elif change == "hex":
        field["data_le_hex"] = "not hex"
    elif change == "length":
        field["data_le_hex"] = "00"
    else:
        field["shape"] = [True, 3] if change == "shape" else [1 << 100, 3]
    with pytest.raises(H3PackedLayoutError, match="snapshot"):
        H3PackedLayout.from_dict(snapshot)
