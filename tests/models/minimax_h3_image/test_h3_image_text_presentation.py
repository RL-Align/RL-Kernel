# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #738 Phase B ``image_text_presentation_contract``: T2I and edit presentations.

Layers of tests, each skipping when its inputs are missing:

* self-contained (torch only): token construction, tags, truncation, reference order and
  fail-closed cases with a stand-in tokenizer that maps special-token text to the pinned ids;
* tiny Qwen3-VL (transformers with Qwen3-VL): the layer-50 feature of a presentation and the
  I9/I19 probes on a random 51-layer model;
* released files (``$RL_KERNEL_H3_IMAGE_DIR``, a download of ``MiniMaxAI/MiniMax-H3-Image``
  with ``tokenizer/`` and ``processor/``): file pins, the sealed CPU digests of every fixture
  and, with Diffusers 875373ac importable, bitwise parity with the pinned Diffusers steps on
  the tiny model;
* released weights (CUDA with about 70 GB free; ``text_encoder/`` under
  ``$RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT``, default ``$RL_KERNEL_H3_IMAGE_DIR``): every sealed
  fixture through the pinned Diffusers step and the contract, bitwise, and against the sealed
  feature digest when this GPU arch and library versions have one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from rl_engine.models.minimax_h3 import qwen3vl_text as q
from rl_engine.models.minimax_h3.trajectory import tensor_digest
from rl_engine.models.minimax_h3_image import fingerprint as image_fp
from rl_engine.models.minimax_h3_image import text_presentation as tp

IMAGE_DIR_ENV = "RL_KERNEL_H3_IMAGE_DIR"
TEXT_ENCODER_ROOT_ENV = "RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT"
IDS = q.vision_token_ids()
START, END, PAD = IDS[q.VISION_START], IDS[q.VISION_END], IDS[q.IMAGE_PAD]


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


HAS_DIFFUSERS_H3_IMAGE = _has_module("diffusers.modular_pipelines.minimax_h3_image")
needs_diffusers = pytest.mark.skipif(
    not HAS_DIFFUSERS_H3_IMAGE, reason="needs diffusers with MiniMax-H3-Image (875373ac)"
)


def _has_qwen3vl() -> bool:
    if not _has_module("transformers"):
        return False
    import transformers

    return hasattr(transformers, "Qwen3VLForConditionalGeneration")


needs_qwen3vl = pytest.mark.skipif(not _has_qwen3vl(), reason="transformers without Qwen3-VL")


# --- stand-in tokenizer -------------------------------------------------------------------------


class CharTokenizer:
    """One id per character, special-token text mapped to its pinned id (as the release does)."""

    def __init__(self) -> None:
        self.special = q.special_token_ids()

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            for name, token in self.special.items():
                if text.startswith(name, index):
                    ids.append(token)
                    index += len(name)
                    break
            else:
                ids.append(1000 + ord(text[index]) % 100_000)
                index += 1
        return {"input_ids": ids}


TOK = CharTokenizer()


def _ids(text: str) -> list[int]:
    return TOK(text)["input_ids"]


GRIDS = [(1, 4, 6), (1, 6, 4), (1, 2, 2)]


# --- constants ----------------------------------------------------------------------------------


def test_constants_follow_the_pins() -> None:
    assert (tp.IMAGE_TAG, tp.TEXT_TAG) == (0, 1)
    # The image tag is the H3 video tag: both select AdaLN row 0.
    assert (tp.IMAGE_TAG, tp.TEXT_TAG) == (q.VIDEO_TAG, q.TEXT_TAG)
    assert tp.MAX_SEQUENCE_LENGTH == image_fp.MAX_SEQUENCE_LENGTH == 4096
    assert tp.TEXT_ENCODER_LAYER == image_fp.TEXT_ENCODER_LAYER == q.TEXT_ENCODER_LAYER == 50
    assert tp.FEATURE_WIDTH == 5120
    assert tp.DIFFUSERS_COMMIT == image_fp.DIFFUSERS_COMMIT
    json.dumps(tp.contract_manifest())


# --- text2image ---------------------------------------------------------------------------------


def test_t2i_is_the_prompt_verbatim_all_text() -> None:
    prompt = "A red fox, 雪地. <Picture 1> stays text."
    p = tp.present_text2image(TOK, prompt)
    assert p.mode == tp.TEXT2IMAGE
    assert list(p.token_ids) == _ids(prompt)
    assert set(p.token_tags) == {tp.TEXT_TAG}
    assert p.segments == (tp.Segment(tp.PROMPT, 0, len(prompt)),)
    assert p.image_grid_thw == ()
    assert not p.prompt_truncated
    tp.check_tags(p)
    assert p.tags_tensor().dtype == torch.long


@pytest.mark.parametrize("length", [1, 4095, 4096, 4097, 9000])
def test_t2i_truncates_to_4096_tokens(length: int) -> None:
    prompt = "".join(chr(0x4E00 + i % 500) for i in range(length))
    p = tp.present_text2image(TOK, prompt)
    assert p.num_rows == min(length, 4096)
    assert list(p.token_ids) == _ids(prompt)[:4096]
    assert p.prompt_num_tokens == length
    assert p.prompt_truncated == (length > 4096)


def test_t2i_digest_depends_on_every_token() -> None:
    a = tp.present_text2image(TOK, "a cat on a mat")
    b = tp.present_text2image(TOK, "a cat on a hat")
    assert a.summary()["token_ids_sha256"] != b.summary()["token_ids_sha256"]
    assert a.summary()["prompt_sha256"] != b.summary()["prompt_sha256"]


# --- edit ---------------------------------------------------------------------------------------


def test_edit_layout_label_block_per_reference_then_prompt() -> None:
    prompt = "Swap <Picture 1> and <Picture 2>, keep <Picture 3>."
    p = tp.present_edit(TOK, prompt, torch.tensor(GRIDS))
    expected: list[int] = []
    tags: list[int] = []
    for index, (t, h, w) in enumerate(GRIDS):
        label = _ids(f"Picture {index + 1}: ")
        block = [START] + [PAD] * (t * h * w // 4) + [END]
        expected += label + block
        tags += [1] * len(label) + [0] * len(block)
    expected += _ids(prompt)
    tags += [1] * len(_ids(prompt))
    assert list(p.token_ids) == expected
    assert list(p.token_tags) == tags
    assert [(s.kind, s.reference) for s in p.segments] == [
        ("label", 0),
        ("vision", 0),
        ("label", 1),
        ("vision", 1),
        ("label", 2),
        ("vision", 2),
        ("prompt", None),
    ]
    assert p.image_grid_thw == tuple(GRIDS)
    tp.check_tags(p)


def test_edit_labels_are_one_based_without_angle_brackets() -> None:
    p = tp.present_edit(TOK, "x", [(1, 2, 2)] * 11)
    labels = [s for s in p.segments if s.kind == tp.LABEL]
    texts = [
        "".join(chr(i - 1000) for i in p.token_ids[s.start : s.start + s.length]) for s in labels
    ]
    assert texts == [f"Picture {i}: " for i in range(1, 12)]


def test_edit_truncates_prompt_and_labels_separately() -> None:
    prompt = "z" * 5000
    p = tp.present_edit(TOK, prompt, GRIDS)
    assert p.segments[-1].length == 4096
    assert p.prompt_truncated
    # The whole presentation is not cut: labels and vision blocks come on top.
    assert p.num_rows == p.segments[-1].start + 4096 > 4096
    short = tp.present_edit(TOK, prompt, GRIDS, max_sequence_length=3)
    assert [s.length for s in short.segments if s.kind == tp.LABEL] == [3, 3, 3]


def test_reference_order_is_part_of_the_presentation() -> None:
    """RFC #738 I9: swapping two references of different geometry changes the token ids."""

    a = tp.present_edit(TOK, "p", GRIDS)
    b = tp.present_edit(TOK, "p", [GRIDS[2], GRIDS[1], GRIDS[0]])
    assert a.token_ids != b.token_ids
    assert a.summary()["token_ids_sha256"] != b.summary()["token_ids_sha256"]
    # 4x6 and 6x4 hold the same number of pads: the ids cannot show that swap, the grids do
    # (check_vision_inputs), and so do the pixels.
    c = tp.present_edit(TOK, "p", [GRIDS[1], GRIDS[0], GRIDS[2]])
    assert c.token_ids == a.token_ids
    assert c.image_grid_thw != a.image_grid_thw
    assert c.summary() != a.summary()


@pytest.mark.parametrize(
    "prompt, grids, match",
    [
        ("", GRIDS, "non-empty string"),
        (None, GRIDS, "non-empty string"),
        (["a", "b"], GRIDS, "non-empty string"),
        ("look at <|image_pad|>", GRIDS, "special tokens"),
        ("<|vision_start|>x", GRIDS, "special tokens"),
        ("ok", [], "at least one reference"),
        ("ok", [(2, 4, 4)], "t=2"),
        ("ok", [(1, 3, 4)], "merged patches"),
        ("ok", [(1, 0, 4)], "merged patches"),
        ("ok", [(1, 4)], "three ints"),
        ("ok", torch.tensor([[1.0, 4.0, 4.0]]), "integer"),
        ("ok", torch.tensor([1, 4, 4]), "integer"),
    ],
)
def test_edit_fails_closed(prompt: Any, grids: Any, match: str) -> None:
    with pytest.raises(tp.PresentationError, match=match):
        tp.present_edit(TOK, prompt, grids)


@pytest.mark.parametrize("prompt", ["", None, "a <|image_pad|> b", "<|endoftext|>"])
def test_t2i_fails_closed(prompt: Any) -> None:
    with pytest.raises(tp.PresentationError):
        tp.present_text2image(TOK, prompt)


@pytest.mark.parametrize("limit", [0, -1, 2.5])
def test_bad_max_sequence_length(limit: Any) -> None:
    with pytest.raises(ValueError):
        tp.present_text2image(TOK, "a", max_sequence_length=limit)


# --- tags (I20) and vision inputs ---------------------------------------------------------------


def _with_tags(p: tp.TextPresentation, tags: list[int]) -> tp.TextPresentation:
    return tp.TextPresentation(
        mode=p.mode,
        token_ids=p.token_ids,
        token_tags=tuple(tags),
        segments=p.segments,
        image_grid_thw=p.image_grid_thw,
        merge_size=p.merge_size,
        prompt_sha256=p.prompt_sha256,
        prompt_num_tokens=p.prompt_num_tokens,
    )


@pytest.mark.parametrize("which", ["text_to_image", "vision_to_text", "start_bracket", "bad_value"])
def test_flipped_tag_is_caught(which: str) -> None:
    """RFC #738 I20: one flipped text or vision tag is rejected before the AdaLN gather."""

    p = tp.present_edit(TOK, "prompt", GRIDS)
    tags = list(p.token_tags)
    vision = next(s for s in p.segments if s.kind == tp.VISION)
    row = {
        "text_to_image": p.segments[-1].start,
        "vision_to_text": vision.start + 1,
        "start_bracket": vision.start,
        "bad_value": 0,
    }[which]
    tags[row] = 2 if which == "bad_value" else 1 - tags[row]
    with pytest.raises(tp.PresentationError):
        tp.check_tags(_with_tags(p, tags))
    with pytest.raises(tp.PresentationError, match="tags for"):
        tp.check_tags(_with_tags(p, list(p.token_tags)[:-1]))


def _vision(grids: list[tuple[int, int, int]], seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    patches = sum(t * h * w for t, h, w in grids)
    return {
        "pixel_values": torch.randn(patches, 3 * 2 * 16 * 16, generator=generator),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long),
    }


def test_check_vision_inputs() -> None:
    p = tp.present_edit(TOK, "p", GRIDS)
    tp.check_vision_inputs(p, _vision(GRIDS))
    with pytest.raises(tp.PresentationError, match="reference order"):
        tp.check_vision_inputs(p, _vision([GRIDS[1], GRIDS[0], GRIDS[2]]))
    short = _vision(GRIDS)
    short["pixel_values"] = short["pixel_values"][:-1]
    with pytest.raises(tp.PresentationError, match="patch rows"):
        tp.check_vision_inputs(p, short)
    extra = {**_vision(GRIDS), "pixel_values_videos": torch.zeros(1)}
    with pytest.raises(tp.PresentationError, match="takes pixel_values"):
        tp.check_vision_inputs(p, extra)
    with pytest.raises(tp.PresentationError, match="takes pixel_values"):
        tp.check_vision_inputs(p, None)
    t2i = tp.present_text2image(TOK, "p")
    tp.check_vision_inputs(t2i, None)
    with pytest.raises(tp.PresentationError, match="no vision inputs"):
        tp.check_vision_inputs(t2i, _vision(GRIDS))


# --- pins and fixtures --------------------------------------------------------------------------


def test_presentation_files_are_the_h3_files() -> None:
    files = tp.inherited_presentation_files()
    assert len(files) == 11
    assert {p.split("/")[0] for p in files} == {"tokenizer", "processor"}
    h3 = q.release_fingerprint()["files"]
    for path, (size, oid, sha) in files.items():
        assert (size, sha) == (int(h3[path]["size_bytes"]), h3[path]["sha256"])
        assert len(oid) == 40


def test_fixture_cases_cover_the_phase_b_gate() -> None:
    cases = {c["name"]: c for c in tp.sealed_fixtures()["cases"]}
    assert {c["mode"] for c in cases.values()} == set(tp.MODES)
    edit = cases["edit"]["references"]
    sizes = [(r["width"], r["height"]) for r in edit]
    assert len(edit) >= 2 and len(set(sizes)) == len(sizes)
    for width, height in sizes:
        # Inside the Diffusers vision bounds (short edge 1024, sides multiples of 32), so the
        # resize step leaves the reference alone.
        assert min(width, height) == 1024 and width % 32 == 0 and height % 32 == 0
    swapped = [(r["width"], r["height"], r["seed"]) for r in cases["edit_swapped"]["references"]]
    original = [(r["width"], r["height"], r["seed"]) for r in edit]
    assert sorted(swapped) == sorted(original) and swapped != original
    assert cases["edit"]["prompt"] == cases["edit_swapped"]["prompt"]
    for case in cases.values():
        assert "expected" in case, f"{case['name']} has no sealed CPU digests"
        assert case.get("features"), f"{case['name']} has no sealed layer-50 feature"
        for key, sealed in case["features"].items():
            assert sealed["shape"] == [1, case["expected"]["num_rows"], tp.FEATURE_WIDTH], key
            assert sealed["dtype"] == "bfloat16", key
    # Every case is sealed for the same set of devices.
    assert len({tuple(sorted(c["features"])) for c in cases.values()}) == 1
    assert cases["t2i_truncated"]["expected"]["prompt_truncated"]
    assert cases["t2i_truncated"]["expected"]["num_rows"] == 4096


def test_fixture_images_are_deterministic() -> None:
    pytest.importorskip("PIL")
    recipe = {"width": 96, "height": 64, "seed": 5}
    a, b = tp.fixture_image(recipe), tp.fixture_image(recipe)
    assert a.size == (96, 64) and a.mode == "RGB"
    assert a.tobytes() == b.tobytes()
    assert tp.fixture_image({**recipe, "seed": 6}).tobytes() != a.tobytes()


# --- tiny Qwen3-VL ------------------------------------------------------------------------------


def _tiny_config(num_layers: int = 51) -> Any:
    import transformers

    return transformers.Qwen3VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 151936,
            "rope_theta": 5_000_000.0,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [4, 2, 2],
                "rope_type": "default",
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [0, 1],
            "num_position_embeddings": 2304,
            "rope_parameters": {"rope_theta": 10_000.0, "rope_type": "axial"},
        },
        image_token_id=PAD,
        video_token_id=IDS[q.VIDEO_PAD],
        vision_start_token_id=START,
        vision_end_token_id=END,
    )


@pytest.fixture(scope="module")
def tiny() -> Any:
    import transformers

    torch.manual_seed(0)
    return transformers.Qwen3VLForConditionalGeneration(_tiny_config()).eval()


@pytest.fixture(scope="module")
def tiny_encoder(tiny: Any) -> q.Qwen3VLLayer50Encoder:
    return q.Qwen3VLLayer50Encoder(tiny, check_release=False)


@torch.no_grad()
def _hidden_states_50(model: Any, ids: list[int], vision: Any = None) -> torch.Tensor:
    input_ids = torch.tensor([ids])
    out = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=torch.tensor([q.mm_token_type_ids(ids)]),
        use_cache=False,
        output_hidden_states=True,
        **(vision or {}),
    )
    return out.hidden_states[50]


def _random_prompt(seed: int, length: int) -> str:
    rng = random.Random(seed)
    return "".join(chr(rng.randrange(0x4E00, 0x9FFF)) for _ in range(length))


@needs_qwen3vl
def test_edit_feature_is_hidden_states_50(tiny: Any, tiny_encoder: Any) -> None:
    p = tp.present_edit(TOK, _random_prompt(0, 12), GRIDS)
    vision = _vision(GRIDS)
    out = tp.encode(tiny_encoder, p, vision)
    assert out.prompt_embeds.shape == (1, p.num_rows, 64)
    assert torch.equal(out.prompt_embeds, _hidden_states_50(tiny, list(p.token_ids), vision))
    assert torch.equal(out.text_token_tags, p.tags_tensor())


@needs_qwen3vl
def test_encode_rejects_mismatched_inputs(tiny_encoder: Any) -> None:
    p = tp.present_edit(TOK, "p", GRIDS)
    with pytest.raises(tp.PresentationError, match="reference order"):
        tp.encode(tiny_encoder, p, _vision([GRIDS[2], GRIDS[1], GRIDS[0]]))
    tags = list(p.token_tags)
    tags[-1] = tp.IMAGE_TAG
    with pytest.raises(tp.PresentationError):
        tp.encode(tiny_encoder, _with_tags(p, tags), _vision(GRIDS))


@needs_qwen3vl
def test_swapping_equal_geometry_references_changes_the_feature(tiny_encoder: Any) -> None:
    """RFC #738 I9 when the ids cannot show it: same grids, references in the other order."""

    grids = [(1, 4, 4), (1, 4, 4)]
    p = tp.present_edit(TOK, _random_prompt(1, 8), grids)
    a = _vision(grids, seed=3)
    rows = 16
    b = {**a, "pixel_values": torch.cat([a["pixel_values"][rows:], a["pixel_values"][:rows]])}
    fa = tp.encode(tiny_encoder, p, a).prompt_embeds
    fb = tp.encode(tiny_encoder, p, b).prompt_embeds
    assert not torch.equal(fa, fb)
    # Vision rows are first to differ; the prompt rows after them see the change too.
    prompt = p.segments[-1]
    assert not torch.equal(fa[:, prompt.start :], fb[:, prompt.start :])


@needs_qwen3vl
@pytest.mark.parametrize("probe", ["prompt_first", "special_token_inserted", "label_dropped"])
def test_presentation_probes_change_the_layer50_feature(tiny_encoder: Any, probe: str) -> None:
    """RFC #738 I19: each alternative presentation policy moves the layer-50 hidden state."""

    grids = [(1, 4, 6), (1, 2, 2)]
    vision = _vision(grids, seed=4)
    p = tp.present_edit(TOK, _random_prompt(2, 10), grids)
    base = tp.encode(tiny_encoder, p, vision).prompt_embeds
    ids = list(p.token_ids)
    cut = p.segments[-1].start
    if probe == "prompt_first":
        alt = ids[cut:] + ids[:cut]
    elif probe == "special_token_inserted":
        alt = [q.special_token_ids()["<|im_start|>"]] + ids
    else:
        label = p.segments[0]
        alt = ids[label.length :]
    other = tiny_encoder.encode(alt, vision).feature
    assert other.shape != base.shape or not torch.equal(other, base)


@needs_qwen3vl
def test_truncation_policy_changes_the_rows(tiny_encoder: Any) -> None:
    """RFC #738 I19: a different truncation policy changes what the transformer is given."""

    prompt = _random_prompt(3, 4100)
    p = tp.present_text2image(TOK, prompt)
    kept = tiny_encoder.encode(list(p.token_ids)).feature
    untruncated = tiny_encoder.encode(_ids(prompt)).feature
    assert kept.shape[1] == 4096 and untruncated.shape[1] == 4100


# --- released tokenizer / processor -------------------------------------------------------------


def _image_dir() -> Path:
    value = os.environ.get(IMAGE_DIR_ENV)
    if not value:
        pytest.skip(f"set {IMAGE_DIR_ENV} to a MiniMax-H3-Image download")
    return Path(value)


@pytest.fixture(scope="module")
def release_parts() -> tuple[Any, Any]:
    root = _image_dir()
    pytest.importorskip("transformers")
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    processor = AutoProcessor.from_pretrained(str(root / "processor"))
    return tokenizer, processor


def test_release_presentation_files_match_both_pins() -> None:
    checked = tp.verify_presentation_files(_image_dir())
    assert len(checked) == 11


def test_release_tokenizer_and_processor_are_the_h3_pins(release_parts: Any) -> None:
    tokenizer, processor = release_parts
    q.verify_release_tokenizer(tokenizer)
    q.verify_release_processor(processor)
    assert tp.merge_size_of(processor) == 2
    for name, token in IDS.items():
        assert tokenizer.convert_tokens_to_ids(name) == token


def test_release_label_is_tokenized_on_its_own(release_parts: Any) -> None:
    tokenizer, _ = release_parts
    label = tokenizer("Picture 1: ", add_special_tokens=False)["input_ids"]
    assert label == [24669, 220, 16, 25, 220]
    prompt = "Put the cup on the table."
    p = tp.present_edit(tokenizer, prompt, [(1, 2, 2)])
    joint = tokenizer("Picture 1: " + prompt, add_special_tokens=False)["input_ids"]
    separate = label + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    assert joint != separate  # the trailing space merges into " Put" in one call
    assert list(p.token_ids[: len(label)]) == label
    assert list(p.token_ids[p.segments[-1].start :]) == separate[len(label) :]


def test_release_prompt_round_trips(release_parts: Any) -> None:
    """RFC #738 I24: what is encoded is the prompt as given, with no rewrite."""

    tokenizer, _ = release_parts
    for case in tp.sealed_fixtures()["cases"]:
        if case["name"] == "t2i_truncated":
            continue
        prompt = tp.fixture_prompt(case)
        p = tp.present_text2image(tokenizer, prompt)
        assert tokenizer.decode(list(p.token_ids)) == prompt


def test_release_sealed_cpu_digests(release_parts: Any) -> None:
    tokenizer, processor = release_parts
    for case in tp.sealed_fixtures()["cases"]:
        presentation, vision = tp.build_fixture_case(case, tokenizer, processor)
        tp.check_tags(presentation)
        tp.check_vision_inputs(presentation, vision)
        got = {**presentation.summary(), "vision": tp.vision_digest(vision)}
        assert got == case["expected"], case["name"]


# --- pinned Diffusers ---------------------------------------------------------------------------


@needs_diffusers
def test_installed_diffusers_is_the_pinned_reference() -> None:
    import diffusers
    from diffusers.modular_pipelines.minimax_h3_image import modular_pipeline as mp

    checked = tp.verify_diffusers_reference(Path(diffusers.__file__).parent)
    assert set(checked) == set(tp.DIFFUSERS_REFERENCE_FILES)
    assert (mp.MINIMAX_H3_IMAGE_TAG, mp.MINIMAX_H3_TEXT_TAG) == (tp.IMAGE_TAG, tp.TEXT_TAG)
    pipeline = mp.MiniMaxH3ImageModularPipeline
    assert getattr(pipeline, "max_sequence_length").fget(None) == tp.MAX_SEQUENCE_LENGTH
    assert getattr(pipeline, "text_encoder_layer").fget(None) == tp.TEXT_ENCODER_LAYER


@needs_diffusers
def test_pinned_pipeline_has_no_prompt_expander() -> None:
    """RFC #738 I24: no component of the pinned blocks rewrites the prompt."""

    from diffusers.modular_pipelines.minimax_h3_image import MiniMaxH3ImageBlocks

    names = {spec.name for spec in getattr(MiniMaxH3ImageBlocks(), "expected_components")}
    assert names >= {"text_encoder", "tokenizer", "processor"}
    assert not any("expand" in n or "rewrite" in n or "llm" in n for n in names), names


@needs_diffusers
@needs_qwen3vl
def test_diffusers_steps_bitwise_on_tiny_model(release_parts: Any, tiny: Any) -> None:
    """Byte-freeze: every fixture through the pinned Diffusers step and the contract."""

    tokenizer, processor = release_parts
    encoder = q.Qwen3VLLayer50Encoder(tiny, check_release=False)
    for case in tp.sealed_fixtures()["cases"]:
        presentation, vision = tp.build_fixture_case(case, tokenizer, processor)
        images = None
        if case["mode"] == tp.IMAGE_CONDITIONED:
            images = [tp.fixture_image(r) for r in case["references"]]
        ref = tp.run_diffusers_reference(
            tiny, tokenizer, processor, tp.fixture_prompt(case), images
        )
        assert ref.token_ids == list(presentation.token_ids), case["name"]
        assert torch.equal(ref.text_token_tags, presentation.tags_tensor()), case["name"]
        assert set(ref.vision_inputs) == set(vision or {})
        for key, value in (vision or {}).items():
            assert torch.equal(ref.vision_inputs[key], value), (case["name"], key)
        ours = tp.encode(encoder, presentation, vision)
        assert torch.equal(ours.prompt_embeds, ref.prompt_embeds), case["name"]


# --- released weights (CUDA) --------------------------------------------------------------------


@pytest.fixture(scope="module")
def release_model() -> Any:
    root = _image_dir()
    te_root = Path(os.environ.get(TEXT_ENCODER_ROOT_ENV, str(root)))
    if not (te_root / "text_encoder" / "model-00014-of-00014.safetensors").is_file():
        pytest.skip(f"no text_encoder weights under {te_root}")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    free, _ = torch.cuda.mem_get_info()
    if free < 70 * 2**30:
        pytest.skip("needs about 70 GB of free GPU memory")
    model, _ = q.load_release_text_encoder(
        te_root / "text_encoder", device="cuda", num_layers=q.RELEASE_NUM_LAYERS
    )
    yield model
    del model
    torch.cuda.empty_cache()


@needs_diffusers
def test_release_fixtures_bitwise_against_diffusers(release_parts: Any, release_model: Any) -> None:
    tokenizer, processor = release_parts
    encoder = q.Qwen3VLLayer50Encoder(release_model)
    key = tp.device_key(torch.device("cuda"))
    digests = {}
    for case in tp.sealed_fixtures()["cases"]:
        presentation, vision = tp.build_fixture_case(case, tokenizer, processor)
        images = None
        if case["mode"] == tp.IMAGE_CONDITIONED:
            images = [tp.fixture_image(r) for r in case["references"]]
        ref = tp.run_diffusers_reference(
            release_model, tokenizer, processor, tp.fixture_prompt(case), images
        )
        assert ref.token_ids == list(presentation.token_ids)
        ours = tp.encode(encoder, presentation, vision)
        assert ours.prompt_embeds.shape == (1, presentation.num_rows, tp.FEATURE_WIDTH)
        assert torch.equal(ours.prompt_embeds, ref.prompt_embeds), case["name"]
        digests[case["name"]] = ours.output.feature_digest
        sealed = case.get("features", {}).get(key)
        if sealed is not None:
            assert ours.output.feature_digest == sealed["sha256"], case["name"]
            assert tensor_digest(ref.prompt_embeds) == sealed["sha256"]
    assert digests["edit"] != digests["edit_swapped"]
