# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 Phase B ``qwen3vl_layer50_contract``: tokenizer, processor and layer-50 extraction.

The extraction tests build a small random ``Qwen3VLForConditionalGeneration`` (53 layers, so
``hidden_states[49..51]`` all exist) and run on CPU wherever transformers ships Qwen3-VL.
Tests that compare with Diffusers skip unless Diffusers with MiniMax-H3 (f53d5520+) is
importable. Tokenizer/processor tests read the released files from
``$RL_KERNEL_H3_QWEN3VL_DIR`` (a download of ``MiniMaxAI/MiniMax-H3`` with ``tokenizer/``,
``processor/`` and ``text_encoder/``). Released-weight tests also need CUDA with about 60 GB
free.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from rl_engine.models.minimax_h3 import qwen3vl_text as q

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    pytest.skip("transformers without Qwen3-VL", allow_module_level=True)

QWEN3VL_DIR_ENV = "RL_KERNEL_H3_QWEN3VL_DIR"

PROMPTS = (
    "A red fox runs through fresh snow at dawn.",
    "镜头缓慢推进，一位老人在雨中撑伞走过石桥。",
    "Close-up of a barista pouring latte art, warm light, shallow depth of field, 24 fps.",
)
IDS = q.vision_token_ids()
VISION_ROPE = {"rope_theta": 10_000.0, "rope_type": "axial"}
SGLANG_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
    "minimax_h3.packed_sequence"
)


# --- tiny model -------------------------------------------------------------------------------


def _tiny_config(num_layers: int = 53) -> Any:
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
            # What transformers derives for the release; explicit so an SGLang import cannot
            # leave it at "default" (see q.restore_vision_rope).
            "rope_parameters": dict(VISION_ROPE),
        },
        image_token_id=IDS[q.IMAGE_PAD],
        video_token_id=IDS[q.VIDEO_PAD],
        vision_start_token_id=IDS[q.VISION_START],
        vision_end_token_id=IDS[q.VISION_END],
    )


@pytest.fixture(scope="module")
def tiny() -> Any:
    torch.manual_seed(0)
    return transformers.Qwen3VLForConditionalGeneration(_tiny_config()).eval()


@pytest.fixture
def tiny_fresh() -> list[Any]:
    """A tiny model handed over in a list, so the test itself keeps no reference to it."""

    torch.manual_seed(0)
    return [transformers.Qwen3VLForConditionalGeneration(_tiny_config()).eval()]


def _encoder(model: Any) -> q.Qwen3VLLayer50Encoder:
    return q.Qwen3VLLayer50Encoder(model, check_release=False)


def _text_ids(seed: int, length: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, 151_000) for _ in range(length)]


def _image_inputs(grids: list[tuple[int, int, int]], seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    patches = sum(t * h * w for t, h, w in grids)
    return {
        "pixel_values": torch.randn(patches, 3 * 2 * 16 * 16, generator=generator),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long),
    }


def _image_presentation(grids: list[tuple[int, int, int]], prompt_len: int = 9) -> list[int]:
    ids: list[int] = []
    for index, (t, h, w) in enumerate(grids):
        ids += _text_ids(100 + index, 4) + q.vision_block("image", t * h * w // 4)
    return ids + _text_ids(7, prompt_len)


@torch.no_grad()
def _hf_hidden_states(
    model: Any, ids: list[int], vision: dict[str, torch.Tensor] | None = None
) -> tuple:
    """The call Diffusers' ``get_qwen3vl_prompt_embeds`` makes, written out."""

    input_ids = torch.tensor([ids])
    out = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=torch.tensor([q.mm_token_type_ids(ids)]),
        use_cache=False,
        output_hidden_states=True,
        **(vision or {}),
    )
    return out.hidden_states


# --- layer indexing ---------------------------------------------------------------------------


def test_feature_is_hidden_states_50_not_49_or_51(tiny: Any) -> None:
    ids = _text_ids(0, 13)
    hidden = _hf_hidden_states(tiny, ids)
    assert len(hidden) == 54  # embeddings + 53 layer outputs (the last one post-norm)
    feature = _encoder(tiny).encode(ids).feature
    assert torch.equal(feature, hidden[50])
    assert not torch.equal(feature, hidden[49])
    assert not torch.equal(feature, hidden[51])


def test_feature_with_vision_block_is_hidden_states_50(tiny: Any) -> None:
    grids = [(1, 4, 4), (1, 4, 6)]
    ids = _image_presentation(grids)
    vision = _image_inputs(grids)
    feature = _encoder(tiny).encode(ids, vision).feature
    assert torch.equal(feature, _hf_hidden_states(tiny, ids, vision)[50])


def test_matches_diffusers_prompt_embeds(tiny: Any) -> None:
    encoders = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.encoders",
        reason="needs diffusers with MiniMax-H3 (f53d5520+)",
    )

    # transformers' own create_mm_token_type_ids with the placeholder ids of the model config,
    # so a wrong mapping in q.mm_token_type_ids cannot cancel out against the reference.
    processor = transformers.Qwen3VLProcessor.__new__(transformers.Qwen3VLProcessor)
    processor.image_token_id = tiny.config.image_token_id
    processor.video_token_id = tiny.config.video_token_id

    grids = [(1, 4, 4)]
    cases: list[tuple[list[int], dict[str, torch.Tensor]]] = [
        (_text_ids(1, 21), {}),
        (_image_presentation(grids), _image_inputs(grids, seed=3)),
    ]
    for ids, vision in cases:
        with torch.no_grad():
            ref = encoders.get_qwen3vl_prompt_embeds(
                tiny,
                processor,
                ids,
                vision,
                text_encoder_layer=50,
                device=torch.device("cpu"),
                dtype=tiny.dtype,
            )
        assert torch.equal(_encoder(tiny).encode(ids, vision or None).feature, ref)


def test_truncated_stack_gives_the_same_bits(tiny: Any) -> None:
    truncated = transformers.Qwen3VLForConditionalGeneration(_tiny_config(num_layers=50)).eval()
    missing, unexpected = truncated.load_state_dict(tiny.state_dict(), strict=False)
    assert not missing
    assert unexpected and all(".layers.5" in key for key in unexpected)
    ids = _text_ids(2, 17)
    feature = _encoder(truncated).encode(ids).feature
    assert torch.equal(feature, _encoder(tiny).encode(ids).feature)
    # The trap Diffusers guards against: the last hidden state of a 50-layer stack is post-norm.
    last = _hf_hidden_states(truncated, ids)[-1]
    assert not torch.equal(last, feature)


def test_diffusers_refuses_a_50_layer_stack_the_contract_does_not() -> None:
    encoders = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.encoders",
        reason="needs diffusers with MiniMax-H3 (f53d5520+)",
    )
    truncated = transformers.Qwen3VLForConditionalGeneration(_tiny_config(num_layers=50)).eval()
    with pytest.raises(ValueError, match="post-norm"):
        encoders.get_qwen3vl_prompt_embeds(truncated, None, [1, 2, 3], {}, text_encoder_layer=50)
    assert _encoder(truncated).encode([1, 2, 3]).feature.shape == (1, 3, 64)


def test_too_shallow_stack_is_rejected() -> None:
    shallow = transformers.Qwen3VLForConditionalGeneration(_tiny_config(num_layers=49)).eval()
    with pytest.raises(ValueError, match="needs >= 50"):
        _encoder(shallow)


def test_tiny_config_fails_the_release_check(tiny: Any) -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        q.Qwen3VLLayer50Encoder(tiny)


@pytest.mark.parametrize(
    "key", ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]
)
def test_release_check_pins_the_vision_placeholder_ids(key: str, tiny_fresh: list[Any]) -> None:
    encoder = _encoder(tiny_fresh.pop())
    pinned = q.release_fingerprint()
    # A config with the pinned text/vision values (no weights): passes the check as is.
    encoder.model.config = transformers.Qwen3VLConfig(
        text_config={**pinned["text_config"], "rope_parameters": dict(pinned["rope"])},
        vision_config={**pinned["vision_config"], "rope_parameters": dict(VISION_ROPE)},
        image_token_id=IDS[q.IMAGE_PAD],
        video_token_id=IDS[q.VIDEO_PAD],
        vision_start_token_id=IDS[q.VISION_START],
        vision_end_token_id=IDS[q.VISION_END],
    )
    encoder._check_release_config()
    setattr(encoder.model.config, key, IDS[q.VISION_START] - 1)
    with pytest.raises(ValueError, match=key):
        encoder._check_release_config()


def test_offload_hook_is_rejected(tiny: Any) -> None:
    tiny._hf_hook = object()
    try:
        with pytest.raises(RuntimeError, match="offload"):
            _encoder(tiny)
    finally:
        del tiny._hf_hook


def test_hook_is_removed_after_encode(tiny: Any) -> None:
    encoder = _encoder(tiny)
    before = len(encoder.target._forward_hooks)
    encoder.encode(_text_ids(4, 5))
    assert len(encoder.target._forward_hooks) == before


# --- repeatability and batch composition ------------------------------------------------------


def test_feature_is_bitwise_repeatable(tiny: Any) -> None:
    encoder = _encoder(tiny)
    ids = _text_ids(5, 31)
    digests = {encoder.encode(ids).feature_digest for _ in range(3)}
    assert len(digests) == 1


def test_feature_is_independent_of_batch_composition(tiny: Any) -> None:
    encoder = _encoder(tiny)
    grids = [(1, 4, 4)]
    rows: list[Any] = [_text_ids(10 + i, n) for i, n in enumerate((3, 17, 40, 8))]
    rows.append((_image_presentation(grids), _image_inputs(grids, seed=9)))
    solo = [encoder.encode_many([row])[0].feature_digest for row in rows]
    rng = random.Random(0)
    for _ in range(4):
        order = list(range(len(rows)))
        rng.shuffle(order)
        size = rng.randrange(1, len(rows) + 1)
        got = encoder.encode_many([rows[i] for i in order[:size]])
        assert [o.feature_digest for o in got] == [solo[i] for i in order[:size]]


# --- presentation checks (pinned token table only) --------------------------------------------


def test_mm_token_type_ids_and_vision_block() -> None:
    block = q.vision_block("image", 3)
    assert block == [IDS[q.VISION_START]] + [IDS[q.IMAGE_PAD]] * 3 + [IDS[q.VISION_END]]
    ids = [5] + block + q.vision_block("video", 2) + [6]
    assert q.mm_token_type_ids(ids) == [0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 0]
    assert q.attention_mask(ids) == [1] * len(ids)


@pytest.mark.parametrize(
    ("ids", "vision", "match"),
    [
        ([1, IDS[q.IMAGE_PAD], 2], None, "outside a vision block"),
        ([IDS[q.VISION_START], IDS[q.IMAGE_PAD], 2], None, "malformed"),
        (
            [IDS[q.VISION_START], IDS[q.IMAGE_PAD], IDS[q.VIDEO_PAD], IDS[q.VISION_END]],
            None,
            "mixes",
        ),
        (q.vision_block("image", 4), None, "processor grids"),
        (q.vision_block("image", 3), _image_inputs([(1, 4, 4)]), "processor grids"),
        ([1, 2], _image_inputs([(1, 4, 4)]), "processor grids"),
    ],
)
def test_check_presentation_fails_closed(ids: list[int], vision: Any, match: str) -> None:
    with pytest.raises(q.PresentationError, match=match):
        q.check_presentation(ids, vision)


def test_check_presentation_accepts_release_layouts() -> None:
    grids = [(1, 4, 4), (1, 6, 8)]
    q.check_presentation(_image_presentation(grids), _image_inputs(grids))
    video = {
        "pixel_values_videos": torch.zeros(2 * 4 * 4, 1536),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
    }
    q.check_presentation([9] + q.vision_block("video", 4) + [8] + q.vision_block("video", 4), video)


def test_required_shards_skip_late_layers() -> None:
    weight_map = {
        "model.language_model.embed_tokens.weight": "a",
        "model.visual.patch_embed.proj.weight": "a",
        "model.language_model.layers.0.mlp.up_proj.weight": "a",
        "model.language_model.layers.49.mlp.up_proj.weight": "b",
        "model.language_model.layers.50.mlp.up_proj.weight": "c",
        "model.language_model.layers.63.mlp.up_proj.weight": "d",
        "model.language_model.norm.weight": "d",
        "lm_head.weight": "e",
    }
    # The truncated model still has the final norm and lm_head, and the loader fails on a
    # missing key, so their shards are required even when they hold no early layer.
    assert q.required_shards(weight_map) == ["a", "b", "d", "e"]
    keys = q.required_weight_keys(weight_map)
    assert "model.language_model.norm.weight" in keys and "lm_head.weight" in keys
    assert not any(".layers.50." in k or ".layers.63." in k for k in keys)


def test_release_policy_is_torch_default_and_restored() -> None:
    code = (
        "import json; from rl_engine.models.minimax_h3 import qwen3vl_text as q; "
        "from dataclasses import asdict; "
        "print(json.dumps([q.runtime_readback(), asdict(q.RELEASE_POLICY)]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    readback, policy = json.loads(out.stdout.strip().splitlines()[-1])
    # Every flag is read back (a missing one raises KeyError); sdpa_backends=None means all four
    # SDPA backends stay enabled, PyTorch's default priority.
    flags = {k: v for k, v in policy.items() if k != "sdpa_backends"}
    assert {k: readback[k] for k in flags} == flags
    assert policy["sdpa_backends"] is None
    sdpa = ("sdpa_flash", "sdpa_mem_efficient", "sdpa_math", "sdpa_cudnn")
    assert set(readback) == set(flags) | set(sdpa)
    assert all(readback[k] for k in sdpa)
    before = q.runtime_readback()
    with q.pinned_runtime(q.RuntimePolicy(matmul_allow_tf32=True, cudnn_benchmark=True)) as inside:
        assert inside["matmul_allow_tf32"] and inside["cudnn_benchmark"]
    assert q.runtime_readback() == before


def test_fingerprint_pins_release_identity() -> None:
    fp = q.release_fingerprint()
    assert fp["text_config"]["hidden_size"] == q.FEATURE_WIDTH
    assert fp["text_config"]["num_hidden_layers"] == q.RELEASE_NUM_LAYERS
    assert fp["vision_config"]["out_hidden_size"] == q.FEATURE_WIDTH
    assert q.special_token_ids()[q.IMAGE_PAD] == 151655
    shards = [k for k in fp["files"] if k.startswith("text_encoder/model-")]
    assert len(shards) == 14


# --- evidence script: the full model is freed before the 50-layer stack loads -----------------


def test_evidence_release_after_frees_the_model_and_catches_a_leak() -> None:
    import gc
    import weakref

    from tools.validation.models import h3_qwen3vl_evidence as ev

    cpu = torch.device("cpu")
    seen: list[Any] = []

    def clean(full: Any) -> dict[str, Any]:
        fn = lambda full=full: full(torch.ones(1, 4))  # noqa: E731 - the pattern that leaked
        seen.append(weakref.ref(full))
        with torch.no_grad():
            return {"sum": float(fn().sum())}

    report, freed = ev.release_after(
        lambda: torch.nn.Linear(4, 4), clean, device=cpu, max_retained_bytes=0
    )
    assert set(report) == {"sum"} and freed == {}
    assert seen[0]() is None

    survivors: list[Any] = []

    def leaky(full: Any) -> dict[str, Any]:
        # A default argument that outlives the stages, as at 97ba3b82 (main's frame held it).
        survivors.append(lambda full=full: full)
        return {}

    with pytest.raises(RuntimeError, match="still referenced"):
        ev.release_after(lambda: torch.nn.Linear(4, 4), leaky, device=cpu, max_retained_bytes=0)
    survivors.clear()
    gc.collect()

    with pytest.raises(TypeError):  # a tensor in the report would keep activations alive
        ev.release_after(
            lambda: torch.nn.Linear(4, 4),
            lambda full: {"t": full.weight.detach()},
            device=cpu,
            max_retained_bytes=0,
        )


@pytest.mark.parametrize("repeats", ["0", "-1"])
def test_evidence_rejects_repeats_below_one(
    repeats: str, tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Fails at argument parsing, before any file is hashed or model loaded."""

    from tools.validation.models import h3_qwen3vl_evidence as ev

    argv = ["h3_qwen3vl_evidence.py", "--root", str(tmp_path / "missing"), "--out", "x.json"]
    monkeypatch.setattr(sys, "argv", [*argv, "--repeats", repeats])
    with pytest.raises(SystemExit) as exc:
        ev.main()
    assert exc.value.code == 2
    assert "--repeats must be at least 1" in capsys.readouterr().err


def test_evidence_sdpa_cell_records_only_unsupported_backends() -> None:
    from tools.validation.models import h3_qwen3vl_evidence as ev

    x = torch.randn(1, 2, 4, 8, generator=torch.Generator().manual_seed(0))

    def attend() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(x, x, x)

    ref = attend()
    assert "error" not in ev.sdpa_backend_cell("MATH", attend, ref)
    # No CUDNN kernel takes CPU tensors: recorded as unsupported.
    cell = ev.sdpa_backend_cell("CUDNN_ATTENTION", attend, ref)
    assert cell["error"].startswith("RuntimeError: No viable backend")

    def broken() -> torch.Tensor:
        raise TypeError("bad call")

    def out_of_memory() -> torch.Tensor:
        raise torch.OutOfMemoryError("CUDA out of memory")

    for run, error in ((broken, TypeError), (out_of_memory, torch.OutOfMemoryError)):
        with pytest.raises(error):
            ev.sdpa_backend_cell("MATH", run, ref)


# --- loading in a process that imported SGLang ------------------------------------------------


@pytest.fixture(params=["simulated", "sglang"])
def sglang_rope_patch(request: Any, monkeypatch: Any) -> None:
    """transformers as SGLang 0.5.21 leaves it after import: ``standardize_rope_params`` returns
    early for configs without ``max_position_embeddings``, such as Qwen3-VL's vision config.
    ``simulated`` installs the same guard; ``sglang`` imports SGLang. Undone after the test."""

    config_cls = transformers.PretrainedConfig
    original = config_cls.standardize_rope_params
    monkeypatch.setattr(config_cls, "standardize_rope_params", original)
    if request.param == "sglang":
        pytest.importorskip(SGLANG_MODULE, reason="needs SGLang")
    else:

        def skip_without_max_positions(self: Any) -> None:
            if hasattr(self, "max_position_embeddings"):
                original(self)

        monkeypatch.setattr(config_cls, "standardize_rope_params", skip_without_max_positions)
        rope = transformers.Qwen3VLVisionConfig().rope_parameters or {}
        assert rope.get("rope_type") != "axial"


def test_loader_restores_vision_rope_under_sglang_patch(
    sglang_rope_patch: None, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(_tiny_config(num_layers=50)).eval()
    model.save_pretrained(tmp_path)
    # Like the released config.json: no RoPE fields in vision_config.
    config_file = tmp_path / "config.json"
    saved = json.loads(config_file.read_text())
    saved["vision_config"].pop("rope_parameters", None)
    config_file.write_text(json.dumps(saved))
    loaded, report = q.load_release_text_encoder(tmp_path, device="cpu", dtype=torch.float32)
    assert report["num_layers"] == 50 and report["skipped_keys"] == 0
    assert loaded.config.vision_config.rope_parameters == VISION_ROPE
    grids = [(1, 4, 4)]
    ids, vision = _image_presentation(grids), _image_inputs(grids)
    assert torch.equal(
        _encoder(loaded).encode(ids, vision).feature, _encoder(model).encode(ids, vision).feature
    )


def test_restore_vision_rope_changes_nothing_without_the_patch(monkeypatch: Any) -> None:
    # Drop a patch installed by an earlier SGLang import in this process, if any.
    if "standardize_rope_params" in vars(transformers.PretrainedConfig):
        monkeypatch.delattr(transformers.PretrainedConfig, "standardize_rope_params")
    config = transformers.Qwen3VLConfig(vision_config={"depth": 2})
    before = config.vision_config.rope_parameters
    assert before == VISION_ROPE
    q.restore_vision_rope(config)
    assert config.vision_config.rope_parameters is before


# --- released tokenizer / processor files -----------------------------------------------------


def _release_dir() -> Path:
    root = os.environ.get(QWEN3VL_DIR_ENV)
    if not root:
        pytest.skip(
            f"{QWEN3VL_DIR_ENV} not set (download of MiniMaxAI/MiniMax-H3 with tokenizer/processor)"
        )
    return Path(root)


@pytest.fixture(scope="module")
def release_tokenizer() -> Any:
    root = _release_dir()
    q.verify_release_files(root, subfolders=("tokenizer", "processor"))
    return transformers.AutoTokenizer.from_pretrained(str(root / "tokenizer"))


@pytest.fixture(scope="module")
def release_processor() -> Any:
    return transformers.AutoProcessor.from_pretrained(str(_release_dir() / "processor"))


def test_release_tokenizer_matches_pin(release_tokenizer: Any, release_processor: Any) -> None:
    q.verify_release_tokenizer(release_tokenizer)
    q.verify_release_tokenizer(release_processor.tokenizer)
    q.verify_release_processor(release_processor)


def test_token_ids_and_masks_identical_across_runs(release_tokenizer: Any) -> None:
    root = _release_dir()
    first = {p: q.tokenize_prompt(release_tokenizer, p) for p in PROMPTS}
    fresh = transformers.AutoTokenizer.from_pretrained(str(root / "processor"))
    assert {p: q.tokenize_prompt(fresh, p) for p in PROMPTS} == first
    code = (
        "import json, sys, transformers\n"
        "from rl_engine.models.minimax_h3 import qwen3vl_text as q\n"
        "t = transformers.AutoTokenizer.from_pretrained(sys.argv[1])\n"
        "rows = [q.tokenize_prompt(t, p) for p in json.loads(sys.argv[2])]\n"
        "print(json.dumps([[q.token_ids_digest(r), q.attention_mask(r)] for r in rows]))"
    )
    runs = []
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-c", code, str(root / "tokenizer"), json.dumps(PROMPTS)],
            capture_output=True,
            text=True,
            check=True,
        )
        runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
    expected = [[q.token_ids_digest(first[p]), [1] * len(first[p])] for p in PROMPTS]
    assert runs[0] == runs[1] == expected


@pytest.mark.parametrize("token", sorted(q.special_token_ids()))
def test_prompt_spelling_a_special_token_is_rejected(release_tokenizer: Any, token: str) -> None:
    assert (
        q.special_token_ids()[token]
        in release_tokenizer(f"look {token} here", add_special_tokens=False)["input_ids"]
    )
    with pytest.raises(q.PresentationError, match="special tokens"):
        q.tokenize_prompt(release_tokenizer, f"look {token} here")


def test_non_special_added_tokens_pass_as_in_the_release(release_tokenizer: Any) -> None:
    # ``<think>`` is an added token with special=False: the release tokenizes it to one id and
    # Diffusers conditions on it; the contract keeps that behaviour.
    ids = q.tokenize_prompt(release_tokenizer, "<think> a cat")
    assert ids[0] == release_tokenizer.convert_tokens_to_ids("<think>")


def test_mm_token_type_ids_match_processor(release_processor: Any) -> None:
    ids = [5, 6] + q.vision_block("image", 7) + [8] + q.vision_block("video", 3) + [9]
    assert q.mm_token_type_ids(ids) == list(release_processor.create_mm_token_type_ids([ids])[0])


def test_image_processor_is_batch_invariant_and_matches_presentation(
    release_tokenizer: Any, release_processor: Any
) -> None:
    np = pytest.importorskip("numpy")
    image_mod = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    images = [
        image_mod.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        for h, w in ((480, 832), (512, 512), (300, 401))
    ]
    processor = release_processor.image_processor
    batch = processor(images=images, return_tensors="pt")
    offset = 0
    for index, image in enumerate(images):
        one = processor(images=[image], return_tensors="pt")
        rows = int(one["image_grid_thw"][0].prod())
        assert torch.equal(batch["pixel_values"][offset : offset + rows], one["pixel_values"])
        assert torch.equal(batch["image_grid_thw"][index], one["image_grid_thw"][0])
        offset += rows
    ids: list[int] = []
    for index, grid in enumerate(batch["image_grid_thw"].tolist()):
        ids += release_tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        ids += q.vision_block("image", grid[0] * grid[1] * grid[2] // processor.merge_size**2)
    ids += q.tokenize_prompt(release_tokenizer, PROMPTS[0])
    q.check_presentation(ids, dict(batch))


def test_video_presentation_from_the_release_processor(release_processor: Any) -> None:
    """transformers' own expansion of one ``<|video_pad|>`` for a video with ``t > 1``."""

    np = pytest.importorskip("numpy")
    frames = np.random.default_rng(0).integers(0, 256, (8, 128, 160, 3), dtype=np.uint8)
    out = release_processor(
        text=["<Video 1>: <|video_pad|>" + PROMPTS[0]],
        videos=[frames],
        video_metadata=[{"fps": 2.0, "total_num_frames": 8, "frames_indices": list(range(8))}],
        do_sample_frames=False,
        return_tensors="pt",
    )
    t, h, w = (int(x) for x in out["video_grid_thw"][0])
    assert t > 1
    ids = out["input_ids"][0].tolist()
    merge = release_processor.video_processor.merge_size**2
    assert q._pad_runs(ids) == [("video", h * w // merge)] * t
    vision = {k: out[k] for k in ("pixel_values_videos", "video_grid_thw")}
    q.check_presentation(ids, vision)
    one_block = q.vision_block("video", t * h * w // merge)
    with pytest.raises(q.PresentationError, match="video blocks hold"):
        q.check_presentation(one_block, vision)


def test_evidence_full_model_stages_release_the_model(
    tiny_fresh: Any, release_tokenizer: Any, release_processor: Any, monkeypatch: Any
) -> None:
    """Stages 3-6 of the evidence script on a tiny model (CPU): every report, then a dead model."""

    pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.encoders",
        reason="needs diffusers with MiniMax-H3 (f53d5520+)",
    )
    import weakref

    from tools.validation.models import h3_qwen3vl_evidence as ev

    # The tiny model is not the release config; only the release-config check is relaxed.
    release_encoder = q.Qwen3VLLayer50Encoder
    monkeypatch.setattr(
        ev.q,
        "Qwen3VLLayer50Encoder",
        lambda model, **kw: release_encoder(model, **{**kw, "check_release": False}),
    )
    cpu = torch.device("cpu")
    rows = {n: q.tokenize_prompt(release_tokenizer, t) for n, t in ev.PROMPTS.items()}
    cases: dict[str, tuple[list[int], dict[str, Any]]] = {n: (r, {}) for n, r in rows.items()}
    cases["fl2va_1kf"] = ev._vision_presentation(
        release_tokenizer, release_processor, 1, ev.PROMPTS["short_en"]
    )
    cases["fl2va_2kf"] = ev._vision_presentation(
        release_tokenizer, release_processor, 2, ev.PROMPTS["mid_en"]
    )
    alive: list[Any] = []

    def load() -> Any:
        model = tiny_fresh.pop()
        alive.append(weakref.ref(model))
        return model

    report, freed = ev.release_after(
        load,
        lambda full: ev._full_model_stages(
            full,
            rows=rows,
            cases=cases,
            tokenizer=release_tokenizer,
            processor=release_processor,
            device=cpu,
            repeats=1,
        ),
        device=cpu,
        max_retained_bytes=0,
    )
    assert alive[0]() is None and freed == {}
    assert set(report) == {
        "reference",
        "full_model_param_bytes",
        "sdpa_backends_vs_default",
        "prefix_invariance",
        "prior_art_batching",
        "vision_tower_batch_composition",
    }
    assert set(report["reference"]) == set(cases)
    for entry in report["reference"].values():
        assert entry["contract_on_full_model_sha256"] == entry["diffusers_sha256"]
        assert entry["hidden_states_50_sha256"] == entry["diffusers_sha256"]


def test_release_required_shards() -> None:
    weight_map = json.loads(
        (_release_dir() / "text_encoder" / "model.safetensors.index.json").read_text()
    )["weight_map"]
    shards = q.required_shards(weight_map)
    assert len(shards) == 12
    assert "model-00012-of-00014.safetensors" not in shards
    assert "model-00013-of-00014.safetensors" not in shards
    for key in ("model.language_model.norm.weight", "lm_head.weight"):
        assert weight_map[key] in shards


# --- released weights (CUDA) ------------------------------------------------------------------


@pytest.fixture(scope="module")
def release_encoder() -> q.Qwen3VLLayer50Encoder:
    root = _release_dir()
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    free, _ = torch.cuda.mem_get_info()
    if free < 60 * 2**30:
        pytest.skip(f"needs ~60 GiB free on the GPU, have {free / 2**30:.1f}")
    model, report = q.load_release_text_encoder(root / "text_encoder", device="cuda")
    assert report["num_layers"] == 50 and report["skipped_keys"] > 0
    return q.Qwen3VLLayer50Encoder(model)


def test_release_feature_repeatable_and_batch_invariant(
    release_encoder: q.Qwen3VLLayer50Encoder, release_tokenizer: Any
) -> None:
    rows = [q.tokenize_prompt(release_tokenizer, p) for p in PROMPTS]
    solo = [release_encoder.encode(ids) for ids in rows]
    for out, ids in zip(solo, rows, strict=True):
        assert out.feature.shape == (1, len(ids), q.FEATURE_WIDTH)
        assert out.feature.dtype == torch.bfloat16
        assert torch.isfinite(out.feature).all()
    for _ in range(2):
        again = release_encoder.encode_many(rows)
        assert [o.feature_digest for o in again] == [o.feature_digest for o in solo]
    reordered = release_encoder.encode_many([rows[2], rows[0]])
    assert [o.feature_digest for o in reordered] == [solo[2].feature_digest, solo[0].feature_digest]
