# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 Qwen3-VL text-encoder contract (RFC #420 Phase B ``qwen3vl_layer50_contract``).

MiniMax-H3 conditions its Omni-Transformer on the full Qwen3-VL-32B (``text_encoder/`` of
``MiniMaxAI/MiniMax-H3``, class ``Qwen3VLForConditionalGeneration``). The feature is not the
model's last hidden state. It is ``hidden_states[50]``, the raw (pre-norm) output of decoder
layer index 49, width 5120, which then feeds the 5120 -> 5376 text projection and the 2-layer
token refiner. This module pins everything that decides which feature bytes come out:

* **Tokenizer and special tokens.** The presentation is the prompt verbatim: no chat template
  and ``add_special_tokens=False``. The released tokenizer adds no BOS/EOS either way.
  :func:`verify_release_tokenizer` pins the special-token table. :func:`tokenize_prompt` fails
  closed when a prompt spells out a special token such as ``<|image_pad|>``, which the
  tokenizer would otherwise turn into a vision placeholder with no pixels behind it. Diffusers
  accepts such a prompt silently.
* **Multimodal processor.** :func:`mm_token_type_ids` derives Qwen3-VL's internal run types
  (0 text, 1 image, 2 video) and :func:`check_presentation` checks every vision block against
  the processor's ``*_grid_thw`` before the model runs. :func:`verify_release_processor` pins
  the image/video processor geometry.
* **Exact layer-50 extraction.** :class:`Qwen3VLLayer50Encoder` runs the unmodified HF
  ``Qwen3VLModel`` forward, the same call Diffusers' ``get_qwen3vl_prompt_embeds`` makes, and
  stops it with a forward hook on decoder layer index 49. The captured tensor is the one
  transformers records as ``hidden_states[50]``: the hook fires on the same layer output.
  Layers 50-63, the final norm and the 65 retained hidden states are never computed.
  :func:`load_release_text_encoder` loads a 50-layer stack, which saves about 14 GB.
* **Batch composition.** One presentation per forward, all-ones attention mask, no padding,
  no truncation, as in the release. :meth:`Qwen3VLLayer50Encoder.encode_many` loops over
  rows, so a feature never depends on which other prompts were encoded with it. Padded HF
  batching is not batch-invariant on B200 (see ``docs/operators/h3-qwen3vl-layer50-contract.md``).

The network is reused from HF transformers unmodified. This module imports transformers only
inside :func:`load_release_text_encoder`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from rl_engine.models.minimax_h3.trajectory import tensor_digest

CONTRACT_ID = "h3-qwen3vl-layer50-v1"
FINGERPRINT_PATH = Path(__file__).with_name("qwen3vl_text_fingerprint.json")

TEXT_ENCODER_LAYER = 50
"""Index into transformers' ``hidden_states`` tuple: the output of decoder layer index 49."""
FEATURE_WIDTH = 5120
RELEASE_NUM_LAYERS = 64

# Packed-sequence modality tags (Diffusers ``MINIMAX_H3_*_TAG``). A vision block's rows carry
# the video tag.
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

# Qwen3-VL's internal per-token run types (``Qwen3VLProcessor.create_mm_token_type_ids``).
MM_TEXT, MM_IMAGE, MM_VIDEO = 0, 1, 2


class PresentationError(ValueError):
    """A token sequence or vision input that the release presentation never produces."""


# --- pinned release fingerprint -----------------------------------------------------------


@lru_cache(maxsize=1)
def release_fingerprint() -> Mapping[str, Any]:
    """The pinned checkpoint/tokenizer/processor identity (``qwen3vl_text_fingerprint.json``)."""

    data = json.loads(FINGERPRINT_PATH.read_text())
    if data.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"{FINGERPRINT_PATH.name} is not for {CONTRACT_ID}")
    return data


def special_token_ids() -> dict[str, int]:
    """Every special (``special=True``) added token of the released tokenizer, by content."""

    return {str(k): int(v) for k, v in release_fingerprint()["special_tokens"].items()}


def vision_token_ids() -> dict[str, int]:
    """The four placeholder tokens a vision block is built from."""

    special = special_token_ids()
    return {name: special[name] for name in (VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release_files(
    root: str | Path,
    *,
    subfolders: Sequence[str] = ("tokenizer", "processor"),
    skip_suffixes: Sequence[str] = (),
) -> dict[str, str]:
    """SHA-256 every pinned file under ``root/<subfolder>/`` and fail on any difference.

    ``root`` is a download of the pinned ``MiniMaxAI/MiniMax-H3`` revision. Files whose name
    ends with one of ``skip_suffixes`` (for example ``.safetensors`` in a quick check) are
    skipped. Returns ``{relative path: sha256}`` of what was checked.
    """

    base = Path(root)
    files: Mapping[str, Any] = release_fingerprint()["files"]
    checked: dict[str, str] = {}
    for rel, meta in sorted(files.items()):
        if rel.split("/", 1)[0] not in subfolders or rel.endswith(tuple(skip_suffixes)):
            continue
        path = base / rel
        if not path.is_file():
            raise FileNotFoundError(f"pinned file {rel} is missing under {base}")
        size = path.stat().st_size
        if size != int(meta["size_bytes"]):
            raise ValueError(f"{rel}: size {size} != pinned {meta['size_bytes']}")
        got = _sha256_file(path)
        if got != meta["sha256"]:
            raise ValueError(f"{rel}: sha256 {got} != pinned {meta['sha256']}")
        checked[rel] = got
    if not checked:
        raise ValueError(f"no pinned files under {base} for {tuple(subfolders)}")
    return checked


_LAYER_KEY = re.compile(r"language_model\.layers\.(\d+)\.")


def required_weight_keys(
    weight_map: Mapping[str, str], num_layers: int = TEXT_ENCODER_LAYER
) -> list[str]:
    """Checkpoint keys a ``num_layers``-deep stack loads: every key but decoder layers >= n.

    This is the complement of what :func:`load_release_text_encoder` skips, so it includes the
    final norm and ``lm_head``: the truncated ``Qwen3VLForConditionalGeneration`` has both as
    parameters and fails on a missing key, although the feature never uses them.
    """

    keys = []
    for key in weight_map:
        match = _LAYER_KEY.search(key)
        if match is None or int(match.group(1)) < num_layers:
            keys.append(key)
    return sorted(keys)


def required_shards(
    weight_map: Mapping[str, str], num_layers: int = TEXT_ENCODER_LAYER
) -> list[str]:
    """Shard files that hold :func:`required_weight_keys`.

    ``from_pretrained`` opens every shard the index lists, so :func:`load_release_text_encoder`
    still needs the full ``text_encoder/`` download; the other shards are opened, not read.
    """

    return sorted({weight_map[key] for key in required_weight_keys(weight_map, num_layers)})


# --- tokenizer and presentation -------------------------------------------------------------


def verify_release_tokenizer(tokenizer: Any) -> None:
    """Fail unless ``tokenizer`` has the pinned special-token table and adds no BOS/EOS."""

    pinned = release_fingerprint()["tokenizer"]
    if len(tokenizer) != int(pinned["len"]):
        raise ValueError(f"tokenizer has {len(tokenizer)} tokens, pinned {pinned['len']}")
    actual = {
        str(token.content): int(index)
        for index, token in tokenizer.added_tokens_decoder.items()
        if bool(token.special)
    }
    expected = special_token_ids()
    if actual != expected:
        drift = sorted(set(actual.items()) ^ set(expected.items()))
        raise ValueError(f"special-token table differs from the release: {drift}")
    probe = str(pinned["probe_prompt"])
    plain = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    if plain != [int(i) for i in pinned["probe_ids"]]:
        raise ValueError(f"probe prompt tokenizes to {plain}, pinned {pinned['probe_ids']}")
    if list(tokenizer(probe, add_special_tokens=True)["input_ids"]) != plain:
        raise ValueError("tokenizer adds special tokens; the release presentation has none")


def tokenize_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Token ids of the release presentation of ``prompt``: verbatim, no template, no specials.

    Raises :class:`PresentationError` when the prompt itself spells out a special token. The
    tokenizer would map ``"<|image_pad|>"`` in user text to the vision placeholder id, which
    shifts every later vision block (or, with no vision input, conditions on a placeholder
    embedding the release never trains against).
    """

    if not isinstance(prompt, str):
        raise PresentationError(f"prompt must be a single string, got {type(prompt).__name__}")
    ids = [int(i) for i in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
    by_id = {index: name for name, index in special_token_ids().items()}
    found = sorted({by_id[i] for i in ids if i in by_id})
    if found:
        raise PresentationError(f"prompt text contains special tokens {found}")
    return ids


def attention_mask(token_ids: Sequence[int]) -> list[int]:
    """The release attention mask: one unpadded sequence, so every position is 1."""

    return [1] * len(token_ids)


def mm_token_type_ids(token_ids: Sequence[int]) -> list[int]:
    """Qwen3-VL's per-token run types: 1 on ``<|image_pad|>``, 2 on ``<|video_pad|>``, else 0."""

    vision = vision_token_ids()
    image, video = vision[IMAGE_PAD], vision[VIDEO_PAD]
    return [MM_IMAGE if i == image else MM_VIDEO if i == video else MM_TEXT for i in token_ids]


def vision_block(kind: str, num_tokens: int) -> list[int]:
    """``<|vision_start|>``, ``num_tokens`` pads of ``kind`` (image/video), ``<|vision_end|>``.

    One block per image. A video is ``t`` blocks, one per merged frame pair, of
    ``h * w / merge_size**2`` pads each, as transformers' ``Qwen3VLProcessor`` and Diffusers emit
    them (each after its ``<x.x seconds>`` text): call this once per block.
    """

    if kind not in ("image", "video") or num_tokens <= 0:
        raise PresentationError(f"bad vision block {kind!r} x {num_tokens}")
    vision = vision_token_ids()
    pad = vision[IMAGE_PAD if kind == "image" else VIDEO_PAD]
    return [vision[VISION_START], *([pad] * num_tokens), vision[VISION_END]]


def _pad_runs(token_ids: Sequence[int]) -> list[tuple[str, int]]:
    """``(kind, length)`` of every bracketed vision block, checking the brackets."""

    vision = vision_token_ids()
    start, end = vision[VISION_START], vision[VISION_END]
    pads = {vision[IMAGE_PAD]: "image", vision[VIDEO_PAD]: "video"}
    runs: list[tuple[str, int]] = []
    index, n = 0, len(token_ids)
    while index < n:
        token = token_ids[index]
        if token == start:
            kind: str | None = None
            length = 0
            index += 1
            while index < n and token_ids[index] in pads:
                this = pads[token_ids[index]]
                if kind is not None and this != kind:
                    raise PresentationError(f"vision block mixes {kind} and {this} pads")
                kind = this
                length += 1
                index += 1
            if kind is None or index >= n or token_ids[index] != end:
                raise PresentationError(f"malformed vision block at position {index}")
            runs.append((kind, length))
        elif token == end or token in pads:
            raise PresentationError(f"vision token {token} outside a vision block at {index}")
        index += 1
    return runs


def check_presentation(
    token_ids: Sequence[int],
    vision_inputs: Mapping[str, torch.Tensor] | None = None,
    *,
    merge_size: int = 2,
) -> None:
    """Check every vision block against the processor's grids before the model runs.

    An image block holds ``prod(image_grid_thw[i]) / merge_size**2`` pads. A video contributes
    ``t`` blocks of ``h * w / merge_size**2`` pads each (one per merged frame pair), in order.
    """

    runs = _pad_runs(token_ids)
    inputs = dict(vision_inputs or {})
    merge = merge_size * merge_size
    expected: dict[str, list[int]] = {"image": [], "video": []}
    for grid in inputs.get("image_grid_thw", torch.empty(0, 3, dtype=torch.long)).tolist():
        expected["image"].append(int(grid[0]) * int(grid[1]) * int(grid[2]) // merge)
    for grid in inputs.get("video_grid_thw", torch.empty(0, 3, dtype=torch.long)).tolist():
        expected["video"].extend([int(grid[1]) * int(grid[2]) // merge] * int(grid[0]))
    for kind, pixel_key in (("image", "pixel_values"), ("video", "pixel_values_videos")):
        got = [length for this, length in runs if this == kind]
        if got != expected[kind]:
            raise PresentationError(
                f"{kind} blocks hold {got} pads, processor grids give {expected[kind]}"
            )
        if expected[kind] and pixel_key not in inputs:
            raise PresentationError(f"{kind} blocks present but {pixel_key} missing")
        if not expected[kind] and pixel_key in inputs:
            raise PresentationError(f"{pixel_key} given but the presentation has no {kind} block")


def verify_release_processor(processor: Any) -> None:
    """Fail unless the image/video processor geometry matches the pinned release."""

    pinned = release_fingerprint()["processor"]
    for part in ("image_processor", "video_processor"):
        component = getattr(processor, part)
        for key, value in pinned[part].items():
            got = getattr(component, key)
            if key == "size":
                items = got.items() if isinstance(got, Mapping) else vars(got).items()
                got = {k: int(v) for k, v in items if v is not None}
            if isinstance(got, (tuple, list)):
                got = [float(x) for x in got]
                value = [float(x) for x in value]
            if got != value:
                raise ValueError(f"{part}.{key} = {got!r}, pinned {value!r}")


# --- runtime policy ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimePolicy:
    """Numeric flags the feature bytes depend on. The release values are PyTorch's defaults."""

    matmul_allow_tf32: bool = False
    matmul_allow_bf16_reduced_precision_reduction: bool = True
    matmul_allow_fp16_reduced_precision_reduction: bool = True
    cudnn_allow_tf32: bool = True
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False
    # None keeps PyTorch's SDPA backend priority (what the release runs); a tuple of
    # ``torch.nn.attention.SDPBackend`` names restricts it.
    sdpa_backends: tuple[str, ...] | None = None


RELEASE_POLICY = RuntimePolicy()


def runtime_readback() -> dict[str, Any]:
    """The current values of every flag :class:`RuntimePolicy` pins."""

    matmul = torch.backends.cuda.matmul
    return {
        "matmul_allow_tf32": bool(matmul.allow_tf32),
        "matmul_allow_bf16_reduced_precision_reduction": bool(
            matmul.allow_bf16_reduced_precision_reduction
        ),
        "matmul_allow_fp16_reduced_precision_reduction": bool(
            matmul.allow_fp16_reduced_precision_reduction
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "sdpa_flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdpa_mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdpa_math": bool(torch.backends.cuda.math_sdp_enabled()),
        "sdpa_cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
    }


@contextlib.contextmanager
def pinned_runtime(policy: RuntimePolicy = RELEASE_POLICY) -> Iterator[dict[str, Any]]:
    """Set the policy flags for the duration of the block, restore them after, yield a readback."""

    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    saved = (
        matmul.allow_tf32,
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_fp16_reduced_precision_reduction,
        cudnn.allow_tf32,
        cudnn.deterministic,
        cudnn.benchmark,
    )
    matmul.allow_tf32 = policy.matmul_allow_tf32
    matmul.allow_bf16_reduced_precision_reduction = (
        policy.matmul_allow_bf16_reduced_precision_reduction
    )
    matmul.allow_fp16_reduced_precision_reduction = (
        policy.matmul_allow_fp16_reduced_precision_reduction
    )
    cudnn.allow_tf32 = policy.cudnn_allow_tf32
    cudnn.deterministic = policy.cudnn_deterministic
    cudnn.benchmark = policy.cudnn_benchmark
    try:
        with contextlib.ExitStack() as stack:
            if policy.sdpa_backends is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                stack.enter_context(
                    sdpa_kernel([getattr(SDPBackend, b) for b in policy.sdpa_backends])
                )
            yield runtime_readback()
    finally:
        (
            matmul.allow_tf32,
            matmul.allow_bf16_reduced_precision_reduction,
            matmul.allow_fp16_reduced_precision_reduction,
            cudnn.allow_tf32,
            cudnn.deterministic,
            cudnn.benchmark,
        ) = saved


# --- layer-50 extraction ----------------------------------------------------------------------


class _LayerReached(Exception):
    """Raised by the capture hook to stop the forward after the target layer."""

    def __init__(self, hidden: torch.Tensor) -> None:
        super().__init__("target layer reached")
        self.hidden = hidden


@dataclass
class Layer50Output:
    """One presentation's conditioning feature and its provenance."""

    feature: torch.Tensor
    token_ids: list[int]
    feature_digest: str
    token_ids_digest: str
    runtime: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "num_tokens": len(self.token_ids),
            "shape": list(self.feature.shape),
            "dtype": str(self.feature.dtype).removeprefix("torch."),
            "feature_sha256": self.feature_digest,
            "token_ids_sha256": self.token_ids_digest,
        }


def token_ids_digest(token_ids: Sequence[int]) -> str:
    """sha256 of the int64 little-endian token ids (with their count in the header)."""

    return tensor_digest(torch.tensor(list(token_ids), dtype=torch.int64))


def _vl_model(text_encoder: Any) -> Any:
    """The ``Qwen3VLModel`` inside a ``Qwen3VLForConditionalGeneration`` (or the model itself)."""

    inner = getattr(text_encoder, "model", text_encoder)
    if not hasattr(inner, "language_model"):
        raise TypeError(f"{type(text_encoder).__name__} is not a Qwen3-VL model")
    return inner


class Qwen3VLLayer50Encoder:
    """Run the HF Qwen3-VL forward one presentation at a time and stop at ``hidden_states[50]``.

    ``text_encoder`` is a ``Qwen3VLForConditionalGeneration`` or ``Qwen3VLModel`` with at least
    ``layer`` decoder layers. A stack truncated to exactly ``layer`` layers works too, because the
    feature is captured before the final norm.
    """

    def __init__(
        self,
        text_encoder: Any,
        *,
        layer: int = TEXT_ENCODER_LAYER,
        policy: RuntimePolicy = RELEASE_POLICY,
        check_release: bool = True,
    ) -> None:
        self.text_encoder = text_encoder
        self.model = _vl_model(text_encoder)
        self.layer = int(layer)
        self.policy = policy
        layers = self.model.language_model.layers
        if not 1 <= self.layer <= len(layers):
            raise ValueError(
                f"hidden_states[{self.layer}] needs >= {self.layer} decoder layers, "
                f"model has {len(layers)}"
            )
        self.target = layers[self.layer - 1]
        if check_release:
            self._check_release_config()
        if (
            getattr(text_encoder, "_hf_hook", None) is not None
            or getattr(self.model, "_hf_hook", None) is not None
        ):
            raise RuntimeError(
                "an offload hook is attached; move the encoder to one device instead"
            )

    def _check_release_config(self) -> None:
        pinned = release_fingerprint()
        text = self.model.config.text_config
        for key, value in pinned["text_config"].items():
            got = getattr(text, key)
            if key == "num_hidden_layers":
                if not self.layer <= int(got) <= int(value):
                    raise ValueError(
                        f"text_config.num_hidden_layers={got} outside [{self.layer}, {value}]"
                    )
                continue
            if got != value:
                raise ValueError(f"text_config.{key}={got!r}, pinned {value!r}")
        rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None)
        rope = {
            k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(rope or {}).items()
        }
        if rope != dict(pinned["rope"]):
            raise ValueError(f"text_config rope {rope!r}, pinned {pinned['rope']!r}")
        vision = self.model.config.vision_config
        for key, value in pinned["vision_config"].items():
            got = getattr(vision, key)
            if isinstance(got, tuple):
                got = list(got)
            if got != value:
                raise ValueError(f"vision_config.{key}={got!r}, pinned {value!r}")
        # Qwen3VLModel scatters the vision features onto the tokens equal to these config ids,
        # while the presentation checks use the pinned token table: the two must agree.
        tokens = vision_token_ids()
        for key, name in (
            ("image_token_id", IMAGE_PAD),
            ("video_token_id", VIDEO_PAD),
            ("vision_start_token_id", VISION_START),
            ("vision_end_token_id", VISION_END),
        ):
            got = getattr(self.model.config, key, None)
            if got != tokens[name]:
                raise ValueError(f"{key}={got!r}, pinned {tokens[name]} ({name})")

    @property
    def device(self) -> torch.device:
        return self.model.language_model.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.language_model.embed_tokens.weight.dtype

    def _forward(
        self, token_ids: Sequence[int], vision_inputs: Mapping[str, torch.Tensor] | None
    ) -> torch.Tensor:
        device = self.device
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        kwargs: dict[str, Any] = {}
        for name, value in (vision_inputs or {}).items():
            kwargs[name] = (
                value.to(device, self.dtype) if name.startswith("pixel_") else value.to(device)
            )

        def capture(module: Any, args: Any, output: Any) -> None:
            raise _LayerReached(output[0] if isinstance(output, tuple) else output)

        handle = self.target.register_forward_hook(capture)
        try:
            self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=torch.tensor(
                    [mm_token_type_ids(token_ids)], dtype=torch.long, device=device
                ),
                use_cache=False,
                **kwargs,
            )
        except _LayerReached as reached:
            return reached.hidden
        finally:
            handle.remove()
        raise RuntimeError(f"decoder layer {self.layer - 1} never ran")

    @torch.no_grad()
    def encode(
        self,
        token_ids: Sequence[int],
        vision_inputs: Mapping[str, torch.Tensor] | None = None,
    ) -> Layer50Output:
        """The ``(1, len(token_ids), 5120)`` feature of one presentation, in the encoder dtype."""

        ids = [int(i) for i in token_ids]
        if not ids:
            raise PresentationError("empty presentation")
        merge = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
        check_presentation(ids, vision_inputs, merge_size=merge)
        with pinned_runtime(self.policy) as readback:
            hidden = self._forward(ids, vision_inputs)
        width = int(self.model.config.text_config.hidden_size)
        if tuple(hidden.shape) != (1, len(ids), width):
            raise RuntimeError(f"layer output {tuple(hidden.shape)} != (1, {len(ids)}, {width})")
        return Layer50Output(
            feature=hidden,
            token_ids=ids,
            feature_digest=tensor_digest(hidden),
            token_ids_digest=token_ids_digest(ids),
            runtime=readback,
        )

    def encode_many(
        self,
        presentations: Sequence[
            Sequence[int] | tuple[Sequence[int], Mapping[str, torch.Tensor] | None]
        ],
    ) -> list[Layer50Output]:
        """Encode several presentations, one forward each, so no row sees its neighbours."""

        outputs = []
        for item in presentations:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and (item[1] is None or isinstance(item[1], Mapping))
            ):
                ids, vision = item
            else:
                ids, vision = item, None
            outputs.append(self.encode(ids, vision))  # type: ignore[arg-type]
        return outputs


# --- loading ----------------------------------------------------------------------------------


def restore_vision_rope(config: Any) -> None:
    """Give ``config.vision_config`` the RoPE parameters transformers itself derives.

    The released ``vision_config`` has no RoPE fields. transformers' ``standardize_rope_params``
    fills in ``rope_theta`` and maps ``rope_type`` ``"default"`` to the class's
    ``default_rope_type`` (``"axial"``, the only type the vision rotary embedding accepts).
    SGLang 0.5.21 patches that method on import to return early for configs without
    ``max_position_embeddings``, which includes the vision config, so after any SGLang import
    the type stays ``"default"`` and building the model fails. This repeats transformers'
    mapping; without the patch the parameters are already standardized and nothing changes.
    """

    vision = getattr(config, "vision_config", None)
    axial = getattr(type(vision), "default_rope_type", "default")
    if vision is None or axial == "default":
        return
    current = getattr(vision, "rope_parameters", None)
    rope = dict(current or {})
    rope.setdefault("rope_theta", getattr(vision, "rope_theta", None) or vision.default_theta)
    rope.setdefault("rope_type", rope.get("type", "default"))
    if rope["rope_type"] == "default":
        rope["rope_type"] = axial
    if rope != current:
        vision.rope_parameters = rope


def load_release_text_encoder(
    path: str | Path,
    *,
    device: str | torch.device = "cuda",
    num_layers: int = TEXT_ENCODER_LAYER,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, dict[str, Any]]:
    """Load ``text_encoder/`` with only ``num_layers`` decoder layers, through ``from_pretrained``.

    The weights load on CPU and move to ``device`` (no ``accelerate`` needed), as the Diffusers
    pipeline does. Fails unless every parameter of the truncated stack came from the checkpoint and
    every skipped key belongs to a layer >= ``num_layers``. Returns the model and a load report.
    """

    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(str(path))
    restore_vision_rope(config)
    full_layers = int(config.text_config.num_hidden_layers)
    if not TEXT_ENCODER_LAYER <= num_layers <= full_layers:
        raise ValueError(f"num_layers={num_layers} outside [{TEXT_ENCODER_LAYER}, {full_layers}]")
    config.text_config.num_hidden_layers = num_layers
    loaded: Any = Qwen3VLForConditionalGeneration.from_pretrained(
        str(path), config=config, dtype=dtype, output_loading_info=True
    )
    model: Any = loaded[0]
    info: dict[str, Any] = dict(loaded[1])
    missing = sorted(info.get("missing_keys", []))
    unexpected = sorted(info.get("unexpected_keys", []))
    mismatched = sorted(str(k) for k in info.get("mismatched_keys", []))
    stray = []
    for key in unexpected:
        match = _LAYER_KEY.search(key)
        if match is None or int(match.group(1)) < num_layers:
            stray.append(key)
    if missing or mismatched or stray:
        raise ValueError(
            f"load mismatch: missing={missing[:5]} mismatched={mismatched[:5]} "
            f"unexpected={stray[:5]}"
        )
    model = model.to(device).eval()
    report = {
        "num_layers": num_layers,
        "checkpoint_layers": full_layers,
        "skipped_keys": len(unexpected),
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
    }
    return model, report


def contract_manifest() -> dict[str, Any]:
    """Machine-readable description of what this contract pins (for evidence files)."""

    pinned = release_fingerprint()
    return {
        "contract_id": CONTRACT_ID,
        "model_id": pinned["model_id"],
        "revision": pinned["revision"],
        "reference": pinned["reference"],
        "text_encoder_layer": TEXT_ENCODER_LAYER,
        "feature_width": FEATURE_WIDTH,
        "tags": {"video": VIDEO_TAG, "text": TEXT_TAG, "audio": AUDIO_TAG},
        "special_tokens": special_token_ids(),
        "runtime_policy": asdict(RELEASE_POLICY),
    }
