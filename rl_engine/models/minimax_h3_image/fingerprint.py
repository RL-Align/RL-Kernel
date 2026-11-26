# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3-Image architecture fingerprint (RFC #738 ``h3_image_arch_fingerprint``).

One machine-readable description of what "the H3-Image graph" means to RL-Kernel:

* the pinned checkpoint (repo, revision, every component directory, file hashes) and the
  pinned Diffusers commit whose H3-Image code is the reference implementation;
* the inherited Qwen3-VL text encoder, checked file by file against the MiniMax-H3 revision
  that RFC #420 pins;
* the transformer configuration in both released layouts (upstream ``MiniMaxH3DiTModel`` and
  Diffusers ``MiniMaxH3ImageTransformer2DModel``) and the quantities derived from it;
* the six Parallel Decoding Distillation (PDD) output heads: shape, dtype, and per-head
  sha256 in both layouts, plus proof that the upstream layout's single
  ``final_layer.video_out`` is none of them (strict mode must not run it);
* the fixed seven-point sigma grid, the float32 round trip the released sampler applies to
  it, and the velocity-form Euler update;
* the VAE configuration facts the pipeline depends on (latent statistics, tiling defaults,
  pixel normalization);
* the executable transformer graph, each node tagged with the RFC and the row that owns
  its arithmetic. Nodes owned by RFC #420 rows are reused H3 operators; nodes owned by
  RFC #738 rows are image-specific.

The module needs neither torch nor a GPU. Everything is derived from
``pinned_checkpoint.json`` (written by ``tools/weights/snapshot_h3_image_checkpoint.py``
from the real Hub files) plus the analytic constants below, and the two are cross-checked.
Any mismatch raises; nothing is normalised away.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from rl_engine.models.minimax_h3 import fingerprint as h3_fp

SCHEMA_VERSION = "rlkernel.h3_image.arch_fingerprint.v1"
PINNED_CHECKPOINT_PATH = Path(__file__).with_name("pinned_checkpoint.json")

REPO = "MiniMaxAI/MiniMax-H3-Image"
REVISION = "8ced60068623def5b74b4d10558117cb61d6482d"
ARCHITECTURE = "MiniMaxH3ImageTransformer2DModel"
UPSTREAM_ARCHITECTURE = "MiniMaxH3DiTModel"
VAE_ARCHITECTURE = "AutoencoderKLMiniMaxH3Image"
UPSTREAM_VAE_ARCHITECTURE = "MiniMaxH3ImageVAE"

# The Diffusers commit that added MiniMax-H3-Image (huggingface/diffusers#15230). Its
# transformer, VAE, scheduler and modular pipeline are the reference implementation.
DIFFUSERS_COMMIT = "875373ac7f5856071e345a3e394c74efccc6b9e6"

# The text encoder is inherited from MiniMax-H3 and pinned at the RFC #420 revision.
TEXT_ENCODER_REPO = h3_fp.REPO
TEXT_ENCODER_REVISION = h3_fp.REVISION
TEXT_ENCODER_LAYER = 50

# Directories of the released repo. ``transformer``/``vae`` are the upstream layout;
# ``*_diffusers`` are the layout the Diffusers modular pipeline loads.
DIFFUSERS_TRANSFORMER_DIR = "transformer_diffusers"
UPSTREAM_TRANSFORMER_DIR = "transformer"
DIFFUSERS_VAE_DIR = "vae_diffusers"
UPSTREAM_VAE_DIR = "vae"

# PDD heads.
PDD_FILE = "pdd_fused_heads.safetensors"
PDD_WEIGHT = "video_out.weight"
PDD_BIAS = "video_out.bias"
NUM_PDD_HEADS = 6
UPSTREAM_SHARED_HEAD = "final_layer.video_out"

# AdaLN modality tags (row = timestep_index * 3 + tag). The audio slot is unused by image
# requests but stays in the table: the checkpoint's AdaLN projections are three rows wide.
MODALITY_TAGS: Mapping[str, int] = {"image": 0, "text": 1, "audio_unused": 2}
ADALN_MODALITIES = 3

# Declared mixed precision of the Diffusers layout. Everything not listed as FP32 is BF16.
FP32_MODULES = ("proj_in", "time_embedder", "pdd_proj_out")
FP32_COMPUTE = ("time_proj", "rope")  # no parameters; frequencies/features built in FP32
_SAFETENSORS_DTYPE = {"fp32": "F32", "bf16": "BF16"}

# Edit references are noised to this t and held there for every step.
REFERENCE_TIMESTEP = 0.999

# Pipeline preprocessing constants (``MiniMaxH3ImageModularPipeline``).
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
MAX_SEQUENCE_LENGTH = 4096
VAE_SPATIAL_COMPRESSION = 16
CANVAS_MULTIPLE = 32  # VAE compression 16 x transformer spatial patch 2

SCHEDULER: Mapping[str, Any] = {
    "family": "pdd_fixed_grid_euler",
    "class": "MiniMaxH3Scheduler",
    "eta": 0.0,
    "shift_provided_sigmas": False,
    "euler_update_form": "velocity",
    "timestep": "t_i = float32(1 - sigma_i), t = 1 is clean",
    "step_sigma": "sigma_eff_i = float32(1 - t_i): the released round trip, one ulp below 0.5",
    "velocity": "data-ward, x0 = x_t + sigma * v",
    "euler_update": "x_next = x_t + (sigma_eff_i - sigma_eff_(i+1)) * v_i, in float32",
    "pdd_head": "head i for loop step i; never chosen from the timestep value",
    "terminal_sigma": "sigma = 0 ends the grid and has no model evaluation",
    "reference_timestep": REFERENCE_TIMESTEP,
}

# --- Graph vocabulary -------------------------------------------------------------------------

RFC_H3 = "420"
RFC_IMAGE = "738"

# Stem nodes in execution order: (path, rfc, owning row, precision).
STEM_NODES: tuple[tuple[str, str, str, str], ...] = (
    ("image_patch_pack", RFC_IMAGE, "image_patch_pack_unpack", "fp32"),
    ("proj_in", RFC_H3, "video_input_projection_fp32", "fp32"),
    ("context_embedder", RFC_H3, "text_context_projection", "bf16"),
    ("time_proj", RFC_H3, "timestep_sinusoid_h3", "fp32"),
    ("time_embedder", RFC_H3, "timestep_mlp_fp32", "fp32"),
    ("position_grid", RFC_IMAGE, "image_position_grid", "fp64"),
    ("rope", RFC_H3, "h3_mm_rope_3axis_partial", "fp32"),
)
PACK_NODE = ("packed_index_copy", RFC_IMAGE, "image_packed_layout_schema", "bf16")
TAIL_NODES: tuple[tuple[str, str, str, str], ...] = (
    ("norm_out", RFC_H3, "final_adaln_out", "bf16"),
    ("pdd_proj_out", RFC_IMAGE, "h3_image_pdd_head_contract", "fp32"),
    ("packed_index_select", RFC_IMAGE, "image_transformer_output_gather", "fp32"),
)
REFINER_FINAL_NODE = ("token_refiner.final_norm", RFC_H3, "h3_rmsnorm", "bf16")
PRECISIONS = ("fp64", "fp32", "bf16")


@dataclass(frozen=True)
class GraphNode:
    """One node of the strict H3-Image transformer graph."""

    path: str  # unique, e.g. ``transformer_blocks.7.attention``
    stage: str  # stem | token_refiner | transformer_blocks | output
    layer: int | None  # block index within its stack, or None
    node: str  # node name inside the block (``attention``) or the module name
    rfc: str  # "420" (reused H3 row) or "738" (image-specific row)
    row: str  # row owning the arithmetic in that RFC
    precision: str  # "fp64", "fp32" or "bf16": the declared compute precision at the node

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "stage": self.stage,
            "layer": self.layer,
            "node": self.node,
            "rfc": self.rfc,
            "row": self.row,
            "precision": self.precision,
        }


# --- Pinned checkpoint -----------------------------------------------------------------------


@lru_cache(maxsize=1)
def pinned_checkpoint() -> Mapping[str, Any]:
    """Load the pinned Hub snapshot (file hashes, configs, tensor-table digests, head hashes)."""

    data = json.loads(PINNED_CHECKPOINT_PATH.read_text())
    if data["repo"] != REPO or data["revision"] != REVISION:
        raise ValueError("pinned_checkpoint.json does not match the pinned repo/revision")
    inherited = data["inherited_text_encoder"]
    if inherited["repo"] != TEXT_ENCODER_REPO or inherited["revision"] != TEXT_ENCODER_REVISION:
        raise ValueError("pinned_checkpoint.json pins a different text-encoder revision")
    return data


def _public(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def transformer_config() -> dict[str, Any]:
    """Public (underscore-free) config of the Diffusers-layout transformer."""

    return _public(pinned_checkpoint()["configs"][DIFFUSERS_TRANSFORMER_DIR])


def upstream_transformer_config() -> dict[str, Any]:
    """Public config of the upstream-layout ``MiniMaxH3DiTModel``."""

    return _public(pinned_checkpoint()["configs"][UPSTREAM_TRANSFORMER_DIR])


def derived_quantities(config: Mapping[str, Any]) -> dict[str, Any]:
    """Quantities the strict operators depend on, derived from the Diffusers config."""

    heads, head_dim, hidden = (
        config["num_attention_heads"],
        config["attention_head_dim"],
        config["hidden_size"],
    )
    patch_t, patch_h, patch_w = config["patch_size"]
    return {
        "attention_inner_dim": heads * head_dim,
        "attention_inner_dim_differs_from_hidden": heads * head_dim != hidden,
        "rotary_channels": 6 * config["rope_freq_dim"],
        "rotary_pass_through_channels": head_dim - 6 * config["rope_freq_dim"],
        "image_patch_width": config["in_channels"] * patch_t * patch_h * patch_w,
        "adaln_modalities": ADALN_MODALITIES,
        "adaln_projection_out": 6 * ADALN_MODALITIES * hidden,
        "final_adaln_out": 2 * hidden,
        "ffn_gate_up_out": 2 * config["ffn_dim"],
        "num_pdd_heads": config["num_pdd_heads"],
        "canvas_multiple": VAE_SPATIAL_COMPRESSION * patch_h,
    }


def _check_layouts_agree(diffusers: Mapping[str, Any], upstream: Mapping[str, Any]) -> None:
    """The two released transformer configs must describe one architecture."""

    pairs = (
        ("hidden_size", "hidden_size"),
        ("num_layers", "num_layers"),
        ("num_refiner_layers", "token_refiner_num_layers"),
        ("num_attention_heads", "num_attention_heads"),
        ("attention_head_dim", "attention_head_dim"),
        ("ffn_dim", "ffn_hidden_size"),
        ("in_channels", "latents_dim"),
        ("patch_size", "patch_size"),
        ("text_dim", "text_dim"),
        ("freq_dim", "timestep_input_dim"),
        ("time_embed_hidden_dim", "time_embed_hidden_size"),
        ("time_embed_dim", "time_embed_dim"),
        ("rope_freq_dim", "rope_inv_freq_len"),
        ("norm_eps", "norm_eps"),
        ("qk_norm_eps", "qk_norm_eps"),
        ("final_norm_eps", "final_norm_eps"),
    )
    problems = [
        f"{d}={diffusers[d]!r} != upstream {u}={upstream[u]!r}"
        for d, u in pairs
        if diffusers[d] != upstream[u]
    ]
    derived = derived_quantities(diffusers)
    if upstream["adaln_out_features"] != derived["adaln_projection_out"]:
        problems.append("upstream adaln_out_features != 6 * 3 * hidden_size")
    if upstream["final_adaln_out_features"] != derived["final_adaln_out"]:
        problems.append("upstream final_adaln_out_features != 2 * hidden_size")
    if upstream.get("audio_enabled") is not False:
        problems.append("upstream audio_enabled must be false for the image checkpoint")
    if problems:
        raise ValueError("transformer layouts disagree:\n  " + "\n  ".join(problems))


# --- Tensor tables ----------------------------------------------------------------------------


def _precision_of(name: str) -> str:
    top = name.split(".", 1)[0]
    return "fp32" if top in FP32_MODULES else "bf16"


def expected_tensor_table(config: Mapping[str, Any]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Analytic ``name -> (safetensors dtype, shape)`` of the Diffusers-layout checkpoint.

    It is the MiniMax-H3 table (RFC #420) without the audio projections and without the
    single ``proj_out``, plus one FP32 ``pdd_proj_out.<i>`` head per distilled step.
    """

    hidden = config["hidden_size"]
    patch_t, patch_h, patch_w = config["patch_size"]
    width = config["in_channels"] * patch_t * patch_h * patch_w
    shapes: dict[str, tuple[int, ...]] = {
        "proj_in.weight": (hidden, width),
        "proj_in.bias": (hidden,),
        "context_embedder.weight": (hidden, config["text_dim"]),
        "context_embedder.bias": (hidden,),
        "time_embedder.linear_1.weight": (config["time_embed_hidden_dim"], config["freq_dim"]),
        "time_embedder.linear_1.bias": (config["time_embed_hidden_dim"],),
        "time_embedder.linear_2.weight": (
            config["time_embed_dim"],
            config["time_embed_hidden_dim"],
        ),
        "time_embedder.linear_2.bias": (config["time_embed_dim"],),
        "norm_out.norm.weight": (hidden,),
        "norm_out.linear.weight": (2 * hidden, config["time_embed_dim"]),
        "norm_out.linear.bias": (2 * hidden,),
        "token_refiner.final_norm.weight": (hidden,),
    }
    for index in range(config["num_pdd_heads"]):
        shapes[f"pdd_proj_out.{index}.weight"] = (width, hidden)
        shapes[f"pdd_proj_out.{index}.bias"] = (width,)
    for index in range(config["num_refiner_layers"]):
        prefix = f"token_refiner.refiner_blocks.{index}"
        shapes[f"{prefix}.norm1.weight"] = (hidden,)
        shapes.update(h3_fp._attention_tensors(prefix, config))
    for index in range(config["num_layers"]):
        prefix = f"transformer_blocks.{index}"
        shapes[f"{prefix}.norm1.weight"] = (hidden,)
        shapes.update(h3_fp._attention_tensors(prefix, config))
        shapes[f"{prefix}.adaln_proj.linear.weight"] = (
            6 * ADALN_MODALITIES * hidden,
            config["time_embed_dim"],
        )
        shapes[f"{prefix}.adaln_proj.linear.bias"] = (6 * ADALN_MODALITIES * hidden,)
    return {
        name: (_SAFETENSORS_DTYPE[_precision_of(name)], shape) for name, shape in shapes.items()
    }


def expected_pdd_table(config: Mapping[str, Any]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Analytic table of ``pdd_fused_heads.safetensors``: all heads stacked, FP32."""

    patch_t, patch_h, patch_w = config["patch_size"]
    width = config["in_channels"] * patch_t * patch_h * patch_w
    heads, hidden = config["num_pdd_heads"], config["hidden_size"]
    return {PDD_WEIGHT: ("F32", (heads, width, hidden)), PDD_BIAS: ("F32", (heads, width))}


tensor_table_digest = h3_fp.tensor_table_digest
parameter_count = h3_fp.parameter_count


# --- PDD heads ---------------------------------------------------------------------------------


def pdd_head_digests() -> list[dict[str, str]]:
    """Per-head sha256 of the fused PDD file, after checking the Diffusers copy is identical."""

    heads = pinned_checkpoint()["head_sha256"]
    fused, diffusers = heads["pdd_fused"], heads["diffusers_pdd_proj_out"]
    if len(fused) != NUM_PDD_HEADS or len(diffusers) != NUM_PDD_HEADS:
        raise ValueError(f"expected {NUM_PDD_HEADS} PDD heads in both layouts")
    for index, (a, b) in enumerate(zip(fused, diffusers)):
        if a != b:
            raise ValueError(f"PDD head {index}: pdd_fused_heads and pdd_proj_out.{index} differ")
    if len({(h["weight"], h["bias"]) for h in fused}) != NUM_PDD_HEADS:
        raise ValueError("PDD heads are not pairwise distinct")
    shared = heads["upstream_video_out"]
    for head in fused:
        if head["weight"] == shared["weight"] or head["bias"] == shared["bias"]:
            raise ValueError(f"{UPSTREAM_SHARED_HEAD} coincides with a PDD head")
    return [dict(head) for head in fused]


# --- Sigma grid --------------------------------------------------------------------------------


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def sigma_grid() -> dict[str, list[float]]:
    """The released grid, the float32 timesteps and the sigmas the sampler actually steps on.

    ``MiniMaxH3ImageSetTimestepsStep`` builds ``t = float32(1 - sigma)`` from the config
    values, then hands ``float32(1 - t)`` to the scheduler. Both operations are exact
    float32 rounding of an exact double difference, so they are reproduced here without torch.
    """

    snapshot = pinned_checkpoint()
    config = [float(s) for s in snapshot["configs"]["pdd_config"]["sigmas"]]
    if [float(s) for s in snapshot["configs"]["modular_model_index"]["sample_sigmas"]] != config:
        raise ValueError("pdd_config.json and modular_model_index.json sigma grids differ")
    if snapshot["configs"]["pdd_config"]["num_inference_steps"] != NUM_PDD_HEADS:
        raise ValueError("pdd_config num_inference_steps != number of PDD heads")
    if len(config) != NUM_PDD_HEADS + 1 or config[0] != 1.0 or config[-1] != 0.0:
        raise ValueError("sigma grid must run from 1.0 to 0.0 over num_heads + 1 points")
    if any(_f32(s) != s for s in config):
        raise ValueError("sigma grid values are not float32-representable")
    if any(b >= a for a, b in zip(config, config[1:])):
        raise ValueError("sigma grid must be strictly decreasing")
    timesteps = [_f32(1.0 - s) for s in config[:-1]]
    step = [_f32(1.0 - t) for t in timesteps] + [0.0]
    scheduler_timesteps = [_f32(1.0 - s) for s in step[:-1]]
    if scheduler_timesteps != timesteps:
        raise ValueError("scheduler timesteps do not survive the sigma round trip")
    return {
        "config_sigmas": config,
        "timesteps": timesteps,
        "step_sigmas": step,
        "round_trip_moved": [i for i, (a, b) in enumerate(zip(config, step)) if a != b],
    }


def _check_scheduler() -> None:
    raw = pinned_checkpoint()["configs"]["scheduler"]
    for key in ("shift_provided_sigmas", "euler_update_form"):
        if raw.get(key) != SCHEDULER[key]:
            raise ValueError(f"scheduler_config {key}={raw.get(key)!r} != {SCHEDULER[key]!r}")
    if raw.get("_class_name") != SCHEDULER["class"]:
        raise ValueError("scheduler class differs from the pinned one")


# --- VAE ---------------------------------------------------------------------------------------


def vae_facts() -> dict[str, Any]:
    """Pipeline-relevant VAE facts, cross-checked between the two released layouts."""

    configs = pinned_checkpoint()["configs"]
    diffusers, upstream = configs[DIFFUSERS_VAE_DIR], configs[UPSTREAM_VAE_DIR]
    problems = []
    if diffusers["_class_name"] != VAE_ARCHITECTURE:
        problems.append(f"Diffusers VAE class {diffusers['_class_name']}")
    if upstream["_class_name"] != UPSTREAM_VAE_ARCHITECTURE:
        problems.append(f"upstream VAE class {upstream['_class_name']}")
    for key in ("latent_channels", "latents_mean", "latents_std"):
        if diffusers[key] != upstream[key]:
            problems.append(f"{key} differs between layouts")
    channels = diffusers["latent_channels"]
    if len(diffusers["latents_mean"]) != channels or len(diffusers["latents_std"]) != channels:
        problems.append("latent statistics do not have latent_channels entries")
    if channels != transformer_config()["in_channels"]:
        problems.append("VAE latent_channels != transformer in_channels")
    if any(_f32(v) != v for v in diffusers["latents_mean"] + diffusers["latents_std"]):
        problems.append("latent statistics are not float32-representable")
    spatial = 1
    for factor in diffusers["spatial_downsample_factors"]:
        spatial *= factor
    if spatial != VAE_SPATIAL_COMPRESSION:
        problems.append(f"spatial compression {spatial} != {VAE_SPATIAL_COMPRESSION}")
    if not (upstream["vae_encoder_tiling"] and upstream["vae_decoder_tiling"]):
        problems.append("released tiling is expected on for encode and decode")
    if problems:
        raise ValueError("VAE configs disagree:\n  " + "\n  ".join(problems))
    stats = hashlib.sha256(
        json.dumps([diffusers["latents_mean"], diffusers["latents_std"]]).encode()
    ).hexdigest()
    return {
        "class": VAE_ARCHITECTURE,
        "upstream_class": UPSTREAM_VAE_ARCHITECTURE,
        "latent_channels": channels,
        "spatial_compression": spatial,
        "latent_stats_sha256": stats,
        "tile_size": upstream["vae_tile_size"],
        "tile_overlap_min": upstream["vae_tile_overlap_min"],
        "encoder_tiling": bool(upstream["vae_encoder_tiling"]),
        "decoder_tiling": bool(upstream["vae_decoder_tiling"]),
        "decoder": {
            "layers": diffusers["decoder_num_layers"],
            "heads": diffusers["decoder_num_attention_heads"],
            "head_dim": diffusers["decoder_attention_head_dim"],
            "register_tokens": diffusers["decoder_num_register_tokens"],
            "rope_theta": diffusers["decoder_rope_theta"],
            "rope_dim_ratio": diffusers["decoder_rope_dim_ratio"],
        },
        "pixel_mean": list(PIXEL_MEAN),
        "pixel_std": list(PIXEL_STD),
    }


# --- Graph -------------------------------------------------------------------------------------


def build_graph(config: Mapping[str, Any] | None = None) -> tuple[GraphNode, ...]:
    """Return the full strict transformer graph in execution order."""

    config = dict(config or transformer_config())
    nodes: list[GraphNode] = []
    for path, rfc, row, precision in STEM_NODES:
        nodes.append(GraphNode(path, "stem", None, path, rfc, row, precision))
    for index in range(config["num_refiner_layers"]):
        for name, row in h3_fp.REFINER_NODES:
            nodes.append(
                GraphNode(
                    f"token_refiner.refiner_blocks.{index}.{name}",
                    "token_refiner",
                    index,
                    name,
                    RFC_H3,
                    row,
                    "bf16",
                )
            )
    path, rfc, row, precision = REFINER_FINAL_NODE
    nodes.append(GraphNode(path, "token_refiner", None, "final_norm", rfc, row, precision))
    path, rfc, row, precision = PACK_NODE
    nodes.append(GraphNode(path, "stem", None, path, rfc, row, precision))
    for index in range(config["num_layers"]):
        for name, row in h3_fp.BLOCK_NODES:
            nodes.append(
                GraphNode(
                    f"transformer_blocks.{index}.{name}",
                    "transformer_blocks",
                    index,
                    name,
                    RFC_H3,
                    row,
                    "bf16",
                )
            )
    for path, rfc, row, precision in TAIL_NODES:
        nodes.append(GraphNode(path, "output", None, path, rfc, row, precision))
    paths = [node.path for node in nodes]
    if len(set(paths)) != len(paths):
        raise ValueError("graph paths must be unique")
    return tuple(nodes)


def graph_digest(graph: Iterable[GraphNode]) -> str:
    payload = json.dumps([node.to_dict() for node in graph], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def rows_in_graph(graph: Iterable[GraphNode] | None = None) -> dict[str, list[str]]:
    """Distinct owning rows of the graph per RFC, sorted."""

    rows: dict[str, set[str]] = {}
    for node in graph or build_graph():
        rows.setdefault(node.rfc, set()).add(node.row)
    return {rfc: sorted(names) for rfc, names in sorted(rows.items())}


# --- Components --------------------------------------------------------------------------------


component_hash = h3_fp.component_hash


def component_hashes() -> dict[str, str]:
    """Hash of every pinned component directory."""

    return {
        directory: component_hash(files)
        for directory, files in sorted(pinned_checkpoint()["dirs"].items())
    }


def _relative(files: Iterable[Mapping[str, Any]]) -> dict[str, tuple[int, str]]:
    return {
        f["path"].split("/", 1)[1]: (int(f["size"]), f["sha256"] or f["git_oid"]) for f in files
    }


def check_inherited_text_encoder() -> str:
    """The released ``text_encoder/`` must equal MiniMax-H3's file by file. Returns its hash."""

    snapshot = pinned_checkpoint()
    ours = _relative(snapshot["dirs"]["text_encoder"])
    theirs = _relative(snapshot["inherited_text_encoder"]["files"])
    if ours != theirs:
        changed = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k))
        raise ValueError(f"text_encoder differs from {TEXT_ENCODER_REPO}: {changed}")
    lines = sorted(f"{path}|{size}|{digest}" for path, (size, digest) in ours.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# --- Fingerprint -------------------------------------------------------------------------------


def build_fingerprint() -> dict[str, Any]:
    """Assemble the fingerprint; every cross-check runs and raises on disagreement."""

    snapshot = pinned_checkpoint()
    config = transformer_config()
    if snapshot["configs"][DIFFUSERS_TRANSFORMER_DIR]["_class_name"] != ARCHITECTURE:
        raise ValueError("Diffusers transformer class differs from the pinned one")
    if snapshot["configs"][UPSTREAM_TRANSFORMER_DIR]["_class_name"] != UPSTREAM_ARCHITECTURE:
        raise ValueError("upstream transformer class differs from the pinned one")
    _check_layouts_agree(config, upstream_transformer_config())
    if config["num_pdd_heads"] != NUM_PDD_HEADS:
        raise ValueError("config num_pdd_heads != the released head count")

    table = expected_tensor_table(config)
    pinned = snapshot["tensor_tables"][DIFFUSERS_TRANSFORMER_DIR]
    if pinned["digest"] != tensor_table_digest(table):
        raise ValueError("analytic tensor table does not match the Diffusers checkpoint headers")
    if pinned["parameters"] != parameter_count(table) or pinned["tensors"] != len(table):
        raise ValueError("Diffusers checkpoint parameter or tensor count mismatch")
    pdd_table = expected_pdd_table(config)
    if snapshot["tensor_tables"]["pdd"]["digest"] != tensor_table_digest(pdd_table):
        raise ValueError("pdd_fused_heads.safetensors header does not match the analytic table")

    model_index = snapshot["configs"]["model_index"]["_minimax_h3"]
    if model_index["pdd_fused_heads"] != PDD_FILE or model_index["tasks"] != ["t2i", "edit"]:
        raise ValueError("model_index.json _minimax_h3 block differs from the pinned one")
    _check_scheduler()
    graph = build_graph(config)
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            "repo": REPO,
            "revision": REVISION,
            "architecture": ARCHITECTURE,
            "upstream_architecture": UPSTREAM_ARCHITECTURE,
            "tasks": list(model_index["tasks"]),
        },
        "reference_implementation": {
            "repo": "huggingface/diffusers",
            "commit": DIFFUSERS_COMMIT,
        },
        "text_encoder": {
            "repo": TEXT_ENCODER_REPO,
            "revision": TEXT_ENCODER_REVISION,
            "hidden_state_layer": TEXT_ENCODER_LAYER,
            "identical_to_released_text_encoder": True,
            "files_sha256": check_inherited_text_encoder(),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        },
        "layouts": {
            directory: dict(info)
            for directory, info in sorted(snapshot["tensor_tables"].items())
            if directory != "pdd"
        },
        "transformer_config": config,
        "derived": derived_quantities(config),
        "parameters": parameter_count(table),
        "tensors": len(table),
        "mixed_precision": {
            "fp32_modules": list(FP32_MODULES),
            "fp32_compute_without_parameters": list(FP32_COMPUTE),
            "fp64_compute": ["position_grid"],
            "bf16": "every other module and the packed residual stream",
        },
        "modality_tags": dict(MODALITY_TAGS),
        "pdd": {
            "file": PDD_FILE,
            "heads": NUM_PDD_HEADS,
            "table_digest": snapshot["tensor_tables"]["pdd"]["digest"],
            "head_sha256": pdd_head_digests(),
            "upstream_shared_head": UPSTREAM_SHARED_HEAD,
            "upstream_shared_head_is_a_pdd_head": False,
        },
        "scheduler": dict(SCHEDULER),
        "sigma_grid": sigma_grid(),
        "vae": vae_facts(),
        "graph": {
            "nodes": len(graph),
            "stem_nodes": len(STEM_NODES) + 1,
            "refiner_blocks": config["num_refiner_layers"],
            "refiner_nodes_per_block": len(h3_fp.REFINER_NODES),
            "blocks": config["num_layers"],
            "block_nodes_per_block": len(h3_fp.BLOCK_NODES),
            "digest": graph_digest(graph),
            "rows": rows_in_graph(graph),
        },
        "components": component_hashes(),
    }
    fingerprint["fingerprint_sha256"] = fingerprint_digest(fingerprint)
    return fingerprint


fingerprint_digest = h3_fp.fingerprint_digest


# --- Validation against a live model / checkpoint ---------------------------------------------


def validate_diffusers_model(model: Any) -> None:
    """Check a Diffusers ``MiniMaxH3ImageTransformer2DModel`` (meta device is fine).

    Compares the class name, config and the state-dict names/shapes to the fingerprint.
    Dtypes are not compared because Diffusers builds modules in one default dtype; use
    ``validate_checkpoint_headers`` for stored dtypes.
    """

    problems: list[str] = []
    if type(model).__name__ != ARCHITECTURE:
        problems.append(f"class {type(model).__name__} != {ARCHITECTURE}")
    config = transformer_config()
    live = _public(dict(model.config))
    for key, value in config.items():
        if live.get(key) != value:
            problems.append(f"config[{key}] = {live.get(key)!r} != {value!r}")
    actual = {
        name: (None, tuple(int(d) for d in tensor.shape))
        for name, tensor in model.state_dict().items()
    }
    problems += h3_fp._diff_tables(actual, expected_tensor_table(config), check_dtype=False)
    if problems:
        raise ValueError(
            "H3-Image model does not match the fingerprint:\n  " + "\n  ".join(problems)
        )


def validate_checkpoint_headers(directory: Path) -> int:
    """Check local Diffusers-layout shard headers (and an optional PDD file) in ``directory``.

    Shards may be partially present: every tensor found must exist in the expected table
    with the declared dtype and shape. ``pdd_fused_heads.safetensors``, when present, is
    checked against the stacked-head table. Returns the number of tensors checked.
    """

    directory = Path(directory)
    config = transformer_config()
    shards = sorted(p for p in directory.glob("*.safetensors") if p.name != PDD_FILE)
    pdd = directory / PDD_FILE
    if not shards and not pdd.is_file():
        raise ValueError(f"no safetensors shards in {directory}")
    found: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard in shards:
        found.update(h3_fp.read_safetensors_header(shard))
    problems = [
        problem
        for problem in h3_fp._diff_tables(found, expected_tensor_table(config), check_dtype=True)
        if not problem.startswith("missing tensor")
    ]
    count = len(found)
    if pdd.is_file():
        heads = h3_fp.read_safetensors_header(pdd)
        problems += h3_fp._diff_tables(heads, expected_pdd_table(config), check_dtype=True)
        count += len(heads)
    if problems:
        raise ValueError(f"{directory} does not match the fingerprint:\n  " + "\n  ".join(problems))
    return count
