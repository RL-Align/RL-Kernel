# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 architecture fingerprint (RFC #420 ``h3_arch_fingerprint``).

One machine-readable description of what "the H3 graph" means to RL-Kernel:

* the pinned checkpoint (repo, revision, both variants, file hashes);
* the transformer configuration and the quantities derived from it;
* the declared mixed-precision boundaries;
* the deterministic rectified-flow scheduler settings;
* the executable graph: stem, two token-refiner blocks, 50 transformer blocks
  and the output path, each node tagged with the RFC row that owns its
  arithmetic.

The module needs neither torch nor a GPU. Everything is derived from
``pinned_checkpoint.json`` (written by ``tools/weights/snapshot_h3_checkpoint.py``
from the real Hub files) plus the analytic constants below, and the two are
cross-checked: the analytic tensor table must hash to the same digest as the
real safetensors headers.

``validate_diffusers_model`` and ``validate_checkpoint_headers`` compare a live
Diffusers module, or a local checkpoint directory, against the fingerprint.
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

SCHEMA_VERSION = "rlkernel.h3.arch_fingerprint.v1"
PINNED_CHECKPOINT_PATH = Path(__file__).with_name("pinned_checkpoint.json")

REPO = "MiniMaxAI/MiniMax-H3"
REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
ARCHITECTURE = "MiniMaxH3Transformer3DModel"

# Variants differ only in weights and inputs: identical config and tensor table.
VARIANTS: Mapping[str, str] = {"FL2VA": "transformer", "Ref2VA": "transformer_ref"}

# AdaLN modality tags (row = timestep_index * 3 + tag).
MODALITY_TAGS: Mapping[str, int] = {"video": 0, "text": 1, "audio": 2}

# Declared mixed precision. Everything not listed as FP32 is BF16.
FP32_MODULES = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")
FP32_COMPUTE = ("time_proj", "rope")  # no parameters; frequencies/features built in FP32
_SAFETENSORS_DTYPE = {"fp32": "F32", "bf16": "BF16"}

# Deterministic rectified-flow Euler scheduler (eta = 0). Shifts come from the
# released scheduler configs and are re-checked against pinned_checkpoint.json.
SCHEDULER: Mapping[str, Any] = {
    "family": "rectified_flow_euler",
    "eta": 0.0,
    "video_shift": 12.0,
    "audio_shift": 3.0,
    "sigma_transform": "s * sigma / (1 + (s - 1) * sigma)",
    "timestep": "t = 1 - sigma",
    "velocity": "data-ward, x0 = xt + sigma * v",
    "euler_update": "x_next = r * xt + (1 - r) * x0, r = sigma_next / sigma",
    "grid": "shifted linspace(1, 0, steps), consecutive FP32 duplicates removed",
}

# --- Graph vocabulary -------------------------------------------------------------------------

# The 17 nodes of one transformer block, in graph order, with the RFC row that
# owns the arithmetic. ``h3_block.NODE_NAMES`` must equal these names.
BLOCK_NODES: tuple[tuple[str, str], ...] = (
    ("adaln_projection", "adaln_projection_3mod"),
    ("norm1", "h3_rmsnorm"),
    ("q_proj", "h3_qkv_gemm"),
    ("k_proj", "h3_qkv_gemm"),
    ("v_proj", "h3_qkv_gemm"),
    ("q_norm", "h3_qk_rmsnorm_d128"),
    ("k_norm", "h3_qk_rmsnorm_d128"),
    ("rope_q", "h3_mm_rope_3axis_partial"),
    ("rope_k", "h3_mm_rope_3axis_partial"),
    ("attention", "h3_full_attention"),
    ("o_proj", "h3_attention_o_gemm"),
    ("residual_attn", "adaln_gate_residual"),
    ("norm2", "h3_rmsnorm"),
    ("ffn_gate_up", "h3_ffn_gate_up_gemm"),
    ("swiglu", "h3_swiglu"),
    ("ffn_down", "h3_ffn_down_gemm"),
    ("residual_mlp", "adaln_gate_residual"),
)
BLOCK_NODE_NAMES = tuple(name for name, _ in BLOCK_NODES)

# The 14 nodes of one token-refiner block: plain pre-norm, no AdaLN, no RoPE,
# plain (ungated) residual adds.
REFINER_NODES: tuple[tuple[str, str], ...] = (
    ("norm1", "h3_rmsnorm"),
    ("q_proj", "h3_qkv_gemm"),
    ("k_proj", "h3_qkv_gemm"),
    ("v_proj", "h3_qkv_gemm"),
    ("q_norm", "h3_qk_rmsnorm_d128"),
    ("k_norm", "h3_qk_rmsnorm_d128"),
    ("attention", "h3_full_attention"),
    ("o_proj", "h3_attention_o_gemm"),
    ("residual_attn", "token_refiner_block"),
    ("norm2", "h3_rmsnorm"),
    ("ffn_gate_up", "h3_ffn_gate_up_gemm"),
    ("swiglu", "h3_swiglu"),
    ("ffn_down", "h3_ffn_down_gemm"),
    ("residual_mlp", "token_refiner_block"),
)

# Stem nodes in execution order: (path, owning row, precision).
STEM_NODES: tuple[tuple[str, str, str], ...] = (
    ("video_patch_pack", "video_patch_pack_unpack", "fp32"),
    ("proj_in", "video_input_projection_fp32", "fp32"),
    ("audio_proj_in", "audio_input_projection_fp32", "fp32"),
    ("context_embedder", "text_context_projection", "bf16"),
    ("time_proj", "timestep_sinusoid_h3", "fp32"),
    ("time_embedder", "timestep_mlp_fp32", "fp32"),
    ("rope", "h3_mm_rope_3axis_partial", "fp32"),
)
PACK_NODE = ("packed_index_copy", "packed_index_copy", "bf16")
TAIL_NODES: tuple[tuple[str, str, str], ...] = (
    ("norm_out", "final_adaln_out", "bf16"),
    ("proj_out", "video_output_projection_fp32", "fp32"),
    ("audio_proj_out", "audio_output_projection_fp32", "fp32"),
    ("packed_index_select", "packed_index_select", "fp32"),
)
REFINER_FINAL_NODE = ("token_refiner.final_norm", "h3_rmsnorm", "bf16")


@dataclass(frozen=True)
class GraphNode:
    """One node of the strict H3 graph."""

    path: str  # unique, e.g. ``transformer_blocks.7.attention``
    stage: str  # stem | token_refiner | transformer_blocks | output
    layer: int | None  # block index within its stack, or None
    node: str  # node name inside the block (``attention``) or the module name
    row: str  # RFC #420 row owning the arithmetic
    precision: str  # "fp32" or "bf16": the declared compute precision at the node

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "stage": self.stage,
            "layer": self.layer,
            "node": self.node,
            "row": self.row,
            "precision": self.precision,
        }


# Checkpoint-compatible FP32 boundary for block nodes: everything in the blocks is BF16.
_BLOCK_PRECISION = "bf16"


# --- Pinned checkpoint -----------------------------------------------------------------------


@lru_cache(maxsize=1)
def pinned_checkpoint() -> Mapping[str, Any]:
    """Load the pinned Hub snapshot (file hashes, configs, tensor-table digests)."""

    data = json.loads(PINNED_CHECKPOINT_PATH.read_text())
    if data["repo"] != REPO or data["revision"] != REVISION:
        raise ValueError("pinned_checkpoint.json does not match the pinned repo/revision")
    return data


def transformer_config(variant: str = "FL2VA") -> dict[str, Any]:
    """Return the public (underscore-free) transformer config of ``variant``."""

    directory = VARIANTS[variant]
    raw = pinned_checkpoint()["configs"][directory]
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def derived_quantities(config: Mapping[str, Any]) -> dict[str, Any]:
    """Quantities the strict operators depend on, derived from the config."""

    heads, head_dim, hidden = (
        config["num_attention_heads"],
        config["attention_head_dim"],
        config["hidden_size"],
    )
    patch_t, patch_h, patch_w = config["patch_size"]
    modalities = len(MODALITY_TAGS)
    return {
        "attention_inner_dim": heads * head_dim,
        "attention_inner_dim_differs_from_hidden": heads * head_dim != hidden,
        "rotary_channels": 6 * config["rope_freq_dim"],
        "rotary_pass_through_channels": head_dim - 6 * config["rope_freq_dim"],
        "video_patch_width": config["in_channels"] * patch_t * patch_h * patch_w,
        "audio_width": config["audio_in_channels"],
        "adaln_modalities": modalities,
        "adaln_projection_out": 6 * modalities * hidden,
        "final_adaln_out": 2 * hidden,
        "ffn_gate_up_out": 2 * config["ffn_dim"],
    }


# --- Tensor table ---------------------------------------------------------------------------


def _precision_of(name: str) -> str:
    top = name.split(".", 1)[0]
    return "fp32" if top in FP32_MODULES else "bf16"


def _attention_tensors(prefix: str, config: Mapping[str, Any]) -> Iterable[tuple[str, tuple[int, ...]]]:
    hidden = config["hidden_size"]
    inner = config["num_attention_heads"] * config["attention_head_dim"]
    head_dim = config["attention_head_dim"]
    ffn = config["ffn_dim"]
    yield f"{prefix}.attn.to_q.weight", (inner, hidden)
    yield f"{prefix}.attn.to_k.weight", (inner, hidden)
    yield f"{prefix}.attn.to_v.weight", (inner, hidden)
    yield f"{prefix}.attn.norm_q.weight", (head_dim,)
    yield f"{prefix}.attn.norm_k.weight", (head_dim,)
    yield f"{prefix}.attn.to_out.0.weight", (hidden, inner)
    yield f"{prefix}.norm2.weight", (hidden,)
    yield f"{prefix}.ff.net.0.proj.weight", (2 * ffn, hidden)
    yield f"{prefix}.ff.net.2.weight", (hidden, ffn)


def expected_tensor_table(config: Mapping[str, Any]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Analytic ``name -> (safetensors dtype, shape)`` of the released checkpoint."""

    hidden = config["hidden_size"]
    patch_t, patch_h, patch_w = config["patch_size"]
    video_width = config["in_channels"] * patch_t * patch_h * patch_w
    shapes: dict[str, tuple[int, ...]] = {
        "proj_in.weight": (hidden, video_width),
        "proj_in.bias": (hidden,),
        "audio_proj_in.weight": (hidden, config["audio_in_channels"]),
        "audio_proj_in.bias": (hidden,),
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
        "proj_out.weight": (video_width, hidden),
        "proj_out.bias": (video_width,),
        "audio_proj_out.weight": (config["audio_in_channels"], hidden),
        "audio_proj_out.bias": (config["audio_in_channels"],),
        "token_refiner.final_norm.weight": (hidden,),
    }
    for index in range(config["num_refiner_layers"]):
        prefix = f"token_refiner.refiner_blocks.{index}"
        shapes[f"{prefix}.norm1.weight"] = (hidden,)
        shapes.update(_attention_tensors(prefix, config))
    for index in range(config["num_layers"]):
        prefix = f"transformer_blocks.{index}"
        shapes[f"{prefix}.norm1.weight"] = (hidden,)
        shapes.update(_attention_tensors(prefix, config))
        shapes[f"{prefix}.adaln_proj.linear.weight"] = (
            6 * len(MODALITY_TAGS) * hidden,
            config["time_embed_dim"],
        )
        shapes[f"{prefix}.adaln_proj.linear.bias"] = (6 * len(MODALITY_TAGS) * hidden,)
    return {
        name: (_SAFETENSORS_DTYPE[_precision_of(name)], shape) for name, shape in shapes.items()
    }


def tensor_table_digest(table: Mapping[str, tuple[str, Iterable[int]]]) -> str:
    """SHA-256 over the sorted ``name|dtype|shape`` lines of a tensor table."""

    lines = [
        f"{name}|{dtype}|{','.join(str(int(dim)) for dim in shape)}"
        for name, (dtype, shape) in sorted(table.items())
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def parameter_count(table: Mapping[str, tuple[str, Iterable[int]]]) -> int:
    total = 0
    for _, shape in table.values():
        count = 1
        for dim in shape:
            count *= int(dim)
        total += count
    return total


# --- Graph -----------------------------------------------------------------------------------


def build_graph(config: Mapping[str, Any] | None = None) -> tuple[GraphNode, ...]:
    """Return the full strict graph in execution order."""

    config = dict(config or transformer_config())
    nodes: list[GraphNode] = []
    for path, row, precision in STEM_NODES:
        nodes.append(GraphNode(path, "stem", None, path, row, precision))
    for index in range(config["num_refiner_layers"]):
        for name, row in REFINER_NODES:
            nodes.append(
                GraphNode(
                    f"token_refiner.refiner_blocks.{index}.{name}",
                    "token_refiner",
                    index,
                    name,
                    row,
                    _BLOCK_PRECISION,
                )
            )
    path, row, precision = REFINER_FINAL_NODE
    nodes.append(GraphNode(path, "token_refiner", None, "final_norm", row, precision))
    path, row, precision = PACK_NODE
    nodes.append(GraphNode(path, "stem", None, path, row, precision))
    for index in range(config["num_layers"]):
        for name, row in BLOCK_NODES:
            nodes.append(
                GraphNode(
                    f"transformer_blocks.{index}.{name}",
                    "transformer_blocks",
                    index,
                    name,
                    row,
                    _BLOCK_PRECISION,
                )
            )
    for path, row, precision in TAIL_NODES:
        nodes.append(GraphNode(path, "output", None, path, row, precision))
    paths = [node.path for node in nodes]
    if len(set(paths)) != len(paths):
        raise ValueError("graph paths must be unique")
    return tuple(nodes)


def graph_digest(graph: Iterable[GraphNode]) -> str:
    payload = json.dumps([node.to_dict() for node in graph], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def rows_in_graph(graph: Iterable[GraphNode] | None = None) -> tuple[str, ...]:
    """Distinct owning rows of the graph, sorted."""

    return tuple(sorted({node.row for node in (graph or build_graph())}))


# --- Components ------------------------------------------------------------------------------


def component_hash(files: Iterable[Mapping[str, Any]]) -> str:
    """SHA-256 over sorted ``path|size|sha256`` lines of a directory's files."""

    lines = sorted(f"{f['path']}|{f['size']}|{f['sha256'] or f['git_oid']}" for f in files)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def component_hashes() -> dict[str, str]:
    """Hash of every pinned component directory (variant transformers, schedulers, wrappers)."""

    return {
        directory: component_hash(files)
        for directory, files in sorted(pinned_checkpoint()["dirs"].items())
    }


# --- Fingerprint ---------------------------------------------------------------------------


def build_fingerprint() -> dict[str, Any]:
    """Assemble the fingerprint; every cross-check runs and raises on disagreement."""

    snapshot = pinned_checkpoint()
    config = transformer_config("FL2VA")
    if transformer_config("Ref2VA") != config:
        raise ValueError("FL2VA and Ref2VA transformer configs differ; the fingerprint is per-variant")
    table = expected_tensor_table(config)
    digest = tensor_table_digest(table)
    for variant, directory in VARIANTS.items():
        pinned = snapshot["tensor_tables"][directory]
        if pinned["digest"] != digest:
            raise ValueError(f"{variant}: analytic tensor table does not match the checkpoint headers")
        if pinned["parameters"] != parameter_count(table):
            raise ValueError(f"{variant}: parameter count mismatch")
    for key, shift in (("scheduler", "video_shift"), ("audio_scheduler", "audio_shift")):
        if float(snapshot["configs"][key]["shift"]) != float(SCHEDULER[shift]):
            raise ValueError(f"{key} shift does not match the pinned scheduler settings")
    graph = build_graph(config)
    rows = rows_in_graph(graph)
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {"repo": REPO, "revision": REVISION, "architecture": ARCHITECTURE},
        "variants": {
            variant: {
                "directory": directory,
                "tensor_table_digest": snapshot["tensor_tables"][directory]["digest"],
                "files": len(snapshot["dirs"][directory]),
                "bytes": sum(f["size"] for f in snapshot["dirs"][directory]),
            }
            for variant, directory in VARIANTS.items()
        },
        "transformer_config": config,
        "derived": derived_quantities(config),
        "parameters": parameter_count(table),
        "tensors": len(table),
        "mixed_precision": {
            "fp32_modules": list(FP32_MODULES),
            "fp32_compute_without_parameters": list(FP32_COMPUTE),
            "bf16": "every other module and the packed residual stream",
        },
        "modality_tags": dict(MODALITY_TAGS),
        "scheduler": dict(SCHEDULER),
        "graph": {
            "nodes": len(graph),
            "stem_nodes": len(STEM_NODES) + 1,
            "refiner_blocks": config["num_refiner_layers"],
            "refiner_nodes_per_block": len(REFINER_NODES),
            "blocks": config["num_layers"],
            "block_nodes_per_block": len(BLOCK_NODES),
            "digest": graph_digest(graph),
            "rows": list(rows),
        },
        "components": component_hashes(),
    }
    fingerprint["fingerprint_sha256"] = fingerprint_digest(fingerprint)
    return fingerprint


def fingerprint_digest(fingerprint: Mapping[str, Any]) -> str:
    """Digest of a fingerprint without its own ``fingerprint_sha256`` key."""

    body = {key: value for key, value in fingerprint.items() if key != "fingerprint_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --- Validation against a live model / checkpoint ---------------------------------------------


def _shape_tuple(shape: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _diff_tables(
    actual: Mapping[str, tuple[str | None, tuple[int, ...]]],
    expected: Mapping[str, tuple[str, tuple[int, ...]]],
    *,
    check_dtype: bool,
) -> list[str]:
    problems: list[str] = []
    for name in sorted(set(expected) - set(actual)):
        problems.append(f"missing tensor {name}")
    for name in sorted(set(actual) - set(expected)):
        problems.append(f"unexpected tensor {name}")
    for name in sorted(set(actual) & set(expected)):
        dtype, shape = actual[name]
        exp_dtype, exp_shape = expected[name]
        if shape != exp_shape:
            problems.append(f"{name}: shape {shape} != {exp_shape}")
        if check_dtype and dtype != exp_dtype:
            problems.append(f"{name}: dtype {dtype} != {exp_dtype}")
    return problems


def validate_diffusers_model(model: Any, variant: str = "FL2VA") -> None:
    """Check a Diffusers ``MiniMaxH3Transformer3DModel`` (meta device is fine).

    Compares the class name, config and the state-dict names/shapes to the
    fingerprint. Dtypes are not compared because Diffusers builds modules in
    one default dtype; use ``validate_checkpoint_headers`` for stored dtypes.
    """

    problems: list[str] = []
    if type(model).__name__ != ARCHITECTURE:
        problems.append(f"class {type(model).__name__} != {ARCHITECTURE}")
    config = transformer_config(variant)
    live = {k: v for k, v in dict(model.config).items() if not k.startswith("_")}
    for key, value in config.items():
        if live.get(key) != value:
            problems.append(f"config[{key}] = {live.get(key)!r} != {value!r}")
    actual = {
        name: (None, _shape_tuple(tensor.shape)) for name, tensor in model.state_dict().items()
    }
    problems += _diff_tables(actual, expected_tensor_table(config), check_dtype=False)
    if problems:
        raise ValueError("H3 model does not match the fingerprint:\n  " + "\n  ".join(problems))


def read_safetensors_header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Read ``name -> (dtype, shape)`` from a safetensors file without loading tensors."""

    with Path(path).open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return {name: (item["dtype"], _shape_tuple(item["shape"])) for name, item in header.items()}


def validate_checkpoint_headers(directory: Path, variant: str = "FL2VA") -> int:
    """Check every local shard header in ``directory`` against the fingerprint.

    Shards may be partially present: every tensor found must exist in the
    expected table with the declared dtype and shape. Returns the number of
    tensors checked. A directory with no shards is an error.
    """

    directory = Path(directory)
    expected = expected_tensor_table(transformer_config(variant))
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards in {directory}")
    found: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard in shards:
        found.update(read_safetensors_header(shard))
    problems = [
        problem
        for problem in _diff_tables(found, expected, check_dtype=True)
        if not problem.startswith("missing tensor")
    ]
    if problems:
        raise ValueError(f"{directory} does not match the fingerprint:\n  " + "\n  ".join(problems))
    return len(found)
