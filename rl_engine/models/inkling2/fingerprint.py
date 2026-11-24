# SPDX-License-Identifier: Apache-2.0
"""INK-01: released checkpoint identity and fixed text architecture."""

import hashlib
import json
from pathlib import Path

HF_MODEL = "thinkingmachines/Inkling-2-Mini"
HF_REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
MILES_REVISION = "43ee2489a76a059801f60961c59310178b1ce74d"
SGLANG_REVISION = "99f0be56153d55b79b08301543ab046583890ccc"
TEXT = {
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "vocab_size": 201024,
    "unpadded_vocab_size": 200058,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "swa_num_attention_heads": 24,
    "swa_num_key_value_heads": 8,
    "swa_head_dim": 128,
    "d_rel": 16,
    "rel_extent": 1024,
    "sliding_window_size": 1024,
    "n_layers_per_attn_res_block": 6,
    "local_layer_ids": [i for i in range(24) if i % 6 != 5],
    "n_routed_experts": 128,
    "num_experts_per_tok": 8,
    "n_shared_experts": 0,
    "intermediate_size": 1152,
    "route_scale": 8.0,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "final_logit_softcapping": 30.0,
    "log_scaling_n_floor": 8192,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "sconv_kernel_size": 4,
    "model_max_length": 1048576,
    "torch_dtype": "bfloat16",
    "attn_res_normalize_values": True,
    "use_value_norm_no_gamma": True,
    "use_gate_bias": True,
    "q_bias": False,
    "o_bias": False,
}
LOOP = dict(core_start=6, core_end=18, num_passes=3, gate_rank=64, is_varloop=False)


def validate_config(config: dict, *, revision: str, converted: bool = False) -> None:
    """Reject wrong architecture; revision is caller-supplied provenance, not authentication."""
    if revision != HF_REVISION:
        raise ValueError("checkpoint revision must match the pinned HF revision")
    if config.get("model_type") != "inkling2_loop":
        raise ValueError("expected inkling2_loop checkpoint")
    if config.get("architectures") != ["Inkling2LoopForConditionalGeneration"]:
        raise ValueError("unexpected checkpoint architecture")
    if config.get("dtype") != "bfloat16":
        raise ValueError("expected top-level bfloat16 dtype")
    text = config.get("text_config", {})
    for name, expected in TEXT.items():
        if text.get(name) != expected or type(text.get(name)) is not type(expected):
            raise ValueError(f"text_config.{name} does not match released Mini")
    if text.get("loop_config") != LOOP:
        raise ValueError("unsupported fixed-loop configuration")
    quant = config.get("quantization_config")
    # Miles copies config.json unchanged; precision is determined by the sidecar.
    if quant != {"quant_method": "modelopt", "quant_algo": "NVFP4"}:
        raise ValueError("unexpected released quantization config")


def validate_expert_precision(quant: dict, *, converted: bool) -> None:
    """Check each expert matrix against the release or Miles BF16 sidecar."""
    groups = quant["model"]["parameter_groups"]
    owners: dict[str, str] = {}
    for group in groups:
        for tensor in group["tensors"]:
            if tensor in owners:
                raise ValueError(f"duplicate precision assignment: {tensor}")
            owners[tensor] = group["compute_dtype"]
    for layer in range(24):
        expected = "bfloat16" if converted else ("nvfp4" if layer < 20 else "mxfp8")
        for suffix in ("w13_weight", "w2_weight"):
            name = f"model.llm.layers.{layer}.mlp.experts.{suffix}"
            if owners.get(name) != expected:
                raise ValueError(f"wrong expert precision: {name}, expected {expected}")
    if converted and any(value in ("nvfp4", "mxfp8") for value in owners.values()):
        raise ValueError("BF16 export still has low-bit model tensors")


def snapshot_fingerprint(directory: Path, *, revision: str, converted: bool = False) -> dict:
    """Hash actual config/quant metadata bytes; not a weight-content attestation."""
    raw = (directory / "config.json").read_bytes()
    validate_config(json.loads(raw), revision=revision, converted=converted)
    quant_path = directory / "hf_quant_config.json"
    quant_raw = quant_path.read_bytes()
    validate_expert_precision(json.loads(quant_raw), converted=converted)
    return {
        "model": HF_MODEL,
        "revision": revision,
        "profile": "bf16-dequant" if converted else "released-mixed-lowbit",
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "quant_metadata_sha256": hashlib.sha256(quant_raw).hexdigest(),
        "physical_layers": 24,
        "logical_visits": 48,
    }
