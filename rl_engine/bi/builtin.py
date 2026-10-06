# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shipped recipes. Future upstream reuse PRs add plans and comparisons here."""

from .catalog import Catalog, ExecutionPlan, ModelProfile, canonical

QWEN3_8B = ModelProfile(
    model_id="qwen3-8b",
    config_match=canonical(
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "hidden_act": "silu",
        }
    ),
)

STRICT_ENVIRONMENT = (
    ("RL_KERNEL_MODE", "strict"),
    ("RL_KERNEL_ATTENTION_CASE", "R/R"),
    ("RL_KERNEL_FFN_CASE", "R/R"),
    ("RL_KERNEL_LOGP_CASE", "R/R"),
    ("RL_KERNEL_VLLM_INTEGRATION", "1"),
)


def builtin_catalog() -> Catalog:
    catalog = Catalog()
    catalog.register_model(QWEN3_8B)
    for platform, result in (
        ("cuda", "scale_reference_s1234_g10_g11_optimized/summary.json"),
        ("rocm", "pr396_rocm_s1234_g10_g11_200/validation-summary.json"),
    ):
        catalog.register_plan(
            ExecutionPlan(
                plan_id=f"qwen3-8b.vime.{platform}.reference.v1",
                model_id=QWEN3_8B.model_id,
                framework="vime",
                platform=platform,
                adapter="vime.qwen3.v1",
                environment=STRICT_ENVIRONMENT
                + (
                    (
                        "RL_KERNEL_DET_GEMM_BACKEND",
                        "cublaslt_nosplitk" if platform == "cuda" else "triton_mfma",
                    ),
                )
                + ((("RL_KERNEL_ROCM_ATTENTION_BACKEND", "triton"),) if platform == "rocm" else ()),
                evidence=(f"examples/vime_qwen3_8b_tp4_cp2_200/results/{result}",),
                reference=True,
            )
        )
    # No invented timings: the existing reference is selected until a reviewed
    # comparison for the exact runtime/workload is shipped with a reuse PR.
    return catalog
