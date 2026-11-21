# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Construct the native VIME/Megatron actor for the fixed Qwen TP4 profile."""

import copy
import os
import sys
from pathlib import Path

from rl_engine.validation.models.qwen3_next.qwen3_next_rl_fixture import (
    OPTIMIZER_SETTINGS,
    POLICY_SETTINGS,
    require_policy_configuration,
)


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Native GPTModel construction, with providers installed before DDP/optimizer."""
    from megatron.training.global_vars import get_args
    from vime.backends.megatron_utils.model_provider import get_model_provider_func

    from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime import (
        install_vime_model_providers,
    )

    args = copy.copy(get_args())
    args.custom_model_provider_path = None
    factory = get_model_provider_func(args, "actor")
    model = factory(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    install_vime_model_providers(model)
    return model


# Stock VIME's own Qwen3-Next model spec (scripts/models/qwen3-next-80B-A3B.sh).
STOCK_SPEC = ("vime_plugins.models.qwen3_next", "get_qwen3_next_spec")
SHARED_SPEC = (
    "rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime",
    "get_qwen3_next_shared_spec",
)
# Stock VIME's parallel layout on the same four GPUs: TP2 with sequence parallel and expert
# tensor parallel 1 as in its recipe (scripts/run-qwen3-next-80B-A3B.sh), experts over all
# four ranks, so data parallel 2. Stock VIME cannot run TP4: Qwen3-Next has two KV heads,
# and Megatron's attention output gate is not narrowed when a rank keeps part of a KV group's
# query heads.
STOCK_TP, STOCK_EP = 2, 4


def runtime_argv(checkpoint, *, training, ep=1, providers=True, seq_length=1040, tp=4):
    """Use native argument parsing/validation; no precision-policy overrides.

    ``providers=False`` builds the actor on stock VIME: its own Qwen3-Next spec and model
    provider, the MoE fusion its recipe enables, and its layout (``tp=STOCK_TP`` with
    sequence parallel, ``ep=STOCK_EP``, expert tensor parallel 1). Only speed comparisons use
    it; every acceptance stage runs the RL-Kernel providers at TP4.
    """
    checkpoint = str(Path(checkpoint).absolute())
    if providers:
        if tp != 4 or ep not in (1, 4) or (training and ep != 1):
            raise ValueError("Native VIME runs EP4 for recompute only; training stays EP1")
    elif (tp, ep) != (STOCK_TP, STOCK_EP):
        raise ValueError(f"Stock VIME runs Qwen3-Next at TP{STOCK_TP} EP{STOCK_EP} here")
    if type(seq_length) is not int or seq_length < 1040:
        raise ValueError("The actor sequence budget cannot be below the certified 1040 tokens")
    values = {
        "hf-checkpoint": checkpoint,
        "load": checkpoint,
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": 4,
        "num-gpus-per-node": 4,
        "tensor-model-parallel-size": tp,
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "expert-model-parallel-size": ep,
        "expert-tensor-parallel-size": 4 // ep if providers else 1,
        "num-attention-heads": 16,
        "num-query-groups": 2,
        "kv-channels": 256,
        "num-layers": 48,
        "hidden-size": 2048,
        "ffn-hidden-size": 5120,
        "normalization": "RMSNorm",
        "position-embedding-type": "rope",
        "norm-epsilon": 1e-6,
        "rotary-percent": 0.25,
        "rotary-base": 10000000,
        "vocab-size": 151936,
        "make-vocab-size-divisible-by": 1,
        "padded-vocab-size": 151936,
        "moe-ffn-hidden-size": 512,
        "moe-shared-expert-intermediate-size": 512,
        "moe-router-score-function": "softmax",
        "moe-router-dtype": "fp32",
        "moe-router-topk": 10,
        "num-experts": 512,
        "moe-layer-freq": "[" + ",".join(["1"] * 48) + "]",
        "moe-aux-loss-coeff": 0,
        "moe-token-dispatcher-type": "alltoall",
        "hidden-dropout": 0,
        "attention-dropout": 0,
        "seq-length": seq_length,
        "max-position-embeddings": 262144,
        "global-batch-size": 4,
        "micro-batch-size": 1,
        "data-pad-size-multiplier": 1,
        "num-rollout": 3 if training else 0,
        "rollout-batch-size": 4,
        "n-samples-per-prompt": 1,
        "rollout-num-gpus": 0,
        "advantage-estimator": "grpo",
        "eps-clip": 0.2,
        "eps-clip-high": 0.2,
        "kl-coef": 0,
        "kl-loss-coef": 0,
        "entropy-coef": 0,
        "optimizer": "adam",
        "lr": 1e-6,
        "lr-decay-style": "constant",
        "adam-beta1": 0.9,
        "adam-beta2": 0.98,
        "adam-eps": 1e-8,
        "weight-decay": 0.1,
        "clip-grad": 1.0,
        "optimizer-offload-fraction": 1.0,
        "custom-model-provider-path": (
            "rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime_runtime"
            ".model_provider"
        ),
    }
    switches = [
        "debug-train-only",
        "disable-bias-linear",
        "qk-layernorm",
        "group-query-attention",
        "use-gated-attention",
        "apply-layernorm-1p",
        "swiglu",
        "untie-embeddings-and-output-weights",
        "attention-output-gate",
        "moe-shared-expert-gate",
        "moe-grouped-gemm",
        "use-rollout-logprobs",
        "disable-compute-advantages-and-returns",
        "optimizer-cpu-offload",
        "use-precision-aware-optimizer",
        # Megatron's fused weight-gradient accumulation needs APEX's
        # fused_weight_gradient_mlp_cuda, which the pinned environment does not
        # ship; the unfused path accumulates the same GEMM result into main_grad.
        "no-gradient-accumulation-fusion",
    ]
    if not providers:
        del values["custom-model-provider-path"]
        switches.extend(("moe-permute-fusion", "sequence-parallel"))
    argv = ["qwen3-next-native-actor"]
    for name, value in values.items():
        argv.extend((f"--{name}", str(value)))
    argv.extend(f"--{name}" for name in switches)
    argv.extend(("--spec", *(SHARED_SPEC if providers else STOCK_SPEC)))
    return argv


def create_actor(checkpoint, *, training, ep=1, providers=True, seq_length=1040, tp=4):
    """Call real native setup/load after the caller initializes Torch distributed.

    ``providers``, ``seq_length`` and ``tp`` are documented on :func:`runtime_argv`.
    """
    import torch
    import torch.distributed as dist
    from vime.backends.megatron_utils.initialize import init
    from vime.utils.arguments import parse_args

    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Native VIME actor must actually run inside the isolated venv")
    if not dist.is_initialized() or dist.get_world_size() != 4 or torch.cuda.device_count() != 4:
        raise ValueError("Native VIME actor requires four initialized ranks and visible GPUs")
    saved_argv = sys.argv
    try:
        sys.argv = runtime_argv(
            checkpoint,
            training=training,
            ep=ep,
            providers=providers,
            seq_length=seq_length,
            tp=tp,
        )
        args = parse_args()
    finally:
        sys.argv = saved_argv
    precision = {name: getattr(args, name, None) for name in ("optimization_dtype", "reduce_dtype")}
    for name, value in {**POLICY_SETTINGS, **OPTIMIZER_SETTINGS}.items():
        setattr(args, name, value)
    args.rank, args.local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    args.world_size = 4
    args.num_rollout = 3 if training else 0
    # The certified strict profile runs without sequence parallel. Stock VIME's recipe runs
    # TP2 with it, and Megatron's MoE layer refuses to train at TP > 1 without it.
    args.sequence_parallel = not providers
    args.fp32_residual_connection = False
    args.overlap_grad_reduce = False
    args.overlap_param_gather = False
    args.recompute_granularity = None
    args.use_critic = False
    require_policy_configuration(args)
    if any(getattr(args, name, None) != value for name, value in precision.items()):
        raise ValueError("Optimizer/reduction precision changed during profile setup")
    init(args)
    # vime's actor bootstrap creates a gloo group next to the NCCL one; the HF weight
    # exporter (HfWeightIteratorDirect) gathers its parameter metadata over it.
    # This runtime bypasses that bootstrap, so create the group here on every rank.
    from vime.utils.distributed_utils import init_gloo_group

    init_gloo_group()
    if (args.vocab_size, args.padded_vocab_size) != (151936, 151936):
        raise ValueError("Tokenizer initialization changed the official LM-head vocabulary")
    from vime.backends.megatron_utils.checkpoint import load_checkpoint

    from rl_engine.models.qwen3_next.qwen3_next_logprob import install_vime_logprob_provider

    if not training:
        from megatron.core.enums import ModelType
        from megatron.training.training import get_model
        from vime.backends.megatron_utils.linear_logp_provider import linear_logp_provider_path
        from vime.backends.megatron_utils.model import (
            _install_linear_logp_capture,
            get_model_provider_func,
        )

        model = get_model(
            get_model_provider_func(args, "actor"),
            ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
        )
        if linear_logp_provider_path(args) is not None:
            _install_linear_logp_capture(model, args)
        load_checkpoint(
            model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False
        )
        if providers:
            install_vime_logprob_provider()
        return args, model, None, None

    from vime.backends.megatron_utils.model import setup_model_and_optimizer

    model, optimizer, scheduler = setup_model_and_optimizer(args, role="actor")
    load_checkpoint(
        model, optimizer, scheduler, checkpointing_context={}, skip_load_to_model_and_opt=False
    )
    if providers:
        install_vime_logprob_provider()
    return args, model, optimizer, scheduler
