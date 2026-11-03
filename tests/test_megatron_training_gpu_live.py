# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 G2/G3 (GPU, SM90): a tiny Megatron GPT on the real strict logp route.

G2: VIME's own LM-head capture hook and request types feed the real provider;
the live snapshot and the recorded provider agree bitwise, and observation
adds no CUDA kernel. G3: real Megatron DDP, two micro-batches; gradient write
evidence, frozen-base absence and read-only observation.

Runs single-process (TP=1, world size 1). Skipped unless a Hopper GPU, the SM90
extension, Megatron-LM and VIME are all available.
"""

import argparse
import socket
from collections import Counter

import pytest
import torch
from p7_doubles import t02_sampling


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        import megatron.core  # noqa: F401
        import vime.backends.megatron_utils.model  # noqa: F401

        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    except Exception:
        return False
    return bool(_EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90"))


pytestmark = pytest.mark.skipif(not _ready(), reason="needs SM90 + extension + Megatron + VIME")

SEQ, VOCAB_PADDED, VOCAB_REAL, HIDDEN = 32, 512, 500, 128


@pytest.fixture(scope="module")
def megatron_runtime():
    import os

    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.update(
        RL_KERNEL_ATTENTION_CASE="P/P",
        RL_KERNEL_FFN_CASE="P/P",
        RL_KERNEL_LOGP_CASE="R/R",
        VIME_RL_KERNEL_STRICT="1",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)

    from rl_engine.integrations.ablation import integration_plan_from_environment
    from rl_engine.integrations.megatron_runtime import install_megatron_integration

    install_megatron_integration(integration_plan_from_environment())
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _build_model():
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from vime.backends.megatron_utils.model import _install_linear_logp_capture

    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    torch.manual_seed(0)
    model = GPTModel(
        config,
        get_gpt_layer_local_spec(),
        vocab_size=VOCAB_PADDED,
        max_sequence_length=SEQ,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
    ).cuda()
    # As in Megatron's get_model (used by VIME): Float16Module casts every parameter,
    # including norms, to BF16 and returns FP32 logits on the last stage.
    model = Float16Module(config, model)
    args = argparse.Namespace(padded_vocab_size=VOCAB_PADDED, vocab_size=VOCAB_REAL)
    _install_linear_logp_capture([model], args)
    return model, config


def _tokens(shift=0):
    return ((torch.arange(SEQ, device="cuda") * 7 + 3 + shift) % VOCAB_REAL).reshape(1, SEQ)


def _active_mask():
    mask = torch.zeros(SEQ, dtype=torch.bool)
    mask[SEQ // 2 : SEQ - 1] = True  # response rows; the last row has no target
    return mask


def _forward_request(model, tokens):
    """Mirror VIME's get_log_probs_and_entropy request for one packed sample (CP=1)."""

    from megatron.core import parallel_state as mpu
    from vime.backends.megatron_utils.linear_logp_provider import LinearLogpRequest, TokenLayout
    from vime.backends.megatron_utils.model import _take_linear_logp_context

    positions = torch.arange(SEQ, device="cuda").reshape(1, SEQ)
    logits = model(tokens, positions, None).squeeze(0).contiguous()
    context = _take_linear_logp_context(model)
    targets = torch.zeros(SEQ, dtype=torch.long, device="cuda")
    targets[: SEQ - 1] = tokens[0, 1:]
    return LinearLogpRequest(
        logits=logits,
        target_ids=targets,
        tensor_parallel_group=mpu.get_tensor_model_parallel_group(),
        token_layout=TokenLayout(world_size=1, rank=0, layout="single"),
        with_entropy=False,
        with_entropy_grad=False,
        chunk_size=1024,
        log_prob_keep_mask=None,
        context=context,
        temperature=1.0,
        metadata={
            "logits_are_temperature_scaled": True,
            "complete_sampling_support": False,
            "real_vocab_size": VOCAB_REAL,
            "padded_vocab_size": VOCAB_PADDED,
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "tp_world_size": mpu.get_tensor_model_parallel_world_size(),
        },
    )


def _context(request, gate):
    from rl_engine.alignment.cross_config.adapters.megatron_training import (
        PreUpdateWeightLock,
        ShardDigest,
    )

    weight = request.context.projection.weight
    shard = ShardDigest.from_tensor(weight, tp_rank=0, vocab_start=0, vocab_end=VOCAB_PADDED)
    return gate.TrainingCallContext(
        checkpoint_id="ckpt-0",
        weight_id="w-0",
        weight_lock=PreUpdateWeightLock("ckpt-0", "w-0", (shard,)),
        row_layout=gate.RowLayout(("s0",), (SEQ,)),
        active_token_mask=_active_mask(),
        lm_head_lora=False,
        sampling_record=t02_sampling.record(t02_sampling.IDENTITY_REQUEST),
    )


def _cuda_kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = Counter()
    for event in prof.events():
        if (
            event.device_type.name == "CUDA"
            and "memcpy" not in event.name.lower()
            and ("memset" not in event.name.lower())
        ):
            names[event.name] += 1
    return names


def test_g2_live_matches_recorded_and_observation_adds_no_kernel(megatron_runtime):
    from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate
    from rl_engine.alignment.cross_config.adapters.megatron_training_provider import (
        M10ProviderInput,
        RecordedTrainingProvider,
    )
    from rl_engine.integrations.megatron_training_readback import TrainingLogpObserver
    from rl_engine.integrations.vime.linear_logp_provider import provider

    model, _ = _build_model()
    with torch.no_grad():
        request = _forward_request(model, _tokens())
        plain_kernels = _cuda_kernels(lambda: provider(request))
        observer = TrainingLogpObserver()
        observer.install()
        try:
            observed_kernels = _cuda_kernels(lambda: provider(request))
        finally:
            observer.uninstall()
    assert observed_kernels == plain_kernels, "observation launched or removed CUDA kernels"

    context = _context(request, gate)
    live = gate.validate_training_call(observer.snapshots[-1], context)
    assert live.path == "recompute" and live.entrypoint == "sm90_deterministic_linear_logp_tp"
    assert live.runtime["runtime.rng_consumed"] is False
    assert live.runtime["runtime.stream"]["stream"] is not None

    recorded = RecordedTrainingProvider().score(
        M10ProviderInput(
            final_decoder_rows=request.context.hidden.detach(),
            lm_head_weight_local=request.context.projection.weight.detach(),
            target_token_ids=request.target_ids,
            active_token_mask=_active_mask(),
            vocab_start=0,
            real_vocab_size=VOCAB_REAL,
            padded_vocab_size=VOCAB_PADDED,
            token_layout_world_size=1,
            token_layout_rank=0,
            token_layout="single",
            row_layout=gate.RowLayout(("s0",), (SEQ,)),
            checkpoint_id="ckpt-0",
            weight_id="w-0",
            weight_lock=context.weight_lock,
            lm_head_lora=False,
            sampling_record=context.sampling_record,
            tensor_parallel_group=request.tensor_parallel_group,
            metadata=dict(request.metadata),
        )
    )
    assert recorded.path == live.path
    assert recorded.row_keys == live.row_keys
    assert recorded.active_raw_bytes == live.active_raw_bytes


def _name(evidence, suffix):
    matches = [r.name for r in evidence.records if r.name.endswith(suffix)]
    assert len(matches) == 1, matches
    return matches[0]


def _ddp_step(observe, *, fusion=False):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    from rl_engine.integrations.megatron_training_gradients import GradientStepObserver
    from rl_engine.integrations.vime.linear_logp_provider import provider

    model, config = _build_model()
    config.gradient_accumulation_fusion = fusion
    for param in model.module.embedding.parameters():
        param.requires_grad_(False)  # frozen base
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=False, grad_reduce_in_fp32=True
        ),
        module=model,
    )
    observer = GradientStepObserver(ddp)
    if observe:
        observer.install()
    ddp.zero_grad_buffer()
    mask = _active_mask().cuda()
    for shift in (0, 1):  # two micro-batches
        result = provider(_forward_request(ddp.module, _tokens(shift)))
        (-(result.logp.squeeze(-1)[mask]).sum()).backward()
    ddp.finish_grad_sync()
    evidence = observer.end_step() if observe else None
    grads = {
        name: param.main_grad.detach().clone()
        for name, param in ddp.module.named_parameters()
        if param.requires_grad
    }
    if observe:
        observer.uninstall()
    return evidence, grads


def test_g3_ddp_gradient_evidence(megatron_runtime):
    from rl_engine.integrations.megatron_training_gradients import check_base_grad_absent

    evidence, _ = _ddp_step(observe=True)
    head = evidence.record(_name(evidence, "output_layer.weight"))
    assert head.wrote is True and head.write_path == "autograd_main_grad"
    assert head.accumulate_hook_calls == 2
    assert head.buffer_index is not None and head.buffer_range is not None
    base = [r.name for r in evidence.records if ".embedding." in f".{r.name}"]
    assert base, "embedding parameters must be observed"
    check_base_grad_absent(evidence, base)
    assert not evidence.unknown


def test_g3_observation_does_not_change_main_grad(megatron_runtime):
    _, first = _ddp_step(observe=False)
    _, second = _ddp_step(observe=False)
    reproducible = all(
        torch.equal(first[n].view(torch.int32), second[n].view(torch.int32)) for n in first
    )
    if not reproducible:
        pytest.skip(
            "plain DDP step is not bitwise reproducible here; read-only check not decidable"
        )
    _, observed = _ddp_step(observe=True)
    for name in first:
        assert torch.equal(first[name].view(torch.int32), observed[name].view(torch.int32)), name


def test_g3_fused_wgrad_path(megatron_runtime):
    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401
    except ImportError:
        pytest.skip("fused wgrad kernel (apex/TE fused_weight_gradient_mlp_cuda) not installed")
    evidence, _ = _ddp_step(observe=True, fusion=True)
    assert evidence.record(_name(evidence, "output_layer.weight")).write_path == "fused_main_grad"
