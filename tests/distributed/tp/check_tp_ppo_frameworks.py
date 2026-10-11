# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Opt-in real Megatron/DeepSpeed TP validation; see examples/post_training/tp_ppo/README.md.

Run TP=1 first with --output, then TP=2 with --baseline pointing at that file.
No mocks or framework source patches are used. All models are random tiny GPTs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.integrations.common.ppo import (
    PPOConfig,
    PPORollout,
    PPOTrainingWorker,
    build_ppo_batch,
)
from rl_engine.integrations.common.tp_scalar import TPScalarExecutor, clip_tp_grad_norm_
from rl_engine.integrations.engines.train.deepspeed.ppo import (
    DeepSpeedTPGroups,
    deepspeed_ppo_worker,
)
from rl_engine.integrations.engines.train.megatron.ppo import (
    MegatronPolicyAdapter,
    MegatronScalarAdapter,
    megatron_parameter_layouts,
)


def make_gpt(tp: int, *, policy: bool, pg_collection=None, hidden_size=32, max_sequence_length=16):
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(44)
    group = (
        pg_collection.tp
        if pg_collection is not None
        else parallel_state.get_tensor_model_parallel_group()
    )
    model_parallel_cuda_manual_seed(
        44,
        tp_rank=dist.get_rank(group),
        ep_rank=0,
        etp_rank=dist.get_rank(group),
        force_reset_rng=True,
    )
    config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=4,
        ffn_hidden_size=2 * hidden_size,
        tensor_model_parallel_size=tp,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        masked_softmax_fusion=False,
        bias_dropout_fusion=False,
    )
    return GPTModel(
        config,
        get_gpt_layer_local_spec(),
        vocab_size=32,
        max_sequence_length=max_sequence_length,
        post_process=policy,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="learned_absolute",
        pg_collection=pg_collection,
    ).cuda()


def make_model(
    tp: int, *, policy: bool, pg_collection=None, hidden_size=32, max_sequence_length=16
):
    backbone = make_gpt(
        tp,
        policy=policy,
        pg_collection=pg_collection,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
    )
    group = backbone.pg_collection.tp
    if policy:
        return MegatronPolicyAdapter(backbone).eval()
    head = torch.nn.Linear(hidden_size // tp, 1, bias=False, device="cuda")
    generator = torch.Generator().manual_seed(144)
    master = torch.randn(1, hidden_size, generator=generator) * 0.05
    with torch.no_grad():
        head.weight.copy_(master.chunk(tp, dim=1)[dist.get_rank(group)])
    return MegatronScalarAdapter(
        backbone, head, group=group, leader=dist.get_process_group_ranks(group)[0]
    ).eval()


def rollout(device):
    ids = torch.tensor([[1, 2, 3, 4, 5, 0], [0, 1, 2, 3, 4, 5]], device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]], dtype=torch.bool, device=device)
    actions = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1]], dtype=torch.bool, device=device)
    return PPORollout(
        ids,
        mask,
        (mask.long().cumsum(1) - 1).clamp(min=0),
        actions,
        torch.zeros_like(actions, dtype=torch.float32),
        torch.tensor([True, False], device=device),
        torch.tensor([13, 7], device=device),
        torch.tensor(0, device=device),
        torch.tensor(0, device=device),
        torch.tensor([1.0, 0.0, 1.0], device=device),
    )


@torch.no_grad()
def snapshot(model, group, *, gradients=False):
    """Reconstruct master tensors using Megatron's actual partition annotations."""
    parameters = dict(model.named_parameters())
    result = {}
    size = dist.get_world_size(group)
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue
        parameter = parameters.get(name)
        if gradients:
            value = (
                parameter.grad
                if parameter is not None and parameter.grad is not None
                else torch.zeros_like(value)
            )
        value = value.detach().contiguous()
        if name == "bias" and isinstance(model, MegatronScalarAdapter):
            value = value.clone()
            dist.all_reduce(value, group=group)
        elif parameter is not None and getattr(parameter, "tensor_model_parallel", False):
            shards = [torch.empty_like(value) for _ in range(size)]
            dist.all_gather(shards, value, group=group)
            axis = parameter.partition_dim
            stride = parameter.partition_stride
            pieces = [shard.chunk(stride, dim=axis) for shard in shards]
            value = torch.cat(
                [pieces[rank][i] for i in range(stride) for rank in range(size)],
                dim=axis,
            )
        result[name] = value.cpu().clone()
    return result


def initialize_engine(model, *, tp_group=None, dp_group=None):
    import deepspeed
    from megatron.core import parallel_state

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        mpu=DeepSpeedTPGroups(
            (
                tp_group
                if tp_group is not None
                else parallel_state.get_tensor_model_parallel_group()
            ),
            (dp_group if dp_group is not None else parallel_state.get_data_parallel_group()),
        ),
        dist_init_required=False,
        config={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "train_batch_size": 2,
            "zero_optimization": {"stage": 0},
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
            "gradient_clipping": 0.0,
            "steps_per_print": 100000,
        },
    )
    return engine


def run(framework: str, tp: int):
    from megatron.core import parallel_state

    group = parallel_state.get_tensor_model_parallel_group()
    leader = dist.get_process_group_ranks(group)[0]
    device = torch.device("cuda", torch.cuda.current_device())
    sample = rollout(device)
    sample.validate(actor_version=0, critic_version=0)
    scalar = make_model(tp, policy=False)
    initial = snapshot(scalar, group)
    reward_model = make_model(tp, policy=False).requires_grad_(False)
    reward_initial = snapshot(reward_model, group)
    reward_executor = TPScalarExecutor(
        reward_model, group=group, leader=leader, output_layout="local_sum"
    )
    engine = initialize_engine(scalar) if framework == "deepspeed" else None
    model = engine if engine is not None else scalar
    executor = TPScalarExecutor(model, group=group, leader=leader, output_layout="local_sum")
    reward = reward_executor.rewards(sample.inputs)
    if reward is None:
        reward = torch.empty(2, device=device)
    dist.broadcast(reward, src=leader, group=group)
    with torch.no_grad():
        values_before = executor.values(sample.inputs)
    expected_rewards = values_before[
        torch.arange(2, device=device), torch.tensor([4, 5], device=device)
    ]
    torch.testing.assert_close(reward, expected_rewards)
    # Padding must not change the valid final state of an identical sequence.
    torch.testing.assert_close(reward[0], reward[1], atol=2e-6, rtol=2e-5)
    scalar.zero_grad(set_to_none=True)
    value_loss = executor.values(sample.inputs)[sample.attention_mask].square().mean()
    if engine is None:
        value_loss.backward()
    else:
        engine.backward(value_loss)
    gradients = snapshot(scalar, group, gradients=True)
    layout = megatron_parameter_layouts(scalar)
    norm = clip_tp_grad_norm_(
        [(p, layout[n]) for n, p in scalar.named_parameters()],
        group=group,
        max_norm=1.0,
        device=device,
    )
    if engine is None:
        torch.optim.SGD(scalar.parameters(), lr=0.01).step()
    else:
        engine.step()
    updated = snapshot(scalar, group)
    assert any(not torch.equal(initial[n], updated[n]) for n in initial)

    actor = make_model(tp, policy=True)
    reference = make_model(tp, policy=True).requires_grad_(False)
    reference_initial = snapshot(reference, group)
    actor_engine = initialize_engine(actor) if framework == "deepspeed" else None
    actor_model = actor_engine if actor_engine is not None else actor
    with torch.no_grad():
        behavior = actor_model(sample.inputs)
        reference_logps = reference(sample.inputs)
        torch.testing.assert_close(behavior, reference_logps)
    sample = PPORollout(**{**sample.packet(), "behavior_logps": behavior})
    with torch.no_grad():
        batch = build_ppo_batch(
            sample,
            state_values=values_before,
            rewards=reward,
            reference_logps=reference_logps,
            config=PPOConfig(),
        )
    config = PPOConfig(epochs=2, minibatch_size=2)
    if framework == "deepspeed":
        critic_worker = deepspeed_ppo_worker(
            engine, role="critic", forward=executor.values, group=group, config=config
        )
        actor_worker = deepspeed_ppo_worker(
            actor_engine, role="actor", forward=actor_engine, group=group, config=config
        )
    else:
        critic_worker = PPOTrainingWorker(
            role="critic",
            model=scalar,
            forward=executor.values,
            optimizer=torch.optim.SGD(scalar.parameters(), lr=0.01),
            parameter_layouts=megatron_parameter_layouts(scalar),
            group=group,
            config=config,
        )
        actor_worker = PPOTrainingWorker(
            role="actor",
            model=actor,
            forward=actor,
            optimizer=torch.optim.SGD(actor.parameters(), lr=0.01),
            parameter_layouts=megatron_parameter_layouts(actor),
            group=group,
            config=config,
        )
    actor_initial = snapshot(actor, group)
    actor_gradients = {}
    backward = actor_worker.backward

    def capture_backward(loss):
        backward(loss)
        actor_gradients.update(snapshot(actor, group, gradients=True))

    actor_worker.backward = capture_backward
    actor_stats = actor_worker.train(batch)
    critic_stats = critic_worker.train(batch)
    actor_after = snapshot(actor, group)
    assert any(not torch.equal(actor_initial[n], actor_after[n]) for n in actor_after)
    if actor_engine is not None:
        assert actor_engine.global_steps == 2 and engine.global_steps == 3
    for frozen, before in (
        (reward_model, reward_initial),
        (reference, reference_initial),
    ):
        after = snapshot(frozen, group)
        assert all(torch.equal(before[name], after[name]) for name in before)
        assert all(p.grad is None for p in frozen.parameters())
    return {
        "initial": initial,
        "values": values_before.cpu(),
        "rewards": reward.cpu(),
        "gradients": gradients,
        "updated": updated,
        "actor_logps": behavior.cpu(),
        "actor_after": actor_after,
        "actor_gradients": actor_gradients,
        "critic_after": snapshot(scalar, group),
        "grad_norm": norm.cpu(),
    }, {"actor": actor_stats, "critic": critic_stats}


def compare(actual, expected):
    errors = {}
    for section in actual:
        left, right = actual[section], expected[section]
        if isinstance(left, dict):
            assert left.keys() == right.keys(), section
            pairs = [(left[key], right[key]) for key in left]
        else:
            pairs = [(left, right)]
        errors[section] = max(float((a - b).abs().max()) for a, b in pairs)
        for a, b in pairs:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4, msg=section)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("megatron", "deepspeed"), required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=args.tp)
    try:
        if dist.get_world_size() != args.tp:
            raise ValueError("this check requires DP=PP=1 and world_size=TP")
        actual, stats = run(args.framework, args.tp)
        if dist.get_rank() == 0:
            errors = (
                compare(actual, torch.load(args.baseline, weights_only=True))
                if args.baseline
                else {}
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(actual, args.output)
            report = {
                "framework": args.framework,
                "megatron_core": version("megatron-core"),
                "deepspeed": (version("deepspeed") if args.framework == "deepspeed" else None),
                "numpy": version("numpy"),
                "tp": args.tp,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "max_abs_errors": errors,
                "metrics": stats,
            }
            args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        dist.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
