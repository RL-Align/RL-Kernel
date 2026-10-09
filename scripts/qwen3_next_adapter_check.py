#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Four-rank real VIME/vLLM GDN adapter acceptance, scoped below complete-model L2."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal as exact


def run(checkpoint, output):
    from megatron.core import mpu
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from vime.backends.megatron_utils import hf_to_megatron as load_api
    from vime.backends.megatron_utils import megatron_to_hf as export_api
    from vime.backends.megatron_utils.hf_to_megatron.common import (
        SafetensorReader,
        shard_mcore_tensor,
    )
    from vime.backends.megatron_utils.update_weight.common import all_gather_param
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from rl_engine.integrations.qwen3_next_vime import get_qwen3_next_shared_spec
    from rl_engine.integrations.qwen3_next_vllm import SharedGDNMetadataBuilder, SharedQwenGDN

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    if dist.get_world_size() != 4:
        raise ValueError("GDN engine adapters require exactly four ranks")
    mpu.initialize_model_parallel(tensor_model_parallel_size=4)
    # vLLM 0.30.0's initialize_model_parallel reads get_current_vllm_config(), so the
    # engine config must exist before the vLLM process groups are created.
    vconfig = EngineArgs(
        model=str(checkpoint),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=4,
        max_model_len=2048,
        max_num_seqs=4,
        enforce_eager=True,
        enable_prefix_caching=False,
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        distributed_executor_backend="external_launcher",
        additional_config={"gdn_prefill_backend": "triton"},
    ).create_engine_config()
    with set_current_vllm_config(vconfig):
        init_distributed_environment(4, rank, "env://", local_rank, "nccl")
        initialize_model_parallel(4)
    args = SimpleNamespace(
        hf_checkpoint=str(checkpoint),
        num_experts=512,
        sequence_parallel=False,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=256,
    )
    config = TransformerConfig(
        num_layers=48,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=256,
        ffn_hidden_size=5120,
        num_moe_experts=512,
        moe_ffn_hidden_size=512,
        moe_router_topk=10,
        moe_shared_expert_intermediate_size=512,
        tensor_model_parallel_size=4,
        expert_tensor_parallel_size=4,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        attention_dropout=0,
        hidden_dropout=0,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        add_bias_linear=False,
    )
    spec = get_qwen3_next_shared_spec(args, config)
    training = build_module(
        spec.layer_specs[0].submodules.self_attention, config=config, layer_number=1
    )
    reader = SafetensorReader(checkpoint)
    prefix = "module.module.decoder.layers.0.self_attention."
    with torch.no_grad():
        for name, parameter in training.named_parameters():
            full_name = prefix + name
            full = load_api._LOADERS["qwen3_next"](full_name, reader, training.hf_config)
            local = shard_mcore_tensor(full_name, full, parameter)
            if local.shape != parameter.shape:
                raise ValueError(f"Incorrect training shard shape: {full_name}")
            parameter.copy_(local.to(device=parameter.device))
        conv_name = prefix + "linear_attn.conv1d.weight"
        conv = all_gather_param(conv_name, training.linear_attn.conv1d.weight)
        converted = export_api.convert_qwen3_next_to_hf(args, conv_name, conv)
        exact(converted[0][1].cpu(), reader.get_tensor("model.layers.0.linear_attn.conv1d.weight"))

    with set_current_vllm_config(vconfig):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            rollout = SharedQwenGDN(
                vconfig.model_config.hf_text_config,
                vconfig,
                prefix="model.layers.0.linear_attn",
                gqa_interleaved_layout=True,
            ).cuda()
        finally:
            torch.set_default_dtype(previous_dtype)
        for name, parameter in rollout.named_parameters():
            value = reader.get_tensor(f"model.layers.0.linear_attn.{name}")
            getattr(parameter, "weight_loader", default_weight_loader)(parameter, value)
            exact(parameter, dict(training.linear_attn.named_parameters())[name], name=name)
        cache_spec = rollout.get_kv_cache_spec(vconfig)
        builder = SharedGDNMetadataBuilder(
            cache_spec, [rollout.prefix], vconfig, torch.device("cuda")
        )

        def initial_cache():
            # Deliberately dirty slots prove the metadata resets a new prompt.
            return tuple(
                torch.full((8, *shape), 0.125, dtype=dtype, device="cuda")
                for shape, dtype in zip(rollout.get_state_shape(), rollout.get_state_dtype())
            )

        def step(hidden, lengths, computed, slots):
            cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
            seq = torch.tensor(lengths, dtype=torch.int32) + torch.tensor(
                computed, dtype=torch.int32
            )
            common = CommonAttentionMetadata(
                query_start_loc=cu.cuda(),
                query_start_loc_cpu=cu,
                seq_lens=seq.cuda(),
                num_reqs=len(lengths),
                num_actual_tokens=sum(lengths),
                max_query_len=max(lengths),
                max_seq_len=int(seq.max()),
                block_table_tensor=torch.tensor(slots, dtype=torch.int32, device="cuda")[:, None],
                slot_mapping=torch.zeros(sum(lengths), dtype=torch.int64, device="cuda"),
            )
            metadata = builder.build(0, common)
            with set_forward_context({rollout.prefix: metadata}, vconfig):
                return rollout(hidden_states=training.input_layernorm(hidden))

        checked = []
        generator = torch.Generator(device="cuda").manual_seed(713)
        for batch in (1, 4):
            for length in (8, 64, 256, 1024):
                hidden = torch.randn(
                    batch * length, 2048, generator=generator, device="cuda", dtype=torch.bfloat16
                )
                cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * length
                packed = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    max_seqlen_q=length,
                    max_seqlen_kv=length,
                )
                with torch.no_grad():
                    train_out, _ = training(hidden.unsqueeze(1), None, packed_seq_params=packed)
                    rollout.kv_cache = initial_cache()
                    whole = step(
                        hidden.clone(), [length] * batch, [0] * batch, list(range(1, batch + 1))
                    )
                    exact(whole, train_out[:, 0], name=f"VIME/vLLM B{batch} L{length}")
                    final = tuple(state.clone() for state in rollout.kv_cache)
                    for cut in (1, 63, 64, 65, 100, 528, 576):
                        if cut >= length:
                            continue
                        rollout.kv_cache = initial_cache()
                        pieces = [[] for _ in range(batch)]
                        for lo, hi in ((0, cut), (cut, length)):
                            order = list(range(batch)) if lo == 0 else list(reversed(range(batch)))
                            rows = [
                                row
                                for seq in order
                                for row in range(seq * length + lo, seq * length + hi)
                            ]
                            part = step(
                                hidden[rows],
                                [hi - lo] * batch,
                                [lo] * batch,
                                [seq + 1 for seq in order],
                            )
                            for position, seq in enumerate(order):
                                pieces[seq].append(
                                    part[position * (hi - lo) : (position + 1) * (hi - lo)]
                                )
                        exact(
                            torch.cat([torch.cat(parts) for parts in pieces]),
                            whole,
                            name=f"chunk cut={cut}",
                        )
                        for current, expected in zip(rollout.kv_cache, final):
                            exact(current, expected, name="owned rollout cache")
                checked.append(f"batch{batch}-length{length}")

        # Exercise a fresh one-token prompt in the same real scheduler batch as
        # prefill requests, then sixteen reordered decode steps on owned caches.
        prompt_lengths = [1, 64, 256, 1024]
        lengths = [length + 16 for length in prompt_lengths]
        sequences = [
            torch.randn(length, 2048, generator=generator, device="cuda", dtype=torch.bfloat16)
            for length in lengths
        ]
        complete = torch.cat(sequences)
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
        )
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )
        with torch.no_grad():
            training_complete, _ = training(complete.unsqueeze(1), None, packed_seq_params=packed)
            rollout.kv_cache = initial_cache()
            whole = step(complete, lengths, [0] * 4, [1, 2, 3, 4])
            exact(whole, training_complete[:, 0], name="mixed VIME/vLLM complete")
            final = tuple(state.clone() for state in rollout.kv_cache)
            rollout.kv_cache = initial_cache()
            first = step(
                torch.cat([value[:length] for value, length in zip(sequences, prompt_lengths)]),
                prompt_lengths,
                [0] * 4,
                [1, 2, 3, 4],
            )
            pieces = [[part] for part in first.split(prompt_lengths)]
            for position in range(16):
                order = [3, 1, 0, 2] if position % 2 else [2, 0, 3, 1]
                positions = [length + position for length in prompt_lengths]
                outputs = step(
                    torch.cat(
                        [
                            sequences[index][positions[index] : positions[index] + 1]
                            for index in order
                        ]
                    ),
                    [1] * 4,
                    [prompt_lengths[index] + position for index in order],
                    [index + 1 for index in order],
                )
                for index, value in zip(order, outputs.split(1)):
                    pieces[index].append(value)
            exact(
                torch.cat([torch.cat(parts) for parts in pieces]),
                whole,
                name="sixteen decode steps",
            )
            for current, expected in zip(rollout.kv_cache, final):
                exact(current, expected, name="mixed owned rollout cache")
        checked.append("mixed-prefill-decode16-reorder")

        hidden = torch.randn(
            8, 1, 2048, generator=generator, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=8, max_seqlen_kv=8
        )
        value, _ = training(hidden, None, packed_seq_params=packed)
        value[4:].float().square().sum().backward()
        if hidden.grad is None or not all(
            (torch.isfinite(hidden.grad).all(), bool(hidden.grad[:4].abs().max() > 0))
        ):
            raise AssertionError("Actual VIME response loss lost finite nonzero prompt gradients")
        grads = [
            parameter.grad for parameter in training.parameters() if parameter.grad is not None
        ]
        if not grads or any(not torch.isfinite(grad).all() for grad in grads):
            raise AssertionError("Actual VIME GDN parameter gradients must be finite")
        from rl_engine.integrations.qwen3_next_binary import imported_provider_identity

        result = {
            "status": "passed",
            "scope": "gdn_engine_adapter_component",
            "rank": rank,
            "provider_binary": imported_provider_identity(),
            "cases": checked,
            "prompt_gradient": "passed",
            "hf_conv_roundtrip": "passed",
            "full_model_l2": "not_evaluated",
            "batch_invariance_capability_registered": False,
            "conv_dim_first": is_conv_state_dim_first(),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.out)
