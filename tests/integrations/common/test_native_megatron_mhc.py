# SPDX-License-Identifier: Apache-2.0
"""Native MCore GPU integration unit; reduced random weights, not checkpoint parity."""

import json

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")
from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from rl_engine.integrations.engines.train.megatron.native_mhc import selected_raw_logprobs
from rl_engine.integrations.orchestrators.verl.mhc import bind_mhc_worker


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
def test_real_hybrid_model_forward_backward_and_version_update(tmp_path):
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(13)
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        num_moe_experts=2,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=2,
        mhc_learned_output_contract=True,
        gradient_accumulation_fusion=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    model = HybridModel(
        cfg, vocab_size=128, max_sequence_length=32, hybrid_layer_pattern="E"
    ).cuda()
    params = {k: id(v) for k, v in model.named_parameters()}
    records = []
    w = bind_mhc_worker(
        model,
        "training",
        backend="rocm",
        mode="split",
        observer=lambda *args: records.append((args[-2], args[-1].shape)),
    )
    assert params == {k: id(v) for k, v in model.named_parameters()}
    x = torch.randint(0, 128, (1, 8), device="cuda")
    pos = torch.arange(8, device="cuda")[None, :]
    with w.request(0):
        logits = model(x, pos, attention_mask=None)
        print("LOGITS", logits.shape, logits.dtype, flush=True)
        values = selected_raw_logprobs(model, logits, torch.roll(x, -1, 1), weight_version=0)
        values.sum().backward()
    assert {"head", "lm_head", "raw_logprob", "residual", "normalized"} <= {n for n, _ in records}
    assert (
        model.decoder.hc_head_fn.grad is not None
        and torch.isfinite(model.decoder.hc_head_fn.grad).all()
    )
    with w.updating(1, torch.cuda.synchronize), torch.no_grad():
        model.output_layer.weight.add_(0.001 * model.output_layer.weight.grad.sign())
    with w.request(1):
        logits2 = model(x, pos, attention_mask=None)
        values2 = selected_raw_logprobs(model, logits2, torch.roll(x, -1, 1), weight_version=1)
    assert not torch.equal(values, values2)
    print(
        json.dumps(
            dict(
                real_model=str(type(model)),
                records=[(n, list(s)) for n, s in records],
                weight_version=w.version,
                backward=True,
                scope="native HybridModel unit test with reduced random weights",
            )
        ),
        flush=True,
    )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_native_mask_translation_rejects_padding():
    from rl_engine.integrations.engines.train.megatron.native_mhc import (
        normalize_native_causal_mask,
    )

    valid = torch.ones(2, 7, dtype=torch.bool)
    assert normalize_native_causal_mask(valid, valid.shape) is None
    valid[1, -1] = False
    with pytest.raises(ValueError, match="padding"):
        normalize_native_causal_mask(valid, valid.shape)
    causal = torch.ones(7, 7, dtype=torch.bool).triu(1)
    assert normalize_native_causal_mask(causal) is None
    causal[2, 0] = True
    with pytest.raises(ValueError, match="document"):
        normalize_native_causal_mask(causal)
