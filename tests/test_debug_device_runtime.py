# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Real CUDA/HIP acceptance, separately gated from CPU protocol tests."""

from __future__ import annotations

import copy

import pytest
import torch

from rl_engine.alignment.debug import (
    EventCapture,
    FusedCapture,
    GraphCapture,
    ModuleBoundary,
    TrainingCapture,
    load_megatron_model,
    megatron_weight_shards,
)
from rl_engine.alignment.debug.entry import analyze
from rl_engine.alignment.debug.events import _merge_trees, encode_tree

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a real CUDA/HIP device"
)
IDENTITY = {
    "weights": "device-acceptance-model",
    "tokens": [1, 2],
    "positions": [0, 1],
    "mask": "causal",
    "sampling": {},
}


if triton is not None:

    @triton.jit
    def _instrumented_affine_relu(X, PROJECTED, OUTPUT, N: tl.constexpr, FAULT: tl.constexpr):
        indices = tl.program_id(0) * 128 + tl.arange(0, 128)
        x = tl.load(X + indices, indices < N, other=0)
        projected = x * 2.0 + 1.0
        if FAULT:
            projected += 0.25
        activated = tl.maximum(projected, 0.0)
        # Debug output and final output come from this SAME fused kernel launch.
        tl.store(PROJECTED + indices, projected, indices < N)
        tl.store(OUTPUT + indices, activated, indices < N)


@pytest.mark.parametrize("fault", [False, True])
@pytest.mark.skipif(triton is None, reason="requires the native Triton compiler")
def test_real_fused_kernel_exports_internal_probe(tmp_path, fault):
    EventCapture.prepare(
        tmp_path,
        FusedCapture.boundaries("fused", ["affine"]),
        scope="one instrumented native Triton affine/ReLU kernel; device acceptance",
    )
    x = torch.linspace(-2, 2, 257, device="cuda")
    for side in ("training", "rollout"):
        with EventCapture(tmp_path, side, IDENTITY) as capture:
            observer = FusedCapture(
                capture,
                "fused",
                ["affine"],
                kernel_id="triton.instrumented_affine_relu",
                probe_contract="affine-relu-fp32-v1",
            )
            injected = fault and side == "rollout"
            projected, output = torch.empty_like(x), torch.empty_like(x)

            def execute(emit, projected=projected, output=output, injected=injected):
                _instrumented_affine_relu[(triton.cdiv(x.numel(), 128),)](
                    x,
                    projected,
                    output,
                    x.numel(),
                    injected,
                )
                emit("affine", projected)
                return output

            result = observer.run(execute, inputs=x)
            assert result is output
            assert torch.equal(result, torch.relu(x * 2 + 1 + (0.25 if injected else 0)))
    report = analyze(tmp_path)
    assert report["status"] == ("diverged" if fault else "equal"), report["errors"]
    if fault:
        assert report["first_divergence"]["boundary"] == "fused.affine"
        assert report["first_divergence"]["input_evidence"] == "equal"


def test_module_hooks_survive_real_graph_replay_without_python_hooks(tmp_path):
    EventCapture.prepare(
        tmp_path,
        [
            {"id": f"relu.{stage}", "module": "relu", "stage": stage}
            for stage in ("input", "output")
        ],
        scope="native captured graph with removable module observers",
    )
    for side in ("training", "rollout"):
        model = torch.nn.ReLU().cuda()
        with EventCapture(tmp_path, side, IDENTITY) as capture:
            observer = GraphCapture(capture)
            x = torch.tensor([-1.0, 2.0], device="cuda")
            observer.declare("relu.input", x)
            observer.declare("relu.output", x, inputs=("relu.input",))
            bindings = {"relu": ModuleBoundary("", inputs=lambda m, a, k: a[0])}
            with observer.observe(model, bindings):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        model(x)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = model(x)
            assert not model._forward_pre_hooks and not model._forward_hooks
            for value in (-3.0, 4.0):
                x.fill_(value)
                observer.replay(graph)
                assert torch.equal(output, x.relu())
    report = analyze(tmp_path)
    assert report["status"] == "equal", report["errors"]


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
def test_native_gpu_backward_and_optimizer_preserve_updates(tmp_path, optimizer_type):
    torch.manual_seed(17)
    initial = torch.nn.Linear(4, 2).cuda()
    x = torch.arange(8.0, device="cuda").reshape(2, 4) / 8

    def train(model, optimizer, observer=None):
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(2):
                loss = model(x).square().sum()
                if observer is None:
                    loss.backward()
                else:
                    observer.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    reference = copy.deepcopy(initial)
    train(reference, optimizer_type(reference.parameters(), lr=0.01))
    TrainingCapture.prepare(tmp_path, scope="native GPU backward, clipping and optimizer updates")
    for side in ("training", "rollout"):
        model = copy.deepcopy(initial)
        optimizer = optimizer_type(model.parameters(), lr=0.01)
        with TrainingCapture(tmp_path, side, IDENTITY, model, optimizer) as observer:
            train(model, optimizer, observer)
        assert all(
            torch.equal(a, b)
            for a, b in zip(model.parameters(), reference.parameters(), strict=True)
        )
    report = analyze(tmp_path)
    assert report["status"] == "equal", report["errors"]


@pytest.mark.parametrize("gated", [False, True])
def test_real_megatron_model_checkpoint_roundtrip(tmp_path, monkeypatch, gated):
    """Native GPTModel, process groups and distributed checkpoint, not mocks."""
    pytest.importorskip("megatron.core", reason="requires the native Megatron installation")
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.dist_checkpointing.core import CheckpointingException
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig
    from torch.distributed.checkpoint.api import CheckpointException

    if torch.distributed.is_initialized():
        pytest.skip("checkpoint smoke requires its own isolated process group")
    for key in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(key, raising=False)

    def provider(vocab_size=32):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            params_dtype=torch.float32,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            masked_softmax_fusion=False,
            bias_dropout_fusion=False,
            bias_activation_fusion=False,
            persist_layer_norm=False,
            attention_backend=AttnBackend.local,
            gated_linear_unit=gated,
            activation_func=torch.nn.functional.silu if gated else torch.nn.functional.gelu,
        )
        return (
            GPTModel(
                config,
                get_gpt_layer_local_spec(normalization="LayerNorm"),
                vocab_size=vocab_size,
                max_sequence_length=16,
                position_embedding_type="learned_absolute",
            )
            .cuda()
            .eval()
        )

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path}/rendezvous",
        rank=0,
        world_size=1,
    )
    try:
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(17)
        source = provider()
        tokens = torch.arange(16, device="cuda").reshape(2, 8)
        positions = torch.arange(8, device="cuda").expand_as(tokens)
        mask = torch.triu(torch.ones(1, 1, 8, 8, dtype=torch.bool, device="cuda"), diagonal=1)
        with torch.no_grad():
            expected = source(tokens, positions, mask)
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        dist_checkpointing.save({"model": source.sharded_state_dict()}, str(checkpoint))
        loaded = load_megatron_model(provider, checkpoint, model_key="model")
        with torch.no_grad():
            actual = loaded(tokens, positions, mask)
        assert torch.equal(actual, expected)
        assert all(
            torch.equal(source.state_dict()[name], value)
            for name, value in loaded.state_dict().items()
            if isinstance(value, torch.Tensor)
        )
        left, right = megatron_weight_shards(source), megatron_weight_shards(loaded)
        assert left.keys() == right.keys()
        for name in left:
            a = _merge_trees([encode_tree(left[name])])["value"]
            b = _merge_trees([encode_tree(right[name])])["value"]
            assert torch.equal(a, b), name
        with pytest.raises((CheckpointingException, CheckpointException), match="[Ss]hape"):
            load_megatron_model(
                provider, checkpoint, model_key="model", provider_kwargs={"vocab_size": 64}
            )
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
