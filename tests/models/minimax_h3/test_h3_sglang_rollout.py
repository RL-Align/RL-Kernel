# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``sglang_runtime_adapter_h3``: strict H3 projections inside SGLang's H3 DiT.

The refusal tests use stand-in modules with SGLang's linear attributes and run on CPU. The
tests marked ``sglang`` build SGLang's own modules and skip when ``sglang.multimodal_gen`` is not
importable (it is not a CI dependency).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.engines.rollout.sglang import h3 as adapter
from rl_engine.models.minimax_h3 import checkpoint_mapper as cm
from rl_engine.models.minimax_h3 import fingerprint as fp

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
PROJECTIONS = (
    "blocks.0.attn.qkv_proj",
    "blocks.0.attn.out_proj",
    "blocks.0.mlp.fc1",
    "blocks.0.mlp.fc2",
)


class _Linear(torch.nn.Module):
    """A stand-in for SGLang's parallel linears: their attributes, a forward that must not run."""

    def __init__(self, n, k, *, row, device="cpu", dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (torch.randn(n, k) / 30).to(device=device, dtype=dtype), requires_grad=False
        )
        self.tp_size = 1
        self.quant_method = None
        if row:
            self.input_is_parallel, self.reduce_results = True, True
        else:
            self.gather_output = False

    def forward(self, x):
        raise AssertionError("SGLang's forward")


class _LoRA(torch.nn.Module):
    """Shaped like SGLang's ``BaseLayerWithLoRA``: the base linear plus its own forward."""

    def __init__(self, base):
        super().__init__()
        self.base_layer = base

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, x):
        raise AssertionError("SGLang's LoRA forward")


class Fp8LinearMethod:  # its class name is what the adapter reads
    pass


def _model(device="cpu", dtype=torch.float32):
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    block = model.blocks[0]
    block.attn, block.mlp = torch.nn.Module(), torch.nn.Module()
    kw = dict(device=device, dtype=dtype)
    block.attn.qkv_proj = _Linear(3 * 1024, 640, row=False, **kw)
    block.attn.out_proj = _Linear(640, 1024, row=True, **kw)
    block.mlp.fc1 = _Linear(2 * 2048, 640, row=False, **kw)
    block.mlp.fc2 = _Linear(640, 2048, row=True, **kw)
    return model


def _no_forward_changed(model):
    return all("forward" not in m.__dict__ for m in model.modules())


def _lora(model):
    model.blocks[0].attn.qkv_proj = _LoRA(model.blocks[0].attn.qkv_proj)


def _quantized(model):
    model.blocks[0].mlp.fc1.quant_method = Fp8LinearMethod()


def _other_tp(model):
    model.blocks[0].mlp.fc2.tp_size = 2


def _gathers(model):
    model.blocks[0].attn.qkv_proj.gather_output = True


def _unsharded_input(model):
    model.blocks[0].attn.out_proj.input_is_parallel = False


def _no_reduce(model):
    model.blocks[0].mlp.fc2.reduce_results = False


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_lora, "qkv_proj: a LoRA wrapper"),
        (_quantized, r"fc1: quantized \(Fp8LinearMethod\)"),
        (_other_tp, "fc2: sharded over 2 TP ranks, but tp=1"),
        (_gathers, "qkv_proj: gathers its output"),
        (_unsharded_input, "out_proj: expects unsharded input"),
        (_no_reduce, "fc2: does not reduce its result"),
    ],
)
def test_a_projection_sglang_would_compute_differently_is_refused_before_any_forward_changes(
    mutate, match
):
    model = _model()
    mutate(model)
    # SGLang's own check runs before the dtype test, so the CPU FP32 stand-ins report it.
    with pytest.raises(RuntimeError, match=match):
        adapter.install_h3_sglang_rollout(model)
    assert _no_forward_changed(model)


def test_a_compiled_module_is_refused_even_without_strict():
    model = _model()
    model.blocks[0].compile()  # lazy: sets the compiled call, compiles nothing yet
    for strict in (True, False):
        with pytest.raises(RuntimeError, match="blocks.0: compiled"):
            adapter.install_h3_sglang_rollout(model, strict=strict)
    assert _no_forward_changed(model)


def test_the_vdn_attention_variants_are_refused_in_strict_mode():
    for attribute, match in (("hybrid", "VDN-H3 hybrid"), ("to_gate_compress", "compression")):
        model = _model()
        setattr(model.blocks[0].attn, attribute, torch.nn.Identity())
        with pytest.raises(RuntimeError, match=f"blocks.0.attn: {match}"):
            adapter.install_h3_sglang_rollout(model)
        handle = adapter.install_h3_sglang_rollout(model, strict=False)
        assert handle.patched == []  # CPU FP32 stand-ins: skipped for their dtype
        assert len(handle.skipped) == 4 and _no_forward_changed(model)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"enable_torch_compile": True}, "enable_torch_compile"),
        ({"regional_compile": True}, "regional_compile"),
        ({"enable_breakable_cuda_graph": True}, "enable_breakable_cuda_graph"),
        ({"cache_dit_config": {"Fn": 1}}, "cache_dit_config"),
        ({"quantization": "fp8"}, "quantization"),
        ({"component_quantizations": {"transformer": "fp8"}}, "quantization"),
        ({"lora_path": "adapter"}, "lora_path"),
    ],
)
def test_server_settings_that_change_the_dit_are_refused(settings, match):
    defaults = dict(
        enable_torch_compile=False,
        regional_compile=False,
        enable_breakable_cuda_graph=False,
        cache_dit_config=None,
        quantization=None,
        component_quantizations={},
        lora_path=None,
    )
    adapter.check_sglang_server_args(SimpleNamespace(**defaults))
    with pytest.raises(RuntimeError, match=match):
        adapter.check_sglang_server_args(SimpleNamespace(**{**defaults, **settings}))


@needs_gpu
def test_without_strict_a_refused_projection_keeps_its_forward_and_is_listed():
    pytest.importorskip("triton")
    model = _model("cuda", torch.bfloat16)
    _lora(model)
    with adapter.install_h3_sglang_rollout(model, strict=False) as handle:
        assert sorted(handle.patched) == sorted(PROJECTIONS[1:])
        assert len(handle.skipped) == 1
        assert handle.skipped[0].startswith("blocks.0.attn.qkv_proj: a LoRA wrapper")
        with pytest.raises(AssertionError, match="LoRA forward"):
            model.blocks[0].attn.qkv_proj(torch.zeros(3, 640, device="cuda").bfloat16())
        report = adapter.report(model, handle)
        assert report["engine"] == "sglang" and report["skipped"] == handle.skipped
        assert report["adaln_cache"] is False
    assert _no_forward_changed(model)


# --- SGLang's own modules ---------------------------------------------------------------------


def _sglang_dit():
    return pytest.importorskip("sglang.multimodal_gen.runtime.models.dits.minimax_h3")


def test_sglang_loads_the_checkpoint_as_the_checkpoint_mapper_does():
    sm = _sglang_dit()
    names = list(fp.expected_tensor_table(fp.transformer_config()))
    gen = torch.Generator().manual_seed(0)
    state = {n: torch.randn(6, 2, generator=gen) for n in names}
    ours = cm.diffusers_to_native(state)
    theirs = dict(sm._diffusers_h3_checkpoint(iter(state.items())))
    assert set(theirs) == set(ours)
    assert all(torch.equal(theirs[n], ours[n]) for n in ours)


def _projection_world(world, weights, rows=64, prefix=13):
    from rl_engine.validation.distributed.h3_ws2 import run_world
    from rl_engine.validation.models import h3_sglang_rollout as hs

    gen = torch.Generator().manual_seed(1)
    inputs = {
        "x": (torch.randn(rows, 5376, generator=gen) * 0.5).bfloat16(),
        "a": (torch.randn(rows, 7168, generator=gen) * 0.5).bfloat16(),
        "h": (torch.randn(rows, 14336, generator=gen) * 0.5).bfloat16(),
        **hs.block_weight_inputs({k: v.cpu() for k, v in weights.items()}),
    }
    return hs, run_world(world, hs.projection_target, inputs, prefix_rows=prefix), prefix


@needs_gpu
def test_bound_sglang_projections_equal_the_training_arithmetic_bitwise(h3_block_weights_cuda):
    _sglang_dit()
    hs, (rank,), prefix = _projection_world(1, h3_block_weights_cuda)
    assert rank["report"]["patched"] == list(PROJECTIONS)
    for name in hs.NAMES:
        assert torch.equal(rank[f"bound_{name}"], rank[f"canonical_{name}"]), name
        assert torch.equal(rank[f"bound_prefix_{name}"], rank[f"bound_{name}"][:prefix]), name
        assert rank[f"restored_equals_native_{name}"], name


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_bound_sglang_projections_at_tp2_equal_tp1(h3_block_weights_cuda):
    _sglang_dit()
    hs, (one,), _ = _projection_world(1, h3_block_weights_cuda)
    hs, ranks, _ = _projection_world(2, h3_block_weights_cuda)
    for name in hs.NAMES:
        for rank in ranks:
            assert torch.equal(rank[f"bound_{name}"], rank[f"canonical_{name}"]), name
        if name in ("qkv", "fc1"):
            parts = 3 if name == "qkv" else 2
            split = [r[f"bound_{name}"].chunk(parts, dim=1) for r in ranks]
            whole = torch.cat([torch.cat([s[i] for s in split], 1) for i in range(parts)], 1)
            assert torch.equal(whole, one[f"bound_{name}"]), name
        else:
            assert all(torch.equal(r[f"bound_{name}"], one[f"bound_{name}"]) for r in ranks), name
