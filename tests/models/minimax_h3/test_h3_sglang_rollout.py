# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``sglang_runtime_adapter_h3``: strict H3 operators inside SGLang's H3 DiT.

The refusal tests use stand-in modules with SGLang's class names and linear attributes and run on
CPU. The
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
TESTED = adapter.TESTED_SGLANG_VERSIONS[0]


@pytest.fixture(autouse=True)
def _validated_sglang(monkeypatch):
    """The stand-ins run as if a validated SGLang were installed (CI has none).

    The SGLang tests below install the adapter in ``run_world``'s spawned ranks, which check the
    SGLang that is really installed.
    """

    monkeypatch.setattr(adapter, "_sglang_version", lambda: TESTED)


PROJECTIONS = (
    "blocks.0.attn.qkv_proj",
    "blocks.0.attn.out_proj",
    "blocks.0.mlp.fc1",
    "blocks.0.mlp.fc2",
)
NORMS = (
    "blocks.0.adaln_proj.linear",
    "blocks.0",
    "final_layer.adaln_proj.linear",
    "final_layer",
    "token_refiner.final_norm",
)
SGLANG_BOUND = (
    PROJECTIONS
    + NORMS[:4]
    + (
        "token_refiner.blocks.0.norm1",
        "token_refiner.blocks.0.norm2",
        "token_refiner.final_norm",
    )
)


class _Linear(torch.nn.Module):
    """A stand-in for SGLang's parallel linears: their attributes, a forward that must not run."""

    def __init__(self, n, k, *, row, bias=False, device="cpu", dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (torch.randn(n, k) / 30).to(device=device, dtype=dtype), requires_grad=False
        )
        if bias:
            self.bias = torch.nn.Parameter(
                (torch.randn(n) / 30).to(device=device, dtype=dtype), requires_grad=False
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


HIDDEN, TEMB = 640, 64


class _AdalnProj(torch.nn.Module):
    """SGLang's ``MiniMaxH3AdalnProj`` at TP1: linear, then ``view(M * modalities, E * H)``."""

    def __init__(self, expand, modalities, **kw):
        super().__init__()
        self.expand, self.modalities = expand, modalities
        self.linear = _Linear(expand * HIDDEN * modalities, TEMB, row=False, bias=True, **kw)

    def forward(self, adaln_input):
        x, _ = self.linear(adaln_input)
        x = x.view(x.shape[0] * self.modalities, self.expand * HIDDEN)
        return tuple(x.chunk(self.expand, dim=-1))


class _Attention(torch.nn.Module):
    """Holds the block's projections; its forward stands in for SGLang's attention."""

    def forward(self, h, *, x_prequant=None, **kwargs):
        assert x_prequant is None and kwargs == {"max_seqlen": 7}
        return h * 0.5


class _MLP(torch.nn.Module):
    def forward(self, h):
        return h.flip(-1)


def _norm(device, dtype):
    norm = torch.nn.RMSNorm(HIDDEN, eps=1e-5, device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(1 + torch.randn(HIDDEN) / 10)
    return norm


class MiniMaxH3DiTBlock(torch.nn.Module):  # the adapter finds blocks by SGLang's class name
    def __init__(self, **kw):
        super().__init__()
        self.norm1, self.norm2 = _norm(**kw), _norm(**kw)
        self.attn, self.mlp = _Attention(), _MLP()
        self.attn.qkv_proj = _Linear(3 * 1024, HIDDEN, row=False, **kw)
        self.attn.out_proj = _Linear(HIDDEN, 1024, row=True, **kw)
        self.mlp.fc1 = _Linear(2 * 2048, HIDDEN, row=False, **kw)
        self.mlp.fc2 = _Linear(HIDDEN, 2048, row=True, **kw)
        self.adaln_proj = _AdalnProj(6, 3, **kw)

    def forward(self, x, **kwargs):
        raise AssertionError("SGLang's block forward")


class MiniMaxH3FinalLayer(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.norm = _norm(**kw)
        self.adaln_proj = _AdalnProj(2, 1, **kw)
        self._pdd_heads = None

    def _project(self, h):
        return h, h.sum(-1)

    def forward(self, x, **kwargs):
        raise AssertionError("SGLang's final layer forward")


class _DiT(torch.nn.Module):
    """A stand-in for ``MiniMaxH3DiTModel``'s switch for batching the blocks' AdaLN under TP."""

    def _can_batch_block_adaln(self):
        return True


def _model(device="cpu", dtype=torch.float32):
    kw = dict(device=device, dtype=dtype)
    model = _DiT()
    model.blocks = torch.nn.ModuleList([MiniMaxH3DiTBlock(**kw)])
    model.final_layer = MiniMaxH3FinalLayer(**kw)
    model.token_refiner = torch.nn.Module()
    model.token_refiner.final_norm = _norm(**kw)
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


@pytest.mark.parametrize(
    ("version", "match"),
    [("0.5.22", "SGLang 0.5.22 is installed"), (None, "SGLang is not installed")],
)
def test_an_sglang_version_the_bound_forwards_were_not_validated_on_is_refused(
    monkeypatch, version, match
):
    monkeypatch.setattr(adapter, "_sglang_version", lambda: version)
    model = _model()
    for strict in (True, False):
        with pytest.raises(RuntimeError, match=f"{match}, but the bound forwards are validated"):
            adapter.install_h3_sglang_rollout(model, strict=strict)
    assert _no_forward_changed(model)


def test_sampling_settings_that_skip_dit_steps_are_refused():
    adapter.check_sglang_sampling_params(
        SimpleNamespace(enable_spectrum=False, enable_teacache=False)
    )
    for flag in ("enable_spectrum", "enable_teacache"):
        with pytest.raises(RuntimeError, match=f"{flag} is set"):
            adapter.check_sglang_sampling_params(SimpleNamespace(**{flag: True}))


def test_a_compiled_module_is_refused_even_without_strict():
    model = _model()
    model.blocks[0].compile()  # lazy: sets the compiled call, compiles nothing yet
    for strict in (True, False):
        with pytest.raises(RuntimeError, match="blocks.0: compiled"):
            adapter.install_h3_sglang_rollout(model, strict=strict)
    assert _no_forward_changed(model)


def test_the_vdn_attention_variants_are_refused_in_strict_mode_and_listed_without():
    for attribute, match in (("hybrid", "VDN-H3 hybrid"), ("to_gate_compress", "compression")):
        model = _model()
        setattr(model.blocks[0].attn, attribute, torch.nn.Identity())
        with pytest.raises(RuntimeError, match=f"blocks.0.attn: {match}"):
            adapter.install_h3_sglang_rollout(model)
        handle = adapter.install_h3_sglang_rollout(model, strict=False)
        assert handle.patched == []  # CPU FP32 stand-ins: skipped for their dtype
        assert handle.skipped[0].startswith(f"blocks.0.attn: {match}")
        assert len(handle.skipped) == 8 and _no_forward_changed(model)


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
        ({"dit_cpu_offload": True}, "dit_cpu_offload"),
        ({"dit_layerwise_offload": True}, "dit_layerwise_offload"),
        ({"use_fsdp_inference": True}, "use_fsdp_inference"),
        ({"cpu_offload_components": ["transformer"]}, "cpu_offload_components"),
        ({"layerwise_offload_components": ["transformer"]}, "layerwise_offload_components"),
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
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        cpu_offload_components=["text_encoder"],
        layerwise_offload_components=None,
    )
    adapter.check_sglang_server_args(SimpleNamespace(**defaults))
    with pytest.raises(RuntimeError, match=match):
        adapter.check_sglang_server_args(SimpleNamespace(**{**defaults, **settings}))


def _adaln_cache(model):
    model.blocks[0].adaln_proj = None


def _fp32_adaln(model):
    pass  # the CPU stand-ins are FP32


def _adaln_lora(model):
    model.final_layer.adaln_proj.linear = _LoRA(model.final_layer.adaln_proj.linear)


def _adaln_quantized(model):
    model.blocks[0].adaln_proj.linear.quant_method = Fp8LinearMethod()


def _mxfp8_block(model):
    model.blocks[0].attn.qkv_proj.quant_method = Fp8LinearMethod()


def _pdd(model):
    model.final_layer._pdd_heads = {}


def _no_eps(model):
    model.token_refiner.final_norm.eps = None


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_adaln_cache, "blocks.0: the AdaLN cache replaces adaln_proj"),
        (_fp32_adaln, "blocks.0: .*FP32 AdaLN weights"),
        (_adaln_lora, "final_layer: adaln_proj.linear: a LoRA wrapper"),
        (_adaln_quantized, r"blocks.0: adaln_proj.linear: quantized \(Fp8LinearMethod\)"),
        (_mxfp8_block, "blocks.0: .*attn.qkv_proj is quantized .*mxfp8 modulation path"),
        (_pdd, "final_layer: .*PDD distilled output heads"),
        (_no_eps, "token_refiner.final_norm: .*no explicit eps"),
    ],
)
def test_an_adaln_path_or_norm_sglang_would_compute_differently_is_refused(mutate, match):
    model = _model()
    mutate(model)
    with pytest.raises(RuntimeError, match=match):
        adapter.install_h3_sglang_rollout(model)
    assert _no_forward_changed(model)


@needs_gpu
def test_without_strict_a_refused_projection_keeps_its_forward_and_is_listed():
    pytest.importorskip("triton")
    model = _model("cuda", torch.bfloat16)
    _lora(model)
    model.blocks[0].adaln_proj = None  # SGLang's AdaLN cache
    with adapter.install_h3_sglang_rollout(model, strict=False) as handle:
        assert sorted(handle.patched) == sorted(PROJECTIONS[1:] + NORMS[2:])
        assert len(handle.skipped) == 2
        assert handle.skipped[0].startswith("blocks.0.attn.qkv_proj: a LoRA wrapper")
        assert handle.skipped[1].startswith("blocks.0: the AdaLN cache")
        with pytest.raises(AssertionError, match="LoRA forward"):
            model.blocks[0].attn.qkv_proj(torch.zeros(3, 640, device="cuda").bfloat16())
        with pytest.raises(AssertionError, match="block forward"):
            model.blocks[0](torch.zeros(3, 640, device="cuda").bfloat16())
        report = adapter.report(model, handle)
        assert report["engine"] == "sglang" and report["skipped"] == handle.skipped
        assert report["external"] == adapter.EXTERNAL and report["sglang"] == TESTED
    assert _no_forward_changed(model)


@needs_gpu
def test_a_strict_refusal_of_an_adaln_path_unbinds_the_projections_it_had_bound():
    pytest.importorskip("triton")
    model = _model("cuda", torch.bfloat16)
    model.final_layer._pdd_heads = {}
    with pytest.raises(RuntimeError, match="final_layer: PDD"):
        adapter.install_h3_sglang_rollout(model)
    assert _no_forward_changed(model)


@needs_gpu
def test_an_operator_that_cannot_be_built_leaves_no_forward_changed(monkeypatch):
    """``_Ops()`` raises when ``rl_engine._C`` lacks the H3 norm or gate symbols."""

    pytest.importorskip("triton")

    def missing():
        raise RuntimeError("rl_engine._C lacks h3_gate_residual_*")

    monkeypatch.setattr(adapter, "_Ops", missing)
    model = _model("cuda", torch.bfloat16)
    with pytest.raises(RuntimeError, match="lacks h3_gate_residual"):
        adapter.install_h3_sglang_rollout(model)
    assert _no_forward_changed(model)


@needs_gpu
def test_the_bound_adaln_paths_and_norms_are_the_training_operators():
    pytest.importorskip("triton")
    from rl_engine.backends.cuda.model_specific.minimax_h3.det_linear import det_linear_forward
    from rl_engine.backends.cuda.model_specific.minimax_h3.gate_residual import H3GateResidualCudaOp
    from rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm import H3RMSNormCudaOp

    torch.manual_seed(0)
    model = _model("cuda", torch.bfloat16)
    block, final = model.blocks[0], model.final_layer
    rows, steps = 37, 2
    x = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
    adaln_input = torch.randn(steps, TEMB, device="cuda").bfloat16()
    step = torch.randint(0, steps, (rows,), device="cuda")
    index = step * 3 + torch.randint(0, 3, (rows,), device="cuda")
    norm, gate = H3RMSNormCudaOp(), H3GateResidualCudaOp()

    def table(linear, expand, modalities):
        (out,) = det_linear_forward(adaln_input, linear.weight, linear.bias)
        return out.view(steps * modalities, expand * HIDDEN).chunk(expand, dim=-1)

    t = table(block.adaln_proj.linear, 6, 3)
    x1 = gate(
        x, norm.forward_modulated(x, block.norm1.weight, t[0], t[1], index) * 0.5, t[2], index
    )
    h2 = norm.forward_modulated(x1, block.norm2.weight, t[3], t[4], index)
    expected_block = gate(x1, h2.flip(-1), t[5], index)
    shift, scale = table(final.adaln_proj.linear, 2, 1)
    expected_final = norm.forward_modulated(x, final.norm.weight, shift, scale, step).float()

    with adapter.install_h3_sglang_rollout(model) as handle:
        assert sorted(handle.patched) == sorted(PROJECTIONS + NORMS)
        x_before = x.clone()
        out = block(x, adaln_input=adaln_input, combined_indices=index, max_seqlen=7)
        assert torch.equal(out, expected_block) and torch.equal(x, x_before)
        video, _ = final(x, adaln_input=adaln_input, inverse_indices=step)
        assert torch.equal(video, expected_final)
        refiner = model.token_refiner.final_norm
        assert torch.equal(refiner(x), norm(x, refiner.weight, refiner.eps))
        with pytest.raises(RuntimeError, match="precomputed AdaLN tables"):
            block(x, adaln_input=adaln_input, combined_indices=index, adaln_params=t)
    assert _no_forward_changed(model)


def test_the_sglang_plugin_is_declared_and_inactive_unless_enabled(monkeypatch):
    from pathlib import Path

    from rl_engine.integrations.engines.rollout.sglang import plugin

    pyproject = (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text()
    assert '[project.entry-points."sglang.multimodal_gen.plugins"]' in pyproject
    entry = f"{plugin.register_sglang_plugin.__module__}:{plugin.register_sglang_plugin.__name__}"
    assert f'rl_kernel_h3 = "{entry}"' in pyproject
    monkeypatch.delenv(plugin.ENABLE, raising=False)
    assert not plugin.enabled()
    plugin.register_sglang_plugin()  # inactive: imports nothing of SGLang (absent in CI)
    monkeypatch.setenv(plugin.ENABLE, "1")
    assert plugin.enabled()


class _Step(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fail = False

    def forward(self, x):
        if self.fail:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return x + 1


class _TwoSteps(torch.nn.Module):
    """A model whose forward runs two bound modules in turn."""

    def __init__(self):
        super().__init__()
        self.first, self.second = _Step(), _Step()

    def forward(self, x):
        return self.second(self.first(x))


def _guarded(monkeypatch, tmp_path):
    """The plugin's guarded forward on a CPU model; SGLang's TP size and forward context are
    stubbed, and the install binds the two modules as ``install_h3_sglang_rollout`` does."""

    import sys

    from rl_engine.integrations.engines.rollout.sglang import plugin

    state = "sglang.multimodal_gen.runtime.distributed.parallel_state"
    monkeypatch.setitem(sys.modules, state, SimpleNamespace(get_tp_world_size=lambda: 1))
    monkeypatch.setattr(plugin, "_forward_batch", lambda: None)

    def install(model, *, collective, tp):
        handle = adapter.H3SGLangRolloutHandle()
        for name in ("first", "second"):
            module = model.get_submodule(name)
            handle._restore.append((module, module.__dict__.get("forward", adapter._ABSENT)))
            module.forward = adapter._counted(module.forward, handle.calls, name)
            handle.patched.append(name)
        return handle

    monkeypatch.setattr(adapter, "install_h3_sglang_rollout", install)
    model = _TwoSteps()
    handle = plugin.install_in_worker(model, SimpleNamespace(), readback_dir=str(tmp_path))
    return plugin, model, handle


def test_a_failed_forward_does_not_fail_the_readback_of_later_forwards(monkeypatch, tmp_path):
    """SGLang's worker logs a failed request (an OOM, say) and serves the next one: the calls of
    the bound modules that ran before the failure are not counted."""

    _, model, handle = _guarded(monkeypatch, tmp_path)
    x = torch.zeros(2)
    model(x)
    model.second.fail = True
    with pytest.raises(torch.OutOfMemoryError):
        model(x)
    assert handle.forwards == 1 and dict(handle.calls) == {"first": 1, "second": 1}
    model.second.fail = False
    model(x)
    assert handle.forwards == 2 and handle.readback(2) == []


def test_the_handle_leaves_the_model_with_remove(monkeypatch, tmp_path):
    plugin, model, handle = _guarded(monkeypatch, tmp_path)
    assert getattr(model, plugin.HANDLE) is handle
    handle.remove()
    assert not hasattr(model, plugin.HANDLE) and _no_forward_changed(model)


@needs_gpu
def test_the_readback_counts_every_bound_forward_and_the_batching_switch_is_put_back():
    pytest.importorskip("triton")
    model = _model("cuda", torch.bfloat16)
    x = torch.randn(5, HIDDEN, device="cuda").bfloat16()
    with adapter.install_h3_sglang_rollout(model) as handle:
        assert model._can_batch_block_adaln() is False and len(handle.overrides) == 1
        assert handle.readback(0) == []
        model.token_refiner.final_norm(x)
        model.blocks[0].attn.qkv_proj(x)
        model.blocks[0].attn.qkv_proj(x)
        assert handle.calls["token_refiner.final_norm"] == 1
        assert handle.calls["blocks.0.attn.qkv_proj"] == 2
        missing = handle.readback(1)
        assert len(missing) == len(handle.patched) - 1
        assert "blocks.0.attn.qkv_proj: ran 2 times in 1 forwards" in missing
        with pytest.raises(RuntimeError, match="blocks.0: ran 0 times in 1 forwards"):
            handle.assert_ran(1)
        report = adapter.report(model, handle)
        implementations = report["implementations"]
        assert report["calls"]["blocks.0.attn.qkv_proj"] == 2
        assert implementations["blocks.0.attn.qkv_proj"] == adapter.IMPLEMENTATIONS["column"]
        assert implementations["blocks.0.mlp.fc1"] == adapter.IMPLEMENTATIONS["column_fixed"]
        assert implementations["blocks.0.mlp.fc2"] == adapter.IMPLEMENTATIONS["row"]
        assert implementations["blocks.0.adaln_proj.linear"] == adapter.IMPLEMENTATIONS["adaln"]
        assert implementations["blocks.0"] == adapter.IMPLEMENTATIONS["block"]
        assert implementations["final_layer"] == adapter.IMPLEMENTATIONS["final"]
        assert report["overrides"] == handle.overrides
    assert model._can_batch_block_adaln() is True
    assert "_can_batch_block_adaln" not in model.__dict__ and _no_forward_changed(model)


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


def _rollout_world(world, weights, rows=64, prefix=13):
    from rl_engine.validation.distributed.h3_ws2 import run_world
    from rl_engine.validation.models import h3_sglang_rollout as hs

    gen = torch.Generator().manual_seed(1)
    inputs = {
        "x": (torch.randn(rows, 5376, generator=gen) * 0.5).bfloat16(),
        "a": (torch.randn(rows, 7168, generator=gen) * 0.5).bfloat16(),
        "h": (torch.randn(rows, 14336, generator=gen) * 0.5).bfloat16(),
        **hs.block_weight_inputs({k: v.cpu() for k, v in weights.items()}),
        **hs.norm_inputs(rows, seed=1),
    }
    return hs, run_world(world, hs.rollout_target, inputs, prefix_rows=prefix), prefix


@needs_gpu
def test_bound_sglang_operators_equal_the_training_arithmetic_bitwise(h3_block_weights_cuda):
    _sglang_dit()
    hs, (rank,), prefix = _rollout_world(1, h3_block_weights_cuda)
    assert rank["report"]["patched"] == list(SGLANG_BOUND)
    for name in hs.NAMES + hs.NORM_NAMES:
        assert torch.equal(rank[f"bound_{name}"], rank[f"canonical_{name}"]), name
        if name not in hs.TABLES:
            bound = rank[f"bound_{name}"]
            assert torch.equal(rank[f"bound_prefix_{name}"], bound[:prefix]), name
        assert rank[f"restored_equals_native_{name}"], name


@pytest.mark.parametrize(
    "tp",
    [
        pytest.param(
            t, marks=pytest.mark.skipif(torch.cuda.device_count() < t, reason=f"needs {t} GPUs")
        )
        for t in (2, 4, 8)
    ],
)
def test_bound_sglang_operators_at_tp_n_equal_tp1(h3_block_weights_cuda, tp):
    _sglang_dit()
    hs, (one,), _ = _rollout_world(1, h3_block_weights_cuda)
    hs, ranks, _ = _rollout_world(tp, h3_block_weights_cuda)
    for name in hs.NAMES + hs.NORM_NAMES:
        for rank in ranks:
            assert torch.equal(rank[f"bound_{name}"], rank[f"canonical_{name}"]), name
        if name in ("qkv", "fc1"):
            parts = 3 if name == "qkv" else 2
            split = [r[f"bound_{name}"].chunk(parts, dim=1) for r in ranks]
            whole = torch.cat([torch.cat([s[i] for s in split], 1) for i in range(parts)], 1)
            assert torch.equal(whole, one[f"bound_{name}"]), name
        else:
            assert all(torch.equal(r[f"bound_{name}"], one[f"bound_{name}"]) for r in ranks), name


@pytest.mark.parametrize(
    "tp",
    [
        pytest.param(
            t, marks=pytest.mark.skipif(torch.cuda.device_count() < t, reason=f"needs {t} GPUs")
        )
        for t in (1, 2)
    ],
)
def test_the_adapter_binds_sglangs_whole_dit_and_every_bound_forward_runs(tp):
    """SGLang's ``MiniMaxH3DiTModel`` (2 blocks at H3's widths, seeded weights) on one denoising
    step packed by SGLang's request builder: all 27 bound forwards run once per model forward.
    Under TP this needs the batching switch off: SGLang's batched AdaLN path would hand the blocks
    precomputed tables."""

    _sglang_dit()
    from rl_engine.validation.distributed.h3_ws2 import run_world
    from rl_engine.validation.models import h3_sglang_rollout as hs

    for rank in run_world(tp, hs.full_model_target, {}, layers=2, forwards=2):
        report = rank["report"]
        assert rank["readback"] == [] and set(report["calls"].values()) == {2}
        assert len(report["patched"]) == 27 and report["skipped"] == []
        assert report["sglang"] == TESTED and len(report["overrides"]) == 1
        assert rank["bound_finite"] and rank["bound_repeatable"]
        assert rank["restored_equals_native"] and rank["batching_restored"]


@pytest.mark.parametrize(
    "tp",
    [
        pytest.param(
            t, marks=pytest.mark.skipif(torch.cuda.device_count() < t, reason=f"needs {t} GPUs")
        )
        for t in (1, 2)
    ],
)
def test_the_sglang_plugin_installs_the_adapter_in_sglangs_worker_lifecycle(tmp_path, tp):
    """RL-Kernel's entry point, found by SGLang's own ``load_plugins()`` / ``apply_plugin_hooks()``,
    hooks ``TransformerLoader.load_customized`` (its body returns the seeded whole DiT: no H3
    checkpoint here). The guard refuses Spectrum before a forward and stops a forward whose bound
    path SGLang went around."""

    _sglang_dit()
    from rl_engine.validation.distributed.h3_ws2 import run_world
    from rl_engine.validation.models import h3_sglang_rollout as hs

    plugin_dir = hs.write_plugin_dist(str(tmp_path))
    for rank in run_world(tp, hs.plugin_target, {}, plugin_dir=plugin_dir, forwards=2):
        assert rank["hooked"] and rank["installed"], rank
        assert rank["forwards"] == 2 and rank["readback"] == [] and rank["repeatable"]
        assert rank["readback_file_forwards"] == 2 and rank["readback_file_calls"] == [2]
        assert len(rank["overrides"]) == 2
        assert "enable_spectrum is set" in rank["spectrum_refused"]
        assert rank["forwards_after_refusal"] == 2
        assert "final_layer: ran 2 times in 3 forwards" in rank["bypass_caught"]
        assert rank["forward_restored"]


@needs_gpu
def test_the_bound_fc1_computes_the_gate_up_rows_fixed_kernel():
    """The label alone cannot tell: the bound ``fc1`` output must be the fixed GEMM's."""

    pytest.importorskip("triton")
    from rl_engine.backends.cuda.model_specific.minimax_h3 import fixed_fp32_gemm as fx

    model = _model("cuda", torch.bfloat16)
    fc1 = model.blocks[0].mlp.fc1
    x = torch.randn(5, fc1.weight.shape[1], device="cuda").bfloat16()
    with adapter.install_h3_sglang_rollout(model):
        out = fc1(x)
        out = out[0] if isinstance(out, tuple) else out
        assert torch.equal(out, fx.gemm(x, fc1.weight, b_t=True))
