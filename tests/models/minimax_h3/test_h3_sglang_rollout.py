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


def _model(device="cpu", dtype=torch.float32):
    kw = dict(device=device, dtype=dtype)
    model = torch.nn.Module()
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
        assert len(handle.skipped) == 7 and _no_forward_changed(model)


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
        for t in (2, 4)
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
