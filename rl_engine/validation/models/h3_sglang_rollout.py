# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validation of ``sglang_runtime_adapter_h3`` on SGLang's own H3 modules (RFC #420).

``rollout_target`` is a ``run_world`` target. On each rank it builds, from ``sglang.multimodal_gen``
at SGLang's TP size, block 0 (``MiniMaxH3DiTBlock``), the final layer, and the token refiner's
norms, loads the pinned Diffusers tensors through SGLang's own loader (``_diffusers_h3_checkpoint``
and each parameter's TP ``weight_loader``), and evaluates every bound operator three ways:

* ``native``: SGLang's own forward;
* ``bound``: SGLang's modules after ``install_h3_sglang_rollout``;
* ``canonical``: the training arithmetic on the Diffusers tensors. For the projections that is
  ``h3-parallel-v2`` on the same rank's shards (``par.column_parallel_linear`` /
  ``par.row_parallel_linear``, as in ``h3_tp.run_tp_block_rows``); for the AdaLN path and the norms
  it is the WS1 operators (``adaln_projection_3mod``, ``h3_rmsnorm``, ``adaln_gate_residual``,
  ``final_adaln_out``), whose TP layouts give the WS1 bytes on every rank.

Column outputs are the rank's shard (``[q; k; v]`` and ``[gate; up]`` of its heads / channels),
row outputs are the full reduced rows. The block runs with stand-ins for its attention and MLP
(``ya`` / ``ym``, recording their inputs), since those are evaluated as projections above or are
external; the final layer's FP32 output heads are replaced by the identity for the same reason.
The refiner blocks' norm weights are not in the pinned block-0 extract and are seeded inputs.
Only CUDA BF16 is meaningful here; ``sglang`` must be importable in the worker.

``full_model_target`` is the whole ``MiniMaxH3DiTModel`` instead: ``layers`` blocks at H3's
widths with seeded weights, one denoising step's packed inputs from SGLang's own request builder
(``minimax_h3_packed_sequence`` and ``MiniMaxH3DenoiseBranch``), and the adapter's readback: that
every bound forward ran once per model forward, so no path of SGLang's model went around it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par

BLOCK = "transformer_blocks.0"
FFN = 14336
NAMES = ("qkv", "out", "fc1", "fc2")
# Outputs of the AdaLN path and the norms; the tables do not depend on the rows.
NORM_NAMES = (
    "adaln_table",
    "norm1",
    "norm2",
    "block_out",
    "final_table",
    "final_out",
    "refiner_norm1",
    "refiner_norm2",
    "refiner_final_norm",
)
TABLES = ("adaln_table", "final_table")


def _sglang_model(weights: dict[str, torch.Tensor], refiner_norms: tuple[torch.Tensor, ...]):
    """SGLang's block 0, final layer and refiner norms, loaded by SGLang's loader."""

    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as sm

    arch = MiniMaxH3DiTArchConfig()
    # The tensors are Diffusers-named: _diffusers_h3_checkpoint fuses q/k/v as [Q; K; V] and
    # swaps the fc1 halves, so SGLang's grouped-row reorder does not apply.
    arch.checkpoint_uses_diffusers_layout = True
    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList([sm.MiniMaxH3DiTBlock(arch, None, prefix="blocks.0")])
    root.final_layer = sm.MiniMaxH3FinalLayer(arch, None, prefix="final_layer")
    refiner, refiner_block = torch.nn.Module(), torch.nn.Module()
    norms = [sm._norm(arch.hidden_size, eps=arch.norm_eps) for _ in refiner_norms]
    refiner_block.norm1, refiner_block.norm2 = norms
    refiner.blocks = torch.nn.ModuleList([refiner_block])
    refiner.final_norm = sm._norm(arch.hidden_size, eps=arch.final_norm_eps)
    root.token_refiner = refiner
    root.cuda()
    with torch.no_grad():
        for norm, weight in zip(norms, refiner_norms, strict=True):
            norm.weight.copy_(weight)
    params = dict(root.named_parameters())
    loaded = []
    for name, tensor in sm._diffusers_h3_checkpoint(iter(weights.items())):
        if name not in params:
            continue
        param = params[name]
        loader = getattr(param, "weight_loader", None)
        if loader is not None:
            loader(param, tensor)
        elif param.shape == tensor.shape:  # SGLang's plain nn.RMSNorm weights
            with torch.no_grad():
                param.copy_(tensor)
        else:
            raise RuntimeError(f"{name}: SGLang parameter without a weight_loader")
        loaded.append(name)
    return root, root.blocks[0], sorted(loaded)


def _canonical(weights, x, a, h, collective, tp, rank):
    w = lambda name: weights[f"{BLOCK}.{name}"]  # noqa: E731
    q, k, v = par.column_parallel_linear(
        x, [par.shard_rows(w(f"attn.to_{n}.weight"), tp, rank) for n in "qkv"], collective, tp
    )
    fused = w("ff.net.0.proj.weight")
    hid, gate = par.column_parallel_linear(
        x,
        [par.shard_rows(fused[:FFN], tp, rank), par.shard_rows(fused[FFN:], tp, rank)],
        collective,
        tp,
        kernel="fixed",
    )
    a_local = par.shard_cols(a, tp, rank).contiguous()
    h_local = par.shard_cols(h, tp, rank).contiguous()
    out = par.row_parallel_linear(
        a_local, par.shard_cols(w("attn.to_out.0.weight"), tp, rank), collective, tp
    )
    fc2 = par.row_parallel_linear(
        h_local, par.shard_cols(w("ff.net.2.weight"), tp, rank), collective, tp
    )
    return {
        "qkv": torch.cat([q, k, v], 1),
        "out": out,
        "fc1": torch.cat([gate, hid], 1),
        "fc2": fc2,
    }, (a_local, h_local)


def _canonical_norms(weights, x, temb, index, final_index, ya, ym, refiner_norms):
    """The WS1 operators of the AdaLN path and the norms on the Diffusers tensors."""

    import torch.nn.functional as F

    from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_projection import (
        H3AdaLNProjectionCudaOp,
    )
    from rl_engine.backends.cuda.model_specific.minimax_h3.det_linear import det_linear_forward
    from rl_engine.backends.cuda.model_specific.minimax_h3.final_adaln_out import (
        H3FinalAdaLNOutCudaOp,
    )
    from rl_engine.backends.cuda.model_specific.minimax_h3.gate_residual import H3GateResidualCudaOp
    from rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm import H3RMSNormCudaOp

    w = lambda name: weights[f"{BLOCK}.{name}"]  # noqa: E731
    norm, gate = H3RMSNormCudaOp(), H3GateResidualCudaOp()
    t = H3AdaLNProjectionCudaOp()(temb, w("adaln_proj.linear.weight"), w("adaln_proj.linear.bias"))
    norm1 = norm.forward_modulated(x, w("norm1.weight"), t[0], t[1], index)
    x1 = gate(x, ya, t[2], index)
    norm2 = norm.forward_modulated(x1, w("norm2.weight"), t[3], t[4], index)
    final_w, final_b = weights["norm_out.linear.weight"], weights["norm_out.linear.bias"]
    (final_table,) = det_linear_forward(F.silu(temb).to(final_w.dtype), final_w, final_b)
    final_out = H3FinalAdaLNOutCudaOp()(
        x, weights["norm_out.norm.weight"], temb, final_w, final_b, final_index
    )
    return {
        "adaln_table": torch.cat(t, dim=1),
        "norm1": norm1,
        "norm2": norm2,
        "block_out": gate(x1, ym, t[5], index),
        "final_table": final_table,
        "final_out": final_out.float(),
        "refiner_norm1": norm(x, refiner_norms[0]),
        "refiner_norm2": norm(x, refiner_norms[1]),
        "refiner_final_norm": norm(x, weights["token_refiner.final_norm.weight"]),
    }


def _evaluate_norms(root, x, adaln_input, index, final_index, ya, ym):
    """SGLang's block 0, final layer and refiner norms as they are now bound (or not)."""

    block, final = root.blocks[0], root.final_layer
    seen = {}

    def attention(h, **kwargs):
        seen["norm1"] = h.clone()
        return ya.clone()

    def mlp(h):
        seen["norm2"] = h.clone()
        return ym.clone()

    block.attn.forward, block.mlp.forward = attention, mlp
    final._project = lambda h: (h, h)
    try:
        block_out = block(
            x.clone(),  # SGLang's gated residual writes into its input
            adaln_input=adaln_input,
            combined_indices=index,
            rope_cache=None,
            cu_seqlens=None,
            max_seqlen=0,
        )
        final_out, _ = final(x.clone(), adaln_input=adaln_input, inverse_indices=final_index)
    finally:
        for module, attribute in (
            (block.attn, "forward"),
            (block.mlp, "forward"),
            (final, "_project"),
        ):
            module.__dict__.pop(attribute, None)
    refiner = root.token_refiner
    return {
        "adaln_table": torch.cat(block.adaln_proj(adaln_input), dim=1),
        "norm1": seen["norm1"],
        "norm2": seen["norm2"],
        "block_out": block_out,
        "final_table": torch.cat(final.adaln_proj(adaln_input), dim=1),
        "final_out": final_out,
        "refiner_norm1": refiner.blocks[0].norm1(x),
        "refiner_norm2": refiner.blocks[0].norm2(x),
        "refiner_final_norm": refiner.final_norm(x),
    }


def _evaluate(block, x, a_local, h_local):
    return {
        "qkv": block.attn.qkv_proj(x)[0],
        "out": block.attn.out_proj(a_local)[0],
        "fc1": block.mlp.fc1(x)[0],
        "fc2": block.mlp.fc2(h_local)[0],
    }


def rollout_target(
    collective,
    *,
    x,
    a,
    h,
    temb,
    index,
    final_index,
    ya,
    ym,
    refiner_norm1,
    refiner_norm2,
    prefix_rows: int = 0,
    **weights,
):
    """One rank: native, bound and canonical outputs of block 0 (see the module docstring).

    ``x (T, 5376)`` feeds ``qkv_proj`` / ``fc1``, the block, the final layer and the refiner
    norms; ``a (T, 7168)`` and ``h (T, 14336)`` are the full inputs of ``out_proj`` / ``fc2``, of
    which each rank takes its TP columns. ``temb (M, 2688)`` is the FP32 timestep embedding,
    ``index`` the block's ``modality * M``-row AdaLN index of each token and ``final_index`` its
    timestep; ``ya`` / ``ym`` stand in for the attention and MLP outputs. With ``prefix_rows``
    the bound and native forwards are also run on the first ``prefix_rows`` rows alone (row
    invariance).
    """

    import torch.distributed as dist
    import torch.nn.functional as F
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    from rl_engine.integrations.engines.rollout.sglang import h3 as adapter

    rank, world = dist.get_rank(), dist.get_world_size()
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    maybe_init_distributed_environment_and_model_parallel(tp_size=world, sp_size=1)
    weights = {name.replace("__", "."): tensor for name, tensor in weights.items()}
    refiner_norms = (refiner_norm1, refiner_norm2)
    root, block, loaded = _sglang_model(weights, refiner_norms)
    tp = world
    coll = collective if tp > 1 else None
    canonical, (a_local, h_local) = _canonical(weights, x, a, h, coll, tp, rank)
    canonical.update(_canonical_norms(weights, x, temb, index, final_index, ya, ym, refiner_norms))
    adaln_input = F.silu(temb).to(torch.bfloat16)  # SGLang's request-step AdaLN input

    def evaluate(rows=None):
        cut = slice(None) if rows is None else slice(0, rows)
        out = _evaluate(block, x[cut], a_local[cut], h_local[cut])
        out.update(
            _evaluate_norms(
                root, x[cut], adaln_input, index[cut], final_index[cut], ya[cut], ym[cut]
            )
        )
        return out

    out: dict = {"loaded": loaded, "tp": tp}
    with torch.no_grad():
        native = evaluate()
        native_prefix = evaluate(prefix_rows) if prefix_rows else None
        with adapter.install_h3_sglang_rollout(root, collective=coll, tp=tp) as handle:
            bound = evaluate()
            bound_prefix = evaluate(prefix_rows) if prefix_rows else None
            out["report"] = adapter.report(root, handle)
        restored = evaluate()
    for name in NAMES + NORM_NAMES:
        out[f"canonical_{name}"] = canonical[name]
        out[f"native_{name}"] = native[name]
        out[f"bound_{name}"] = bound[name]
        out[f"restored_equals_native_{name}"] = bool(torch.equal(restored[name], native[name]))
        if prefix_rows and name not in TABLES:
            assert native_prefix is not None and bound_prefix is not None
            out[f"native_prefix_{name}"] = native_prefix[name]
            out[f"bound_prefix_{name}"] = bound_prefix[name]
    return out


def block_weight_inputs(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The pinned tensors ``rollout_target`` loads, as ``run_world`` inputs (keys without dots)."""

    names = [f"{BLOCK}.attn.to_{n}.weight" for n in "qkv"] + [
        f"{BLOCK}.attn.to_out.0.weight",
        f"{BLOCK}.ff.net.0.proj.weight",
        f"{BLOCK}.ff.net.2.weight",
        f"{BLOCK}.adaln_proj.linear.weight",
        f"{BLOCK}.adaln_proj.linear.bias",
        f"{BLOCK}.norm1.weight",
        f"{BLOCK}.norm2.weight",
        "norm_out.linear.weight",
        "norm_out.linear.bias",
        "norm_out.norm.weight",
        "token_refiner.final_norm.weight",
    ]
    return {name.replace(".", "__"): weights[name] for name in names}


def norm_inputs(rows: int, *, timesteps: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    """Seeded inputs of the AdaLN path and the norms (CPU; ``run_world`` moves them)."""

    hidden, temb_dim, modalities = 5376, 2688, 3
    gen = torch.Generator().manual_seed(seed)
    step = torch.randint(0, timesteps, (rows,), generator=gen)
    modality = torch.randint(0, modalities, (rows,), generator=gen)
    return {
        "temb": torch.randn(timesteps, temb_dim, generator=gen),
        "index": step * modalities + modality,
        "final_index": step,
        "ya": (torch.randn(rows, hidden, generator=gen) * 0.5).bfloat16(),
        "ym": (torch.randn(rows, hidden, generator=gen) * 0.5).bfloat16(),
        "refiner_norm1": (1 + torch.randn(hidden, generator=gen) * 0.1).bfloat16(),
        "refiner_norm2": (1 + torch.randn(hidden, generator=gen) * 0.1).bfloat16(),
    }


def _server_args():
    """The process-wide server settings SGLang's model reads; SGLang picks the attention backend."""

    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime import server_args as sa

    if sa._global_server_args is None:
        sa.set_global_server_args(
            SimpleNamespace(
                attention_backend=None,
                attention_backend_config=None,
                comfyui_mode=False,
                disable_autocast=False,
                enable_cfg_parallel=False,
                enable_breakable_cuda_graph=False,
                enable_layerwise_nvtx_marker=False,
                enable_torch_compile=False,
                kv_gather_degree=1,
                sp_split_auto=False,
                model_loaded={},
                model_paths={},
                pipeline_config=SimpleNamespace(dit_precision="bfloat16", vae_precision="bfloat16"),
            )
        )


def _whole_model(layers: int, seed: int):
    """SGLang's ``MiniMaxH3DiTModel`` with ``layers`` blocks at H3's widths, seeded per rank, and
    one denoising step's forward kwargs packed by SGLang's own request builder."""

    import torch.distributed as dist
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTConfig,
    )
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as sm
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (  # noqa: E501
        MiniMaxH3DenoiseBranch,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (  # noqa: E501
        minimax_h3_packed_sequence,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    config = MiniMaxH3DiTConfig(arch_config=MiniMaxH3DiTArchConfig(num_layers=layers))
    with torch.device(device):
        model = sm.MiniMaxH3DiTModel(config=config, hf_config={}, quant_config=None)
    gen = torch.Generator(device=device).manual_seed(seed + dist.get_rank())
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.shape, generator=gen, device=device, dtype=torch.float32)
            param.copy_((noise * 0.02).to(param.dtype))

    # One denoising step of a small text-to-video+audio request, as SGLang's pipeline packs it.
    cpu = torch.Generator().manual_seed(seed)
    packed = minimax_h3_packed_sequence(
        text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3, include_keyframe_cond=False
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.randn(3, 5120, generator=cpu).to(device),
        token_tags=packed["token_tags"],
        device=device,
    )
    (step,) = branch.prepare_timestep_plan(
        video_timesteps=[500.0],
        audio_timesteps=[500.0],
        imgvid_cond_noise_aug=0.0,
        audio_ref_cond_noise_aug=0.0,
    )
    kwargs = branch.forward_kwargs(
        video_rows=torch.randn(packed["img_pos"].numel(), 96, generator=cpu).to(device),
        audio_rows=torch.randn(packed["audio_pos"].numel(), 32, generator=cpu).to(device),
        step_timesteps=step,
    )

    def run():
        video, audio = model(**kwargs)
        return torch.cat((video.flatten(), audio.flatten())).cpu()

    return model, run


def _init_rank():
    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    maybe_init_distributed_environment_and_model_parallel(tp_size=world, sp_size=1)
    _server_args()
    return rank, world


def full_model_target(collective, *, layers: int = 2, forwards: int = 2, seed: int = 0):
    """One rank: SGLang's whole H3 DiT with the adapter installed (see the module docstring).

    Returns the adapter's report and readback (``forwards`` model forwards), whether the bound
    forwards are finite and repeat bitwise, and whether ``remove()`` restores SGLang's output.
    The weights are seeded per rank (each rank's own shards), so outputs are not compared across
    TP sizes; the operators' bytes are compared by ``rollout_target``.
    """

    from rl_engine.integrations.engines.rollout.sglang import h3 as adapter

    _, world = _init_rank()
    model, run = _whole_model(layers, seed)
    out: dict = {"tp": world, "layers": layers, "forwards": forwards}
    with torch.no_grad():
        native = run()
        coll = collective if world > 1 else None
        with adapter.install_h3_sglang_rollout(model, collective=coll, tp=world) as handle:
            bound = [run() for _ in range(forwards)]
            out["report"] = adapter.report(model, handle)
            out["readback"] = handle.readback(forwards)
        restored = run()
    out["bound_finite"] = bool(torch.isfinite(bound[0]).all())
    out["bound_repeatable"] = all(torch.equal(b, bound[0]) for b in bound)
    out["restored_equals_native"] = bool(torch.equal(restored, native))
    out["batching_restored"] = "_can_batch_block_adaln" not in model.__dict__
    out["native_max_abs_diff"] = (bound[0].double() - native.double()).abs().max().item()
    return out


def plugin_target(
    collective, *, plugin_dir: str, layers: int = 2, forwards: int = 2, seed: int = 0
):
    """One rank: the SGLang plugin (``plugin.py``) as SGLang's worker bootstrap runs it.

    ``plugin_dir`` holds a ``*.dist-info`` that declares RL-Kernel's entry point in the group
    ``sglang.multimodal_gen.plugins``, as an install of RL-Kernel does. With
    ``RL_KERNEL_SGLANG_H3_STRICT=1`` this rank runs SGLang's own ``load_plugins()`` and
    ``apply_plugin_hooks()``, then calls the hooked ``TransformerLoader.load_customized``. Only the
    loader's body is replaced, by one that returns the seeded whole DiT (``_whole_model``): there is
    no H3 checkpoint here. ``collective`` is unused: the plugin builds its own collective over
    SGLang's TP group.

    Then: ``forwards`` guarded forwards, the readback file, a request with Spectrum on (refused
    before the forward), and a bound forward put back to SGLang's own (the guard's readback stops
    the forward).
    """

    import importlib
    import sys
    from types import SimpleNamespace

    from rl_engine.integrations.engines.rollout.sglang import plugin

    rank, world = _init_rank()
    sys.path.insert(0, plugin_dir)
    importlib.invalidate_caches()
    os.environ[plugin.ENABLE] = "1"
    readback = os.path.join(plugin_dir, f"readback-tp{world}")
    os.environ[plugin.READBACK_DIR] = readback

    from sglang.multimodal_gen.runtime.loader.component_loaders import transformer_loader as tl
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.runtime.platforms import plugins as sp

    built: dict = {}

    def load_customized(
        self, component_model_path, server_args, component_name, cpu_offload_flag=False
    ):
        model, run = _whole_model(layers, seed)
        built["run"] = run
        return model

    tl.TransformerLoader.load_customized = load_customized
    sp.load_plugins()
    sp.apply_plugin_hooks()
    out: dict = {"tp": world, "hooked": plugin.TARGET in sp.HookRegistry._patched}
    loader = object.__new__(tl.TransformerLoader)
    server_args = SimpleNamespace(dit_cpu_offload=False, use_fsdp_inference=False)
    with torch.no_grad():
        model = loader.load_customized("unused", server_args, "transformer")
        handle = getattr(model, plugin.HANDLE, None)
        out["installed"] = handle is not None
        if handle is None:
            return out
        run = built["run"]
        outputs = [run() for _ in range(forwards)]
        out["forwards"] = handle.forwards
        out["readback"] = handle.readback(forwards)
        out["repeatable"] = all(torch.equal(o, outputs[0]) for o in outputs)
        written = json.loads(Path(readback, f"rank{rank}.json").read_text())
        out["readback_file_forwards"] = written["forwards"]
        out["readback_file_calls"] = sorted(set(written["calls"].values()))
        out["overrides"] = written["overrides"]
        with set_forward_context(0, None, SimpleNamespace(enable_spectrum=True)):
            try:
                run()
                out["spectrum_refused"] = None
            except RuntimeError as error:
                out["spectrum_refused"] = str(error)
        out["forwards_after_refusal"] = handle.forwards
        # A bound forward that SGLang's model would bypass: put SGLang's own back on one module.
        del model.final_layer.__dict__["forward"]
        try:
            run()
            out["bypass_caught"] = None
        except RuntimeError as error:
            out["bypass_caught"] = str(error)
        handle.remove()
        out["forward_restored"] = "forward" not in model.__dict__
    return out


def write_plugin_dist(directory: str) -> str:
    """A ``*.dist-info`` declaring RL-Kernel's SGLang entry point, as ``pyproject.toml`` does."""

    root = Path(directory)
    info = root / "rl_kernel_sglang_plugin_test-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rl-kernel-sglang-plugin-test\nVersion: 0.0\n"
    )
    (info / "entry_points.txt").write_text(
        "[sglang.multimodal_gen.plugins]\n"
        "rl_kernel_h3 = "
        "rl_engine.integrations.engines.rollout.sglang.plugin:register_sglang_plugin\n"
    )
    return str(root)
