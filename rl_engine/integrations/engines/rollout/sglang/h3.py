# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``sglang_runtime_adapter_h3``: strict H3 projections inside SGLang's H3 DiT.

SGLang (``sglang.multimodal_gen``, checked at 0.5.21) runs MiniMax-H3 as ``MiniMaxH3DiTModel``.
Its block and token-refiner projections have the paths, fused layouts and ``(output, bias)``
return of vLLM-Omni's, and SGLang's loader maps the Diffusers checkpoint the way
``models/minimax_h3/checkpoint_mapper`` does:

* ``attn.qkv_proj``: ``[Q; K; V]`` rows (``MergedColumnParallelLinear``, each of Q, K, V sharded
  by heads under TP);
* ``mlp.fc1``: ``[gate; up]`` rows (``MergedColumnParallelLinear``);
* ``attn.out_proj`` / ``mlp.fc2``: ``RowParallelLinear``, all-reduced by SGLang.

``install_h3_sglang_rollout`` binds the kernels of ``install_h3_rollout`` (``vllm/h3_omni.py``):
``qkv_proj`` / ``fc1`` on ``h3-projection-gemm-v1``, ``out_proj`` / ``fc2`` on the canonical
row-parallel reduction of ``h3-parallel-v1``, the arithmetic of the strict training block
(``h3_tp.run_tp_block_rows``). Before any forward changes, it refuses what SGLang can put in the
way of a bound forward:

* a compiled module (``torch.compile`` / ``nn.Module.compile``): the compiled graph keeps
  running the forward it traced;
* a LoRA wrapper (``BaseLayerWithLoRA``) at a projection path: it adds the LoRA output in its own
  forward, which a bound forward would drop;
* a quantized linear: its weight is not the BF16 checkpoint tensor, and mxfp8 hands the
  projection pre-quantized input;
* a module sharded over another TP size than ``tp``, a column-parallel module that gathers its
  output, or a row-parallel module that does not expect sharded input or reduce its result;
* the VDN-H3 hybrid attention and the compression gate: these are not the released dense H3.

``check_sglang_server_args`` refuses the server settings that change what the DiT computes or
bypass a bound forward: ``enable_torch_compile`` / ``regional_compile``,
``enable_breakable_cuda_graph``, ``cache_dit_config``, quantization and an unmerged LoRA.

Attention, rotary, QK norm, SwiGLU and AdaLN stay SGLang's and are reported ``external``; the
AdaLN cache (``adaln_proj is None``) is reported, not refused, because it replaces AdaLN, which
is external either way. ``handle.remove()`` restores every forward.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.integrations.engines.rollout.vllm.h3_omni import H3RolloutHandle, install_h3_rollout

ENGINE = "sglang"


def _compiled(module: torch.nn.Module) -> bool:
    if getattr(module, "_compiled_call_impl", None) is not None:  # nn.Module.compile()
        return True
    return type(module).__name__ == "OptimizedModule"  # torch.compile(module)


def _model_refusals(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    """``(compiled, variant)``: modules whose bound forward would not run, and non-H3 attention."""

    compiled, variant = [], []
    for name, module in model.named_modules():
        where = name or "<model>"
        if _compiled(module):
            compiled.append(f"{where}: compiled; the compiled graph keeps the traced forward")
        if name.endswith("attn"):
            if getattr(module, "hybrid", None) is not None:
                variant.append(f"{where}: VDN-H3 hybrid attention, not the released dense H3")
            if getattr(module, "to_gate_compress", None) is not None:
                variant.append(f"{where}: compression gate, not the released dense H3")
    return compiled, variant


def _check_projection(tp: int):
    def check(name: str, module: torch.nn.Module, kind: str) -> str | None:
        if hasattr(module, "base_layer"):
            return (
                "a LoRA wrapper adds the LoRA output in its own forward; merge the adapter into "
                "the checkpoint first"
            )
        method = getattr(module, "quant_method", None)
        if method is not None and type(method).__name__ != "UnquantizedLinearMethod":
            return f"quantized ({type(method).__name__}); the strict kernels read BF16 weights"
        shards = getattr(module, "tp_size", 1)
        if shards != tp:
            return f"sharded over {shards} TP ranks, but tp={tp}"
        if kind == "column" and getattr(module, "gather_output", False):
            return "gathers its output; the strict column kernel returns the local shard"
        if kind == "row" and not getattr(module, "input_is_parallel", True):
            return "expects unsharded input; the strict row kernel takes the local shard"
        if kind == "row" and not getattr(module, "reduce_results", True):
            return "does not reduce its result; the strict row kernel always reduces"
        return None

    return check


def check_sglang_server_args(server_args: Any) -> None:
    """Raise if SGLang's server settings change the DiT's arithmetic or bypass a bound forward."""

    refusals = []
    for flag in ("enable_torch_compile", "regional_compile", "enable_breakable_cuda_graph"):
        if getattr(server_args, flag, False):
            refusals.append(f"{flag} is set")
    if getattr(server_args, "cache_dit_config", None):
        refusals.append("cache_dit_config is set (cached blocks are not recomputed)")
    if getattr(server_args, "quantization", None) or getattr(
        server_args, "component_quantizations", None
    ):
        refusals.append("quantization is set")
    if getattr(server_args, "lora_path", None):
        refusals.append("lora_path is set (load a checkpoint with the adapter merged)")
    if refusals:
        raise RuntimeError("strict H3 rollout on SGLang: " + "; ".join(refusals))


def install_h3_sglang_rollout(
    model: torch.nn.Module, *, collective=None, tp: int = 1, strict: bool = True
) -> H3RolloutHandle:
    """Bind the strict projections of every block and refiner block of SGLang's H3 DiT.

    ``tp`` is SGLang's TP size and ``collective`` a deterministic collective over SGLang's TP
    group (``tp > 1``). With ``strict``, any refusal raises before a forward changes; without it,
    refused projections keep SGLang's forward and are listed in ``handle.skipped`` (model-level
    refusals still raise, since no forward of a compiled model can be bound).
    """

    compiled, variant = _model_refusals(model)
    refusals = compiled + variant if strict else compiled
    if refusals:
        raise RuntimeError("H3 rollout on SGLang: " + "; ".join(refusals))
    return install_h3_rollout(
        model, collective=collective, tp=tp, strict=strict, check=_check_projection(tp)
    )


def report(model: torch.nn.Module, handle: H3RolloutHandle) -> dict:
    """The handle's report, plus whether SGLang's AdaLN cache replaces the AdaLN projections."""

    out = handle.report()
    out["engine"] = ENGINE
    out["adaln_cache"] = any(getattr(m, "adaln_proj", "absent") is None for m in model.modules())
    return out
